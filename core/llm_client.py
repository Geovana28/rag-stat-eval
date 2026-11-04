"""
Real LLM & Embedding Client for Gemini REST API with Disk Caching, Latency, and Token Tracking.
Supports deterministic offline/CI execution via local JSON cache (`cache/gemini_cache.json`)
and live execution when `GEMINI_API_KEY` is provided in `.env` or environment variables.
"""

import os
import json
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_env_key(env_path: Optional[Path] = None) -> Optional[str]:
    """Loads GEMINI_API_KEY from environment or local .env file without external dependencies."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key and key != "COLE_SUA_CHAVE_AQUI" and not key.startswith("your_"):
        return key

    search_paths = [
        env_path,
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for p in search_paths:
        if p and p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val and val != "COLE_SUA_CHAVE_AQUI" and not val.startswith("your_"):
                            return val
            except Exception:
                pass
    return None


class GeminiRAGClient:
    """
    Production-minded Gemini API client with:
    - Disk cache (`cache/gemini_cache.json`) so CI/tests run deterministically with zero API cost
    - Latency measurement (ms) and token usage accounting
    - Clean fallback/mock support for unit tests
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        generation_model: str = "gemini-3.8-flash",
        embedding_model: str = "text-embedding-004",
        cache_path: Optional[Path] = None,
    ):
        self.api_key = api_key or load_env_key()
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.cache_path = cache_path or (
            Path(__file__).resolve().parent.parent / "cache" / "gemini_cache.json"
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    @staticmethod
    def _hash_key(prefix: str, payload: str) -> str:
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"{prefix}:{h}"

    def generate_answer(
        self,
        prompt: str,
        temperature: float = 0.1,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Calls Gemini generateContent endpoint (or returns cached response) with latency & token metrics.
        """
        cache_key = self._hash_key(f"gen:{self.generation_model}:{temperature}", prompt)
        if use_cache and cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["from_cache"] = True
            return cached

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not configured and prompt not found in cache.")

        models_to_try = [
            self.generation_model,
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash-lite",
        ]
        # Deduplicate while preserving order
        seen = set()
        models_chain = [m for m in models_to_try if not (m in seen or seen.add(m))]

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 350,
            },
        }
        data_bytes = json.dumps(body).encode("utf-8")

        raw = None
        used_model = self.generation_model
        latency_ms = 0.0
        last_err = None

        for m in models_chain:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{m}:generateContent?key={self.api_key}"
            )
            req = urllib.request.Request(
                url, data=data_bytes, headers={"Content-Type": "application/json"}, method="POST"
            )
            try:
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=20) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                used_model = m
                break
            except Exception as e:
                last_err = e
                continue

        if raw is None:
            raise RuntimeError(f"All Gemini models in fallback chain failed: {last_err}")

        text = ""
        candidates = raw.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text = parts[0].get("text", "").strip()

        usage = raw.get("usageMetadata", {})
        prompt_tokens = int(usage.get("promptTokenCount", max(1, len(prompt) // 4)))
        completion_tokens = int(usage.get("candidatesTokenCount", max(1, len(text) // 4)))
        total_tokens = int(usage.get("totalTokenCount", prompt_tokens + completion_tokens))

        result = {
            "text": text,
            "model": used_model,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "from_cache": False,
        }
        if use_cache:
            self._cache[cache_key] = result
            self._save_cache()
        return result