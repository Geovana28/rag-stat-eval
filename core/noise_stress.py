"""
Noise & Adversarial Perturbation Stress Testing Module
Implements:
- Typographical noise (character swap, insertion, deletion, QWERTY keyboard distance)
- Lexical & Synonym perturbation (masking and semantic jitter)
- RAG Distractor Context injection (irrelevant retrieval fragments, contradictory facts)
- Perturbation Fragility Curve generation across ascending perturbation intensity levels
"""

import random
import string
import numpy as np
from typing import List, Dict, Any, Callable

# Standard QWERTY proximity map
KEYBOARD_ADJACENT = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy', 'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
}

SYNONYMS = {
    "important": ["crucial", "critical", "vital", "essential"],
    "increase": ["boost", "elevate", "expand", "grow"],
    "decrease": ["reduce", "lower", "drop", "diminish"],
    "fast": ["quick", "rapid", "swift", "speedy"],
    "error": ["fault", "mistake", "defect", "glitch"],
    "show": ["display", "exhibit", "demonstrate", "reveal"],
    "support": ["sustain", "back", "uphold", "reinforce"],
    "patient": ["individual", "subject", "case"],
    "result": ["outcome", "finding", "consequence"]
}

DISTRACTOR_SENTENCES = [
    "Recent atmospheric measurements indicate variable solar irradiance over the oceanic basin.",
    "A secondary committee suggested rescheduling the third quarter fiscal assembly to November.",
    "The regional wildlife preservation agency documented migratory patterns of tundra swans.",
    "Engineers completed stress tests on the hydraulic bypass valve with nominal friction coefficients."
]

def inject_typo(text: str, severity: float = 0.1, rng: random.Random = None) -> str:
    """Injects QWERTY keyboard adjacent typos or swaps into string."""
    if rng is None:
        rng = random.Random()
        
    chars = list(text)
    n_perturbs = max(1, int(len(chars) * severity))
    
    for _ in range(n_perturbs):
        idx = rng.randint(0, len(chars) - 1)
        c = chars[idx].lower()
        if c in KEYBOARD_ADJACENT:
            options = list(KEYBOARD_ADJACENT[c])
            chars[idx] = str(rng.choice(options))
        elif (hasattr(rng, 'random') and rng.random() < 0.5) and idx > 0:
            # Transposition swap
            chars[idx - 1], chars[idx] = chars[idx], chars[idx - 1]
            
    return "".join(chars)


def inject_synonyms(text: str, severity: float = 0.2, rng: random.Random = None) -> str:
    """Replaces words with semantic synonyms to test lexical resilience."""
    if rng is None:
        rng = random.Random()
        
    words = text.split()
    perturbed = []
    for w in words:
        clean_w = w.lower().strip(string.punctuation)
        if clean_w in SYNONYMS and rng.random() < severity:
            syn = rng.choice(SYNONYMS[clean_w])
            # Preserve punctuation if any
            if w[-1] in string.punctuation:
                syn += w[-1]
            perturbed.append(syn)
        else:
            perturbed.append(w)
    return " ".join(perturbed)


def inject_rag_distractor(context: str, n_distractors: int = 2, rng: random.Random = None) -> str:
    """Injects irrelevant distractor sentences into retrieved context to test retrieval filtering."""
    if rng is None:
        rng = random.Random()
        
    parts = context.split(". ")
    chosen_distractors = rng.sample(DISTRACTOR_SENTENCES, min(n_distractors, len(DISTRACTOR_SENTENCES)))
    
    for dist in chosen_distractors:
        insert_idx = rng.randint(0, len(parts))
        parts.insert(insert_idx, dist)
        
    return ". ".join(parts)


def evaluate_perturbation_curve(
    eval_fn: Callable[[float], float],
    severity_levels: List[float] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates model performance across ascending levels of perturbation severity (0% to 50%).
    
    Returns curve data points: { severity, performance, degradation_pct }
    """
    if severity_levels is None:
        severity_levels = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
        
    baseline_perf = eval_fn(0.0)
    curve = []
    
    for s in severity_levels:
        perf = eval_fn(s)
        deg = ((baseline_perf - perf) / max(baseline_perf, 1e-6)) * 100.0
        curve.append({
            "severity": round(s, 2),
            "performance": round(float(perf), 4),
            "degradation_pct": round(float(max(deg, 0.0)), 2),
            "retention_rate": round(float(max(100.0 - deg, 0.0)), 2)
        })
        
    return curve
