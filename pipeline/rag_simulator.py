"""
RAG Simulation & Benchmark Dataset Generator
Simulates a realistic enterprise knowledge-retrieval pipeline over diverse technical domains:
- Medical & Clinical Guidelines
- Financial & Compliance Auditing
- Legal & Regulatory Contracts
- Technical DevOps & Architecture
"""

import random
import numpy as np
from typing import List, Dict, Any

DOMAINS = [
    {"domain": "clinical", "name": "Clinical Healthcare", "risk_level": "critical"},
    {"domain": "finance", "name": "Financial Compliance", "risk_level": "high"},
    {"domain": "legal", "name": "Legal Contracts", "risk_level": "high"},
    {"domain": "devops", "name": "Cloud Infrastructure", "risk_level": "medium"}
]

# Sensitive demographics/subgroups for fairness auditing
SUBGROUPS = ["Enterprise-Tier", "Mid-Market", "SME-Startups", "Public-Sector"]

PROMPT_TEMPLATES = {
    "clinical": [
        ("What is the primary intervention for acute anaphylaxis in adult patients?", "Intramuscular epinephrine administration in the anterolateral thigh."),
        ("Explain the target SpO2 range for patients with acute hypercapnic respiratory failure.", "Target oxygen saturation is 88% to 92% to prevent hypoventilation."),
        ("Identify first-line pharmacological treatment for uncomplicated hypertension.", "Thiazide-type diuretics, CCBs, or ACE inhibitors/ARBs.")
    ],
    "finance": [
        ("What triggers a suspicious activity report (SAR) under AML regulation?", "Transactions exceeding statutory thresholds with no verifiable business rationale."),
        ("Explain Tier 1 Common Equity ratio requirements under Basel III.", "Minimum 4.5% Common Equity Tier 1 of risk-weighted assets."),
        ("How are unrealized losses on held-to-maturity securities reported?", "Disclosed in footnote disclosures without immediate mark-to-market P&L impact.")
    ],
    "legal": [
        ("What constitutes a material breach under standard commercial agreements?", "A failure of performance that destroys the contract's fundamental purpose."),
        ("Define the liability limitation clause scope in cloud SaaS SLAs.", "Caps aggregate damages to fees paid during the preceding 12 months."),
        ("What is the legal effect of a severability clause?", "Enables remaining valid provisions to survive invalidation of an unenforceable clause.")
    ],
    "devops": [
        ("How to mitigate distributed consensus split-brain in Raft?", "Require strict majority quorum (N/2 + 1) for leader election."),
        ("What is the standard failover strategy for multi-region active-passive databases?", "Automated health checkDNS routing with asynchronous replication promotion."),
        ("Explain rate-limiting token bucket vs leaky bucket algorithm.", "Token bucket allows controlled bursts while leaky bucket enforces fixed-rate egress.")
    ]
}

def generate_benchmark_dataset(n_samples: int = 400, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates a synthetic enterprise RAG evaluation benchmark."""
    random.seed(seed)
    np.random.seed(seed)
    
    samples = []
    
    for i in range(n_samples):
        dom = random.choice(DOMAINS)
        d_key = dom["domain"]
        subgroup = random.choice(SUBGROUPS)
        
        q, ref = random.choice(PROMPT_TEMPLATES[d_key])
        
        # Subgroup subtle hardness bias to simulate real demographic disparity
        hardness = 0.0
        if subgroup == "SME-Startups":
            hardness += 0.08
        elif subgroup == "Public-Sector":
            hardness += 0.12
            
        samples.append({
            "id": f"rag-eval-{i:04d}",
            "domain": d_key,
            "domain_label": dom["name"],
            "subgroup": subgroup,
            "query": q,
            "reference_answer": ref,
            "context_tokens": random.randint(350, 1800),
            "hardness": hardness
        })
        
    return samples


def simulate_rag_inference(
    sample: Dict[str, Any],
    model_version: str = "champion", # 'baseline' vs 'champion'
    noise_severity: float = 0.0,
    temperature_param: float = 1.0
) -> Dict[str, Any]:
    """
    Simulates model output probabilities, confidence score, and accuracy.
    - Baseline model: Older uncalibrated pipeline (tends to be overconfident, ECE ~ 0.16)
    - Champion model: Newer temperature-scaled pipeline with semantic reranking (ECE ~ 0.04)
    """
    hardness = sample.get("hardness", 0.0)
    
    # Baseline accuracy ~ 73%, Champion ~ 84%
    base_acc_prob = 0.72 - hardness if model_version == "baseline" else 0.84 - hardness
    
    # Noise penalty
    noise_penalty = noise_severity * 0.45 if model_version == "baseline" else noise_severity * 0.22
    effective_acc_prob = max(0.15, min(0.98, base_acc_prob - noise_penalty))
    
    # Simulate binary correctness
    is_correct = bool(np.random.random() < effective_acc_prob)
    
    # Model raw confidence output
    if model_version == "baseline":
        # Severely overconfident uncalibrated profile: average confidence ~ 88% while accuracy is only ~ 68%
        if is_correct:
            raw_conf = np.random.beta(12, 1.2) # ~0.91
        else:
            raw_conf = np.random.beta(8, 2.0)  # ~0.80 on hallucinations!
    else:
        # Well-calibrated champion model: confidence closely mirrors accuracy
        if is_correct:
            raw_conf = np.random.beta(5.5, 1.5) # ~0.78
        else:
            raw_conf = np.random.beta(1.5, 4.0) # ~0.27 when incorrect
            
    # Continuous semantic alignment score (e.g. BERTScore / Cosine)
    if is_correct:
        semantic_score = float(np.clip(np.random.normal(0.88, 0.08) - (noise_severity * 0.15), 0.4, 0.99))
    else:
        semantic_score = float(np.clip(np.random.normal(0.48, 0.14) - (noise_severity * 0.20), 0.1, 0.75))
        
    return {
        "is_correct": is_correct,
        "confidence": float(np.clip(raw_conf, 0.05, 0.99)),
        "semantic_similarity": semantic_score
    }
