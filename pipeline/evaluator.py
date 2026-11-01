"""
Full Statistical Evaluation Orchestrator
Runs all 4 statistical pillars:
1. Uncertainty Calibration (ECE / MCE / Reliability bins)
2. Hypothesis Testing & Significance (McNemar, Paired Bootstrap 95% CI, Wilcoxon)
3. Noise Fragility & Adversarial Degradation Curves
4. Fairness & Disparate Impact Audit across Subgroups
"""

import numpy as np
from typing import Dict, Any, List

from core.calibration import compute_ece, compute_brier_score
from core.hypothesis_tests import mcnemar_test, paired_bootstrap_test, wilcoxon_signed_rank
from core.fairness_audit import audit_fairness
from core.noise_stress import evaluate_perturbation_curve
from pipeline.rag_simulator import generate_benchmark_dataset, simulate_rag_inference

def run_full_statistical_evaluation(n_samples: int = 500, seed: int = 42) -> Dict[str, Any]:
    """
    Executes end-to-end evaluation comparing Baseline RAG pipeline vs Champion RAG pipeline.
    Returns complete structured payload suitable for reports and frontend HUD.
    """
    np.random.seed(seed)
    samples = generate_benchmark_dataset(n_samples=n_samples, seed=seed)
    
    # 1. Simulate evaluations for both models on identical benchmark questions
    baseline_correct = []
    baseline_conf = []
    baseline_semantic = []
    
    champion_correct = []
    champion_conf = []
    champion_semantic = []
    
    subgroups = []
    domains = []
    
    for s in samples:
        subgroups.append(s["subgroup"])
        domains.append(s["domain"])
        
        # Baseline model
        b_res = simulate_rag_inference(s, model_version="baseline", noise_severity=0.0)
        baseline_correct.append(b_res["is_correct"])
        baseline_conf.append(b_res["confidence"])
        baseline_semantic.append(b_res["semantic_similarity"])
        
        # Champion model
        c_res = simulate_rag_inference(s, model_version="champion", noise_severity=0.0)
        champion_correct.append(c_res["is_correct"])
        champion_conf.append(c_res["confidence"])
        champion_semantic.append(c_res["semantic_similarity"])
        
    b_corr = np.array(baseline_correct, dtype=bool)
    b_conf = np.array(baseline_conf, dtype=float)
    b_sem = np.array(baseline_semantic, dtype=float)
    
    c_corr = np.array(champion_correct, dtype=bool)
    c_conf = np.array(champion_conf, dtype=float)
    c_sem = np.array(champion_semantic, dtype=float)
    
    subgroups_arr = np.array(subgroups)
    
    # === PILLAR 1: CALIBRATION & ECE ===
    b_ece, b_mce, b_bins = compute_ece(b_conf, b_corr, n_bins=10)
    c_ece, c_mce, c_bins = compute_ece(c_conf, c_corr, n_bins=10)
    b_brier = compute_brier_score(b_conf, b_corr.astype(float))
    c_brier = compute_brier_score(c_conf, c_corr.astype(float))
    
    # === PILLAR 2: HYPOTHESIS TESTING & STATISTICAL SIGNIFICANCE ===
    mcnemar_res = mcnemar_test(b_corr, c_corr)
    bootstrap_res = paired_bootstrap_test(b_corr.astype(float), c_corr.astype(float), n_resamples=2000)
    wilcoxon_res = wilcoxon_signed_rank(b_sem, c_sem)
    
    # === PILLAR 3: ADVERSARIAL NOISE FRAGILITY CURVES ===
    def eval_baseline_noise(severity: float) -> float:
        accs = []
        for s in samples[:150]:
            r = simulate_rag_inference(s, model_version="baseline", noise_severity=severity)
            accs.append(r["is_correct"])
        return float(np.mean(accs))
        
    def eval_champion_noise(severity: float) -> float:
        accs = []
        for s in samples[:150]:
            r = simulate_rag_inference(s, model_version="champion", noise_severity=severity)
            accs.append(r["is_correct"])
        return float(np.mean(accs))
        
    severities = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    b_noise_curve = evaluate_perturbation_curve(eval_baseline_noise, severities)
    c_noise_curve = evaluate_perturbation_curve(eval_champion_noise, severities)
    
    # === PILLAR 4: FAIRNESS AUDIT & SUBGROUP DISPARITY ===
    fairness_champion = audit_fairness(
        y_true=np.ones(len(c_corr), dtype=int), # Ground truth correctness goal
        y_pred=c_corr.astype(int),
        subgroups=subgroups_arr,
        favorable_class=1
    )
    
    return {
        "metadata": {
            "total_benchmark_samples": n_samples,
            "domains_evaluated": ["clinical", "finance", "legal", "devops"],
            "subgroups_audited": list(np.unique(subgroups_arr)),
            "baseline_model": "RAG Baseline (Uncalibrated Standard Retrieval)",
            "champion_model": "RAG Champion (Reranked & Temperature-Calibrated)"
        },
        "summary": {
            "baseline_accuracy": round(float(np.mean(b_corr)) * 100, 2),
            "champion_accuracy": round(float(np.mean(c_corr)) * 100, 2),
            "accuracy_lift_pct": round((float(np.mean(c_corr)) - float(np.mean(b_corr))) * 100, 2),
            "baseline_ece": round(b_ece, 4),
            "champion_ece": round(c_ece, 4),
            "ece_reduction_pct": round(((b_ece - c_ece) / max(b_ece, 1e-6)) * 100, 1),
            "p_value_mcnemar": mcnemar_res["p_value"],
            "statistically_significant": mcnemar_res["significant_at_05"],
            "fairness_four_fifths_compliant": fairness_champion["passes_four_fifths_rule"]
        },
        "pillar1_calibration": {
            "baseline": {
                "ece": round(b_ece, 4),
                "mce": round(b_mce, 4),
                "brier_score": round(b_brier, 4),
                "bins": b_bins
            },
            "champion": {
                "ece": round(c_ece, 4),
                "mce": round(c_mce, 4),
                "brier_score": round(c_brier, 4),
                "bins": c_bins
            }
        },
        "pillar2_hypothesis_testing": {
            "mcnemar": mcnemar_res,
            "paired_bootstrap": bootstrap_res,
            "wilcoxon": wilcoxon_res
        },
        "pillar3_robustness_curves": {
            "severity_levels": severities,
            "baseline_curve": b_noise_curve,
            "champion_curve": c_noise_curve
        },
        "pillar4_fairness_audit": fairness_champion
    }
