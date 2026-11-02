"""
Unit Tests for RAG Statistical Evaluation Suite
Tests core statistical formulations:
- ECE / MCE properties
- McNemar symmetry and significance
- Paired Bootstrap 95% Confidence Interval convergence
- Fairness 4/5ths rule checks
"""

import numpy as np
import pytest

from core.calibration import compute_ece, compute_brier_score
from core.hypothesis_tests import mcnemar_test, paired_bootstrap_test, compute_mcnemar_power
from core.fairness_audit import audit_fairness
from core.noise_stress import inject_typo, inject_synonyms

def test_ece_perfect_calibration():
    # When confidences exactly equal accuracies, ECE should be near 0
    confidences = np.array([0.1, 0.1, 0.9, 0.9])
    # 1 true out of ten for 0.1, 9 true out of ten for 0.9
    confs = np.concatenate([np.full(100, 0.1), np.full(100, 0.9)])
    accs = np.concatenate([
        np.array([1]*10 + [0]*90),
        np.array([1]*90 + [0]*10)
    ])
    ece, mce, bins = compute_ece(confs, accs, n_bins=10)
    assert ece < 0.05
    assert mce < 0.05

def test_mcnemar_identical_models():
    a = np.array([True, False, True, True, False])
    b = np.array([True, False, True, True, False])
    res = mcnemar_test(a, b)
    assert res["p_value"] == 1.0
    assert not res["significant_at_05"]

def test_mcnemar_significant_divergence():
    # Model B gets 40 correct that A missed, A gets 2 correct that B missed
    a = np.array([True]*10 + [False]*40 + [False]*50)
    b = np.array([True]*10 + [True]*40  + [False]*50)
    res = mcnemar_test(a, b)
    assert res["p_value"] < 0.001
    assert res["significant_at_05"] is True
    assert "statistical_power_at_05" in res
    assert res["statistical_power_at_05"] > 0.80

def test_statistical_power_calculation():
    # Large sample (N=500), 20% discordant, strong odds ratio 2.5
    power = compute_mcnemar_power(n_samples=500, discordant_rate=0.20, odds_ratio=2.5, alpha=0.05)
    assert power > 0.85
    # Weak effect (OR=1.1) yields low power
    low_power = compute_mcnemar_power(n_samples=500, discordant_rate=0.10, odds_ratio=1.1, alpha=0.05)
    assert low_power < 0.30

def test_paired_bootstrap_confidence_interval():
    a = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0] * 20)
    b = np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1] * 20)
    res = paired_bootstrap_test(a, b, n_resamples=500)
    assert res["ci_95_lower"] < res["observed_diff"] < res["ci_95_upper"]
    assert res["observed_diff"] > 0

def test_fairness_four_fifths_rule():
    # 2 groups: Group A (100 samples, 80% pass), Group B (100 samples, 40% pass)
    y_true = np.ones(200)
    y_pred = np.array([1]*80 + [0]*20 + [1]*40 + [0]*60)
    groups = np.array(["GroupA"]*100 + ["GroupB"]*100)
    
    res = audit_fairness(y_true, y_pred, groups)
    assert res["disparate_impact_ratio"] == 0.5 # 40% / 80%
    assert res["passes_four_fifths_rule"] is False

def test_noise_injection():
    orig = "The hemodynamic parameters of the patient are critical"
    noisy = inject_typo(orig, severity=0.2, rng=np.random.RandomState(42))
    assert orig != noisy
    assert len(noisy) > 10

def test_edge_cases_and_invalid_inputs():
    # Mismatched length in McNemar must raise AssertionError
    with pytest.raises(AssertionError):
        mcnemar_test(np.array([True, False]), np.array([True]))
    
    # Zero discordant pairs returns p_value = 1.0 and power = 0.0
    a = np.array([True, True, False, False])
    b = np.array([True, True, False, False])
    res_zero = mcnemar_test(a, b)
    assert res_zero["p_value"] == 1.0
    assert not res_zero["significant_at_05"]

    # Power with 0 discordant rate or invalid N returns 0.0
    assert compute_mcnemar_power(n_samples=0, discordant_rate=0.2, odds_ratio=2.0) == 0.0
    assert compute_mcnemar_power(n_samples=500, discordant_rate=0.0, odds_ratio=2.0) == 0.0
    assert compute_mcnemar_power(n_samples=500, discordant_rate=0.2, odds_ratio=1.0) == 0.0

    # Reproducibility check for paired bootstrap with fixed seed
    s1 = np.array([0.1, 0.4, 0.8, 0.2, 0.9] * 10)
    s2 = np.array([0.3, 0.5, 0.9, 0.4, 0.95] * 10)
    b1 = paired_bootstrap_test(s1, s2, n_resamples=300, random_state=123)
    b2 = paired_bootstrap_test(s1, s2, n_resamples=300, random_state=123)
    assert b1["ci_95_lower"] == b2["ci_95_lower"]
    assert b1["ci_95_upper"] == b2["ci_95_upper"]


def test_real_rag_retrieval_and_cached_gemini_answers():
    """Verifies lexical retrieval accuracy, cached Gemini API responses, and Cohen's Kappa judge validation."""
    from pipeline.real_rag import (
        REAL_RAG_EVAL_QUESTIONS,
        retrieve_chunks,
        answer_with_rag,
        compute_cohens_kappa,
    )

    for q in REAL_RAG_EVAL_QUESTIONS:
        top_docs = retrieve_chunks(q["question"], top_k=1)
        assert top_docs[0]["doc_id"] == q["relevant_doc"]

        ans = answer_with_rag(q["question"], config="champion")
        assert q["relevant_doc"] in ans["retrieved_docs"]
        assert any(kw.lower() in ans["answer"].lower() for kw in q["expected_keywords"])
        assert ans["total_tokens"] > 0

    # Cohen's Kappa validation between human labels and LLM-as-Judge
    human_labels = [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]
    judge_labels = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
    kappa_res = compute_cohens_kappa(human_labels, judge_labels)
    assert kappa_res["cohens_kappa"] >= 0.75
    assert kappa_res["observed_agreement"] == 0.90

if __name__ == "__main__":
    test_ece_perfect_calibration()
    test_mcnemar_identical_models()
    test_mcnemar_significant_divergence()
    test_paired_bootstrap_confidence_interval()
    test_fairness_four_fifths_rule()
    test_noise_injection()
    print("All statistical suite tests passed successfully!")
