"""
Hypothesis Testing & Statistical Significance Module
Implements:
- McNemar's Test (paired binary accuracy comparison between Model A and Model B)
- Wilcoxon Signed-Rank Test (paired continuous score comparison, e.g. RAG faithfulness/similarity)
- Paired Bootstrap Resampling with 95% Confidence Intervals
- Effect Size calculation (Cohen's d, Odds Ratio)
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple

def compute_mcnemar_power(
    n_samples: int,
    discordant_rate: float,
    odds_ratio: float,
    alpha: float = 0.05
) -> float:
    """
    Computes asymptotic statistical power (1 - beta) for McNemar's paired test
    using the asymptotic normal approximation for discordant proportions.

    Args:
        n_samples: Total number of paired evaluation queries (N).
        discordant_rate: Proportion of paired samples with discordant outcomes (p01 + p10).
        odds_ratio: Expected effect size psi = p01 / p10.
        alpha: Type I error significance threshold (default: 0.05).

    Returns:
        Estimated statistical power in the range [0.0, 1.0].
    """
    if discordant_rate <= 0 or odds_ratio <= 0 or odds_ratio == 1.0 or n_samples <= 0:
        return 0.0

    # Probabilities of discordance
    p_disc = float(discordant_rate)
    p01 = p_disc * (odds_ratio / (1.0 + odds_ratio))
    p10 = p_disc * (1.0 / (1.0 + odds_ratio))

    delta = p01 - p10
    var_null = p_disc / n_samples
    var_alt = (p_disc - delta**2) / n_samples

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    numerator = abs(delta) - z_alpha * np.sqrt(var_null)
    denominator = np.sqrt(var_alt)

    z_beta = numerator / max(denominator, 1e-9)
    power = float(stats.norm.cdf(z_beta))
    return round(float(np.clip(power, 0.0, 1.0)), 4)


def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    exact: bool = False
) -> Dict[str, Any]:
    """
    Performs McNemar's test for paired nominal data (Model A vs Model B).
    
    Contingency table:
                Model B Correct   Model B Wrong
    Model A Corr       n00              n01 (A right, B wrong)
    Model A Wrong      n10 (A wrong, B right)   n11
    
    Statistic (with continuity correction):
    chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    assert len(correct_a) == len(correct_b), "Both model evaluations must have identical sample length."
    
    n01 = int(np.sum(correct_a & ~correct_b)) # A correct, B incorrect
    n10 = int(np.sum(~correct_a & correct_b)) # A incorrect, B correct
    n00 = int(np.sum(correct_a & correct_b))  # Both correct
    n11 = int(np.sum(~correct_a & ~correct_b)) # Both incorrect
    
    discordant = n01 + n10
    
    if discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_05": False,
            "table": {"both_correct": n00, "only_a_correct": n01, "only_b_correct": n10, "both_wrong": n11},
            "odds_ratio": 1.0,
            "interpretation": "Identical performance on all samples. No statistical divergence."
        }
        
    # If sample size is small or exact requested, binomial test
    if exact or discordant < 25:
        p_val = stats.binomtest(min(n01, n10), n=discordant, p=0.5, alternative='two-sided').pvalue
        stat = float(min(n01, n10))
    else:
        # Edwards continuity correction
        stat = ((abs(n01 - n10) - 1.0) ** 2) / discordant
        p_val = stats.chi2.sf(stat, df=1)
        
    odds_ratio = (n01 / max(n10, 1e-6)) if n10 > 0 else float('inf')
    disc_rate = discordant / len(correct_a)
    
    # Effect size for power: ratio of larger discordant to smaller discordant (>= 1.0)
    eff_ratio = max(n01, n10) / max(min(n01, n10), 1)
    estimated_power = compute_mcnemar_power(
        n_samples=len(correct_a),
        discordant_rate=disc_rate,
        odds_ratio=float(eff_ratio),
        alpha=0.05
    )
    
    return {
        "statistic": round(float(stat), 4),
        "p_value": float(p_val),
        "significant_at_05": bool(p_val < 0.05),
        "significant_at_01": bool(p_val < 0.01),
        "statistical_power_at_05": estimated_power,
        "table": {"both_correct": n00, "only_a_correct": n01, "only_b_correct": n10, "both_wrong": n11},
        "odds_ratio": round(float(odds_ratio), 3) if odds_ratio != float('inf') else "inf",
        "interpretation": "Statistically significant difference detected (reject H0)." if p_val < 0.05 else "Differences are not statistically significant (fail to reject H0)."
    }


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 2000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes paired bootstrap difference between two models with 95% Confidence Interval.
    delta = metric(B) - metric(A)
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    n = len(scores_a)
    rng = np.random.RandomState(random_state)
    
    observed_diff = float(np.mean(scores_b) - np.mean(scores_a))
    
    boot_diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_diffs[i] = np.mean(scores_b[idx]) - np.mean(scores_a[idx])
        
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    
    # Two-sided empirical p-value for difference = 0
    if observed_diff > 0:
        p_value = 2.0 * float(np.mean(boot_diffs <= 0))
    else:
        p_value = 2.0 * float(np.mean(boot_diffs >= 0))
    p_value = min(max(p_value, 1.0 / n_resamples), 1.0)
    
    return {
        "observed_diff": round(observed_diff, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "significant_at_05": bool(ci_lower > 0 or ci_upper < 0),
        "resamples": n_resamples
    }


def wilcoxon_signed_rank(
    scores_a: np.ndarray,
    scores_b: np.ndarray
) -> Dict[str, Any]:
    """
    Non-parametric Wilcoxon signed-rank test for paired continuous scores (e.g. BERTScore, ROUGE, Cosine Similarity).
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    diffs = scores_b - scores_a
    
    # Remove zero diffs for standard Wilcoxon
    non_zero_diffs = diffs[diffs != 0]
    if len(non_zero_diffs) < 10:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_05": False,
            "mean_diff": float(np.mean(diffs)),
            "cohens_d": 0.0
        }
        
    res = stats.wilcoxon(scores_a, scores_b, alternative='two-sided')
    
    # Effect size: Cohen's d for paired samples
    sd_diff = np.std(diffs, ddof=1)
    cohens_d = float(np.mean(diffs) / (sd_diff + 1e-8))
    
    return {
        "statistic": round(float(res.statistic), 4),
        "p_value": float(res.pvalue),
        "significant_at_05": bool(res.pvalue < 0.05),
        "mean_diff": round(float(np.mean(diffs)), 4),
        "cohens_d": round(cohens_d, 3),
        "effect_size_magnitude": "negligible" if abs(cohens_d) < 0.2 else ("small" if abs(cohens_d) < 0.5 else ("medium" if abs(cohens_d) < 0.8 else "large"))
    }
