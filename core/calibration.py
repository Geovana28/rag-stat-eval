"""
Calibration & Uncertainty Quantification Module
Implements:
- Reliability diagram binning
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score & Negative Log-Likelihood (NLL)
- Post-hoc Temperature Scaling
"""

import numpy as np
from typing import Dict, Any, Tuple, List

def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    MCE = max_{m=1}^M |acc(B_m) - conf(B_m)|
    
    Returns:
        ece: float,
        mce: float,
        bin_details: list of dicts with bin statistics for reliability diagrams.
    """
    confidences = np.asarray(confidences, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    bin_details = []
    n_samples = len(confidences)
    
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # For the last bin, include the upper edge 1.0
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            calibration_gap = abs(bin_acc - bin_conf)
            
            ece += (bin_size / n_samples) * calibration_gap
            mce = max(mce, calibration_gap)
            
            bin_details.append({
                "bin_idx": i,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": int(bin_size),
                "proportion": round(float(bin_size / n_samples), 4),
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "gap": round(calibration_gap, 4)
            })
        else:
            bin_details.append({
                "bin_idx": i,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": 0,
                "proportion": 0.0,
                "accuracy": None,
                "confidence": None,
                "gap": 0.0
            })
            
    return float(ece), float(mce), bin_details


def compute_brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes standard multi-class or binary Brier Score:
    BS = (1 / N) * sum (p_i - y_i)^2
    """
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return float(np.mean((probabilities - targets) ** 2))


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Applies temperature scaling to raw logits:
    p_i = softmax(z_i / T)
    """
    scaled_logits = logits / max(temperature, 1e-4)
    # Stable softmax
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def optimize_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Finds optimal temperature parameter T > 0 minimizing Negative Log-Likelihood (NLL).
    Uses a simple grid/golden-section search.
    """
    from scipy.optimize import minimize_scalar
    
    def nll_obj(T):
        scaled = logits / max(T, 1e-4)
        exp_logits = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        # NLL for true labels
        chosen_probs = probs[np.arange(len(labels)), labels]
        chosen_probs = np.clip(chosen_probs, 1e-12, 1.0)
        return -np.mean(np.log(chosen_probs))
        
    res = minimize_scalar(nll_obj, bounds=(0.05, 5.0), method='bounded')
    return float(res.x)
