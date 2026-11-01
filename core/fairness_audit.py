"""
Fairness & Subgroup Disparity Audit Module
Implements:
- Disparate Impact Ratio (DIR / 80% rule)
- Demographic Parity Difference
- Equal Opportunity Difference (True Positive Rate disparity)
- Equalized Odds (TPR and FPR parity across sensitive attributes)
- Subgroup error disparity analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

def audit_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroups: np.ndarray,
    favorable_class: int = 1
) -> Dict[str, Any]:
    """
    Computes standard algorithmic fairness metrics across distinct subgroups.
    
    Metrics:
    - Selection Rate (P(Y_hat = 1 | A = a))
    - Disparate Impact Ratio (min(SR) / max(SR))
    - Demographic Parity Difference (max(SR) - min(SR))
    - True Positive Rate (TPR / Recall) per group -> Equal Opportunity Difference
    - False Positive Rate (FPR) per group
    - Accuracy per group
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    subgroups = np.asarray(subgroups)
    
    unique_groups = np.unique(subgroups)
    group_stats = {}
    
    selection_rates = []
    tprs = []
    fprs = []
    accuracies = []
    
    for g in unique_groups:
        mask = (subgroups == g)
        n_g = int(np.sum(mask))
        if n_g == 0:
            continue
            
        yt_g = y_true[mask]
        yp_g = y_pred[mask]
        
        # Selection rate
        sr = float(np.mean(yp_g == favorable_class))
        selection_rates.append(sr)
        
        # Accuracy
        acc = float(np.mean(yt_g == yp_g))
        accuracies.append(acc)
        
        # Positives
        pos_mask = (yt_g == favorable_class)
        neg_mask = (yt_g != favorable_class)
        
        tpr = float(np.mean(yp_g[pos_mask] == favorable_class)) if np.sum(pos_mask) > 0 else 0.0
        fpr = float(np.mean(yp_g[neg_mask] == favorable_class)) if np.sum(neg_mask) > 0 else 0.0
        
        tprs.append(tpr)
        fprs.append(fpr)
        
        group_stats[str(g)] = {
            "sample_count": n_g,
            "selection_rate": round(sr, 4),
            "accuracy": round(acc, 4),
            "true_positive_rate": round(tpr, 4),
            "false_positive_rate": round(fpr, 4)
        }
        
    # Global Disparity Metrics
    max_sr = max(selection_rates) if selection_rates else 1.0
    min_sr = min(selection_rates) if selection_rates else 0.0
    
    # 4/5ths / 80% rule
    disparate_impact_ratio = (min_sr / max(max_sr, 1e-6)) if max_sr > 0 else 1.0
    demographic_parity_diff = max_sr - min_sr
    
    max_tpr = max(tprs) if tprs else 1.0
    min_tpr = min(tprs) if tprs else 0.0
    equal_opportunity_diff = max_tpr - min_tpr
    
    max_fpr = max(fprs) if fprs else 0.0
    min_fpr = min(fprs) if fprs else 0.0
    equalized_odds_diff = max(equal_opportunity_diff, max_fpr - min_fpr)
    
    max_acc = max(accuracies) if accuracies else 1.0
    min_acc = min(accuracies) if accuracies else 0.0
    accuracy_disparity = max_acc - min_acc
    
    # Four-Fifths compliance check
    passes_80_rule = bool(disparate_impact_ratio >= 0.80)
    
    return {
        "groups": group_stats,
        "disparate_impact_ratio": round(float(disparate_impact_ratio), 4),
        "passes_four_fifths_rule": passes_80_rule,
        "demographic_parity_difference": round(float(demographic_parity_diff), 4),
        "equal_opportunity_difference": round(float(equal_opportunity_diff), 4),
        "equalized_odds_difference": round(float(equalized_odds_diff), 4),
        "accuracy_disparity": round(float(accuracy_disparity), 4),
        "verdict": "Compliant with 80% legal fairness threshold" if passes_80_rule else "Violates 80% disparate impact threshold (adverse disparity detected)"
    }
