"""
Evaluation metrics: KL divergence, TV distance, L1 error, per-class relative error.
"""

import numpy as np
from typing import Dict, List, Tuple


def kl_divergence(
    p_hist: Dict[str, int],
    q_hist: Dict[str, int],
    labels: List[str] = None,
    epsilon: float = 1e-10
) -> float:
    """
    Compute KL divergence D_KL(P||Q) between two histograms.

    D_KL(P||Q) = sum P(l) * log(P(l) / Q(l))

    Args:
        p_hist: True/reference histogram (P)
        q_hist: Estimated histogram (Q)
        labels: Ordered labels (uses p_hist keys if not provided)
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence value
    """
    if labels is None:
        labels = list(p_hist.keys())

    # Convert to distributions
    p_total = sum(p_hist.get(l, 0) for l in labels)
    q_total = sum(q_hist.get(l, 0) for l in labels)

    if p_total == 0 or q_total == 0:
        return float('inf')

    kl = 0.0
    for label in labels:
        p_val = p_hist.get(label, 0) / p_total
        q_val = q_hist.get(label, 0) / q_total

        # Add epsilon to avoid log(0)
        q_val = max(q_val, epsilon)

        if p_val > 0:
            kl += p_val * np.log(p_val / q_val)

    return kl


def tv_distance(
    p_hist: Dict[str, int],
    q_hist: Dict[str, int],
    labels: List[str] = None
) -> float:
    """
    Compute Total Variation distance between two histograms.

    TV(P, Q) = 0.5 * sum |P(l) - Q(l)|

    Args:
        p_hist: True histogram
        q_hist: Estimated histogram
        labels: Ordered labels

    Returns:
        TV distance (in [0, 1])
    """
    if labels is None:
        labels = list(p_hist.keys())

    # Convert to distributions
    p_total = sum(p_hist.get(l, 0) for l in labels)
    q_total = sum(q_hist.get(l, 0) for l in labels)

    if p_total == 0 and q_total == 0:
        return 0.0
    if p_total == 0 or q_total == 0:
        return 1.0

    tv = 0.0
    for label in labels:
        p_val = p_hist.get(label, 0) / p_total
        q_val = q_hist.get(label, 0) / q_total
        tv += abs(p_val - q_val)

    return tv / 2


def l1_error(
    true_hist: Dict[str, int],
    est_hist: Dict[str, int],
    labels: List[str] = None
) -> float:
    """
    Compute L1 error (sum of absolute differences) between histograms.

    L1 = sum |H_true[l] - H_est[l]|

    Args:
        true_hist: True histogram
        est_hist: Estimated histogram
        labels: Ordered labels

    Returns:
        L1 error
    """
    if labels is None:
        labels = list(true_hist.keys())

    error = 0.0
    for label in labels:
        true_val = true_hist.get(label, 0)
        est_val = est_hist.get(label, 0)
        error += abs(true_val - est_val)

    return error


def per_class_relative_error(
    true_hist: Dict[str, int],
    est_hist: Dict[str, int],
    labels: List[str] = None,
    epsilon: float = 1.0
) -> Dict[str, float]:
    """
    Compute per-class relative error.

    RelError[l] = |H_true[l] - H_est[l]| / max(H_true[l], epsilon)

    Args:
        true_hist: True histogram
        est_hist: Estimated histogram
        labels: Ordered labels
        epsilon: Minimum denominator to avoid division by zero

    Returns:
        Dictionary of per-class relative errors
    """
    if labels is None:
        labels = list(true_hist.keys())

    errors = {}
    for label in labels:
        true_val = true_hist.get(label, 0)
        est_val = est_hist.get(label, 0)
        denom = max(true_val, epsilon)
        errors[label] = abs(true_val - est_val) / denom

    return errors


def mean_relative_error(
    true_hist: Dict[str, int],
    est_hist: Dict[str, int],
    labels: List[str] = None
) -> float:
    """
    Compute mean relative error across all classes.

    Args:
        true_hist: True histogram
        est_hist: Estimated histogram
        labels: Ordered labels

    Returns:
        Mean relative error
    """
    per_class = per_class_relative_error(true_hist, est_hist, labels)
    return np.mean(list(per_class.values()))


def compute_all_metrics(
    true_hist: Dict[str, int],
    est_hist: Dict[str, int],
    labels: List[str] = None
) -> Dict[str, float]:
    """
    Compute all metrics at once.

    Args:
        true_hist: True histogram
        est_hist: Estimated histogram
        labels: Ordered labels

    Returns:
        Dictionary with all metric values
    """
    return {
        "kl_divergence": kl_divergence(true_hist, est_hist, labels),
        "tv_distance": tv_distance(true_hist, est_hist, labels),
        "l1_error": l1_error(true_hist, est_hist, labels),
        "mean_relative_error": mean_relative_error(true_hist, est_hist, labels),
    }


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval.

    Args:
        values: List of metric values from multiple trials
        confidence: Confidence level (default 95%)

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0

    # Use t-distribution for small samples
    from scipy import stats
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1) if n > 1 else 0

    margin = t_val * std / np.sqrt(n) if n > 1 else 0

    return mean, mean - margin, mean + margin


def aggregate_trial_metrics(
    trial_metrics: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics from multiple trials with confidence intervals.

    Args:
        trial_metrics: List of metric dictionaries from each trial

    Returns:
        Dictionary with mean and CI for each metric
    """
    if not trial_metrics:
        return {}

    metric_names = trial_metrics[0].keys()
    aggregated = {}

    for metric in metric_names:
        values = [tm[metric] for tm in trial_metrics if not np.isinf(tm[metric])]
        if values:
            mean, ci_lower, ci_upper = compute_confidence_interval(values)
            aggregated[metric] = {
                "mean": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "std": np.std(values),
                "n_valid": len(values),
            }
        else:
            aggregated[metric] = {
                "mean": float('inf'),
                "ci_lower": float('inf'),
                "ci_upper": float('inf'),
                "std": 0,
                "n_valid": 0,
            }

    return aggregated
