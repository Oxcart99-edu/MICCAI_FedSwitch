"""FedSwitch client weighting."""

from __future__ import annotations

import numpy as np

from .base import normalize_positive_weights


def compute_fedswitch_weights(
    *,
    round_idx: int,
    switch_round: int,
    step_sizes: list[int],
    step_counts: list[np.ndarray],
    target_estimate: dict[str, float],
    num_classes: int,
) -> list[float]:
    if round_idx < switch_round:
        return normalize_positive_weights(step_sizes)

    weights: list[float] = []
    per_class = 1.0 / float(num_classes)
    for counts in step_counts:
        value = 0.0
        for class_idx in range(num_classes):
            denom = float(target_estimate.get(str(class_idx), 0.0))
            if denom <= 0:
                continue
            value += (float(counts[class_idx]) / denom) * per_class
        weights.append(value)
    return normalize_positive_weights(weights)
