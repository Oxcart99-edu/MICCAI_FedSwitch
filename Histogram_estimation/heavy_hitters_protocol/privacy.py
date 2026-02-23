"""
Privacy mechanisms: histogram clipping and variance computation.
"""

import numpy as np
from typing import Dict, List
from math import exp


def clip_histogram(
    histogram: Dict[str, int], C: int, method: str = "proportional"
) -> Dict[str, int]:
    """
    Clip histogram counts to bound ℓ1-sensitivity to exactly C.

    Args:
        histogram: Label counts for a client.
        C: Maximum total samples (clipping threshold).
        method: "proportional" (scale + largest-remainder rounding),
                "uniform" (sample C without replacement),
                "uniform_with_replacement" (sample C with replacement).

    Returns:
        Clipped histogram with ||clipped||_1 = C (or original if total <= C).
    """
    total = sum(histogram.values())

    if total <= C:
        return histogram.copy()

    if method == "uniform":
        return clip_histogram_uniform(histogram, C)
    if method in ("uniform_with_replacement", "uniform_wr"):
        return clip_histogram_uniform(histogram, C, replace=True)

    # Proportional clipping with largest-remainder rounding.
    scale = C / total
    sorted_labels = sorted(histogram.keys())

    raw = {label: histogram[label] * scale for label in sorted_labels}
    clipped = {label: int(raw[label]) for label in sorted_labels}
    remaining = C - sum(clipped.values())

    if remaining > 0:
        remainders = sorted(
            sorted_labels,
            key=lambda label: (raw[label] - clipped[label], label),
            reverse=True,
        )
        for label in remainders[:remaining]:
            clipped[label] += 1

    return clipped


def clip_histogram_uniform(
    histogram: Dict[str, int],
    C: int,
    rng: np.random.Generator = None,
    replace: bool = False,
) -> Dict[str, int]:
    """
    Clip histogram by uniformly sampling C items from the empirical distribution.

    Args:
        histogram: Label counts for a client.
        C: Number of samples to draw.
        rng: Random generator.
        replace: If True, sample with replacement.

    Returns:
        Clipped histogram with ||clipped||_1 = C.
    """
    if rng is None:
        rng = np.random.default_rng()

    total = sum(histogram.values())
    if total <= C:
        return histogram.copy()

    labels = []
    for label, count in histogram.items():
        labels.extend([label] * count)

    sampled_indices = rng.choice(len(labels), size=C, replace=replace)
    sampled_labels = [labels[i] for i in sampled_indices]

    clipped = {label: 0 for label in histogram.keys()}
    for label in sampled_labels:
        clipped[label] += 1

    return clipped
