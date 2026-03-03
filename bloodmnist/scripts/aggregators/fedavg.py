"""FedAvg client weighting."""

from __future__ import annotations

from .base import normalize_positive_weights


def compute_fedavg_weights(client_sizes: list[int]) -> list[float]:
    return normalize_positive_weights(client_sizes)
