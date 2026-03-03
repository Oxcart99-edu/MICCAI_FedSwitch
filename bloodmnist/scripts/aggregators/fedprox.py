"""FedProx uses FedAvg aggregation weights."""

from __future__ import annotations

from .base import normalize_positive_weights


def compute_fedprox_weights(client_sizes: list[int]) -> list[float]:
    return normalize_positive_weights(client_sizes)
