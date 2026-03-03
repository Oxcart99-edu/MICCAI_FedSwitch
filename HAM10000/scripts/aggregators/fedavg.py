"""
FedAvg aggregator.
"""

from __future__ import annotations

from typing import List, Optional

from .base import Aggregator, StateDict, weighted_average_state_dicts


class FedAvgAggregator(Aggregator):
    name = "fedavg"

    def compute_client_weights(
        self,
        client_sizes: List[int],
        class_counts: Optional[List[dict]] = None,
        num_classes: Optional[int] = None,
    ) -> List[float]:
        if not client_sizes:
            return []
        total = float(sum(client_sizes))
        if total == 0:
            raise ValueError("Total client samples is zero; cannot compute FedAvg weights.")
        return [size / total for size in client_sizes]

    def aggregate(
        self,
        state_dicts: List[StateDict],
        client_sizes: List[int],
        class_counts: Optional[List[dict]] = None,
        num_classes: Optional[int] = None,
        client_weights: Optional[List[float]] = None,
    ) -> StateDict:
        if len(state_dicts) != len(client_sizes):
            raise ValueError("state_dicts and client_sizes must have the same length.")
        weights = client_weights or self.compute_client_weights(client_sizes)
        if len(weights) != len(state_dicts):
            raise ValueError("state_dicts and computed weights must have the same length.")
        return weighted_average_state_dicts(state_dicts, weights)
