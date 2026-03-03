"""
Stratified aggregator.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import Aggregator, StateDict, weighted_average_state_dicts


class StratifiedAggregator(Aggregator):
    name = "stratified"

    def compute_client_weights(
        self,
        client_sizes: List[int],
        class_counts: Optional[List[dict]] = None,
        num_classes: Optional[int] = None,
    ) -> List[float]:
        if class_counts is None or num_classes is None:
            raise ValueError("class_counts and num_classes are required for stratified aggregation.")
        if len(client_sizes) != len(class_counts):
            raise ValueError("client_sizes and class_counts must have the same length.")

        totals: Dict[str, int] = {}
        for counts in class_counts:
            for label, cnt in counts.items():
                totals[label] = totals.get(label, 0) + cnt

        client_weights: List[float] = []
        for counts in class_counts:
            weight = 0.0
            for label, total_cnt in totals.items():
                if total_cnt == 0:
                    continue
                client_cnt = counts.get(label, 0)
                weight += (client_cnt / total_cnt) * (1.0 / num_classes)
            client_weights.append(weight)

        if sum(client_weights) == 0:
            raise ValueError("All client weights computed to zero; check class_counts input.")
        return client_weights

    def aggregate(
        self,
        state_dicts: List[StateDict],
        client_sizes: List[int],
        class_counts: Optional[List[dict]] = None,
        num_classes: Optional[int] = None,
        client_weights: Optional[List[float]] = None,
    ) -> StateDict:
        if class_counts is None or num_classes is None:
            raise ValueError("class_counts and num_classes are required for stratified aggregation.")
        if len(state_dicts) != len(class_counts):
            raise ValueError("state_dicts and class_counts must have the same length.")

        weights = client_weights or self.compute_client_weights(
            client_sizes, class_counts=class_counts, num_classes=num_classes
        )
        return weighted_average_state_dicts(state_dicts, weights)
