"""Shared utilities for federated aggregation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch


def normalize_positive_weights(weights: Sequence[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return [float(weight) / total for weight in weights]


def aggregate_state_dicts(
    client_states: Sequence[dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> OrderedDict[str, torch.Tensor]:
    if len(client_states) != len(weights):
        raise ValueError("client_states and weights must have the same length.")
    if not client_states:
        raise ValueError("At least one client state is required.")

    normalized_weights = normalize_positive_weights(weights)
    aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in client_states[0]:
        tensors = [state[key] for state in client_states]
        if not torch.is_floating_point(tensors[0]):
            aggregated[key] = tensors[0].clone()
            continue
        aggregated[key] = sum(tensor.float() * weight for tensor, weight in zip(tensors, normalized_weights))
    return aggregated
