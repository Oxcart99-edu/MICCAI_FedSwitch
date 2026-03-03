"""
Aggregator registry for federated training scripts.
"""

from __future__ import annotations

from .base import Aggregator, StateDict
from .fedavg import FedAvgAggregator
from .fedswitch import FedSwitchAggregator
from .fediic import FedIICAggregator
from .fedlc import FedLCAggregator
from .fedprox import FedProxAggregator

AGGREGATOR_REGISTRY = {
    FedAvgAggregator.name: FedAvgAggregator,
    FedSwitchAggregator.name: FedSwitchAggregator,
    FedIICAggregator.name: FedIICAggregator,
    FedLCAggregator.name: FedLCAggregator,
    FedProxAggregator.name: FedProxAggregator,
}


def get_aggregator(name: str) -> Aggregator:
    key = name.lower()
    if key not in AGGREGATOR_REGISTRY:
        available = ", ".join(sorted(AGGREGATOR_REGISTRY.keys()))
        raise ValueError(f"Unknown aggregator '{name}'. Available: {available}")
    return AGGREGATOR_REGISTRY[key]()


__all__ = [
    "Aggregator",
    "StateDict",
    "FedAvgAggregator",
    "FedSwitchAggregator",
    "FedIICAggregator",
    "FedLCAggregator",
    "FedProxAggregator",
    "AGGREGATOR_REGISTRY",
    "get_aggregator",
]
