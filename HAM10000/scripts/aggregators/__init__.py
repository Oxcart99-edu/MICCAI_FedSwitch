"""
Aggregator registry for federated training scripts.
"""

from __future__ import annotations

from .base import Aggregator, StateDict
from .fedavg import FedAvgAggregator
from .fediic import FedIICAggregator
from .fedlc import FedLCAggregator
from .fedprox import FedProxAggregator
from .stratified import StratifiedAggregator
from .switch_stratified import SwitchStratifiedAggregator

AGGREGATOR_REGISTRY = {
    FedAvgAggregator.name: FedAvgAggregator,
    FedIICAggregator.name: FedIICAggregator,
    FedLCAggregator.name: FedLCAggregator,
    FedProxAggregator.name: FedProxAggregator,
    SwitchStratifiedAggregator.name: SwitchStratifiedAggregator,
    StratifiedAggregator.name: StratifiedAggregator,
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
    "FedIICAggregator",
    "FedLCAggregator",
    "FedProxAggregator",
    "SwitchStratifiedAggregator",
    "StratifiedAggregator",
    "AGGREGATOR_REGISTRY",
    "get_aggregator",
]
