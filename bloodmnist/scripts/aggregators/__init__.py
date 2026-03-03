"""Aggregator registry for the BloodMNIST federated trainer."""

from .base import aggregate_state_dicts, normalize_positive_weights
from .fedavg import compute_fedavg_weights
from .fediic import (
    FedIICCNN,
    build_global_prototypes,
    compute_class_loss_vector,
    compute_fediic_weights,
    local_train_fediic,
)
from .fedswitch import compute_fedswitch_weights
from .fedprox import compute_fedprox_weights

SUPPORTED_AGGREGATORS = ("fedavg", "fedprox", "fedswitch", "fediic")

__all__ = [
    "SUPPORTED_AGGREGATORS",
    "FedIICCNN",
    "aggregate_state_dicts",
    "build_global_prototypes",
    "compute_class_loss_vector",
    "compute_fedavg_weights",
    "compute_fediic_weights",
    "compute_fedswitch_weights",
    "compute_fedprox_weights",
    "local_train_fediic",
    "normalize_positive_weights",
]
