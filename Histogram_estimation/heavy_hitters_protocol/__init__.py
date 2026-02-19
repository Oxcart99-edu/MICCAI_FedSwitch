"""
Heavy Hitters Label Distribution Estimation Protocol

A privacy-preserving protocol for estimating label distributions across
federated medical imaging datasets using:
- Shamir Secret Sharing
- Boneh Factorial Technique
- Distributed Discrete Laplace noise for differential privacy
"""

from .config import DatasetConfig, ExperimentConfig
from .crypto import ShamirSecretSharing, compute_lagrange_coefficients
from .privacy import DiscreteLaplaceNoise, clip_histogram, clip_histogram_uniform
from .distributed_noise import (
    DistributedNoiseGenerator,
    run_distributed_noise_protocol,
    generate_decryptor_noise,
    sample_partial_discrete_laplace
)
from .data_loaders import load_ham10000, load_mosmed, load_arrhythmia, load_client_data
from .protocol import Client, Decryptor, Coordinator, ProtocolResult
from .metrics import kl_divergence, tv_distance, l1_error, per_class_relative_error
from .experiment import ExperimentRunner

__version__ = "1.0.0"
__all__ = [
    "DatasetConfig",
    "ExperimentConfig",
    "ShamirSecretSharing",
    "DiscreteLaplaceNoise",
    "clip_histogram",
    "clip_histogram_uniform",
    "DistributedNoiseGenerator",
    "run_distributed_noise_protocol",
    "generate_decryptor_noise",
    "sample_partial_discrete_laplace",
    "load_ham10000",
    "load_mosmed",
    "load_arrhythmia",
    "load_client_data",
    "Client",
    "Decryptor",
    "Coordinator",
    "ProtocolResult",
    "kl_divergence",
    "tv_distance",
    "l1_error",
    "per_class_relative_error",
    "ExperimentRunner",
]
