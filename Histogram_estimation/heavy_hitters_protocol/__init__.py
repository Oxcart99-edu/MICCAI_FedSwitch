"""
Privacy-preserving histogram estimation via Shamir secret sharing
and distributed discrete Laplace noise (Pólya decomposition).
"""

from .config import DatasetConfig, ExperimentConfig
from .crypto import ShamirSecretSharing, compute_lagrange_coefficients
from .privacy import clip_histogram, clip_histogram_uniform
from .distributed_noise import (
    sample_partial_noise,
    sample_polya,
    compute_trigger_round,
    compute_target_cohort,
    compute_dp_variance,
)
from .data_loaders import load_client_data, compute_aggregate_histogram
from .protocol import Client, Decryptor, Coordinator, ProtocolResult
from .metrics import kl_divergence, tv_distance, l1_error, compute_all_metrics
from .experiment import ExperimentRunner, ConfigurationResult

__version__ = "2.0.0"
