"""
Configuration classes for datasets and experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


# Ground truth distributions for each dataset
GROUND_TRUTH = {
    "ham10000": {
        "akiec": 327,
        "bcc": 514,
        "bkl": 1099,
        "df": 115,
        "mel": 1113,
        "nv": 6705,
        "vasc": 142,
    },
    "mosmed": {
        "CT-0": 254,
        "CT-1": 684,
        "CT-2": 125,
        "CT-3": 45,
        "CT-4": 2,
    },
    "arrhythmia": {
        "0": 81424,
        "1": 2498,
        "2": 6506,
        "3": 722,
        "4": 7224,
    },
}

# Label ordering for each dataset
LABEL_ORDER = {
    "ham10000": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    "mosmed": ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"],
    "arrhythmia": ["0", "1", "2", "3", "4"],
}


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""

    name: str
    n_clients: int
    labels: List[str]
    ground_truth: Dict[str, int]
    data_path: Path
    gamma: float = 1.0  # Fraction of clients per round
    rounds: int = 1  # Number of buffering rounds
    sampling_with_replacement: bool = False  # Sample clients with reinsertion across rounds
    count_client_once: bool = False  # If True, each client contributes at most once across all rounds

    @classmethod
    def ham10000(cls, base_path: Path, alpha: str = "0_1", seed: int = 42) -> "DatasetConfig":
        """Create HAM10000 dataset config."""
        return cls(
            name="ham10000",
            n_clients=100,
            labels=LABEL_ORDER["ham10000"],
            ground_truth=GROUND_TRUTH["ham10000"],
            data_path=base_path / f"HAM_10000_client_splits/alpha_{alpha}/distributions_json/clients_summary_alpha_{alpha}_seed_{seed}.json",
            gamma=0.1,  # ~10% of clients per round
            rounds=7,   # 7 rounds to collect all clients
        )

    @classmethod
    def mosmed(cls, base_path: Path, alpha: str = "0_1", seed: int = 42) -> "DatasetConfig":
        """Create MOSMED dataset config."""
        return cls(
            name="mosmed",
            n_clients=10,
            labels=LABEL_ORDER["mosmed"],
            ground_truth=GROUND_TRUTH["mosmed"],
            data_path=base_path / f"MOSMED_client_splits/alpha_{alpha}/distributions_json/clients_summary_alpha_{alpha}_seed_{seed}.json",
            gamma=1.0,
            rounds=1,
        )

    @classmethod
    def arrhythmia(cls, base_path: Path, gamma: float = 1.0, rounds: int = 1) -> "DatasetConfig":
        """Create Arrhythmia dataset config."""
        return cls(
            name="arrhythmia",
            n_clients=48,
            labels=LABEL_ORDER["arrhythmia"],
            ground_truth=GROUND_TRUTH["arrhythmia"],
            data_path=base_path / "Arrythmia_client_splits/patient_label_counts.json",
            gamma=gamma,
            rounds=rounds,
        )


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""

    dataset: DatasetConfig
    C_values: List[int]  # Clipping thresholds
    epsilon_values: List[float]  # Privacy budgets
    failure_rates: List[float]  # Decryptor failure rates
    n_trials: int = 100  # Number of trials per configuration
    seed: int = 42

    @classmethod
    def ham10000_config(cls, base_path: Path) -> "ExperimentConfig":
        """Default experiment config for HAM10000."""
        return cls(
            dataset=DatasetConfig.ham10000(base_path),
            C_values=[50, 100, 200],
            epsilon_values=[0.5, 1.0, 2.0, 4.0, 8.0],
            failure_rates=[0.0, 0.10, 0.20, 0.33],
            n_trials=100,
        )

    @classmethod
    def mosmed_config(cls, base_path: Path) -> "ExperimentConfig":
        """Default experiment config for MOSMED."""
        return cls(
            dataset=DatasetConfig.mosmed(base_path),
            C_values=[20, 40],
            epsilon_values=[0.5, 1.0, 2.0, 4.0, 8.0],
            failure_rates=[0.0, 0.10, 0.20, 0.33],
            n_trials=100,
        )

    @classmethod
    def arrhythmia_config(cls, base_path: Path, gamma: float = 1.0, rounds: int = 1) -> "ExperimentConfig":
        """Default experiment config for Arrhythmia."""
        return cls(
            dataset=DatasetConfig.arrhythmia(base_path, gamma=gamma, rounds=rounds),
            C_values=[100, 500, 1000],
            epsilon_values=[0.5, 1.0, 2.0, 4.0, 8.0],
            failure_rates=[0.0, 0.10, 0.20, 0.33],
            n_trials=100,
        )
