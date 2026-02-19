"""
Data loaders for HAM10000, MOSMED, and Arrhythmia datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from .config import DatasetConfig, LABEL_ORDER, GROUND_TRUTH


def load_client_data(config: DatasetConfig) -> List[Dict[str, int]]:
    """
    Load client histogram data based on dataset configuration.

    Args:
        config: Dataset configuration

    Returns:
        List of client histograms (label -> count)
    """
    if config.name == "ham10000":
        return load_ham10000(config.data_path)
    elif config.name == "mosmed":
        return load_mosmed(config.data_path)
    elif config.name == "arrhythmia":
        return load_arrhythmia(config.data_path)
    else:
        raise ValueError(f"Unknown dataset: {config.name}")


def load_ham10000(data_path: Path) -> List[Dict[str, int]]:
    """
    Load HAM10000 client distributions.

    Args:
        data_path: Path to clients_summary_*.json file

    Returns:
        List of client histograms
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    client_distributions = data["client_label_distribution"]

    # Ensure all labels are present (with 0 if missing)
    labels = LABEL_ORDER["ham10000"]
    normalized = []

    for client_hist in client_distributions:
        full_hist = {label: client_hist.get(label, 0) for label in labels}
        normalized.append(full_hist)

    return normalized


def load_mosmed(data_path: Path) -> List[Dict[str, int]]:
    """
    Load MOSMED client distributions.

    Args:
        data_path: Path to clients_summary_*.json file

    Returns:
        List of client histograms
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    client_distributions = data["client_label_distribution"]

    # Ensure all labels are present
    labels = LABEL_ORDER["mosmed"]
    normalized = []

    for client_hist in client_distributions:
        full_hist = {label: client_hist.get(label, 0) for label in labels}
        normalized.append(full_hist)

    return normalized


def load_arrhythmia(data_path: Path) -> List[Dict[str, int]]:
    """
    Load Arrhythmia patient/client distributions.

    Args:
        data_path: Path to patient_label_counts.json file

    Returns:
        List of client histograms (one per patient)
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    labels = LABEL_ORDER["arrhythmia"]
    normalized = []

    # Data is keyed by patient ID, values are label counts
    for patient_id, patient_hist in sorted(data.items()):
        # Convert string keys to match our label format
        full_hist = {label: patient_hist.get(label, 0) for label in labels}
        normalized.append(full_hist)

    return normalized


def compute_aggregate_histogram(client_histograms: List[Dict[str, int]], labels: List[str]) -> Dict[str, int]:
    """
    Compute the aggregate histogram by summing all client histograms.

    Args:
        client_histograms: List of client histograms
        labels: Ordered list of labels

    Returns:
        Aggregate histogram
    """
    aggregate = {label: 0 for label in labels}

    for client_hist in client_histograms:
        for label in labels:
            aggregate[label] += client_hist.get(label, 0)

    return aggregate


def histogram_to_distribution(histogram: Dict[str, int]) -> Dict[str, float]:
    """
    Convert a histogram to a probability distribution.

    Args:
        histogram: Count dictionary

    Returns:
        Probability distribution
    """
    total = sum(histogram.values())
    if total == 0:
        return {label: 0.0 for label in histogram}
    return {label: count / total for label, count in histogram.items()}


def get_ground_truth(dataset_name: str) -> Tuple[Dict[str, int], List[str]]:
    """
    Get ground truth histogram and labels for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Tuple of (ground truth histogram, ordered labels)
    """
    return GROUND_TRUTH[dataset_name].copy(), LABEL_ORDER[dataset_name].copy()


def validate_loaded_data(client_histograms: List[Dict[str, int]], config: DatasetConfig) -> bool:
    """
    Validate that loaded data matches expected configuration.

    Args:
        client_histograms: Loaded client data
        config: Expected configuration

    Returns:
        True if valid
    """
    # Check number of clients
    if len(client_histograms) != config.n_clients:
        print(f"Warning: Expected {config.n_clients} clients, got {len(client_histograms)}")
        return False

    # Check labels
    for i, client_hist in enumerate(client_histograms):
        for label in config.labels:
            if label not in client_hist:
                print(f"Warning: Client {i} missing label {label}")
                return False

    return True


def print_data_summary(client_histograms: List[Dict[str, int]], config: DatasetConfig):
    """Print a summary of the loaded data."""
    aggregate = compute_aggregate_histogram(client_histograms, config.labels)

    print(f"\n{'='*50}")
    print(f"Dataset: {config.name.upper()}")
    print(f"{'='*50}")
    print(f"Number of clients: {len(client_histograms)}")
    print(f"Labels: {config.labels}")
    print(f"\nAggregate histogram from loaded data:")
    for label in config.labels:
        print(f"  {label}: {aggregate[label]}")
    print(f"Total samples: {sum(aggregate.values())}")

    print(f"\nGround truth:")
    for label in config.labels:
        print(f"  {label}: {config.ground_truth[label]}")
    print(f"Total samples: {sum(config.ground_truth.values())}")
    print(f"{'='*50}\n")
