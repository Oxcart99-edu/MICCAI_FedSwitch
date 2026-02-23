"""
Experiment runner: runs multiple trials and aggregates metrics.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np

from .config import DatasetConfig
from .data_loaders import compute_aggregate_histogram
from .protocol import Coordinator, ProtocolResult
from .metrics import compute_all_metrics, aggregate_trial_metrics


@dataclass
class ConfigurationResult:
    """Aggregated result for a single (C, ε, failure_rate) configuration."""
    dataset: str
    clipping_threshold: int
    epsilon: float
    failure_rate: float
    gamma: float
    alpha: float
    n_rounds: int
    clipping_method: str
    n_trials: int
    true_histogram: Dict[str, int]
    target_cohort: int
    trigger_round: int
    aggregated_metrics: Dict[str, Dict[str, float]]
    sample_estimated_histograms: List[Dict[str, int]] = field(default_factory=list)
    sample_noise_values: List[Dict[str, int]] = field(default_factory=list)
    sample_client_sets: Optional[List[List[List[int]]]] = None
    per_trial_metrics: Optional[List[Dict[str, float]]] = None


class ExperimentRunner:
    """Runs histogram estimation experiments over multiple trials."""

    def __init__(self, base_path: Path, output_dir: Path = None, verbose: bool = True):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir) if output_dir else self.base_path / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def run_configuration(
        self,
        config: DatasetConfig,
        client_histograms: List[Dict[str, int]],
        clipping_threshold: int,
        epsilon: float,
        failure_rate: float,
        n_trials: int,
        alpha: float = 0.9,
        clipping_method: str = "proportional",
        base_seed: int = 42,
        save_samples: int = 5,
        fixed_client_sets_per_round: Optional[List[List[int]]] = None,
    ) -> ConfigurationResult:
        """
        Run multiple trials for a single configuration.

        Args:
            config: Dataset configuration.
            client_histograms: Per-client label count dictionaries.
            clipping_threshold: Clipping value C.
            epsilon: Privacy budget ε.
            failure_rate: Fraction of decryptors that fail.
            n_trials: Number of independent trials.
            alpha: Target coverage fraction for noise calibration.
            clipping_method: Clipping strategy.
            base_seed: Base random seed.
            save_samples: How many individual trial outputs to keep.
            fixed_client_sets_per_round: Optional pre-determined client IDs per round.

        Returns:
            ConfigurationResult with aggregated metrics.
        """
        trial_metrics = []
        sample_histograms = []
        sample_noise_values = []
        sample_client_sets = []

        n_decryptors = config.n_clients

        # Build one coordinator to read off target_cohort / trigger_round
        first_coord = Coordinator(
            n_clients=config.n_clients,
            n_decryptors=n_decryptors,
            labels=config.labels,
            clipping_threshold=clipping_threshold,
            epsilon=epsilon,
            gamma=config.gamma,
            n_rounds=config.rounds,
            alpha=alpha,
            sampling_with_replacement=getattr(config, "sampling_with_replacement", False),
            count_client_once=getattr(config, "count_client_once", False),
            fixed_client_sets_per_round=fixed_client_sets_per_round,
            failure_rate=failure_rate,
            clipping_method=clipping_method,
            seed=base_seed,
        )
        target_cohort = first_coord.target_cohort
        trigger_round = first_coord.trigger_round

        for trial in range(n_trials):
            seed = base_seed + trial

            coordinator = Coordinator(
                n_clients=config.n_clients,
                n_decryptors=n_decryptors,
                labels=config.labels,
                clipping_threshold=clipping_threshold,
                epsilon=epsilon,
                gamma=config.gamma,
                n_rounds=config.rounds,
                alpha=alpha,
                sampling_with_replacement=getattr(config, "sampling_with_replacement", False),
                count_client_once=getattr(config, "count_client_once", False),
                fixed_client_sets_per_round=fixed_client_sets_per_round,
                failure_rate=failure_rate,
                clipping_method=clipping_method,
                seed=seed,
            )

            result = coordinator.run_protocol(client_histograms)

            true_hist = compute_aggregate_histogram(client_histograms, config.labels)
            metrics = compute_all_metrics(true_hist, result.estimated_histogram, config.labels)
            trial_metrics.append(metrics)

            if trial < save_samples:
                sample_histograms.append(result.estimated_histogram)
                sample_noise_values.append(result.noise_values)
                sample_client_sets.append(result.client_sets_per_round)

        aggregated = aggregate_trial_metrics(trial_metrics)
        true_hist = compute_aggregate_histogram(client_histograms, config.labels)

        return ConfigurationResult(
            dataset=config.name,
            clipping_threshold=clipping_threshold,
            epsilon=epsilon,
            failure_rate=failure_rate,
            gamma=config.gamma,
            alpha=alpha,
            n_rounds=config.rounds,
            clipping_method=clipping_method,
            n_trials=n_trials,
            true_histogram=true_hist,
            target_cohort=target_cohort,
            trigger_round=trigger_round,
            aggregated_metrics=aggregated,
            sample_estimated_histograms=sample_histograms,
            sample_noise_values=sample_noise_values,
            sample_client_sets=sample_client_sets,
            per_trial_metrics=trial_metrics,
        )

    def save_results(self, results: List[ConfigurationResult], filename: str):
        """Save results to JSON file."""
        output_path = self.output_dir / filename

        serializable = []
        for r in results:
            serializable.append({
                "dataset": r.dataset,
                "clipping_threshold": r.clipping_threshold,
                "epsilon": r.epsilon,
                "failure_rate": r.failure_rate,
                "gamma": r.gamma,
                "alpha": r.alpha,
                "n_rounds": r.n_rounds,
                "clipping_method": r.clipping_method,
                "n_trials": r.n_trials,
                "true_histogram": r.true_histogram,
                "target_cohort": r.target_cohort,
                "trigger_round": r.trigger_round,
                "aggregated_metrics": r.aggregated_metrics,
                "sample_estimated_histograms": r.sample_estimated_histograms,
                "sample_noise_values": r.sample_noise_values,
                "sample_client_sets": r.sample_client_sets,
            })

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        if self.verbose:
            print(f"Results saved to {output_path}")
