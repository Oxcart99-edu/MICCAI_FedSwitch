"""
Experiment runner and visualization.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np

from .config import DatasetConfig, ExperimentConfig
from .data_loaders import load_client_data, compute_aggregate_histogram
from .protocol import Coordinator, ProtocolResult
from .metrics import compute_all_metrics, aggregate_trial_metrics


@dataclass
class TrialResult:
    """Result of a single trial."""
    estimated_histogram: Dict[str, int]
    metrics: Dict[str, float]
    client_sets_per_round: List[List[int]]
    noise_values: Dict[str, int]
    runtime_seconds: float


@dataclass
class ConfigurationResult:
    """Result for a single configuration (C, epsilon, failure_rate)."""
    dataset: str
    clipping_threshold: int
    epsilon: float
    failure_rate: float
    gamma: float
    n_rounds: int
    clipping_method: str
    n_trials: int
    true_histogram: Dict[str, int]
    aggregated_metrics: Dict[str, Dict[str, float]]
    sample_estimated_histograms: List[Dict[str, int]] = field(default_factory=list)
    sample_noise_values: List[Dict[str, int]] = field(default_factory=list)
    sample_client_sets: Optional[List[List[List[int]]]] = None
    per_trial_metrics: Optional[List[Dict[str, float]]] = None


class ExperimentRunner:
    """
    Runs experiments for the Heavy Hitters protocol.
    """

    def __init__(self, base_path: Path, output_dir: Path = None, verbose: bool = True):
        """
        Initialize the experiment runner.

        Args:
            base_path: Base path for data files
            output_dir: Directory for output files
            verbose: Print progress
        """
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
        clipping_method: str = "proportional",
        base_seed: int = 42,
        save_samples: int = 5,
        fixed_client_sets_per_round: Optional[List[List[int]]] = None,
    ) -> ConfigurationResult:
        """
        Run multiple trials for a single configuration.

        Args:
            config: Dataset configuration
            client_histograms: Loaded client data
            clipping_threshold: C value
            epsilon: Privacy budget
            failure_rate: Decryptor failure rate
            n_trials: Number of trials
            clipping_method: "proportional", "uniform", or "uniform_with_replacement"
            base_seed: Base random seed
            save_samples: Number of sample estimated histograms to save
            fixed_client_sets_per_round: Optional fixed picked IDs for each round.
                If provided, protocol sampling uses these exact client IDs.

        Returns:
            Configuration result with aggregated metrics
        """
        trial_metrics = []
        sample_histograms = []
        sample_noise_values = []
        sample_client_sets = []

        # Use number of decryptors equal to number of clients
        n_decryptors = config.n_clients

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
                sampling_with_replacement=getattr(config, "sampling_with_replacement", False),
                count_client_once=getattr(config, "count_client_once", False),
                fixed_client_sets_per_round=fixed_client_sets_per_round,
                failure_rate=failure_rate,
                clipping_method=clipping_method,
                seed=seed
            )

            start_time = time.time()
            result = coordinator.run_protocol(client_histograms, return_detailed=True)
            runtime = time.time() - start_time

            # Compute metrics against actual aggregate (true distribution from loaded data)
            true_hist = compute_aggregate_histogram(client_histograms, config.labels)
            metrics = compute_all_metrics(
                true_hist,
                result.estimated_histogram,
                config.labels
            )
            trial_metrics.append(metrics)

            # Save sample histograms, noise values, and client sets
            if trial < save_samples:
                sample_histograms.append(result.estimated_histogram)
                sample_noise_values.append(result.noise_values)
                sample_client_sets.append(result.client_sets_per_round)

        # Aggregate results
        aggregated = aggregate_trial_metrics(trial_metrics)
        true_hist = compute_aggregate_histogram(client_histograms, config.labels)

        return ConfigurationResult(
            dataset=config.name,
            clipping_threshold=clipping_threshold,
            epsilon=epsilon,
            failure_rate=failure_rate,
            gamma=config.gamma,
            n_rounds=config.rounds,
            clipping_method=clipping_method,
            n_trials=n_trials,
            true_histogram=true_hist,
            aggregated_metrics=aggregated,
            sample_estimated_histograms=sample_histograms,
            sample_noise_values=sample_noise_values,
            sample_client_sets=sample_client_sets,
            per_trial_metrics=trial_metrics
        )

    def run_experiment(
        self,
        exp_config: ExperimentConfig,
        clipping_method: str = "proportional"
    ) -> List[ConfigurationResult]:
        """
        Run full experiment with all configurations.

        Args:
            exp_config: Experiment configuration
            clipping_method: "proportional", "uniform", or "uniform_with_replacement"

        Returns:
            List of results for all configurations
        """
        dataset_config = exp_config.dataset

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running experiment for {dataset_config.name.upper()}")
            print(f"gamma={dataset_config.gamma}, rounds={dataset_config.rounds}")
            print(f"clipping_method={clipping_method}")
            print(f"{'='*60}")

        # Load data
        client_histograms = load_client_data(dataset_config)

        if self.verbose:
            aggregate = compute_aggregate_histogram(client_histograms, dataset_config.labels)
            print(f"Loaded {len(client_histograms)} clients")
            print(f"Aggregate histogram: {aggregate}")
            print(f"Ground truth: {dataset_config.ground_truth}")

        results = []
        total_configs = (
            len(exp_config.C_values) *
            len(exp_config.epsilon_values) *
            len(exp_config.failure_rates)
        )
        config_num = 0

        for C in exp_config.C_values:
            for epsilon in exp_config.epsilon_values:
                for failure_rate in exp_config.failure_rates:
                    config_num += 1

                    if self.verbose:
                        print(f"\n[{config_num}/{total_configs}] "
                              f"C={C}, epsilon={epsilon}, failure={failure_rate:.0%}")

                    result = self.run_configuration(
                        config=dataset_config,
                        client_histograms=client_histograms,
                        clipping_threshold=C,
                        epsilon=epsilon,
                        failure_rate=failure_rate,
                        n_trials=exp_config.n_trials,
                        clipping_method=clipping_method,
                        base_seed=exp_config.seed
                    )

                    results.append(result)

                    if self.verbose:
                        kl = result.aggregated_metrics.get('kl_divergence', {})
                        print(f"  KL divergence: {kl.get('mean', 'N/A'):.4f} "
                              f"[{kl.get('ci_lower', 'N/A'):.4f}, {kl.get('ci_upper', 'N/A'):.4f}]")
                        if result.sample_estimated_histograms:
                            print(f"  Sample estimate: {result.sample_estimated_histograms[0]}")

        return results

    def save_results(self, results: List[ConfigurationResult], filename: str):
        """Save results to JSON file."""
        output_path = self.output_dir / filename

        # Convert to serializable format
        serializable = []
        for r in results:
            d = {
                "dataset": r.dataset,
                "clipping_threshold": r.clipping_threshold,
                "epsilon": r.epsilon,
                "failure_rate": r.failure_rate,
                "gamma": r.gamma,
                "n_rounds": r.n_rounds,
                "clipping_method": r.clipping_method,
                "n_trials": r.n_trials,
                "true_histogram": r.true_histogram,
                "aggregated_metrics": r.aggregated_metrics,
                "sample_estimated_histograms": r.sample_estimated_histograms,
                "sample_noise_values": r.sample_noise_values,
                "sample_client_sets": r.sample_client_sets,
            }
            serializable.append(d)

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        if self.verbose:
            print(f"\nResults saved to {output_path}")

    def run_all_datasets(
        self,
        clipping_method: str = "proportional"
    ) -> Dict[str, List[ConfigurationResult]]:
        """Run experiments on all three datasets."""
        all_results = {}

        # HAM10000
        ham_config = ExperimentConfig.ham10000_config(self.base_path)
        ham_results = self.run_experiment(ham_config, clipping_method)
        all_results['ham10000'] = ham_results
        self.save_results(ham_results, 'ham10000_results.json')

        # MOSMED
        mosmed_config = ExperimentConfig.mosmed_config(self.base_path)
        mosmed_results = self.run_experiment(mosmed_config, clipping_method)
        all_results['mosmed'] = mosmed_results
        self.save_results(mosmed_results, 'mosmed_results.json')

        # Arrhythmia
        arr_config = ExperimentConfig.arrhythmia_config(self.base_path)
        arr_results = self.run_experiment(arr_config, clipping_method)
        all_results['arrhythmia'] = arr_results
        self.save_results(arr_results, 'arrhythmia_results.json')

        return all_results


def create_visualizations(results_dir: Path, output_dir: Path = None):
    """
    Create visualization plots from saved results.

    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory for output plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualizations")
        return

    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir

    datasets = ['ham10000', 'mosmed', 'arrhythmia']

    for dataset in datasets:
        results_file = results_dir / f'{dataset}_results.json'
        if not results_file.exists():
            print(f"Skipping {dataset} - no results file found")
            continue

        with open(results_file, 'r') as f:
            results = json.load(f)

        # Plot KL divergence vs epsilon for each C value
        plot_kl_vs_epsilon(results, dataset, output_dir)

        # Plot failure rate impact
        plot_failure_impact(results, dataset, output_dir)

    print(f"Visualizations saved to {output_dir}")


def plot_kl_vs_epsilon(results: List[dict], dataset: str, output_dir: Path):
    """Plot KL divergence vs epsilon for different C values."""
    import matplotlib.pyplot as plt

    # Group by C and failure_rate=0
    c_values = sorted(set(r['clipping_threshold'] for r in results))

    fig, ax = plt.subplots(figsize=(10, 6))

    for C in c_values:
        # Filter results for this C with no failures
        filtered = [r for r in results
                   if r['clipping_threshold'] == C and r['failure_rate'] == 0.0]
        filtered.sort(key=lambda x: x['epsilon'])

        epsilons = [r['epsilon'] for r in filtered]
        kl_means = [r['aggregated_metrics']['kl_divergence']['mean'] for r in filtered]
        kl_lower = [r['aggregated_metrics']['kl_divergence']['ci_lower'] for r in filtered]
        kl_upper = [r['aggregated_metrics']['kl_divergence']['ci_upper'] for r in filtered]

        ax.plot(epsilons, kl_means, 'o-', label=f'C={C}')
        ax.fill_between(epsilons, kl_lower, kl_upper, alpha=0.2)

    ax.set_xlabel('Privacy Budget (ε)')
    ax.set_ylabel('KL Divergence')
    ax.set_title(f'{dataset.upper()} - KL Divergence vs Privacy Budget')
    ax.legend()
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset}_kl_vs_epsilon.png', dpi=150)
    plt.close()


def plot_failure_impact(results: List[dict], dataset: str, output_dir: Path):
    """Plot impact of decryptor failure rate on KL divergence."""
    import matplotlib.pyplot as plt

    # Use middle C value and middle epsilon
    c_values = sorted(set(r['clipping_threshold'] for r in results))
    C = c_values[len(c_values) // 2]

    epsilon_values = sorted(set(r['epsilon'] for r in results))
    epsilon = epsilon_values[len(epsilon_values) // 2]

    # Filter and sort by failure rate
    filtered = [r for r in results
               if r['clipping_threshold'] == C and r['epsilon'] == epsilon]
    filtered.sort(key=lambda x: x['failure_rate'])

    failure_rates = [r['failure_rate'] * 100 for r in filtered]
    kl_means = [r['aggregated_metrics']['kl_divergence']['mean'] for r in filtered]
    kl_lower = [r['aggregated_metrics']['kl_divergence']['ci_lower'] for r in filtered]
    kl_upper = [r['aggregated_metrics']['kl_divergence']['ci_upper'] for r in filtered]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(range(len(failure_rates)), kl_means, yerr=[
        [m - l for m, l in zip(kl_means, kl_lower)],
        [u - m for m, u in zip(kl_means, kl_upper)]
    ], capsize=5, alpha=0.7)

    ax.set_xticks(range(len(failure_rates)))
    ax.set_xticklabels([f'{fr:.0f}%' for fr in failure_rates])
    ax.set_xlabel('Decryptor Failure Rate')
    ax.set_ylabel('KL Divergence')
    ax.set_title(f'{dataset.upper()} - Impact of Decryptor Failures\n(C={C}, ε={epsilon})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset}_failure_impact.png', dpi=150)
    plt.close()


def print_summary_table(all_results: Dict[str, List[ConfigurationResult]]):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}")
        print("-" * 80)
        print(f"{'C':>6} {'epsilon':>8} {'failure':>8} {'gamma':>6} {'rounds':>6} {'KL mean':>10} {'KL CI':>24}")
        print("-" * 80)

        for r in results:
            kl = r.aggregated_metrics.get('kl_divergence', {})
            mean = kl.get('mean', float('inf'))
            ci_l = kl.get('ci_lower', float('inf'))
            ci_u = kl.get('ci_upper', float('inf'))

            print(f"{r.clipping_threshold:>6} {r.epsilon:>8.1f} {r.failure_rate:>8.0%} "
                  f"{r.gamma:>6.2f} {r.n_rounds:>6} "
                  f"{mean:>10.4f} [{ci_l:.4f}, {ci_u:.4f}]")
