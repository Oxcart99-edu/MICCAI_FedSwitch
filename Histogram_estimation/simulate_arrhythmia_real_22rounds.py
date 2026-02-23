#!/usr/bin/env python3
"""
Arrhythmia simulation from real patient-level distributions.

No synthetic generation:
- no min-Dirichlet for client sizes
- no Dirichlet for labels
- uses original patient-by-patient histograms directly

Estimation setup (same style as previous runs):
- 16 clients per round
- 10 rounds
- random sampling with replacement across rounds
- each client contributes at most once
- random clipping by uniform sampling with replacement, C=1200
- privacy noise enabled (epsilon=4.0)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from heavy_hitters_protocol.config import DatasetConfig, LABEL_ORDER
from heavy_hitters_protocol.data_loaders import compute_aggregate_histogram, load_client_data
from heavy_hitters_protocol.experiment import ExperimentRunner


CLIENTS_PER_ROUND = 16
N_ROUNDS = 6
CLIPPING_THRESHOLD = 1200
CLIPPING_METHOD = "uniform_with_replacement"
EPSILON = 4
FAILURE_RATE = 0.0
N_TRIALS = 30
SEED = 10
TEST_RATIO = 0.10
PICKED_IDS_TOTAL_ROUNDS = 1000


def split_histogram_by_ratio(
    histogram: Dict[str, int], test_ratio: float
) -> tuple[Dict[str, int], Dict[str, int]]:
    """Split a histogram into train/test while preserving totals exactly."""
    from math import floor

    total = sum(histogram.values())
    target_test_total = int(round(total * test_ratio))

    raw_test = {label: histogram[label] * test_ratio for label in histogram}
    base_test = {label: int(floor(raw_test[label])) for label in histogram}
    base_total = sum(base_test.values())

    remainders = sorted(
        histogram.keys(),
        key=lambda label: (raw_test[label] - base_test[label]),
        reverse=True,
    )
    missing = target_test_total - base_total
    for label in remainders[:missing]:
        base_test[label] += 1

    test_hist = {label: min(base_test[label], histogram[label]) for label in histogram}
    train_hist = {label: histogram[label] - test_hist[label] for label in histogram}
    return train_hist, test_hist


def write_clients_csv(
    csv_path: Path,
    labels: List[str],
    client_histograms: List[Dict[str, int]],
) -> List[int]:
    """Write per-patient histograms to CSV."""
    fieldnames = ["client_id", "client_size"] + [f"count_{l}" for l in labels] + [
        f"prop_{l}" for l in labels
    ]

    client_sizes = [sum(hist.values()) for hist in client_histograms]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for client_id, hist in enumerate(client_histograms):
            size = client_sizes[client_id]
            row = {"client_id": client_id, "client_size": size}
            for label in labels:
                row[f"count_{label}"] = hist[label]
                row[f"prop_{label}"] = hist[label] / size if size > 0 else 0.0
            writer.writerow(row)

    return client_sizes


def save_client_size_histogram(
    output_path: Path, client_sizes: List[int], title_suffix: str
) -> None:
    """Save histogram of samples-per-client."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping client-size histogram plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = min(20, max(5, int(np.sqrt(len(client_sizes)))))
    ax.hist(client_sizes, bins=bins, alpha=0.85, edgecolor="black")
    ax.set_title(f"Client Sample Count Histogram ({title_suffix})")
    ax.set_xlabel("Samples per client")
    ax.set_ylabel("Number of clients")
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_client_label_stacked_bars(
    output_path: Path,
    labels: List[str],
    client_histograms: List[Dict[str, int]],
    normalize: bool,
    title_suffix: str,
) -> None:
    """Save per-client stacked bars with one color per label."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping client-label stacked plot")
        return

    n_clients = len(client_histograms)
    x = np.arange(n_clients)
    values = np.array(
        [[client_histograms[i][label] for i in range(n_clients)] for label in labels],
        dtype=float,
    )

    if normalize:
        totals = values.sum(axis=0)
        totals[totals == 0] = 1.0
        values = values / totals

    colors = {
        "0": "#457b9d",
        "1": "#e63946",
        "2": "#f4a261",
        "3": "#2a9d8f",
        "4": "#8d99ae",
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    bottom = np.zeros(n_clients, dtype=float)
    for i, label in enumerate(labels):
        ax.bar(
            x,
            values[i],
            bottom=bottom,
            width=0.9,
            label=label,
            color=colors.get(label, f"C{i}"),
            linewidth=0,
        )
        bottom += values[i]

    ax.set_title(
        f"Per-Client Label Composition ({title_suffix}, "
        f"{'proportions' if normalize else 'counts'})"
    )
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Proportion" if normalize else "Samples")
    if normalize:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.5, n_clients - 0.5)
    ax.set_xticks(np.arange(0, n_clients, 2))
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 1.13))

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def unique_coverage_by_round(client_sets: List[List[int]]) -> List[int]:
    """Unique-client coverage progression."""
    seen = set()
    coverage = []
    for ids in client_sets:
        seen.update(ids)
        coverage.append(len(seen))
    return coverage


def save_picked_ids_and_estimate(
    output_path: Path,
    picked_ids_by_round: List[List[int]],
    estimated_histogram: Dict[str, int] | None,
    noise_values: Dict[str, int] | None,
    estimate_trigger_round: int,
    picked_ids_used_for_estimate: List[List[int]],
    first_rounds_match_estimate: bool,
) -> None:
    """Save the exact picked IDs used before one estimate (trial 1)."""
    payload = {
        "seed": SEED,
        "clients_per_round": CLIENTS_PER_ROUND,
        "n_rounds": N_ROUNDS,
        "sampling_with_replacement_across_rounds": True,
        "replacement_within_round": False,
        "picked_ids_total_rounds": len(picked_ids_by_round),
        "estimate_trigger_round": estimate_trigger_round,
        "first_rounds_match_estimate": first_rounds_match_estimate,
        "picked_ids_by_round": [
            {"round": i + 1, "picked_ids": ids} for i, ids in enumerate(picked_ids_by_round)
        ],
        "picked_ids_used_for_this_estimate": [
            {"round": i + 1, "picked_ids": ids}
            for i, ids in enumerate(picked_ids_used_for_estimate)
        ],
        "estimate_histogram_from_these_ids": estimated_histogram,
        "noise_values_for_this_estimate": noise_values,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def generate_and_save_picked_ids_trace(
    output_dir: Path,
    n_clients: int,
    clients_per_round: int,
    n_rounds: int,
    seed: int,
) -> List[List[int]]:
    """Generate Arrhythmia client picks for many rounds and save trace + summary."""
    replace_within_round = clients_per_round > n_clients
    rng = np.random.default_rng(seed)

    picks_payload = []
    picks: List[List[int]] = []
    seen = set()
    unique_coverage = []

    for round_idx in range(n_rounds):
        picked_ids = rng.choice(
            n_clients, size=clients_per_round, replace=replace_within_round
        ).tolist()
        picks_payload.append({"round": round_idx + 1, "picked_ids": picked_ids})
        picks.append(picked_ids)
        seen.update(picked_ids)
        unique_coverage.append(len(seen))

    picks_path = output_dir / f"picked_ids_{n_rounds}_rounds_seed{seed}.json"
    with open(picks_path, "w") as f:
        json.dump(picks_payload, f, indent=2)

    summary = {
        "seed": seed,
        "n_clients": n_clients,
        "clients_per_round": clients_per_round,
        "n_rounds": n_rounds,
        "sampling_with_replacement_across_rounds": True,
        "replacement_within_round": replace_within_round,
        "final_unique_clients_seen": int(unique_coverage[-1]) if unique_coverage else 0,
        "first_round_full_coverage": next(
            (i + 1 for i, x in enumerate(unique_coverage) if x == n_clients), None
        ),
        "unique_coverage_by_round": unique_coverage,
    }
    summary_path = output_dir / f"picked_ids_{n_rounds}_rounds_seed{seed}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return picks


def main() -> None:
    base_path = Path(__file__).parent
    output_dir = base_path / "results" / "arrhythmia_real_22rounds"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = DatasetConfig.arrhythmia(base_path)
    labels = LABEL_ORDER["arrhythmia"]
    full_client_histograms = load_client_data(base_config)

    train_client_histograms = []
    test_client_histograms = []
    for hist in full_client_histograms:
        train_hist, test_hist = split_histogram_by_ratio(hist, TEST_RATIO)
        train_client_histograms.append(train_hist)
        test_client_histograms.append(test_hist)

    n_clients = len(train_client_histograms)
    gamma = CLIENTS_PER_ROUND / n_clients
    true_hist = compute_aggregate_histogram(train_client_histograms, labels)
    real_hist = compute_aggregate_histogram(full_client_histograms, labels)
    test_hist = compute_aggregate_histogram(test_client_histograms, labels)

    csv_path = output_dir / "arrhythmia_clients_real.csv"
    client_sizes = write_clients_csv(csv_path, labels, train_client_histograms)

    size_hist_path = output_dir / "arrhythmia_clients_samples_hist.png"
    stacked_counts_path = output_dir / "arrhythmia_clients_stacked_counts.png"
    stacked_props_path = output_dir / "arrhythmia_clients_stacked_proportions.png"
    save_client_size_histogram(size_hist_path, client_sizes, "Arrhythmia real")
    save_client_label_stacked_bars(
        stacked_counts_path, labels, train_client_histograms, normalize=False, title_suffix="Arrhythmia train(90%)"
    )
    save_client_label_stacked_bars(
        stacked_props_path, labels, train_client_histograms, normalize=True, title_suffix="Arrhythmia train(90%)"
    )

    run_config = DatasetConfig(
        name="arrhythmia_real_patients",
        n_clients=n_clients,
        labels=labels,
        ground_truth=true_hist,
        data_path=base_config.data_path,
        gamma=gamma,
        rounds=N_ROUNDS,
        sampling_with_replacement=True,
        count_client_once=True,
    )

    picked_ids_1000 = generate_and_save_picked_ids_trace(
        output_dir=output_dir,
        n_clients=n_clients,
        clients_per_round=CLIENTS_PER_ROUND,
        n_rounds=PICKED_IDS_TOTAL_ROUNDS,
        seed=SEED,
    )
    estimate_round_picks = picked_ids_1000[:N_ROUNDS]

    runner = ExperimentRunner(base_path, output_dir=output_dir, verbose=False)
    result = runner.run_configuration(
        config=run_config,
        client_histograms=train_client_histograms,
        clipping_threshold=CLIPPING_THRESHOLD,
        epsilon=EPSILON,
        failure_rate=FAILURE_RATE,
        n_trials=N_TRIALS,
        clipping_method=CLIPPING_METHOD,
        base_seed=SEED,
        fixed_client_sets_per_round=estimate_round_picks,
    )
    runner.save_results([result], "arrhythmia_real_22rounds_results.json")

    first_trial_sets = result.sample_client_sets[0] if result.sample_client_sets else []
    coverage = unique_coverage_by_round(first_trial_sets)
    within_round_unique = all(
        len(round_ids) == len(set(round_ids)) for round_ids in first_trial_sets
    )

    first_rounds_match_estimate = picked_ids_1000[:N_ROUNDS] == first_trial_sets

    picked_ids_path = output_dir / f"picked_ids_used_for_estimate_seed{SEED}.json"
    save_picked_ids_and_estimate(
        output_path=picked_ids_path,
        picked_ids_by_round=picked_ids_1000,
        estimated_histogram=(
            result.sample_estimated_histograms[0]
            if result.sample_estimated_histograms
            else None
        ),
        noise_values=(
            result.sample_noise_values[0]
            if result.sample_noise_values
            else None
        ),
        estimate_trigger_round=N_ROUNDS,
        picked_ids_used_for_estimate=first_trial_sets,
        first_rounds_match_estimate=first_rounds_match_estimate,
    )
    summary = {
        "dataset": "arrhythmia_real_patients",
        "seed": SEED,
        "test_ratio": TEST_RATIO,
        "n_clients": n_clients,
        "labels": labels,
        "real_histogram": real_hist,
        "test_histogram": test_hist,
        "true_histogram": true_hist,
        "clients_per_round": CLIENTS_PER_ROUND,
        "gamma": gamma,
        "n_rounds": N_ROUNDS,
        "sampling_with_replacement": True,
        "count_client_once": True,
        "clipping_threshold": CLIPPING_THRESHOLD,
        "clipping_method": CLIPPING_METHOD,
        "epsilon": EPSILON,
        "failure_rate": FAILURE_RATE,
        "n_trials": N_TRIALS,
        "client_size_stats": {
            "min": int(min(client_sizes)),
            "max": int(max(client_sizes)),
            "mean": float(np.mean(client_sizes)),
            "std": float(np.std(client_sizes)),
        },
        "coverage_first_trial": coverage,
        "within_round_unique_first_trial": within_round_unique,
        "picked_ids_file_trial_1": str(picked_ids_path),
        "picked_ids_trial_1": first_trial_sets,
        "metrics": result.aggregated_metrics,
        "sample_estimate_trial_1": (
            result.sample_estimated_histograms[0]
            if result.sample_estimated_histograms
            else None
        ),
        "outputs": {
            "clients_csv": str(csv_path),
            "size_histogram_png": str(size_hist_path),
            "stacked_counts_png": str(stacked_counts_path),
            "stacked_proportions_png": str(stacked_props_path),
            "results_json": str(output_dir / "arrhythmia_real_22rounds_results.json"),
        },
    }

    summary_path = output_dir / "arrhythmia_real_22rounds_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    kl = result.aggregated_metrics["kl_divergence"]
    tv = result.aggregated_metrics["tv_distance"]
    l1 = result.aggregated_metrics["l1_error"]

    print("=" * 80)
    print(f"ARRHYTHMIA REAL PATIENT SIMULATION (TRAIN 90%, {N_ROUNDS} ROUNDS)")
    print("=" * 80)
    print(f"Patients(clients): {n_clients}")
    print(f"Real aggregate: {real_hist}")
    print(f"Train aggregate (true used): {true_hist}")
    print(f"Test aggregate: {test_hist}")
    print(
        f"Client size stats: min={min(client_sizes)}, max={max(client_sizes)}, mean={np.mean(client_sizes):.2f}"
    )
    print(
        f"Sampling: {CLIENTS_PER_ROUND}/round, rounds={N_ROUNDS}, with_replacement=True, count_client_once=True"
    )
    print(
        f"KL mean={kl['mean']:.4f} [{kl['ci_lower']:.4f}, {kl['ci_upper']:.4f}] | "
        f"TV mean={tv['mean']:.4f} | L1 mean={l1['mean']:.2f}"
    )
    print(f"Within-round IDs are all distinct (trial 1): {within_round_unique}")
    print(
        f"Picked-ID trace rounds saved: {len(picked_ids_1000)} "
        f"(first {N_ROUNDS} match estimate trial-1 picks: {first_rounds_match_estimate})"
    )
    print(f"Picked IDs used for trial-1 estimate saved to: {picked_ids_path}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
