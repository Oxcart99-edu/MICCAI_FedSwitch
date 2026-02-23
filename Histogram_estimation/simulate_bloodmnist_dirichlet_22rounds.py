#!/usr/bin/env python3
"""
BloodMNIST synthetic client simulation with 3 label-Dirichlet scenarios.

Setup mirrors HAM10000 pipeline:
- 100 clients
- Client size: min 50 samples each + remaining from Dirichlet(alpha=0.5)
- Label distributions per client from Dirichlet(alpha in {0.1, 0.5, 1.0})
- Estimation with fixed picked IDs source (1000 rounds), using first 22 rounds
- 10 clients/round, count each client once, clipping/noise as configured
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from heavy_hitters_protocol.config import DatasetConfig
from heavy_hitters_protocol.experiment import ExperimentRunner


N_CLIENTS = 100
MIN_SAMPLES_PER_CLIENT = 70
CLIENT_SIZE_ALPHA = 0.5
LABEL_ALPHAS = [0.1, 0.5, 1.0]
CLIENTS_PER_ROUND = 10
N_ROUNDS = 22
CLIPPING_THRESHOLD = 70
CLIPPING_METHOD = "uniform_with_replacement"
EPSILON = 4
FAILURE_RATE = 0.0
N_TRIALS = 30
BASE_SEED = 42
PICKED_IDS_TOTAL_ROUNDS = 1000

LABELS = [str(i) for i in range(8)]
LABEL_NAMES = {
    "0": "basophil",
    "1": "eosinophil",
    "2": "erythroblast",
    "3": "immature_granulocytes",
    "4": "lymphocyte",
    "5": "monocyte",
    "6": "neutrophil",
    "7": "platelet",
}

# User-provided counts
TRAIN_HIST = {
    "0": 852,
    "1": 2181,
    "2": 1085,
    "3": 2026,
    "4": 849,
    "5": 993,
    "6": 2330,
    "7": 1643,
}
VAL_HIST = {
    "0": 122,
    "1": 312,
    "2": 155,
    "3": 290,
    "4": 122,
    "5": 143,
    "6": 333,
    "7": 235,
}
TEST_HIST = {
    "0": 244,
    "1": 624,
    "2": 311,
    "3": 579,
    "4": 243,
    "5": 284,
    "6": 666,
    "7": 470,
}
TRAIN_VAL_HIST = {k: TRAIN_HIST[k] + VAL_HIST[k] for k in LABELS}
TOTAL_HIST = {k: TRAIN_VAL_HIST[k] + TEST_HIST[k] for k in LABELS}


def sample_client_sizes(
    total_samples: int,
    n_clients: int,
    min_samples: int,
    alpha: float,
    rng: np.random.Generator,
) -> List[int]:
    baseline = n_clients * min_samples
    if baseline > total_samples:
        raise ValueError(
            f"Minimum requirement invalid: {baseline} > total samples {total_samples}"
        )
    remaining = total_samples - baseline
    if remaining == 0:
        return [min_samples] * n_clients
    proportions = rng.dirichlet(np.full(n_clients, alpha))
    extra = rng.multinomial(remaining, proportions)
    return [min_samples + int(x) for x in extra]


def sample_client_histograms(
    labels: List[str],
    client_sizes: List[int],
    label_alpha: float,
    target_histogram: Dict[str, int],
    rng: np.random.Generator,
) -> List[Dict[str, int]]:
    """Dirichlet per-client label heterogeneity with exact global totals."""
    n_clients = len(client_sizes)
    n_labels = len(labels)
    probs = rng.dirichlet(np.full(n_labels, label_alpha), size=n_clients)
    counts = np.vstack(
        [rng.multinomial(size, probs[i]) for i, size in enumerate(client_sizes)]
    ).astype(int)

    target = np.array([target_histogram[label] for label in labels], dtype=int)
    diff = target - counts.sum(axis=0)
    eps = 1e-12

    max_passes = 2000
    for _ in range(max_passes):
        if np.all(diff == 0):
            break
        moved_any = False
        deficit_labels = np.where(diff > 0)[0]
        surplus_labels = np.where(diff < 0)[0]
        deficit_labels = deficit_labels[np.argsort(-diff[deficit_labels])]

        for d in deficit_labels:
            need = int(diff[d])
            if need <= 0:
                continue
            ordered_surplus = surplus_labels[np.argsort(diff[surplus_labels])]
            for s in ordered_surplus:
                surplus = int(-diff[s])
                if surplus <= 0 or need <= 0:
                    continue
                candidate_rows = np.where(counts[:, s] > 0)[0]
                if candidate_rows.size == 0:
                    continue
                scores = probs[candidate_rows, d] / (probs[candidate_rows, s] + eps)
                candidate_rows = candidate_rows[np.argsort(-scores)]

                for r in candidate_rows:
                    if need <= 0 or diff[s] >= 0:
                        break
                    transferable = min(int(counts[r, s]), need, int(-diff[s]))
                    if transferable <= 0:
                        continue
                    counts[r, s] -= transferable
                    counts[r, d] += transferable
                    diff[s] += transferable
                    diff[d] -= transferable
                    need -= transferable
                    moved_any = True

        if not moved_any:
            raise RuntimeError("Could not reconcile client histograms to exact totals.")

    if not np.all(diff == 0):
        raise RuntimeError("Failed to match exact global label totals.")

    return [
        {label: int(counts[i, j]) for j, label in enumerate(labels)}
        for i in range(n_clients)
    ]


def aggregate(histograms: List[Dict[str, int]], labels: List[str]) -> Dict[str, int]:
    out = {label: 0 for label in labels}
    for hist in histograms:
        for label in labels:
            out[label] += hist[label]
    return out


def unique_coverage_by_round(client_sets: List[List[int]]) -> List[int]:
    seen = set()
    coverage = []
    for ids in client_sets:
        seen.update(ids)
        coverage.append(len(seen))
    return coverage


def generate_picked_ids(
    n_clients: int,
    clients_per_round: int,
    n_rounds: int,
    seed: int,
) -> List[List[int]]:
    """Generate round-wise picked client IDs from scratch."""
    rng = np.random.default_rng(seed)
    replace_within_round = clients_per_round > n_clients
    picks: List[List[int]] = []
    for _ in range(n_rounds):
        ids = rng.choice(
            n_clients, size=clients_per_round, replace=replace_within_round
        ).tolist()
        picks.append(ids)
    return picks


def save_generated_picked_ids(
    output_dir: Path,
    seed: int,
    picks: List[List[int]],
    n_clients: int,
    clients_per_round: int,
) -> Path:
    """Save generated picked IDs (and summary) in HAM-compatible format."""
    payload = [
        {"round": i + 1, "picked_ids": ids}
        for i, ids in enumerate(picks)
    ]
    picks_path = output_dir / f"picked_ids_1000_rounds_seed{seed}.json"
    with open(picks_path, "w") as f:
        json.dump(payload, f, indent=2)

    seen = set()
    unique_coverage = []
    for ids in picks:
        seen.update(ids)
        unique_coverage.append(len(seen))

    summary = {
        "seed": seed,
        "n_clients": n_clients,
        "clients_per_round": clients_per_round,
        "n_rounds": len(picks),
        "sampling_with_replacement_across_rounds": True,
        "replacement_within_round": clients_per_round > n_clients,
        "final_unique_clients_seen": int(unique_coverage[-1]) if unique_coverage else 0,
        "first_round_full_coverage": next(
            (i + 1 for i, x in enumerate(unique_coverage) if x == n_clients), None
        ),
        "unique_coverage_by_round": unique_coverage,
    }
    summary_path = output_dir / f"picked_ids_1000_rounds_seed{seed}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return picks_path


def write_clients_csv(
    csv_path: Path,
    labels: List[str],
    client_sizes: List[int],
    client_histograms: List[Dict[str, int]],
) -> None:
    fieldnames = ["client_id", "client_size"] + [f"count_{l}" for l in labels] + [
        f"prop_{l}" for l in labels
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for client_id, (size, hist) in enumerate(zip(client_sizes, client_histograms)):
            row = {"client_id": client_id, "client_size": size}
            for label in labels:
                row[f"count_{label}"] = hist[label]
                row[f"prop_{label}"] = hist[label] / size if size > 0 else 0.0
            writer.writerow(row)


def save_client_samples_histogram(
    output_path: Path, client_sizes: List[int], label_alpha: float
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping client-size histogram plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = min(20, max(5, int(np.sqrt(len(client_sizes)))))
    ax.hist(client_sizes, bins=bins, alpha=0.85, edgecolor="black")
    ax.set_title(f"BloodMNIST Client Sample Count Histogram (label alpha={label_alpha})")
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
    label_alpha: float,
    normalize: bool = False,
) -> None:
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

    fig, ax = plt.subplots(figsize=(20, 7))
    bottom = np.zeros(n_clients, dtype=float)
    for i, label in enumerate(labels):
        ax.bar(
            x,
            values[i],
            bottom=bottom,
            width=0.9,
            label=f"{label}:{LABEL_NAMES[label]}",
            color=f"C{i}",
            linewidth=0,
        )
        bottom += values[i]

    ax.set_title(
        f"BloodMNIST Per-Client Label Composition (label alpha={label_alpha}, "
        f"{'proportions' if normalize else 'counts'})"
    )
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Proportion" if normalize else "Samples")
    if normalize:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.5, n_clients - 0.5)
    ax.set_xticks(np.arange(0, n_clients, 5))
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_picked_ids_and_estimate(
    output_path: Path,
    label_alpha: float,
    seed: int,
    picked_ids_by_round: List[List[int]],
    picked_ids_used_for_estimate: List[List[int]],
    estimated_histogram: Dict[str, int] | None,
    noise_values: Dict[str, int] | None,
    first_rounds_match_estimate: bool,
    picked_ids_source_file: str,
) -> None:
    payload = {
        "label_alpha": label_alpha,
        "seed": seed,
        "clients_per_round": CLIENTS_PER_ROUND,
        "n_rounds": N_ROUNDS,
        "estimate_trigger_round": N_ROUNDS,
        "sampling_with_replacement_across_rounds": True,
        "replacement_within_round": False,
        "picked_ids_source_file": picked_ids_source_file,
        "picked_ids_total_rounds": len(picked_ids_by_round),
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


def main() -> None:
    base_path = Path(__file__).parent
    output_dir = base_path / "results" / "bloodmnist_dirichlet_22rounds"
    output_dir.mkdir(parents=True, exist_ok=True)

    gamma = CLIENTS_PER_ROUND / N_CLIENTS
    picked_ids_1000 = generate_picked_ids(
        n_clients=N_CLIENTS,
        clients_per_round=CLIENTS_PER_ROUND,
        n_rounds=PICKED_IDS_TOTAL_ROUNDS,
        seed=BASE_SEED,
    )
    estimate_round_picks = picked_ids_1000[:N_ROUNDS]
    picked_ids_source_path = save_generated_picked_ids(
        output_dir=output_dir,
        seed=BASE_SEED,
        picks=picked_ids_1000,
        n_clients=N_CLIENTS,
        clients_per_round=CLIENTS_PER_ROUND,
    )
    if len(estimate_round_picks) < N_ROUNDS:
        raise ValueError(
            f"Generated picked IDs only {len(estimate_round_picks)} rounds, need {N_ROUNDS}."
        )
    for round_idx, ids in enumerate(estimate_round_picks, start=1):
        if len(ids) != CLIENTS_PER_ROUND:
            raise ValueError(
                f"Generated round {round_idx} has {len(ids)} ids, expected {CLIENTS_PER_ROUND}."
            )

    print("=" * 80)
    print("BLOODMNIST DIRICHLET CLIENT SIMULATION (3 LABEL SCENARIOS, TRAIN+VAL)")
    print("=" * 80)
    print(f"Train histogram: {TRAIN_HIST}")
    print(f"Val histogram: {VAL_HIST}")
    print(f"Train+Val (used target): {TRAIN_VAL_HIST}")
    print(f"Test histogram (holdout): {TEST_HIST}")
    print(f"Total histogram: {TOTAL_HIST}")
    print(
        f"Client size model: min={MIN_SAMPLES_PER_CLIENT} + Dirichlet(alpha={CLIENT_SIZE_ALPHA})"
    )
    print(
        f"Estimation setup: {CLIENTS_PER_ROUND}/round, rounds={N_ROUNDS}, with_replacement=True, "
        f"count_client_once=True, clipping={CLIPPING_METHOD}, C={CLIPPING_THRESHOLD}, epsilon={EPSILON}"
    )
    print(f"Picked-ID source file (generated): {picked_ids_source_path}")
    print(
        f"Picked rounds loaded: {len(picked_ids_1000)} (estimation uses first {N_ROUNDS})"
    )

    total_samples = int(sum(TRAIN_VAL_HIST.values()))
    runner = ExperimentRunner(base_path, output_dir=output_dir, verbose=False)
    summary: List[dict] = []

    for label_alpha in LABEL_ALPHAS:
        seed = BASE_SEED
        rng = np.random.default_rng(seed)

        client_sizes = sample_client_sizes(
            total_samples=total_samples,
            n_clients=N_CLIENTS,
            min_samples=MIN_SAMPLES_PER_CLIENT,
            alpha=CLIENT_SIZE_ALPHA,
            rng=rng,
        )
        client_histograms = sample_client_histograms(
            labels=LABELS,
            client_sizes=client_sizes,
            label_alpha=label_alpha,
            target_histogram=TRAIN_VAL_HIST,
            rng=rng,
        )
        true_hist = aggregate(client_histograms, LABELS)

        csv_path = output_dir / f"clients_label_alpha_{label_alpha}.csv"
        write_clients_csv(csv_path, LABELS, client_sizes, client_histograms)
        hist_path = output_dir / f"clients_label_alpha_{label_alpha}_samples_hist.png"
        save_client_samples_histogram(hist_path, client_sizes, label_alpha)
        stacked_counts_path = (
            output_dir / f"clients_label_alpha_{label_alpha}_stacked_counts.png"
        )
        stacked_props_path = (
            output_dir / f"clients_label_alpha_{label_alpha}_stacked_proportions.png"
        )
        save_client_label_stacked_bars(
            stacked_counts_path, LABELS, client_histograms, label_alpha, normalize=False
        )
        save_client_label_stacked_bars(
            stacked_props_path, LABELS, client_histograms, label_alpha, normalize=True
        )

        dataset_config = DatasetConfig(
            name=f"bloodmnist_dirichlet_alpha_{label_alpha}",
            n_clients=N_CLIENTS,
            labels=LABELS,
            ground_truth=true_hist,
            data_path=Path("synthetic_bloodmnist"),
            gamma=gamma,
            rounds=N_ROUNDS,
            sampling_with_replacement=True,
            count_client_once=True,
        )

        result = runner.run_configuration(
            config=dataset_config,
            client_histograms=client_histograms,
            clipping_threshold=CLIPPING_THRESHOLD,
            epsilon=EPSILON,
            failure_rate=FAILURE_RATE,
            n_trials=N_TRIALS,
            clipping_method=CLIPPING_METHOD,
            base_seed=seed,
            fixed_client_sets_per_round=estimate_round_picks,
        )
        runner.save_results([result], f"bloodmnist_dirichlet_alpha_{label_alpha}_results.json")

        first_trial_sets = result.sample_client_sets[0] if result.sample_client_sets else []
        coverage = unique_coverage_by_round(first_trial_sets)
        first_rounds_match_estimate = estimate_round_picks == first_trial_sets

        picked_ids_path = (
            output_dir / f"picked_ids_used_for_estimate_alpha_{label_alpha}_seed_{seed}.json"
        )
        save_picked_ids_and_estimate(
            output_path=picked_ids_path,
            label_alpha=label_alpha,
            seed=seed,
            picked_ids_by_round=picked_ids_1000,
            picked_ids_used_for_estimate=first_trial_sets,
            estimated_histogram=(
                result.sample_estimated_histograms[0]
                if result.sample_estimated_histograms
                else None
            ),
            noise_values=(
                result.sample_noise_values[0] if result.sample_noise_values else None
            ),
            first_rounds_match_estimate=first_rounds_match_estimate,
            picked_ids_source_file=str(picked_ids_source_path),
        )

        summary.append(
            {
                "label_alpha": label_alpha,
                "seed": seed,
                "labels": LABELS,
                "label_names": LABEL_NAMES,
                "clients_csv": str(csv_path),
                "stacked_counts_plot": str(stacked_counts_path),
                "stacked_proportions_plot": str(stacked_props_path),
                "train_histogram": TRAIN_HIST,
                "val_histogram": VAL_HIST,
                "train_val_histogram": TRAIN_VAL_HIST,
                "test_histogram": TEST_HIST,
                "total_histogram": TOTAL_HIST,
                "client_size_stats": {
                    "min": int(min(client_sizes)),
                    "max": int(max(client_sizes)),
                    "mean": float(np.mean(client_sizes)),
                },
                "true_histogram": true_hist,
                "coverage_first_trial": coverage,
                "picked_ids_file_trial_1": str(picked_ids_path),
                "picked_ids_total_rounds": len(picked_ids_1000),
                "picked_ids_first_rounds_match_estimate": first_rounds_match_estimate,
                "picked_ids_source_file": str(picked_ids_source_path),
                "metrics": result.aggregated_metrics,
                "sample_estimate_trial_1": (
                    result.sample_estimated_histograms[0]
                    if result.sample_estimated_histograms
                    else None
                ),
            }
        )

        kl = result.aggregated_metrics["kl_divergence"]
        tv = result.aggregated_metrics["tv_distance"]
        l1 = result.aggregated_metrics["l1_error"]
        print("-" * 80)
        print(f"Scenario label alpha={label_alpha}")
        print(f"True aggregate: {true_hist}")
        print(
            f"KL mean={kl['mean']:.4f} [{kl['ci_lower']:.4f}, {kl['ci_upper']:.4f}] | "
            f"TV mean={tv['mean']:.4f} | L1 mean={l1['mean']:.2f}"
        )
        print(f"First-trial unique coverage by round: {coverage}")
        print(
            f"Picked-ID trace rounds saved: {len(picked_ids_1000)} "
            f"(first {N_ROUNDS} match estimate trial-1 picks: {first_rounds_match_estimate})"
        )
        print(f"Picked IDs used for trial-1 estimate saved to: {picked_ids_path}")

    summary_path = output_dir / "bloodmnist_dirichlet_22rounds_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Summary saved to: {summary_path}")
    print(f"All outputs saved in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
