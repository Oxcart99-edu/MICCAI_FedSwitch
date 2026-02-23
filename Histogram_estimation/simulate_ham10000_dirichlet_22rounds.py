#!/usr/bin/env python3
"""
HAM10000 synthetic client simulation with 3 label-Dirichlet scenarios.

Requested setup:
- 100 clients
- Client size: min 50 samples each + remaining samples from Dirichlet(alpha=0.5)
- Label distributions per client from Dirichlet(alpha in {0.1, 0.5, 1.0})
- Save per-client histograms to CSV
- Run protocol estimation as before:
  10 clients/round, 22 rounds, with replacement, count each client once,
  random clipping with replacement C=50, privacy noise enabled
"""

from __future__ import annotations

import csv
import json
from math import floor
from pathlib import Path
from typing import Dict, List

import numpy as np

from heavy_hitters_protocol.config import DatasetConfig, GROUND_TRUTH, LABEL_ORDER
from heavy_hitters_protocol.experiment import ExperimentRunner


N_CLIENTS = 100
MIN_SAMPLES_PER_CLIENT = 50
CLIENT_SIZE_ALPHA = 0.5
LABEL_ALPHAS = [0.1, 0.5, 1.0]
CLIENTS_PER_ROUND = 10
N_ROUNDS = 22
CLIPPING_THRESHOLD = 50
CLIPPING_METHOD = "uniform_with_replacement"
EPSILON = 4.0
FAILURE_RATE = 0.0
N_TRIALS = 30
BASE_SEED = 42
TEST_RATIO = 0.10
PICKED_IDS_TOTAL_ROUNDS = 1000


def sample_client_sizes(
    total_samples: int,
    n_clients: int,
    min_samples: int,
    alpha: float,
    rng: np.random.Generator,
) -> List[int]:
    """Sample client sizes with a hard minimum per client."""
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


def split_histogram_by_ratio(
    histogram: Dict[str, int], test_ratio: float
) -> tuple[Dict[str, int], Dict[str, int]]:
    """Split a histogram into train/test while preserving total exactly."""
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


def sample_client_histograms(
    labels: List[str],
    client_sizes: List[int],
    label_alpha: float,
    target_histogram: Dict[str, int],
    rng: np.random.Generator,
) -> List[Dict[str, int]]:
    """
    Sample client label counts with Dirichlet heterogeneity while matching
    the global real label histogram exactly.
    """
    n_clients = len(client_sizes)
    n_labels = len(labels)

    # Per-client label preferences (scenario-controlled heterogeneity)
    probs = rng.dirichlet(np.full(n_labels, label_alpha), size=n_clients)

    # Initial row-wise multinomial draws (match row sums only)
    counts = np.vstack(
        [rng.multinomial(size, probs[i]) for i, size in enumerate(client_sizes)]
    ).astype(int)

    # Enforce exact global column totals via within-row label transfers.
    target = np.array([target_histogram[label] for label in labels], dtype=int)
    diff = target - counts.sum(axis=0)
    eps = 1e-12

    # Move mass from surplus labels to deficit labels, preserving client sizes.
    # This keeps heterogeneity shape while forcing exact real global totals.
    max_passes = 2000
    for _ in range(max_passes):
        if np.all(diff == 0):
            break

        moved_any = False
        deficit_labels = np.where(diff > 0)[0]
        surplus_labels = np.where(diff < 0)[0]

        # Fill larger deficits first.
        deficit_labels = deficit_labels[np.argsort(-diff[deficit_labels])]

        for d in deficit_labels:
            need = int(diff[d])
            if need <= 0:
                continue

            # Drain larger surpluses first.
            ordered_surplus = surplus_labels[np.argsort(diff[surplus_labels])]
            for s in ordered_surplus:
                surplus = int(-diff[s])
                if surplus <= 0 or need <= 0:
                    continue

                candidate_rows = np.where(counts[:, s] > 0)[0]
                if candidate_rows.size == 0:
                    continue

                # Prefer rows where switching s->d is more consistent with preferences.
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
            raise RuntimeError(
                "Could not reconcile client histograms with exact global label totals."
            )

    if not np.all(diff == 0):
        raise RuntimeError("Failed to match exact global label totals within iteration budget.")

    histograms: List[Dict[str, int]] = []
    for i in range(n_clients):
        histograms.append({label: int(counts[i, j]) for j, label in enumerate(labels)})
    return histograms


def aggregate(histograms: List[Dict[str, int]], labels: List[str]) -> Dict[str, int]:
    """Aggregate all client histograms."""
    out = {label: 0 for label in labels}
    for hist in histograms:
        for label in labels:
            out[label] += hist[label]
    return out


def unique_coverage_by_round(client_sets: List[List[int]]) -> List[int]:
    """Unique-client coverage progression."""
    seen = set()
    coverage = []
    for ids in client_sets:
        seen.update(ids)
        coverage.append(len(seen))
    return coverage


def generate_and_save_picked_ids_source(
    output_dir: Path,
    n_clients: int,
    clients_per_round: int,
    n_rounds_total: int,
    n_rounds_for_estimate: int,
    seed: int,
) -> tuple[List[List[int]], List[List[int]], Path]:
    """Generate fixed picked IDs trace, save it, and return estimate prefix."""
    replace_within_round = clients_per_round > n_clients
    rng = np.random.default_rng(seed)

    raw_payload = []
    all_rounds: List[List[int]] = []
    seen = set()
    unique_coverage = []

    for round_idx in range(n_rounds_total):
        ids = rng.choice(
            n_clients, size=clients_per_round, replace=replace_within_round
        ).tolist()
        raw_payload.append({"round": round_idx + 1, "picked_ids": ids})
        all_rounds.append(ids)
        seen.update(ids)
        unique_coverage.append(len(seen))

    if len(all_rounds) < n_rounds_for_estimate:
        raise ValueError(
            f"Generated {len(all_rounds)} rounds, need at least {n_rounds_for_estimate}."
        )

    estimate_rounds = all_rounds[:n_rounds_for_estimate]

    picks_path = output_dir / f"picked_ids_{n_rounds_total}_rounds_seed{seed}.json"
    with open(picks_path, "w") as f:
        json.dump(raw_payload, f, indent=2)

    summary = {
        "seed": seed,
        "n_clients": n_clients,
        "clients_per_round": clients_per_round,
        "n_rounds": n_rounds_total,
        "sampling_with_replacement_across_rounds": True,
        "replacement_within_round": replace_within_round,
        "final_unique_clients_seen": int(unique_coverage[-1]) if unique_coverage else 0,
        "first_round_full_coverage": next(
            (i + 1 for i, x in enumerate(unique_coverage) if x == n_clients), None
        ),
        "unique_coverage_by_round": unique_coverage,
    }
    summary_path = output_dir / f"picked_ids_{n_rounds_total}_rounds_seed{seed}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return all_rounds, estimate_rounds, picks_path


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
    """Save 1000-round picked IDs and link them to trial-1 estimate."""
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


def write_clients_csv(
    csv_path: Path,
    labels: List[str],
    client_sizes: List[int],
    client_histograms: List[Dict[str, int]],
) -> None:
    """Write per-client distributions to CSV."""
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
    """Save histogram of samples-per-client."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping client-size histogram plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = min(20, max(5, int(np.sqrt(len(client_sizes)))))
    ax.hist(client_sizes, bins=bins, alpha=0.85, edgecolor="black")
    ax.set_title(f"Client Sample Count Histogram (label alpha={label_alpha})")
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
    """Save a per-client stacked bar chart with one color per label."""
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
        "akiec": "#e63946",
        "bcc": "#ff7f11",
        "bkl": "#f4d35e",
        "df": "#2a9d8f",
        "mel": "#1d3557",
        "nv": "#457b9d",
        "vasc": "#8d99ae",
    }

    fig, ax = plt.subplots(figsize=(20, 7))
    bottom = np.zeros(n_clients, dtype=float)
    for i, label in enumerate(labels):
        color = colors.get(label, f"C{i}")
        ax.bar(
            x,
            values[i],
            bottom=bottom,
            width=0.9,
            label=label,
            color=color,
            linewidth=0,
        )
        bottom += values[i]

    ax.set_title(
        f"Per-Client Label Composition (label alpha={label_alpha}, "
        f"{'proportions' if normalize else 'counts'})"
    )
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Proportion" if normalize else "Samples")
    if normalize:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.5, n_clients - 0.5)
    ax.set_xticks(np.arange(0, n_clients, 5))
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 1.13))

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    base_path = Path(__file__).parent
    output_dir = base_path / "results" / "ham10000_dirichlet_22rounds"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = LABEL_ORDER["ham10000"]
    real_hist = {label: int(GROUND_TRUTH["ham10000"][label]) for label in labels}
    train_hist, test_hist = split_histogram_by_ratio(real_hist, TEST_RATIO)
    total_samples = int(sum(train_hist.values()))
    gamma = CLIENTS_PER_ROUND / N_CLIENTS
    picked_ids_1000, estimate_round_picks, picked_ids_source_path = (
        generate_and_save_picked_ids_source(
            output_dir=output_dir,
            n_clients=N_CLIENTS,
            clients_per_round=CLIENTS_PER_ROUND,
            n_rounds_total=PICKED_IDS_TOTAL_ROUNDS,
            n_rounds_for_estimate=N_ROUNDS,
            seed=BASE_SEED,
        )
    )

    print("=" * 80)
    print("HAM10000 DIRICHLET CLIENT SIMULATION (3 LABEL SCENARIOS, TRAIN 90%)")
    print("=" * 80)
    print(f"Real total samples: {sum(real_hist.values())}")
    print(f"Train samples (used): {sum(train_hist.values())}")
    print(f"Test samples (holdout): {sum(test_hist.values())}")
    print(f"Real global histogram: {real_hist}")
    print(f"Train global histogram (fixed target): {train_hist}")
    print(
        f"Client size model: min={MIN_SAMPLES_PER_CLIENT} + Dirichlet(alpha={CLIENT_SIZE_ALPHA})"
    )
    print(
        f"Estimation setup: 10/round, rounds={N_ROUNDS}, with_replacement=True, "
        f"count_client_once=True, clipping={CLIPPING_METHOD}, "
        f"C={CLIPPING_THRESHOLD}, epsilon={EPSILON}"
    )
    print(f"Picked-ID source file: {picked_ids_source_path}")
    print(
        f"Picked rounds loaded: {len(picked_ids_1000)} (estimation uses first {N_ROUNDS})"
    )

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
            labels=labels,
            client_sizes=client_sizes,
            label_alpha=label_alpha,
            target_histogram=train_hist,
            rng=rng,
        )
        true_hist = aggregate(client_histograms, labels)

        csv_path = output_dir / f"clients_label_alpha_{label_alpha}.csv"
        write_clients_csv(csv_path, labels, client_sizes, client_histograms)
        hist_path = output_dir / f"clients_label_alpha_{label_alpha}_samples_hist.png"
        save_client_samples_histogram(hist_path, client_sizes, label_alpha)
        stacked_counts_path = (
            output_dir / f"clients_label_alpha_{label_alpha}_stacked_counts.png"
        )
        stacked_props_path = (
            output_dir / f"clients_label_alpha_{label_alpha}_stacked_proportions.png"
        )
        save_client_label_stacked_bars(
            stacked_counts_path, labels, client_histograms, label_alpha, normalize=False
        )
        save_client_label_stacked_bars(
            stacked_props_path, labels, client_histograms, label_alpha, normalize=True
        )

        print("\n" + "-" * 80)
        print(f"Scenario label alpha={label_alpha}")
        print(
            f"Client size stats: min={min(client_sizes)}, max={max(client_sizes)}, "
            f"mean={np.mean(client_sizes):.2f}"
        )
        print(f"True aggregate: {true_hist}")
        print(f"Per-client histograms written to: {csv_path}")
        print(f"Client sample histogram saved to: {hist_path}")
        print(f"Client label stacked counts plot: {stacked_counts_path}")
        print(f"Client label stacked proportions plot: {stacked_props_path}")
        print("Client histograms:")
        for client_id, hist in enumerate(client_histograms):
            print(f"  client {client_id:03d}: size={client_sizes[client_id]}, hist={hist}")

        dataset_config = DatasetConfig(
            name=f"ham10000_dirichlet_alpha_{label_alpha}",
            n_clients=N_CLIENTS,
            labels=labels,
            ground_truth=true_hist,
            data_path=Path("synthetic"),
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
        runner.save_results([result], f"ham10000_dirichlet_alpha_{label_alpha}_results.json")

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
                result.sample_noise_values[0]
                if result.sample_noise_values
                else None
            ),
            first_rounds_match_estimate=first_rounds_match_estimate,
            picked_ids_source_file=str(picked_ids_source_path),
        )

        kl = result.aggregated_metrics["kl_divergence"]
        tv = result.aggregated_metrics["tv_distance"]
        l1 = result.aggregated_metrics["l1_error"]
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
        if result.sample_estimated_histograms:
            print(f"Sample estimate (trial 1): {result.sample_estimated_histograms[0]}")

        summary.append(
            {
                "label_alpha": label_alpha,
                "seed": seed,
                "clients_csv": str(csv_path),
                "stacked_counts_plot": str(stacked_counts_path),
                "stacked_proportions_plot": str(stacked_props_path),
                "test_ratio": TEST_RATIO,
                "real_histogram": real_hist,
                "train_histogram": train_hist,
                "test_histogram": test_hist,
                "client_size_stats": {
                    "min": int(min(client_sizes)),
                    "max": int(max(client_sizes)),
                    "mean": float(np.mean(client_sizes)),
                },
                "true_histogram": true_hist,
                "coverage_first_trial": coverage,
                "picked_ids_file_trial_1": str(picked_ids_path),
                "picked_ids_total_rounds": PICKED_IDS_TOTAL_ROUNDS,
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

    summary_path = output_dir / "ham10000_dirichlet_22rounds_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Summary saved to: {summary_path}")
    print(f"All outputs saved in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
