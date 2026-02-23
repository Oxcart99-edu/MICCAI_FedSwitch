"""
Shared utilities for simulation scripts.

Provides:
  - Synthetic client generation (size skew + label skew via Dirichlet).
  - Histogram train/test splitting.
  - Picked-ID generation and persistence.
  - CSV and plot helpers.
  - A high-level run_simulation() entry point used by all four scripts.
"""

from __future__ import annotations

import csv
import json
from math import floor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import DatasetConfig
from .data_loaders import compute_aggregate_histogram
from .distributed_noise import compute_trigger_round, compute_target_cohort
from .experiment import ExperimentRunner


# ---------------------------------------------------------------------------
# Client generation
# ---------------------------------------------------------------------------

def sample_client_sizes(
    total_samples: int,
    n_clients: int,
    min_samples: int,
    alpha: float,
    rng: np.random.Generator,
) -> List[int]:
    """Min-Dirichlet client size allocation."""
    baseline = n_clients * min_samples
    if baseline > total_samples:
        raise ValueError(f"min requirement {baseline} > total {total_samples}")
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
    """
    Dirichlet per-client label heterogeneity with exact global totals.

    Draws per-client multinomial counts then iteratively transfers mass
    between surplus and deficit labels to match the target exactly.
    """
    n_clients = len(client_sizes)
    n_labels = len(labels)
    probs = rng.dirichlet(np.full(n_labels, label_alpha), size=n_clients)
    counts = np.vstack(
        [rng.multinomial(size, probs[i]) for i, size in enumerate(client_sizes)]
    ).astype(int)

    target = np.array([target_histogram[label] for label in labels], dtype=int)
    diff = target - counts.sum(axis=0)
    eps = 1e-12

    for _ in range(2000):
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
            for s in surplus_labels[np.argsort(diff[surplus_labels])]:
                if int(-diff[s]) <= 0 or need <= 0:
                    continue
                rows = np.where(counts[:, s] > 0)[0]
                if rows.size == 0:
                    continue
                scores = probs[rows, d] / (probs[rows, s] + eps)
                rows = rows[np.argsort(-scores)]
                for r in rows:
                    if need <= 0 or diff[s] >= 0:
                        break
                    t = min(int(counts[r, s]), need, int(-diff[s]))
                    if t <= 0:
                        continue
                    counts[r, s] -= t
                    counts[r, d] += t
                    diff[s] += t
                    diff[d] -= t
                    need -= t
                    moved_any = True

        if not moved_any:
            raise RuntimeError("Could not reconcile client histograms.")

    if not np.all(diff == 0):
        raise RuntimeError("Failed to match exact global label totals.")

    return [
        {label: int(counts[i, j]) for j, label in enumerate(labels)}
        for i in range(n_clients)
    ]


# ---------------------------------------------------------------------------
# Train / test splitting
# ---------------------------------------------------------------------------

def split_histogram_by_ratio(
    histogram: Dict[str, int], test_ratio: float
) -> tuple[Dict[str, int], Dict[str, int]]:
    """Deterministic largest-remainder split of a histogram."""
    total = sum(histogram.values())
    target_test = int(round(total * test_ratio))

    raw = {l: histogram[l] * test_ratio for l in histogram}
    base = {l: int(floor(raw[l])) for l in histogram}
    missing = target_test - sum(base.values())

    remainders = sorted(histogram, key=lambda l: raw[l] - base[l], reverse=True)
    for l in remainders[:missing]:
        base[l] += 1

    test = {l: min(base[l], histogram[l]) for l in histogram}
    train = {l: histogram[l] - test[l] for l in histogram}
    return train, test


# ---------------------------------------------------------------------------
# Picked-ID generation
# ---------------------------------------------------------------------------

def generate_picked_ids(
    n_clients: int,
    clients_per_round: int,
    n_rounds: int,
    seed: int,
) -> List[List[int]]:
    """Generate n_rounds of client picks (with replacement across rounds)."""
    rng = np.random.default_rng(seed)
    replace = clients_per_round > n_clients
    return [
        rng.choice(n_clients, size=clients_per_round, replace=replace).tolist()
        for _ in range(n_rounds)
    ]


def save_picked_ids(
    output_dir: Path,
    picks: List[List[int]],
    n_clients: int,
    clients_per_round: int,
    seed: int,
) -> Path:
    """Persist picked IDs and a coverage summary; return the picks path."""
    payload = [{"round": i + 1, "picked_ids": ids} for i, ids in enumerate(picks)]
    path = output_dir / f"picked_ids_{len(picks)}_rounds_seed{seed}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    seen: set = set()
    coverage = []
    for ids in picks:
        seen.update(ids)
        coverage.append(len(seen))

    summary = {
        "seed": seed,
        "n_clients": n_clients,
        "clients_per_round": clients_per_round,
        "n_rounds": len(picks),
        "final_unique_clients": coverage[-1] if coverage else 0,
        "unique_coverage_by_round": coverage,
    }
    with open(path.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return path


# ---------------------------------------------------------------------------
# Coverage helper
# ---------------------------------------------------------------------------

def unique_coverage_by_round(client_sets: List[List[int]]) -> List[int]:
    seen: set = set()
    coverage = []
    for ids in client_sets:
        seen.update(ids)
        coverage.append(len(seen))
    return coverage


# ---------------------------------------------------------------------------
# CSV / plot helpers
# ---------------------------------------------------------------------------

def write_clients_csv(
    csv_path: Path,
    labels: List[str],
    client_histograms: List[Dict[str, int]],
) -> List[int]:
    """Write per-client histograms to CSV. Returns client sizes."""
    client_sizes = [sum(h.values()) for h in client_histograms]
    fieldnames = (
        ["client_id", "client_size"]
        + [f"count_{l}" for l in labels]
        + [f"prop_{l}" for l in labels]
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid, hist in enumerate(client_histograms):
            sz = client_sizes[cid]
            row = {"client_id": cid, "client_size": sz}
            for l in labels:
                row[f"count_{l}"] = hist[l]
                row[f"prop_{l}"] = hist[l] / sz if sz > 0 else 0.0
            writer.writerow(row)
    return client_sizes


def save_stacked_bars(
    output_path: Path,
    labels: List[str],
    client_histograms: List[Dict[str, int]],
    title: str,
    normalize: bool = False,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """Save a per-client stacked bar chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    nc = len(client_histograms)
    x = np.arange(nc)
    values = np.array(
        [[client_histograms[i][l] for i in range(nc)] for l in labels], dtype=float
    )
    if normalize:
        totals = values.sum(axis=0)
        totals[totals == 0] = 1.0
        values /= totals

    fig, ax = plt.subplots(figsize=(min(20, max(10, nc * 0.2)), 6))
    bottom = np.zeros(nc)
    for i, l in enumerate(labels):
        c = (colors or {}).get(l, f"C{i}")
        ax.bar(x, values[i], bottom=bottom, width=0.9, label=l, color=c, linewidth=0)
        bottom += values[i]

    ax.set_title(f"{title} ({'proportions' if normalize else 'counts'})")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Proportion" if normalize else "Samples")
    if normalize:
        ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, nc - 0.5)
    ax.legend(ncol=min(len(labels), 7), loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# High-level simulation entry point
# ---------------------------------------------------------------------------

def run_simulation(
    *,
    name: str,
    labels: List[str],
    client_histograms: List[Dict[str, int]],
    n_clients: int,
    clients_per_round: int,
    n_rounds: int,
    clipping_threshold: int,
    clipping_method: str,
    epsilon: float,
    alpha: float,
    n_trials: int,
    seed: int,
    output_dir: Path,
    base_path: Path,
    failure_rate: float = 0.0,
    picked_ids_total_rounds: int = 1000,
) -> dict:
    """
    Run the full estimation pipeline for one scenario and return a summary dict.

    Steps:
      1. Generate and save picked IDs.
      2. Build DatasetConfig.
      3. Run ExperimentRunner.
      4. Return a JSON-serialisable summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gamma = clients_per_round / n_clients

    # Picked IDs
    all_picks = generate_picked_ids(n_clients, clients_per_round, picked_ids_total_rounds, seed)
    estimate_picks = all_picks[:n_rounds]
    picks_path = save_picked_ids(output_dir, all_picks, n_clients, clients_per_round, seed)

    true_hist = compute_aggregate_histogram(client_histograms, labels)

    config = DatasetConfig(
        name=name,
        n_clients=n_clients,
        labels=labels,
        ground_truth=true_hist,
        data_path=Path("synthetic"),
        gamma=gamma,
        rounds=n_rounds,
        sampling_with_replacement=True,
        count_client_once=True,
    )

    runner = ExperimentRunner(base_path, output_dir=output_dir, verbose=False)
    result = runner.run_configuration(
        config=config,
        client_histograms=client_histograms,
        clipping_threshold=clipping_threshold,
        epsilon=epsilon,
        failure_rate=failure_rate,
        n_trials=n_trials,
        alpha=alpha,
        clipping_method=clipping_method,
        base_seed=seed,
        fixed_client_sets_per_round=estimate_picks,
    )
    runner.save_results([result], f"{name}_results.json")

    # Computed protocol parameters
    target_cohort = compute_target_cohort(alpha, n_clients)
    trigger_round = compute_trigger_round(alpha, gamma, n_clients)

    first_sets = result.sample_client_sets[0] if result.sample_client_sets else []
    coverage = unique_coverage_by_round(first_sets)

    kl = result.aggregated_metrics["kl_divergence"]
    tv = result.aggregated_metrics["tv_distance"]
    l1 = result.aggregated_metrics["l1_error"]

    client_sizes = [sum(h.values()) for h in client_histograms]

    summary = {
        "dataset": name,
        "seed": seed,
        "n_clients": n_clients,
        "labels": labels,
        "true_histogram": true_hist,
        "clients_per_round": clients_per_round,
        "gamma": gamma,
        "alpha": alpha,
        "target_cohort": target_cohort,
        "trigger_round_R_star": trigger_round,
        "n_rounds": n_rounds,
        "clipping_threshold": clipping_threshold,
        "clipping_method": clipping_method,
        "epsilon": epsilon,
        "failure_rate": failure_rate,
        "n_trials": n_trials,
        "client_size_stats": {
            "min": int(min(client_sizes)),
            "max": int(max(client_sizes)),
            "mean": float(np.mean(client_sizes)),
        },
        "coverage_first_trial": coverage,
        "metrics": result.aggregated_metrics,
        "sample_estimate_trial_1": (
            result.sample_estimated_histograms[0]
            if result.sample_estimated_histograms
            else None
        ),
        "picked_ids_source": str(picks_path),
    }
    return summary
