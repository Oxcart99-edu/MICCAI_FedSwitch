#!/usr/bin/env python3
"""
Arrhythmia simulation from real patient-level distributions.

Uses original patient-by-patient histograms directly (no synthetic generation).
A 90/10 train/test split is applied per patient before estimation.
"""

from __future__ import annotations

import json
from pathlib import Path

from heavy_hitters_protocol.config import DatasetConfig, LABEL_ORDER
from heavy_hitters_protocol.data_loaders import load_client_data
from heavy_hitters_protocol.simulation import (
    split_histogram_by_ratio,
    write_clients_csv,
    save_stacked_bars,
    run_simulation,
)

# ── Configuration ────────────────────────────────────────────────────────────
CLIENTS_PER_ROUND = 16
N_ROUNDS = 6
CLIPPING_THRESHOLD = 1024
CLIPPING_METHOD = "uniform_with_replacement"
EPSILON = 2
ALPHA = 0.9
N_TRIALS = 30
SEED = 10
TEST_RATIO = 0.10


def main() -> None:
    base_path = Path(__file__).parent
    output_dir = base_path / "results" / "arrhythmia_real"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load real patient histograms and split train/test
    base_config = DatasetConfig.arrhythmia(base_path)
    labels = LABEL_ORDER["arrhythmia"]
    full = load_client_data(base_config)

    train_hists, test_hists = [], []
    for hist in full:
        tr, te = split_histogram_by_ratio(hist, TEST_RATIO)
        train_hists.append(tr)
        test_hists.append(te)

    n_clients = len(train_hists)

    # Save per-client CSV and plots
    write_clients_csv(output_dir / "clients.csv", labels, train_hists)
    save_stacked_bars(
        output_dir / "clients_stacked_counts.png",
        labels, train_hists, "Arrhythmia train (90%)",
    )
    save_stacked_bars(
        output_dir / "clients_stacked_props.png",
        labels, train_hists, "Arrhythmia train (90%)", normalize=True,
    )

    # Run estimation
    summary = run_simulation(
        name="arrhythmia_real",
        labels=labels,
        client_histograms=train_hists,
        n_clients=n_clients,
        clients_per_round=CLIENTS_PER_ROUND,
        n_rounds=N_ROUNDS,
        clipping_threshold=CLIPPING_THRESHOLD,
        clipping_method=CLIPPING_METHOD,
        epsilon=EPSILON,
        alpha=ALPHA,
        n_trials=N_TRIALS,
        seed=SEED,
        output_dir=output_dir,
        base_path=base_path,
    )

    summary["test_ratio"] = TEST_RATIO
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    kl = summary["metrics"]["kl_divergence"]
    print("=" * 70)
    print(f"Arrhythmia (real, {n_clients} patients, {N_ROUNDS} rounds)")
    print(f"  target cohort |A| = {summary['target_cohort']}, "
          f"R* = {summary['trigger_round_R_star']}")
    print(f"  KL = {kl['mean']:.4f} [{kl['ci_lower']:.4f}, {kl['ci_upper']:.4f}]")
    print(f"  Summary: {output_dir / 'summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
