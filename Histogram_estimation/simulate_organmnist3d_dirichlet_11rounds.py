#!/usr/bin/env python3
"""
OrganMNIST3D synthetic client simulation with 3 label-Dirichlet scenarios.

Client sizes: min 12 + Dirichlet(0.5) residuals.
Label skew: Dirichlet(α_d) with α_d ∈ {0.1, 0.5, 1.0}.
Target histogram: train + val counts (test held out).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from heavy_hitters_protocol.simulation import (
    sample_client_sizes,
    sample_client_histograms,
    write_clients_csv,
    save_stacked_bars,
    run_simulation,
)

# ── Configuration ────────────────────────────────────────────────────────────
N_CLIENTS = 50
MIN_SAMPLES_PER_CLIENT = 12
CLIENT_SIZE_ALPHA = 0.5
LABEL_ALPHAS = [0.1, 0.5, 1.0]
CLIENTS_PER_ROUND = 10
N_ROUNDS = 11
CLIPPING_THRESHOLD = 12
CLIPPING_METHOD = "uniform_with_replacement"
EPSILON = 2
ALPHA = 0.9
N_TRIALS = 30
SEED = 42

LABELS = [str(i) for i in range(8)]

TRAIN_HIST = {"0": 115, "1": 230, "2": 184, "3": 94, "4": 40, "5": 78, "6": 115, "7": 115}
VAL_HIST = {"0": 16, "1": 30, "2": 29, "3": 15, "4": 13, "5": 26, "6": 16, "7": 16}
TEST_HIST = {"0": 69, "1": 137, "2": 130, "3": 66, "4": 28, "5": 42, "6": 69, "7": 69}
TRAIN_VAL_HIST = {k: TRAIN_HIST[k] + VAL_HIST[k] for k in LABELS}


def main() -> None:
    base_path = Path(__file__).parent
    output_dir = base_path / "results" / "organmnist3d_dirichlet"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = sum(TRAIN_VAL_HIST.values())
    all_summaries: List[dict] = []

    for label_alpha in LABEL_ALPHAS:
        rng = np.random.default_rng(SEED)

        sizes = sample_client_sizes(total_samples, N_CLIENTS, MIN_SAMPLES_PER_CLIENT, CLIENT_SIZE_ALPHA, rng)
        hists = sample_client_histograms(LABELS, sizes, label_alpha, TRAIN_VAL_HIST, rng)

        tag = f"alpha_{label_alpha}"
        write_clients_csv(output_dir / f"clients_{tag}.csv", LABELS, hists)
        save_stacked_bars(
            output_dir / f"clients_{tag}_counts.png",
            LABELS, hists, f"OrganMNIST3D D({label_alpha})",
        )
        save_stacked_bars(
            output_dir / f"clients_{tag}_props.png",
            LABELS, hists, f"OrganMNIST3D D({label_alpha})", normalize=True,
        )

        summary = run_simulation(
            name=f"organmnist3d_D{label_alpha}",
            labels=LABELS,
            client_histograms=hists,
            n_clients=N_CLIENTS,
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
        summary["label_alpha"] = label_alpha
        all_summaries.append(summary)

        kl = summary["metrics"]["kl_divergence"]
        print(f"  D({label_alpha}): KL={kl['mean']:.4f} "
              f"[{kl['ci_lower']:.4f}, {kl['ci_upper']:.4f}]")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"Saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
