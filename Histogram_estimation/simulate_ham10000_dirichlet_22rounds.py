#!/usr/bin/env python3
"""
HAM10000 synthetic client simulation with 3 label-Dirichlet scenarios.

Client sizes: min 50 + Dirichlet(0.5) residuals.
Label skew: Dirichlet(α_d) with α_d ∈ {0.1, 0.5, 1.0}.
Uses a 90/10 train/test split of the real global histogram as target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from heavy_hitters_protocol.config import GROUND_TRUTH, LABEL_ORDER
from heavy_hitters_protocol.simulation import (
    split_histogram_by_ratio,
    sample_client_sizes,
    sample_client_histograms,
    write_clients_csv,
    save_stacked_bars,
    run_simulation,
)

# ── Configuration ────────────────────────────────────────────────────────────
N_CLIENTS = 100
MIN_SAMPLES_PER_CLIENT = 50
CLIENT_SIZE_ALPHA = 0.5
LABEL_ALPHAS = [0.1, 0.5, 1.0]
CLIENTS_PER_ROUND = 10
N_ROUNDS = 22
CLIPPING_THRESHOLD = 50
CLIPPING_METHOD = "uniform_with_replacement"
EPSILON = 1.0
ALPHA = 0.9
N_TRIALS = 30
SEED = 42
TEST_RATIO = 0.10


def main() -> None:
    base_path = Path(__file__).parent
    output_dir = base_path / "results" / "ham10000_dirichlet"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = LABEL_ORDER["ham10000"]
    real_hist = {l: int(GROUND_TRUTH["ham10000"][l]) for l in labels}
    train_hist, test_hist = split_histogram_by_ratio(real_hist, TEST_RATIO)
    total_train = sum(train_hist.values())

    all_summaries: List[dict] = []

    for label_alpha in LABEL_ALPHAS:
        rng = np.random.default_rng(SEED)

        sizes = sample_client_sizes(total_train, N_CLIENTS, MIN_SAMPLES_PER_CLIENT, CLIENT_SIZE_ALPHA, rng)
        hists = sample_client_histograms(labels, sizes, label_alpha, train_hist, rng)

        tag = f"alpha_{label_alpha}"
        write_clients_csv(output_dir / f"clients_{tag}.csv", labels, hists)
        save_stacked_bars(
            output_dir / f"clients_{tag}_counts.png",
            labels, hists, f"HAM10000 D({label_alpha})",
        )
        save_stacked_bars(
            output_dir / f"clients_{tag}_props.png",
            labels, hists, f"HAM10000 D({label_alpha})", normalize=True,
        )

        summary = run_simulation(
            name=f"ham10000_D{label_alpha}",
            labels=labels,
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
        summary["test_ratio"] = TEST_RATIO
        all_summaries.append(summary)

        kl = summary["metrics"]["kl_divergence"]
        print(f"  D({label_alpha}): KL={kl['mean']:.4f} "
              f"[{kl['ci_lower']:.4f}, {kl['ci_upper']:.4f}]")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"Saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
