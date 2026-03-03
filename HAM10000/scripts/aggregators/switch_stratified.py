"""
Switch-stratified aggregator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .base import RoundAggregationPlan, normalize_weights
from .stratified import StratifiedAggregator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DISTRIBUTION_DIR = PROJECT_ROOT / "sample_distribution"


class SwitchStratifiedAggregator(StratifiedAggregator):
    name = "switch_stratified"

    def __init__(self) -> None:
        self.rounds = 0
        self.steps_per_round: int | None = None
        self.switch_round = 22
        self.switch_weight_source = "client"
        self._estimate_json_path: Path | None = None
        self._target_estimate: dict[str, float] | None = None
        self._class_labels: List[str] = []

    def configure(
        self,
        *,
        rounds: int,
        steps_per_round: int | None,
        switch_round: int | None,
        switch_weight_source: str,
        dirichlet_alpha: float,
        fedprox_mu: float,
        learning_rate: float,
    ) -> None:
        del fedprox_mu, learning_rate
        self.rounds = int(rounds)
        self.steps_per_round = steps_per_round
        self.switch_round = int(switch_round) if switch_round is not None else 22
        self.switch_weight_source = switch_weight_source
        self._target_estimate = self._load_target_estimate(dirichlet_alpha=dirichlet_alpha)
        self._class_labels = sorted(self._target_estimate.keys())

        if self.switch_round < 1 or self.switch_round > self.rounds:
            raise ValueError("switch-round must be in [1, rounds].")
        if self.steps_per_round is None or self.steps_per_round <= 0:
            raise ValueError(
                "switch_stratified requires --steps-per-round > 0 because post-switch weighting "
                "must use only step-seen samples."
            )
        if self.switch_weight_source == "steps" and (self.steps_per_round is None or self.steps_per_round <= 0):
            raise ValueError(
                "switch_stratified with --switch-stratified-weight-source steps requires --steps-per-round > 0."
            )

    def experiment_note(self) -> str:
        estimate_source = str(self._estimate_json_path) if self._estimate_json_path is not None else "<unresolved>"
        estimate_values = "<unresolved>"
        if self._target_estimate is not None:
            parts = []
            for label, value in self._target_estimate.items():
                value_str = str(int(value)) if float(value).is_integer() else f"{value:g}"
                parts.append(f"{label}:{value_str}")
            estimate_values = ", ".join(parts)
        return (
            f"[switch_stratified] Using FedAvg-on-step-seen weights before round {self.switch_round}, "
            "then stratified-style weights from step-seen labels/samples "
            f"using sample_estimated_histograms[0] from {estimate_source}.\n"
            f"[switch_stratified] denominator_distribution={estimate_values}"
        )

    def _resolve_estimate_json_path(self, dirichlet_alpha: float) -> Path:
        if not SAMPLE_DISTRIBUTION_DIR.exists():
            raise FileNotFoundError(f"Missing sample_distribution directory: {SAMPLE_DISTRIBUTION_DIR}")

        candidates = [
            SAMPLE_DISTRIBUTION_DIR / f"ham10000_dirichlet_alpha_{dirichlet_alpha}_results.json",
            SAMPLE_DISTRIBUTION_DIR / f"ham10000_dirichlet_alpha_{dirichlet_alpha:g}_results.json",
            SAMPLE_DISTRIBUTION_DIR / f"ham10000_dirichlet_alpha_{dirichlet_alpha:.1f}_results.json",
        ]
        for path in candidates:
            if path.exists():
                return path

        prefix = "ham10000_dirichlet_alpha_"
        suffix = "_results.json"
        parsed_paths: List[tuple[float, Path]] = []
        for path in SAMPLE_DISTRIBUTION_DIR.glob("ham10000_dirichlet_alpha_*_results.json"):
            tag = path.name[len(prefix) : -len(suffix)]
            try:
                alpha_val = float(tag)
            except ValueError:
                continue
            parsed_paths.append((alpha_val, path))

        if not parsed_paths:
            raise FileNotFoundError(
                "No estimate JSON found in sample_distribution matching "
                "'ham10000_dirichlet_alpha_*_results.json'."
            )

        parsed_paths.sort(key=lambda item: abs(item[0] - dirichlet_alpha))
        best_alpha, best_path = parsed_paths[0]
        if abs(best_alpha - dirichlet_alpha) > 1e-12:
            available = ", ".join(str(alpha) for alpha, _ in sorted(parsed_paths, key=lambda item: item[0]))
            raise FileNotFoundError(
                "No exact estimate JSON for dirichlet alpha "
                f"{dirichlet_alpha}. Available alphas: {available}"
            )
        return best_path

    def _load_target_estimate(self, *, dirichlet_alpha: float) -> dict[str, float]:
        estimate_json_path = self._resolve_estimate_json_path(dirichlet_alpha=dirichlet_alpha)
        self._estimate_json_path = estimate_json_path
        payload = json.loads(estimate_json_path.read_text())
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"Expected a non-empty list in {estimate_json_path}.")
        first_trial = payload[0]
        if not isinstance(first_trial, dict):
            raise ValueError(f"Expected first JSON entry to be an object in {estimate_json_path}.")
        histograms = first_trial.get("sample_estimated_histograms")
        if not isinstance(histograms, list) or not histograms:
            raise ValueError(f"Missing non-empty 'sample_estimated_histograms' in {estimate_json_path}.")
        estimate = histograms[0]
        if not isinstance(estimate, dict):
            raise ValueError(
                "Expected sample_estimated_histograms[0] to be an object "
                f"in {estimate_json_path}."
            )
        parsed: dict[str, float] = {}
        for label, value in estimate.items():
            try:
                parsed[str(label)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid value for label {label!r} in sample_estimated_histograms[0]: {value!r}"
                ) from exc
        return parsed

    def _count_for_class(self, counts: dict, class_idx: int) -> float:
        if class_idx in counts:
            return float(counts.get(class_idx, 0))
        idx_key = str(class_idx)
        if idx_key in counts:
            return float(counts.get(idx_key, 0))
        if class_idx < len(self._class_labels):
            label_key = self._class_labels[class_idx]
            if label_key in counts:
                return float(counts.get(label_key, 0))
        return 0.0

    def _estimate_for_class(self, estimate: dict[str, float], class_idx: int) -> float:
        if class_idx in estimate:
            return float(estimate.get(class_idx, 0.0))
        idx_key = str(class_idx)
        if idx_key in estimate:
            return float(estimate.get(idx_key, 0.0))
        if class_idx < len(self._class_labels):
            label_key = self._class_labels[class_idx]
            if label_key in estimate:
                return float(estimate.get(label_key, 0.0))
        return 0.0

    def _compute_estimate_stratified_weights(
        self,
        class_counts: List[dict],
        num_classes: int,
    ) -> List[float]:
        if self._target_estimate is None:
            raise RuntimeError("Target estimate distribution not initialized.")
        per_class = 1.0 / float(num_classes)
        weights: List[float] = []
        for counts in class_counts:
            weight = 0.0
            for class_idx in range(num_classes):
                client_cnt = self._count_for_class(counts, class_idx)
                estimate_cnt = self._estimate_for_class(self._target_estimate, class_idx)
                if estimate_cnt <= 0:
                    continue
                weight += (client_cnt / estimate_cnt) * per_class
            weights.append(weight)
        if sum(weights) <= 0:
            raise ValueError("All switch_stratified weights computed to zero; check class counts and estimate.")
        return weights

    def resolve_round_aggregation(
        self,
        *,
        round_idx: int,
        selected_client_sizes: List[int],
        selected_client_counts: List[dict],
        step_seen_sizes: List[int],
        step_seen_label_counts: List[Dict[int, int]],
        num_classes: int,
    ) -> RoundAggregationPlan:
        if len(step_seen_sizes) != len(selected_client_sizes):
            raise ValueError("step_seen_sizes and selected_client_sizes must have the same length.")
        if len(step_seen_label_counts) != len(selected_client_counts):
            raise ValueError("step_seen_label_counts and selected_client_counts must have the same length.")
        if any(size <= 0 for size in step_seen_sizes):
            raise ValueError(
                "switch_stratified step-based weighting found a client with zero seen samples. "
                "Increase --steps-per-round."
            )
        agg_sizes = step_seen_sizes
        agg_counts: List[dict] = [dict(row) for row in step_seen_label_counts]

        if round_idx < self.switch_round:
            return RoundAggregationPlan(
                client_sizes=agg_sizes,
                class_counts=agg_counts,
                client_weights=normalize_weights(agg_sizes),
                raw_client_weights=[float(x) for x in agg_sizes],
                phase_suffix=" phase=fedavg_steps",
            )

        weights = self._compute_estimate_stratified_weights(
            class_counts=agg_counts,
            num_classes=num_classes,
        )
        return RoundAggregationPlan(
            client_sizes=agg_sizes,
            class_counts=agg_counts,
            client_weights=normalize_weights(weights),
            raw_client_weights=[float(w) for w in weights],
            phase_suffix=" phase=estimate_stratified_steps",
        )
