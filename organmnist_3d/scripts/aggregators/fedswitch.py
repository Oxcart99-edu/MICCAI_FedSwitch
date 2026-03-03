"""
FedSwitch aggregator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .base import RoundAggregationPlan, normalize_weights
from .fedavg import FedAvgAggregator


class FedSwitchAggregator(FedAvgAggregator):
    name = "fedswitch"

    def __init__(self) -> None:
        self.rounds = 0
        self.steps_per_round: int | None = None
        self.switch_round = 1
        self.switch_weight_source = "client"
        self.scenario_alpha: Optional[str] = None
        self.estimate_json_dir: Path = Path("sample_distribution")
        self.estimate_json_path: Optional[Path] = None
        self.estimate_histogram: Optional[Dict[int, float]] = None
        self.estimate_trigger_round: Optional[int] = None

    def configure(
        self,
        *,
        rounds: int,
        steps_per_round: int | None,
        switch_round: int | None,
        switch_weight_source: str,
        fedprox_mu: float,
        learning_rate: float,
        fedlc_tau: float = 0.5,
        scenario_alpha: str | None = None,
        estimate_json_dir: str | None = None,
    ) -> None:
        del fedprox_mu, learning_rate, fedlc_tau
        self.rounds = int(rounds)
        self.steps_per_round = steps_per_round
        self.switch_weight_source = switch_weight_source
        self.scenario_alpha = str(scenario_alpha) if scenario_alpha is not None else None
        if estimate_json_dir:
            self.estimate_json_dir = Path(estimate_json_dir).expanduser().resolve()
        self.estimate_json_path = None
        self.estimate_histogram = None
        self.estimate_trigger_round = None

        json_switch_round = self._load_alpha_estimate_metadata()
        if json_switch_round is not None:
            self.estimate_trigger_round = int(json_switch_round)
            self.switch_round = min(max(1, self.estimate_trigger_round), self.rounds)
        else:
            self.switch_round = int(switch_round) if switch_round is not None else 1

        if self.switch_round < 1 or self.switch_round > self.rounds:
            raise ValueError("switch-round must be in [1, rounds].")
        if self.switch_weight_source == "steps" and (self.steps_per_round is None or self.steps_per_round <= 0):
            raise ValueError(
                "fedswitch with --fedswitch-weight-source steps requires --steps-per-round > 0."
            )

    def experiment_note(self) -> str:
        step_mode_forced = self.steps_per_round is not None and self.steps_per_round > 0
        source_desc = "step-seen labels/samples" if self.switch_weight_source == "steps" else "selected-client full distributions"
        if step_mode_forced and self.switch_weight_source != "steps":
            source_desc = "step-seen labels/samples (forced by --steps-per-round)"
        estimate_note = ""
        if self.estimate_histogram is not None:
            if self.estimate_json_path is not None:
                estimate_note = f" using estimated class histogram from {self.estimate_json_path.name}"
            else:
                estimate_note = " using estimated class histogram"
        return f"[fedswitch] Using uniform weights before round {self.switch_round}, then stratified weights from {source_desc}{estimate_note}."

    def compute_client_weights(
        self,
        client_sizes: List[int],
        *,
        class_counts: List[dict],
        num_classes: int,
    ) -> List[float]:
        del client_sizes
        totals: Dict[int, int] = {}
        for counts in class_counts:
            for label, cnt in counts.items():
                label_id = int(label)
                totals[label_id] = totals.get(label_id, 0) + int(cnt)

        client_weights: List[float] = []
        for counts in class_counts:
            weight = 0.0
            for label, total_cnt in totals.items():
                if total_cnt <= 0:
                    continue
                client_cnt = int(counts.get(label, 0))
                weight += (client_cnt / total_cnt) * (1.0 / num_classes)
            client_weights.append(weight)

        if sum(client_weights) <= 0:
            raise ValueError("All client weights computed to zero; check class_counts input.")
        return client_weights

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
        if round_idx < self.switch_round:
            raw_weights = [1.0] * len(selected_client_sizes)
            return RoundAggregationPlan(
                client_sizes=selected_client_sizes,
                class_counts=selected_client_counts,
                client_weights=normalize_weights(raw_weights),
                raw_client_weights=raw_weights,
                phase_suffix=" phase=uniform",
            )

        use_step_seen = (self.steps_per_round is not None and self.steps_per_round > 0) or self.switch_weight_source == "steps"
        if use_step_seen:
            if len(step_seen_sizes) != len(selected_client_sizes):
                raise ValueError("step_seen_sizes and selected_client_sizes must have the same length.")
            if len(step_seen_label_counts) != len(selected_client_counts):
                raise ValueError("step_seen_label_counts and selected_client_counts must have the same length.")
            if any(size <= 0 for size in step_seen_sizes):
                raise ValueError(
                    "fedswitch step-based weighting found a client with zero seen samples. Increase --steps-per-round."
                )
            agg_sizes = step_seen_sizes
            agg_counts: List[dict] = [dict(row) for row in step_seen_label_counts]
        else:
            agg_sizes = selected_client_sizes
            agg_counts = selected_client_counts

        if self.estimate_histogram is not None:
            weights = self._compute_estimated_stratified_weights(
                class_counts=agg_counts,
                num_classes=num_classes,
            )
            phase_suffix = " phase=stratified_estimated_steps" if use_step_seen else " phase=stratified_estimated"
        else:
            weights = self.compute_client_weights(
                agg_sizes,
                class_counts=agg_counts,
                num_classes=num_classes,
            )
            phase_suffix = " phase=stratified_steps" if use_step_seen else " phase=stratified"
        return RoundAggregationPlan(
            client_sizes=agg_sizes,
            class_counts=agg_counts,
            client_weights=normalize_weights(weights),
            raw_client_weights=[float(w) for w in weights],
            phase_suffix=phase_suffix,
        )

    def _load_alpha_estimate_metadata(self) -> int | None:
        if self.scenario_alpha is None:
            return None
        if not self.estimate_json_dir.exists():
            return None

        alpha_tokens = {self.scenario_alpha}
        try:
            alpha_value = float(self.scenario_alpha)
            alpha_tokens.add(str(alpha_value))
            alpha_tokens.add(f"{alpha_value:.1f}")
        except ValueError:
            pass

        candidates: List[Path] = []
        for alpha_token in sorted(alpha_tokens):
            patterns = [
                f"picked_ids_used_for_estimate_alpha_{alpha_token}_seed_*.json",
                f"picked_ids_used_for_estimate_alpha_{alpha_token}_seed_42.json",
            ]
            for pattern in patterns:
                candidates.extend(sorted(self.estimate_json_dir.glob(pattern)))
        candidates = sorted(set(candidates))
        if not candidates:
            return None
        estimate_json = candidates[0]

        with estimate_json.open(encoding="utf-8") as f:
            payload = json.load(f)
        trigger_round = payload.get("estimate_trigger_round")
        estimate_hist = payload.get("estimate_histogram_from_these_ids")
        if not isinstance(trigger_round, int):
            return None
        if not isinstance(estimate_hist, dict):
            return None

        parsed_hist: Dict[int, float] = {}
        for raw_label, raw_count in estimate_hist.items():
            label = int(raw_label)
            count = float(raw_count)
            if count > 0:
                parsed_hist[label] = count
        if not parsed_hist:
            return None

        self.estimate_json_path = estimate_json
        self.estimate_histogram = parsed_hist
        return int(trigger_round)

    def _compute_estimated_stratified_weights(
        self,
        *,
        class_counts: List[dict],
        num_classes: int,
    ) -> List[float]:
        if self.estimate_histogram is None:
            raise ValueError("estimate_histogram is required for estimated stratified weights.")

        client_weights: List[float] = []
        for counts in class_counts:
            weight = 0.0
            for label in range(num_classes):
                total_cnt = self.estimate_histogram.get(label, 0.0)
                if total_cnt <= 0:
                    continue
                client_cnt = float(counts.get(label, 0))
                weight += (client_cnt / total_cnt) * (1.0 / num_classes)
            client_weights.append(weight)

        if sum(client_weights) <= 0:
            raise ValueError("All client weights computed to zero from estimate histogram.")
        return client_weights
