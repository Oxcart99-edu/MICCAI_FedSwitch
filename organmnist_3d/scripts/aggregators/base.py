"""
Base interfaces and shared utilities for federated aggregators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

StateDict = Dict[str, torch.Tensor]


@dataclass
class LocalTrainResult:
    train_loss: float
    local_steps: int
    seen_samples: int
    seen_label_counts: Dict[int, int]
    seen_examples: Optional[List[Tuple[int, int]]] = None


@dataclass
class RoundAggregationPlan:
    client_sizes: List[int]
    class_counts: List[dict]
    client_weights: List[float]
    raw_client_weights: Optional[List[float]] = None
    phase_suffix: str = ""


def normalize_weights(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Weights must have a strictly positive sum.")
    return [float(w) / total for w in weights]


def weighted_average_state_dicts(state_dicts: List[StateDict], weights: List[float]) -> StateDict:
    if not state_dicts:
        raise ValueError("No state_dicts provided for aggregation.")
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have the same length.")

    agg_state: StateDict = {}
    for key in state_dicts[0].keys():
        sample_tensor = state_dicts[0][key]
        agg_tensor = torch.zeros_like(sample_tensor, dtype=torch.float32)
        for state, weight in zip(state_dicts, weights):
            agg_tensor += state[key].float() * weight
        if sample_tensor.dtype.is_floating_point:
            agg_state[key] = agg_tensor.type_as(sample_tensor)
        else:
            agg_state[key] = agg_tensor.round().to(dtype=sample_tensor.dtype)
    return agg_state


class Aggregator(ABC):
    name: str

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
    ) -> None:
        del rounds, switch_round, switch_weight_source, fedprox_mu, learning_rate, fedlc_tau
        self.steps_per_round = int(steps_per_round) if steps_per_round is not None else None

    def experiment_note(self) -> str:
        return ""

    def on_experiment_start(self, global_model: nn.Module, total_clients: int) -> None:
        del global_model, total_clients

    def on_round_start(self, round_idx: int) -> None:
        del round_idx

    def on_global_model_updated(self, total_clients: int) -> None:
        del total_clients

    @abstractmethod
    def compute_client_weights(
        self,
        client_sizes: List[int],
        class_counts: Optional[List[dict]] = None,
        num_classes: Optional[int] = None,
    ) -> List[float]:
        """Compute the aggregation weight for each client."""
        raise NotImplementedError

    @abstractmethod
    def aggregate(
        self,
        state_dicts: List[StateDict],
        client_sizes: List[int],
        class_counts: Optional[List[dict]] = None,
        num_classes: Optional[int] = None,
        client_weights: Optional[List[float]] = None,
    ) -> StateDict:
        """Aggregate client state dicts into a single global state."""
        raise NotImplementedError

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
        del round_idx
        use_step_seen = self.steps_per_round is not None and self.steps_per_round > 0
        if use_step_seen:
            if len(step_seen_sizes) != len(selected_client_sizes):
                raise ValueError("step_seen_sizes and selected_client_sizes must have the same length.")
            if len(step_seen_label_counts) != len(selected_client_counts):
                raise ValueError("step_seen_label_counts and selected_client_counts must have the same length.")
            if any(size <= 0 for size in step_seen_sizes):
                raise ValueError(
                    "Step-based weighting found a client with zero seen samples. Increase --steps-per-round."
                )
            agg_sizes = step_seen_sizes
            agg_counts: List[dict] = [dict(row) for row in step_seen_label_counts]
        else:
            agg_sizes = selected_client_sizes
            agg_counts = selected_client_counts
        weights = self.compute_client_weights(
            agg_sizes,
            class_counts=agg_counts,
            num_classes=num_classes,
        )
        return RoundAggregationPlan(
            client_sizes=agg_sizes,
            class_counts=agg_counts,
            client_weights=normalize_weights(weights),
            raw_client_weights=[float(w) for w in weights],
            phase_suffix="",
        )

    def train_local(
        self,
        *,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        local_epochs: int,
        steps_per_round: int | None,
        client_id: int,
        global_named_params: Dict[str, torch.Tensor],
        learning_rate: float,
        use_amp: bool,
        client_class_counts: Optional[Dict[int, int]] = None,
        num_classes: Optional[int] = None,
        collect_seen_examples: bool = False,
    ) -> LocalTrainResult:
        model.train()
        total_loss = 0.0
        total_steps = 0
        seen_samples = 0
        seen_label_counts: Dict[int, int] = {}
        seen_examples: List[Tuple[int, int]] = []
        amp_enabled = bool(use_amp and device.type == "cuda")
        scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
        named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        local_context = self._build_local_context(
            client_id=client_id,
            device=device,
            named_params=named_params,
            global_named_params=global_named_params,
        )
        if client_class_counts is not None:
            local_context.setdefault("client_class_counts", {int(k): int(v) for k, v in client_class_counts.items()})
        if num_classes is not None:
            local_context.setdefault("num_classes", int(num_classes))

        def run_step(images: torch.Tensor, labels: torch.Tensor, sample_ids: Optional[torch.Tensor]) -> None:
            nonlocal total_loss, total_steps, seen_samples
            labels_list = [int(label_idx) for label_idx in labels.tolist()]
            for cls in labels_list:
                seen_label_counts[cls] = seen_label_counts.get(cls, 0) + 1
            seen_samples += len(labels_list)

            if collect_seen_examples and sample_ids is not None:
                sample_ids_list = [int(sample_idx) for sample_idx in sample_ids.tolist()]
                if len(sample_ids_list) != len(labels_list):
                    raise ValueError("sample_ids and labels must have the same length.")
                seen_examples.extend(list(zip(sample_ids_list, labels_list)))

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(images)
                loss = self._compute_training_loss(
                    outputs=outputs,
                    labels=labels,
                    criterion=criterion,
                    named_params=named_params,
                    local_context=local_context,
                )
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                self._adjust_gradients(named_params=named_params, local_context=local_context)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                self._adjust_gradients(named_params=named_params, local_context=local_context)
                optimizer.step()
            total_loss += loss.item()
            total_steps += 1

        def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            if not isinstance(batch, (list, tuple)):
                raise ValueError("Each batch must be a tuple/list like (images, labels) or (images, labels, sample_ids).")
            if len(batch) == 2:
                images, labels = batch
                return images, labels, None
            if len(batch) >= 3:
                images, labels, sample_ids = batch[0], batch[1], batch[2]
                return images, labels, sample_ids
            raise ValueError("Empty batch tuple is not supported.")

        if steps_per_round is not None and steps_per_round > 0:
            data_iter = iter(loader)
            for _ in range(steps_per_round):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)
                images, labels, sample_ids = unpack_batch(batch)
                run_step(images, labels, sample_ids)
        else:
            for _ in range(local_epochs):
                for batch in loader:
                    images, labels, sample_ids = unpack_batch(batch)
                    run_step(images, labels, sample_ids)

        self._on_client_trained(
            client_id=client_id,
            client_model=model,
            local_steps=total_steps,
            global_named_params=global_named_params,
            learning_rate=learning_rate,
            local_context=local_context,
        )

        return LocalTrainResult(
            train_loss=total_loss / max(total_steps, 1),
            local_steps=total_steps,
            seen_samples=seen_samples,
            seen_label_counts=seen_label_counts,
            seen_examples=seen_examples if collect_seen_examples else None,
        )

    def _build_local_context(
        self,
        *,
        client_id: int,
        device: torch.device,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        global_named_params: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        del client_id, device, named_params, global_named_params
        return {}

    def _augment_loss(
        self,
        *,
        loss: torch.Tensor,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        local_context: Dict[str, Any],
    ) -> torch.Tensor:
        del named_params, local_context
        return loss

    def _compute_training_loss(
        self,
        *,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        local_context: Dict[str, Any],
    ) -> torch.Tensor:
        loss = criterion(outputs, labels)
        return self._augment_loss(loss=loss, named_params=named_params, local_context=local_context)

    def _adjust_gradients(
        self,
        *,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        local_context: Dict[str, Any],
    ) -> None:
        del named_params, local_context

    def _on_client_trained(
        self,
        *,
        client_id: int,
        client_model: nn.Module,
        local_steps: int,
        global_named_params: Dict[str, torch.Tensor],
        learning_rate: float,
        local_context: Dict[str, Any],
    ) -> None:
        del client_id, client_model, local_steps, global_named_params, learning_rate, local_context
