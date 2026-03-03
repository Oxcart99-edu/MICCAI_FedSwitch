"""
FedIIC-style aggregator.

This implementation keeps FedAvg server aggregation (as in the paper/repo) and
adds the DALA local classification loss from FedIIC.

Reference:
- FedIIC: https://arxiv.org/abs/2206.13803
- Repo: https://github.com/wnn2000/FedIIC
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .fedavg import FedAvgAggregator


class FedIICAggregator(FedAvgAggregator):
    name = "fediic"

    def __init__(self) -> None:
        super().__init__()
        self.fediic_d = 0.25
        self.fediic_tau = 1.0
        self.fediic_loss_scope = "all"
        self._round_class_loss: torch.Tensor | None = None

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
        fediic_d: float = 0.25,
        fediic_tau: float = 1.0,
        fediic_loss_scope: str = "all",
    ) -> None:
        super().configure(
            rounds=rounds,
            steps_per_round=steps_per_round,
            switch_round=switch_round,
            switch_weight_source=switch_weight_source,
            dirichlet_alpha=dirichlet_alpha,
            fedprox_mu=fedprox_mu,
            learning_rate=learning_rate,
        )
        self.fediic_d = float(fediic_d)
        self.fediic_tau = float(fediic_tau)
        self.fediic_loss_scope = str(fediic_loss_scope).lower()
        if self.fediic_d < 0:
            raise ValueError("fediic-d must be >= 0.")
        if self.fediic_tau < 0:
            raise ValueError("fediic-tau must be >= 0.")
        if self.fediic_loss_scope not in {"all", "selected"}:
            raise ValueError("fediic-loss-scope must be 'all' or 'selected'.")

    def experiment_note(self) -> str:
        return (
            "[fediic] FedAvg aggregation + local DALA loss (FedIIC-compatible; "
            "contrastive/prototype terms not included in this pipeline) "
            f"with d={self.fediic_d:g}, tau={self.fediic_tau:g}, "
            f"class-loss-scope={self.fediic_loss_scope}."
        )

    @torch.no_grad()
    def prepare_round_training(
        self,
        *,
        round_idx: int,
        global_model: nn.Module,
        client_loaders: Sequence[Tuple[DataLoader, int, dict]],
        selected_client_ids: Sequence[int],
        device: torch.device,
        num_classes: int,
    ) -> None:
        del round_idx
        target_client_ids = list(range(len(client_loaders))) if self.fediic_loss_scope == "all" else list(selected_client_ids)
        if not target_client_ids:
            raise ValueError("FedIIC requires at least one client to estimate per-class loss.")

        was_training = global_model.training
        global_model.eval()
        try:
            class_loss_sum = torch.zeros(num_classes, dtype=torch.float64)
            class_count_sum = torch.zeros(num_classes, dtype=torch.float64)

            for client_id in target_client_ids:
                loader = client_loaders[client_id][0]
                for images, labels in loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    logits = global_model(images)
                    losses = F.cross_entropy(logits, labels, reduction="none")

                    labels_cpu = labels.detach().to(device="cpu", dtype=torch.long)
                    losses_cpu = losses.detach().to(device="cpu", dtype=torch.float64)
                    ones = torch.ones_like(losses_cpu, dtype=torch.float64)
                    class_loss_sum.scatter_add_(0, labels_cpu, losses_cpu)
                    class_count_sum.scatter_add_(0, labels_cpu, ones)

            observed = class_count_sum > 0
            if not bool(observed.any()):
                raise ValueError("FedIIC could not estimate class losses: no samples observed.")

            avg_class_loss = torch.zeros(num_classes, dtype=torch.float32)
            avg_class_loss[observed] = (class_loss_sum[observed] / class_count_sum[observed]).to(torch.float32)

            # With partial participation, some classes may be absent in a round. Use the mean observed loss as fallback.
            if not bool(observed.all()):
                fallback = float(avg_class_loss[observed].mean().item())
                avg_class_loss[~observed] = fallback

            self._round_class_loss = avg_class_loss
        finally:
            if was_training:
                global_model.train()

    def _compute_training_loss(
        self,
        *,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        local_context: Dict[str, Any],
    ) -> torch.Tensor:
        num_classes = local_context.get("num_classes")
        client_class_counts = local_context.get("client_class_counts")
        if num_classes is None or client_class_counts is None:
            raise ValueError("FedIIC requires client_class_counts and num_classes in local training context.")
        if self._round_class_loss is None:
            raise RuntimeError("FedIIC round class loss is not initialized. Call prepare_round_training() first.")

        margin = local_context.get("fediic_margin")
        if margin is None:
            counts = torch.zeros(int(num_classes), dtype=torch.float32, device=outputs.device)
            for cls_idx, count in client_class_counts.items():
                try:
                    class_idx = int(cls_idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= class_idx < int(num_classes):
                    counts[class_idx] = float(count)
            total = float(counts.sum().item())
            if total <= 0:
                return self._augment_loss(loss=criterion(outputs, labels), named_params=named_params, local_context=local_context)

            cls_p = counts / total
            cls_loss = self._round_class_loss.to(device=outputs.device, dtype=torch.float32)
            eps = 1e-5
            t = cls_p / (torch.pow(cls_loss.clamp_min(eps), self.fediic_d) + eps)
            margin = self.fediic_tau * torch.log(t.clamp_min(eps))
            local_context["fediic_margin"] = margin
        else:
            margin = margin.to(device=outputs.device, dtype=torch.float32)

        calibrated_logits = outputs + margin.to(dtype=outputs.dtype).unsqueeze(0)
        loss = criterion(calibrated_logits, labels)
        return self._augment_loss(loss=loss, named_params=named_params, local_context=local_context)
