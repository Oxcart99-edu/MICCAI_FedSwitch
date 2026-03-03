"""
FedLC aggregator: FedAvg aggregation + local logits calibration for label skew.

Paper: Federated Learning with Label Distribution Skew via Logits Calibration
https://arxiv.org/abs/2209.00189
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from .fedavg import FedAvgAggregator


class FedLCAggregator(FedAvgAggregator):
    name = "fedlc"

    def __init__(self) -> None:
        super().__init__()
        self.fedlc_tau = 0.5

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
        fedlc_tau: float = 0.5,
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
        self.fedlc_tau = float(fedlc_tau)
        if self.fedlc_tau < 0:
            raise ValueError("fedlc-tau must be >= 0.")

    def experiment_note(self) -> str:
        return (
            f"[fedlc] FedAvg aggregation + local logits calibration "
            f"(z_c <- z_c - tau * n_c^(-1/4), tau={self.fedlc_tau:g})."
        )

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
            raise ValueError("FedLC requires client_class_counts and num_classes in local training context.")

        calibration = local_context.get("fedlc_calibration")
        if calibration is None:
            counts = torch.zeros(int(num_classes), dtype=torch.float32, device=outputs.device)
            for cls_idx, count in client_class_counts.items():
                try:
                    class_idx = int(cls_idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= class_idx < int(num_classes):
                    counts[class_idx] = float(count)
            # Clamp to 1 to avoid inf for absent classes while still applying a strong downshift.
            counts = counts.clamp_min(1.0)
            calibration = self.fedlc_tau * torch.pow(counts, -0.25)
            local_context["fedlc_calibration"] = calibration
        else:
            calibration = calibration.to(device=outputs.device, dtype=torch.float32)

        calibrated_logits = outputs - calibration.to(dtype=outputs.dtype).unsqueeze(0)
        loss = criterion(calibrated_logits, labels)
        return self._augment_loss(loss=loss, named_params=named_params, local_context=local_context)
