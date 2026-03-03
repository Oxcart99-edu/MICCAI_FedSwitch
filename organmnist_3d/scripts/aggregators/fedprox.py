"""
FedProx aggregator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from .fedavg import FedAvgAggregator


class FedProxAggregator(FedAvgAggregator):
    name = "fedprox"

    def __init__(self) -> None:
        self.fedprox_mu = 0.0

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
        super().configure(
            rounds=rounds,
            steps_per_round=steps_per_round,
            switch_round=switch_round,
            switch_weight_source=switch_weight_source,
            fedprox_mu=fedprox_mu,
            learning_rate=learning_rate,
            fedlc_tau=fedlc_tau,
        )
        del learning_rate
        self.fedprox_mu = float(fedprox_mu)
        if self.fedprox_mu < 0:
            raise ValueError("fedprox-mu must be >= 0")

    def _build_local_context(
        self,
        *,
        client_id: int,
        device: torch.device,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        global_named_params: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        del client_id
        if self.fedprox_mu <= 0:
            return {}
        return {
            "prox_ref": {name: global_named_params[name].detach().to(device) for name, _ in named_params},
        }

    def _augment_loss(
        self,
        *,
        loss: torch.Tensor,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        local_context: Dict[str, Any],
    ) -> torch.Tensor:
        if self.fedprox_mu <= 0:
            return loss
        prox_ref = local_context.get("prox_ref")
        if prox_ref is None:
            raise ValueError("FedProx context missing prox_ref.")

        prox_term = 0.0
        for name, param in named_params:
            prox_term = prox_term + torch.sum((param - prox_ref[name]) ** 2)
        return loss + 0.5 * self.fedprox_mu * prox_term
