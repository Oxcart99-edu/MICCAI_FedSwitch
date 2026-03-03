"""
FedIIC aggregator: FedAvg server aggregation with DALA/IntraSCL/InterSCL local training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from .fedavg import FedAvgAggregator


class DALA(nn.Module):
    def __init__(
        self,
        cls_num_list: Sequence[int],
        cls_loss: torch.Tensor,
        difficulty: float,
        tau: float = 1.0,
        weight: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        cls_num = torch.as_tensor(list(cls_num_list), dtype=torch.float32)
        cls_num = torch.clamp(cls_num, min=0.0)
        total = torch.clamp(cls_num.sum(), min=1.0)
        cls_p = cls_num / total

        cls_loss_t = torch.as_tensor(cls_loss, dtype=torch.float32)
        cls_loss_t = torch.clamp(cls_loss_t, min=eps)
        t = cls_p / (torch.pow(cls_loss_t, difficulty) + eps)
        t = torch.clamp(t, min=eps)
        m_list = tau * torch.log(t)

        self.register_buffer("m_list", m_list.view(1, -1))
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_adj = logits + self.m_list.to(device=logits.device, dtype=logits.dtype)
        weight = None if self.weight is None else self.weight.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits_adj, target, weight=weight)


class IntraSCL(nn.Module):
    def __init__(self, cls_num_list: Sequence[int], temperature: float = 0.1) -> None:
        super().__init__()
        cls_num = torch.as_tensor(list(cls_num_list), dtype=torch.float32)
        cls_num = torch.clamp(cls_num, min=0.0)
        total = torch.clamp(cls_num.sum(), min=1.0)
        self.register_buffer("cls_prior", cls_num / total)
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.size(1) != 2:
            raise ValueError("IntraSCL expects features with shape [B, 2, D].")
        batch_size = int(features.shape[0])
        if batch_size <= 1:
            return features.new_tensor(0.0)

        device = features.device
        targets = targets.contiguous().view(-1, 1)
        targets_2v = targets.repeat(2, 1)

        mask = torch.eq(targets_2v, targets_2v.T).float().to(device)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(1, torch.arange(batch_size * 2, device=device).view(-1, 1), 0.0)
        mask = mask * logits_mask

        feats = torch.cat(torch.unbind(features, dim=1), dim=0)
        logits = feats.mm(feats.T)

        priors = self.cls_prior.to(device=device)
        temp = priors.gather(0, targets_2v.view(-1)).view(-1, 1)
        temp = temp.mm(temp.T)
        temp = torch.clamp(temp.sqrt(), min=0.07)
        logits = logits / temp

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(torch.clamp(exp_logits.sum(1, keepdim=True), min=1e-12))

        denom = torch.clamp(mask.sum(1), min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / denom
        loss = -mean_log_prob_pos
        return loss.view(2, batch_size).mean()


class InterSCL(nn.Module):
    def __init__(self, cls_num_list: Sequence[int], temperature: float = 0.1) -> None:
        super().__init__()
        self.n_classes = len(list(cls_num_list))
        self.temperature = float(temperature)

    def forward(self, centers: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.size(1) != 2:
            raise ValueError("InterSCL expects features with shape [B, 2, D].")
        batch_size = int(features.shape[0])
        if batch_size <= 0:
            return features.new_tensor(0.0)

        device = features.device
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(self.n_classes, device=device).view(-1, 1)
        targets_all = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)

        mask = torch.eq(targets_all[: 2 * batch_size], targets_all.T).float().to(device)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(1, torch.arange(batch_size * 2, device=device).view(-1, 1), 0.0)
        logits_mask[: 2 * batch_size, : 2 * batch_size] = 0.0
        mask = mask * logits_mask

        feats = torch.cat(torch.unbind(features, dim=1), dim=0)
        feats_all = torch.cat([feats, centers], dim=0)
        logits = feats_all[: 2 * batch_size].mm(feats_all.T) / self.temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(torch.clamp(exp_logits.sum(1, keepdim=True), min=1e-12))

        denom = torch.clamp(mask.sum(1), min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / denom
        loss = -mean_log_prob_pos
        return loss.view(2, batch_size).mean()


@dataclass
class FedIICLocalState:
    loss_class: torch.Tensor
    prototypes: torch.Tensor


class FedIICAggregator(FedAvgAggregator):
    name = "fediic"

    def __init__(self) -> None:
        self.k1 = 2.0
        self.k2 = 2.0
        self.difficulty = 0.25
        self.tau = 1.0
        self.projector_dim = 256
        self.prototype_opt_steps = 100
        self.prototype_opt_lr = 0.1
        self.prototype_every_round = True
        self._cached_prototypes: torch.Tensor | None = None
        self._round_state: FedIICLocalState | None = None
        self._num_classes: int | None = None
        self._client_eval_loaders: List[DataLoader] = []
        self._client_class_counts: List[Dict[int, int]] = []
        self._device: torch.device | None = None

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
        fediic_k1: float = 2.0,
        fediic_k2: float = 2.0,
        fediic_difficulty: float = 0.25,
        fediic_tau: float = 1.0,
        fediic_projector_dim: int = 256,
        fediic_prototype_opt_steps: int = 100,
        fediic_prototype_opt_lr: float = 0.1,
        fediic_prototype_every_round: bool = True,
        client_eval_loaders: List[DataLoader] | None = None,
        client_class_counts_all: List[Dict[int, int]] | None = None,
        num_classes: int | None = None,
        device: torch.device | None = None,
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
        del rounds, switch_round, switch_weight_source, fedprox_mu, learning_rate
        self.k1 = float(fediic_k1)
        self.k2 = float(fediic_k2)
        self.difficulty = float(fediic_difficulty)
        self.tau = float(fediic_tau)
        self.projector_dim = int(fediic_projector_dim)
        self.prototype_opt_steps = int(fediic_prototype_opt_steps)
        self.prototype_opt_lr = float(fediic_prototype_opt_lr)
        self.prototype_every_round = bool(fediic_prototype_every_round)
        self._client_eval_loaders = client_eval_loaders or []
        self._client_class_counts = client_class_counts_all or []
        self._num_classes = num_classes
        self._device = device
        if self.tau <= 0:
            raise ValueError("fediic-tau must be > 0.")
        if self.prototype_opt_steps < 0:
            raise ValueError("fediic-prototype-opt-steps must be >= 0.")
        if self.prototype_opt_lr <= 0 and self.prototype_opt_steps > 0:
            raise ValueError("fediic-prototype-opt-lr must be > 0.")
        if self._num_classes is None or self._device is None:
            raise ValueError("FedIIC requires num_classes and device during configuration.")

    def experiment_note(self) -> str:
        return (
            "[fediic] FedAvg aggregation with local DALA + IntraSCL + InterSCL training "
            f"(k1={self.k1:g}, k2={self.k2:g}, difficulty={self.difficulty:g}, tau={self.tau:g})."
        )

    def on_experiment_start(self, global_model: nn.Module, total_clients: int) -> None:
        del total_clients
        self._cached_prototypes = None
        self._prepare_round_state(global_model)

    def on_round_start(self, round_idx: int) -> None:
        del round_idx

    def on_global_model_updated(self, total_clients: int) -> None:
        del total_clients

    def prepare_round(self, global_model: nn.Module) -> None:
        self._prepare_round_state(global_model)

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
        client_class_counts: Dict[int, int] | None = None,
        num_classes: int | None = None,
        collect_seen_examples: bool = False,
    ):
        del criterion, global_named_params, learning_rate, use_amp, collect_seen_examples, num_classes
        if self._round_state is None:
            raise RuntimeError("FedIIC round state is not initialized.")
        if client_class_counts is None:
            raise ValueError("FedIIC requires client_class_counts.")

        class_num_list = [int(client_class_counts.get(class_idx, 0)) for class_idx in range(int(self._num_classes or 0))]
        local_result = _local_train_fediic(
            model=model,
            loader=loader,
            device=device,
            optimizer=optimizer,
            local_epochs=local_epochs,
            steps_per_round=steps_per_round,
            class_num_list=class_num_list,
            loss_class=self._round_state.loss_class,
            prototypes=self._round_state.prototypes,
            k1=self.k1,
            k2=self.k2,
            difficulty=self.difficulty,
            tau=self.tau,
            num_classes=int(self._num_classes or 0),
        )
        return local_result

    def _prepare_round_state(self, global_model: nn.Module) -> None:
        if self._device is None or self._num_classes is None:
            raise RuntimeError("FedIIC is not fully configured.")
        if self.prototype_every_round or self._cached_prototypes is None:
            self._cached_prototypes = _build_global_prototypes(
                global_model,
                device=self._device,
                steps=self.prototype_opt_steps,
                lr=self.prototype_opt_lr,
            )

        loss_matrix = torch.zeros((len(self._client_eval_loaders), self._num_classes), dtype=torch.float32)
        class_num_matrix = torch.zeros((len(self._client_class_counts), self._num_classes), dtype=torch.float32)
        for client_id, counts in enumerate(self._client_class_counts):
            for class_idx in range(self._num_classes):
                class_num_matrix[client_id, class_idx] = float(counts.get(class_idx, 0))
            loss_matrix[client_id] = _compute_loss_of_classes(
                global_model,
                self._client_eval_loaders[client_id],
                device=self._device,
                num_classes=self._num_classes,
            )
        global_class_num = torch.sum(class_num_matrix, dim=0)
        loss_matrix = loss_matrix / (1e-5 + global_class_num.unsqueeze(0))
        loss_class = torch.sum(loss_matrix, dim=0)
        self._round_state = FedIICLocalState(
            loss_class=loss_class,
            prototypes=self._cached_prototypes,
        )


@torch.no_grad()
def _compute_loss_of_classes(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    loss_class = torch.zeros(num_classes, dtype=torch.float32)

    for images, labels, *_ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        losses = criterion(logits, labels).detach().cpu()
        labels_cpu = labels.detach().cpu()
        for class_idx in range(num_classes):
            indices = torch.where(labels_cpu == class_idx)[0]
            if indices.numel() > 0:
                loss_class[class_idx] += losses[indices].sum()
    return loss_class


def _orthogonalize_prototypes(feature_avg: torch.Tensor, steps: int, lr: float) -> torch.Tensor:
    if steps <= 0:
        return feature_avg
    feat = feature_avg.detach().clone().requires_grad_(True)
    optimizer_f = torch.optim.SGD([feat], lr=lr)
    num_classes = int(feat.shape[0])
    mask = torch.ones((num_classes, num_classes), device=feat.device) - torch.eye(num_classes, device=feat.device)

    for _ in range(int(steps)):
        feat_n = F.normalize(feat, dim=1)
        cos_sim = torch.matmul(feat_n, feat_n.T)
        objective = (cos_sim * mask).max(dim=1)[0].sum()
        optimizer_f.zero_grad(set_to_none=True)
        objective.backward()
        optimizer_f.step()
    return feat.detach()


def _build_global_prototypes(model: nn.Module, device: torch.device, steps: int, lr: float) -> torch.Tensor:
    with torch.no_grad():
        class_embedding = model.class_embedding().detach().clone().to(device)
        feature_avg = model.projector(class_embedding).detach().clone()
    feature_avg = _orthogonalize_prototypes(feature_avg, steps=steps, lr=lr)
    return F.normalize(feature_avg, dim=1).detach()


def _local_train_fediic(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    local_epochs: int,
    steps_per_round: int | None,
    class_num_list: Sequence[int],
    loss_class: torch.Tensor,
    prototypes: torch.Tensor,
    k1: float,
    k2: float,
    difficulty: float,
    tau: float,
    num_classes: int,
):
    from .base import LocalTrainResult

    model.train()
    dala_criterion = DALA(cls_num_list=class_num_list, cls_loss=loss_class, difficulty=difficulty, tau=tau)
    intra_criterion = IntraSCL(cls_num_list=class_num_list)
    inter_criterion = InterSCL(cls_num_list=class_num_list)
    prototypes = F.normalize(prototypes, dim=1).detach().to(device)

    total_loss = 0.0
    total_steps = 0
    seen_samples = 0
    seen_label_counts: Dict[int, int] = {}

    def run_batch(images: torch.Tensor, labels: torch.Tensor) -> None:
        nonlocal total_loss, total_steps, seen_samples
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images_view_a = images
        images_view_b = torch.flip(images, dims=[4])
        inputs = torch.cat([images_view_a, images_view_b], dim=0)

        proj_features, logits_all = model(inputs, project=True)
        split_size = int(labels.shape[0])
        feat_a, feat_b = torch.split(proj_features, [split_size, split_size], dim=0)
        feature_pairs = torch.stack([feat_a, feat_b], dim=1)
        logits, _ = torch.split(logits_all, [split_size, split_size], dim=0)

        loss_ce = dala_criterion(logits, labels)
        loss_intra = intra_criterion(feature_pairs, labels)
        loss_inter = inter_criterion(prototypes, feature_pairs, labels)
        loss = loss_ce + float(k1) * loss_intra + float(k2) * loss_inter

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1
        labels_list = [int(label_idx) for label_idx in labels.detach().cpu().tolist()]
        seen_samples += len(labels_list)
        for class_idx in labels_list:
            seen_label_counts[class_idx] = seen_label_counts.get(class_idx, 0) + 1

    if steps_per_round is not None and steps_per_round > 0:
        data_iter = iter(loader)
        for _ in range(int(steps_per_round)):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            images, labels, *_ = batch
            run_batch(images, labels)
    else:
        for _ in range(int(local_epochs)):
            for batch in loader:
                images, labels, *_ = batch
                run_batch(images, labels)

    return LocalTrainResult(
        train_loss=total_loss / max(total_steps, 1),
        local_steps=total_steps,
        seen_samples=seen_samples,
        seen_label_counts=seen_label_counts,
    )
