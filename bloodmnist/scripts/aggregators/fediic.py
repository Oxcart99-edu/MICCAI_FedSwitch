"""FedIIC helpers for the BloodMNIST federated trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from .base import normalize_positive_weights
from scripts.models import ResNetCIFAR


def compute_fediic_weights(client_sizes: list[int]) -> list[float]:
    return normalize_positive_weights(client_sizes)


class FedIICCNN(ResNetCIFAR):
    """ResNet-18 CIFAR backbone with projector head and FedIIC-compatible forward path."""

    def __init__(self, num_classes: int, projector_dim: int = 256):
        super().__init__(in_channels=3, num_classes=num_classes, layers=(2, 2, 2, 2))
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projector_dim),
        )

    def forward(self, x: torch.Tensor, project: bool = False):
        feats = self.forward_backbone(x)
        logits = self.fc(feats)
        if project:
            proj = self.projector(feats)
            return proj, logits
        return logits

    def class_embedding(self) -> torch.Tensor:
        return self.fc.weight


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
class FedIICTrainResult:
    state_dict: dict[str, torch.Tensor]
    loss_total: float
    loss_ce: float
    loss_intra: float
    loss_inter: float
    local_steps: int
    seen_samples: int
    seen_label_counts: np.ndarray


@torch.no_grad()
def compute_class_loss_vector(
    model: FedIICCNN,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    loss_class = torch.zeros(num_classes, dtype=torch.float32)

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        losses = criterion(logits, labels).detach().cpu()
        label_cpu = labels.detach().cpu()
        for class_idx in range(num_classes):
            indices = torch.where(label_cpu == class_idx)[0]
            if indices.numel() > 0:
                loss_class[class_idx] += losses[indices].sum()
    return loss_class


def orthogonalize_prototypes(feature_avg: torch.Tensor, steps: int, lr: float) -> torch.Tensor:
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


def build_global_prototypes(model: FedIICCNN, device: torch.device, steps: int, lr: float) -> torch.Tensor:
    with torch.no_grad():
        class_embedding = model.class_embedding().detach().clone().to(device)
        feature_avg = model.projector(class_embedding).detach().clone()
    feature_avg = orthogonalize_prototypes(feature_avg, steps=steps, lr=lr)
    return F.normalize(feature_avg, dim=1).detach()


def local_train_fediic(
    *,
    model: FedIICCNN,
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
    augment: Callable[[torch.Tensor], torch.Tensor] | None,
    num_classes: int,
) -> FedIICTrainResult:
    model.train()
    dala_criterion = DALA(cls_num_list=class_num_list, cls_loss=loss_class, difficulty=difficulty, tau=tau)
    intra_criterion = IntraSCL(cls_num_list=class_num_list)
    inter_criterion = InterSCL(cls_num_list=class_num_list)
    prototypes = F.normalize(prototypes, dim=1).detach().to(device)

    total_loss = 0.0
    total_ce = 0.0
    total_intra = 0.0
    total_inter = 0.0
    total_steps = 0
    seen_samples = 0
    seen_counts = np.zeros(num_classes, dtype=np.int64)

    def run_batch(images: torch.Tensor, labels: torch.Tensor) -> None:
        nonlocal total_loss, total_ce, total_intra, total_inter, total_steps, seen_samples, seen_counts
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if augment is not None:
            view_a = augment(images.clone())
            view_b = augment(images.clone())
        else:
            view_a = images
            view_b = images.clone()

        inputs = torch.cat([view_a, view_b], dim=0)
        proj_features, logits_all = model(inputs, project=True)
        proj_features = F.normalize(proj_features, dim=1)
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

        total_steps += 1
        total_loss += float(loss.item())
        total_ce += float(loss_ce.item())
        total_intra += float(loss_intra.item())
        total_inter += float(loss_inter.item())

        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        seen_samples += int(labels_np.shape[0])
        seen_counts += np.bincount(labels_np, minlength=num_classes)

    if steps_per_round is not None and steps_per_round > 0:
        data_iter = iter(loader)
        for _ in range(int(steps_per_round)):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                images, labels = next(data_iter)
            run_batch(images, labels)
    else:
        for _ in range(int(local_epochs)):
            for images, labels in loader:
                run_batch(images, labels)

    if total_steps == 0:
        return FedIICTrainResult(
            state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
            loss_total=0.0,
            loss_ce=0.0,
            loss_intra=0.0,
            loss_inter=0.0,
            local_steps=0,
            seen_samples=0,
            seen_label_counts=seen_counts,
        )

    return FedIICTrainResult(
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        loss_total=total_loss / total_steps,
        loss_ce=total_ce / total_steps,
        loss_intra=total_intra / total_steps,
        loss_inter=total_inter / total_steps,
        local_steps=total_steps,
        seen_samples=seen_samples,
        seen_label_counts=seen_counts,
    )
