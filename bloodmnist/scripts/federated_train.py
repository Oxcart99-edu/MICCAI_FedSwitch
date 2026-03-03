#!/usr/bin/env python3
"""Federated trainer for BloodMNIST."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: numpy. Install the local requirements with "
        "`python -m pip install -r requirements.txt` from the `bloodmnist/` directory."
    ) from exc

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: torch. Install the local requirements with "
        "`python -m pip install -r requirements.txt` from the `bloodmnist/` directory."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models import ResNetCIFAR  # noqa: E402
from scripts.aggregators import (  # noqa: E402
    FedIICCNN,
    SUPPORTED_AGGREGATORS,
    aggregate_state_dicts,
    build_global_prototypes,
    compute_class_loss_vector,
    compute_fedavg_weights,
    compute_fediic_weights,
    compute_fedswitch_weights,
    compute_fedprox_weights,
    local_train_fediic,
)

CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature_granulocytes",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float
    macro_f1: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BloodMNISTDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels.reshape(-1).astype(np.int64)

    @classmethod
    def from_npz(cls, npz_path: Path, split: str) -> "BloodMNISTDataset":
        archive = np.load(npz_path)
        return cls(archive[f"{split}_images"], archive[f"{split}_labels"])

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).permute(2, 0, 1).float().div(255.0)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


class TrainAugment:
    def __init__(self, crop_padding: int = 4, hflip_p: float = 0.5):
        self.crop_padding = int(crop_padding)
        self.hflip_p = float(hflip_p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.hflip_p > 0:
            flip_mask = torch.rand(x.size(0), device=x.device) < self.hflip_p
            if flip_mask.any():
                x = x.clone()
                x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

        if self.crop_padding > 0:
            padding = self.crop_padding
            x = torch.nn.functional.pad(x, (padding, padding, padding, padding), mode="reflect")
            out_h = x.shape[-2] - 2 * padding
            out_w = x.shape[-1] - 2 * padding
            top = torch.randint(0, 2 * padding + 1, (x.size(0),), device=x.device)
            left = torch.randint(0, 2 * padding + 1, (x.size(0),), device=x.device)
            crops = []
            for idx in range(x.size(0)):
                crops.append(x[idx : idx + 1, :, top[idx] : top[idx] + out_h, left[idx] : left[idx] + out_w])
            x = torch.cat(crops, dim=0)
        return x


def compute_macro_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    scores = []
    for class_idx in range(num_classes):
        tp = float(np.sum((preds == class_idx) & (targets == class_idx)))
        fp = float(np.sum((preds == class_idx) & (targets != class_idx)))
        fn = float(np.sum((preds != class_idx) & (targets == class_idx)))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        scores.append(2.0 * precision * recall / (precision + recall + 1e-12))
    return float(np.mean(scores))


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * images.size(0)
            total_examples += images.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return Metrics(
        loss=total_loss / max(1, total_examples),
        accuracy=float((preds == targets).mean()),
        macro_f1=compute_macro_f1(preds, targets, len(CLASS_NAMES)),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    augment: TrainAugment | None,
    prox_mu: float = 0.0,
    global_named_params: dict[str, torch.Tensor] | None = None,
    steps_limit: int | None = None,
) -> tuple[Metrics, int, np.ndarray]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    seen_label_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    local_steps = 0
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    def run_batch(images: torch.Tensor, labels: torch.Tensor) -> None:
        nonlocal total_loss, total_examples, local_steps
        images = images.to(device)
        labels = labels.to(device)
        if augment is not None:
            images = augment(images)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        if prox_mu > 0.0:
            if global_named_params is None:
                raise ValueError("global_named_params is required when prox_mu > 0.")
            prox_term = 0.0
            for name, param in named_params:
                prox_term = prox_term + torch.sum((param - global_named_params[name]) ** 2)
            loss = loss + 0.5 * prox_mu * prox_term
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        local_steps += 1

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets)
        seen_label_counts[:] += np.bincount(targets, minlength=len(CLASS_NAMES))
        total_loss += float(loss.item()) * images.size(0)
        total_examples += images.size(0)

    if steps_limit is None:
        for batch_images, batch_labels in loader:
            run_batch(batch_images, batch_labels)
    else:
        if steps_limit <= 0:
            raise ValueError("--steps-per-round must be > 0 when provided.")
        iterator = iter(loader)
        while local_steps < steps_limit:
            try:
                batch_images, batch_labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch_images, batch_labels = next(iterator)
            run_batch(batch_images, batch_labels)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return (
        Metrics(
            loss=total_loss / max(1, total_examples),
            accuracy=float((preds == targets).mean()),
            macro_f1=compute_macro_f1(preds, targets, len(CLASS_NAMES)),
        ),
        int(seen_label_counts.sum()),
        seen_label_counts,
    )


def make_loss(labels: np.ndarray, device: torch.device, use_class_weights: bool) -> nn.Module:
    if not use_class_weights:
        return nn.CrossEntropyLoss()
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))


def sample_distribution_partition(
    labels: np.ndarray,
    num_clients: int,
    seed: int,
    clients_label_csv: Path,
) -> list[np.ndarray]:
    labels_np = np.asarray(labels).astype(np.int64).flatten()
    num_classes = int(labels_np.max()) + 1
    rng = np.random.default_rng(seed)

    class_pools: list[list[int]] = []
    for class_id in range(num_classes):
        indices = np.where(labels_np == class_id)[0].tolist()
        rng.shuffle(indices)
        class_pools.append(indices)

    client_bins: list[list[int]] = [[] for _ in range(num_clients)]
    with clients_label_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            client_id = int(row["client_id"])
            if client_id < 0 or client_id >= num_clients:
                raise ValueError(f"CSV contains client_id={client_id}, outside [0, {num_clients - 1}].")
            for class_id in range(num_classes):
                key = f"count_{class_id}"
                count = int(float(row.get(key, "0") or 0))
                if count > len(class_pools[class_id]):
                    raise ValueError(
                        f"Requested {count} samples of class {class_id} for client {client_id}, "
                        f"but only {len(class_pools[class_id])} remain."
                    )
                if count <= 0:
                    continue
                taken = class_pools[class_id][:count]
                del class_pools[class_id][:count]
                client_bins[client_id].extend(taken)

    leftovers: list[int] = []
    for pool in class_pools:
        leftovers.extend(pool)
    rng.shuffle(leftovers)
    for idx in leftovers:
        target_client = min(range(num_clients), key=lambda client_id: len(client_bins[client_id]))
        client_bins[target_client].append(idx)

    splits: list[np.ndarray] = []
    for client_id in range(num_clients):
        rng.shuffle(client_bins[client_id])
        splits.append(np.asarray(client_bins[client_id], dtype=np.int64))
    return splits


def client_split_counts(splits: list[np.ndarray], labels: np.ndarray, num_classes: int) -> list[np.ndarray]:
    labels_np = np.asarray(labels).astype(np.int64).flatten()
    out: list[np.ndarray] = []
    for indices in splits:
        if len(indices) == 0:
            out.append(np.zeros(num_classes, dtype=np.int64))
        else:
            out.append(np.bincount(labels_np[np.asarray(indices, dtype=np.int64)], minlength=num_classes))
    return out


def load_picked_ids_schedule(path: Path, total_clients: int) -> dict[int, list[int]]:
    payload = json.loads(path.read_text())
    schedule: dict[int, list[int]] = {}
    for item in payload:
        round_idx = int(item["round"])
        picked_ids = []
        seen = set()
        for raw_id in item["picked_ids"]:
            client_id = int(raw_id)
            if client_id < 0 or client_id >= total_clients:
                raise ValueError(f"Client id {client_id} out of range [0, {total_clients - 1}].")
            if client_id in seen:
                continue
            seen.add(client_id)
            picked_ids.append(client_id)
        if not picked_ids:
            raise ValueError(f"Round {round_idx} has no picked clients.")
        schedule[round_idx] = picked_ids
    return schedule


def default_data_path() -> Path:
    return PROJECT_ROOT / "dataset" / "bloodmnist.npz"


def default_splits_dir() -> Path:
    return PROJECT_ROOT / "client_selection"


def parse_args() -> argparse.Namespace:
    assets_dir = default_splits_dir()
    parser = argparse.ArgumentParser(description="Standalone federated training for BloodMNIST.")
    parser.add_argument("--data-path", type=Path, default=default_data_path())
    parser.add_argument(
        "--clients-label-csv",
        type=Path,
        default=assets_dir / "clients_label_alpha_0.5.csv",
        help="CSV with per-client class counts.",
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round.")
    parser.add_argument("--steps-per-round", type=int, default=None)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--aggregator", choices=SUPPORTED_AGGREGATORS, default="fedavg")
    parser.add_argument("--switch-round", type=int, default=10)
    parser.add_argument("--fediic-k1", type=float, default=2.0)
    parser.add_argument("--fediic-k2", type=float, default=2.0)
    parser.add_argument("--fediic-difficulty", type=float, default=0.25)
    parser.add_argument("--fediic-tau", type=float, default=0.5)
    parser.add_argument("--fediic-prototype-opt-steps", type=int, default=100)
    parser.add_argument("--fediic-prototype-opt-lr", type=float, default=0.1)
    parser.add_argument("--fediic-projector-dim", type=int, default=256)
    parser.add_argument("--fediic-prototype-every-round", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--picked-ids-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--verbose-round-details", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "federated")
    return parser.parse_args()


def compute_client_weights(
    *,
    aggregator_name: str,
    round_idx: int,
    client_sizes: list[int],
    step_seen_sizes: list[int],
    step_seen_counts: list[np.ndarray],
    switch_round: int,
    switch_target_estimate: dict[str, float] | None,
) -> tuple[list[float], str | None]:
    if aggregator_name == "fedswitch":
        if switch_target_estimate is None:
            raise RuntimeError("fedswitch estimate not initialized.")
        method = "fedavg_steps" if round_idx < switch_round else "estimate_stratified_steps"
        return (
            compute_fedswitch_weights(
                round_idx=round_idx,
                switch_round=switch_round,
                step_sizes=step_seen_sizes,
                step_counts=step_seen_counts,
                target_estimate=switch_target_estimate,
                num_classes=len(CLASS_NAMES),
            ),
            method,
        )
    if aggregator_name == "fedprox":
        return compute_fedprox_weights(client_sizes), None
    if aggregator_name == "fediic":
        return compute_fediic_weights(client_sizes), None
    return compute_fedavg_weights(client_sizes), None


def main() -> int:
    args = parse_args()
    if not args.data_path.exists():
        raise SystemExit(f"Dataset not found: {args.data_path}")
    if not args.clients_label_csv.exists():
        raise SystemExit(f"Client split CSV not found: {args.clients_label_csv}")
    if args.rounds <= 0:
        raise SystemExit("--rounds must be > 0")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.clients <= 0 or args.clients_per_round <= 0:
        raise SystemExit("--clients and --clients-per-round must be > 0")
    if args.clients_per_round > args.clients:
        raise SystemExit("--clients-per-round cannot exceed --clients")
    if args.eval_every <= 0:
        raise SystemExit("--eval-every must be > 0")
    if args.aggregator == "fedswitch" and (args.steps_per_round is None or args.steps_per_round <= 0):
        raise SystemExit("--steps-per-round must be > 0 for fedswitch")
    if args.switch_round < 1 or args.switch_round > args.rounds:
        raise SystemExit("--switch-round must be in [1, --rounds]")
    if args.fediic_tau <= 0:
        raise SystemExit("--fediic-tau must be > 0")
    if args.fediic_prototype_opt_steps < 0:
        raise SystemExit("--fediic-prototype-opt-steps must be >= 0")
    if args.fediic_prototype_opt_lr <= 0 and args.fediic_prototype_opt_steps > 0:
        raise SystemExit("--fediic-prototype-opt-lr must be > 0")

    seed_everything(args.seed)
    device = torch.device(args.device)

    train_ds = BloodMNISTDataset.from_npz(args.data_path, "train")
    val_ds = BloodMNISTDataset.from_npz(args.data_path, "val")
    test_ds = BloodMNISTDataset.from_npz(args.data_path, "test")

    train_pool_images = np.concatenate([train_ds.images, val_ds.images], axis=0)
    train_pool_labels = np.concatenate([train_ds.labels, val_ds.labels], axis=0)
    train_pool_ds = BloodMNISTDataset(train_pool_images, train_pool_labels)

    loss_fn = make_loss(train_pool_labels, device=device, use_class_weights=not args.no_class_weights)
    if args.aggregator == "fediic":
        model: nn.Module = FedIICCNN(
            num_classes=len(CLASS_NAMES),
            projector_dim=args.fediic_projector_dim,
        ).to(device)
    else:
        model = ResNetCIFAR(in_channels=3, num_classes=len(CLASS_NAMES), layers=(2, 2, 2, 2)).to(device)

    client_indices = sample_distribution_partition(
        labels=train_pool_labels,
        num_clients=args.clients,
        seed=args.seed,
        clients_label_csv=args.clients_label_csv,
    )
    available_client_ids = [client_id for client_id, indices in enumerate(client_indices) if len(indices) > 0]
    if args.clients_per_round > len(available_client_ids):
        raise SystemExit(
            f"--clients-per-round={args.clients_per_round} exceeds available non-empty clients ({len(available_client_ids)})"
        )
    client_class_counts = client_split_counts(client_indices, train_pool_labels, num_classes=len(CLASS_NAMES))

    train_loader_all = DataLoader(
        train_pool_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    client_loaders: dict[int, DataLoader] = {}
    client_eval_loaders: dict[int, DataLoader] = {}
    for client_id in available_client_ids:
        subset = Subset(train_pool_ds, indices=client_indices[client_id].tolist())
        client_loaders[client_id] = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        client_eval_loaders[client_id] = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    picked_schedule: dict[int, list[int]] = {}
    if args.picked_ids_json is not None:
        if not args.picked_ids_json.exists():
            raise SystemExit(f"Picked-ids JSON not found: {args.picked_ids_json}")
        picked_schedule = load_picked_ids_schedule(args.picked_ids_json, total_clients=args.clients)

    selection_rng = random.Random(args.seed)
    train_augment = None if args.no_augment else TrainAugment()
    switch_target_estimate: dict[str, float] | None = None
    if args.aggregator == "fedswitch":
        alpha_token = args.clients_label_csv.stem.replace("clients_label_alpha_", "")
        estimate_path = default_splits_dir() / f"bloodmnist_dirichlet_alpha_{alpha_token}_results.json"
        if not estimate_path.exists():
            raise SystemExit(f"Missing estimate JSON for fedswitch: {estimate_path}")
        payload = json.loads(estimate_path.read_text())
        switch_target_estimate = payload[0]["sample_estimated_histograms"][0]
    cached_fediic_prototypes: torch.Tensor | None = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.pt"
    metrics_path = args.output_dir / "metrics.json"
    best_test_f1 = -1.0
    history: list[dict[str, float | int | list[int] | str]] = []

    print(f"Dataset: {args.data_path}")
    print(f"Train-pool/test: {len(train_pool_ds)}/{len(test_ds)}")
    print(f"Aggregator: {args.aggregator} | rounds={args.rounds} | clients={args.clients} | clients_per_round={args.clients_per_round}")
    print(f"Client CSV: {args.clients_label_csv}")

    for round_idx in range(1, args.rounds + 1):
        if round_idx in picked_schedule:
            selected_client_ids = list(picked_schedule[round_idx])
        else:
            selected_client_ids = selection_rng.sample(available_client_ids, k=args.clients_per_round)

        if args.verbose_round_details:
            print(f"[round {round_idx:03d}] selected_clients={selected_client_ids}")

        client_states = []
        client_sizes: list[int] = []
        step_seen_sizes: list[int] = []
        step_seen_counts: list[np.ndarray] = []
        round_client_losses: list[float] = []
        round_client_ce_losses: list[float] = []
        round_client_intra_losses: list[float] = []
        round_client_inter_losses: list[float] = []
        global_named_params = {
            name: param.detach().clone().to(device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        fediic_loss_class: torch.Tensor | None = None
        fediic_prototypes: torch.Tensor | None = None
        if args.aggregator == "fediic":
            if not isinstance(model, FedIICCNN):
                raise RuntimeError("fediic requires a FedIICCNN model.")
            if args.fediic_prototype_every_round or cached_fediic_prototypes is None:
                cached_fediic_prototypes = build_global_prototypes(
                    model,
                    device=device,
                    steps=args.fediic_prototype_opt_steps,
                    lr=args.fediic_prototype_opt_lr,
                )
            fediic_prototypes = cached_fediic_prototypes
            loss_matrix = torch.zeros((args.clients, len(CLASS_NAMES)), dtype=torch.float32)
            class_num_matrix = torch.zeros((args.clients, len(CLASS_NAMES)), dtype=torch.float32)
            for client_id in available_client_ids:
                class_num_matrix[client_id] = torch.tensor(client_class_counts[client_id], dtype=torch.float32)
                loss_matrix[client_id] = compute_class_loss_vector(
                    model,
                    client_eval_loaders[client_id],
                    device=device,
                    num_classes=len(CLASS_NAMES),
                )
            global_class_num = torch.sum(class_num_matrix, dim=0)
            loss_matrix = loss_matrix / (1e-5 + global_class_num.unsqueeze(0))
            fediic_loss_class = torch.sum(loss_matrix, dim=0)

        for client_id in selected_client_ids:
            client_model = copy.deepcopy(model).to(device)
            optimizer = torch.optim.Adam(client_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            loader = client_loaders[client_id]

            seen_samples_total = 0
            seen_counts_total = np.zeros(len(CLASS_NAMES), dtype=np.int64)
            if args.aggregator == "fediic":
                if fediic_loss_class is None or fediic_prototypes is None or not isinstance(client_model, FedIICCNN):
                    raise RuntimeError("fediic round state not initialized.")
                result = local_train_fediic(
                    model=client_model,
                    loader=loader,
                    device=device,
                    optimizer=optimizer,
                    local_epochs=args.epochs,
                    steps_per_round=args.steps_per_round,
                    class_num_list=client_class_counts[client_id].astype(int).tolist(),
                    loss_class=fediic_loss_class,
                    prototypes=fediic_prototypes,
                    k1=args.fediic_k1,
                    k2=args.fediic_k2,
                    difficulty=args.fediic_difficulty,
                    tau=args.fediic_tau,
                    augment=train_augment,
                    num_classes=len(CLASS_NAMES),
                )
                seen_samples_total = result.seen_samples
                seen_counts_total = result.seen_label_counts.copy()
                round_client_losses.append(result.loss_total)
                round_client_ce_losses.append(result.loss_ce)
                round_client_intra_losses.append(result.loss_intra)
                round_client_inter_losses.append(result.loss_inter)
                client_states.append(result.state_dict)
            elif args.steps_per_round is not None:
                local_metrics, seen_samples_total, seen_counts_total = train_one_epoch(
                    client_model,
                    loader,
                    optimizer,
                    loss_fn,
                    device,
                    train_augment,
                    prox_mu=(args.fedprox_mu if args.aggregator == "fedprox" else 0.0),
                    global_named_params=global_named_params,
                    steps_limit=args.steps_per_round,
                )
                round_client_losses.append(local_metrics.loss)
                client_states.append({key: value.detach().cpu() for key, value in client_model.state_dict().items()})
                step_seen_sizes.append(seen_samples_total)
                step_seen_counts.append(seen_counts_total.copy())
            else:
                epoch_metrics: list[Metrics] = []
                for _ in range(args.epochs):
                    metrics, seen_samples, seen_counts = train_one_epoch(
                        client_model,
                        loader,
                        optimizer,
                        loss_fn,
                        device,
                        train_augment,
                        prox_mu=(args.fedprox_mu if args.aggregator == "fedprox" else 0.0),
                        global_named_params=global_named_params,
                        steps_limit=None,
                    )
                    epoch_metrics.append(metrics)
                    seen_samples_total += seen_samples
                    seen_counts_total += seen_counts
                local_metrics = Metrics(
                    loss=float(np.mean([metrics.loss for metrics in epoch_metrics])),
                    accuracy=float(np.mean([metrics.accuracy for metrics in epoch_metrics])),
                    macro_f1=float(np.mean([metrics.macro_f1 for metrics in epoch_metrics])),
                )
                round_client_losses.append(local_metrics.loss)
                client_states.append({key: value.detach().cpu() for key, value in client_model.state_dict().items()})
                if args.aggregator == "fedswitch":
                    step_seen_sizes.append(seen_samples_total)
                    step_seen_counts.append(seen_counts_total.copy())

            client_sizes.append(int(len(client_indices[client_id])))

        client_weights, aggregation_method = compute_client_weights(
            aggregator_name=args.aggregator,
            round_idx=round_idx,
            client_sizes=client_sizes,
            step_seen_sizes=step_seen_sizes,
            step_seen_counts=step_seen_counts,
            switch_round=args.switch_round,
            switch_target_estimate=switch_target_estimate,
        )

        aggregated_state = aggregate_state_dicts(client_states, client_weights)
        model.load_state_dict(aggregated_state)
        model.to(device)

        test_metrics = evaluate(model, test_loader, loss_fn, device)
        train_metrics = None
        if round_idx % args.eval_every == 0 or round_idx == args.rounds:
            train_metrics = evaluate(model, train_loader_all, loss_fn, device)

        history_entry: dict[str, float | int | list[int] | str] = {
            "round": round_idx,
            "selected_clients": selected_client_ids,
            "client_loss_mean": float(np.mean(round_client_losses)),
            "test_loss": test_metrics.loss,
            "test_accuracy": test_metrics.accuracy,
            "test_macro_f1": test_metrics.macro_f1,
        }
        if aggregation_method is not None:
            history_entry["aggregation_method"] = aggregation_method
        if args.aggregator == "fediic":
            history_entry["client_ce_mean"] = float(np.mean(round_client_ce_losses)) if round_client_ce_losses else 0.0
            history_entry["client_intra_mean"] = (
                float(np.mean(round_client_intra_losses)) if round_client_intra_losses else 0.0
            )
            history_entry["client_inter_mean"] = (
                float(np.mean(round_client_inter_losses)) if round_client_inter_losses else 0.0
            )
        if train_metrics is not None:
            history_entry["train_loss"] = train_metrics.loss
            history_entry["train_accuracy"] = train_metrics.accuracy
            history_entry["train_macro_f1"] = train_metrics.macro_f1
        history.append(history_entry)

        if args.aggregator == "fediic":
            print(
                f"round {round_idx:03d} | client_loss_mean={history_entry['client_loss_mean']:.4f} | "
                f"client_ce={history_entry['client_ce_mean']:.4f} | "
                f"intra={history_entry['client_intra_mean']:.4f} | "
                f"inter={history_entry['client_inter_mean']:.4f} | "
                f"test_loss={test_metrics.loss:.4f} | test_acc={test_metrics.accuracy:.4f} | "
                f"test_f1={test_metrics.macro_f1:.4f}"
            )
        elif args.aggregator == "fedswitch":
            print(
                f"round {round_idx:03d} | aggregation_method={history_entry['aggregation_method']} | "
                f"client_loss_mean={history_entry['client_loss_mean']:.4f} | "
                f"test_loss={test_metrics.loss:.4f} | test_acc={test_metrics.accuracy:.4f} | "
                f"test_f1={test_metrics.macro_f1:.4f}"
            )
        else:
            print(
                f"round {round_idx:03d} | client_loss_mean={history_entry['client_loss_mean']:.4f} | "
                f"test_loss={test_metrics.loss:.4f} | test_acc={test_metrics.accuracy:.4f} | "
                f"test_f1={test_metrics.macro_f1:.4f}"
            )

        if test_metrics.macro_f1 > best_test_f1:
            best_test_f1 = test_metrics.macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "round": round_idx,
                    "aggregator": args.aggregator,
                    "test_metrics": asdict(test_metrics),
                },
                checkpoint_path,
            )

    payload = {
        "config": {
            "data_path": str(args.data_path),
            "clients_label_csv": str(args.clients_label_csv),
            "aggregator": args.aggregator,
            "rounds": args.rounds,
            "epochs": args.epochs,
            "steps_per_round": args.steps_per_round,
            "clients": args.clients,
            "clients_per_round": args.clients_per_round,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
        },
        "best_test_macro_f1": best_test_f1,
        "history": history,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved model to: {checkpoint_path}")
    print(f"Saved metrics to: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
