#!/usr/bin/env python3
"""Centralized trainer for BloodMNIST."""

from __future__ import annotations

import argparse
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
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: torch. Install the local requirements with "
        "`python -m pip install -r requirements.txt` from the `bloodmnist/` directory."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models import ResNetCIFAR  # noqa: E402

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
    def __init__(self, npz_path: Path, split: str):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        archive = np.load(npz_path)
        self.images = archive[f"{split}_images"]
        self.labels = archive[f"{split}_labels"].reshape(-1).astype(np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).permute(2, 0, 1).float().div(255.0)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Metrics:
    model.train()
    total_loss = 0.0
    total_examples = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        total_examples += images.size(0)
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return Metrics(
        loss=total_loss / max(1, total_examples),
        accuracy=float((preds == targets).mean()),
        macro_f1=compute_macro_f1(preds, targets, len(CLASS_NAMES)),
    )


def make_loss(labels: np.ndarray, device: torch.device, use_class_weights: bool) -> nn.Module:
    if not use_class_weights:
        return nn.CrossEntropyLoss()
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))


def default_data_path() -> Path:
    return PROJECT_ROOT / "dataset" / "bloodmnist.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet-18 CIFAR model on BloodMNIST.")
    parser.add_argument("--data-path", type=Path, default=default_data_path())
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_path.exists():
        raise SystemExit(f"Dataset not found: {args.data_path}")

    seed_everything(args.seed)
    device = torch.device(args.device)

    train_ds = BloodMNISTDataset(args.data_path, "train")
    val_ds = BloodMNISTDataset(args.data_path, "val")
    test_ds = BloodMNISTDataset(args.data_path, "test")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_ds, **eval_loader_kwargs)
    test_loader = DataLoader(test_ds, **eval_loader_kwargs)

    model = ResNetCIFAR(in_channels=3, num_classes=len(CLASS_NAMES), layers=(2, 2, 2, 2)).to(device)
    loss_fn = make_loss(train_ds.labels, device=device, use_class_weights=not args.no_class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.pt"
    metrics_path = args.output_dir / "metrics.json"

    history: list[dict[str, float | int]] = []
    best_val_f1 = -1.0

    print(f"Dataset: {args.data_path}")
    print(f"Train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Device: {device}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "train_macro_f1": train_metrics.macro_f1,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "val_macro_f1": val_metrics.macro_f1,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} train_f1={train_metrics.macro_f1:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.macro_f1:.4f}"
        )

        if val_metrics.macro_f1 > best_val_f1:
            best_val_f1 = val_metrics.macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "val_metrics": asdict(val_metrics),
                },
                checkpoint_path,
            )

    test_metrics = evaluate(model, test_loader, loss_fn, device)
    payload = {
        "config": {
            "data_path": str(args.data_path),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
        },
        "best_val_macro_f1": best_val_f1,
        "test_metrics": asdict(test_metrics),
        "history": history,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    print(
        "test | "
        f"loss={test_metrics.loss:.4f} acc={test_metrics.accuracy:.4f} f1={test_metrics.macro_f1:.4f}"
    )
    print(f"Saved model to: {checkpoint_path}")
    print(f"Saved metrics to: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
