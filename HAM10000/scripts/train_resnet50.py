"""
Centralized training script for ResNet-50 on HAM10000.

Also exposes reusable utilities used by the federated trainer:
  - HamDataset
  - build_transforms
  - create_model
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class HamDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        label_to_index: Dict[str, int],
        image_dirs: Sequence[str],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.transform = transform
        self.image_lookup = self._build_image_lookup(image_dirs)

    @staticmethod
    def _build_image_lookup(image_dirs: Sequence[str]) -> Dict[str, Path]:
        lookup: Dict[str, Path] = {}
        for image_dir in image_dirs:
            root = Path(image_dir)
            if not root.exists():
                continue
            for image_path in root.glob("*.jpg"):
                lookup[image_path.stem] = image_path
        return lookup

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.frame.iloc[idx]
        image_id = str(row["image_id"])
        label = str(row["dx"])
        label_idx = self.label_to_index[label]

        image_path = self.image_lookup.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"Image not found for image_id={image_id}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def create_model(num_classes: int, use_pretrained: bool = True) -> nn.Module:
    from torchvision.models import resnet50

    # Support both newer and older torchvision APIs.
    try:
        from torchvision.models import ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        model = resnet50(weights=weights)
    except Exception:
        model = resnet50(pretrained=use_pretrained)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-50 on HAM10000 (centralized)")
    parser.add_argument("--metadata", default="data/HAM10000_metadata.csv")
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        default=["data/HAM10000_images_part_1", "data/HAM10000_images_part_2"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output", default="resnet50_ham10000.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    from sklearn.metrics import f1_score

    model.eval()
    total_loss = 0.0
    total = 0
    preds: List[int] = []
    labels_all: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        preds.extend(outputs.argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    if total == 0:
        return float("nan"), float("nan"), float("nan")

    avg_loss = total_loss / total
    acc = sum(int(p == t) for p, t in zip(preds, labels_all)) / total
    macro_f1 = f1_score(labels_all, preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def main() -> None:
    from sklearn.model_selection import train_test_split

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame = pd.read_csv(args.metadata)
    label_names = sorted(frame["dx"].unique())
    label_to_index = {name: idx for idx, name in enumerate(label_names)}
    num_classes = len(label_names)

    if args.test_split > 0:
        train_val_frame, test_frame = train_test_split(
            frame,
            test_size=args.test_split,
            stratify=frame["dx"],
            random_state=args.seed,
        )
    else:
        train_val_frame, test_frame = frame, frame.iloc[0:0]

    if args.val_split > 0:
        train_frame, val_frame = train_test_split(
            train_val_frame,
            test_size=args.val_split,
            stratify=train_val_frame["dx"],
            random_state=args.seed,
        )
    else:
        train_frame, val_frame = train_val_frame, train_val_frame.iloc[0:0]

    train_tf, eval_tf = build_transforms()
    train_ds = HamDataset(train_frame, label_to_index, args.image_dirs, transform=train_tf)
    val_ds = HamDataset(val_frame, label_to_index, args.image_dirs, transform=eval_tf)
    test_ds = HamDataset(test_frame, label_to_index, args.image_dirs, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=num_classes, use_pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1

        train_loss, train_acc, train_f1 = evaluate(model, train_loader, criterion, device)
        if len(val_ds) > 0:
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc, val_f1 = float("nan"), float("nan"), float("nan")
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
        mean_step_loss = running_loss / max(steps, 1)

        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"step_loss={mean_step_loss:.4f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
        )

        score = val_f1 if len(val_ds) > 0 else test_f1
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state": best_state,
                    "label_names": label_names,
                    "best_epoch": best_epoch,
                    "best_score": float(best_f1),
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output} (epoch={best_epoch}, score={best_f1:.4f})")

    print(f"Training completed. Best epoch={best_epoch}, best_score={best_f1:.4f}")


if __name__ == "__main__":
    main()
