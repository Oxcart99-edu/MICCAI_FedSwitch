from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r3d_18

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_LABEL_NAMES: Dict[int, str] = {
    0: "liver",
    1: "kidney_right",
    2: "kidney_left",
    3: "femur_right",
    4: "femur_left",
    5: "bladder",
    6: "heart",
    7: "lung_right",
    8: "lung_left",
    9: "spleen",
    10: "pancreas",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ToTensor3D:
    def __call__(self, volume: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(volume, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim == 4 and x.shape[-1] == 1:
            x = x.permute(3, 0, 1, 2)
        if x.max() > 1:
            x = x / 255.0
        return x


def normalize_label_name(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def anatomical_group_name(raw_name: str) -> str:
    name = normalize_label_name(raw_name)
    kidney_names = {"right kidney", "kidney right", "kidney left", "left kidney"}
    adrenal_names = {
        "left adrenal gland",
        "right adrenal gland",
        "adrenal right",
        "adrenal left",
        "left adrenal",
        "right adrenal",
        "adrenal gland right",
        "adrenal gland left",
    }
    vessel_names = {
        "portalvein",
        "vena cava inferior",
        "ivc",
        "portal and splenic vein",
        "aorta",
        "inferior vena cava",
        "portal vein and splenic vein",
        "postcava",
        "portal vein",
    }
    femur_names = {"right femur", "femur left", "left femur", "femur right"}
    lung_names = {"left lung", "right lung", "lung left", "lung right"}

    if name in kidney_names or ("kidney" in name and ("left" in name or "right" in name)):
        return "kidney"
    if name in adrenal_names or ("adrenal" in name and ("left" in name or "right" in name)):
        return "adrenal"
    if name in vessel_names:
        return "vessels"
    if name in femur_names or ("femur" in name and ("left" in name or "right" in name)):
        return "femur"
    if name in lung_names or ("lung" in name and ("left" in name or "right" in name)):
        return "lung"
    return raw_name


def build_anatomical_label_map(label_names: Dict[int, str]) -> Tuple[Dict[int, int], Dict[int, str]]:
    raw_to_merged: Dict[int, int] = {}
    merged_name_to_id: Dict[str, int] = {}
    merged_id_to_name: Dict[int, str] = {}

    for raw_id in sorted(label_names.keys()):
        merged_name = anatomical_group_name(label_names[raw_id])
        if merged_name not in merged_name_to_id:
            merged_id = len(merged_name_to_id)
            merged_name_to_id[merged_name] = merged_id
            merged_id_to_name[merged_id] = merged_name
        raw_to_merged[raw_id] = merged_name_to_id[merged_name]

    return raw_to_merged, merged_id_to_name


class NpzVolumeDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: ToTensor3D,
        label_map: Dict[int, int],
    ) -> None:
        self.images = images
        self.labels = labels.reshape(-1)
        self.transform = transform
        self.label_map = label_map

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = self.transform(self.images[index])
        raw_label = int(self.labels[index])
        return image, int(self.label_map[raw_label])


def build_model(num_classes: int) -> nn.Module:
    model = r3d_18(weights=None)
    model.stem[0] = nn.Conv3d(
        in_channels=1,
        out_channels=64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class FedIICR3D18(nn.Module):
    """R3D-18 backbone with projector head and FedIIC-compatible forward path."""

    def __init__(self, num_classes: int, projector_dim: int = 256) -> None:
        super().__init__()
        backbone = build_model(num_classes=num_classes)
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc
        feat_dim = int(self.fc.in_features)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, projector_dim),
        )

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor, project: bool = False):
        feats = self.forward_backbone(x)
        logits = self.fc(feats)
        if project:
            proj = self.projector(feats)
            return F.normalize(proj, dim=1), logits
        return logits

    def class_embedding(self) -> torch.Tensor:
        return self.fc.weight


def build_fediic_model(num_classes: int, projector_dim: int = 256) -> nn.Module:
    return FedIICR3D18(num_classes=num_classes, projector_dim=projector_dim)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += int((preds == labels).sum().item())
            total_seen += batch_size

    if total_seen == 0:
        return float("nan"), float("nan")
    return total_loss / total_seen, total_correct / total_seen


def load_datasets(data_dir: Path, merge_anatomical: bool) -> Tuple[NpzVolumeDataset, NpzVolumeDataset, NpzVolumeDataset, Dict[int, str]]:
    npz_path = data_dir / "organmnist3d.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    raw_to_merged, merged_id_to_name = build_anatomical_label_map(RAW_LABEL_NAMES)
    if not merge_anatomical:
        raw_to_merged = {idx: idx for idx in RAW_LABEL_NAMES}
        merged_id_to_name = {idx: name for idx, name in sorted(RAW_LABEL_NAMES.items())}

    payload = np.load(npz_path)
    transform = ToTensor3D()

    train_ds = NpzVolumeDataset(payload["train_images"], payload["train_labels"], transform, raw_to_merged)
    val_ds = NpzVolumeDataset(payload["val_images"], payload["val_labels"], transform, raw_to_merged)
    test_ds = NpzVolumeDataset(payload["test_images"], payload["test_labels"], transform, raw_to_merged)
    return train_ds, val_ds, test_ds, merged_id_to_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal OrganMNIST3D trainer")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-path", type=Path, default=BASE_DIR / "best_organmnist3d_r3d18.pt")
    parser.add_argument(
        "--merge-anatomical",
        dest="merge_anatomical",
        action="store_true",
        default=True,
        help="Merge left/right femur, kidney and lung labels into shared anatomical classes.",
    )
    parser.add_argument(
        "--no-merge-anatomical",
        dest="merge_anatomical",
        action="store_false",
        help="Keep the original 11 OrganMNIST3D labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    requested_device = args.device
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, label_names = load_datasets(args.data_dir, args.merge_anatomical)
    num_classes = len(label_names)

    print(f"Using {num_classes} classes on device={device}")
    for class_id, class_name in sorted(label_names.items()):
        print(f"  [{class_id}] {class_name}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
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

    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += int((preds == labels).sum().item())
            running_seen += batch_size

        scheduler.step()

        train_loss = running_loss / max(running_seen, 1)
        train_acc = running_correct / max(running_seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "best_val_acc": best_val_acc,
                "label_names": label_names,
                "merge_anatomical": args.merge_anatomical,
            }

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.save_path)

    model.load_state_dict(best_state["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test loss: {test_loss:.4f} test_acc={test_acc:.4f}")
    print(f"Checkpoint saved to: {args.save_path}")


if __name__ == "__main__":
    main()
