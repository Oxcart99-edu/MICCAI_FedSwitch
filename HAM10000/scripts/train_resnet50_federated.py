"""
Federated training of ResNet-50 on HAM10000 with multiple aggregation strategies.

This script only supports precomputed client distributions loaded from
``sample_distribution/clients_label_alpha_*.csv``.

Requirements: torch, torchvision, pandas, scikit-learn, pillow
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep matplotlib fully headless and avoid writing cache under an unwritable/home-mapped path.
os.environ.setdefault("MPLCONFIGDIR", str(Path("plots") / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from scripts.aggregators import AGGREGATOR_REGISTRY, get_aggregator
from scripts.train_resnet50 import HamDataset, build_transforms, create_model


def _can_use_multiprocess_dataloader(num_workers: int) -> tuple[bool, str]:
    """Validate DataLoader workers with a real tiny batch transfer."""
    if num_workers <= 0:
        return True, ""
    try:
        probe_images = torch.zeros((8, 3, 8, 8), dtype=torch.float32)
        probe_labels = torch.zeros(8, dtype=torch.long)
        probe_ds = torch.utils.data.TensorDataset(probe_images, probe_labels)
        probe_loader = DataLoader(
            probe_ds,
            batch_size=4,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )
        next(iter(probe_loader))
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, str(exc)
    return True, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated ResNet-50 on HAM10000")
    parser.add_argument(
        "--metadata",
        default="data/HAM10000_metadata.csv",
        help="Path to HAM10000 metadata CSV (contains image_id, dx, etc.)",
    )
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        default=["data/HAM10000_images_part_1", "data/HAM10000_images_part_2"],
        help="Directories containing HAM10000 images",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30, help="Total effective epochs (rounds * local_epochs)")
    parser.add_argument("--rounds", type=int, default=500, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per selected client per round")
    parser.add_argument(
        "--steps-per-round",
        type=int,
        default=None,
        help="If set, run a fixed number of Adam steps per client per round instead of full local_epochs",
    )
    parser.add_argument("--num-clients", type=int, default=100, help="Total simulated clients")
    parser.add_argument(
        "--client-partition",
        type=str,
        default="sample_distribution",
        choices=["sample_distribution"],
        help="Client partitioning is loaded from precomputed CSVs in sample_distribution.",
    )
    parser.add_argument(
        "--sample-distribution-dir",
        type=str,
        default="sample_distribution",
        help="Directory containing precomputed clients_label_alpha_*.csv files.",
    )
    parser.add_argument(
        "--clients-label-csv",
        type=str,
        default=None,
        help=(
            "Optional path to a specific CSV with per-client label counts. "
            "If omitted, it is resolved from --sample-distribution-dir using --dirichlet-alpha."
        ),
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=1.0,
        help=(
            "Alpha tag used to resolve the precomputed CSV "
            "(e.g. sample_distribution/clients_label_alpha_1.0.csv)."
        ),
    )
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=10,
        help="Number of clients sampled each round (must be <= num_clients)",
    )
    parser.add_argument(
        "--picked-ids-json",
        type=str,
        default=None,
        help=(
            "Optional JSON schedule for selected client ids per round "
            "(e.g. sample_distribution/picked_ids_1000_rounds_seed42.json)."
        ),
    )
    parser.add_argument(
        "--aggregators",
        nargs="+",
        default=["stratified"],
        choices=sorted(AGGREGATOR_REGISTRY.keys()),
        help="Aggregation strategies to run (each run executes independently)",
    )
    parser.add_argument(
        "--all-aggregators",
        action="store_true",
        help="Run experiments for all registered aggregators (overrides --aggregators)",
    )
    parser.add_argument(
        "--switch-round",
        type=int,
        default=None,
        help=(
            "Round at which switch_stratified starts using stratified weights "
            "(rounds before this use uniform client weights)."
        ),
    )
    parser.add_argument(
        "--switch-stratified-weight-source",
        type=str,
        default="client",
        choices=["client", "steps"],
        help=(
            "Weight source for switch_stratified after --switch-round: "
            "'client' uses full selected-client distributions, "
            "'steps' uses only labels/samples seen during local training steps."
        ),
    )
    parser.add_argument(
        "--switch-stratified-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Verbose per-round logging for switch_stratified: prints selected client IDs, "
            "samples seen during local steps, aggregation sample counts, and final weights."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--fedprox-mu",
        type=float,
        default=0.01,
        help="FedProx proximal coefficient (used only when aggregator=fedprox)",
    )
    parser.add_argument(
        "--fedlc-tau",
        type=float,
        default=0.5,
        help="FedLC logits calibration strength tau (used only when aggregator=fedlc)",
    )
    parser.add_argument(
        "--fediic-d",
        type=float,
        default=0.25,
        help="FedIIC DALA difficulty exponent d (used only when aggregator=fediic)",
    )
    parser.add_argument(
        "--fediic-tau",
        type=float,
        default=0.5,
        help="FedIIC DALA margin scale tau (used only when aggregator=fediic)",
    )
    parser.add_argument(
        "--fediic-loss-scope",
        type=str,
        default="all",
        choices=["all", "selected"],
        help=(
            "Client scope used to estimate FedIIC per-class loss each round: "
            "'all' (paper-like, more expensive) or 'selected' (faster)."
        ),
    )
    parser.add_argument("--val-split", type=float, default=0, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio taken from the full dataset")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers. Set >0 only if the system allows multiprocessing sockets.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision on CUDA (use --no-amp to disable).",
    )
    parser.add_argument(
        "--output",
        default="resnet50_fedavg.pt",
        help="Where to save the best model checkpoint",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fallback seed used for both partitioning and model randomness unless specific seeds are provided.",
    )
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=None,
        help="Seed used for dataset split and deterministic assignment from the precomputed CSV.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Seed used for model/training randomness (init, client sampling, torch RNG).",
    )
    return parser.parse_args()


def _to_int_series(series: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    rounded = numeric.round()
    if not (numeric == rounded).all():
        raise ValueError(f"Column '{column_name}' must contain integer values.")
    integers = rounded.astype(int)
    if (integers < 0).any():
        raise ValueError(f"Column '{column_name}' must contain non-negative values.")
    return integers


def _sample_distribution_alpha_candidates(alpha: float) -> List[str]:
    candidates = [str(alpha), f"{alpha:.1f}", f"{alpha:g}"]
    ordered: List[str] = []
    seen = set()
    for value in candidates:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def resolve_sample_distribution_csv(
    clients_label_csv: Optional[str],
    sample_distribution_dir: str,
    alpha: float,
) -> str:
    if clients_label_csv:
        return clients_label_csv

    root = Path(sample_distribution_dir).expanduser()
    for alpha_token in _sample_distribution_alpha_candidates(alpha):
        candidate = root / f"clients_label_alpha_{alpha_token}.csv"
        if candidate.exists():
            return str(candidate)

    searched = ", ".join(
        str(root / f"clients_label_alpha_{alpha_token}.csv")
        for alpha_token in _sample_distribution_alpha_candidates(alpha)
    )
    raise FileNotFoundError(
        "Could not resolve sample-distribution CSV. "
        f"Searched: {searched}. "
        "Either provide --clients-label-csv explicitly or add the matching file."
    )


def allocate_from_sample_distribution_csv(
    *,
    train_frame: pd.DataFrame,
    label_to_indices: Dict[str, List[int]],
    label_order: Sequence[str],
    num_clients: int,
    csv_path: str,
) -> Tuple[List[List[int]], List[int]]:
    csv_file = Path(csv_path).expanduser().resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"Sample distribution CSV not found: {csv_file}")

    dist_df = pd.read_csv(csv_file)
    if "client_id" not in dist_df.columns:
        raise ValueError(f"CSV {csv_file} must contain a 'client_id' column.")

    client_ids = _to_int_series(dist_df["client_id"], "client_id")
    dist_df = dist_df.copy()
    dist_df["client_id"] = client_ids
    if dist_df["client_id"].duplicated().any():
        dupes = sorted(dist_df.loc[dist_df["client_id"].duplicated(), "client_id"].unique().tolist())
        raise ValueError(f"CSV {csv_file} has duplicate client_id values: {dupes}")

    expected_ids = set(range(num_clients))
    actual_ids = set(dist_df["client_id"].tolist())
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            f"CSV {csv_file} client_id mismatch. missing={missing_ids}, extra={extra_ids}, expected 0..{num_clients - 1}."
        )

    dist_df = dist_df.set_index("client_id").reindex(list(range(num_clients)))
    count_columns: Dict[str, str] = {}
    for label in label_order:
        candidate_count = f"count_{label}"
        if candidate_count in dist_df.columns:
            count_columns[label] = candidate_count
        elif label in dist_df.columns:
            count_columns[label] = label
        else:
            raise ValueError(
                f"CSV {csv_file} missing column for label '{label}'. "
                f"Expected '{candidate_count}' or '{label}'."
            )

    for label, col_name in count_columns.items():
        dist_df[col_name] = _to_int_series(dist_df[col_name], col_name)

    client_bins: List[List[int]] = [[] for _ in range(num_clients)]
    client_counts = [0 for _ in range(num_clients)]
    for label in label_order:
        indices = label_to_indices.get(label, [])
        requests = dist_df[count_columns[label]].astype(int).tolist()
        requested_total = int(sum(requests))
        available_total = len(indices)
        if requested_total != available_total:
            raise ValueError(
                f"Label '{label}' count mismatch in {csv_file}: requested={requested_total}, available={available_total}."
            )

        start = 0
        for client_idx, cnt in enumerate(requests):
            if cnt <= 0:
                continue
            end = start + cnt
            chunk = indices[start:end]
            if len(chunk) != cnt:
                raise ValueError(
                    f"Insufficient samples while assigning label '{label}' to client {client_idx}: "
                    f"expected {cnt}, got {len(chunk)}."
                )
            client_bins[client_idx].extend(chunk)
            client_counts[client_idx] += len(chunk)
            start = end

    if any(size <= 0 for size in client_counts):
        empty_ids = [idx for idx, size in enumerate(client_counts) if size <= 0]
        raise ValueError(
            f"CSV {csv_file} produced empty clients {empty_ids}. "
            "Provide at least one sample per client."
        )
    return client_bins, client_counts


def load_picked_ids_schedule(
    *,
    json_path: str,
    rounds: int,
    clients_per_round: int,
    num_clients: int,
) -> List[List[int]]:
    schedule_file = Path(json_path).expanduser().resolve()
    if not schedule_file.exists():
        raise FileNotFoundError(f"Picked-ids JSON not found: {schedule_file}")

    with schedule_file.open(encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Picked-ids JSON {schedule_file} must contain a list of round entries.")

    round_to_ids: Dict[int, List[int]] = {}
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid entry at index {idx} in {schedule_file}: expected object.")
        if "round" not in entry or "picked_ids" not in entry:
            raise ValueError(f"Entry {idx} in {schedule_file} must contain 'round' and 'picked_ids'.")
        round_idx = int(entry["round"])
        if round_idx < 1:
            raise ValueError(f"Invalid round {round_idx} in {schedule_file}: rounds must start at 1.")
        if round_idx in round_to_ids:
            raise ValueError(f"Duplicate round {round_idx} in {schedule_file}.")

        picked_ids_raw = entry["picked_ids"]
        if not isinstance(picked_ids_raw, list):
            raise ValueError(f"'picked_ids' for round {round_idx} in {schedule_file} must be a list.")
        picked_ids = [int(client_id) for client_id in picked_ids_raw]
        if len(picked_ids) != clients_per_round:
            raise ValueError(
                f"Round {round_idx} in {schedule_file} has {len(picked_ids)} clients, "
                f"expected {clients_per_round}."
            )
        if len(set(picked_ids)) != len(picked_ids):
            raise ValueError(f"Round {round_idx} in {schedule_file} contains duplicate client ids.")
        invalid_ids = sorted(client_id for client_id in picked_ids if client_id < 0 or client_id >= num_clients)
        if invalid_ids:
            raise ValueError(
                f"Round {round_idx} in {schedule_file} contains invalid client ids {invalid_ids}. "
                f"Allowed range is 0..{num_clients - 1}."
            )
        round_to_ids[round_idx] = picked_ids

    missing_rounds = [round_idx for round_idx in range(1, rounds + 1) if round_idx not in round_to_ids]
    if missing_rounds:
        preview = missing_rounds[:10]
        suffix = "..." if len(missing_rounds) > 10 else ""
        raise ValueError(
            f"Picked-ids schedule {schedule_file} is missing rounds: {preview}{suffix} "
            f"(required 1..{rounds})."
        )

    return [round_to_ids[round_idx] for round_idx in range(1, rounds + 1)]


def prepare_data(
    metadata_path: str,
    image_dirs: Sequence[str],
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[
    pd.DataFrame,
    HamDataset,
    HamDataset,
    List[str],
    Dict[str, int],
    int,
    Tuple[transforms.Compose, transforms.Compose],
]:
    frame = pd.read_csv(metadata_path)
    label_names = sorted(frame["dx"].unique())
    label_to_index = {label: idx for idx, label in enumerate(label_names)}
    num_classes = len(label_names)
    overall_counts = frame["dx"].value_counts().reindex(label_names, fill_value=0)
    print("Overall label distribution (full metadata):")
    for label in label_names:
        print(f"  {label}: {overall_counts[label]}")

    # First carve out test (if requested), then build validation from remaining train_val.
    if test_split and test_split > 0:
        train_val_frame, test_frame = train_test_split(
            frame,
            test_size=test_split,
            stratify=frame["dx"],
            random_state=seed,
        )
    else:
        train_val_frame, test_frame = frame, frame.iloc[0:0]

    if val_split and val_split > 0:
        train_frame, val_frame = train_test_split(
            train_val_frame,
            test_size=val_split,
            stratify=train_val_frame["dx"],
            random_state=seed,
        )
    else:
        train_frame, val_frame = train_val_frame, train_val_frame.iloc[0:0]

    train_tf, val_tf = build_transforms()
    val_ds = HamDataset(val_frame, label_to_index, image_dirs, transform=val_tf)
    test_ds = HamDataset(test_frame, label_to_index, image_dirs, transform=val_tf)
    return (
        train_frame.reset_index(drop=True),
        val_ds,
        test_ds,
        label_names,
        label_to_index,
        num_classes,
        (train_tf, val_tf),
    )


def build_client_loaders(
    train_frame: pd.DataFrame,
    label_to_index: Dict[str, int],
    image_dirs: Sequence[str],
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    num_clients: int,
    seed: int,
    clients_label_csv: str,
) -> List[Tuple[DataLoader, int, Dict[str, int]]]:
    rng = random.Random(seed)
    if len(train_frame) < num_clients:
        raise ValueError("Not enough samples to allocate at least one item per client.")
    if not clients_label_csv:
        raise ValueError("--clients-label-csv is required.")

    grouped_indices = train_frame.groupby("dx").groups
    label_order = sorted(label_to_index.keys())
    label_to_indices: Dict[str, List[int]] = {}
    for label in label_order:
        source_indices = list(grouped_indices.get(label, []))
        label_to_indices[label] = rng.sample(source_indices, k=len(source_indices))

    client_bins, client_counts = allocate_from_sample_distribution_csv(
        train_frame=train_frame,
        label_to_indices=label_to_indices,
        label_order=label_order,
        num_clients=num_clients,
        csv_path=clients_label_csv,
    )

    loaders: List[Tuple[DataLoader, int, Dict[str, int]]] = []
    print(f"Client label distribution (sample_distribution, csv={clients_label_csv}):")
    for client_idx in range(num_clients):
        client_frame = train_frame.iloc[client_bins[client_idx]]
        counts = client_frame["dx"].value_counts().reindex(label_order, fill_value=0)
        counts_str = ", ".join(f"{label}: {counts[label]}" for label in label_order)
        print(f"  Client {client_idx}: {counts_str} (total {len(client_frame)})")
        client_ds = HamDataset(client_frame, label_to_index, image_dirs, transform=transform)
        loader = DataLoader(
            client_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        loaders.append((loader, len(client_ds), counts.to_dict()))
    return loaders


@torch.no_grad()
def evaluate_with_f1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_samples += labels.size(0)

    if total_samples == 0:
        return float("nan"), float("nan"), float("nan")

    avg_loss = total_loss / total_samples
    avg_acc = sum(int(p == t) for p, t in zip(all_preds, all_labels)) / total_samples
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, avg_acc, macro_f1


@torch.no_grad()
def compute_confusion(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> List[List[int]]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    conf = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes))).tolist()
    return conf, all_labels, all_preds


def save_confusion_csv(confusion: List[List[int]], label_names: List[str], path: Path) -> None:
    frame = pd.DataFrame(confusion, index=label_names, columns=label_names)
    frame.index.name = "true_label"
    frame.columns.name = "pred_label"
    frame.to_csv(path)


def main() -> None:
    args = parse_args()
    args.clients_label_csv = resolve_sample_distribution_csv(
        clients_label_csv=args.clients_label_csv,
        sample_distribution_dir=args.sample_distribution_dir,
        alpha=args.dirichlet_alpha,
    )
    print(f"Using sample-distribution CSV: {args.clients_label_csv}")
    if args.test_split != 0.1 or args.val_split != 0:
        print(
            "For sample_distribution, forcing --test-split=0.1 and --val-split=0 "
            "to match CSV class totals on the 90% train split."
        )
    args.test_split = 0.1
    args.val_split = 0.0
    if getattr(args, "all_aggregators", False):
        args.aggregators = sorted(AGGREGATOR_REGISTRY.keys())
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients-per-round must be <= num-clients")
    partition_seed = args.partition_seed if args.partition_seed is not None else args.seed
    model_seed = args.model_seed if args.model_seed is not None else args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.multiprocessing.set_sharing_strategy("file_system")
    if args.num_workers > 0:
        workers_ok, reason = _can_use_multiprocess_dataloader(args.num_workers)
        if not workers_ok:
            print(
                "Multiprocess DataLoader unavailable in this environment "
                f"({reason}). Falling back to --num-workers=0."
            )
            args.num_workers = 0
    pin_memory = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    random.seed(model_seed)
    torch.manual_seed(model_seed)
    print(f"Seeds: partition_seed={partition_seed}, model_seed={model_seed}")

    train_frame, val_ds, test_ds, label_names, label_to_index, num_classes, (train_tf, eval_tf) = prepare_data(
        metadata_path=args.metadata,
        image_dirs=args.image_dirs,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=partition_seed,
    )

    client_loaders = build_client_loaders(
        train_frame=train_frame,
        label_to_index=label_to_index,
        image_dirs=args.image_dirs,
        transform=train_tf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        num_clients=args.num_clients,
        seed=partition_seed,
        clients_label_csv=args.clients_label_csv,
    )
    available_clients = len(client_loaders)
    if available_clients == 0:
        raise ValueError("No clients contain data; check num-clients vs dataset size.")
    picked_ids_schedule: Optional[List[List[int]]] = None
    if args.picked_ids_json:
        picked_ids_schedule = load_picked_ids_schedule(
            json_path=args.picked_ids_json,
            rounds=args.rounds,
            clients_per_round=args.clients_per_round,
            num_clients=available_clients,
        )
        print(f"Using picked client ids from {args.picked_ids_json} for {args.rounds} rounds.")
    label_order = sorted(label_to_index.keys())

    has_validation = len(val_ds) > 0
    val_loader: Optional[DataLoader] = None
    if has_validation:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=args.num_workers > 0,
        )
    else:
        print("Validation split disabled (--val-split=0). Skipping validation metrics and reports.")
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.amp and not amp_enabled:
        print("AMP requested but CUDA is unavailable; running local training in FP32.")
    output_path = Path(args.output)
    preferred_agg_order = ["fedavg", "fediic", "fedlc", "fedprox", "switch_stratified", "stratified"]
    ordered_aggs: List[str] = [name for name in preferred_agg_order if name in args.aggregators]
    for name in args.aggregators:
        if name not in ordered_aggs:
            ordered_aggs.append(name)

    multiple_aggs = len(ordered_aggs) > 1
    run_root = Path("plots") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    alpha_str = f"{args.dirichlet_alpha:g}"

    for aggregator_name in ordered_aggs:
        aggregator = get_aggregator(aggregator_name)
        configure_kwargs = dict(
            rounds=args.rounds,
            steps_per_round=args.steps_per_round,
            switch_round=args.switch_round,
            switch_weight_source=args.switch_stratified_weight_source,
            dirichlet_alpha=args.dirichlet_alpha,
            fedprox_mu=args.fedprox_mu,
            learning_rate=args.learning_rate,
        )
        if aggregator.name == "fedlc":
            configure_kwargs["fedlc_tau"] = args.fedlc_tau
        if aggregator.name == "fediic":
            configure_kwargs["fediic_d"] = args.fediic_d
            configure_kwargs["fediic_tau"] = args.fediic_tau
            configure_kwargs["fediic_loss_scope"] = args.fediic_loss_scope
        aggregator.configure(**configure_kwargs)
        switch_verbose = bool(args.switch_stratified_verbose and aggregator.name == "switch_stratified")
        experiment_dir_name = f"{aggregator.name}_{alpha_str}_{model_seed}"
        run_dir = run_root / experiment_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Running aggregator: {aggregator.name} for {args.rounds} rounds ===")
        note = aggregator.experiment_note()
        if note:
            print(note)
        if switch_verbose:
            print(
                "[switch_stratified][verbose] Enabled per-round client selection and weight logging "
                "(shows step-seen samples and aggregation weights)."
            )

            def _format_verbose_label_counts(counts: Dict[object, int]) -> str:
                parts: List[str] = []
                for label in label_order:
                    class_idx = label_to_index[label]
                    value = 0
                    if class_idx in counts:
                        value = int(counts[class_idx])
                    elif str(class_idx) in counts:
                        value = int(counts[str(class_idx)])
                    elif label in counts:
                        value = int(counts[label])
                    if value > 0:
                        parts.append(f"{label}:{value}")
                return ", ".join(parts) if parts else "-"

        client_sizes_all = [entry[1] for entry in client_loaders]
        client_counts_all = [entry[2] for entry in client_loaders]
        client_weights_all = aggregator.compute_client_weights(
            client_sizes_all, class_counts=client_counts_all, num_classes=num_classes
        )
        print("Client label distribution with aggregation weights:")
        for idx, (counts, weight, size) in enumerate(zip(client_counts_all, client_weights_all, client_sizes_all)):
            counts_str = ", ".join(f"{label}:{counts.get(label, 0)}" for label in label_order)
            print(f"  Client {idx}: weight={weight:.4f}, {counts_str} (total {size})")

        # Create a per-experiment client distribution CSV and a stacked bar plot
        # Columns: per-class counts, plus 'total_samples' and 'weight'
        clients_data = []
        for idx, (counts, weight, size) in enumerate(zip(client_counts_all, client_weights_all, client_sizes_all)):
            row = {label: counts.get(label, 0) for label in label_order}
            row["total_samples"] = size
            row["weight"] = weight
            row["client_id"] = idx
            clients_data.append(row)

        clients_df = pd.DataFrame(clients_data).set_index("client_id")
        csv_path = run_dir / f"{aggregator.name}_client_distribution_{args.num_clients}clients.csv"
        clients_df.to_csv(csv_path)

        # Stacked bar plot of class counts per client, annotated with weight and total
        ax = clients_df[label_order].plot(kind="bar", stacked=True, figsize=(14, 6), width=0.8)
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Samples per Class")
        ax.set_title(f"Client class distribution and aggregation weights ({aggregator.name})")
        # Annotate weight and total samples above each bar
        for i, (tot, w) in enumerate(zip(clients_df["total_samples"].tolist(), clients_df["weight"].tolist())):
            ax.text(i, tot + max(1, tot * 0.02), f"w={w:.3f}\nn={int(tot)}", ha="center", va="bottom", fontsize=8)

        plot_path = run_dir / f"{aggregator.name}_client_distribution_{args.num_clients}clients.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved client distribution CSV and plot to {run_dir}")

        global_model = create_model(num_classes=len(label_names), use_pretrained=not args.no_pretrained)
        global_model.to(device)
        aggregator.on_experiment_start(global_model, available_clients)

        model_output = (
            run_dir / f"{output_path.stem}_{aggregator.name}{output_path.suffix}"
            if multiple_aggs
            else output_path
        )

        best_f1 = -1.0
        best_round: int | None = None
        best_state: Dict[str, torch.Tensor] | None = None
        history_rounds: List[int] = []
        test_losses: List[float] = []
        test_f1s: List[float] = []
        metrics_rows: List[Dict[str, float | int]] = []
        metrics_csv_path = run_dir / f"{aggregator.name}_metrics_{args.num_clients}clients.csv"
        metrics_columns = [
            "round",
            "local_train_loss",
            "train_loss",
            "test_loss",
            "train_acc",
            "test_acc",
            "train_f1",
            "test_f1",
        ]
        if has_validation:
            metrics_columns[3:3] = ["val_loss"]
            metrics_columns[6:6] = ["val_acc"]
            metrics_columns[9:9] = ["val_f1"]
        rounds = args.rounds
        interrupted = False

        try:
            all_client_ids = list(range(available_clients))
            # Create client model once, reuse by loading state_dict (faster than deepcopy)
            client_model = create_model(num_classes=len(label_names), use_pretrained=False)
            client_model.to(device)
            
            for round_idx in range(1, rounds + 1):
                aggregator.on_round_start(round_idx)
                if picked_ids_schedule is not None:
                    selected_client_ids = picked_ids_schedule[round_idx - 1]
                elif args.clients_per_round == available_clients:
                    selected_client_ids = all_client_ids
                else:
                    selected_client_ids = random.sample(all_client_ids, k=args.clients_per_round)
                    selected_client_ids.sort()

                selected_client_sizes = [client_sizes_all[idx] for idx in selected_client_ids]
                selected_client_counts = [client_counts_all[idx] for idx in selected_client_ids]

                prepare_round_training = getattr(aggregator, "prepare_round_training", None)
                if callable(prepare_round_training):
                    prepare_round_training(
                        round_idx=round_idx,
                        global_model=global_model,
                        client_loaders=client_loaders,
                        selected_client_ids=selected_client_ids,
                        device=device,
                        num_classes=num_classes,
                    )

                client_states: List[Dict[str, torch.Tensor]] = []
                round_train_losses: List[float] = []
                global_named_params = {
                    name: param.detach().clone() for name, param in global_model.named_parameters() if param.requires_grad
                }
                step_seen_sizes: List[int] = []
                step_seen_label_counts: List[Dict[int, int]] = []
                
                for client_id in selected_client_ids:
                    loader, _, client_label_counts = client_loaders[client_id]
                    client_class_counts = {
                        int(label_to_index[label]): int(count)
                        for label, count in client_label_counts.items()
                        if label in label_to_index and int(count) > 0
                    }
                    # Load global state instead of deepcopy - much faster
                    client_model.load_state_dict(global_model.state_dict())
                    optimizer = optim.Adam(
                        client_model.parameters(),
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                    )
                    local_result = aggregator.train_local(
                        model=client_model,
                        loader=loader,
                        device=device,
                        criterion=criterion,
                        optimizer=optimizer,
                        local_epochs=args.local_epochs,
                        steps_per_round=args.steps_per_round,
                        client_id=client_id,
                        global_named_params=global_named_params,
                        learning_rate=args.learning_rate,
                        use_amp=amp_enabled,
                        num_classes=num_classes,
                        client_class_counts=client_class_counts,
                    )
                    step_seen_sizes.append(local_result.seen_samples)
                    step_seen_label_counts.append(local_result.seen_label_counts)

                    client_states.append({k: v.detach().clone() for k, v in client_model.state_dict().items()})
                    round_train_losses.append(local_result.train_loss)

                round_plan = aggregator.resolve_round_aggregation(
                    round_idx=round_idx,
                    selected_client_sizes=selected_client_sizes,
                    selected_client_counts=selected_client_counts,
                    step_seen_sizes=step_seen_sizes,
                    step_seen_label_counts=step_seen_label_counts,
                    num_classes=num_classes,
                )
                if switch_verbose:
                    source = "step_seen_samples"
                    print(
                        f"[switch_stratified][verbose] round={round_idx:03d} source={source} "
                        f"selected_clients={selected_client_ids}"
                    )
                    for idx, client_id in enumerate(selected_client_ids):
                        total_samples = int(selected_client_sizes[idx]) if idx < len(selected_client_sizes) else -1
                        seen_samples = int(step_seen_sizes[idx]) if idx < len(step_seen_sizes) else -1
                        agg_samples = int(round_plan.client_sizes[idx]) if idx < len(round_plan.client_sizes) else -1
                        raw_weight = (
                            float(round_plan.raw_client_weights[idx])
                            if round_plan.raw_client_weights is not None and idx < len(round_plan.raw_client_weights)
                            else float("nan")
                        )
                        weight = float(round_plan.client_weights[idx]) if idx < len(round_plan.client_weights) else float("nan")
                        print(
                            f"[switch_stratified][verbose]   client={client_id:03d} "
                            f"total_samples={total_samples} seen_samples={seen_samples} "
                            f"agg_samples={agg_samples} raw_weight={raw_weight:.6f} "
                            f"normalized_weight={weight:.6f}"
                        )
                        seen_labels = _format_verbose_label_counts(step_seen_label_counts[idx])
                        agg_labels = (
                            _format_verbose_label_counts(round_plan.class_counts[idx])
                            if idx < len(round_plan.class_counts)
                            else "-"
                        )
                        print(
                            f"[switch_stratified][verbose]     seen_labels={seen_labels} "
                            f"agg_labels={agg_labels}"
                        )

                averaged_state = aggregator.aggregate(
                    client_states,
                    round_plan.client_sizes,
                    class_counts=round_plan.class_counts,
                    num_classes=num_classes,
                    client_weights=round_plan.client_weights,
                )
                global_model.load_state_dict(averaged_state)
                aggregator.on_global_model_updated(available_clients)

                train_loss_eval, train_acc, train_f1 = float("nan"), float("nan"), float("nan")
                if has_validation and val_loader is not None:
                    val_loss, val_acc, val_f1 = evaluate_with_f1(global_model, val_loader, device, criterion)
                else:
                    val_loss, val_acc, val_f1 = float("nan"), float("nan"), float("nan")
                test_loss, test_acc, test_f1 = evaluate_with_f1(global_model, test_loader, device, criterion)
                effective_epoch = round_idx * args.local_epochs
                mean_train_loss = sum(round_train_losses) / max(len(round_train_losses), 1)
                phase_suffix = round_plan.phase_suffix
                val_segment = (
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
                    if has_validation
                    else ""
                )
                print(
                    f"[{aggregator.name}] Round {round_idx:02d}/{rounds} (approx epoch {effective_epoch:02d})"
                    f"{phase_suffix} "
                    f"local_train_loss={mean_train_loss:.4f} "
                    f"{val_segment}"
                    f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
                )

                history_rounds.append(round_idx)
                test_losses.append(test_loss)
                test_f1s.append(test_f1)
                row: Dict[str, float | int] = {
                    "round": round_idx,
                    "local_train_loss": mean_train_loss,
                    "train_loss": train_loss_eval,
                    "test_loss": test_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_f1": train_f1,
                    "test_f1": test_f1,
                }
                if has_validation:
                    row["val_loss"] = val_loss
                    row["val_acc"] = val_acc
                    row["val_f1"] = val_f1
                metrics_rows.append(row)
                pd.DataFrame(metrics_rows, columns=metrics_columns).to_csv(metrics_csv_path, index=False)

                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_round = round_idx
                    best_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
                    torch.save(
                        {
                            "model_state": best_state,
                            "label_names": label_names,
                            "test_f1": test_f1,
                            "round": round_idx,
                            "aggregator": aggregator.name,
                        },
                        str(model_output),
                    )
                    print(f"Saved new best global model to {model_output} (test_f1={test_f1:.4f})")
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[{aggregator.name}] Interrupted by user. Partial metrics saved to {metrics_csv_path}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_rounds, test_losses, label="Test Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss per Round")

        plt.subplot(1, 2, 2)
        plt.plot(history_rounds, test_f1s, label="Test F1")
        plt.xlabel("Round")
        plt.ylabel("Macro F1")
        plt.legend()
        plt.title("F1 per Round")

        plt.tight_layout()
        metrics_filename = run_dir / f"{aggregator.name}_metrics_{args.num_clients}clients.png"
        plt.savefig(metrics_filename)
        plt.close()
        pd.DataFrame(metrics_rows, columns=metrics_columns).to_csv(metrics_csv_path, index=False)
        print(f"Saved training curves to {metrics_filename}")
        print(f"Saved per-round metrics CSV to {metrics_csv_path}")
        print(f"[{aggregator.name}] Best test F1: {best_f1:.4f}")
        if interrupted:
            return

        # Final metrics matrix for quick reference.
        val_loss_final = float("nan")
        val_acc_final = float("nan")
        val_f1_final = float("nan")
        if has_validation and val_loader is not None:
            val_loss_final, val_acc_final, val_f1_final = evaluate_with_f1(global_model, val_loader, device, criterion)
        test_loss_final, test_acc_final, test_f1_final = evaluate_with_f1(global_model, test_loader, device, criterion)
        print(f"\n[{aggregator.name}] Final metrics matrix:")
        print("           loss      acc      macro_f1")
        if has_validation:
            print(f"  val   {val_loss_final:8.4f} {val_acc_final:8.4f} {val_f1_final:11.4f}")
        print(f"  test  {test_loss_final:8.4f} {test_acc_final:8.4f} {test_f1_final:11.4f}\n")

        # Per-label confusion matrices (rows=true, cols=pred).
        final_val_conf = None
        final_val_true: List[int] = []
        final_val_pred: List[int] = []
        if has_validation and val_loader is not None:
            final_val_conf, final_val_true, final_val_pred = compute_confusion(
                global_model, val_loader, device, num_classes
            )
        final_test_conf, final_test_true, final_test_pred = compute_confusion(global_model, test_loader, device, num_classes)
        header = "          " + " ".join(f"{name:>6}" for name in label_names)
        if final_val_conf is not None:
            print(f"[{aggregator.name}] Final confusion matrix - Validation (rows=true, cols=pred):")
            print(header)
            for name, row in zip(label_names, final_val_conf):
                row_str = " ".join(f"{val:6d}" for val in row)
                print(f"{name:>8} {row_str}")
        print(f"\n[{aggregator.name}] Final confusion matrix - Test (rows=true, cols=pred):")
        print(header)
        for name, row in zip(label_names, final_test_conf):
            row_str = " ".join(f"{val:6d}" for val in row)
            print(f"{name:>8} {row_str}")
        print()

        final_test_conf_path = run_dir / f"{aggregator.name}_confusion_final_test_{args.num_clients}clients.csv"
        final_val_conf_path = None
        if final_val_conf is not None:
            final_val_conf_path = run_dir / f"{aggregator.name}_confusion_final_val_{args.num_clients}clients.csv"
            save_confusion_csv(final_val_conf, label_names, final_val_conf_path)
        save_confusion_csv(final_test_conf, label_names, final_test_conf_path)

        best_val_conf = None
        best_test_conf = None
        best_val_conf_path = None
        best_test_conf_path = None
        if best_state is not None:
            best_model = create_model(num_classes=len(label_names), use_pretrained=not args.no_pretrained)
            best_model.load_state_dict(best_state)
            best_model.to(device)
            if has_validation and val_loader is not None:
                best_val_conf, _, _ = compute_confusion(best_model, val_loader, device, num_classes)
            best_test_conf, _, _ = compute_confusion(best_model, test_loader, device, num_classes)

            if best_val_conf is not None:
                print(
                    f"[{aggregator.name}] Best confusion matrix (round {best_round}) - "
                    "Validation (rows=true, cols=pred):"
                )
                print(header)
                for name, row in zip(label_names, best_val_conf):
                    row_str = " ".join(f"{val:6d}" for val in row)
                    print(f"{name:>8} {row_str}")
            print(f"\n[{aggregator.name}] Best confusion matrix (round {best_round}) - Test (rows=true, cols=pred):")
            print(header)
            for name, row in zip(label_names, best_test_conf):
                row_str = " ".join(f"{val:6d}" for val in row)
                print(f"{name:>8} {row_str}")
            print()

            best_test_conf_path = run_dir / f"{aggregator.name}_confusion_best_test_{args.num_clients}clients.csv"
            if best_val_conf is not None:
                best_val_conf_path = run_dir / f"{aggregator.name}_confusion_best_val_{args.num_clients}clients.csv"
                save_confusion_csv(best_val_conf, label_names, best_val_conf_path)
            save_confusion_csv(best_test_conf, label_names, best_test_conf_path)

        # Save confusion and per-class metrics to text file.
        report_path = run_dir / f"{aggregator.name}_report_{args.num_clients}clients.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"Aggregator: {aggregator.name}\n")
            f.write(f"Rounds: {args.rounds}\n")
            f.write(f"Best test F1: {best_f1:.4f}\n\n")
            f.write("Final metrics:\n")
            f.write("           loss      acc      macro_f1\n")
            if has_validation:
                f.write(f"  val   {val_loss_final:8.4f} {val_acc_final:8.4f} {val_f1_final:11.4f}\n")
            f.write(f"  test  {test_loss_final:8.4f} {test_acc_final:8.4f} {test_f1_final:11.4f}\n\n")

            if final_val_conf is not None:
                f.write("Final validation confusion matrix (rows=true, cols=pred):\n")
                f.write(header + "\n")
                for name, row in zip(label_names, final_val_conf):
                    row_str = " ".join(f"{val:6d}" for val in row)
                    f.write(f"{name:>8} {row_str}\n")
                f.write("\n")
            f.write("Final test confusion matrix (rows=true, cols=pred):\n")
            f.write(header + "\n")
            for name, row in zip(label_names, final_test_conf):
                row_str = " ".join(f"{val:6d}" for val in row)
                f.write(f"{name:>8} {row_str}\n")

            if best_test_conf is not None:
                f.write(f"\nBest round: {best_round}\n")
                if best_val_conf is not None:
                    f.write("Best validation confusion matrix (rows=true, cols=pred):\n")
                    f.write(header + "\n")
                    for name, row in zip(label_names, best_val_conf):
                        row_str = " ".join(f"{val:6d}" for val in row)
                        f.write(f"{name:>8} {row_str}\n")
                f.write("\nBest test confusion matrix (rows=true, cols=pred):\n")
                f.write(header + "\n")
                for name, row in zip(label_names, best_test_conf):
                    row_str = " ".join(f"{val:6d}" for val in row)
                    f.write(f"{name:>8} {row_str}\n")

            if final_val_conf is not None:
                f.write("\nValidation per-class metrics:\n")
                f.write(
                    classification_report(
                        final_val_true,
                        final_val_pred,
                        labels=list(range(num_classes)),
                        target_names=label_names,
                        zero_division=0,
                    )
                )
                f.write("\n\n")
            f.write("Test per-class metrics:\n")
            f.write(
                classification_report(
                    final_test_true,
                    final_test_pred,
                    labels=list(range(num_classes)),
                    target_names=label_names,
                    zero_division=0,
                )
            )
        confusion_paths = [str(final_test_conf_path)]
        if final_val_conf_path is not None:
            confusion_paths.insert(0, str(final_val_conf_path))
        if best_val_conf_path is not None:
            confusion_paths.append(str(best_val_conf_path))
        if best_test_conf_path is not None:
            confusion_paths.append(str(best_test_conf_path))
        print(f"Saved confusion matrices to {', '.join(confusion_paths)}")
        print(f"Saved confusion matrices and per-class metrics to {report_path}\n")


if __name__ == "__main__":
    main()
