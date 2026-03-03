from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregators import AGGREGATOR_REGISTRY, get_aggregator
from train import (
    NpzVolumeDataset,
    RAW_LABEL_NAMES,
    ToTensor3D,
    build_anatomical_label_map,
    build_fediic_model,
    build_model,
    seed_everything,
)

BASE_DIR = PROJECT_ROOT


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = [int(idx) for idx in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        source_idx = self.indices[idx]
        image, label = self.dataset[source_idx]
        return image, label, source_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated training for OrganMNIST3D")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--steps-per-round", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=50)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--clients-label-alpha-dir", type=Path, default=BASE_DIR / "sample_distribution")
    parser.add_argument("--clients-label-csvs", nargs="*", default=None)
    parser.add_argument(
        "--picked-ids-json",
        type=Path,
        default=BASE_DIR / "sample_distribution" / "picked_ids_1000_rounds_seed42.json",
    )
    parser.add_argument(
        "--aggregators",
        nargs="+",
        default=["fedswitch"],
        choices=sorted(AGGREGATOR_REGISTRY.keys()),
    )
    parser.add_argument("--all-aggregators", action="store_true")
    parser.add_argument("--switch-round", type=int, default=None)
    parser.add_argument(
        "--fedswitch-weight-source",
        type=str,
        default="client",
        choices=["client", "steps"],
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--fedlc-tau", type=float, default=0.5)
    parser.add_argument("--fediic-k1", type=float, default=2.0)
    parser.add_argument("--fediic-k2", type=float, default=2.0)
    parser.add_argument("--fediic-difficulty", type=float, default=0.25)
    parser.add_argument("--fediic-tau", type=float, default=1.0)
    parser.add_argument("--fediic-projector-dim", type=int, default=256)
    parser.add_argument("--fediic-prototype-opt-steps", type=int, default=100)
    parser.add_argument("--fediic-prototype-opt-lr", type=float, default=0.1)
    parser.add_argument("--fediic-prototype-every-round", action="store_true", default=True)
    parser.add_argument("--no-fediic-prototype-every-round", dest="fediic_prototype_every_round", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.add_argument("--verbose-round-details", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs")
    parser.add_argument(
        "--merge-anatomical",
        dest="merge_anatomical",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-merge-anatomical", dest="merge_anatomical", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    return parser.parse_args()


def parse_scenario_alpha(csv_path: Path) -> str:
    match = re.search(r"alpha_([0-9.]+)", csv_path.stem)
    return match.group(1) if match else csv_path.stem


def discover_clients_label_csvs(args: argparse.Namespace) -> List[Path]:
    if args.clients_label_csvs:
        paths = [Path(path).expanduser().resolve() for path in args.clients_label_csvs]
    else:
        root = args.clients_label_alpha_dir.expanduser().resolve()
        paths = sorted(root.glob("clients_label_alpha_*.csv"))
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError(
            "No scenario CSV found. Use --clients-label-csvs or populate sample_distribution/."
        )
    return sorted(existing, key=lambda path: float(parse_scenario_alpha(path)))


def load_picked_ids_schedule(
    *,
    json_path: Path,
    rounds: int,
    clients_per_round: int,
    num_clients: int,
) -> List[List[int]]:
    schedule_file = json_path.expanduser().resolve()
    if not schedule_file.exists():
        raise FileNotFoundError(f"Picked-ids JSON not found: {schedule_file}")

    payload = json.loads(schedule_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Picked-ids JSON {schedule_file} must contain a list.")

    round_to_ids: Dict[int, List[int]] = {}
    for entry in payload:
        round_idx = int(entry["round"])
        picked_ids = [int(client_id) for client_id in entry["picked_ids"]]
        if len(picked_ids) != clients_per_round:
            raise ValueError(
                f"Round {round_idx} in {schedule_file} has {len(picked_ids)} clients, expected {clients_per_round}."
            )
        if len(set(picked_ids)) != len(picked_ids):
            raise ValueError(f"Round {round_idx} in {schedule_file} contains duplicate client ids.")
        invalid_ids = [client_id for client_id in picked_ids if client_id < 0 or client_id >= num_clients]
        if invalid_ids:
            raise ValueError(f"Round {round_idx} in {schedule_file} contains invalid client ids: {invalid_ids}")
        round_to_ids[round_idx] = picked_ids

    missing_rounds = [round_idx for round_idx in range(1, rounds + 1) if round_idx not in round_to_ids]
    if missing_rounds:
        raise ValueError(f"Schedule {schedule_file} is missing rounds: {missing_rounds[:10]}")
    return [round_to_ids[round_idx] for round_idx in range(1, rounds + 1)]


def remap_label_array(labels: np.ndarray, label_map: Dict[int, int]) -> np.ndarray:
    return np.asarray([label_map[int(label)] for label in labels.reshape(-1).tolist()], dtype=np.int64)


def load_organmnist3d_data(
    *,
    data_dir: Path,
    merge_anatomical: bool,
) -> Tuple[Dataset, Dataset, np.ndarray, List[str], int]:
    npz_path = data_dir.expanduser().resolve() / "organmnist3d.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    if merge_anatomical:
        raw_to_merged, merged_id_to_name = build_anatomical_label_map(RAW_LABEL_NAMES)
    else:
        raw_to_merged = {idx: idx for idx in RAW_LABEL_NAMES}
        merged_id_to_name = {idx: name for idx, name in sorted(RAW_LABEL_NAMES.items())}

    payload = np.load(npz_path)
    train_val_images = np.concatenate([payload["train_images"], payload["val_images"]], axis=0)
    train_val_raw_labels = np.concatenate([payload["train_labels"], payload["val_labels"]], axis=0)
    test_images = payload["test_images"]
    test_raw_labels = payload["test_labels"]

    train_val_labels = remap_label_array(train_val_raw_labels, raw_to_merged)
    class_names = [merged_id_to_name[idx] for idx in sorted(merged_id_to_name.keys())]
    transform = ToTensor3D()
    train_dataset = NpzVolumeDataset(train_val_images, train_val_raw_labels, transform, raw_to_merged)
    test_dataset = NpzVolumeDataset(test_images, test_raw_labels, transform, raw_to_merged)
    return train_dataset, test_dataset, train_val_labels, class_names, len(class_names)


def allocate_from_sample_distribution_csv(
    *,
    train_labels: np.ndarray,
    num_classes: int,
    num_clients: int,
    csv_path: Path,
    rng: random.Random,
) -> Tuple[List[List[int]], List[int], List[Dict[int, int]]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Scenario CSV is empty: {csv_path}")

    by_client_id: Dict[int, Dict[str, str]] = {}
    for row in rows:
        client_id = int(row["client_id"])
        if client_id in by_client_id:
            raise ValueError(f"Duplicate client_id={client_id} in {csv_path}")
        by_client_id[client_id] = row

    expected_ids = set(range(num_clients))
    actual_ids = set(by_client_id.keys())
    if actual_ids != expected_ids:
        raise ValueError(
            f"CSV {csv_path} client ids mismatch. missing={sorted(expected_ids - actual_ids)} extra={sorted(actual_ids - expected_ids)}"
        )

    label_to_indices: Dict[int, List[int]] = {}
    for class_idx in range(num_classes):
        indices = np.where(train_labels == class_idx)[0].tolist()
        rng.shuffle(indices)
        label_to_indices[class_idx] = indices

    client_bins: List[List[int]] = [[] for _ in range(num_clients)]
    client_sizes = [0 for _ in range(num_clients)]
    client_counts: List[Dict[int, int]] = [{class_idx: 0 for class_idx in range(num_classes)} for _ in range(num_clients)]

    for class_idx in range(num_classes):
        count_col = f"count_{class_idx}"
        requests = [int(by_client_id[client_id][count_col]) for client_id in range(num_clients)]
        available_total = len(label_to_indices[class_idx])
        requested_total = sum(requests)
        if requested_total != available_total:
            raise ValueError(
                f"Class {class_idx} mismatch in {csv_path}: requested={requested_total}, available={available_total}"
            )

        cursor = 0
        for client_id, count in enumerate(requests):
            if count <= 0:
                continue
            next_cursor = cursor + count
            chunk = label_to_indices[class_idx][cursor:next_cursor]
            if len(chunk) != count:
                raise ValueError(
                    f"Insufficient samples for class {class_idx} while filling client {client_id}: expected {count}, got {len(chunk)}"
                )
            client_bins[client_id].extend(chunk)
            client_sizes[client_id] += count
            client_counts[client_id][class_idx] += count
            cursor = next_cursor

    for client_id, size in enumerate(client_sizes):
        if size <= 0:
            raise ValueError(f"Scenario {csv_path} produced empty client {client_id}")

    return client_bins, client_sizes, client_counts


def build_client_loaders(
    *,
    train_dataset: Dataset,
    client_bins: List[List[int]],
    client_sizes: List[int],
    client_counts: List[Dict[int, int]],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> List[Tuple[DataLoader, int, Dict[int, int]]]:
    client_loaders: List[Tuple[DataLoader, int, Dict[int, int]]] = []
    for client_idx in range(len(client_bins)):
        subset = IndexedSubset(train_dataset, client_bins[client_idx])
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        client_loaders.append((loader, client_sizes[client_idx], client_counts[client_idx]))
    return client_loaders


def _can_use_multiprocess_dataloader(num_workers: int) -> tuple[bool, str]:
    if num_workers <= 0:
        return True, ""
    try:
        probe_images = torch.zeros((8, 1, 8, 8, 8), dtype=torch.float32)
        probe_labels = torch.zeros(8, dtype=torch.long)
        probe_ds = torch.utils.data.TensorDataset(probe_images, probe_labels)
        probe_loader = DataLoader(probe_ds, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=False)
        next(iter(probe_loader))
    except Exception as exc:
        return False, str(exc)
    return True, ""


def compute_confusion_matrix(labels: Sequence[int], preds: Sequence[int], num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, pred_label in zip(labels, preds):
        matrix[int(true_label)][int(pred_label)] += 1
    return matrix


def macro_f1_from_confusion(matrix: Sequence[Sequence[int]]) -> Tuple[float, List[Dict[str, float]]]:
    num_classes = len(matrix)
    per_class: List[Dict[str, float]] = []
    f1_values: List[float] = []

    for class_idx in range(num_classes):
        tp = float(matrix[class_idx][class_idx])
        fp = float(sum(matrix[row][class_idx] for row in range(num_classes) if row != class_idx))
        fn = float(sum(matrix[class_idx][col] for col in range(num_classes) if col != class_idx))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = float(sum(matrix[class_idx]))
        per_class.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        f1_values.append(f1)

    return sum(f1_values) / max(len(f1_values), 1), per_class


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
) -> Tuple[float, float, float, List[List[int]], List[Dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1).long()
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    if total_samples == 0:
        return float("nan"), float("nan"), float("nan"), compute_confusion_matrix([], [], num_classes), []

    confusion = compute_confusion_matrix(all_labels, all_preds, num_classes)
    macro_f1, per_class = macro_f1_from_confusion(confusion)
    accuracy = sum(int(pred == label) for pred, label in zip(all_preds, all_labels)) / total_samples
    return total_loss / total_samples, accuracy, macro_f1, confusion, per_class


def save_metrics_csv(rows: Sequence[Dict[str, float | int]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_confusion_csv(confusion: Sequence[Sequence[int]], class_names: Sequence[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *class_names])
        for class_name, row in zip(class_names, confusion):
            writer.writerow([class_name, *row])


def write_report(
    *,
    path: Path,
    dataset_name: str,
    scenario_csv: Path,
    aggregator_name: str,
    rounds: int,
    best_round: Optional[int],
    best_test_f1: float,
    class_names: Sequence[str],
    final_confusion: Sequence[Sequence[int]],
    final_per_class: Sequence[Dict[str, float]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"Dataset: {dataset_name}\n")
        handle.write(f"Scenario CSV: {scenario_csv}\n")
        handle.write(f"Aggregator: {aggregator_name}\n")
        handle.write(f"Rounds: {rounds}\n")
        handle.write(f"Best test F1: {best_test_f1:.4f}\n")
        handle.write(f"Best round: {best_round}\n\n")
        handle.write("Final test confusion matrix (rows=true, cols=pred):\n")
        header = "          " + " ".join(f"{name:>10}" for name in class_names)
        handle.write(header + "\n")
        for class_name, row in zip(class_names, final_confusion):
            row_str = " ".join(f"{value:10d}" for value in row)
            handle.write(f"{class_name:>10} {row_str}\n")
        handle.write("\nPer-class metrics:\n")
        for class_name, stats in zip(class_names, final_per_class):
            handle.write(
                f"{class_name}: precision={stats['precision']:.4f} recall={stats['recall']:.4f} "
                f"f1={stats['f1']:.4f} support={int(stats['support'])}\n"
            )


def format_label_counts(counts: Dict[int, int], class_names: Sequence[str]) -> str:
    return ", ".join(f"{class_names[idx]}:{int(counts.get(idx, 0))}" for idx in range(len(class_names)))


def main() -> None:
    args = parse_args()
    if args.all_aggregators:
        args.aggregators = sorted(AGGREGATOR_REGISTRY.keys())
    if args.clients_per_round > args.num_clients:
        raise ValueError("--clients-per-round must be <= --num-clients")

    partition_seed = args.partition_seed if args.partition_seed is not None else args.seed
    model_seed = args.model_seed if args.model_seed is not None else args.seed
    seed_everything(model_seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    pin_memory = device.type == "cuda"

    if args.num_workers > 0:
        workers_ok, reason = _can_use_multiprocess_dataloader(args.num_workers)
        if not workers_ok:
            print(f"Multiprocess DataLoader unavailable ({reason}). Falling back to --num-workers=0.")
            args.num_workers = 0

    train_dataset, test_dataset, train_labels, class_names, num_classes = load_organmnist3d_data(
        data_dir=args.data_dir,
        merge_anatomical=args.merge_anatomical,
    )
    print(f"Using {num_classes} classes on device={device}")
    for class_idx, class_name in enumerate(class_names):
        print(f"  [{class_idx}] {class_name}")

    scenario_csvs = discover_clients_label_csvs(args)
    schedule = load_picked_ids_schedule(
        json_path=args.picked_ids_json,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        num_clients=args.num_clients,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    criterion = nn.CrossEntropyLoss()

    run_root = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    for scenario_csv in scenario_csvs:
        scenario_alpha = parse_scenario_alpha(scenario_csv)
        scenario_dir = run_root / f"alpha_{scenario_alpha}_seed{model_seed}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(partition_seed)
        client_bins, client_sizes_all, client_counts_all = allocate_from_sample_distribution_csv(
            train_labels=train_labels,
            num_classes=num_classes,
            num_clients=args.num_clients,
            csv_path=scenario_csv,
            rng=rng,
        )
        client_loaders = build_client_loaders(
            train_dataset=train_dataset,
            client_bins=client_bins,
            client_sizes=client_sizes_all,
            client_counts=client_counts_all,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        print(f"\n=== Scenario alpha={scenario_alpha} ({scenario_csv.name}) ===")
        for client_id, (size, counts) in enumerate(zip(client_sizes_all, client_counts_all)):
            print(f"  Client {client_id}: {format_label_counts(counts, class_names)} (total {size})")

        for aggregator_name in args.aggregators:
            aggregator = get_aggregator(aggregator_name)
            if aggregator.name == "fedswitch":
                aggregator.configure(
                    rounds=args.rounds,
                    steps_per_round=args.steps_per_round,
                    switch_round=args.switch_round,
                    switch_weight_source=args.fedswitch_weight_source,
                    fedprox_mu=args.fedprox_mu,
                    learning_rate=args.learning_rate,
                    fedlc_tau=args.fedlc_tau,
                    scenario_alpha=scenario_alpha,
                    estimate_json_dir=str(args.clients_label_alpha_dir),
                )
            elif aggregator.name == "fediic":
                client_eval_loaders = [
                    DataLoader(
                        IndexedSubset(train_dataset, client_bins[client_id]),
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=pin_memory,
                    )
                    for client_id in range(len(client_bins))
                ]
                aggregator.configure(
                    rounds=args.rounds,
                    steps_per_round=args.steps_per_round,
                    switch_round=args.switch_round,
                    switch_weight_source=args.fedswitch_weight_source,
                    fedprox_mu=args.fedprox_mu,
                    learning_rate=args.learning_rate,
                    fedlc_tau=args.fedlc_tau,
                    fediic_k1=args.fediic_k1,
                    fediic_k2=args.fediic_k2,
                    fediic_difficulty=args.fediic_difficulty,
                    fediic_tau=args.fediic_tau,
                    fediic_projector_dim=args.fediic_projector_dim,
                    fediic_prototype_opt_steps=args.fediic_prototype_opt_steps,
                    fediic_prototype_opt_lr=args.fediic_prototype_opt_lr,
                    fediic_prototype_every_round=args.fediic_prototype_every_round,
                    client_eval_loaders=client_eval_loaders,
                    client_class_counts_all=client_counts_all,
                    num_classes=num_classes,
                    device=device,
                )
            else:
                aggregator.configure(
                    rounds=args.rounds,
                    steps_per_round=args.steps_per_round,
                    switch_round=args.switch_round,
                    switch_weight_source=args.fedswitch_weight_source,
                    fedprox_mu=args.fedprox_mu,
                    learning_rate=args.learning_rate,
                    fedlc_tau=args.fedlc_tau,
                )

            run_dir = scenario_dir / aggregator.name
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- Running aggregator: {aggregator.name} ---")
            note = aggregator.experiment_note()
            if note:
                print(note)

            if aggregator.name == "fediic":
                model_builder = lambda: build_fediic_model(  # noqa: E731
                    num_classes=num_classes,
                    projector_dim=args.fediic_projector_dim,
                )
            else:
                model_builder = lambda: build_model(num_classes=num_classes)  # noqa: E731

            global_model = model_builder().to(device)
            client_model = model_builder().to(device)
            aggregator.on_experiment_start(global_model, args.num_clients)

            best_f1 = -1.0
            best_round: Optional[int] = None
            best_state: Optional[Dict[str, torch.Tensor]] = None
            metrics_rows: List[Dict[str, float | int]] = []
            metrics_csv_path = run_dir / f"{aggregator.name}_metrics.csv"
            checkpoint_path = run_dir / f"best_{aggregator.name}_alpha_{scenario_alpha}.pt"

            for round_idx in range(1, args.rounds + 1):
                aggregator.on_round_start(round_idx)
                if aggregator.name == "fediic":
                    aggregator.prepare_round(global_model)
                selected_client_ids = schedule[round_idx - 1]
                selected_client_sizes = [client_sizes_all[client_id] for client_id in selected_client_ids]
                selected_client_counts = [client_counts_all[client_id] for client_id in selected_client_ids]
                global_named_params = {
                    name: param.detach().clone()
                    for name, param in global_model.named_parameters()
                    if param.requires_grad
                }
                client_states: List[Dict[str, torch.Tensor]] = []
                round_train_losses: List[float] = []
                step_seen_sizes: List[int] = []
                step_seen_label_counts: List[Dict[int, int]] = []

                for client_id in selected_client_ids:
                    loader, _, _ = client_loaders[client_id]
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
                        use_amp=args.use_amp,
                        client_class_counts=client_counts_all[client_id],
                        num_classes=num_classes,
                        collect_seen_examples=args.verbose_round_details,
                    )
                    step_seen_sizes.append(local_result.seen_samples)
                    step_seen_label_counts.append(local_result.seen_label_counts)
                    client_states.append({name: tensor.detach().cpu() for name, tensor in client_model.state_dict().items()})
                    round_train_losses.append(local_result.train_loss)

                round_plan = aggregator.resolve_round_aggregation(
                    round_idx=round_idx,
                    selected_client_sizes=selected_client_sizes,
                    selected_client_counts=selected_client_counts,
                    step_seen_sizes=step_seen_sizes,
                    step_seen_label_counts=step_seen_label_counts,
                    num_classes=num_classes,
                )
                if args.verbose_round_details:
                    print(
                        f"[{aggregator.name}] round={round_idx:04d}/{args.rounds}{round_plan.phase_suffix} "
                        f"selected_clients={selected_client_ids} weights={round_plan.client_weights}"
                    )

                averaged_state = aggregator.aggregate(
                    client_states,
                    round_plan.client_sizes,
                    class_counts=round_plan.class_counts,
                    num_classes=num_classes,
                    client_weights=round_plan.client_weights,
                )
                global_model.load_state_dict(averaged_state)
                aggregator.on_global_model_updated(args.num_clients)

                test_loss, test_acc, test_f1, _, _ = evaluate_model(
                    global_model,
                    test_loader,
                    device,
                    criterion,
                    num_classes,
                )
                mean_train_loss = sum(round_train_losses) / max(len(round_train_losses), 1)
                print(
                    f"[{aggregator.name}] round={round_idx:04d}/{args.rounds}{round_plan.phase_suffix} "
                    f"local_train_loss={mean_train_loss:.4f} test_loss={test_loss:.4f} "
                    f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
                )

                metrics_rows.append(
                    {
                        "round": round_idx,
                        "local_train_loss": mean_train_loss,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "test_f1": test_f1,
                    }
                )

                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_round = round_idx
                    best_state = {name: tensor.detach().cpu().clone() for name, tensor in global_model.state_dict().items()}
                    torch.save(
                        {
                            "model_state": best_state,
                            "label_names": {idx: class_names[idx] for idx in range(num_classes)},
                            "num_classes": num_classes,
                            "test_f1": test_f1,
                            "round": round_idx,
                            "aggregator": aggregator.name,
                            "scenario_csv": str(scenario_csv),
                            "picked_ids_json": str(args.picked_ids_json),
                        },
                        checkpoint_path,
                    )

            save_metrics_csv(metrics_rows, metrics_csv_path)
            final_test_loss, final_test_acc, final_test_f1, final_confusion, final_per_class = evaluate_model(
                global_model,
                test_loader,
                device,
                criterion,
                num_classes,
            )
            final_confusion_path = run_dir / f"{aggregator.name}_confusion_final.csv"
            save_confusion_csv(final_confusion, class_names, final_confusion_path)

            best_confusion_path = run_dir / f"{aggregator.name}_confusion_best.csv"
            if best_state is not None:
                best_model = model_builder().to(device)
                best_model.load_state_dict(best_state)
                _, _, _, best_confusion, _ = evaluate_model(best_model, test_loader, device, criterion, num_classes)
                save_confusion_csv(best_confusion, class_names, best_confusion_path)

            report_path = run_dir / f"{aggregator.name}_report.txt"
            write_report(
                path=report_path,
                dataset_name="organmnist3d",
                scenario_csv=scenario_csv,
                aggregator_name=aggregator.name,
                rounds=args.rounds,
                best_round=best_round,
                best_test_f1=best_f1,
                class_names=class_names,
                final_confusion=final_confusion,
                final_per_class=final_per_class,
            )
            print(
                f"Saved outputs to {run_dir} "
                f"(best_f1={best_f1:.4f}, final_test_loss={final_test_loss:.4f}, "
                f"final_test_acc={final_test_acc:.4f}, final_test_f1={final_test_f1:.4f})"
            )


if __name__ == "__main__":
    main()
