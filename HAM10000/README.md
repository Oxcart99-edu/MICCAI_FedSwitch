# HAM10000

This directory contains code, data, and a small set of example images for experiments on the HAM10000 skin lesion dataset.

## Structure

- `data/`
  Dataset files.
  Contains:
  `HAM10000_metadata.csv`, `HAM10000_images_part_1/`, `HAM10000_images_part_2/`, and the HAM10000 CSV exports (`hmnist_8_8_*`, `hmnist_28_28_*`).

- `scripts/`
  Training and utility code.
  Main entry points:
  `train_resnet50.py` for centralized training and `train_resnet50_federated.py` for federated training.

- `scripts/aggregators/`
  Federated aggregation methods such as FedAvg, FedProx, SCAFFOLD, stratified variants, FedLC, and FedIIC.

- `examples/`
  One sample image per class for quick visual inspection.

- `sample_distribution/`
  Precomputed client-distribution files for federated experiments, including per-client label-count CSV files and a picked-client schedule JSON.
  `train_resnet50_federated.py` reads client assignments only from this folder.

- `requirements.txt`
  Minimal Python dependencies needed by the code in this directory.

## Notes

- Federated runs no longer simulate client distributions inside the training script. Use the CSV and JSON files under `sample_distribution/`.
