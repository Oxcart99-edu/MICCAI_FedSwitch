# OrganMNIST3D

This folder contains the code and assets for centralized and federated training on `OrganMNIST3D`.

## Contents

- `scripts/train.py`: self-contained centralized trainer based on 3D `r3d_18`, with optional anatomical label merging.
- `scripts/train_federated.py`: federated training loop using the local `scripts/aggregators/` package.
- `scripts/aggregators/`: local implementations of `fedavg`, `fedprox`, `fedlc`, `fedswitch`, and `fediic`.
- `sample_distribution/`: client distribution CSVs and round-selection JSON files used by the federated loop.
- `requirements.txt`: runtime dependencies.
- `data/organmnist3d.npz`: local dataset file used by both trainers.

## Setup

```bash
cd organmnist_3d
python -m pip install -r requirements.txt
```

## Centralized Run

```bash
python scripts/train.py --epochs 1 --device cpu
```

## Federated Run

```bash
python scripts/train_federated.py --rounds 2 --local-epochs 1 --aggregators fedavg --device cpu
```

## Notes

- Both trainers read directly from `data/organmnist3d.npz`, so `medmnist` is not required.
- The default batch size is `8`.
- The 3D ResNet-18 is not pretrained.
- Anatomical merging produces 8 classes: `liver`, `kidney`, `femur`, `bladder`, `heart`, `lung`, `spleen`, `pancreas`.
- To use the original 11 OrganMNIST3D classes instead:

```bash
python scripts/train.py --no-merge-anatomical
python scripts/train_federated.py --no-merge-anatomical
```

- The federated loop uses the files in `sample_distribution/` by default.
