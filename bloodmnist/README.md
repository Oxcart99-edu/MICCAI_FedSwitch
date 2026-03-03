# BloodMNIST

This directory contains:

- `scripts/aggregators/` for federated aggregation logic
- `scripts/` for centralized and federated training entrypoints
- a `ResNet-18 CIFAR` backbone
- the local dataset in `dataset/bloodmnist.npz`
- client split assets in `client_selection/`
- dependencies listed in `requirements.txt`

## Quick Start

Run these commands from `bloodmnist/`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/train.py
```

Outputs are written to `results/`:

- `model.pt`
- `metrics.json`

## Federated Training

Base example:

```bash
python scripts/federated_train.py --rounds 100
```

`FedProx` example:

```bash
python scripts/federated_train.py --aggregator fedprox --rounds 100 --fedprox-mu 0.01
```

`fedswitch` example:

```bash
python scripts/federated_train.py \
  --aggregator fedswitch \
  --clients-label-csv client_selection/clients_label_alpha_0.5.csv \
  --rounds 100 \
  --steps-per-round 20 \
  --switch-round 10
```

`fediic` example:

```bash
python scripts/federated_train.py \
  --aggregator fediic \
  --clients-label-csv client_selection/clients_label_alpha_0.5.csv \
  --rounds 100 \
  --clients-per-round 10
```

Federated outputs are written to `results/federated/`.

## Layout

```text
bloodmnist/
├── dataset/
│   └── bloodmnist.npz
├── client_selection/
├── pyproject.toml
├── README.md
├── requirements.txt
└── scripts/
    ├── aggregators/
    │   ├── base.py
    │   ├── fedavg.py
    │   ├── fediic.py
    │   ├── fedswitch.py
    │   └── fedprox.py
    ├── federated_train.py
    └── train.py
```
