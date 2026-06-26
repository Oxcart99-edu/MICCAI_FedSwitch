# FedSwitch

**From Standard to Balanced Aggregation via Private Histogram Estimation in Federated Medical Image Classification**

*MICCAI 2026*

---

FedSwitch is a model-agnostic federated-learning framework for class-imbalanced medical image
classification. It estimates the global label histogram **once**, under client-level ε-differential
privacy, and uses that estimate to **switch** the server from standard `FedAvg` to
**distribution-aware aggregation** — without any client revealing its local label distribution and
without a trusted aggregator.

## How it works

1. **Warm-up (rounds `1 … R*−1`).** The server trains with ordinary FedAvg. In parallel, each
   sampled client bounds its label counts (ℓ₁ = `C`), adds partial discrete-Laplace noise (Pólya
   decomposition), and secret-shares the noisy counts to decryptors via Shamir's scheme. Shares
   accumulate across rounds at no extra privacy cost.

2. **Reconstruction (round `R*`).** Decryptors combine their accumulated shares and the server
   reconstructs the ε-DP global histogram in a single one-shot release — so privacy does not compose
   over rounds. The switch round `R* = ⌊ log(1−ρ) / log(1−m/N) ⌋` is the earliest round at which the
   target client coverage ρ is reached.

3. **Distribution-aware aggregation (rounds `R*+1 … T`).** The server replaces dataset-size weights
   with inverse-frequency weights derived from the estimated histogram, upweighting clients that
   hold globally rare classes.

## Repository

- `Histogram_estimation/` — the private ε-DP histogram estimation protocol and simulation drivers.
- `HAM10000/`, `bloodmnist/`, `organmnist_3d/` — federated trainers (FedSwitch + FedAvg, FedProx,
  FedIIC, FedLC baselines) on the three medical-imaging benchmarks.

The estimation step writes the histogram estimate to a JSON file that the trainers' FedSwitch
aggregator consumes. See each subdirectory's README for run instructions.
</content>
