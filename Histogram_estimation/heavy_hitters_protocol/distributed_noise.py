"""
Distributed Discrete Laplace Noise via Pólya random variables.

Based on Bagdasaryan et al. (PoPETs 2022), Theorem 4.1:
  A Pólya(a, β) random variable X has PMF:
      P(X = k) = C(a+k-1, a-1) · β^k · (1 - β)^a

  If S participants each independently sample X_i, Y_i ~ iid Pólya(1/S, β)
  with β = exp(-ε/C), then Z = Σ(X_i - Y_i) ~ DLap(C/ε),
  achieving ε-DP for queries with ℓ1-sensitivity C.

In our protocol, **clients** (not decryptors) add the noise before secret sharing.
Each client divides by the target cohort size |A| = αN.
"""

import numpy as np
from math import exp, log, ceil, floor
from typing import Dict


def sample_polya(a: float, beta: float, rng: np.random.Generator) -> int:
    """
    Sample from a Pólya(a, β) distribution.

    Pólya(a, β) is equivalent to NegBin(a, 1 - β) in numpy's parametrisation,
    where numpy's negative_binomial(n, p) counts failures before n successes
    with success probability p.

    Args:
        a: Shape parameter (1/|A| in our protocol).
        beta: e^{-ε/C}.
        rng: Numpy random generator.

    Returns:
        Non-negative integer sample.
    """
    # numpy NegBin(n, p): p = success prob = 1 - β
    return int(rng.negative_binomial(a, 1.0 - beta))


def sample_partial_noise(
    n_contributors: int,
    sensitivity: int,
    epsilon: float,
    n_labels: int,
    rng: np.random.Generator,
) -> Dict[int, int]:
    """
    Sample one contributor's share of DLap(C/ε) noise for all labels.

    Each contributor samples η = X⁺ - X⁻ with X± ~ iid Pólya(1/S, β),
    β = exp(-ε/C), S = n_contributors.
    When S contributors sum their η values, the result is DLap(C/ε) per label.

    Args:
        n_contributors: Target number of contributors (|A|).
        sensitivity: ℓ1-sensitivity C.
        epsilon: Privacy budget ε.
        n_labels: Number of labels k.
        rng: Numpy random generator.

    Returns:
        Dict mapping label_idx -> noise value η.
    """
    beta = exp(-epsilon / sensitivity)
    a = 1.0 / n_contributors

    noise = {}
    for label_idx in range(n_labels):
        x_plus = sample_polya(a, beta, rng)
        x_minus = sample_polya(a, beta, rng)
        noise[label_idx] = x_plus - x_minus
    return noise


def compute_trigger_round(alpha: float, gamma: float, n_clients: int) -> int:
    """
    Compute trigger round R* = floor(log(1-α) / log(1 - m/N)).

    Args:
        alpha: Target coverage fraction (e.g. 0.9).
        gamma: Fraction of clients sampled per round (m/N).
        n_clients: Total number of clients N.

    Returns:
        Trigger round R*.
    """
    m = max(1, int(gamma * n_clients))
    ratio = m / n_clients
    if ratio >= 1.0:
        return 1
    return max(1, floor(log(1.0 - alpha) / log(1.0 - ratio)))


def compute_target_cohort(alpha: float, n_clients: int) -> int:
    """
    Compute target cohort size |A| = ceil(α·N).

    Args:
        alpha: Target coverage fraction.
        n_clients: Total number of clients N.

    Returns:
        Target cohort size |A|.
    """
    return max(1, ceil(alpha * n_clients))


def compute_dp_variance(alpha: float, epsilon: float, sensitivity: int) -> float:
    """
    Compute DP noise variance σ²_DP = α · 2β / (1 - β)².

    This is the variance from |A| = αN client-side Pólya contributions.
    When α = 1 this equals the full DLap(C/ε) variance.

    Args:
        alpha: Coverage fraction.
        epsilon: Privacy budget.
        sensitivity: ℓ1-sensitivity C.

    Returns:
        Noise variance σ²_DP.
    """
    beta = exp(-epsilon / sensitivity)
    return alpha * 2.0 * beta / ((1.0 - beta) ** 2)
