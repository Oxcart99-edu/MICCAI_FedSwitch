"""
Distributed Discrete Laplace Noise for Heavy Hitters Protocol.

Based on Distributed DP (Kairouz et al. 2021):
- Each decryptor adds INDEPENDENT noise to their partial result
- Reconstruction: Σ(λ_i * share_i + noise_i) = secret + Σ(noise_i)
- The SUM of independent noises equals DLaplace(b)

Key insight: NegBin is infinitely divisible!
- DLaplace(b) = NegBin(1, p) - NegBin(1, p) where p = exp(-1/b)
- NegBin(1, p) = NegBin(1/n, p) + NegBin(1/n, p) + ... (n terms)
- So each decryptor samples: NegBin(1/n, p) - NegBin(1/n, p)
- Sum of n such samples = DLaplace(b) exactly!

No shared seed needed - fully independent sampling per decryptor!
"""

import numpy as np
from typing import Dict, List
from math import exp


def sample_partial_discrete_laplace(
    b: float,
    n_active_decryptors: int,
    rng: np.random.Generator
) -> int:
    """
    Sample one decryptor's share of Discrete Laplace noise.

    When n_active_decryptors independently sample from this and sum,
    the result is DLaplace(b).

    Uses: NegBin(1/n, 1-p) - NegBin(1/n, 1-p) where p = exp(-1/b)

    Args:
        b: Scale parameter (b = sensitivity/epsilon)
        n_active_decryptors: Number of active decryptors contributing noise
        rng: Numpy random generator (can be different per decryptor!)

    Returns:
        This decryptor's noise contribution
    """
    p = exp(-1.0 / b)
    geom_p = 1.0 - p

    # Each decryptor contributes 1/n of the noise
    # NegBin(r, p) with r = 1/n (non-integer r is valid in numpy)
    r = 1.0 / n_active_decryptors

    # Sample two NegBin(1/n, 1-p) and take difference
    y1 = rng.negative_binomial(r, geom_p)
    y2 = rng.negative_binomial(r, geom_p)

    return int(y1 - y2)


def generate_decryptor_noise(
    decryptor_seed: int,
    n_active_decryptors: int,
    n_labels: int,
    sensitivity: int,
    epsilon: float
) -> Dict[int, int]:
    """
    Generate this decryptor's independent noise contribution.

    Each decryptor calls this with their OWN seed (can be different!).
    The sum of all decryptors' noise = DLaplace(C/ε) for each label.

    Args:
        decryptor_seed: This decryptor's random seed (independent per decryptor)
        n_active_decryptors: Total number of active decryptors
        n_labels: Number of labels in histogram
        sensitivity: L1 sensitivity (clipping threshold C)
        epsilon: Privacy budget

    Returns:
        Dictionary mapping label_idx -> this decryptor's noise contribution
    """
    rng = np.random.default_rng(decryptor_seed)
    b = sensitivity / epsilon

    noise = {}
    for label_idx in range(n_labels):
        noise[label_idx] = sample_partial_discrete_laplace(b, n_active_decryptors, rng)

    return noise


class DistributedNoiseGenerator:
    """
    Distributed noise generator for a single decryptor.

    Each decryptor creates their own instance with their own seed.
    No coordination needed - each samples independently!

    Example:
        # Each decryptor has their own generator with their own seed
        dec1_gen = DistributedNoiseGenerator(seed=111, ...)
        dec2_gen = DistributedNoiseGenerator(seed=222, ...)

        # Each generates their noise share independently
        noise1 = dec1_gen.generate_noise(n_active=3)
        noise2 = dec2_gen.generate_noise(n_active=3)
        noise3 = dec3_gen.generate_noise(n_active=3)

        # Sum gives DLaplace(b) per label!
        total_noise = {k: noise1[k] + noise2[k] + noise3[k] for k in noise1}
    """

    def __init__(
        self,
        seed: int,
        n_labels: int,
        sensitivity: int,
        epsilon: float
    ):
        self.seed = seed
        self.n_labels = n_labels
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.b = sensitivity / epsilon
        self.rng = np.random.default_rng(seed)

    def generate_noise(self, n_active_decryptors: int) -> Dict[int, int]:
        """
        Generate this decryptor's noise contribution.

        Args:
            n_active_decryptors: Number of active decryptors

        Returns:
            Noise values for each label
        """
        noise = {}
        for label_idx in range(self.n_labels):
            noise[label_idx] = sample_partial_discrete_laplace(
                self.b, n_active_decryptors, self.rng
            )
        return noise


# For backward compatibility with protocol.py
def run_distributed_noise_protocol(
    n_decryptors: int,
    n_labels: int,
    sensitivity: int,
    epsilon: float,
    active_decryptor_ids: List[int],
    seed: int = None
) -> Dict[int, int]:
    """
    Simulate the distributed noise protocol.

    In the real protocol:
    - Each decryptor generates their noise share independently
    - Each adds it to their partial result (λ_i * share_i + noise_i)
    - Coordinator sums to get: secret + Σ(noise_i) = secret + DLaplace(b)

    This function simulates the SUM of all decryptors' noise contributions,
    which equals DLaplace(b) per label.

    Args:
        n_decryptors: Total number of decryptors (unused, for compatibility)
        n_labels: Number of labels
        sensitivity: L1 sensitivity (C)
        epsilon: Privacy budget
        active_decryptor_ids: IDs of active decryptors
        seed: Base seed for reproducibility

    Returns:
        Dictionary of label_idx -> total noise (sum of all decryptors' contributions)
    """
    if seed is None:
        seed = 42

    n_active = len(active_decryptor_ids)
    b = sensitivity / epsilon

    # Simulate each decryptor generating their noise share independently
    # and sum them (this is what happens after reconstruction)
    total_noise = {label_idx: 0 for label_idx in range(n_labels)}

    for i, dec_id in enumerate(active_decryptor_ids):
        # Each decryptor has their own seed (derived from base seed + their ID)
        dec_seed = seed + dec_id * 1000 + i
        dec_rng = np.random.default_rng(dec_seed)

        for label_idx in range(n_labels):
            noise_share = sample_partial_discrete_laplace(b, n_active, dec_rng)
            total_noise[label_idx] += noise_share

    return total_noise


def verify_distributed_noise():
    """
    Verify that sum of partial noises equals DLaplace distribution.
    """
    print("Verifying distributed DP noise property...")

    n_decryptors = 5
    n_labels = 3
    sensitivity = 100
    epsilon = 1.0
    b = sensitivity / epsilon
    n_samples = 10000

    # Sample full DLaplace for comparison
    rng_full = np.random.default_rng(42)
    p = exp(-1.0 / b)
    full_samples = []
    for _ in range(n_samples):
        y1 = rng_full.negative_binomial(1, 1-p)
        y2 = rng_full.negative_binomial(1, 1-p)
        full_samples.append(y1 - y2)

    full_mean = np.mean(full_samples)
    full_var = np.var(full_samples)

    # Sample sum of partial noises
    sum_samples = []
    for sample_idx in range(n_samples):
        total = 0
        for dec_id in range(n_decryptors):
            dec_rng = np.random.default_rng(sample_idx * 1000 + dec_id)
            partial = sample_partial_discrete_laplace(b, n_decryptors, dec_rng)
            total += partial
        sum_samples.append(total)

    sum_mean = np.mean(sum_samples)
    sum_var = np.var(sum_samples)

    print(f"  Full DLaplace(b={b}): mean={full_mean:.3f}, var={full_var:.1f}")
    print(f"  Sum of {n_decryptors} partial noises: mean={sum_mean:.3f}, var={sum_var:.1f}")
    print(f"  Theoretical: mean=0, var={2*p/(1-p)**2:.1f}")

    # Check they're close
    assert abs(sum_mean) < 5, f"Mean too far from 0: {sum_mean}"
    assert abs(sum_var - full_var) / full_var < 0.1, f"Variance mismatch: {sum_var} vs {full_var}"

    print("  ✓ Distributed noise sums to correct DLaplace distribution!")


if __name__ == "__main__":
    verify_distributed_noise()
