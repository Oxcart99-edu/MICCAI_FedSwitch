"""
Cryptographic primitives: Shamir Secret Sharing.

This implementation uses standard Shamir Secret Sharing without the Boneh factorial
technique for simplicity, as we're working with a large prime field where modular
inverse computation is straightforward.
"""

import secrets
from typing import List, Tuple
from math import ceil


def generate_large_prime(bits: int = 256) -> int:
    """
    Generate a large prime number with the specified number of bits.
    Uses a probabilistic primality test.
    """
    while True:
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if is_prime(candidate):
            return candidate


def is_prime(n: int, k: int = 40) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


def find_prime_for_protocol(n: int, min_bits: int = 256) -> int:
    """
    Find a suitable prime for the protocol.
    Uses a 256-bit prime which is sufficient for our aggregation needs.
    """
    return generate_large_prime(min_bits)


class ShamirSecretSharing:
    """
    Shamir Secret Sharing implementation.

    Shares a secret s by constructing a random polynomial f(x) = s + a1*x + ... + a_{t-1}*x^{t-1}
    and evaluating at points 1, 2, ..., n.
    """

    def __init__(self, n: int, t: int, q: int = None, seed: int = None):
        """
        Initialize Shamir Secret Sharing.

        Args:
            n: Number of total shares
            t: Threshold (minimum shares needed for reconstruction)
            q: Prime modulus (generated if not provided)
            seed: Random seed for reproducibility (only for testing)
        """
        self.n = n
        self.t = t
        self.q = q if q is not None else find_prime_for_protocol(n)
        self._rng = secrets.SystemRandom()
        if seed is not None:
            import random
            self._rng = random.Random(seed)

    def share(self, secret: int) -> List[Tuple[int, int]]:
        """
        Create n shares of a secret with threshold t.

        The polynomial is: f(x) = s + a1*x + a2*x^2 + ... + a_{t-1}*x^{t-1} mod q

        Args:
            secret: The secret value to share

        Returns:
            List of (x, y) tuples where y = f(x) mod q
        """
        # Ensure secret is in valid range
        secret = secret % self.q

        # Generate random polynomial coefficients a_1, ..., a_{t-1}
        coeffs = [secret]  # a_0 = secret
        for _ in range(self.t - 1):
            coeffs.append(self._rng.randrange(self.q))

        # Evaluate polynomial at points 1, 2, ..., n
        shares = []
        for x in range(1, self.n + 1):
            y = self._eval_polynomial(coeffs, x)
            shares.append((x, y))

        return shares

    def _eval_polynomial(self, coeffs: List[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method."""
        result = 0
        for coeff in reversed(coeffs):
            result = (result * x + coeff) % self.q
        return result

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct the secret from shares using Lagrange interpolation.

        Args:
            shares: List of (x, y) tuples (at least t shares needed)

        Returns:
            The reconstructed secret
        """
        if len(shares) < self.t:
            raise ValueError(f"Need at least {self.t} shares, got {len(shares)}")

        # Use first t shares
        shares = shares[:self.t]

        # Compute Lagrange interpolation at x=0
        # f(0) = sum_{i} y_i * L_i(0)
        # L_i(0) = prod_{j!=i} (0 - x_j) / (x_i - x_j) = prod_{j!=i} (-x_j) / (x_i - x_j)

        result = 0
        for i, (x_i, y_i) in enumerate(shares):
            # Compute Lagrange coefficient L_i(0)
            numerator = 1
            denominator = 1

            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-x_j)) % self.q
                    denominator = (denominator * (x_i - x_j)) % self.q

            # Compute L_i(0) = numerator / denominator mod q
            denom_inv = pow(denominator, -1, self.q)
            lagrange_coeff = (numerator * denom_inv) % self.q

            # Add contribution: y_i * L_i(0)
            result = (result + y_i * lagrange_coeff) % self.q

        return result

    def reconstruct_from_sum(self, summed_shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct from aggregated/summed shares.

        When multiple clients' shares are summed at each decryptor,
        this method reconstructs the sum of all secrets.

        Args:
            summed_shares: List of (x, aggregated_y) tuples

        Returns:
            The sum of all original secrets
        """
        return self.reconstruct(summed_shares)


def compute_threshold(n: int) -> int:
    """Compute threshold t = ceil(2n/3) for Byzantine fault tolerance."""
    return ceil(2 * n / 3)


def compute_lagrange_coefficients(active_ids: List[int], q: int) -> dict:
    """
    Compute Lagrange coefficients for a set of active decryptor IDs.

    For reconstruction at x=0:
    λ_i = prod_{j != i} (0 - x_j) / (x_i - x_j)
        = prod_{j != i} (-x_j) / (x_i - x_j)

    These coefficients allow each decryptor to compute their partial decryption
    locally, enabling distributed DP where noise is added AFTER Lagrange scaling.

    Args:
        active_ids: List of active decryptor IDs (1-indexed)
        q: Prime modulus

    Returns:
        Dictionary mapping decryptor_id -> Lagrange coefficient (mod q)
    """
    coeffs = {}

    for i, x_i in enumerate(active_ids):
        numerator = 1
        denominator = 1

        for j, x_j in enumerate(active_ids):
            if i != j:
                numerator = (numerator * (-x_j)) % q
                denominator = (denominator * (x_i - x_j)) % q

        # λ_i = numerator / denominator mod q
        denom_inv = pow(denominator, -1, q)
        coeffs[x_i] = (numerator * denom_inv) % q

    return coeffs


# Convenience functions for testing
def share_secret(secret: int, n: int, t: int, q: int = None) -> List[Tuple[int, int]]:
    """Share a secret using Shamir SS."""
    ss = ShamirSecretSharing(n, t, q)
    return ss.share(secret)


def reconstruct_secret(shares: List[Tuple[int, int]], n: int, t: int, q: int) -> int:
    """Reconstruct a secret from shares."""
    ss = ShamirSecretSharing(n, t, q)
    return ss.reconstruct(shares)
