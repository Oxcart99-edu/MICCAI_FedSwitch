"""
Protocol implementation: Client, Decryptor, and Coordinator.

Matches the paper's protocol:
  1. Server broadcasts target cohort size |A| = αN at initialisation.
  2. Each client clips its histogram, adds Pólya(1/|A|, β) partial noise,
     and secret-shares the *noisy* count via Shamir's scheme.
  3. Decryptors accumulate shares (pure accumulators, no noise addition).
  4. At trigger round R*, decryptors reveal aggregated shares and the
     coordinator reconstructs via Lagrange interpolation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .crypto import ShamirSecretSharing, compute_threshold, compute_lagrange_coefficients
from .privacy import clip_histogram, clip_histogram_uniform
from .distributed_noise import sample_partial_noise, compute_trigger_round, compute_target_cohort


@dataclass
class ProtocolResult:
    """Result of running the protocol."""
    estimated_histogram: Dict[str, int]
    client_sets_per_round: List[List[int]] = field(default_factory=list)
    n_active_decryptors: int = 0
    clipping_applied: bool = False
    noise_values: Optional[Dict[str, int]] = None
    target_cohort: int = 0
    trigger_round: int = 0


class Client:
    """
    Client in the federated protocol.

    Each client:
      1. Clips its local histogram to ℓ1-norm = C.
      2. Adds partial Pólya noise (calibrated to target cohort |A|).
      3. Secret-shares the noisy count via Shamir's scheme.
    """

    def __init__(self, client_id: int, histogram: Dict[str, int], labels: List[str]):
        self.client_id = client_id
        self.histogram = histogram
        self.labels = labels

    def compute_shares(
        self,
        n_decryptors: int,
        threshold: int,
        prime: int,
        clipping_threshold: int,
        target_cohort: int,
        epsilon: float,
        clipping_method: str = "proportional",
        rng: np.random.Generator = None,
    ) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, int]]:
        """
        Clip, add noise, then create Shamir shares of noisy counts.

        Returns:
            Tuple of (decryptor_shares, noise_per_label).
            - decryptor_shares: {dec_id: [(label_idx, share_value), ...]}.
            - noise_per_label: {label_idx: noise_value} for tracking.
        """
        # Step 1: Clip the histogram
        if clipping_method in ("uniform_with_replacement", "uniform_wr") and rng is not None:
            clipped = clip_histogram_uniform(
                self.histogram, clipping_threshold, rng, replace=True
            )
        elif clipping_method == "uniform" and rng is not None:
            clipped = clip_histogram_uniform(self.histogram, clipping_threshold, rng)
        else:
            clipped = clip_histogram(self.histogram, clipping_threshold, method=clipping_method)

        # Step 2: Add partial Pólya noise (client-side, before secret sharing)
        noise_rng = np.random.default_rng(
            rng.integers(0, 2**31) if rng is not None else None
        )
        noise = sample_partial_noise(
            n_contributors=target_cohort,
            sensitivity=clipping_threshold,
            epsilon=epsilon,
            n_labels=len(self.labels),
            rng=noise_rng,
        )

        # Step 3: Secret-share the noisy count
        ss = ShamirSecretSharing(n_decryptors, threshold, prime)
        decryptor_shares = {d: [] for d in range(n_decryptors)}

        for label_idx, label in enumerate(self.labels):
            noisy_count = clipped.get(label, 0) + noise[label_idx]
            shares = ss.share(noisy_count)
            for decryptor_id, (x, y) in enumerate(shares):
                decryptor_shares[decryptor_id].append((label_idx, y))

        return decryptor_shares, noise


class Decryptor:
    """
    Decryptor: pure share accumulator.

    Each decryptor aggregates the Shamir shares it receives from clients.
    No noise is added by decryptors (noise is added client-side).
    """

    def __init__(self, decryptor_id: int, n_labels: int, prime: int):
        self.decryptor_id = decryptor_id
        self.n_labels = n_labels
        self.prime = prime
        self.aggregated_shares = {l: 0 for l in range(n_labels)}
        self.is_failed = False

    def receive_shares_from_client(self, shares: List[Tuple[int, int]]):
        """Accumulate shares from a client."""
        for label_idx, share_value in shares:
            self.aggregated_shares[label_idx] = (
                self.aggregated_shares[label_idx] + share_value
            ) % self.prime

    def get_partial_decryption(self, label_idx: int, lagrange_coeff: int) -> int:
        """Compute partial decryption: λ_i · aggregated_share_i (mod prime)."""
        return (lagrange_coeff * self.aggregated_shares[label_idx]) % self.prime

    def reset(self):
        """Reset for a new trial."""
        self.aggregated_shares = {l: 0 for l in range(self.n_labels)}


class Coordinator:
    """
    Coordinator for the histogram estimation protocol.

    Protocol flow (matching the paper):
      1. Compute target cohort |A| and trigger round R*.
      2. Over R* rounds, sample clients; each client clips + adds noise + shares.
      3. Decryptors accumulate shares across rounds.
      4. At R*, reconstruct via Lagrange interpolation.
    """

    def __init__(
        self,
        n_clients: int,
        n_decryptors: int,
        labels: List[str],
        clipping_threshold: int,
        epsilon: float,
        gamma: float = 1.0,
        n_rounds: int = 1,
        alpha: float = 0.9,
        sampling_with_replacement: bool = False,
        count_client_once: bool = False,
        fixed_client_sets_per_round: Optional[List[List[int]]] = None,
        failure_rate: float = 0.0,
        clipping_method: str = "proportional",
        seed: int = None,
    ):
        self.n_clients = n_clients
        self.n_decryptors = n_decryptors
        self.labels = labels
        self.n_labels = len(labels)
        self.clipping_threshold = clipping_threshold
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_rounds = n_rounds
        self.alpha = alpha
        self.sampling_with_replacement = sampling_with_replacement
        self.count_client_once = count_client_once
        self.fixed_client_sets_per_round = fixed_client_sets_per_round
        self.failure_rate = failure_rate
        self.clipping_method = clipping_method

        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # Target cohort size |A| = ceil(α·N) — broadcast to all clients
        self.target_cohort = compute_target_cohort(alpha, n_clients)

        # Trigger round R* from the paper's formula
        self.trigger_round = compute_trigger_round(alpha, gamma, n_clients)

        # Shamir threshold for Byzantine fault tolerance
        self.threshold = compute_threshold(n_decryptors)

        # Prime for secret sharing
        from .crypto import find_prime_for_protocol
        self.prime = find_prime_for_protocol(n_decryptors)

        # Initialise decryptors (no seed needed, they don't sample)
        self.decryptors = [
            Decryptor(
                decryptor_id=d + 1,  # 1-indexed for Shamir
                n_labels=self.n_labels,
                prime=self.prime,
            )
            for d in range(n_decryptors)
        ]

    def run_protocol(
        self,
        client_histograms: List[Dict[str, int]],
        return_detailed: bool = False,
    ) -> ProtocolResult:
        """
        Run the full protocol.

        Steps:
          1. Determine failing decryptors, compute Lagrange coefficients.
          2. Over n_rounds rounds, sample clients. Each client clips its
             histogram, adds Pólya(1/|A|, β) noise, and secret-shares
             the noisy result.
          3. Decryptors accumulate shares.
          4. Reconstruct via Lagrange interpolation.
        """
        # Reset decryptors
        for dec in self.decryptors:
            dec.reset()

        # Create client objects
        clients = [
            Client(i, hist, self.labels)
            for i, hist in enumerate(client_histograms)
        ]

        # Determine failing decryptors
        n_failures = int(self.failure_rate * self.n_decryptors)
        failed_indices = set(
            self.rng.choice(self.n_decryptors, size=n_failures, replace=False)
        ) if n_failures > 0 else set()

        for idx in failed_indices:
            self.decryptors[idx].is_failed = True

        active_decryptors = [d for d in self.decryptors if not d.is_failed]
        active_ids = [d.decryptor_id for d in active_decryptors]
        n_active = len(active_ids)

        if n_active < self.threshold:
            raise RuntimeError(
                f"Too many failures: {n_failures} failed, "
                f"only {n_active} active, need {self.threshold}"
            )

        lagrange_coeffs = compute_lagrange_coefficients(active_ids, self.prime)

        # Phase: distribute client shares over rounds
        client_sets_per_round = []
        already_contributed = set()
        total_noise = {l: 0 for l in range(self.n_labels)}

        if self.gamma < 1.0:
            clients_per_round = max(1, int(self.gamma * self.n_clients))

            for round_idx in range(self.n_rounds):
                # Determine which clients are sampled this round
                if self.fixed_client_sets_per_round is not None:
                    round_client_indices = list(
                        self.fixed_client_sets_per_round[round_idx]
                    )
                elif self.sampling_with_replacement:
                    replace = clients_per_round > len(clients)
                    round_client_indices = self.rng.choice(
                        len(clients), size=clients_per_round, replace=replace
                    ).tolist()
                else:
                    # Without replacement: sequential partitioning
                    if round_idx == 0:
                        self._shuffled = list(range(len(clients)))
                        self.rng.shuffle(self._shuffled)
                    start = round_idx * clients_per_round
                    end = min(start + clients_per_round, len(clients))
                    if start >= len(clients):
                        self.rng.shuffle(self._shuffled)
                        start, end = 0, min(clients_per_round, len(clients))
                    round_client_indices = self._shuffled[start:end]

                client_sets_per_round.append(round_client_indices)

                # Filter to first-time contributors if count_client_once
                if self.count_client_once:
                    effective = [
                        i for i in round_client_indices
                        if i not in already_contributed
                    ]
                    already_contributed.update(effective)
                else:
                    effective = round_client_indices

                # Each client: clip + noise + share
                self._distribute_client_shares(
                    [clients[i] for i in effective], total_noise
                )
        else:
            # All clients in one round
            client_sets_per_round.append(list(range(len(clients))))
            self._distribute_client_shares(clients, total_noise)

        # Reconstruction via Lagrange interpolation
        estimated_histogram = {}
        for label_idx, label in enumerate(self.labels):
            total = 0
            for dec in active_decryptors:
                lambda_i = lagrange_coeffs[dec.decryptor_id]
                partial = dec.get_partial_decryption(label_idx, lambda_i)
                total = (total + partial) % self.prime

            # Handle field representation of negative values
            if total > self.prime // 2:
                total = total - self.prime

            estimated_histogram[label] = max(0, int(total))

        noise_str = {self.labels[k]: v for k, v in total_noise.items()}

        # Reset failure status
        for dec in self.decryptors:
            dec.is_failed = False

        return ProtocolResult(
            estimated_histogram=estimated_histogram,
            client_sets_per_round=client_sets_per_round,
            n_active_decryptors=n_active,
            clipping_applied=True,
            noise_values=noise_str,
            target_cohort=self.target_cohort,
            trigger_round=self.trigger_round,
        )

    def _distribute_client_shares(
        self,
        clients: List[Client],
        total_noise: Dict[int, int],
    ):
        """Each client clips, adds noise, secret-shares, and sends to decryptors."""
        for client in clients:
            shares, noise = client.compute_shares(
                self.n_decryptors,
                self.threshold,
                self.prime,
                self.clipping_threshold,
                self.target_cohort,
                self.epsilon,
                self.clipping_method,
                self.rng,
            )

            # Track total noise for output
            for label_idx, val in noise.items():
                total_noise[label_idx] = total_noise.get(label_idx, 0) + val

            # Distribute shares to non-failed decryptors
            for dec_id, dec_shares in shares.items():
                if not self.decryptors[dec_id].is_failed:
                    self.decryptors[dec_id].receive_shares_from_client(dec_shares)
