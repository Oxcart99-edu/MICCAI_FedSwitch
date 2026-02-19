"""
Protocol implementation: Client, Decryptor, and Coordinator classes.

Implements the Heavy Hitters protocol with:
- Shamir Secret Sharing for secure aggregation
- Distributed Discrete Laplace noise (Kairouz 2021 style)
- Support for decryptor failures up to threshold

Key insight for distributed DP:
- Each decryptor computes: partial_i = λ_i * share_i + noise_i
- Coordinator sums: Σ(partial_i) = secret + Σ(noise_i) = secret + DLaplace(b)
- noise_i ~ NegBin(1/n, p) - NegBin(1/n, p), sum gives DLaplace(b)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from math import ceil
from dataclasses import dataclass, field

from .crypto import ShamirSecretSharing, compute_threshold, compute_lagrange_coefficients
from .privacy import clip_histogram, clip_histogram_uniform
from .distributed_noise import sample_partial_discrete_laplace


@dataclass
class ProtocolResult:
    """Result of running the protocol."""
    estimated_histogram: Dict[str, int]
    client_sets_per_round: List[List[int]] = field(default_factory=list)
    n_active_decryptors: int = 0
    clipping_applied: bool = False
    noise_values: Optional[Dict[str, int]] = None  # Total noise per label


class Client:
    """
    Client in the federated protocol.

    Each client holds local label counts and creates Shamir shares
    for secure aggregation.
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
        clipping_method: str = "proportional",
        rng: np.random.Generator = None
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Create Shamir shares of clipped histogram for each decryptor."""
        # Clip the histogram
        if clipping_method == "uniform" and rng is not None:
            clipped = clip_histogram_uniform(self.histogram, clipping_threshold, rng)
        elif clipping_method in ("uniform_with_replacement", "uniform_wr") and rng is not None:
            clipped = clip_histogram_uniform(
                self.histogram, clipping_threshold, rng, replace=True
            )
        else:
            clipped = clip_histogram(self.histogram, clipping_threshold, method=clipping_method)

        # Initialize secret sharing
        ss = ShamirSecretSharing(n_decryptors, threshold, prime)

        # Create shares for each label
        decryptor_shares = {d: [] for d in range(n_decryptors)}

        for label_idx, label in enumerate(self.labels):
            count = clipped.get(label, 0)
            shares = ss.share(count)

            for decryptor_id, (x, y) in enumerate(shares):
                decryptor_shares[decryptor_id].append((label_idx, y))

        return decryptor_shares


class Decryptor:
    """
    Decryptor in the protocol.

    Each decryptor:
    1. Aggregates shares from clients
    2. Generates independent noise (NegBin(1/n, p) - NegBin(1/n, p))
    3. Computes partial decryption: λ_i * aggregated_share + noise_i
    """

    def __init__(
        self,
        decryptor_id: int,
        n_labels: int,
        prime: int,
        seed: int = None
    ):
        self.decryptor_id = decryptor_id
        self.n_labels = n_labels
        self.prime = prime

        # Each decryptor has their own RNG for independent noise
        self.rng = np.random.default_rng(seed)

        # Aggregated shares per label
        self.aggregated_shares = {l: 0 for l in range(n_labels)}

        # This decryptor's noise contribution (generated independently)
        self.noise = {l: 0 for l in range(n_labels)}

        self.is_failed = False

    def receive_share(self, label_idx: int, share_value: int):
        """Receive and aggregate a share from a client."""
        self.aggregated_shares[label_idx] = (
            self.aggregated_shares[label_idx] + share_value
        ) % self.prime

    def receive_shares_from_client(self, shares: List[Tuple[int, int]]):
        """Receive all shares from a single client."""
        for label_idx, share_value in shares:
            self.receive_share(label_idx, share_value)

    def generate_noise(self, n_active: int, sensitivity: int, epsilon: float):
        """
        Generate this decryptor's independent noise contribution.

        Uses NegBin(1/n, p) - NegBin(1/n, p) so sum of n decryptors = DLaplace(b).
        """
        b = sensitivity / epsilon
        self.noise = {}
        for label_idx in range(self.n_labels):
            self.noise[label_idx] = sample_partial_discrete_laplace(b, n_active, self.rng)

    def get_partial_decryption(self, label_idx: int, lagrange_coeff: int) -> int:
        """
        Compute partial decryption: λ_i * share_i + noise_i

        This is the key for distributed DP:
        - Apply Lagrange coefficient first
        - Then add noise
        - Sum of partials = secret + Σ(noise_i) = secret + DLaplace(b)
        """
        # λ_i * share_i (mod prime)
        partial = (lagrange_coeff * self.aggregated_shares[label_idx]) % self.prime

        # Add noise (convert negative to field element)
        noise_val = self.noise.get(label_idx, 0)
        if noise_val < 0:
            noise_val = self.prime + noise_val

        return (partial + noise_val) % self.prime

    def reset(self):
        """Reset for a new round."""
        self.aggregated_shares = {l: 0 for l in range(self.n_labels)}
        self.noise = {l: 0 for l in range(self.n_labels)}


class Coordinator:
    """
    Coordinator for the Heavy Hitters protocol.

    Protocol flow:
    1. Clients send Shamir shares to decryptors
    2. Each decryptor generates independent noise
    3. Each decryptor computes partial decryption: λ_i * share + noise_i
    4. Coordinator sums partials: secret + Σ(noise_i) = secret + DLaplace(b)
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
        sampling_with_replacement: bool = False,
        count_client_once: bool = False,
        fixed_client_sets_per_round: Optional[List[List[int]]] = None,
        failure_rate: float = 0.0,
        clipping_method: str = "proportional",
        seed: int = None,
        max_rounds: int = 1000
    ):
        self.n_clients = n_clients
        self.n_decryptors = n_decryptors
        self.labels = labels
        self.n_labels = len(labels)
        self.clipping_threshold = clipping_threshold
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_rounds = min(n_rounds, max_rounds)
        self.sampling_with_replacement = sampling_with_replacement
        self.count_client_once = count_client_once
        self.fixed_client_sets_per_round = fixed_client_sets_per_round
        self.failure_rate = failure_rate
        self.clipping_method = clipping_method

        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # Compute threshold for Byzantine fault tolerance
        self.threshold = compute_threshold(n_decryptors)

        # Generate prime for secret sharing (256-bit)
        from .crypto import find_prime_for_protocol
        self.prime = find_prime_for_protocol(n_decryptors)

        # Initialize decryptors - each with their own independent seed
        self.decryptors = []
        for d in range(n_decryptors):
            dec_seed = self.rng.integers(0, 2**31) if seed is not None else None
            self.decryptors.append(
                Decryptor(
                    decryptor_id=d + 1,  # 1-indexed for Shamir
                    n_labels=self.n_labels,
                    prime=self.prime,
                    seed=dec_seed
                )
            )

    def run_protocol(
        self,
        client_histograms: List[Dict[str, int]],
        return_detailed: bool = False
    ) -> ProtocolResult:
        """
        Run the full Heavy Hitters protocol with distributed DP.

        Protocol:
        1. Setup: Determine failing decryptors, compute Lagrange coefficients
        2. Share Distribution: Clients send Shamir shares to decryptors
        3. Noise Generation: Each decryptor generates independent noise
        4. Partial Decryption: Each computes λ_i * share + noise_i
        5. Aggregation: Sum partials to get secret + DLaplace(b)
        """
        # Reset decryptors
        for dec in self.decryptors:
            dec.reset()

        # Create client objects
        clients = [
            Client(i, hist, self.labels)
            for i, hist in enumerate(client_histograms)
        ]

        # Phase 1: Determine failing decryptors
        n_failures = int(self.failure_rate * self.n_decryptors)
        failed_indices = set(self.rng.choice(
            self.n_decryptors,
            size=n_failures,
            replace=False
        )) if n_failures > 0 else set()

        for idx in failed_indices:
            self.decryptors[idx].is_failed = True

        # Get active decryptors
        active_decryptors = [d for d in self.decryptors if not d.is_failed]
        active_ids = [d.decryptor_id for d in active_decryptors]
        n_active = len(active_ids)

        if n_active < self.threshold:
            raise RuntimeError(
                f"Too many failures: {n_failures} failed, "
                f"only {n_active} active, need {self.threshold}"
            )

        # Compute Lagrange coefficients for active decryptors
        # Each decryptor knows who's active and can compute this locally
        lagrange_coeffs = compute_lagrange_coefficients(active_ids, self.prime)

        # Track client sets per round
        client_sets_per_round = []
        already_contributed = set()

        # Phase 2: Client share distribution
        if self.gamma < 1.0:
            clients_per_round = max(1, int(self.gamma * self.n_clients))
            if self.fixed_client_sets_per_round is not None:
                if len(self.fixed_client_sets_per_round) < self.n_rounds:
                    raise ValueError(
                        f"fixed_client_sets_per_round has {len(self.fixed_client_sets_per_round)} rounds, "
                        f"but n_rounds={self.n_rounds}"
                    )

                for round_idx in range(self.n_rounds):
                    round_client_indices = list(self.fixed_client_sets_per_round[round_idx])

                    for i in round_client_indices:
                        if i < 0 or i >= len(clients):
                            raise ValueError(
                                f"Invalid client id {i} at fixed round {round_idx + 1}; "
                                f"must be in [0, {len(clients)-1}]"
                            )

                    if self.count_client_once:
                        effective_indices = []
                        for i in round_client_indices:
                            if i not in already_contributed:
                                already_contributed.add(i)
                                effective_indices.append(i)
                    else:
                        effective_indices = round_client_indices

                    round_clients = [clients[i] for i in effective_indices]
                    client_sets_per_round.append(round_client_indices)
                    self._distribute_client_shares(round_clients)
            elif self.sampling_with_replacement:
                # Each round draws from the full client pool; clients can reappear across rounds.
                for _ in range(self.n_rounds):
                    replace = clients_per_round > len(clients)
                    round_client_indices = self.rng.choice(
                        len(clients), size=clients_per_round, replace=replace
                    ).tolist()
                    if self.count_client_once:
                        effective_indices = []
                        for i in round_client_indices:
                            if i not in already_contributed:
                                already_contributed.add(i)
                                effective_indices.append(i)
                    else:
                        effective_indices = round_client_indices

                    round_clients = [clients[i] for i in effective_indices]
                    client_sets_per_round.append(round_client_indices)
                    self._distribute_client_shares(round_clients)
            else:
                client_indices = list(range(len(clients)))
                self.rng.shuffle(client_indices)

                for round_idx in range(self.n_rounds):
                    start_idx = round_idx * clients_per_round
                    end_idx = min(start_idx + clients_per_round, len(clients))

                    if start_idx >= len(clients):
                        self.rng.shuffle(client_indices)
                        start_idx = 0
                        end_idx = min(clients_per_round, len(clients))

                    round_client_indices = client_indices[start_idx:end_idx]
                    if self.count_client_once:
                        effective_indices = []
                        for i in round_client_indices:
                            if i not in already_contributed:
                                already_contributed.add(i)
                                effective_indices.append(i)
                    else:
                        effective_indices = round_client_indices

                    round_clients = [clients[i] for i in effective_indices]

                    client_sets_per_round.append(list(round_client_indices))
                    self._distribute_client_shares(round_clients)
        else:
            client_sets_per_round.append(list(range(len(clients))))
            self._distribute_client_shares(clients)

        # Phase 3: Each decryptor generates independent noise
        # noise_i ~ NegBin(1/n, p) - NegBin(1/n, p)
        # Sum of n such noises = DLaplace(C/ε)
        for dec in active_decryptors:
            dec.generate_noise(n_active, self.clipping_threshold, self.epsilon)

        # Phase 4: Reconstruction via partial decryptions
        # Each decryptor computes: λ_i * share_i + noise_i
        # Sum = secret + Σ(noise_i) = secret + DLaplace(b)
        estimated_histogram = {}
        total_noise = {}

        for label_idx, label in enumerate(self.labels):
            total = 0
            label_noise = 0

            for dec in active_decryptors:
                lambda_i = lagrange_coeffs[dec.decryptor_id]
                partial = dec.get_partial_decryption(label_idx, lambda_i)
                total = (total + partial) % self.prime

                # Track noise for output
                label_noise += dec.noise.get(label_idx, 0)

            # Handle field representation of negative values
            if total > self.prime // 2:
                total = total - self.prime

            estimated_histogram[label] = max(0, int(total))
            total_noise[label] = label_noise

        # Reset failure status for next run
        for dec in self.decryptors:
            dec.is_failed = False

        return ProtocolResult(
            estimated_histogram=estimated_histogram,
            client_sets_per_round=client_sets_per_round,
            n_active_decryptors=n_active,
            clipping_applied=True,
            noise_values=total_noise
        )

    def _distribute_client_shares(self, clients: List[Client]):
        """Distribute shares from clients to decryptors."""
        for client in clients:
            shares = client.compute_shares(
                self.n_decryptors,
                self.threshold,
                self.prime,
                self.clipping_threshold,
                self.clipping_method,
                self.rng
            )

            for dec_id, dec_shares in shares.items():
                if not self.decryptors[dec_id].is_failed:
                    self.decryptors[dec_id].receive_shares_from_client(dec_shares)


def run_single_trial(
    client_histograms: List[Dict[str, int]],
    labels: List[str],
    n_decryptors: int,
    clipping_threshold: int,
    epsilon: float,
    gamma: float = 1.0,
    n_rounds: int = 1,
    sampling_with_replacement: bool = False,
    count_client_once: bool = False,
    failure_rate: float = 0.0,
    clipping_method: str = "proportional",
    seed: int = None
) -> ProtocolResult:
    """Run a single trial of the protocol."""
    coordinator = Coordinator(
        n_clients=len(client_histograms),
        n_decryptors=n_decryptors,
        labels=labels,
        clipping_threshold=clipping_threshold,
        epsilon=epsilon,
        gamma=gamma,
        n_rounds=n_rounds,
        sampling_with_replacement=sampling_with_replacement,
        count_client_once=count_client_once,
        failure_rate=failure_rate,
        clipping_method=clipping_method,
        seed=seed
    )

    return coordinator.run_protocol(client_histograms, return_detailed=True)
