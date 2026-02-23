# Mathematical Foundations for the Heavy Hitters Label Distribution Protocol

This document provides formal mathematical justification for the protocol's correctness, privacy guarantees, and error bounds.

---

## TL;DR

### Goal

We want to estimate the **global label distribution** across multiple federated medical imaging sites (e.g., hospitals) **without** any site revealing its local data. This is useful for:
- Understanding class imbalance before federated learning
- Enabling label-aware aggregation strategies (e.g., FedLA)
- Auditing dataset composition across institutions

### Problem Setting

- **N clients** (hospitals/sites), each holding a local dataset with labels from K classes
- **n decryptors** (semi-trusted servers) that help with secure aggregation
- We want to compute $H(\ell) = \sum_{i=1}^{N} H_i(\ell)$ for each label $\ell$ (global histogram)
- **Constraints**:
  - No client reveals its local histogram $H_i(\ell)$
  - Output must satisfy $(\varepsilon, 0)$-differential privacy (site-level)
  - Protocol must tolerate up to $\lfloor n/3 \rfloor$ decryptor failures

### Our Approach

1. **Shamir Secret Sharing**: Each client secret-shares its (clipped) histogram counts among decryptors
2. **Homomorphic Aggregation**: Decryptors sum received shares locally (no communication needed)
3. **Distributed Discrete Laplace Noise**: Each decryptor independently adds noise sampled from $\text{NegBin}(1/n, 1-p) - \text{NegBin}(1/n, 1-p)$
4. **Reconstruction**: Coordinator collects partial results and sums them to get:
   $$\hat{H}(\ell) = H(\ell) + \underbrace{\sum_{i=1}^{n} Z_i(\ell)}_{\sim \text{DLaplace}(C/\varepsilon)}$$

### Key Insight

The **Negative Binomial distribution is infinitely divisible**: if each of $n$ parties independently samples $\text{NegBin}(1/n, p)$, their sum equals $\text{NegBin}(1, p)$. This means:
- **No shared seed required** between decryptors
- Each decryptor generates noise **independently** with their own randomness
- The sum of all partial noises **exactly** equals Discrete Laplace noise

This is the distributed DP approach from Kairouz et al. (2021).

### Privacy Guarantee

The protocol achieves $(\varepsilon, 0)$-differential privacy at the **site level**: replacing any single client's entire contribution with arbitrary data changes the output distribution by at most $e^\varepsilon$.

---

## Notation and Setup

| Symbol | Definition |
|--------|------------|
| $K$ | Number of distinct labels |
| $N$ | Number of clients (data sites) |
| $n$ | Number of decryptors |
| $t$ | Threshold for Shamir SS, $t = \lceil 2n/3 \rceil$ |
| $C$ | Clipping threshold (L₁ sensitivity) |
| $\varepsilon$ | Privacy budget |
| $H_i(\ell)$ | True histogram count for label $\ell$ at client $i$ |
| $H(\ell)$ | $\sum_i H_i(\ell)$ = global true count for label $\ell$ |
| $\hat{H}(\ell)$ | Estimated count for label $\ell$ (protocol output) |
| $q$ | Prime modulus for finite field arithmetic |

---

## Section 1: Shamir Secret Sharing Correctness

### Definition 1.1 (Shamir Secret Sharing)

For secret $s \in \mathbb{Z}_q$, threshold $t$, and $n$ parties:
1. Sample random coefficients $a_1, \ldots, a_{t-1} \in \mathbb{Z}_q$ uniformly
2. Define polynomial $f(x) = s + a_1 x + a_2 x^2 + \cdots + a_{t-1} x^{t-1} \mod q$
3. Share$_j = (j, f(j))$ for $j \in \{1, \ldots, n\}$

### Theorem 1.1 (Perfect Secrecy)

Any $t-1$ or fewer shares reveal no information about the secret $s$.

**Formally:** For any subset $S \subset \{1,\ldots,n\}$ with $|S| < t$:
$$H(s \mid \{\text{Share}_j : j \in S\}) = H(s)$$
where $H$ denotes Shannon entropy.

**Proof:**
For $|S| = t-1$ shares, we have $t-1$ linear equations in $t$ unknowns $(s, a_1, \ldots, a_{t-1})$. For any fixed value $s' \in \mathbb{Z}_q$, there exists exactly one choice of $(a_1, \ldots, a_{t-1})$ satisfying all equations. Thus, each possible secret is equally likely given the shares. ∎

### Theorem 1.2 (Lagrange Reconstruction)

Given $t$ or more shares $\{(x_i, y_i)\}_{i \in S}$ where $|S| \geq t$, the secret can be reconstructed as:
$$s = f(0) = \sum_{i \in S} y_i \cdot \lambda_i(0) \mod q$$

where the Lagrange coefficients are:
$$\lambda_i(0) = \prod_{j \in S, j \neq i} \frac{-x_j}{x_i - x_j} \mod q$$

**Proof:**
By Lagrange interpolation, $f(x) = \sum_{i \in S} y_i \cdot L_i(x)$ where $L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$.

Evaluating at $x = 0$:
$$f(0) = s = \sum_{i \in S} y_i \cdot \prod_{j \neq i} \frac{-x_j}{x_i - x_j} = \sum_{i \in S} y_i \cdot \lambda_i(0)$$
∎

### Proposition 1.1 (Additive Homomorphism)

If $\{s^{(k)}\}_k$ are secrets with shares $\{\text{Share}_j^{(k)} = (j, y_j^{(k)})\}_k$, then the sum $\sum_k s^{(k)}$ has shares $(j, \sum_k y_j^{(k)} \mod q)$.

**Proof:**
Let $f^{(k)}(x)$ be the polynomial for secret $s^{(k)}$. Then $g(x) = \sum_k f^{(k)}(x)$ is a polynomial with $g(0) = \sum_k s^{(k)}$. The share at point $j$ is $g(j) = \sum_k f^{(k)}(j) = \sum_k y_j^{(k)} \mod q$. ∎

---

## Section 2: Distributed Discrete Laplace Noise

### Definition 2.1 (Discrete Laplace Distribution)

$X \sim \text{DLaplace}(b)$ if:
$$P(X = k) = \frac{1-p}{1+p} \cdot p^{|k|} \quad \text{for } k \in \mathbb{Z}$$

where $p = \exp(-1/b)$.

**Properties:**
- $\mathbb{E}[X] = 0$
- $\text{Var}(X) = \frac{2p}{(1-p)^2}$

### Definition 2.2 (Negative Binomial Distribution)

$Y \sim \text{NegBin}(r, p)$ has PMF:
$$P(Y = k) = \frac{\Gamma(k+r)}{k! \cdot \Gamma(r)} \cdot p^r \cdot (1-p)^k \quad \text{for } k \in \{0,1,2,\ldots\}$$

For non-integer $r > 0$, this is well-defined via the Gamma function.

**Properties:**
- $\mathbb{E}[Y] = r(1-p)/p$
- $\text{Var}(Y) = r(1-p)/p^2$

### Lemma 2.1 (Geometric-Laplace Connection)

If $Y_1, Y_2 \sim \text{Geom}(1-p)$ independently, where $\text{Geom}(q)$ has PMF $P(Y=k) = q(1-q)^k$, then:
$$X = Y_1 - Y_2 \sim \text{DLaplace}(b) \quad \text{with } p = \exp(-1/b)$$

**Proof:**
For $k \geq 0$:
$$P(Y_1 - Y_2 = k) = \sum_{j=0}^{\infty} P(Y_1 = k+j) \cdot P(Y_2 = j)$$
$$= \sum_{j=0}^{\infty} (1-p) p^{k+j} \cdot (1-p) p^j = (1-p)^2 p^k \sum_{j=0}^{\infty} p^{2j}$$
$$= (1-p)^2 p^k \cdot \frac{1}{1-p^2} = \frac{(1-p)^2 p^k}{(1-p)(1+p)} = \frac{1-p}{1+p} \cdot p^k$$

By symmetry, $P(X = -k) = P(X = k)$. This matches Definition 2.1. ∎

### Theorem 2.1 (Infinite Divisibility of Negative Binomial)

If $Y \sim \text{NegBin}(r, p)$, then:
$$Y \stackrel{d}{=} \sum_{i=1}^{n} Y_i \quad \text{where } Y_i \sim \text{NegBin}(r/n, p) \text{ are independent}$$

**Proof:**
The characteristic function of $\text{NegBin}(r, p)$ is:
$$\phi(t) = \left(\frac{p}{1 - (1-p) e^{it}}\right)^r$$

For $n$ independent $\text{NegBin}(r/n, p)$ variables:
$$\phi_{\text{sum}}(t) = [\phi_{r/n}(t)]^n = \left[\left(\frac{p}{1 - (1-p) e^{it}}\right)^{r/n}\right]^n = \left(\frac{p}{1 - (1-p) e^{it}}\right)^r = \phi(t)$$

Since characteristic functions uniquely determine distributions, the sum of $n$ independent $\text{NegBin}(r/n, p)$ equals $\text{NegBin}(r, p)$ in distribution. ∎

### Corollary 2.1 (Distributed Discrete Laplace) ⭐

Let $Z_i = Y_{i,1} - Y_{i,2}$ for $i = 1, \ldots, n$, where each $Y_{i,j} \sim \text{NegBin}(1/n, 1-p)$ independently with $p = \exp(-1/b)$.

**Then:**
$$\sum_{i=1}^{n} Z_i \sim \text{DLaplace}(b)$$

**Proof:**
By Theorem 2.1:
$$\sum_{i=1}^{n} Y_{i,1} \sim \text{NegBin}(1, 1-p) = \text{Geom}(p)$$
$$\sum_{i=1}^{n} Y_{i,2} \sim \text{NegBin}(1, 1-p) = \text{Geom}(p)$$

Since $\{Y_{i,1}\}$ and $\{Y_{i,2}\}$ are independent collections:
$$\sum_{i=1}^{n} Z_i = \sum_{i=1}^{n} (Y_{i,1} - Y_{i,2}) = \left(\sum_{i=1}^{n} Y_{i,1}\right) - \left(\sum_{i=1}^{n} Y_{i,2}\right) \sim \text{Geom}(p) - \text{Geom}(p) \sim \text{DLaplace}(b)$$

by Lemma 2.1. ∎

**Remark 2.1 (Independence Property):**
Crucially, Corollary 2.1 requires **NO coordination** between decryptors. Each decryptor $i$ independently samples $(Y_{i,1}, Y_{i,2})$ using their own random seed. The sum automatically yields the correct distribution.

---

## Section 3: Protocol Correctness

### Definition 3.1 (Protocol Output)

For each label $\ell$, the protocol computes:
$$\hat{H}(\ell) = \sum_{i \in A} (\lambda_i \cdot S_i(\ell) + Z_i(\ell)) \mod q$$

where:
- $A$ = set of active (non-failed) decryptors, $|A| \geq t$
- $\lambda_i$ = Lagrange coefficient for decryptor $i$
- $S_i(\ell)$ = aggregated share for label $\ell$ at decryptor $i$
- $Z_i(\ell)$ = noise contribution from decryptor $i$ for label $\ell$

### Theorem 3.1 (Reconstruction with Distributed Noise)

Let $\tilde{H}(\ell) = \sum_c \tilde{H}_c(\ell)$ be the sum of clipped histograms. Then:
$$\hat{H}(\ell) = \tilde{H}(\ell) + \sum_{i \in A} Z_i(\ell) \mod q$$

where $\sum_{i \in A} Z_i(\ell) \sim \text{DLaplace}(C/\varepsilon)$ when $|A|$ decryptors are active.

**Proof:**

**Step 1 (Share Aggregation):**
By Proposition 1.1, each decryptor $i$ receives shares of $\tilde{H}_c(\ell)$ from all clients $c$, and sums them:
$$S_i(\ell) = \sum_c f_c^{(\ell)}(i) \mod q$$
where $f_c^{(\ell)}(x)$ is client $c$'s Shamir polynomial for label $\ell$.

By additive homomorphism, $S_i(\ell)$ is a valid share of $\sum_c \tilde{H}_c(\ell) = \tilde{H}(\ell)$.

**Step 2 (Lagrange Reconstruction):**
By Theorem 1.2:
$$\sum_{i \in A} \lambda_i \cdot S_i(\ell) = \tilde{H}(\ell) \mod q$$

**Step 3 (Noise Addition):**
Each $Z_i(\ell) = \text{NegBin}(1/|A|, 1-p) - \text{NegBin}(1/|A|, 1-p)$ independently.
By Corollary 2.1, $\sum_{i \in A} Z_i(\ell) \sim \text{DLaplace}(b)$ with $b = C/\varepsilon$.

**Step 4 (Combining):**
$$\hat{H}(\ell) = \sum_{i \in A} (\lambda_i \cdot S_i(\ell) + Z_i(\ell)) = \sum_{i \in A} \lambda_i \cdot S_i(\ell) + \sum_{i \in A} Z_i(\ell) = \tilde{H}(\ell) + \text{DLaplace}(C/\varepsilon)$$
∎

---

## Section 4: Privacy Guarantees

### Definition 4.1 (Differential Privacy)

A mechanism $M: \mathcal{X} \to \mathcal{Y}$ is $(\varepsilon, \delta)$-differentially private if for all adjacent datasets $x, x' \in \mathcal{X}$ and all measurable $S \subseteq \mathcal{Y}$:
$$P(M(x) \in S) \leq e^\varepsilon \cdot P(M(x') \in S) + \delta$$

### Definition 4.2 (Adjacency for Site-Level DP)

Datasets $D$ and $D'$ are adjacent if they differ in the entire contribution of one client (site). That is, one client's histogram $H_c$ can be replaced with any other histogram $H'_c$.

### Lemma 4.1 (Sensitivity with Clipping)

After clipping each client's histogram to have L₁ norm at most $C$, the L₁ sensitivity of the aggregated histogram is $C$.

**Proof:**
Let $D$ and $D'$ differ in client $c$'s contribution. After clipping: $\|\tilde{H}_c\|_1 \leq C$ and $\|\tilde{H}'_c\|_1 \leq C$.

For client removal ($\tilde{H}'_c = 0$):
$$\sum_\ell |\tilde{H}(\ell) - \tilde{H}'(\ell)| = \|\tilde{H}_c\|_1 \leq C$$
∎

### Theorem 4.1 (Privacy of Discrete Laplace Mechanism)

The mechanism $M(H) = H + \text{DLaplace}(C/\varepsilon)$ for each label independently satisfies $(\varepsilon, 0)$-differential privacy for site-level adjacency.

**Proof:**
For adjacent datasets $D, D'$ differing in one site with clipped histograms $\tilde{H}, \tilde{H}'$:

For $Z_\ell \sim \text{DLaplace}(b)$ with $b = C/\varepsilon$:
$$\frac{P(Z_\ell = k)}{P(Z_\ell = k')} = \frac{p^{|k|}}{p^{|k'|}} = p^{|k| - |k'|}$$

where $p = \exp(-1/b) = \exp(-\varepsilon/C)$.

The probability ratio for the mechanism:
$$\frac{P(M(D) = h)}{P(M(D') = h)} = \prod_\ell \frac{P(Z_\ell = h_\ell - \tilde{H}(\ell))}{P(Z_\ell = h_\ell - \tilde{H}'(\ell))}$$

Taking logarithms and using the triangle inequality:
$$\log\frac{P(M(D) = h)}{P(M(D') = h)} = \sum_\ell (|h_\ell - \tilde{H}'(\ell)| - |h_\ell - \tilde{H}(\ell)|) \cdot \log(p)$$

By reverse triangle inequality: $|h - H'| - |h - H| \leq |H - H'|$

Therefore:
$$\leq \sum_\ell |\tilde{H}(\ell) - \tilde{H}'(\ell)| \cdot |\log(p)| = \sum_\ell |\tilde{H}(\ell) - \tilde{H}'(\ell)| \cdot \frac{\varepsilon}{C} \leq C \cdot \frac{\varepsilon}{C} = \varepsilon$$

Thus $P(M(D) = h) \leq e^\varepsilon \cdot P(M(D') = h)$. ∎

### Corollary 4.1 (Privacy of the Full Protocol)

The Heavy Hitters protocol satisfies $(\varepsilon, 0)$-differential privacy for site-level adjacency, as long as at least $t$ decryptors are honest.

**Proof:**
By Theorem 3.1, the output distribution is $\hat{H}(\ell) = \tilde{H}(\ell) + \text{DLaplace}(C/\varepsilon)$, which is exactly the Discrete Laplace mechanism. By Theorem 4.1, this is $(\varepsilon, 0)$-DP.

The distributed computation via Shamir SS does not leak additional information because individual shares reveal nothing (Theorem 1.1), and the final output is exactly the noisy histogram. ∎

---

## Section 5: Statistical Properties

### Proposition 5.1 (Unbiasedness Before Clamping)

$$\mathbb{E}[\hat{H}(\ell)] = \tilde{H}(\ell)$$

**Proof:**
$\hat{H}(\ell) = \tilde{H}(\ell) + Z(\ell)$ where $Z(\ell) \sim \text{DLaplace}(C/\varepsilon)$.
$\mathbb{E}[Z(\ell)] = 0$ by Definition 2.1.
Therefore, $\mathbb{E}[\hat{H}(\ell)] = \tilde{H}(\ell)$. ∎

### Proposition 5.2 (Variance)

$$\text{Var}(\hat{H}(\ell)) = \frac{2p}{(1-p)^2} \quad \text{where } p = \exp(-\varepsilon/C)$$

### Theorem 5.1 (Mean Squared Error)

For each label $\ell$:
$$\text{MSE}(\hat{H}(\ell)) = \text{Var}(Z(\ell)) + \text{Bias}^2(\ell)$$

where:
- $\text{Var}(Z(\ell)) = \frac{2 \exp(-\varepsilon/C)}{(1 - \exp(-\varepsilon/C))^2}$
- $\text{Bias}(\ell) = \tilde{H}(\ell) - H(\ell)$ (clipping bias, always $\leq 0$)

**Proof:**
$$\text{MSE}(\hat{H}(\ell)) = \mathbb{E}[(\hat{H}(\ell) - H(\ell))^2] = \mathbb{E}[(\tilde{H}(\ell) + Z(\ell) - H(\ell))^2]$$
$$= \mathbb{E}[Z(\ell)^2] + 2\mathbb{E}[Z(\ell)](\tilde{H}(\ell) - H(\ell)) + (\tilde{H}(\ell) - H(\ell))^2 = \text{Var}(Z(\ell)) + \text{Bias}^2(\ell)$$
∎

### Corollary 5.1 (Privacy-Utility Tradeoff)

As $\varepsilon \to \infty$ (less privacy), $\text{Var}(Z(\ell)) \to 0$.
As $\varepsilon \to 0$ (more privacy), $\text{Var}(Z(\ell)) \to \infty$.

For large $\varepsilon/C$:
$$\text{Var}(Z(\ell)) \approx \frac{2C^2}{\varepsilon^2}$$

---

## Section 6: Decryptor Failure Tolerance

### Theorem 7.1 (Byzantine Fault Tolerance)

With $n$ decryptors and threshold $t = \lceil 2n/3 \rceil$:
- The protocol tolerates up to $n - t = \lfloor n/3 \rfloor$ decryptor failures
- Reconstruction succeeds if $|A| \geq t$ decryptors are active

### Proposition 7.1 (Noise Scaling with Failures)

When $|A| < n$ decryptors are active, the noise distribution remains $\text{DLaplace}(C/\varepsilon)$, **not** $\text{DLaplace}(|A| \cdot C/(n \cdot \varepsilon))$.

**Proof:**
Each active decryptor samples $Z_i \sim \text{NegBin}(1/|A|, 1-p) - \text{NegBin}(1/|A|, 1-p)$.
By Corollary 2.1, $\sum_{i \in A} Z_i \sim \text{DLaplace}(C/\varepsilon)$ exactly.

The noise parameter adapts: $r = 1/|A|$ ensures the sum of $|A|$ terms equals the target $\text{DLaplace}(b)$ regardless of the number of failures. ∎

---

## Section 7: Estimation Quality Metrics

### Definition 8.1 (KL Divergence)

For true distribution $P$ and estimated distribution $Q$:
$$D_{KL}(P \| Q) = \sum_\ell P(\ell) \cdot \log\frac{P(\ell)}{Q(\ell)}$$

### Theorem 8.1 (Expected KL Divergence Bound)

Under mild conditions ($H(\ell) \gg \sqrt{\text{Var}(Z)}$ for all $\ell$):

$$\mathbb{E}[D_{KL}(P \| Q)] \approx \frac{\text{Var}(Z)}{2 \|H\|_1} \cdot \sum_\ell \frac{1}{P(\ell)}$$

This decreases as total count $\|H\|_1$ increases.

### Proposition 8.1 (TV Distance Bound)

$$\mathbb{E}[\text{TV}(P, Q)] \leq \frac{\sqrt{\text{Var}(Z)}}{2 \|H\|_1} \cdot \sqrt{K}$$

where $K$ is the number of labels.

---

## Section 8: Bias Mitigation for Downstream Tasks (e.g., FedLA)

When using noisy histogram estimates for downstream tasks like FedLA weight computation, bias arises from ratio estimation.

### Problem Statement

FedLA requires weights:
$$W(c_i, \ell) = \frac{h_i(\ell)}{H(\ell)}$$

But we only have noisy estimates $\hat{H}(\ell) = H(\ell) + Z(\ell)$, leading to:
$$\hat{W}(c_i, \ell) = \frac{h_i(\ell)}{\hat{H}(\ell)}$$

### Lemma 8.1 (Ratio Estimator Bias)

For $\hat{H} = H + Z$ with $\mathbb{E}[Z] = 0$ and $\text{Var}(Z) = \sigma^2$:
$$\mathbb{E}\left[\frac{1}{\hat{H}}\right] = \frac{1}{H} + \frac{\sigma^2}{H^3} + O\left(\frac{\sigma^4}{H^5}\right)$$

**Proof:**
Taylor expand $1/(H+Z)$ around $Z=0$:
$$\frac{1}{H+Z} = \frac{1}{H} - \frac{Z}{H^2} + \frac{Z^2}{H^3} - \frac{Z^3}{H^4} + \cdots$$

Taking expectations and using $\mathbb{E}[Z] = 0$, $\mathbb{E}[Z^2] = \sigma^2$, and $\mathbb{E}[Z^3] = 0$ (DLaplace is symmetric):
$$\mathbb{E}\left[\frac{1}{H+Z}\right] = \frac{1}{H} + \frac{\sigma^2}{H^3} + O\left(\frac{\sigma^4}{H^5}\right)$$
∎

### Corollary 8.1 (Weight Estimator Bias)

$$\mathbb{E}[\hat{W}(c_i, \ell)] \approx W(c_i, \ell) \cdot \left(1 + \frac{\sigma^2}{H(\ell)^2}\right)$$

The relative bias is $\sigma^2/H(\ell)^2$, which is large for rare labels.

### Theorem 8.2 (Bias-Corrected Weight Estimator)

The estimator:
$$\hat{W}_{\text{corrected}}(c_i, \ell) = \frac{h_i(\ell)}{\hat{H}(\ell)} \cdot \left(1 - \frac{\sigma^2}{\hat{H}(\ell)^2}\right)$$

has bias $O(\sigma^4/H^5)$, a significant improvement over the naive estimator.

**Proof:**
Substituting the bias-corrected inverse:
$$\frac{1}{\hat{H}} - \frac{\sigma^2}{\hat{H}^3} \approx \frac{1}{H} + \frac{\sigma^2}{H^3} - \frac{\sigma^2}{H^3} + O\left(\frac{\sigma^4}{H^5}\right) = \frac{1}{H} + O\left(\frac{\sigma^4}{H^5}\right)$$

The leading bias term cancels. ∎

### Proposition 8.2 (Practical Bias Correction)

For our protocol with $\sigma^2 = \frac{2p}{(1-p)^2}$ where $p = e^{-\varepsilon/C}$:

$$\hat{W}_{\text{corrected}}(c_i, \ell) = \frac{h_i(\ell)}{\hat{H}(\ell)} \cdot \left(1 - \frac{2p}{(1-p)^2 \cdot \hat{H}(\ell)^2}\right)$$

**Implementation note:** Compute $\sigma^2$ once from known parameters $\varepsilon$ and $C$.

### Alternative: Smoothed Estimator

Adding regularization to the denominator:
$$\hat{W}_{\text{smooth}}(c_i, \ell) = \frac{h_i(\ell)}{\hat{H}(\ell) + \alpha}$$

**Proposition 8.3:** With $\alpha = \sigma$, the smoothed estimator has:
- Reduced variance: $\text{Var}(\hat{W}_{\text{smooth}}) < \text{Var}(\hat{W})$
- Bounded bias: $|\text{Bias}| \leq \frac{\alpha \cdot h_i(\ell)}{H(\ell)^2}$

**Tradeoff:** Smoothing trades upward bias for downward bias, while reducing variance.

### Alternative: Thresholding

For labels where noise dominates signal:
$$\hat{W}_{\text{threshold}}(c_i, \ell) = \begin{cases}
\frac{h_i(\ell)}{\hat{H}(\ell)} & \text{if } \hat{H}(\ell) > k \cdot \sigma \\[1ex]
\frac{1}{N} & \text{otherwise}
\end{cases}$$

**Recommendation:** Use $k \in [3, 5]$. This avoids catastrophic errors for rare labels at the cost of falling back to uniform weighting.

### Comparison of Methods

| Method | Bias | Variance | Best For |
|--------|------|----------|----------|
| Naive $h_i/\hat{H}$ | $O(\sigma^2/H^2)$ | High for small $H$ | Large $H(\ell)$ |
| Bias-corrected | $O(\sigma^4/H^4)$ | Same as naive | Moderate $H(\ell)$ |
| Smoothed ($+\alpha$) | $O(\alpha/H)$ | Reduced | High variance regime |
| Thresholding | Zero (falls back) | N/A | Very small $H(\ell)$ |

### Recommendation

Use a **hybrid approach**:
1. If $\hat{H}(\ell) > 5\sigma$: Use bias-corrected estimator
2. If $2\sigma < \hat{H}(\ell) \leq 5\sigma$: Use smoothed estimator with $\alpha = \sigma$
3. If $\hat{H}(\ell) \leq 2\sigma$: Fall back to uniform weighting

This adapts to the signal-to-noise ratio per label.

---

## Summary

The Heavy Hitters protocol achieves:

1. **Correctness**: Output equals true clipped histogram plus $\text{DLaplace}(C/\varepsilon)$ noise
2. **Privacy**: $(\varepsilon, 0)$-differential privacy for site-level adjacency
3. **Distributed Trust**: No single decryptor learns the true histogram; $t = \lceil 2n/3 \rceil$ honest decryptors required
4. **Fault Tolerance**: Tolerates up to $\lfloor n/3 \rfloor$ decryptor failures
5. **Independence**: Each decryptor generates noise independently using only their own random seed
6. **Unbiasedness**: $\mathbb{E}[\hat{H}(\ell)] = \tilde{H}(\ell)$ (equals clipped true count)
