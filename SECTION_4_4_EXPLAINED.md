# Section 4.4 — Histogram Uncertainty and Estimation Trigger

## Step-by-step explanation with examples

---

## The Big Picture

We want to estimate the **global label distribution** across all hospitals (clients) in the federation. But we have three sources of error:

1. **Clipping** — each client samples only `C` data points (not their full dataset)
2. **Client sampling** — we don't see all `N` clients, only those that participated so far
3. **DP noise** — we add discrete Laplace noise for privacy

Section 4.4 derives a formula that tells us **how much total error** we have, and uses it to decide **when is the earliest round** we can trigger the histogram query and get a useful estimate.

---

## Step 1: Setup — What are we estimating?

Focus on a **single label** (e.g., "melanoma"). Each client `i` has a true proportion `p_i` of that label.

### Example: 5 hospitals, label = "melanoma"

| Client | Total samples | Melanoma samples | True proportion `p_i` |
|--------|--------------|------------------|-----------------------|
| H1     | 200          | 10               | 0.05                  |
| H2     | 150          | 3                | 0.02                  |
| H3     | 80           | 40               | 0.50 (specialist)     |
| H4     | 300          | 15               | 0.05                  |
| H5     | 120          | 6                | 0.05                  |

The **global mean proportion** across clients:

```
mu = (0.05 + 0.02 + 0.50 + 0.05 + 0.05) / 5 = 0.134
```

The **variance across clients** (how different hospitals are from each other):

```
S^2 = Var(p_i) = mean of (p_i - mu)^2
    = [(0.05-0.134)^2 + (0.02-0.134)^2 + (0.50-0.134)^2 + (0.05-0.134)^2 + (0.05-0.134)^2] / 5
    = [0.00706 + 0.01300 + 0.13396 + 0.00706 + 0.00706] / 5
    = 0.03363
```

---

## Step 2: The clipping step — what each client reports

Each client doesn't report their full histogram. They sample `C` points **with replacement** from their dataset and report the counts from that sample.

If client `i` has true proportion `p_i` for melanoma and draws `C` samples:

```
X_i ~ Binomial(C, p_i)       # number of melanoma in the C samples
p_hat_i = X_i / C            # estimated proportion
```

### Example: C = 20

Client H3 (specialist, `p_i = 0.50`):
- Expected melanoma count: `C * p_i = 20 * 0.50 = 10`
- But it's random! Could be 8, 9, 10, 11, 12...
- **Variance of p_hat_i**: `p_i(1 - p_i) / C = 0.50 * 0.50 / 20 = 0.0125`

Client H1 (general, `p_i = 0.05`):
- Expected: `20 * 0.05 = 1`
- **Variance of p_hat_i**: `0.05 * 0.95 / 20 = 0.002375`

**Key insight**: small `C` = more clipping noise. Large `C` = less noise but higher privacy sensitivity (need more DP noise).

---

## Step 3: Client sampling — we don't see everyone

In FL, each round samples a fraction `gamma` of clients. After `R` rounds, the **observed fraction** is:

```
alpha = fraction of distinct clients seen so far
K_obs = alpha * N
```

### Example: N = 5 clients, gamma = 0.4 (2 clients per round)

| Round | Clients sampled | Cumulative observed | alpha |
|-------|----------------|---------------------|-------|
| 1     | {H1, H3}       | {H1, H3}            | 2/5 = 0.40 |
| 2     | {H2, H5}       | {H1, H2, H3, H5}   | 4/5 = 0.80 |
| 3     | {H1, H4}       | {H1, H2, H3, H4, H5} | 5/5 = 1.00 |

Without replacement across rounds: `alpha = min(R * gamma, 1)` approximately.

With replacement: `alpha = 1 - (1 - gamma)^R`.

**If we don't see all clients, our average is biased by which clients we happened to see.**

---

## Step 4: The variance formula — derivation intuition

We want: `Var(p_hat - mu)` = how far is our estimate from the truth?

There are **two sources of randomness**:
1. **Which clients** were sampled (client sampling)
2. **What each client reports** given clipping (local sampling)

Using the **law of total variance**:

```
Var(p_hat - mu) = Var(E[p_hat - mu | clients seen])  +  E[Var(p_hat - mu | clients seen)]
                  \_________________________________/     \___________________________________/
                     client sampling term                    local clipping term
```

### Client sampling term

If we see `K_obs` out of `N` clients uniformly, the sample mean of their true `p_i` values has variance:

```
Var(mu_obs - mu) = (1 - alpha) * S^2 / K_obs
```

This is the **finite population correction** — if `alpha = 1` (all clients seen), this term vanishes.

### Local clipping term

Given which clients we see, the clipping randomness contributes:

```
E[Var(p_hat | obs)] = (1/K_obs) * (1/C) * E[p_i(1 - p_i)]
```

### Key identity

```
E[p_i(1 - p_i)] = mu(1 - mu) - S^2
```

This connects the local variance to the global statistics.

### Combining everything

```
Var(p_hat - mu) = (1/K_obs) * [(1 - alpha - 1/C) * S^2  +  (1/C) * mu(1 - mu)]
```

---

## Step 5: Normalized skew (lambda)

Define:

```
lambda = S^2 / [mu(1 - mu)]    in [0, 1]
```

This measures **how heterogeneous** the clients are for this label, normalized to [0, 1].

- **lambda = 0**: all clients have the same proportion (IID). `S^2 = 0`.
- **lambda = 1**: maximum skew. Some clients have all of this label, others have none.

### Example (continuing from above):

```
lambda = S^2 / [mu(1 - mu)]
       = 0.03363 / [0.134 * 0.866]
       = 0.03363 / 0.11604
       = 0.290
```

So melanoma has moderate skew (lambda = 0.29) — mostly because H3 is a specialist centre.

### The final variance formula

Substituting `S^2 = lambda * mu(1 - mu)`:

```
                    mu(1 - mu)   [ 1          (         1 ) ]
Var(p_hat - mu) = ------------ * [ --- + lambda( (1-alpha) - --- ) ]
                    K_obs        [ C          (         C ) ]
                                   ^^^          ^^^^^^^^^^^^
                                 clipping      client sampling
```

---

## Step 6: Understanding the two extremes

### Extreme 1: lambda = 0 (IID — all hospitals identical)

```
Var = mu(1-mu) / (C * K_obs)
```

Only clipping matters. More clients or larger `C` reduces variance. Client sampling doesn't matter because every client looks the same.

**Numerical example** (mu = 0.134, C = 20, K_obs = 3):
```
Var = 0.134 * 0.866 / (20 * 3) = 0.11604 / 60 = 0.001934
StdDev = 0.044
```

### Extreme 2: lambda = 1 (max skew — completely heterogeneous)

```
Var = mu(1-mu) * (1-alpha) / K_obs
```

Clipping doesn't matter anymore! Only **how many clients you've seen** matters. Even with perfect local reports (`C -> infinity`), you still have high variance if you've only seen a fraction of clients.

**Numerical example** (mu = 0.134, alpha = 0.4, K_obs = 2):
```
Var = 0.134 * 0.866 * (1-0.4) / 2 = 0.11604 * 0.6 / 2 = 0.03481
StdDev = 0.187
```

Much higher! You need to see more clients.

---

## Step 7: Adding DP noise

On top of the estimation variance, we add discrete Laplace noise with scale `b = C/epsilon`:

```
p = exp(-epsilon/C)
sigma^2_DP = 2p / (1-p)^2     (per label, normalized by K_obs^2 if needed)
```

### Example: C = 20, epsilon = 1.0

```
p = exp(-1.0/20) = exp(-0.05) = 0.9512
sigma^2_DP = 2 * 0.9512 / (1 - 0.9512)^2
           = 1.9025 / 0.002381
           = 799.0
```

This is the variance **in counts** (not proportions). To convert to proportion variance, divide by `K_obs^2 * C^2`:

```
sigma^2_DP_proportion = 799.0 / (K_obs * C)^2
```

With K_obs = 5, C = 20: `799.0 / 10000 = 0.0799` — this is very large! So small epsilon with small C means a lot of DP noise.

With C = 100, epsilon = 1.0:
```
p = exp(-0.01) = 0.99005
sigma^2_DP = 2*0.99005 / (0.00995)^2 = 1.9801 / 0.0000990 = 20001
sigma^2_DP_proportion = 20001 / (5*100)^2 = 20001 / 250000 = 0.0800
```

**Takeaway**: DP noise scales with `C/epsilon`. Larger `C` = more DP noise but less clipping noise. There's a **tradeoff**.

---

## Step 8: The estimation trigger — when to query?

### The idea

The server wants to wait until the estimate is "good enough" before triggering reconstruction. We define "good enough" as:

```
Total_Variance <= delta^2 * mu^2
```

where `delta` is the target relative precision (e.g., `delta = 0.10` means we want the estimate within 10% of the true value).

### Solving for alpha*

Rearranging:

```
mu(1-mu) / K_obs * [1/C + lambda((1-alpha) - 1/C)] + sigma^2_DP <= delta^2 * mu^2
```

Since `K_obs = alpha * N`, more clients observed = smaller variance. We solve for the minimum `alpha*` that satisfies this.

### Converting alpha* to rounds

With sampling rate gamma (fraction of clients per round):

**Without replacement across rounds**:
```
R* = ceil(alpha* / gamma)
```

**With replacement across rounds**:
```
alpha = 1 - (1-gamma)^R
R* = ceil(log(1-alpha*) / log(1-gamma))
```

### Full numerical example

**Setting**: N=100 clients, C=50, epsilon=2.0, gamma=0.10, delta=0.10

**For melanoma** (mu=0.134, lambda=0.29):

Step A — DP noise:
```
p = exp(-2.0/50) = exp(-0.04) = 0.9608
sigma^2_DP_counts = 2 * 0.9608 / (1-0.9608)^2 = 1.9216 / 0.001537 = 1250.3
sigma^2_DP_prop = 1250.3 / (K_obs * C)^2   (depends on K_obs)
```

Step B — Target:
```
delta^2 * mu^2 = 0.01 * 0.01796 = 0.0001796
```

Step C — Solve iteratively for alpha*:

| alpha | K_obs | Var(sampling+clipping) | Var(DP) | Total | Target | OK? |
|-------|-------|----------------------|---------|-------|--------|-----|
| 0.10  | 10    | 0.00893              | 0.00500 | 0.01393 | 0.0001796 | NO |
| 0.30  | 30    | 0.00248              | 0.00056 | 0.00304 | 0.0001796 | NO |
| 0.50  | 50    | 0.00129              | 0.00020 | 0.00149 | 0.0001796 | NO |
| 0.80  | 80    | 0.00060              | 0.00008 | 0.00068 | 0.0001796 | NO |
| 1.00  | 100   | 0.00002              | 0.00005 | 0.00007 | 0.0001796 | YES |

In this case, with such a small `mu` and tight `delta`, we need almost all clients. This is realistic: **rare classes are hard to estimate precisely**.

Now relax `delta = 0.30` (30% relative error is acceptable):
```
delta^2 * mu^2 = 0.09 * 0.01796 = 0.001616
```

| alpha | K_obs | Total Var | Target   | OK? |
|-------|-------|-----------|----------|-----|
| 0.30  | 30    | 0.00304   | 0.001616 | NO  |
| 0.50  | 50    | 0.00149   | 0.001616 | YES |

So `alpha* = 0.50`, meaning:
```
R* = ceil(0.50 / 0.10) = 5 rounds
```

**After 5 rounds of standard FedAvg, we trigger the histogram query.**

---

## Step 9: Why this matters — zero overhead

The key insight:

1. Rounds 1 to R* are **standard FedAvg** — you'd run them anyway
2. Clients generate their Shamir shares **once** at initialization
3. Decryptors just **buffer shares** during warm-up (no computation)
4. At round R*, the server says "reconstruct now" — one message
5. From round R*+1 onward: distribution-aware aggregation

**No extra rounds. No extra communication. No extra privacy cost.**

The histogram query costs exactly `epsilon` of privacy budget, spent **once**, at the moment it's most useful.

---

## Summary table

| Symbol | Meaning | Example value |
|--------|---------|---------------|
| `N` | Total number of clients | 100 |
| `K_obs` | Clients observed so far | `alpha * N` |
| `alpha` | Fraction of clients observed | 0.5 |
| `C` | Clipping threshold (samples per client) | 50 |
| `epsilon` | Privacy budget | 2.0 |
| `gamma` | Client sampling rate per round | 0.10 |
| `mu` | True global proportion for a label | 0.134 |
| `S^2` | Variance of `p_i` across clients | 0.0336 |
| `lambda` | Normalised skew = `S^2 / [mu(1-mu)]` | 0.29 |
| `delta` | Target relative precision | 0.10-0.30 |
| `R*` | Optimal trigger round | `ceil(alpha*/gamma)` |

## Intuitions

- **More skew (lambda -> 1)**: need more clients observed -> more rounds
- **Less skew (lambda -> 0)**: clipping dominates, C matters more than alpha
- **Rarer class (small mu)**: harder to estimate precisely -> more rounds or relax delta
- **Larger C**: less clipping noise, but more DP noise (sensitivity = C)
- **Larger epsilon**: less DP noise, but weaker privacy
- **Larger gamma**: see clients faster -> fewer rounds needed
