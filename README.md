# QuantStorm — Mathematical Foundations of Adversarial Market-Making Under Incomplete Information

> **A study in stochastic filtering, Bayesian inference on lattice distributions, mechanism design under energy constraints, and inventory-optimal execution in a zero-sum sequential game.**

---

## Table of Contents

1. [Core Theoretical Framework](#1-core-theoretical-framework)
2. [The Hidden Score — Stochastic Process on the Rademacher Lattice](#2-the-hidden-score--stochastic-process-on-the-rademacher-lattice)
3. [Bayesian Signal Model — Conditional Expectation and Posterior Updates](#3-bayesian-signal-model--conditional-expectation-and-posterior-updates)
4. [Maker-Taker Negotiation — Straddle Pricing on the Binomial Lattice](#4-maker-taker-negotiation--straddle-pricing-on-the-binomial-lattice)
5. [Maker Obligation — Information-Theoretic Transfer Pricing](#5-maker-obligation--information-theoretic-transfer-pricing)
6. [Auction Theory — First-Price Sealed Bidding with Energy Constraints](#6-auction-theory--first-price-sealed-bidding-with-energy-constraints)
7. [Inventory-Adjusted Reservation Pricing (Avellaneda-Stoikov)](#7-inventory-adjusted-reservation-pricing-avellaneda-stoikov)
8. [Power Valuation — Discrete Option Pricing on Game States](#8-power-valuation--discrete-option-pricing-on-game-states)
9. [Zero-Sum Invariant and Settlement Algebra](#9-zero-sum-invariant-and-settlement-algebra)
10. [Empirical Simulation Framework](#10-empirical-simulation-framework)
11. [Results and Measured Axes](#11-results-and-measured-axes)

---

## 1. Core Theoretical Framework

### The Fundamental Paradox

Two agents observe **disjoint private subsets** of a common random vector and must trade a contract whose payoff depends on the **full sum they can never individually observe**. The game reduces to:

> *How do you price a quantity you cannot compute, when the only additional information comes from an adversary whose incentive is to mislead you?*

This is a **zero-sum Bayesian game of incomplete information** — a stochastic filtering problem embedded inside a mechanism design problem.

### Mathematical Setup

Let the probability space be $(\Omega, \mathcal{F}, \mathbb{P})$. Define:

$$C = (c_1, c_2, \ldots, c_N) \in \{-1, +1\}^N, \quad c_i \stackrel{\text{iid}}{\sim} \text{Rademacher}\left(\tfrac{1}{2}\right)$$

The **hidden score** (the settlement value) is:

$$S = \sum_{i=1}^{N} c_i$$

The coin vector is partitioned into two **private hands**:

$$C = H_A \cup H_B, \quad |H_A| = |H_B| = n = \tfrac{N}{2}$$

Each player observes their hand through a sequence of **progressive reveals**:

$$\mathcal{F}_t^{(j)} = \sigma\left(H_j[1:\Delta t]\right), \quad \Delta = \frac{n}{R}$$

where $R$ is the number of rounds and $\Delta$ is the reveal rate per round.

---

## 2. The Hidden Score — Stochastic Process on the Rademacher Lattice

### Distribution of $S$

Since each $c_i$ is an independent Rademacher random variable:

$$S = \sum_{i=1}^{N} c_i = 2B - N, \quad B \sim \text{Binomial}(N, \tfrac{1}{2})$$

**Key distributional properties:**

| Property | Formula | Value ($N=40$) |
|----------|---------|----------------|
| Mean | $\mathbb{E}[S] = 0$ | $0$ |
| Variance | $\text{Var}(S) = N$ | $40$ |
| Std dev | $\sigma_S = \sqrt{N}$ | $\approx 6.32$ |
| Support | $\{-N, -N+2, \ldots, N-2, N\}$ | $\{-40, -38, \ldots, 40\}$ |
| Parity | $S \equiv N \pmod{2}$ always | $S$ is always even |

The **parity constraint** is load-bearing: $S$ lives on a sublattice of the integers. Any pricing model using a continuous normal approximation will misprice the straddle rate by up to 7 percentage points at certain widths — an exploitable arbitrage.

### The Revealed-Sum Decomposition

Define the **revealed partial sums**:

$$k_j^{(r)} = \sum_{i=1}^{\Delta r} H_j[i], \quad j \in \{A, B\}$$

Then at round $r$:

$$S = k_A^{(r)} + k_B^{(r)} + U_A^{(r)} + U_B^{(r)}$$

where $U_j^{(r)} = \sum_{i=\Delta r + 1}^{n} H_j[i]$ is the **unrevealed residual** for player $j$.

Each unrevealed coin is still $\text{Rademacher}(\frac{1}{2})$ conditioned on the reveal filtration, so:

$$U_j^{(r)} \sim 2\tilde{B}_j - (n - \Delta r), \quad \tilde{B}_j \sim \text{Bin}\left(n - \Delta r,\, \tfrac{1}{2}\right)$$

---

## 3. Bayesian Signal Model — Conditional Expectation and Posterior Updates

### Player A's Posterior on $S$

Player $A$ knows $k_A^{(r)}$ but not $k_B^{(r)}$. The conditional expectation:

$$\mathbb{E}\left[S \mid \mathcal{F}_r^{(A)}\right] = k_A^{(r)} + \underbrace{\mathbb{E}\left[k_B^{(r)}\right]}_{= 0} + \underbrace{\mathbb{E}\left[U_A^{(r)}\right]}_{= 0} + \underbrace{\mathbb{E}\left[U_B^{(r)}\right]}_{= 0} = k_A^{(r)}$$

So the **naïve estimator** is simply $\hat{S}_{\text{naive}} = k_{\text{mine}}$.

### Two-Way Bayesian Inference (The VR Bot Signal Engine)

The opponent's opening quote leaks information. An honest Maker centres on $k_{\text{theirs}}$, giving:

**Taker inference (Turn 2):**

$$\hat{k}_{\text{opp}} \approx \eta_T \cdot \text{clamp}\left(\frac{q_{\text{bid}} + q_{\text{ask}}}{2},\; \pm 3\sigma_r\right), \quad \eta_T = 0.98$$

**Maker inference (Turn 3):**

$$\hat{k}_{\text{opp}} \approx \eta_M \cdot \text{clamp}\left(\frac{c_{\text{bid}} + c_{\text{ask}}}{2} - k_{\text{mine}},\; \pm 3\sigma_r\right), \quad \eta_M = 0.90$$

where $\sigma_r = \sqrt{N - \Delta r}$ is the residual standard deviation.

The **posterior estimate** then becomes:

$$\hat{S}_{\text{Bayes}} = k_{\text{mine}} + \hat{k}_{\text{opp}}$$

### FORESIGHT Integration — Sample-Scaled Estimation

When holding a FORESIGHT leak of $m$ coins from the opponent's revealed $\Delta r$ coins with sum $k_{\text{leak}}$:

$$\hat{k}_{\text{opp}}^{(\text{FS})} = k_{\text{leak}} \cdot \frac{\Delta r}{m}$$

When both a quote anchor and a FORESIGHT sample are available, we blend:

$$\hat{k}_{\text{opp}} = \frac{m}{\Delta r} \cdot \hat{k}_{\text{opp}}^{(\text{FS})} + \left(1 - \frac{m}{\Delta r}\right) \cdot \hat{k}_{\text{opp}}^{(\text{quote})}$$

This is optimal in the MSE sense under the assumption that the FORESIGHT sample is uniformly drawn.

---

## 4. Maker-Taker Negotiation — Straddle Pricing on the Binomial Lattice

### The Straddle Probability

Given a quote $[b, a]$ with width $w = a - b$, the **straddle probability** is:

$$p_w^{(r)} = \mathbb{P}\left(b \leq S \leq a \mid \mathcal{F}_r^{(\text{maker})}\right)$$

Because the residual $S - k_{\text{mine}}$ lives on a lattice:

$$p_w^{(r)} = \sum_{\substack{j = -\lfloor w/2 \rfloor \\ j \equiv m \pmod{2}}}^{w - \lfloor w/2 \rfloor} \binom{m}{\frac{j+m}{2}} \cdot 2^{-m}$$

where $m = N - \Delta r$ is the number of coins the Maker has not seen.

### Why the Normal Approximation Fails

The CDF-based approximation $p_w \approx \Phi\left(\frac{w}{2\sigma}\right) - \Phi\left(\frac{-w}{2\sigma}\right)$ ignores the lattice structure and the parity constraint. At width $w = 4$ vs $w = 5$ with $m = 20$:

| Width | Lattice $p_w$ | Normal $p_w$ | Error |
|-------|---------------|--------------|-------|
| 4 | 0.4119 | 0.3829 | −2.9% |
| 5 | 0.4119 | 0.4647 | +5.3% |

The lattice shows $p_4 = p_5$ due to parity — a width-5 window holds exactly the same lattice points as width-4. The normal approximation sees an 8-point difference. **A bot preferring odd widths captures a parity arbitrage worth ~1 tick/deal.**

### Spread Schedule

The spread caps follow a **backward volatility schedule**:

$$w_{\text{final}}(r) = \max\left(1,\; \left\lfloor 4C\sqrt{R + 1 - r}\right\rceil\right)$$

$$w_{\text{cap}}(r) = w_{\text{final}}(r) + (T - 1) \cdot \delta_{\min}$$

where $C = 0.5$ is the spread coefficient and $T = 6$ is the number of negotiation turns.

---

## 5. Maker Obligation — Information-Theoretic Transfer Pricing

### The Obligation Mechanism

The Maker is charged/rewarded based on whether the score straddles its quote:

$$\Pi_{\text{obligation}} = \begin{cases}
+\lambda(1 - p_w) & \text{if } b \leq S \leq a \quad (\text{straddle}) \\
-\lambda \cdot p_w & \text{if } S \notin [b, a] \quad (\text{miss})
\end{cases}$$

where $\lambda = 3.0$ and $p_w = p_w^{(r)}$ is the straddle probability at the **width actually quoted**.

**Key property:** $\mathbb{E}[\Pi_{\text{obligation}}] = \lambda(1-p_w) \cdot p_w - \lambda p_w \cdot (1-p_w) = 0$

An honest Maker breaks even in expectation at **every width**. The transfer is paid by the Taker — zero-sum preserved.

### Width Premium

Additionally, the Maker pays a per-tick premium for quoting wider than the floor:

$$\Pi_{\text{width}} = -\rho \cdot \max\left(0,\; w - w_{\text{final}}(r)\right)$$

where $\rho = 0.22$ is the width premium. This prices the **option value** of a wide quote: harder to cross, more room to retreat.

### Combined Maker PnL from Obligation

$$\Pi_{\text{maker}}^{(r)} = \Pi_{\text{obligation}}^{(r)} + \Pi_{\text{width}}^{(r)}$$

---

## 6. Auction Theory — First-Price Sealed Bidding with Energy Constraints

### The Auction Model

Each round, powers $\mathcal{P}_r \subseteq \mathcal{P}$ are drawn from the eligible set. Both players submit blind bids:

$$b_j(p) \in \mathbb{Z}_{\geq 0}, \quad \sum_{p \in \mathcal{P}_r} b_j(p) \leq E_j$$

where $E_j$ is player $j$'s remaining Tactical Energy. The auction is **first-price sealed-bid**:

$$\text{Winner}(p) = \arg\max_j\; b_j(p), \quad \text{Cost} = b_{\text{winner}}(p)$$

Equal non-zero bids resolve by coin flip (the winner still pays).

### Optimal Shading

In a first-price auction with independent private values, the **Bayesian Nash equilibrium** bid function for value $v$ is:

$$b^*(v) = v - \frac{\int_0^v F(x)\,dx}{F(v)}$$

For symmetric bidders with uniform values on $[0, \bar{v}]$, this simplifies to $b^* = v/2$.

Our implementation uses a **state-conditional shade**:

$$b(p, r) = \left\lfloor \frac{V(p, r)}{\alpha} \cdot s \right\rfloor$$

where:
- $V(p, r)$ is the calibrated tick-value surface
- $\alpha = 0.08$ converts ticks to TE units
- $s \in \{0.58, 0.62\}$ is the empirically optimised shade factor

### Budget Sniping

An opponent with $E_{\text{opp}}$ remaining cannot bid more than $E_{\text{opp}}$, so:

$$b_{\text{snipe}}(p) = \min\left(b(p, r),\; E_{\text{opp}} + 1\right)$$

### TE Salvage and Opportunity Cost

Unspent TE has direct PnL value:

$$\Pi_{\text{salvage}} = \alpha \cdot (E_A - E_B)$$

So a power is worth buying only if its edge exceeds the opportunity cost: $V(p,r) > \alpha \cdot b(p,r)$.

---

## 7. Inventory-Adjusted Reservation Pricing (Avellaneda-Stoikov)

### The Reservation Price

Adapting the Avellaneda-Stoikov (2008) framework to discrete multi-round settlement:

$$p^* = \hat{S}_{\text{Bayes}} - \gamma \cdot I_t \cdot \frac{n_{\text{unseen}}}{N}$$

where:
- $I_t = \sum_{r'<r} \mathbb{1}\{\text{long}\} - \mathbb{1}\{\text{short}\}$ is the **cumulative inventory**
- $\gamma = 0.22$ is the **risk aversion coefficient**
- $\frac{n_{\text{unseen}}}{N}$ scales the adjustment by remaining uncertainty

### Intuition

A player already long 3 contracts faces **correlated tail risk** — if $S$ drops, all three lose simultaneously. The inventory skew pushes the reservation price **away** from the accumulated position, increasing the willingness to sell (reducing $p^*$) when long and increasing willingness to buy (raising $p^*$) when short.

This is the discrete analogue of the continuous-time result:

$$r(s, q, t) = s - q\gamma\sigma^2(T-t)$$

from Avellaneda-Stoikov, adapted to our setting where $\sigma^2$ is replaced by the fraction of remaining uncertainty.

---

## 8. Power Valuation — Discrete Option Pricing on Game States

### The Value Surface $V(p, r)$

Each power is a **state-contingent option**. Its value depends on the round and the information structure:

| Power | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Type |
|-------|---------|---------|---------|---------|---------|------|
| FORESIGHT | 0.85 | 1.30 | 1.70 | 2.25 | 2.45 | ↑ Information |
| TRICK_ROOM | 1.20 | 0.30 | 0.30 | 0.65 | 0.60 | Fill shift |
| SUBSTITUTE | 1.55 | 1.25 | 0.95 | 0.60 | 0.30 | ↓ Convexity |
| STEALTH_ROCK | 1.80 | 1.05 | 0.80 | 0.80 | — | Persistent |
| TRANSFORM | 1.65 | 1.35 | 1.35 | — | — | Hand swap |

### FORESIGHT: Information Value Grows with Precision

FORESIGHT reveals $m = 16$ of the opponent's $\Delta r$ revealed coins. Its value is:

$$V_{\text{FS}}(r) \propto \frac{\text{Var}(k_{\text{opp}}^{(r)} \mid \text{no leak}) - \text{Var}(k_{\text{opp}}^{(r)} \mid \text{leak})}{\text{Var}(k_{\text{opp}}^{(r)} \mid \text{no leak})}$$

In round 5, $\Delta r = 20$ and $m = 16$, so the leak covers 80% of the opponent's hand — near-complete information.

### SUBSTITUTE: Truncated Loss (Put Option)

SUBSTITUTE caps the holder's loss at $\kappa = 2$ ticks on that round's contract:

$$\Pi_{\text{sub}} = \max\left(\Pi_{\text{raw}},\; -\kappa\right)$$

This is equivalent to a **put option** with strike $\kappa$ below the contract price. Its value decreases with rounds because there are fewer remaining contracts to protect.

### TRANSFORM: Dual-Use Option

TRANSFORM grants the **option** (not the obligation) to swap hands:

$$\text{Value} = \begin{cases}
V_{\text{swap}} & \text{if } |k_{\text{mine}}| \leq \tau \quad (\text{flat hand: fire}) \\
V_{\text{swap}} \cdot w_{\text{denial}} & \text{if opponent appears flat (denial)} \\
0 & \text{otherwise}
\end{cases}$$

The power is **consumed** regardless of the decision — buying and declining is a **defence**, removing the option from the opponent's action space.

---

## 9. Zero-Sum Invariant and Settlement Algebra

### The Settlement Equation

At the end of each deal, all PnL components sum to zero:

$$\sum_{j \in \{A, B\}} \Pi_j = 0$$

where:

$$\Pi_j = \underbrace{\sum_r (S - p_r) \cdot \text{sgn}_j(r)}_{\text{contract PnL}} + \underbrace{\Pi_{\text{obligation}}^{(j)}}_{\text{maker}} + \underbrace{\Pi_{\text{fee}}^{(j)}}_{\text{forcing}} + \underbrace{\Pi_{\text{sub}}^{(j)}}_{\text{substitute}} + \underbrace{\alpha(E_j - E_{-j})}_{\text{salvage}}$$

Each component is a **transfer** — one player's gain is the other's loss. This is verified by assertion after every deal:

$$|\Pi_A + \Pi_B| < \epsilon = 10^{-9}$$

### Score Invariance Under Powers

TRANSFORM permutes the partition of $C$ between the seats but does not alter $C$ itself:

$$S = \sum H_A + \sum H_B = \sum H_B + \sum H_A = S'$$

This is verified after every power application.

---

## 10. Empirical Simulation Framework

### Mirrored Matches

To cancel **first-mover advantage** and **coin luck**, each deal is played twice:

1. **Direct:** $H_A = C[1:n],\; H_B = C[n+1:N]$
2. **Mirror:** $H_A = C[n+1:N],\; H_B = C[1:n]$ with roles inverted

A bot playing against itself nets exactly zero over a mirrored match. Measured edges are **differences from this baseline**.

### Deterministic Seed Derivation

All randomness is derived from a master seed via BLAKE2b:

$$\text{seed}(k_1, k_2, \ldots) = \text{int}_{64}\left(\text{BLAKE2b}(k_1 \| k_2 \| \cdots)\right)$$

This guarantees reproducibility while keeping the streams statistically independent.

### Measured Performance Axes

The three independent axes of bot performance, measured by ablation:

| Axis | Ablation Method | Measured Edge |
|------|----------------|---------------|
| **Auction** | Value bidder vs non-bidder (otherwise identical) | $+56.1 \pm 4.8$ ticks/match |
| **Quote reading** | Quote reader vs naïve pricer (otherwise identical) | $+31.0 \pm 7.0$ ticks/match |
| **Width choice** | Floor quoter vs cap quoter (otherwise identical) | $+22.1 \pm 5.3$ ticks/match |

### Running Simulations

```bash
# Duel two strategies
python backtester.py --bot1 strategies/VR_bot.py --bot2 strategies/naive_ev.py --seed 42

# Isolated tournament conditions
python backtester.py --bot1 strategies/VR_bot.py --bot2 strategies/rational.py --isolate

# Quick validation
python backtester.py --validate strategies/VR_bot.py
```

---

## 11. Results and Measured Axes

### VR Bot Performance Summary

Against reference bots over 1000-match samples ($n = 20{,}000$ deals):

| Opponent | VR Bot PnL/match | σ | Sharpe |
|----------|-----------------|---|--------|
| NaiveEV | $+115.5$ | $28.3$ | $4.08$ |
| Rational | $+62.8$ | $31.7$ | $1.98$ |
| AdaptiveBidder | $+18.4$ | $34.2$ | $0.54$ |

### Component Attribution

| Component | Contribution (ticks/match) | Mechanism |
|-----------|---------------------------|-----------|
| Auction participation | $+56.1$ | First-price shade at $s = 0.58$ |
| Two-way quote reading | $+31.0$ | Bayesian k̂_opp from quote midpoints |
| Width optimisation | $+22.1$ | Floor quoting ($w = w_{\text{final}}$) |
| Inventory adjustment | $+6.4$ | Avellaneda-Stoikov skewing ($\gamma = 0.22$) |

### Sensitivity Analysis

| Parameter | Range Tested | Optimal | Basin Width |
|-----------|-------------|---------|-------------|
| Shade $s$ | $[0.35, 0.75]$ | $0.58$ | $\pm 0.07$ |
| Risk aversion $\gamma$ | $[0.05, 0.50]$ | $0.22$ | $\pm 0.10$ |
| SUBSTITUTE bonus | $[0.0, 0.8]$ | $0.35$ | $\pm 0.15$ |

---

## References

1. **Avellaneda, M. & Stoikov, S.** (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217–224.
2. **Krishna, V.** (2010). *Auction Theory.* Academic Press. 2nd edition.
3. **Feller, W.** (1968). *An Introduction to Probability Theory and Its Applications, Vol. I.* Wiley. Ch. III (Rademacher sums).
4. **Harsanyi, J.C.** (1967). *Games with Incomplete Information Played by "Bayesian" Players.* Management Science, 14(3).
5. **Myerson, R.B.** (1981). *Optimal Auction Design.* Mathematics of Operations Research, 6(1).

---

<p align="center"><i>Built for the Divided Oracle competition — a zero-sum adversarial market-making game on the Rademacher lattice.</i></p>
