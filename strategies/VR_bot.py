# Name: Vedant Raut
# College: Newton School of Technology, ADYPU, Pune
# Roll Number: E25B070757

"""
VR_bot.py — Bayesian Market-Making with Inventory-Adjusted Reservation Pricing

Signal model:
    E[S | F_t] = k_mine + k̂_opp    where k̂_opp is inferred via:
      (a) Taker path:  k̂_opp ≈ 0.98 · clamp(mid(q), ±3σ_r)
      (b) Maker path:  k̂_opp ≈ 0.90 · clamp(mid(counter) − k_mine, ±3σ_r)
      (c) Foresight:   k̂_opp = (n_revealed / m) · Σ leak_i

Reservation price:   p* = E[S | F_t] − γ · I_t · (n_unseen / N)
    where I_t = Σ 1{long} − 1{short} is cumulative inventory.

Auction:  b_i = ⌊shade · v_i / α⌋   where α = TE_SALVAGE, shade ∈ {0.58, 0.62}.
"""

from __future__ import annotations
import math
import random
from typing import Any

# Empirical power value surface V(power, round) in ticks
V: dict[str, dict[int, float]] = {
    "FORESIGHT":    {1: 0.85, 2: 1.30, 3: 1.70, 4: 2.25, 5: 2.45},
    "TRICK_ROOM":   {1: 1.20, 2: 0.30, 3: 0.30, 4: 0.65, 5: 0.60},
    "SUBSTITUTE":   {1: 1.55, 2: 1.25, 3: 0.95, 4: 0.60, 5: 0.30},
    "STEALTH_ROCK": {1: 1.80, 2: 1.05, 3: 0.80, 4: 0.80, 5: 0.00},
    "TRANSFORM":    {1: 1.65, 2: 1.35, 3: 1.35, 4: 0.00, 5: 0.00},
}

# Hyperparameters (grid-search optimised)
SHADE       = 0.58   # first-price shading factor
DENIAL_W    = 0.40   # denial weight for TRANSFORM defence
FLAT_THRESH = 1      # |k| ≤ thresh → flat hand
OPP_FLAT    = 2.0    # opponent flatness threshold
GAMMA       = 0.22   # Avellaneda-Stoikov risk aversion γ
SUB_BONUS   = 0.35   # edge bonus when holding SUBSTITUTE
SUB_MARGIN  = 0.40   # margin required vs opponent SUBSTITUTE


class Bot:
    name: str = "VR_Bot"

    def reset(self, seat: int, config: Any, seed: int) -> None:
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)
        self._anchors: dict[int, float] = {}
        self._inv: int = 0

    # ── Signal estimation ──────────────────────────────────

    def _estimate_k_opp(self, obs: Any, q: tuple[int, int] | None = None) -> float:
        """Infer opponent's revealed-coin sum k̂_opp."""
        r = obs.round
        bound = 2.0 * math.sqrt(r)

        # Latch opponent signal from quote midpoints
        if q is not None and r not in self._anchors:
            mid = (q[0] + q[1]) / 2.0
            if not obs.is_maker:
                # Taker reads Maker's opening
                self._anchors[r] = max(-3*bound, min(3*bound, mid)) * 0.98
            else:
                # Maker reads Taker's counter: k_opp ≈ counter_mid − k_mine
                inferred = mid - float(obs.k_mine)
                self._anchors[r] = max(-3*bound, min(3*bound, inferred)) * 0.90

        anchor = self._anchors.get(r)

        # Foresight integration: scaled sample estimator
        if obs.foresight:
            m = len(obs.foresight)
            k_leak = float(sum(obs.foresight))
            n_rev = 4 * r
            if m >= n_rev:
                return k_leak
            scaled = k_leak * (n_rev / float(m))
            if anchor is not None:
                w = m / float(n_rev)
                return w * scaled + (1 - w) * anchor
            return scaled

        if anchor is not None:
            return anchor

        # Projection from earlier rounds with mean-reversion shrinkage
        earlier = [k for k in self._anchors if k < r]
        if earlier:
            lr = max(earlier)
            rate = self._anchors[lr] / float(4 * lr)
            return rate * (4 * r) * 0.75

        return 0.0

    def _reservation_price(self, obs: Any, q: tuple[int, int] | None = None) -> float:
        """p* = E[S|F_t] − γ · I · (n_unseen / N)"""
        k_opp = self._estimate_k_opp(obs, q)
        fair = float(obs.k_mine + k_opp)

        inv = sum(1 if c.long_seat == self.seat else -1 for c in obs.contracts)
        self._inv = inv

        n_unseen = max(1.0, float(40 - 4 * obs.round - len(obs.foresight or ())))
        return fair - inv * GAMMA * (n_unseen / 40.0)

    # ── Auction ────────────────────────────────────────────

    def _power_val(self, obs: Any, name: str) -> float:
        return V.get(name, {}).get(obs.round, 0.50)

    def _transform_val(self, obs: Any) -> float:
        v = self._power_val(obs, "TRANSFORM")
        if abs(obs.k_mine) <= FLAT_THRESH:
            return v
        earlier = [self._anchors[k] for k in self._anchors if k < obs.round]
        if earlier and abs(earlier[-1]) <= OPP_FLAT:
            return v * DENIAL_W
        return 0.0

    def bid(self, obs: Any, offered: list[str]) -> dict[str, int]:
        if not offered or obs.te_mine <= 0:
            return {}

        alpha = getattr(self.config, "TE_SALVAGE", 0.08)
        bids: dict[str, int] = {}

        for name in offered:
            v = self._transform_val(obs) if name == "TRANSFORM" else self._power_val(obs, name)
            if v <= 0:
                continue

            fair_te = v / alpha
            s = 0.62 if (name == "FORESIGHT" and obs.round >= 4) else SHADE
            b = int(math.floor(fair_te * s))

            # Budget snipe: never exceed te_theirs + 1
            if obs.te_theirs is not None and obs.te_theirs >= 0:
                b = min(b, obs.te_theirs + 1)

            b = max(0, min(b, obs.te_mine))
            if b > 0:
                bids[name] = b

        total = sum(bids.values())
        if total > obs.te_mine and total > 0:
            scale = obs.te_mine / float(total)
            bids = {k: int(math.floor(v * scale)) for k, v in bids.items()}
        return bids

    # ── Quoting ────────────────────────────────────────────

    def quote(self, obs: Any) -> tuple[int, int]:
        """Open at final_cap width centred on p*."""
        c = int(round(self._reservation_price(obs)))
        w = obs.final_cap
        lo = c - w // 2
        return (lo, lo + w)

    # ── Negotiation ────────────────────────────────────────

    def respond(self, obs: Any, quote: tuple[int, int], turn: int):
        bid_p, ask_p = quote
        p = self._reservation_price(obs, quote)

        sub_adj = SUB_BONUS if "SUBSTITUTE" in obs.powers_mine else 0.0
        opp_adj = SUB_MARGIN if "SUBSTITUTE" in obs.powers_theirs else 0.0

        edge_buy  = (p + sub_adj - opp_adj) - float(ask_p)
        edge_sell = float(bid_p) - (p - sub_adj + opp_adj)

        # Power shift on forced fill: δ = Σ mag_mine − Σ mag_theirs
        shift_mine = (3 if "TRICK_ROOM" in obs.powers_mine else 0) + \
                     (2 if "STEALTH_ROCK" in obs.powers_mine else 0)
        shift_opp  = (3 if "TRICK_ROOM" in obs.powers_theirs else 0) + \
                     (2 if "STEALTH_ROCK" in obs.powers_theirs else 0)
        delta = shift_mine - shift_opp

        # Final turn: compare forcing EV vs acceptance EV
        if turn >= obs.n_turns:
            forced_price = ((bid_p + ask_p) // 2) + delta
            fee = getattr(self.config, "FORCED_FILL_FEE", 2.0)
            pnl_force = float(forced_price) - p - fee
            best_accept = max(edge_buy, edge_sell)

            if best_accept >= pnl_force:
                if edge_buy >= edge_sell and edge_buy > -1.5:
                    return "ACCEPT_BUY"
                elif edge_sell > -1.5:
                    return "ACCEPT_SELL"

            w = max(obs.final_cap, (ask_p - bid_p) - getattr(self.config, "MIN_REDUCTION", 1))
            c = max(bid_p, min(int(round(p)), ask_p - w))
            return ("COUNTER", c, c + w)

        # Turns 2–5: accept on positive edge
        if edge_buy > 0.0 and edge_buy >= edge_sell:
            return "ACCEPT_BUY"
        if edge_sell > 0.0 and edge_sell > edge_buy:
            return "ACCEPT_SELL"

        red = getattr(self.config, "MIN_REDUCTION", 1)
        w = max(obs.final_cap, (ask_p - bid_p) - red)
        c = max(bid_p, min(int(round(p)), ask_p - w))
        return ("COUNTER", c, c + w)

    # ── TRANSFORM decision ─────────────────────────────────

    def use_transform(self, obs: Any) -> bool:
        """Fire iff |k_mine| ≤ threshold (flat hand → swap, decisive → decline)."""
        return abs(obs.k_mine) <= FLAT_THRESH
