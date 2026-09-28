"""
game_config.py — Game parameters, derived quantities, and frozen data types.

All tunable constants live here. Derived quantities (spread caps, straddle
rates) are computed from the lattice distribution of the residual score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import comb, sqrt
from types import MappingProxyType
from typing import Any, NamedTuple


class ConfigError(Exception):
    pass


@lru_cache(maxsize=512)
def _lattice_straddle(m: int, w: int) -> float:
    """P(|X| ≤ w/2) where X = 2B − m, B ~ Bin(m, 1/2).

    Exact lattice probability that a width-w window centred on the posterior
    mean contains the residual score. X has the same parity as m.
    """
    if m <= 0:
        return 1.0
    total = 0
    for j in range(-(w // 2), w - w // 2 + 1):
        if (j - m) % 2:
            continue
        k = (j + m) // 2
        if 0 <= k <= m:
            total += comb(m, k)
    return total / (1 << m)


# ═══════════════════════════════════════════════════════════════
#  GAME CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class GameConfig:
    """Frozen parameter set Θ for the Divided Oracle game.

    Handed to every bot via reset(seat, config, seed). Read-only after
    construction — mutation raises ConfigError.
    """

    _frozen = False

    def __init__(self, **overrides):
        self._overrides: dict[str, Any] = dict(overrides)

        # ── Asset structure ────────────────────────────────
        self.N_COINS: int = 40               # |C| = 2·N_PRIVATE
        self.N_PRIVATE: int = 20             # coins per player
        self.N_ROUNDS: int = 5               # rounds per deal
        self.REVEAL_PER_ROUND: int = 4       # Δ coins revealed/round

        # ── Match structure ────────────────────────────────
        self.DEALS_PER_PHASE: int = 10

        # ── Tactical Energy (TE) ───────────────────────────
        self.TE_BUDGET: int = 24
        self.TE_SALVAGE: float = 0.08        # PnL per unspent TE point

        # ── Auction ────────────────────────────────────────
        self.TIE_RULE: str = "coin_flip"

        # ── Negotiation ────────────────────────────────────
        self.N_TURNS: int = 6
        self.MIN_REDUCTION: int = 1
        self.SPREAD_C: float = 0.5

        # ── Fees & obligations ─────────────────────────────
        self.FORCED_FILL_FEE: float = 2.0
        self.MAKER_OBLIGATION: float = 3.0
        self.WIDTH_PREMIUM: float = 0.22

        # ── Midpoint rule ──────────────────────────────────
        self.MIDPOINT_SIDE_RULE: str = "last_quoter_sells"

        # ── Powers P ───────────────────────────────────────
        self.POWERS: dict[str, dict[str, Any]] = {
            "FORESIGHT":    {"magnitude": 16, "rounds": (1, 2, 3, 4, 5), "once_per_deal": False},
            "TRICK_ROOM":   {"magnitude": 3,  "rounds": (1, 2, 3, 4, 5), "once_per_deal": False},
            "SUBSTITUTE":   {"magnitude": 2,  "rounds": (1, 2, 3, 4, 5), "once_per_deal": False},
            "STEALTH_ROCK": {"magnitude": 2,  "rounds": (1, 2, 3, 4),    "once_per_deal": True},
            "TRANSFORM":    {"magnitude": 0,  "rounds": (1, 2, 3),       "once_per_deal": True},
        }
        self.SLOTS_PER_ROUND: int = 1
        self.SLATE_MODE: str = "draw"
        self.POWERS_PER_ROUND: int = 2

        # ── Time limits ────────────────────────────────────
        self.TIME_BUDGET_MS: float = 2.0
        self.HARD_TIME_LIMIT_MS: float = 50.0

        # ── Enforcement ────────────────────────────────────
        self.MAX_TIME_VIOLATIONS: int = 5
        self.FORFEIT_PNL: float = 250.0
        self.BOT_MEMORY_LIMIT_MB: int = 512
        self.MATCH_TIMEOUT_S: float = 600.0

        # ── Internal ───────────────────────────────────────
        self.ZERO_SUM_TOL: float = 1e-9
        self.MAX_REASONABLE_VALUE: int = 10_000

        for key, value in overrides.items():
            if not hasattr(self, key):
                raise ConfigError(f"Unknown config parameter: {key!r}")
            object.__setattr__(self, key, value)

        self._validate()

        object.__setattr__(self, "POWERS", MappingProxyType({
            name: MappingProxyType({
                k: (tuple(v) if k == "rounds" else v)
                for k, v in spec.items()
            })
            for name, spec in self.POWERS.items()
        }))
        object.__setattr__(self, "_frozen", True)

    # ── Immutability ───────────────────────────────────────

    def __setattr__(self, key, value):
        if self._frozen:
            raise ConfigError(f"GameConfig is read-only: cannot set {key!r}.")
        object.__setattr__(self, key, value)

    def __delattr__(self, key):
        raise ConfigError(f"GameConfig is read-only: cannot delete {key!r}.")

    def __reduce__(self):
        return (_rebuild_config, (self._overrides,))

    # ── Validation ─────────────────────────────────────────

    def _validate(self):
        if self.TIE_RULE not in ("coin_flip", "refund"):
            raise ConfigError(f"Unknown TIE_RULE: {self.TIE_RULE!r}")
        if self.MAX_TIME_VIOLATIONS < 1:
            raise ConfigError("MAX_TIME_VIOLATIONS must be >= 1")
        if self.FORFEIT_PNL < 0:
            raise ConfigError("FORFEIT_PNL must be non-negative")
        if self.HARD_TIME_LIMIT_MS < self.TIME_BUDGET_MS:
            raise ConfigError("HARD_TIME_LIMIT_MS must be >= TIME_BUDGET_MS")
        if self.N_COINS != 2 * self.N_PRIVATE:
            raise ValueError(f"N_COINS must == 2·N_PRIVATE")
        if self.REVEAL_PER_ROUND * self.N_ROUNDS != self.N_PRIVATE:
            raise ValueError(f"REVEAL_PER_ROUND·N_ROUNDS must == N_PRIVATE")
        if self.N_TURNS < 2:
            raise ValueError("N_TURNS must be >= 2")
        if self.MIN_REDUCTION < 1:
            raise ValueError("MIN_REDUCTION must be >= 1")
        if self.TE_BUDGET < 0:
            raise ValueError("TE_BUDGET must be non-negative")
        if self.TE_SALVAGE < 0:
            raise ValueError("TE_SALVAGE must be non-negative")
        if self.DEALS_PER_PHASE < 1:
            raise ValueError("DEALS_PER_PHASE must be >= 1")
        if self.MAKER_OBLIGATION < 0:
            raise ValueError("MAKER_OBLIGATION must be non-negative")
        if self.WIDTH_PREMIUM < 0:
            raise ValueError("WIDTH_PREMIUM must be non-negative")
        if self.FORCED_FILL_FEE < 0:
            raise ValueError("FORCED_FILL_FEE must be non-negative")
        if self.MIDPOINT_SIDE_RULE not in ("last_quoter_sells", "maker_sells", "coin_flip"):
            raise ValueError(f"Unknown MIDPOINT_SIDE_RULE: {self.MIDPOINT_SIDE_RULE!r}")
        if self.POWERS_PER_ROUND < 1:
            raise ValueError("POWERS_PER_ROUND must be >= 1")
        if self.SLATE_MODE not in ("draw", "fixed"):
            raise ConfigError(f"Unknown SLATE_MODE: {self.SLATE_MODE!r}")
        if not (1 <= self.SLOTS_PER_ROUND <= self.POWERS_PER_ROUND):
            raise ValueError("SLOTS_PER_ROUND must be in [1, POWERS_PER_ROUND]")

        for name, spec in self.POWERS.items():
            if not name.isupper() or not name.replace("_", "").isalpha():
                raise ValueError(f"Power name must be UPPER_SNAKE: {name!r}")
            if "magnitude" not in spec or "rounds" not in spec:
                raise ValueError(f"Power {name!r} missing 'magnitude' or 'rounds'")
            if "once_per_match" in spec:
                raise ConfigError(f"Power {name!r}: 'once_per_match' renamed to 'once_per_deal'")
            for r in spec["rounds"]:
                if not (1 <= r <= self.N_ROUNDS):
                    raise ValueError(f"Power {name!r}: round {r} outside 1..{self.N_ROUNDS}")

        for r in range(1, self.N_ROUNDS + 1):
            eligible = [n for n, s in self.POWERS.items() if r in s["rounds"]]
            if self.SLATE_MODE == "fixed" and len(eligible) > self.POWERS_PER_ROUND:
                raise ValueError(f"Round {r}: {len(eligible)} powers exceeds POWERS_PER_ROUND")

    # ── Derived quantities ─────────────────────────────────
    # σ_r = √(N − Δ·r),  cap_final(r) = max(1, ⌊4·C·√(N+1−r)⌋)

    def final_cap(self, r: int) -> int:
        """max(1, round(4·C·√(N_ROUNDS+1−r)))"""
        self._check_round(r)
        return max(1, round(4 * self.SPREAD_C * sqrt(self.N_ROUNDS + 1 - r)))

    def spread_cap(self, r: int) -> int:
        """final_cap + (N_TURNS − 1)·MIN_REDUCTION"""
        return self.final_cap(r) + (self.N_TURNS - 1) * self.MIN_REDUCTION

    def unknown_to_both(self, r: int) -> int:
        """N − 2·Δ·r"""
        self._check_round(r)
        return self.N_COINS - 2 * self.REVEAL_PER_ROUND * r

    def residual_sd(self, r: int) -> float:
        """σ_r = √(N − Δ·r)"""
        self._check_round(r)
        return sqrt(self.N_COINS - self.REVEAL_PER_ROUND * r)

    def open_width_floor(self, r: int) -> int:
        return self.final_cap(r)

    def straddle_prob(self, r: int, width: int = None, unseen: int = None) -> float:
        """P(bid ≤ S ≤ ask) on the exact binomial lattice."""
        self._check_round(r)
        w = self.spread_cap(r) if width is None else int(width)
        m = self.N_COINS - self.REVEAL_PER_ROUND * r if unseen is None else int(unseen)
        return _lattice_straddle(max(0, m), max(0, w))

    def baseline_straddle(self, r: int) -> float:
        return self.straddle_prob(r, self.spread_cap(r))

    def offered_powers(self, r: int) -> list[str]:
        self._check_round(r)
        return [name for name, spec in self.POWERS.items() if r in spec["rounds"]]

    def draw_slate(self, rng) -> dict[int, tuple[str, ...]]:
        """Draw the full deal's power slate from `rng`."""
        if self.SLATE_MODE == "fixed":
            return {r: tuple(self.offered_powers(r)) for r in range(1, self.N_ROUNDS + 1)}

        placed_once: set[str] = set()
        out: dict[int, tuple[str, ...]] = {}
        for r in range(1, self.N_ROUNDS + 1):
            pool = [n for n in self.offered_powers(r) if n not in placed_once]
            drawn = rng.sample(pool, min(self.SLOTS_PER_ROUND, len(pool)))
            placed_once |= {n for n in drawn if self.POWERS[n]["once_per_deal"]}
            out[r] = tuple(drawn)
        return out

    def sigma_score(self) -> float:
        """σ_S = √N"""
        return sqrt(self.N_COINS)

    def _check_round(self, r: int):
        if not (1 <= r <= self.N_ROUNDS):
            raise ValueError(f"Round {r} outside 1..{self.N_ROUNDS}")

    def __repr__(self) -> str:
        return (f"GameConfig(N={self.N_COINS}, R={self.N_ROUNDS}, "
                f"TE={self.TE_BUDGET}, D={self.DEALS_PER_PHASE})")


# ═══════════════════════════════════════════════════════════════
#  DEFAULT
# ═══════════════════════════════════════════════════════════════

def _rebuild_config(overrides: dict) -> GameConfig:
    return GameConfig(**overrides)

DEFAULT_CONFIG = GameConfig()


# ═══════════════════════════════════════════════════════════════
#  DATA TYPES (frozen = immutable)
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Obs:
    """Observable state vector o_t for a bot at round t."""
    seat: int
    round: int
    my_revealed: tuple[int, ...]
    te_mine: int
    te_theirs: int
    spread_cap: int
    final_cap: int
    is_maker: bool
    powers_mine: frozenset
    powers_theirs: frozenset
    auction_log: tuple
    contracts: tuple
    foresight: tuple
    n_unknown_both: int
    n_turns: int

    @property
    def k_mine(self) -> int:
        """k = Σ c_i for revealed coins."""
        return sum(self.my_revealed)


class Contract(NamedTuple):
    """Settlement contract from one round's negotiation."""
    round: int
    price: int
    long_seat: int
    forced: bool
    forcer: int = -1
    shift: int = 0
    maker_seat: int = -1
    open_bid: int = 0
    open_ask: int = 0


@dataclass
class DealResult:
    pnl: tuple
    score: int
    contracts: tuple
    te_left: tuple
    logs: list = field(default_factory=list)


@dataclass
class MatchResult:
    pnl: tuple
    deals: list = field(default_factory=list)
    bot_a_times: list = field(default_factory=list)
    bot_b_times: list = field(default_factory=list)
    bot_a_warnings: list = field(default_factory=list)
    bot_b_warnings: list = field(default_factory=list)
    bot_a_violations: int = 0
    bot_b_violations: int = 0
    bot_a_clamps: int = 0
    bot_b_clamps: int = 0
    forfeits: tuple = (False, False)
