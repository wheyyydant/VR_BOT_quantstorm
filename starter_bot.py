# Name: [Your Name Here]
# College: [Your College Here]
# Roll Number: [Your Roll Number Here]

"""
starter_bot.py — Template. Implement the five methods below.

Observable fields (obs.*):
    seat, round, my_revealed, k_mine, te_mine, te_theirs, spread_cap,
    final_cap, is_maker, powers_mine, powers_theirs, auction_log,
    contracts, foresight, n_unknown_both, n_turns
"""

import random


class Bot:
    name = "StarterBot"

    def reset(self, seat: int, config, seed: int) -> None:
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)

    def bid(self, obs, offered: list) -> dict:
        return {}

    def quote(self, obs) -> tuple:
        v = obs.k_mine
        cap = obs.spread_cap
        return (v - cap // 2, v + (cap - cap // 2))

    def respond(self, obs, quote: tuple, turn: int):
        bid, ask = quote
        v = obs.k_mine

        if v > ask:
            return "ACCEPT_BUY"
        if v < bid:
            return "ACCEPT_SELL"

        w = max(0, (ask - bid) - self.config.MIN_REDUCTION)
        c = max(bid, min(round(v), ask - w))
        return ("COUNTER", c, c + w)

    def use_transform(self, obs) -> bool:
        return abs(obs.k_mine) <= 1
