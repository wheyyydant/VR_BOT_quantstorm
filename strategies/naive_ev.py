# Name: Quantstorm Reference Bot
# College: Quantstorm
# Roll Number: REF-000

"""
naive_ev.py — Baseline: E[S] = k_mine (no opponent inference, no auction).

The zero-shot archetype. Prices S using only private information.
"""

import random


class Bot:
    name = "NaiveEV"

    def reset(self, seat, config, seed):
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)

    def bid(self, obs, offered):
        return {}

    def quote(self, obs):
        v = obs.k_mine + sum(obs.foresight)
        cap = obs.spread_cap
        lo = round(v) - cap // 2
        return (lo, lo + cap)

    def respond(self, obs, quote, turn):
        bid, ask = quote
        v = obs.k_mine + sum(obs.foresight)

        if v > ask:
            return "ACCEPT_BUY"
        if v < bid:
            return "ACCEPT_SELL"

        w = max(0, (ask - bid) - self.config.MIN_REDUCTION)
        c = max(bid, min(round(v), ask - w))
        return ("COUNTER", c, c + w)

    def use_transform(self, obs):
        return abs(obs.k_mine) <= 1
