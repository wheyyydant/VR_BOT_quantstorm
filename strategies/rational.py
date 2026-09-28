# Name: Quantstorm Reference Bot
# College: Quantstorm
# Roll Number: REF-000

"""
rational.py — Rational Pricing: E[S] = k_mine + k̂_opp via quote reading.

As Taker, infers k̂_opp from the Maker's opening quote midpoint.
No auction participation.
"""

import random


class Bot:
    name = "Rational"

    def reset(self, seat, config, seed):
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)
        self._anchor = {}

    def _get_anchor(self, obs, quote):
        r = obs.round
        if r not in self._anchor:
            if not obs.is_maker and quote is not None:
                self._anchor[r] = (quote[0] + quote[1]) / 2
            else:
                self._anchor[r] = 0.0
        return self._anchor[r]

    def _value(self, obs, quote=None):
        if obs.is_maker or quote is None:
            return float(obs.k_mine + sum(obs.foresight))
        anchor = self._get_anchor(obs, quote)
        if obs.foresight:
            anchor = 0.5 * anchor + 0.5 * sum(obs.foresight)
        return float(anchor + obs.k_mine)

    def bid(self, obs, offered):
        return {}

    def quote(self, obs):
        v = round(obs.k_mine + sum(obs.foresight))
        cap = obs.spread_cap
        lo = v - cap // 2
        return (lo, lo + cap)

    def respond(self, obs, quote, turn):
        bid, ask = quote
        v = self._value(obs, quote)
        edge_buy = v - ask
        edge_sell = bid - v

        if edge_buy > 0 and edge_buy >= edge_sell:
            return "ACCEPT_BUY"
        if edge_sell > 0:
            return "ACCEPT_SELL"

        w = max(0, (ask - bid) - self.config.MIN_REDUCTION)
        c = max(bid, min(round(v), ask - w))
        return ("COUNTER", c, c + w)

    def use_transform(self, obs):
        return abs(obs.k_mine) <= 1
