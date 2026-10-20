"""Random generation and mutation of grammar features."""
from __future__ import annotations

import math
import random

from recfe.features.spec import Feature, Grammar, Ratio, Spec

AGG_WEIGHTS = {"cnt": 0.34, "dec": 0.16, "last": 0.14, "mean": 0.12, "sum": 0.04, "first": 0.06,
               "nuniq": 0.07, "days": 0.07}
# Yambda: played ratio exists only for listens
VALUE_EVENTS = {"yambda": {"listen", "lplus", "skip", "rec", "recplus", "recskip", "org"}}
NEIGHBOR_KEYS = {
    "user": ["ua:*"], "item": ["a:*", "s:*"], "ui": ["ua:*", "item"],
    "a:*": ["item", "a:*", "ua:*"], "ua:*": ["ui", "ua:*", "user", "a:*"], "s:*": ["item", "s:*"],
}


class Generator:
    def __init__(self, g: Grammar, dataset: str, seed: int = 0):
        self.g = g
        self.dataset = dataset
        self.rng = random.Random(seed)
        self.value_events = VALUE_EVENTS.get(dataset, set(g.events))

    def window(self) -> int:
        lo, hi = math.log(self.g.min_window), math.log(self.g.max_window)
        return max(self.g.min_window, min(self.g.max_window, round(math.exp(self.rng.uniform(lo, hi)))))

    def hl(self) -> float:
        return round(math.exp(self.rng.uniform(math.log(0.5), math.log(self.g.max_window / 2))), 1)

    def red(self, key: str) -> str | None:
        attr = key.split(":")[1] if ":" in key else None
        return self.rng.choice(self.g.reducers) if attr in self.g.multi_attrs else None

    def valid(self, s: Spec) -> bool:
        if s.agg in ("mean", "sum") and s.event not in self.value_events:
            return False
        # a (user, item) pair has exactly one counterpart, nuniq is constant there
        return not (s.agg == "nuniq" and s.key == "ui")

    def concrete_key(self, pattern: str) -> str:
        if pattern.endswith("*"):
            opts = [k for k in self.g.keys if k.startswith(pattern[:-1])]
            return self.rng.choice(opts) if opts else self.rng.choice(self.g.keys)
        return pattern if pattern in self.g.keys else self.rng.choice(self.g.keys)

    @staticmethod
    def key_class(key: str) -> str:
        return key if key in ("user", "item", "ui") else key.split(":")[0] + ":*"

    def spec(self, key=None, event=None, agg=None) -> Spec:
        for _ in range(100):
            k = key or self.rng.choice(self.g.keys)
            e = event or self.rng.choice(self.g.events)
            a = agg or self.rng.choices(list(AGG_WEIGHTS), weights=list(AGG_WEIGHTS.values()))[0]
            w = None if self.rng.random() < (0.6 if a == "dec" else 0.2) else self.window()
            s = Spec(k, e, a, w,
                     value=self.rng.choice(self.g.values) if a in ("mean", "sum") else None,
                     hl=self.hl() if a == "dec" else None,
                     red=self.red(k))
            if self.valid(s):
                return s
        return Spec(key or "user", self.g.events[0], "cnt", 7)

    def ratio(self, base: Spec | None = None) -> Ratio:
        kind = self.rng.choice(["conv", "conv", "share", "trend"])
        b = base or self.spec(agg="cnt")
        w = b.window if b.window is not None else self.window()
        if kind == "conv":
            pairs = [p for p in self.g.conversions if b.event in p] or self.g.conversions
            e1, e2 = self.rng.choice(pairs)
            return Ratio(Spec(b.key, e1, "cnt", w, red=b.red), Spec(b.key, e2, "cnt", w, red=b.red))
        if kind == "share":
            num_keys = [k for k in self.g.keys if k.startswith(("ua:", "s:")) or k == "ui"]
            k = b.key if b.key in num_keys else self.rng.choice(num_keys)
            den = "item" if k.startswith("s:") else "user"
            return Ratio(Spec(k, b.event, "cnt", w, red=self.red(k) if k != b.key else b.red),
                         Spec(den, b.event, "cnt", w))
        w2 = self.window()
        w1, w2 = sorted((w, w2)) if w != w2 else (max(1, w // 3), w)
        return Ratio(Spec(b.key, b.event, "cnt", w1, red=b.red), Spec(b.key, b.event, "cnt", w2, red=b.red))

    def random(self) -> Feature:
        return self.ratio() if self.rng.random() < 0.25 else self.spec()

    def mutate_spec(self, s: Spec) -> Spec:
        for _ in range(50):
            op = self.rng.choice(["window", "window", "window", "agg", "event", "key", "red"])
            k, e, a, w, v, h, r = s.key, s.event, s.agg, s.window, s.value, s.hl, s.red
            if op == "window":
                if a == "dec" and self.rng.random() < 0.6:
                    h = round(max(0.5, min(self.g.max_window, h * math.exp(self.rng.gauss(0, 0.6)))), 1)
                elif w is None or self.rng.random() < 0.1:
                    w = self.window() if w is None else None
                else:
                    w = max(self.g.min_window, min(self.g.max_window, round(w * math.exp(self.rng.gauss(0, 0.6)))))
            elif op == "agg":
                a = self.rng.choices(list(AGG_WEIGHTS), weights=list(AGG_WEIGHTS.values()))[0]
                v = self.rng.choice(self.g.values) if a in ("mean", "sum") else None
                h = (round(w / 2, 1) if w else self.hl()) if a == "dec" else None
            elif op == "event":
                e = self.rng.choice(self.g.events)
            elif op == "key":
                k = self.concrete_key(self.rng.choice(NEIGHBOR_KEYS[self.key_class(k)]))
                r = self.red(k)
            elif op == "red":
                r = self.red(k)
            m = Spec(k, e, a, w, v, h, r)
            if m != s and self.valid(m):
                return m
        return self.spec()

    def mutate(self, f: Feature) -> Feature:
        if isinstance(f, Ratio):
            x = self.rng.random()
            if x < 0.5:  # shift the shared window
                w = f.num.window or self.window()
                w2 = max(self.g.min_window, min(self.g.max_window, round(w * math.exp(self.rng.gauss(0, 0.6)))))
                scale = w2 / w
                den_w = f.den.window
                if den_w is not None and f.num.window is not None and den_w != f.num.window:  # trend ratio
                    den_w = max(w2 + 1, min(self.g.max_window, round(den_w * scale)))
                else:
                    den_w = w2 if den_w is not None else None
                return Ratio(Spec(f.num.key, f.num.event, f.num.agg, w2, f.num.value, f.num.hl, f.num.red),
                             Spec(f.den.key, f.den.event, f.den.agg, den_w, f.den.value, f.den.hl, f.den.red))
            if x < 0.8:
                return self.ratio(f.num)
            return f.num
        if self.rng.random() < 0.2 and f.agg == "cnt":
            return self.ratio(f)
        return self.mutate_spec(f)
