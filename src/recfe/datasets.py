"""Dataset definitions: event table, named event filters, value columns, item attributes, user segments."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from recfe.features.spec import Grammar
from recfe.paths import processed

DAY = 86400


@dataclass
class Attr:
    name: str
    mapping: pl.DataFrame  # item_id, <name>
    multi: bool
    on_events: bool  # the event table already carries this column


@dataclass
class Dataset:
    name: str
    events: pl.DataFrame  # sorted by ts
    items: pl.DataFrame
    users: pl.DataFrame | None
    event_filters: dict[str, pl.Expr | None]
    values: dict[str, pl.Expr]
    attrs: dict[str, Attr]
    segs: list[str]
    grammar: Grammar
    _ts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._ts = self.events["ts"].to_numpy()
        # unknown attribute values (-1) must not form one giant pseudo-artist / pseudo-genre
        for a in self.attrs.values():
            a.mapping = a.mapping.filter(pl.col(a.name) >= 0)

    def history(self, cutoff: int, start: int | None = None) -> pl.DataFrame:
        """Events with start <= ts < cutoff. Zero-copy slice of the time-sorted table."""
        hi = int(np.searchsorted(self._ts, cutoff, side="left"))
        lo = 0 if start is None else int(np.searchsorted(self._ts, start, side="left"))
        return self.events.slice(lo, hi - lo)


def load_yambda() -> Dataset:
    p = processed("yambda")
    ev = pl.read_parquet(p / "events.parquet")
    items = pl.read_parquet(p / "items.parquet")
    listen = pl.col("ev") == 0
    plus = pl.col("played") >= 50
    rec = pl.col("org") == 0
    filters = {
        "any": None,
        "listen": listen,
        "lplus": listen & plus,               # listened > 50% (Listen+ in the Yambda paper)
        "skip": listen & ~plus,
        "rec": listen & rec,                  # recommendation-driven listen = impression
        "recplus": listen & rec & plus,
        "recskip": listen & rec & ~plus,
        "org": listen & ~rec,                 # organic listen: user picked the track
        "like": pl.col("ev") == 1,
        "dislike": pl.col("ev") == 2,
        "unlike": pl.col("ev") == 3,
        "undislike": pl.col("ev") == 4,
    }
    values = {"played": pl.col("played").cast(pl.Float32).clip(0, 100)}
    attrs = {
        "artist": Attr("artist", items.select("item_id", "artist"), multi=False, on_events=True),
        "album": Attr("album", items.select("item_id", "album"), multi=False, on_events=True),
    }
    grammar = Grammar(
        keys=["user", "item", "ui", "a:artist", "a:album", "ua:artist", "ua:album"],
        events=["listen", "lplus", "skip", "rec", "recplus", "recskip", "org", "like", "dislike", "unlike"],
        values=["played"],
        max_window=300,
        conversions=[("lplus", "listen"), ("skip", "listen"), ("recplus", "rec"), ("recskip", "rec"),
                     ("org", "listen"), ("like", "listen"), ("dislike", "listen"), ("unlike", "like"),
                     ("like", "lplus"), ("dislike", "rec")],
    )
    return Dataset("yambda", ev, items, None, filters, values, attrs, [], grammar)


def load_kion() -> Dataset:
    p = processed("kion")
    ev = pl.read_parquet(p / "events.parquet")
    items = pl.read_parquet(p / "items.parquet")
    users = pl.read_parquet(p / "users.parquet")
    long = pl.read_parquet(p / "item_attrs.parquet")
    filters = {
        "watch": None,
        "full": pl.col("pct") >= 50,          # watched at least half
        "drop": pl.col("pct") < 10,           # abandoned at the start
        "long": pl.col("dur") >= 1800,        # at least 30 minutes in total
    }
    values = {
        "pct": pl.col("pct").clip(0, 100),
        "hours": pl.col("dur").cast(pl.Float32) / 3600,
    }
    attrs = {}
    for a in ["genre", "country", "actor", "director", "studio", "ctype"]:
        m = long.filter(pl.col("attr") == a).select("item_id", pl.col("value").alias(a))
        attrs[a] = Attr(a, m, multi=a != "ctype", on_events=False)
    segs = ["age", "income", "sex", "kids"]
    grammar = Grammar(
        keys=["user", "item"] + [f"a:{a}" for a in attrs] + [f"ua:{a}" for a in attrs] + [f"s:{s}" for s in segs],
        events=["watch", "full", "drop", "long"],
        values=["pct", "hours"],
        multi_attrs={a for a, v in attrs.items() if v.multi},
        max_window=140,
        conversions=[("full", "watch"), ("drop", "watch"), ("long", "watch")],
    )
    return Dataset("kion", ev, items, users, filters, values, attrs, segs, grammar)


def load(name: str) -> Dataset:
    return {"yambda": load_yambda, "kion": load_kion}[name]()
