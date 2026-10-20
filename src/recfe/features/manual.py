"""Hand-crafted feature sets (the strong baseline) and static base features.

Hand-crafted features are written in the same grammar the automatic search uses, so the
comparison isolates the question "who chooses keys, events, aggregations and windows":
an engineer (round windows 1/7/30/all, familiar ratios) or a search procedure.
"""
from __future__ import annotations

import polars as pl

from recfe.datasets import Dataset
from recfe.features.spec import R, S


def manual_yambda():
    f = []
    # user activity and taste for recommendations
    f += [S("user", "listen", "cnt", w) for w in (1, 7, 30)]
    f += [S("user", "rec", "cnt", 7), S("user", "rec", "cnt", 30)]
    f += [R(S("user", "recplus", "cnt", w), S("user", "rec", "cnt", w)) for w in (7, 30)]
    f += [R(S("user", "lplus", "cnt", 30), S("user", "listen", "cnt", 30)),
          R(S("user", "org", "cnt", 30), S("user", "listen", "cnt", 30))]
    f += [S("user", "listen", "mean", 30, "played"), S("user", "like", "cnt", 30), S("user", "dislike", "cnt", 30),
          S("user", "listen", "last"), S("user", "listen", "nuniq", 30)]
    # item popularity and quality
    f += [S("item", "listen", "cnt", w) for w in (1, 7, 30, None)]
    f += [S("item", "rec", "cnt", 7)]
    f += [R(S("item", "recplus", "cnt", w), S("item", "rec", "cnt", w)) for w in (7, 30)]
    f += [R(S("item", "lplus", "cnt", 30), S("item", "listen", "cnt", 30)),
          R(S("item", "like", "cnt", 30), S("item", "listen", "cnt", 30))]
    f += [S("item", "like", "cnt", 30), S("item", "dislike", "cnt", 30), S("item", "listen", "first"),
          S("item", "listen", "nuniq", 30)]
    # artist / album popularity and quality
    f += [S("a:artist", "listen", "cnt", 7), S("a:artist", "listen", "cnt", 30), S("a:artist", "like", "cnt", 30),
          R(S("a:artist", "recplus", "cnt", 30), S("a:artist", "rec", "cnt", 30)),
          S("a:album", "listen", "cnt", 30)]
    # direct user x item links
    f += [S("ui", "listen", "cnt", w) for w in (7, 30, None)]
    f += [S("ui", "lplus", "cnt"), S("ui", "skip", "cnt", 30), S("ui", "listen", "last"),
          S("ui", "listen", "mean", None, "played"), S("ui", "like", "cnt"), S("ui", "unlike", "cnt"),
          S("ui", "dislike", "cnt"), S("ui", "rec", "cnt", 30),
          R(S("ui", "recplus", "cnt", 30), S("ui", "rec", "cnt", 30))]
    # indirect links: user x artist, user x album
    f += [S("ua:artist", "listen", "cnt", w) for w in (7, 30, None)]
    f += [S("ua:artist", "lplus", "cnt", 30),
          R(S("ua:artist", "skip", "cnt", 30), S("ua:artist", "listen", "cnt", 30)),
          S("ua:artist", "like", "cnt"), S("ua:artist", "dislike", "cnt"), S("ua:artist", "listen", "last"),
          R(S("ua:artist", "listen", "cnt", 30), S("user", "listen", "cnt", 30))]
    f += [S("ua:album", "listen", "cnt", 30), S("ua:album", "listen", "cnt"), S("ua:album", "like", "cnt"),
          S("ua:album", "listen", "last")]
    return f


def manual_kion():
    f = []
    # user
    f += [S("user", "watch", "cnt", w) for w in (7, 30, None)]
    f += [S("user", "watch", "mean", 30, "pct"), S("user", "watch", "mean", None, "pct"),
          S("user", "watch", "sum", 30, "hours"), S("user", "watch", "last"), S("user", "watch", "first"),
          R(S("user", "full", "cnt"), S("user", "watch", "cnt"))]
    # item popularity, trend, quality, novelty
    f += [S("item", "watch", "cnt", w) for w in (1, 7, 14, 30, None)]
    f += [R(S("item", "watch", "cnt", 7), S("item", "watch", "cnt", 30)),
          S("item", "watch", "mean", 30, "pct"), S("item", "watch", "mean", None, "pct"),
          R(S("item", "full", "cnt", 30), S("item", "watch", "cnt", 30)),
          S("item", "watch", "first"), S("item", "watch", "last")]
    # item popularity inside the user's demographic segment
    f += [S(f"s:{s}", "watch", "cnt", 30) for s in ("age", "sex", "income", "kids")]
    f += [R(S("s:age", "watch", "cnt", 30), S("item", "watch", "cnt", 30))]
    # attribute popularity
    f += [S("a:genre", "watch", "cnt", 30, red="mean"), S("a:actor", "watch", "cnt", 30, red="max"),
          S("a:director", "watch", "cnt", 30, red="max")]
    # indirect user x attribute links
    f += [S("ua:genre", "watch", "cnt", None, red="sum"), S("ua:genre", "watch", "cnt", None, red="max"),
          S("ua:genre", "watch", "cnt", 30, red="max"),
          R(S("ua:genre", "watch", "cnt", None, red="mean"), S("user", "watch", "cnt")),
          S("ua:genre", "watch", "mean", None, "pct", red="mean"),
          S("ua:genre", "watch", "last", red="min"),
          S("ua:actor", "watch", "cnt", None, red="sum"), S("ua:director", "watch", "cnt", None, red="sum"),
          S("ua:country", "watch", "cnt", None, red="max"), S("ua:studio", "watch", "cnt", None, red="max"),
          R(S("ua:ctype", "watch", "cnt"), S("user", "watch", "cnt"))]
    return f


MANUAL = {"yambda": manual_yambda, "kion": manual_kion}

KION_CAND = ["als_rank", "als_score", "pop_rank", "in_als", "in_pop"]


def base_features(ds: Dataset, rows: pl.DataFrame) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Static attributes (+ first-stage scores for KION). Shared by every ranker variant."""
    if ds.name == "yambda":
        out = rows.select("row_id", "item_id").join(ds.items.select("item_id", "length"), on="item_id", how="left")
        cols, cats = ["length"], []
    else:
        out = (rows.select(["row_id", "user_id", "item_id"] + KION_CAND)
                   .join(ds.users.select("user_id", "age", "income", "sex", "kids"), on="user_id", how="left")
                   .join(ds.items.select("item_id", "ctype", "year", "for_kids", "age_rating"), on="item_id", how="left"))
        cats = ["age", "income", "sex", "ctype"]
        cols = KION_CAND + ["age", "income", "sex", "kids", "ctype", "year", "for_kids", "age_rating"]
    out = out.select(["row_id"] + [pl.col(c).cast(pl.Float32) for c in cols]).sort("row_id")
    return out, cols, cats
