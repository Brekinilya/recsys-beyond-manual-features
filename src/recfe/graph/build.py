"""Heterogeneous, time-aware interaction graph at a cutoff.

Nodes: users | items | attribute values (artist, album / genre, actor, ...) | user segment values.
Edges:
  user-item  one undirected edge per pair seen before the cutoff, carrying R_ui relation weights:
             for every event type t and time scale tau, w = sum over events of exp(-age / tau).
             The model mixes these relations with learned weights, i.e. it learns how much to trust
             recent vs old and positive vs negative interactions instead of a hand-picked window.
  item-attr, user-segment   static relations, weight 1.
Every relation is symmetrically normalised on its own (w / sqrt(deg_a * deg_b)).
Also returned: raw log-degrees per relation (node features) and, for the rows of this cutoff,
the raw weights of the (user, item) edge itself (pair features; zero if the edge is absent).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from recfe.datasets import DAY, Dataset

GRAPH_CFG = {
    "yambda": {
        "types": {"lplus": (pl.col("ev") == 0) & (pl.col("played") >= 50),
                  "skip": (pl.col("ev") == 0) & (pl.col("played") < 50),
                  "like": pl.col("ev") == 1,
                  "dislike": pl.col("ev") == 2},
        "scales": (3.0, 30.0, 300.0),
        "item_attrs": ["artist", "album"],
        "segs": [],
    },
    "kion": {
        "types": {"watch": pl.lit(True), "full": pl.col("pct") >= 50},
        "scales": (7.0, 30.0, 140.0),
        "item_attrs": ["genre", "country", "actor", "director", "studio", "ctype"],
        "segs": ["age", "income", "sex", "kids"],
    },
}


@dataclass
class Layout:
    n_users: int
    n_items: int
    attr_off: dict      # attr -> (offset, size)
    seg_off: dict
    n_nodes: int
    rel_names: list     # user-item relation names, then static relation names


def layout(ds: Dataset) -> Layout:
    cfg = GRAPH_CFG[ds.name]
    n_users = int(ds.events["user_id"].max()) + 1
    if ds.users is not None:
        n_users = max(n_users, int(ds.users["user_id"].max()) + 1)
    n_items = int(ds.items["item_id"].max()) + 1
    off = n_users + n_items
    attr_off, seg_off = {}, {}
    for a in cfg["item_attrs"]:
        size = int(ds.attrs[a].mapping[a].max()) + 1
        attr_off[a] = (off, size)
        off += size
    for s in cfg["segs"]:
        size = int(ds.users[s].max()) + 1
        seg_off[s] = (off, size)
        off += size
    rels = [f"{t}@{tau:g}" for t in cfg["types"] for tau in cfg["scales"]]
    rels += [f"item-{a}" for a in cfg["item_attrs"]] + [f"user-{s}" for s in cfg["segs"]]
    return Layout(n_users, n_items, attr_off, seg_off, off, rels)


def _sym_norm(w, a, b, na, nb):
    da = np.bincount(a, weights=w, minlength=na)
    db = np.bincount(b, weights=w, minlength=nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = w / np.sqrt(da[a] * db[b])
    return np.nan_to_num(out).astype(np.float32), da, db


def build(ds: Dataset, lay: Layout, cutoff: int, rows_c: pl.DataFrame | None = None) -> dict:
    cfg = GRAPH_CFG[ds.name]
    h = ds.history(cutoff)
    age = (cutoff - pl.col("ts")).cast(pl.Float32) / DAY
    aggs = []
    for t, cond in cfg["types"].items():
        for tau in cfg["scales"]:
            aggs.append((-age / tau).exp().filter(cond).sum().cast(pl.Float32).alias(f"{t}@{tau:g}"))
    rel_ui = [f"{t}@{tau:g}" for t in cfg["types"] for tau in cfg["scales"]]
    pairs = h.group_by("user_id", "item_id").agg(aggs)
    pairs = pairs.filter(pl.sum_horizontal(rel_ui) > 0)
    u = pairs["user_id"].to_numpy().astype(np.int64)
    i = pairs["item_id"].to_numpy().astype(np.int64)
    W = pairs.select(rel_ui).to_numpy().astype(np.float32)
    Wn = np.zeros_like(W)
    deg_u = np.zeros((lay.n_users, len(rel_ui)), np.float32)
    deg_i = np.zeros((lay.n_items, len(rel_ui)), np.float32)
    for r in range(len(rel_ui)):
        Wn[:, r], du, di = _sym_norm(W[:, r].astype(np.float64), u, i, lay.n_users, lay.n_items)
        deg_u[:, r], deg_i[:, r] = du, di

    # static relations
    st_src, st_dst, st_rel, st_w = [], [], [], []
    r0 = len(rel_ui)
    for k, a in enumerate(cfg["item_attrs"]):
        m = ds.attrs[a].mapping
        it = m["item_id"].to_numpy().astype(np.int64)
        av = m[a].to_numpy().astype(np.int64)
        w, _, _ = _sym_norm(np.ones(len(it)), it, av, lay.n_items, lay.attr_off[a][1])
        st_src.append(lay.n_users + it); st_dst.append(lay.attr_off[a][0] + av)
        st_rel.append(np.full(len(it), r0 + k, np.int16)); st_w.append(w)
    for k, s in enumerate(cfg["segs"]):
        us = ds.users.filter(pl.col(s) >= 0)
        uu = us["user_id"].to_numpy().astype(np.int64)
        sv = us[s].to_numpy().astype(np.int64)
        w, _, _ = _sym_norm(np.ones(len(uu)), uu, sv, lay.n_users, lay.seg_off[s][1])
        st_src.append(uu); st_dst.append(lay.seg_off[s][0] + sv)
        st_rel.append(np.full(len(uu), r0 + len(cfg["item_attrs"]) + k, np.int16)); st_w.append(w)

    g = {
        "ui_src": u.astype(np.int32), "ui_dst": (lay.n_users + i).astype(np.int32), "ui_w": Wn.astype(np.float16),
        "st_src": np.concatenate(st_src).astype(np.int32), "st_dst": np.concatenate(st_dst).astype(np.int32),
        "st_rel": np.concatenate(st_rel), "st_w": np.concatenate(st_w).astype(np.float32),
        "node_u": np.log1p(deg_u).astype(np.float16), "node_i": np.log1p(deg_i).astype(np.float16),
    }
    if rows_c is not None:
        pf = (rows_c.select("row_id", "user_id", "item_id")
                    .join(pairs, on=["user_id", "item_id"], how="left").sort("row_id"))
        g["pair"] = np.log1p(pf.select(rel_ui).fill_null(0).to_numpy()).astype(np.float16)
        g["row_id"] = pf["row_id"].to_numpy()
    return g
