"""Engine vs brute force on random rows, plus a leakage check.

Run from the project root: python tests/test_engine.py
"""
import math
import time

import numpy as np
import polars as pl

from recfe.datasets import DAY, load
from recfe.features.engine import Engine
from recfe.features.spec import R, S
from recfe.paths import processed


def brute(ds, row, spec):
    c = row["cutoff"]
    h = ds.events.filter(pl.col("ts") < c)
    if spec.window is not None:
        h = h.filter(pl.col("ts") >= c - spec.window * DAY)
    f = ds.event_filters[spec.event]
    if f is not None:
        h = h.filter(f)
    key = spec.key
    if key == "user":
        h = h.filter(pl.col("user_id") == row["user_id"])
    elif key == "item":
        h = h.filter(pl.col("item_id") == row["item_id"])
    elif key == "ui":
        h = h.filter((pl.col("user_id") == row["user_id"]) & (pl.col("item_id") == row["item_id"]))
    elif key.startswith("s:"):
        seg = key[2:]
        v = ds.users.filter(pl.col("user_id") == row["user_id"])[seg][0]
        h = h.filter((pl.col(seg) == v) & (pl.col("item_id") == row["item_id"]))
    else:
        a = ds.attrs[key.split(":")[1]]
        vals = a.mapping.filter(pl.col("item_id") == row["item_id"])[a.name].to_list()
        if not a.on_events:
            h = h.join(a.mapping, on="item_id")
        out = []
        for v in vals:
            hv = h.filter(pl.col(a.name) == v)
            if key.startswith("ua:"):
                hv = hv.filter(pl.col("user_id") == row["user_id"])
            out.append(agg(ds, hv, spec, c))
        out = [x for x in out if x is not None]
        if not out:
            return 0.0 if spec.fill_zero else None
        red = {"max": max, "sum": sum, "mean": lambda x: sum(x) / len(x), "min": min}[spec.red or "max"]
        return red(out)
    return agg(ds, h, spec, c)


def agg(ds, h, spec, c):
    if h.height == 0:
        return 0.0 if spec.fill_zero else None
    age = (c - h["ts"].to_numpy()) / DAY
    if spec.agg == "cnt":
        return float(h.height)
    if spec.agg == "last":
        return float(age.min())
    if spec.agg == "first":
        return float(age.max())
    if spec.agg == "dec":
        return float(np.exp(-age * math.log(2) / spec.hl).sum())
    if spec.agg == "mean":
        return float(h.select(ds.values[spec.value]).to_series().mean())
    if spec.agg == "sum":
        return float(h.select(ds.values[spec.value]).to_series().sum())
    raise ValueError(spec.agg)


def check(name, specs, n=25):
    ds = load(name)
    rows = pl.read_parquet(processed(name) / "rows.parquet")
    sample = rows.sample(n, seed=1).sort("row_id")
    eng = Engine(ds)
    t = time.perf_counter()
    feats = eng.compute(specs, sample)
    print(f"{name}: engine {time.perf_counter() - t:.1f}s for {n} rows x {len(specs)} specs")
    bad = 0
    for i, row in enumerate(sample.iter_rows(named=True)):
        for s in specs:
            a = feats[s.name][i]
            b = brute(ds, row, s)
            ok = (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-3 * max(1, abs(b)))
            if not ok:
                bad += 1
                print("MISMATCH", s.name, row["row_id"], a, b)
    assert bad == 0, bad
    # leakage: events at or after the cutoff must not change anything
    c = sample["cutoff"].min()
    ds2 = load(name)
    ds2.events = ds2.events.filter(pl.col("ts") < c)
    ds2.__post_init__()
    first = sample.filter(pl.col("cutoff") == c)
    f1 = Engine(ds).compute(specs, first)
    f2 = Engine(ds2).compute(specs, first)
    assert f1.equals(f2), "features changed after removing future events"
    print(f"{name}: OK ({n} rows, {len(specs)} specs, leakage check passed)")


if __name__ == "__main__":
    check("yambda", [S("ui", "listen", "cnt", 30), S("user", "rec", "cnt", 7), S("item", "lplus", "cnt", 1),
                     S("ua:artist", "listen", "last"), S("a:album", "like", "cnt", 30),
                     S("user", "listen", "mean", 30, "played"), S("ui", "listen", "dec", None, hl=7.0),
                     S("item", "listen", "first")])
    check("kion", [S("user", "watch", "cnt", 30), S("item", "full", "cnt", 7), S("ua:genre", "watch", "cnt", None, red="sum"),
                   S("ua:actor", "watch", "cnt", None, red="max"), S("a:genre", "watch", "cnt", 30, red="mean"),
                   S("s:age", "watch", "cnt", 30), S("ua:genre", "watch", "last", red="min"),
                   S("item", "watch", "mean", 30, "pct")])
