"""Point-in-time computation of grammar features.

For every distinct cutoff c in the rows table, only events with ts < c are visible
(`Dataset.history` slices the time-sorted log), so neither the target event nor anything
after it can leak into a feature. Specs that share (key, event) are computed in one
group_by pass; windows are applied inside the aggregation as filters on event age.
"""
from __future__ import annotations

import math
from collections import defaultdict

import polars as pl

from recfe.datasets import DAY, Dataset
from recfe.features.spec import Feature, Ratio, Spec

LN2 = math.log(2)


class Engine:
    def __init__(self, ds: Dataset):
        self.ds = ds

    def _expr(self, s: Spec, key: str) -> pl.Expr:
        age = pl.col("_age")
        # window [c - w days, c): ages in (0, w]
        mask = age <= s.window if s.window is not None else None

        def sel(e):
            return e.filter(mask) if mask is not None else e

        if s.agg == "cnt":
            e = sel(age).count().cast(pl.Float32)
        elif s.agg == "sum":
            e = sel(self.ds.values[s.value]).sum()
        elif s.agg == "mean":
            e = sel(self.ds.values[s.value]).mean()
        elif s.agg == "last":
            e = sel(age).min()
        elif s.agg == "first":
            e = sel(age).max()
        elif s.agg == "dec":
            e = sel((-age * (LN2 / s.hl)).exp()).sum()
        elif s.agg == "nuniq":
            other = "item_id" if key == "user" or key.startswith("ua:") else "user_id"
            e = sel(pl.col(other)).n_unique().cast(pl.Float32)
        elif s.agg == "days":
            e = sel(age.floor()).n_unique().cast(pl.Float32)
        else:
            raise ValueError(s.agg)
        return e.cast(pl.Float32).alias(s.name)

    def _group(self, key: str, event: str, specs: list[Spec], c: int, rc: pl.DataFrame) -> pl.DataFrame:
        ds = self.ds
        wins = [s.window for s in specs]
        start = None if any(w is None for w in wins) else c - max(wins) * DAY
        h = ds.history(c, start)
        f = ds.event_filters[event]
        if f is not None:
            h = h.filter(f)

        multi = False
        if key in ("user", "item", "ui"):
            kcols = {"user": ["user_id"], "item": ["item_id"], "ui": ["user_id", "item_id"]}[key]
            rx = rc
        elif key.startswith(("a:", "ua:")):
            attr = ds.attrs[key.split(":")[1]]
            multi = attr.multi
            a = attr.name
            rx = rc.join(attr.mapping, on="item_id", how="inner")
            if not attr.on_events:
                h = h.join(attr.mapping, on="item_id", how="inner")
            kcols = [a] if key.startswith("a:") else ["user_id", a]
        elif key.startswith("s:"):
            seg = key[2:]
            rx = rc.join(ds.users.select("user_id", seg), on="user_id", how="left")
            kcols = [seg, "item_id"]
        else:
            raise ValueError(key)

        h = h.join(rx.select(kcols).unique(), on=kcols, how="semi")
        h = h.with_columns(((c - pl.col("ts")).cast(pl.Float32) / DAY).alias("_age"))
        stats = h.group_by(kcols).agg([self._expr(s, key) for s in specs])
        names = [s.name for s in specs]
        out = rx.join(stats, on=kcols, how="left")
        if multi:
            red = {"max": pl.Expr.max, "mean": pl.Expr.mean, "sum": pl.Expr.sum, "min": pl.Expr.min}
            fills = [pl.col(s.name).fill_null(0) if s.fill_zero else pl.col(s.name) for s in specs]
            out = out.with_columns(fills).group_by("row_id").agg(
                [red[s.red or "max"](pl.col(s.name)).alias(s.name) for s in specs])
        out = rc.select("row_id").join(out.select(["row_id"] + names), on="row_id", how="left")
        out = out.with_columns([pl.col(s.name).fill_null(0) for s in specs if s.fill_zero])
        return out.sort("row_id")

    def compute(self, feats: list[Feature], rows: pl.DataFrame, verbose: bool = False) -> pl.DataFrame:
        """rows: row_id, user_id, item_id, cutoff. Returns row_id + one Float32 column per feature, sorted by row_id."""
        base: dict[str, Spec] = {}
        for f in feats:
            for s in f.base():
                base[s.name] = s
        groups: dict[tuple[str, str], list[Spec]] = defaultdict(list)
        for s in base.values():
            groups[(s.key, s.event)].append(s)

        parts = []
        for c in sorted(rows["cutoff"].unique().to_list()):
            rc = rows.filter(pl.col("cutoff") == c).select("row_id", "user_id", "item_id").sort("row_id")
            cols = [rc.select("row_id")]
            for (key, event), specs in groups.items():
                cols.append(self._group(key, event, specs, c, rc).drop("row_id"))
            parts.append(pl.concat(cols, how="horizontal"))
            if verbose:
                print(f"  cutoff day {c / DAY:.0f}: {rc.height} rows")
        out = pl.concat(parts).sort("row_id")

        res = [pl.col("row_id")]
        for f in feats:
            if isinstance(f, Ratio):
                num, den = pl.col(f.num.name), pl.col(f.den.name)
                res.append(pl.when(den > 0).then(num / den).otherwise(None).cast(pl.Float32).alias(f.name))
            else:
                res.append(pl.col(f.name))
        # duplicate names (a spec listed twice) would break select
        seen, uniq = set(), []
        for e, f in zip(res[1:], feats):
            if f.name not in seen:
                seen.add(f.name)
                uniq.append(e)
        return out.select([res[0]] + uniq)
