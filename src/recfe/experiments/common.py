"""Shared experiment plumbing: rows, matrices, train / evaluate / save."""
from __future__ import annotations

import json
import time

import numpy as np
import polars as pl

from recfe import metrics
from recfe.datasets import Dataset, load
from recfe.features.manual import base_features
from recfe.features.spec import Feature
from recfe.features.store import FeatureStore
from recfe.models import gbdt
from recfe.paths import processed, results

SPLITS = ("train", "val", "test")


class Context:
    def __init__(self, name: str):
        self.name = name
        self.ds: Dataset = load(name)
        self.rows = pl.read_parquet(processed(name) / "rows.parquet").sort("row_id")
        self.store = FeatureStore(self.ds, self.rows)
        base, self.base_cols, self.cats = base_features(self.ds, self.rows)
        self.split_rows = {s: self.rows.filter(pl.col("split") == s) for s in SPLITS}
        self.base = {s: base.join(self.split_rows[s].select("row_id"), on="row_id", how="semi").sort("row_id")
                     .select(self.base_cols).to_numpy().astype(np.float32) for s in SPLITS}
        self.y = {s: self.split_rows[s]["label"].to_numpy() for s in SPLITS}

    def matrix(self, feats: list[Feature], split: str, extra: dict | None = None, base: bool = True):
        """Feature matrix filled column by column into one preallocated array (no hstack copies)."""
        extra = extra or {}
        names = (self.base_cols if base else []) + [f.name for f in feats] + list(extra)
        X = np.empty((len(self.y[split]), len(names)), np.float32)
        j = 0
        if base:
            X[:, : len(self.base_cols)] = self.base[split]
            j = len(self.base_cols)
        self.store.compute(feats, split)
        for f in feats:
            X[:, j] = self.store.one(f, split)
            j += 1
        for k, v in extra.items():
            X[:, j] = v[split]
            j += 1
        return X, names


def fit_eval(ctx: Context, feats: list[Feature], tag: str, extra: dict | None = None, base: bool = True,
             params: dict | None = None, save: bool = True, splits=("val", "test")) -> dict:
    t0 = time.perf_counter()
    for s in ("train",) + tuple(splits):
        ctx.store.compute(feats, s)
    feat_sec = time.perf_counter() - t0
    Xtr, names = ctx.matrix(feats, "train", extra, base)
    Xva, _ = ctx.matrix(feats, "val", extra, base)
    model, info = gbdt.train(Xtr, ctx.y["train"], Xva, ctx.y["val"], names, ctx.cats, params)
    del Xtr
    out = {"dataset": ctx.name, "tag": tag, "n_features": len(names), "feature_sec": feat_sec, **info}
    preds = {}
    for s in splits:
        X = Xva if s == "val" else ctx.matrix(feats, s, extra, base)[0]
        t = time.perf_counter()
        p = gbdt.predict(model, X)
        out[f"{s}_predict_sec"] = time.perf_counter() - t
        out[s] = metrics.evaluate(ctx.split_rows[s], p, ctx.name)
        preds[s] = p
    out["importance"] = gbdt.importance(model, names)[:40]
    out["features"] = names
    if save:
        results(f"{ctx.name}__{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
        for s, p in preds.items():
            np.save(processed(ctx.name) / f"pred__{tag}__{s}.npy", p)
    short = {s: {k: round(v, 5) for k, v in out[s].items() if isinstance(v, float)} for s in splits}
    print(f"[{ctx.name}:{tag}] feats={len(names)} iters={info['best_iter']} train={info['train_sec']:.0f}s "
          f"featcomp={feat_sec:.0f}s {short}")
    return out
