"""On-disk cache of computed features: one float32 .npy per (split, feature)."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import polars as pl

from recfe.datasets import Dataset
from recfe.features.engine import Engine
from recfe.features.spec import Feature
from recfe.paths import processed


class FeatureStore:
    def __init__(self, ds: Dataset, rows: pl.DataFrame):
        self.ds = ds
        self.rows = rows
        self.engine = Engine(ds)
        self.dir = processed(ds.name) / "fc"
        self.cost_path = self.dir / "cost.json"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cost = json.loads(self.cost_path.read_text()) if self.cost_path.exists() else {}

    def _path(self, split: str, name: str):
        h = hashlib.md5(name.encode()).hexdigest()[:16]
        return self.dir / split / f"{h}.npy"

    def has(self, f: Feature, split: str) -> bool:
        return self._path(split, f.name).exists()

    def compute(self, feats: list[Feature], split: str, verbose=False) -> float:
        missing = list({f.name: f for f in feats if not self.has(f, split)}.values())
        if not missing:
            return 0.0
        r = self.rows.filter(pl.col("split") == split)
        t = time.perf_counter()
        df = self.engine.compute(missing, r, verbose=verbose)
        sec = time.perf_counter() - t
        (self.dir / split).mkdir(exist_ok=True)
        for f in missing:
            np.save(self._path(split, f.name), df[f.name].to_numpy().astype(np.float32))
            self.cost[f"{split}::{f.name}"] = sec / len(missing)
        self.cost_path.write_text(json.dumps(self.cost))
        return sec

    def get(self, feats: list[Feature], split: str, verbose=False) -> np.ndarray:
        self.compute(feats, split, verbose)
        if not feats:
            return np.zeros((int((self.rows["split"] == split).sum()), 0), np.float32)
        return np.column_stack([np.load(self._path(split, f.name)) for f in feats])

    def one(self, f: Feature, split: str) -> np.ndarray:
        self.compute([f], split)
        return np.load(self._path(split, f.name))
