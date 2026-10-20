"""LightGBM ranker (binary objective, early stopping on the validation fold)."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np

PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, verbose=-1, num_threads=16, seed=0,
)


def train(Xtr, ytr, Xva, yva, names, cats=(), params=None, rounds=3000, es=100,
          init_tr=None, init_va=None):
    p = dict(PARAMS, **(params or {}))
    # grammar names contain characters LightGBM rejects; use positional names inside the model
    cats = [i for i, n in enumerate(names) if n in set(cats)]
    dtr = lgb.Dataset(Xtr, ytr, feature_name=[f"f{i}" for i in range(len(names))],
                      categorical_feature=cats or "auto", init_score=init_tr, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr, init_score=init_va, free_raw_data=True)
    t = time.perf_counter()
    model = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(es, verbose=False)])
    info = {"train_sec": time.perf_counter() - t, "best_iter": model.best_iteration,
            "val_logloss": model.best_score["valid_0"]["binary_logloss"]}
    return model, info


def predict(model, X) -> np.ndarray:
    return model.predict(X, num_iteration=model.best_iteration)


def importance(model, names) -> list[tuple[str, float]]:
    g = model.feature_importance("gain")
    order = np.argsort(-g)
    tot = g.sum() or 1.0
    return [(names[i], float(g[i] / tot)) for i in order]
