"""KION task: which new items will a user watch during the next 7 days.

Two-stage setup. For a cutoff day T the first stage, trained only on interactions before T,
proposes candidates: iALS top-K (items the user has not watched) plus the most popular items
of the last 7 days. The second stage (ranker) scores every candidate; label = the user watched
it in [T, T + 7). Folds are three consecutive weeks:

    train: T = 2021-08-02   val: T = 2021-08-09   test: T = 2021-08-16 (last week of the log)

Users of a fold = warm users (at least one interaction before T) with at least one target
interaction, sampled with a fixed seed. Cold users can only get popularity and are identical
for all compared rankers, so they are left out. `npos` keeps the full number of target items
(also those missed by the candidates), so Recall/NDCG/MAP are honest end-to-end values.
"""
import time

import implicit
import numpy as np
import polars as pl
import scipy.sparse as sp

from recfe.datasets import DAY, load_kion
from recfe.paths import processed

FOLDS = {"train": 142, "val": 149, "test": 156}
HORIZON = 7
N_USERS = {"train": 25000, "val": 15000, "test": 30000}
K_ALS = 60
K_POP = 60
SEED = 42


def fit_als(hist: pl.DataFrame, n_users: int, n_items: int):
    # confidence grows with how much of the item was watched
    w = (1.0 + hist["pct"].to_numpy() / 100.0 * 4.0).astype(np.float32)
    X = sp.csr_matrix((w, (hist["user_id"].to_numpy(), hist["item_id"].to_numpy())), shape=(n_users, n_items))
    model = implicit.cpu.als.AlternatingLeastSquares(factors=64, regularization=0.1, alpha=10.0,
                                                     iterations=15, random_state=SEED)
    model.fit(X, show_progress=False)
    return model, X


def fold_candidates(ds, T: int, n_users_sample: int, rng) -> tuple[pl.DataFrame, dict]:
    ev = ds.events
    hist = ds.history(T * DAY)
    target = ev.filter((pl.col("ts") >= T * DAY) & (pl.col("ts") < (T + HORIZON) * DAY)).select("user_id", "item_id")
    warm = hist.select("user_id").unique()
    # unique() has no fixed order; without sorting the same seed would sample different users
    tusers = np.sort(target.select("user_id").unique().join(warm, on="user_id", how="semi")["user_id"].to_numpy())
    users = np.sort(rng.choice(tusers, size=min(n_users_sample, len(tusers)), replace=False)).astype(np.int32)

    n_users = int(ev["user_id"].max()) + 1
    n_items = int(ds.items["item_id"].max()) + 1
    t = time.perf_counter()
    model, X = fit_als(hist, n_users, n_items)
    als_sec = time.perf_counter() - t
    ids, scores = model.recommend(users, X[users], N=K_ALS, filter_already_liked_items=True)
    als = pl.DataFrame({
        "user_id": np.repeat(users, K_ALS),
        "item_id": ids.reshape(-1).astype(np.int32),
        "als_rank": np.tile(np.arange(1, K_ALS + 1, dtype=np.int32), len(users)),
    }).filter(pl.col("item_id") >= 0)

    recent = hist.filter(pl.col("ts") >= (T - 7) * DAY)
    pop = (recent.group_by("item_id").agg(pl.len().alias("n"))
                 .sort(["n", "item_id"], descending=[True, False])
                 .head(K_POP + 300).with_row_index("pop_rank", offset=1)
                 .select("item_id", pl.col("pop_rank").cast(pl.Int32)))
    seen = hist.join(pl.DataFrame({"user_id": users}), on="user_id", how="semi").select("user_id", "item_id")
    popc = (pl.DataFrame({"user_id": users}).join(pop, how="cross")
              .join(seen, on=["user_id", "item_id"], how="anti")
              .sort("user_id", "pop_rank")
              .with_columns(pl.col("pop_rank").rank("ordinal").over("user_id").alias("_r"))
              .filter(pl.col("_r") <= K_POP).drop("_r"))

    cand = als.join(popc, on=["user_id", "item_id"], how="full", coalesce=True)
    cand = cand.with_columns(
        pl.col("als_rank").is_not_null().cast(pl.Int8).alias("in_als"),
        pl.col("pop_rank").is_not_null().cast(pl.Int8).alias("in_pop"),
    ).drop("pop_rank").join(pop, on="item_id", how="left")
    uf = model.user_factors[cand["user_id"].to_numpy()]
    itf = model.item_factors[cand["item_id"].to_numpy()]
    cand = cand.with_columns(pl.Series("als_score", (uf * itf).sum(1).astype(np.float32)))

    tgt = target.join(pl.DataFrame({"user_id": users}), on="user_id", how="semi")
    npos = tgt.group_by("user_id").agg(pl.len().cast(pl.Int32).alias("npos"))
    cand = (cand.join(tgt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["user_id", "item_id"], how="left")
                .with_columns(pl.col("label").fill_null(0))
                .join(npos, on="user_id", how="left"))
    info = {
        "users": len(users), "warm_target_users": len(tusers), "rows": cand.height,
        "cand_per_user": cand.height / len(users), "als_fit_sec": als_sec,
        "recall_all": cand["label"].sum() / npos["npos"].sum(),
        "recall_als": cand.filter(pl.col("in_als") == 1)["label"].sum() / npos["npos"].sum(),
        "recall_pop": cand.filter(pl.col("in_pop") == 1)["label"].sum() / npos["npos"].sum(),
    }
    return cand, info


def build_rows() -> pl.DataFrame:
    ds = load_kion()
    rng = np.random.default_rng(SEED)
    parts = []
    for split, T in FOLDS.items():
        cand, info = fold_candidates(ds, T, N_USERS[split], rng)
        print(split, {k: round(v, 4) if isinstance(v, float) else v for k, v in info.items()})
        parts.append(cand.with_columns(
            pl.lit(split).alias("split"),
            pl.lit(T * DAY, pl.Int32).alias("cutoff"),
            (pl.col("user_id").cast(pl.Int64) * 1000 + T).alias("group"),
        ))
    rows = (pl.concat(parts, how="diagonal_relaxed")
              .sort("cutoff", "user_id", "item_id").with_row_index("row_id")
              .with_columns(pl.col("row_id").cast(pl.Int64)))
    rows.write_parquet(processed("kion") / "rows.parquet")
    print(rows.group_by("split").agg(pl.len(), pl.col("label").mean().alias("pos_rate")).sort("split"))
    return rows


if __name__ == "__main__":
    build_rows()
