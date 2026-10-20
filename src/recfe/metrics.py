"""Classification and ranking metrics.

Ranking metrics are averaged over groups (KION: user at a cutoff, Yambda: user-day).
`npos` is the number of relevant items of the group, including relevant items the
candidates missed (KION), so Recall/NDCG/MAP are end-to-end, not candidate-conditional.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score


def ranking(df: pl.DataFrame, k: int = 10, require_neg: bool = False) -> dict:
    """df: group, label, score, npos."""
    df = df.select("group", "label", "score", "npos")
    if require_neg:
        size = pl.len().over("group")
        df = df.filter(pl.col("npos") < size)
    df = df.filter(pl.col("npos") > 0)
    df = df.sort(["group", "score"], descending=[False, True])
    df = df.with_columns(pl.int_range(1, pl.len() + 1).over("group").alias("rank"))
    df = df.with_columns(pl.col("label").cum_sum().over("group").alias("cum"))
    top = df.filter(pl.col("rank") <= k)
    g = top.group_by("group").agg(
        (pl.col("label") / (pl.col("rank") + 1).log(2)).sum().alias("dcg"),
        pl.col("label").sum().alias("hits"),
        (pl.col("cum") / pl.col("rank") * pl.col("label")).sum().alias("ap"),
        pl.col("npos").first(),
    )
    idcg_tab = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))])
    m = np.minimum(g["npos"].to_numpy(), k)
    ndcg = g["dcg"].to_numpy() / idcg_tab[m]
    recall = g["hits"].to_numpy() / g["npos"].to_numpy()
    ap = g["ap"].to_numpy() / m
    return {f"ndcg@{k}": float(ndcg.mean()), f"recall@{k}": float(recall.mean()), f"map@{k}": float(ap.mean()),
            "groups": int(g.height)}


def gauc(df: pl.DataFrame) -> float:
    """Impression-weighted mean of per-group ROC-AUC over groups with both classes."""
    df = df.with_columns(pl.col("score").rank("average").over("group").alias("r"),
                         pl.len().over("group").alias("n"))
    g = df.group_by("group").agg(
        (pl.col("r") * pl.col("label")).sum().alias("rs"),
        pl.col("label").sum().alias("p"), pl.col("n").first())
    g = g.filter((pl.col("p") > 0) & (pl.col("p") < pl.col("n")))
    p, n, rs = g["p"].to_numpy().astype(float), g["n"].to_numpy().astype(float), g["rs"].to_numpy()
    auc = (rs - p * (p + 1) / 2) / (p * (n - p))
    return float(np.average(auc, weights=n))


def evaluate(rows: pl.DataFrame, score: np.ndarray, dataset: str, k: int = 10) -> dict:
    y = rows["label"].to_numpy()
    p = np.clip(score, 1e-6, 1 - 1e-6)
    out = {"logloss": float(log_loss(y, p)) if score.min() >= 0 and score.max() <= 1 else None,
           "auc": float(roc_auc_score(y, score))}
    df = rows.select("group", "label", "npos").with_columns(pl.Series("score", score))
    if dataset == "yambda":
        out["gauc"] = gauc(df)
        out.update(ranking(df, k, require_neg=True))
    else:
        out.update(ranking(df, k))
    return out
