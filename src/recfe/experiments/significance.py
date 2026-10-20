"""Paired bootstrap over groups for differences of per-group test metrics against the baseline.

Groups are resampled with replacement (Yambda: user-day, KION: user); both models are scored on the
same resample, so the interval is for the difference itself.
python -m recfe.experiments.significance yambda|kion
"""
import json
import sys

import numpy as np
import polars as pl

from recfe.experiments.hybrid import logit
from recfe.paths import processed, results

TAGS = ["grid", "autofe", "manual_autofe", "gnn", "blend_manual_gnn", "blend_autofe_gnn"]


def per_group(rows: pl.DataFrame, score: np.ndarray, dataset: str, k=10) -> pl.DataFrame:
    df = rows.select("group", "label", "npos").with_columns(pl.Series("score", score))
    df = df.with_columns(pl.len().over("group").alias("n"))
    if dataset == "yambda":
        df = df.filter(pl.col("npos") < pl.col("n"))
    df = df.filter(pl.col("npos") > 0).sort(["group", "score"], descending=[False, True])
    df = df.with_columns(pl.int_range(1, pl.len() + 1).over("group").alias("rank"),
                         pl.col("label").cum_sum().over("group").alias("cum"),
                         pl.col("score").rank("average").over("group").alias("r"))
    top = pl.col("rank") <= k
    g = df.group_by("group").agg(
        (pl.col("label") / (pl.col("rank") + 1).log(2)).filter(top).sum().alias("dcg"),
        (pl.col("cum") / pl.col("rank") * pl.col("label")).filter(top).sum().alias("ap"),
        (pl.col("r") * pl.col("label")).sum().alias("rs"),
        pl.col("npos").first(), pl.col("n").first(),
    ).sort("group")
    idcg = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))])
    m = np.minimum(g["npos"].to_numpy(), k)
    p, n = g["npos"].to_numpy().astype(float), g["n"].to_numpy().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (g["rs"].to_numpy() - p * (p + 1) / 2) / (p * (n - p))
    return pl.DataFrame({"group": g["group"], "ndcg": g["dcg"].to_numpy() / idcg[m], "ap": g["ap"].to_numpy() / m,
                         "auc": auc, "w": n})


def group_logloss(rows: pl.DataFrame, score: np.ndarray) -> pl.DataFrame:
    """Per-group sum of the log loss over all rows (all groups), for a bootstrap of the global LogLoss."""
    p = np.clip(score, 1e-6, 1 - 1e-6)
    y = rows["label"].to_numpy()
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return (rows.select("group").with_columns(pl.Series("ll", ll))
                .group_by("group").agg(pl.col("ll").sum(), pl.len().alias("n")).sort("group"))


def main(name: str, B: int = 1000):
    P = processed(name)
    rows = pl.read_parquet(P / "rows.parquet").filter(pl.col("split") == "test").sort("row_id")
    hyb = json.loads(results(f"{name}__hybrid_summary.json").read_text())

    def pred(tag):
        if tag.startswith("blend_"):
            _, k, g = tag.split("_")
            w = hyb[tag]["w_gbdt"]
            z = w * logit(np.load(P / f"pred__{k}__test.npy")) + (1 - w) * logit(np.load(P / f"pred__{g}__test.npy"))
            return 1 / (1 + np.exp(-z))
        return np.load(P / f"pred__{tag}__test.npy")

    base_pred = pred("manual")
    base = per_group(rows, base_pred, name)
    base_ll = group_logloss(rows, base_pred)
    metrics = ["auc", "ndcg"] if name == "yambda" else ["ap", "ndcg"]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, base.height, size=(B, base.height))
    idx_ll = rng.integers(0, base_ll.height, size=(B, base_ll.height))
    out = {}
    for tag in TAGS:
        try:
            p_other = pred(tag)
        except FileNotFoundError:
            continue
        other = per_group(rows, p_other, name)
        assert other["group"].equals(base["group"])
        res = {}
        oll = group_logloss(rows, p_other)
        n = base_ll["n"].to_numpy()
        d = oll["ll"].to_numpy() - base_ll["ll"].to_numpy()
        diff = d[idx_ll].sum(1) / n[idx_ll].sum(1)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        res["logloss"] = {"diff": float(d.sum() / n.sum()), "ci95": [float(lo), float(hi)],
                          "p_ge_0": float((diff >= 0).mean())}
        for m in metrics:
            a, b = base[m].to_numpy(), other[m].to_numpy()
            if m == "auc":  # GAUC: impression-weighted mean over groups with both classes
                ok = ~np.isnan(a)
                w = base["w"].to_numpy() * ok
                a, b = np.nan_to_num(a), np.nan_to_num(b)
                diff = ((b - a) * w)[idx].sum(1) / w[idx].sum(1)
                point = float(((b - a) * w).sum() / w.sum())
            else:
                diff = (b - a)[idx].mean(1)
                point = float((b - a).mean())
            lo, hi = np.percentile(diff, [2.5, 97.5])
            res[{"auc": "gauc", "ndcg": "ndcg@10", "ap": "map@10"}[m]] = {
                "diff": point, "ci95": [float(lo), float(hi)], "p_le_0": float((diff <= 0).mean())}
        out[tag] = res
        print(tag, {k: f"{v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in res.items()})
    results(f"{name}__significance.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
