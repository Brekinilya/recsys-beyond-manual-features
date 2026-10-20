"""Combining graph models with boosting.

1. GBDT(+ link-prediction scores): the self-supervised LightGCN (trained on edges before the first
   train cutoff, never on labels) gives user.item and user.artist dot products for every row;
   they are added as features to the hand-crafted / AutoFE sets.
2. Blend: logit-average of GBDT and the supervised GNN, weight chosen on val.
python -m recfe.experiments.hybrid yambda|kion
"""
import json
import sys

import numpy as np

from recfe import metrics
from recfe.autofe.search import load_selected
from recfe.experiments.common import Context, fit_eval
from recfe.features.manual import MANUAL
from recfe.paths import processed, results


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(name):
    ctx = Context(name)
    P = processed(name)
    out = {}
    lp = {s: np.load(P / f"lp__lgcn_bpr__{s}.npy") for s in ("train", "val", "test")}
    names = ["lgcn_ui"] + (["lgcn_artist", "lgcn_album"] if name == "yambda" else ["lgcn_ctype"])
    extra = {n: {s: lp[s][:, j] for s in lp} for j, n in enumerate(names)}
    sets = {"manual": MANUAL[name]()}
    try:
        sets["autofe"] = load_selected(name, "autofe")
    except FileNotFoundError:
        pass
    for k, feats in sets.items():
        if not (P / f"pred__{k}__val.npy").exists():
            fit_eval(ctx, feats, k)
        r = fit_eval(ctx, feats, f"{k}+lgcn", extra=extra)
        out[f"{k}+lgcn"] = {"val": r["val"], "test": r["test"]}
    r = fit_eval(ctx, [], "base+lgcn", extra=extra)
    out["base+lgcn"] = {"val": r["val"], "test": r["test"]}

    # blend GBDT and the supervised GNN on the logit scale, weight picked on val logloss
    for g in ("gnn",):
        for k in sets:
            try:
                a = {s: np.load(P / f"pred__{k}__{s}.npy") for s in ("val", "test")}
                b = {s: np.load(P / f"pred__{g}__{s}.npy") for s in ("val", "test")}
            except FileNotFoundError:
                continue
            y = ctx.y["val"]
            best = None
            for w in np.linspace(0, 1, 21):
                z = w * logit(a["val"]) + (1 - w) * logit(b["val"])
                p = 1 / (1 + np.exp(-z))
                ll = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
                if best is None or ll < best[1]:
                    best = (w, ll)
            w = best[0]
            res = {"w_gbdt": float(w)}
            for s in ("val", "test"):
                z = w * logit(a[s]) + (1 - w) * logit(b[s])
                res[s] = metrics.evaluate(ctx.split_rows[s], 1 / (1 + np.exp(-z)), name)
            out[f"blend_{k}_{g}"] = res
            print(f"[{name}] blend {k}+{g}: w_gbdt={w:.2f} test {res['test']}")
    results(f"{name}__hybrid_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
