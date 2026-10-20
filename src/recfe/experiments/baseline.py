"""Strong baseline: LightGBM on hand-crafted features, plus reference rankers and group ablations.

python -m recfe.experiments.baseline yambda|kion
"""
import json
import sys

import numpy as np

from recfe import metrics
from recfe.experiments.common import Context, fit_eval
from recfe.features.manual import MANUAL
from recfe.features.spec import S
from recfe.paths import results


def group_of(f) -> str:
    key = (f.num if hasattr(f, "num") else f).key
    if key in ("user", "item", "ui"):
        return key
    return key.split(":")[0]  # a / ua / s


def main(name: str):
    ctx = Context(name)
    manual = MANUAL[name]()
    out = {}

    # reference rankers without a learned model
    ref = {}
    if name == "kion":
        for col, sign in [("als_score", 1), ("pop_rank", -1)]:
            i = ctx.base_cols.index(col)
            ref[col] = {s: metrics.evaluate(ctx.split_rows[s], np.nan_to_num(sign * ctx.base[s][:, i], nan=-1e9), name)
                        for s in ("val", "test")}
    else:
        pop = S("item", "listen", "cnt", 7)
        ref["item_pop_7d"] = {s: metrics.evaluate(ctx.split_rows[s], ctx.store.one(pop, s), name) for s in ("val", "test")}
    for k, v in ref.items():
        print("reference", k, {s: {m: round(x, 4) for m, x in v[s].items() if isinstance(x, float)} for s in v})
    out["reference"] = ref

    out["base"] = fit_eval(ctx, [], "base")
    out["manual"] = fit_eval(ctx, manual, "manual")

    # which groups of hand-crafted features matter: drop one group at a time
    groups = sorted({group_of(f) for f in manual})
    abl = {}
    for g in groups:
        feats = [f for f in manual if group_of(f) != g]
        r = fit_eval(ctx, feats, f"manual_minus_{g}", save=False, splits=("val", "test"))
        abl[g] = {"n_dropped": len(manual) - len(feats), "val": r["val"], "test": r["test"]}
    out["ablation"] = abl
    results(f"{name}__baseline_summary.json").write_text(json.dumps(
        {"reference": ref, "ablation": abl,
         "manual": {k: out["manual"][k] for k in ("val", "test", "n_features", "train_sec", "feature_sec", "best_iter")},
         "base": {k: out["base"][k] for k in ("val", "test", "n_features")}}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
