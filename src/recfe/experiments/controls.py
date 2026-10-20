"""Controls for the search: brute-force grid and random features of the same size.

python -m recfe.experiments.controls yambda grid
python -m recfe.experiments.controls yambda random   (needs the autofe search json for K and the gen-0 pool)
"""
import json
import sys

import numpy as np

from recfe.autofe.generate import VALUE_EVENTS
from recfe.autofe.search import from_json
from recfe.experiments.common import Context, fit_eval
from recfe.features.spec import S
from recfe.paths import results

GRID_WINDOWS = (1, 7, 30, None)


def grid(ctx: Context):
    g = ctx.ds.grammar
    value_events = VALUE_EVENTS.get(ctx.name, set(g.events))
    feats = []
    # KION: only the two main events, so that 2.7M train rows x all grid columns fit into 16 GB RAM
    events = g.events if ctx.name == "yambda" else ["watch", "full"]
    for key in g.keys:
        attr = key.split(":")[1] if ":" in key else None
        red = "max" if attr in g.multi_attrs else None
        for ev in events:
            feats += [S(key, ev, "cnt", w, red=red) for w in GRID_WINDOWS]
            feats.append(S(key, ev, "last", red="min" if red else None))
            if ev in value_events and ctx.name == "yambda":
                feats += [S(key, ev, "mean", w, g.values[0], red=red) for w in (30, None)]
    if ctx.name == "kion":  # value means only for the main event, to keep memory in check
        for key in g.keys:
            attr = key.split(":")[1] if ":" in key else None
            red = "mean" if attr in g.multi_attrs else None
            feats += [S(key, "watch", "mean", w, "pct", red=red) for w in (30, None)]
    return feats


def main(name: str, what: str):
    ctx = Context(name)
    if what == "grid":
        feats = grid(ctx)
        print("grid features:", len(feats))
        fit_eval(ctx, feats, "grid")
    elif what == "random":
        d = json.loads(results(f"{name}__autofe_search.json").read_text())
        k = len(d["selected"])
        pool = [from_json(a["feature"]) for a in d["archive"] if a["gen"] == 0]
        res = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            feats = [pool[i] for i in rng.choice(len(pool), size=min(k, len(pool)), replace=False)]
            r = fit_eval(ctx, feats, f"random{k}_s{seed}", save=seed == 0)
            res.append({"val": r["val"], "test": r["test"]})
        results(f"{name}__random_summary.json").write_text(json.dumps({"k": k, "runs": res}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
