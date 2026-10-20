"""Training-seed variance of the main rankers: same rows, same features, different random seeds.

LightGBM sets are retrained with seeds 0, 1, 2 (row and column subsampling); the graph ranker with
seeds 1 and 2 (seed 0 is the main run). Differences to the hand-crafted model are taken per seed.
python -m recfe.experiments.seeds yambda|kion
"""
import gc
import json
import sys

import torch

from recfe.autofe.search import load_selected
from recfe.experiments.common import Context, fit_eval
from recfe.experiments.controls import grid
from recfe.features.manual import MANUAL
from recfe.graph import train as graph_train
from recfe.paths import results

SEEDS = (0, 1, 2)


def main(name: str):
    ctx = Context(name)
    sets = {"manual": MANUAL[name](), "autofe": load_selected(name, "autofe"),
            "manual_autofe": load_selected(name, "manual_autofe"), "grid": grid(ctx)}
    out = {"gbdt": {}, "gnn": []}
    for tag, feats in sets.items():
        out["gbdt"][tag] = []
        for s in SEEDS:
            r = fit_eval(ctx, feats, f"{tag}_seed{s}", params={"seed": s}, save=False)
            out["gbdt"][tag].append({"seed": s, "val": r["val"], "test": r["test"]})
        results(f"{name}__seeds.json").write_text(json.dumps(out, indent=1))
    del ctx
    gc.collect()

    main_gnn = json.loads(results(f"{name}__gnn.json").read_text())
    out["gnn"].append({"seed": 0, "val": main_gnn["val"], "test": main_gnn["test"]})
    for s in SEEDS[1:]:
        graph_train.main(name, "gnn", seed=s)
        r = json.loads(results(f"{name}__gnn_seed{s}.json").read_text())
        out["gnn"].append({"seed": s, "val": r["val"], "test": r["test"]})
        torch.cuda.empty_cache()
        results(f"{name}__seeds.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
