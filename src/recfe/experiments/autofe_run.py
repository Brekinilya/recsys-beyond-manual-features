"""Run the feature search and evaluate the found set on test.

python -m recfe.experiments.autofe_run yambda autofe            # search from static features only
python -m recfe.experiments.autofe_run yambda manual_autofe     # search additions to the hand-crafted set
"""
import sys

from recfe.autofe.search import AutoFE
from recfe.experiments.common import Context, fit_eval
from recfe.features.manual import MANUAL

CONFIG = {
    "autofe": dict(gen0=160, per_gen=60, generations=10, topk=8),
    "manual_autofe": dict(gen0=120, per_gen=60, generations=6, topk=6),
}


def finalize(name: str, tag: str):
    """Evaluate the saved selection of an interrupted search (the search json is written every generation)."""
    import json
    from recfe.autofe.search import load_selected
    from recfe.paths import results
    ctx = Context(name)
    res = fit_eval(ctx, load_selected(name, tag), tag)
    p = results(f"{name}__{tag}_search.json")
    d = json.loads(p.read_text())
    d["timing"]["final_train_sec"] = res["train_sec"]
    d["interrupted_after_gen"] = d["log"][-1]["gen"]
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False))


def main(name: str, tag: str, seed: int = 0):
    ctx = Context(name)
    start = MANUAL[name]() if tag.startswith("manual") else []
    search = AutoFE(ctx, tag=tag, start=start, seed=seed, **CONFIG[tag.split("_s")[0]])
    selected = search.run()
    res = fit_eval(ctx, selected, tag)
    search.timing["final_train_sec"] = res["train_sec"]
    search.save()


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "finalize":
        finalize(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
