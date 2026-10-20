"""How does the value of a feature depend on its window? ("why 7 days and not 10?")

For a few templates, the window runs over a log grid. The value of each version is its FeatureBoost
gain (val logloss drop of a 1-feature booster started from a reference model's scores):
  * alone   - reference = static features only (how informative the aggregate is by itself)
  * in-set  - reference = hand-crafted model without the template's (key, event) family
              (how much the template adds, at this window, to everything else).
python -m recfe.experiments.window_sweep yambda|kion
"""
import json
import sys

import lightgbm as lgb
import numpy as np

from recfe.autofe.search import SCREEN_PARAMS, logloss
from recfe.experiments.common import Context
from recfe.features.manual import MANUAL
from recfe.features.spec import Ratio, S
from recfe.models import gbdt
from recfe.paths import results

TEMPLATES = {
    "yambda": [
        ("ui", "listen", "cnt", None), ("ua:artist", "listen", "cnt", None), ("item", "listen", "cnt", None),
        ("user", "rec", "conv", ("recplus", "rec")), ("ua:artist", "skip", "conv", ("skip", "listen")),
    ],
    "kion": [
        ("item", "watch", "cnt", None), ("ua:genre", "watch", "cnt", "max"), ("s:age", "watch", "cnt", None),
        ("user", "watch", "cnt", None), ("item", "full", "conv", ("full", "watch")),
    ],
}
WINDOWS = {"yambda": [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 150, 300],
           "kion": [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 140]}


def make(t, w):
    key, ev, agg, extra = t
    if agg == "conv":
        return Ratio(S(key, extra[0], "cnt", w), S(key, extra[1], "cnt", w))
    return S(key, ev, "cnt", w, red=extra)


def family(f):
    s = f.num if isinstance(f, Ratio) else f
    return s.key, s.event


def margins(ctx, feats, idx):
    Xtr, names = ctx.matrix(feats, "train")
    Xva, _ = ctx.matrix(feats, "val")
    m, info = gbdt.train(Xtr, ctx.y["train"], Xva, ctx.y["val"], names, ctx.cats)
    return (m.predict(Xtr[idx], raw_score=True, num_iteration=m.best_iteration),
            m.predict(Xva, raw_score=True, num_iteration=m.best_iteration))


def gain(ctx, f, idx, m_tr, m_va):
    x_tr = ctx.store.one(f, "train")[idx]
    x_va = ctx.store.one(f, "val")
    d = lgb.Dataset(x_tr.reshape(-1, 1), ctx.y["train"][idx], init_score=m_tr, free_raw_data=True)
    b = lgb.train(dict(SCREEN_PARAMS, num_threads=16), d, 40)
    return logloss(ctx.y["val"], m_va) - logloss(ctx.y["val"], m_va + b.predict(x_va.reshape(-1, 1), raw_score=True))


def main(name):
    ctx = Context(name)
    rng = np.random.default_rng(0)
    n = len(ctx.y["train"])
    idx = np.sort(rng.choice(n, min(700_000, n), replace=False))
    manual = MANUAL[name]()
    base_m = margins(ctx, [], idx)
    out = {}
    for t in TEMPLATES[name]:
        feats = [make(t, w) for w in WINDOWS[name]] + [make(t, None)]
        for s in ("train", "val"):
            ctx.store.compute(feats, s)
        fam = (t[0], t[1] if t[2] != "conv" else t[3][0])
        rest = [f for f in manual if family(f) != fam and not (isinstance(f, Ratio) and family(f)[0] == t[0]
                                                               and t[2] == "conv")]
        ref_m = margins(ctx, rest, idx)
        alone = [gain(ctx, f, idx, *base_m) for f in feats]
        inset = [gain(ctx, f, idx, *ref_m) for f in feats]
        tname = feats[0].name.replace("w1", "w*")
        out[tname] = {"windows": WINDOWS[name] + ["all"], "alone": alone, "in_set": inset,
                      "manual_dropped": len(manual) - len(rest)}
        best = int(np.argmax(inset))
        print(f"{tname}: best window {out[tname]['windows'][best]} (in-set gain {inset[best]:.5f}); "
              f"alone best {out[tname]['windows'][int(np.argmax(alone))]}")
    results(f"{name}__window_sweep.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
