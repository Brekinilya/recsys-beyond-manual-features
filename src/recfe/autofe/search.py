"""Evolutionary search over the feature grammar with FeatureBoost screening.

Loop (one generation):
  1. propose candidates: random grammar samples (exploration) + mutations / ratio compositions
     of the best features seen so far (exploitation);
  2. compute them point-in-time on train and val (FeatureStore caches every result);
  3. screen: for each candidate fit a small booster on that single feature, starting from the
     current model's raw scores (init_score) on train, and measure the val logloss drop
     (FeatureBoost, OpenFE). This estimates the marginal value of the feature given
     everything already selected, at the cost of a 1-feature model;
     The best not-yet-selected candidates of earlier generations are re-screened every generation
     against the current model (their old gains were measured before the last accepted batch);
  4. accept: take the top non-redundant candidates, retrain the full LightGBM with them and keep
     the batch only if val logloss improves; the new model's scores become the next init_score.
Only train and val are touched; test is evaluated once at the end.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np

from recfe.autofe.generate import Generator
from recfe.experiments.common import Context
from recfe.features.spec import Feature, Ratio, Spec
from recfe.models import gbdt
from recfe.paths import results

SCREEN_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=15, min_data_in_leaf=500, max_bin=63,
                     verbose=-1, num_threads=2, feature_pre_filter=False, seed=0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logloss(y, raw):
    p = np.clip(sigmoid(raw), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def to_json(f: Feature):
    if isinstance(f, Ratio):
        return {"num": to_json(f.num), "den": to_json(f.den)}
    return {k: v for k, v in f.__dict__.items() if v is not None}


def from_json(d) -> Feature:
    if "num" in d:
        return Ratio(from_json(d["num"]), from_json(d["den"]))
    return Spec(**d)


class AutoFE:
    def __init__(self, ctx: Context, tag="autofe", start: list[Feature] = (), seed=0, gen0=160, per_gen=60,
                 generations=8, topk=8, screen_rows=700_000, rounds=40, elite=24, explore=0.25,
                 corr_max=0.97, min_gain=2e-5, workers=1, rescreen=40):
        self.ctx, self.tag = ctx, tag
        self.rescreen = rescreen
        self.gen = Generator(ctx.ds.grammar, ctx.name, seed)
        self.rng = np.random.default_rng(seed)
        self.selected = list(start)
        self.n_start = len(self.selected)
        self.gen0, self.per_gen, self.G, self.topk = gen0, per_gen, generations, topk
        self.rounds, self.elite, self.explore = rounds, elite, explore
        self.corr_max, self.min_gain, self.workers = corr_max, min_gain, workers
        n_tr = len(ctx.y["train"])
        self.tr_idx = np.sort(self.rng.choice(n_tr, min(screen_rows, n_tr), replace=False))
        n_va = len(ctx.y["val"])
        self.corr_idx = np.sort(self.rng.choice(n_va, min(60_000, n_va), replace=False))
        self.archive: dict[str, dict] = {}
        self.log: list[dict] = []
        self.timing = {"features": 0.0, "screen": 0.0, "train": 0.0}

    def fit(self, feats):
        ctx = self.ctx
        t = time.perf_counter()
        Xtr, names = ctx.matrix(feats, "train")
        Xva, _ = ctx.matrix(feats, "val")
        model, info = gbdt.train(Xtr, ctx.y["train"], Xva, ctx.y["val"], names, ctx.cats)
        m_tr = model.predict(Xtr[self.tr_idx], raw_score=True, num_iteration=model.best_iteration)
        m_va = model.predict(Xva, raw_score=True, num_iteration=model.best_iteration)
        self.timing["train"] += time.perf_counter() - t
        return model, info, m_tr, m_va

    def screen(self, cands, m_tr, m_va):
        ctx = self.ctx
        y_tr, y_va = ctx.y["train"][self.tr_idx], ctx.y["val"]
        base_ll = logloss(y_va, m_va)

        def one(f):
            x_tr = self.ctx.store.one(f, "train")[self.tr_idx]
            x_va = self.ctx.store.one(f, "val")
            if np.nanstd(x_tr) == 0 or np.all(np.isnan(x_tr)):
                return f.name, 0.0
            d = lgb.Dataset(x_tr.reshape(-1, 1), y_tr, init_score=m_tr, free_raw_data=True)
            params = SCREEN_PARAMS if self.workers > 1 else dict(SCREEN_PARAMS, num_threads=16)
            b = lgb.train(params, d, self.rounds)
            raw = b.predict(x_va.reshape(-1, 1), raw_score=True)
            return f.name, base_ll - logloss(y_va, m_va + raw)

        t = time.perf_counter()
        if self.workers > 1:
            with ThreadPoolExecutor(self.workers) as ex:
                gains = dict(ex.map(one, cands))
        else:
            # several LightGBM trainings in parallel Python threads occasionally crash lib_lightgbm
            # on Windows (access violation in concurrent OpenMP regions); one model at a time is safe
            gains = dict(map(one, cands))
        self.timing["screen"] += time.perf_counter() - t
        return gains

    def propose(self, g: int) -> list[Feature]:
        seen = set(self.archive) | {f.name for f in self.selected}
        out: dict[str, Feature] = {}
        if g == 0:
            target = self.gen0
            make = lambda: self.gen.random()
        else:
            target = self.per_gen
            ranked = sorted((a for a in self.archive.values() if a["gain"] > 0), key=lambda a: -a["gain"])
            parents = [a["feature"] for a in ranked[: self.elite]] + self.selected[self.n_start:]
            if not parents:
                parents = self.selected or [self.gen.random()]
            w = np.array([1.0 / (i + 2) for i in range(len(parents))])
            w /= w.sum()

            def make():
                if self.gen.rng.random() < self.explore:
                    return self.gen.random()
                return self.gen.mutate(parents[self.rng.choice(len(parents), p=w)])
        tries = 0
        while len(out) < target and tries < target * 50:
            tries += 1
            f = make()
            if f.name not in seen and f.name not in out:
                out[f.name] = f
        return list(out.values())

    def _vec(self, f):
        x = self.ctx.store.one(f, "val")[self.corr_idx].astype(np.float64)
        x = np.where(np.isnan(x), np.nanmedian(x) if not np.all(np.isnan(x)) else 0.0, x)
        x = np.sign(x) * np.log1p(np.abs(x))
        s = x.std()
        return (x - x.mean()) / s if s > 0 else None

    def pick(self, cands, gains):
        ranked = sorted(cands, key=lambda f: -gains[f.name])
        chosen, vecs = [], [v for v in (self._vec(f) for f in self.selected) if v is not None]
        for f in ranked:
            if gains[f.name] < self.min_gain or len(chosen) >= self.topk:
                break
            v = self._vec(f)
            if v is None:
                continue
            if vecs and max(abs(float(v @ u)) / len(v) for u in vecs) > self.corr_max:
                continue
            chosen.append(f)
            vecs.append(v)
        return chosen

    def run(self):
        ctx = self.ctx
        t_start = time.perf_counter()
        model, info, m_tr, m_va = self.fit(self.selected)
        cur = info["val_logloss"]
        print(f"[{self.tag}] start: {len(self.selected)} feats, val logloss {cur:.5f}")
        stall = 0
        for g in range(self.G):
            cands = self.propose(g)
            t = time.perf_counter()
            ctx.store.compute(cands, "train")
            ctx.store.compute(cands, "val")
            ft = time.perf_counter() - t
            self.timing["features"] += ft
            # re-screen the best not-yet-selected candidates of earlier generations against the
            # current model: their old gains were measured before the last accepted batch
            sel = {f.name for f in self.selected}
            pool = [a["feature"] for a in sorted(self.archive.values(), key=lambda a: -a["gain"])
                    if a["feature"].name not in sel][: self.rescreen]
            gains = self.screen(cands + pool, m_tr, m_va)
            for f in cands:
                self.archive[f.name] = {"feature": f, "gain": gains[f.name], "gen": g}
            for f in pool:
                self.archive[f.name]["gain"] = gains[f.name]
            chosen = self.pick(cands + pool, gains)
            accepted = []
            while chosen:
                model2, info2, m_tr2, m_va2 = self.fit(self.selected + chosen)
                if info2["val_logloss"] < cur - 1e-6:
                    accepted = chosen
                    self.selected += chosen
                    cur, m_tr, m_va, model = info2["val_logloss"], m_tr2, m_va2, model2
                    break
                chosen = chosen[: len(chosen) // 2]
            stall = 0 if accepted else stall + 1
            best = max(gains.values()) if gains else 0.0
            rec = {"gen": g, "candidates": len(cands), "rescreened": len(pool), "best_gain": best,
                   "accepted": [f.name for f in accepted],
                   "n_selected": len(self.selected), "val_logloss": cur, "feature_sec": ft,
                   "elapsed_sec": time.perf_counter() - t_start}
            self.log.append(rec)
            print(f"[{self.tag}] gen {g}: {len(cands)} cands, best gain {best:.5f}, +{len(accepted)} -> "
                  f"{len(self.selected)} feats, val logloss {cur:.5f}, {rec['elapsed_sec'] / 60:.1f} min")
            for n in rec["accepted"]:
                print("    +", n, f"{gains[n]:.5f}")
            self.save()
            if stall >= 3:
                break
        self.timing["total"] = time.perf_counter() - t_start
        self.save()
        return self.selected

    def save(self):
        out = {
            "tag": self.tag, "dataset": self.ctx.name, "n_start": self.n_start,
            "selected": [to_json(f) for f in self.selected],
            "selected_names": [f.name for f in self.selected],
            "log": self.log, "timing": self.timing, "n_evaluated": len(self.archive),
            "archive": sorted(({"name": k, "gain": v["gain"], "gen": v["gen"], "feature": to_json(v["feature"])}
                               for k, v in self.archive.items()), key=lambda a: -a["gain"]),
        }
        results(f"{self.ctx.name}__{self.tag}_search.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


def load_selected(dataset: str, tag: str) -> list[Feature]:
    d = json.loads(results(f"{dataset}__{tag}_search.json").read_text())
    return [from_json(x) for x in d["selected"]]
