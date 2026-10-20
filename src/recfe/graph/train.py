"""Train / evaluate the graph ranker on exactly the same rows, splits and cutoffs as LightGBM.

python -m recfe.graph.train yambda gnn             # supervised ranker (BCE on task labels + BPR aux)
python -m recfe.graph.train yambda gnn_emb         # same, embeddings only (no degree / pair / base inputs)
python -m recfe.graph.train yambda lgcn_bpr        # self-supervised link prediction only -> scores as GBDT features

The graph for a row is always built from events before that row's cutoff. Model parameters are
fitted on train rows (graphs at train cutoffs); at val/test cutoffs the same parameters are
applied by propagating over the newer graph.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from recfe import metrics
from recfe.datasets import DAY
from recfe.experiments.common import Context
from recfe.features.manual import base_features
from recfe.graph.build import GRAPH_CFG, build, layout
from recfe.graph.model import GPUGraph, GraphRanker, to_csr
from recfe.paths import processed, results

VARIANTS = {
    "gnn": dict(use_feats=True, bce_w=1.0, bpr_w=0.1),
    "gnn_emb": dict(use_feats=False, bce_w=1.0, bpr_w=0.1),
    "lgcn_bpr": dict(use_feats=False, bce_w=0.0, bpr_w=1.0),
}
BATCH = {"yambda": 8192, "kion": 32768}


class GraphExperiment:
    def __init__(self, name: str, tag: str, d=64, layers=2, lr=1e-3, epochs=4, patience=4, emb_l2=1e-4, seed=0):
        self.emb_l2 = emb_l2
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.dev = torch.device("cuda")
        self.ctx = Context(name)
        self.name, self.tag = name, tag
        self.cfg = VARIANTS[tag]
        self.lay = layout(self.ctx.ds)
        self.dir = processed(name) / "graphs"
        self.dir.mkdir(exist_ok=True)
        self.n_ui_rel = len(GRAPH_CFG[name]["types"]) * len(GRAPH_CFG[name]["scales"])
        self.attr_slots = ["artist", "album"] if name == "yambda" else ["ctype"]
        self.epochs, self.patience, self.lr = epochs, patience, lr
        self._prep_rows()
        self.model = GraphRanker(self.lay.n_nodes, len(self.lay.rel_names), self.n_ui_rel, len(self.attr_slots),
                                 self.base.shape[1], d=d, layers=layers, use_feats=self.cfg["use_feats"]).to(self.dev)
        self.timing = {"graph_build": 0.0, "train": 0.0, "eval": 0.0}

    def _prep_rows(self):
        ctx, ds = self.ctx, self.ctx.ds
        rows = ctx.rows
        base, cols, cats = base_features(ds, rows)
        num = [c for c in cols if c not in cats]
        X = base.select(num).to_numpy().astype(np.float64)
        X = np.sign(X) * np.log1p(np.abs(X))
        tr = (rows["split"] == "train").to_numpy()
        med = np.nanmedian(X[tr], 0)
        X = np.where(np.isnan(X), med, X)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        self.base = ((X - mu) / sd).astype(np.float32)          # aligned with rows (sorted by row_id)
        u = rows["user_id"].to_numpy().astype(np.int64)
        i = self.lay.n_users + rows["item_id"].to_numpy().astype(np.int64)
        slots = []
        for a in self.attr_slots:
            m = ds.attrs[a].mapping
            v = rows.select("item_id").join(m, on="item_id", how="left")[a].fill_null(-1).to_numpy().astype(np.int64)
            slots.append(np.where(v >= 0, self.lay.attr_off[a][0] + v, -1))
        self.u, self.i, self.slots = u, i, slots
        self.y = rows["label"].to_numpy().astype(np.float32)
        self.cutoff = rows["cutoff"].to_numpy()
        self.split = rows["split"].to_numpy()

    def cutoffs(self, split):
        return sorted(np.unique(self.cutoff[self.split == split]).tolist())

    def graph(self, c: int) -> tuple[GPUGraph, np.ndarray, np.ndarray]:
        p = self.dir / f"c{c // DAY}.npz"
        if not p.exists():
            t = time.perf_counter()
            rows_c = self.ctx.rows.filter(pl.col("cutoff") == c)
            g = build(self.ctx.ds, self.lay, c, rows_c)
            g.update(to_csr(g, self.lay.n_nodes))
            np.savez(p, **g)
            self.timing["graph_build"] += time.perf_counter() - t
        g = dict(np.load(p))
        return GPUGraph(g, self.lay.n_nodes, self.n_ui_rel, self.dev), g["row_id"], g["pair"]

    def batch(self, idx: np.ndarray, pair: torch.Tensor, pos: np.ndarray):
        t = lambda a, dt=torch.int64: torch.from_numpy(a[idx]).to(self.dev, dt)
        return {"u": t(self.u), "i": t(self.i), "attrs": [t(s) for s in self.slots],
                "pair": pair[torch.from_numpy(pos).to(self.dev)], "base": t(self.base, torch.float32)}

    def run(self):
        m = self.model
        other = [p for n, p in m.named_parameters() if n not in ("emb.weight", "mix_logits")]
        # the dozen relation-mixing logits get a larger step: with the shared lr they barely leave uniform
        opt = torch.optim.Adam([{"params": [m.emb.weight], "lr": self.lr}, {"params": other, "lr": self.lr},
                                {"params": [m.mix_logits], "lr": 0.03}])
        bs = BATCH[self.name]
        bce_w, bpr_w = self.cfg["bce_w"], self.cfg["bpr_w"]
        lo, hi = self.lay.n_users, self.lay.n_users + self.lay.n_items
        train_c = self.cutoffs("train")
        if bce_w == 0:          # link prediction only: learn on the graph of the first train cutoff
            train_c = train_c[:1]
        best, best_state, bad, hist = np.inf, None, 0, []
        for ep in range(self.epochs):
            t = time.perf_counter()
            # an epoch is split into units (Yambda: one train cutoff, KION: a quarter of the single
            # cutoff's rows); val is checked after every unit because the ID embeddings overfit fast
            units = []
            for c in self.rng.permutation(train_c):
                n_c = int((self.cutoff == c).sum()) if bce_w > 0 else 0
                perm = self.rng.permutation(max(n_c, 1))
                for part in np.array_split(perm, 1 if self.name == "yambda" or bce_w == 0 else 4):
                    units.append((int(c), part))
            stop = False
            for u_i, (c, part) in enumerate(units):
                m.train()
                losses = []
                g, row_id, pair = self.graph(c)
                pair_t = torch.from_numpy(pair).to(self.dev, torch.float32)
                rows_idx = np.searchsorted(self.ctx.rows["row_id"].to_numpy(), row_id)
                steps = int(np.ceil(len(part) / bs)) if bce_w > 0 else 150
                for s in range(steps):
                    E = m.propagate(g)
                    loss = bpr_w * m.bpr(E, g, 8192, lo, hi) if bpr_w > 0 else 0.0
                    if bce_w > 0:
                        pos = part[s * bs:(s + 1) * bs]
                        b = self.batch(rows_idx[pos], pair_t, pos)
                        y = torch.from_numpy(self.y[rows_idx[pos]]).to(self.dev)
                        loss = loss + bce_w * F.binary_cross_entropy_with_logits(m.score(E, g, b), y)
                        e0 = m.emb.weight
                        loss = loss + self.emb_l2 * ((e0[b["u"]] ** 2).sum(1).mean() + (e0[b["i"]] ** 2).sum(1).mean())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.detach()))
                del g, pair_t, E
                torch.cuda.empty_cache()
                self.timing["train"] += time.perf_counter() - t
                if bce_w == 0:
                    continue
                t = time.perf_counter()
                val = self.predict("val")
                ll = metrics.evaluate(self.ctx.split_rows["val"], val, self.name)
                hist.append({"epoch": ep, "unit": u_i, "loss": float(np.mean(losses)), **ll})
                print(f"[{self.name}:{self.tag}] epoch {ep} unit {u_i} loss {np.mean(losses):.4f} "
                      f"val logloss {ll['logloss']:.5f} auc {ll['auc']:.4f} ndcg {ll['ndcg@10']:.4f} "
                      f"mix {np.round(m.mix().detach().cpu().numpy(), 3).tolist()}", flush=True)
                if ll["logloss"] < best - 1e-5:
                    best, bad = ll["logloss"], 0
                    best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        stop = True
                        break
                t = time.perf_counter()
            if bce_w == 0:
                hist.append({"epoch": ep, "loss": float(np.mean(losses))})
                print(f"[{self.name}:{self.tag}] epoch {ep} bpr {np.mean(losses):.4f}", flush=True)
            if stop:
                break
        if best_state is not None:
            m.load_state_dict(best_state)
        self.history = hist

    @torch.no_grad()
    def predict(self, split: str, what: str = "score") -> np.ndarray:
        m = self.model
        m.eval()
        t = time.perf_counter()
        all_ids = self.ctx.rows["row_id"].to_numpy()
        split_ids = self.ctx.split_rows[split]["row_id"].to_numpy()
        out = np.zeros(len(split_ids) if what == "score" else (len(split_ids), 1 + len(self.slots)), np.float32)
        for c in self.cutoffs(split):
            g, row_id, pair = self.graph(int(c))
            pair_t = torch.from_numpy(pair).to(self.dev, torch.float32)
            E = m.propagate(g)
            rows_idx = np.searchsorted(all_ids, row_id)
            dst = np.searchsorted(split_ids, row_id)
            for s in range(0, len(rows_idx), 65536):
                pos = np.arange(s, min(s + 65536, len(rows_idx)))
                b = self.batch(rows_idx[pos], pair_t, pos)
                if what == "score":
                    out[dst[pos]] = torch.sigmoid(m.score(E, g, b)).cpu().numpy()
                else:  # link-prediction scores used as GBDT features
                    eu = E[b["u"]]
                    cols = [(eu * E[b["i"]]).sum(1)]
                    for a in b["attrs"]:
                        cols.append(((eu * E[a.clamp(min=0)]).sum(1)) * (a >= 0))
                    out[dst[pos]] = torch.stack(cols, 1).cpu().numpy()
            del g, pair_t, E
            torch.cuda.empty_cache()
        self.timing["eval"] += time.perf_counter() - t
        return out


def main(name: str, tag: str):
    ex = GraphExperiment(name, tag)
    t = time.perf_counter()
    ex.run()
    res = {"dataset": name, "tag": tag, "history": ex.history,
           "mix": dict(zip(ex.lay.rel_names, ex.model.mix().detach().cpu().numpy().round(4).tolist())),
           "n_params": int(sum(p.numel() for p in ex.model.parameters()))}
    if ex.cfg["bce_w"] > 0:
        for s in ("val", "test"):
            p = ex.predict(s)
            res[s] = metrics.evaluate(ex.ctx.split_rows[s], p, name)
            np.save(processed(name) / f"pred__{tag}__{s}.npy", p)
        print(f"[{name}:{tag}] val {res['val']}\n  test {res['test']}")
    else:
        for s in ("train", "val", "test"):
            np.save(processed(name) / f"lp__{tag}__{s}.npy", ex.predict(s, what="lp"))
    res["timing"] = ex.timing
    res["total_sec"] = time.perf_counter() - t
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    results(f"{name}__{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
