"""Relation-mixing LightGCN encoder + MLP ranking head."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_csr(g: dict, n_nodes: int) -> dict:
    """Directed CSR layout of the symmetric adjacency; `und` maps each nnz to its undirected edge."""
    src = np.concatenate([g["ui_src"], g["st_src"]]).astype(np.int64)
    dst = np.concatenate([g["ui_dst"], g["st_dst"]]).astype(np.int64)
    E = len(src)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    und = np.concatenate([np.arange(E), np.arange(E)])
    order = np.argsort(row * n_nodes + col, kind="stable")
    row, col, und = row[order], col[order], und[order]
    crow = np.zeros(n_nodes + 1, np.int64)
    crow[1:] = np.cumsum(np.bincount(row, minlength=n_nodes))
    return {"crow": crow.astype(np.int32), "col": col.astype(np.int32), "und": und.astype(np.int32)}


class GPUGraph:
    def __init__(self, g: dict, n_nodes: int, n_ui_rel: int, device):
        t = lambda a, dt=None: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
        self.n = n_nodes
        self.crow, self.col, self.und = t(g["crow"]), t(g["col"]), t(g["und"], torch.int64)
        counts = self.crow[1:] - self.crow[:-1]
        self.row = torch.repeat_interleave(torch.arange(n_nodes, device=device, dtype=torch.int32), counts)
        self.ui_w = t(g["ui_w"])                       # fp16 (E_ui, R_ui)
        self.st_rel = t(g["st_rel"], torch.int64)
        self.st_w = t(g["st_w"])
        self.ui_src = t(g["ui_src"], torch.int64)
        self.ui_dst = t(g["ui_dst"], torch.int64)
        self.node_u = t(g["node_u"], torch.float32)
        self.node_i = t(g["node_i"], torch.float32)
        self.n_ui_rel = n_ui_rel


class MixSpMM(torch.autograd.Function):
    """Y = A(vals) @ X for a symmetric CSR adjacency.

    d/dX is exact (A^T = A). d/dvals is estimated on a random subset of nonzeros
    (unbiased, scaled by nnz / sample): the exact value gradient needs one gather per nonzero
    and costs ~20x a spmm, while only a dozen relation-mixing weights depend on it.
    """

    @staticmethod
    def forward(ctx, vals, X, graph, sample):
        A = torch.sparse_csr_tensor(graph.crow, graph.col, vals, (graph.n, graph.n))
        ctx.save_for_backward(vals, X)
        ctx.graph, ctx.sample = graph, sample
        return A @ X

    @staticmethod
    def backward(ctx, G):
        vals, X = ctx.saved_tensors
        g = ctx.graph
        A = torch.sparse_csr_tensor(g.crow, g.col, vals, (g.n, g.n))
        gX = A @ G.contiguous()
        gv = None
        if ctx.needs_input_grad[0]:
            nnz = vals.numel()
            S = min(ctx.sample, nnz)
            idx = torch.randint(0, nnz, (S,), device=vals.device)
            est = (G[g.row[idx].long()] * X[g.col[idx].long()]).sum(1) * (nnz / S)
            gv = torch.zeros_like(vals).index_add_(0, idx, est)
        return gv, gX, None, None


class GraphRanker(nn.Module):
    def __init__(self, n_nodes, n_rel, n_ui_rel, n_attr_slots, n_base, d=64, layers=2, hidden=256,
                 use_feats=True, sample=2_000_000):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, d)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.mix_logits = nn.Parameter(torch.zeros(n_rel))
        self.layers, self.n_ui_rel, self.sample, self.use_feats = layers, n_ui_rel, sample, use_feats
        n_in = d * (3 + n_attr_slots)
        if use_feats:
            n_in += 3 * n_ui_rel + n_base
        self.drop = nn.Dropout(0.2)
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def mix(self):
        return torch.softmax(self.mix_logits, 0)

    def propagate(self, g: GPUGraph) -> torch.Tensor:
        m = self.mix()
        v_ui = (g.ui_w @ m[: self.n_ui_rel].to(g.ui_w.dtype)).float()
        v_st = g.st_w * m[g.st_rel]
        vals = torch.cat([v_ui, v_st])[g.und]
        X = self.emb.weight
        acc = X
        for _ in range(self.layers):
            X = MixSpMM.apply(vals, X, g, self.sample)
            acc = acc + X
        return acc / (self.layers + 1)

    def score(self, E, g: GPUGraph, b: dict) -> torch.Tensor:
        eu, ei = self.drop(E[b["u"]]), self.drop(E[b["i"]])
        z = [eu, ei, eu * ei]
        for a in b["attrs"]:
            ea = self.drop(E[a.clamp(min=0)]) * (a >= 0).unsqueeze(1)
            z.append(eu * ea)
        if self.use_feats:
            z += [g.node_u[b["u"]], g.node_i[b["i"] - g.node_u.shape[0]], b["pair"], b["base"]]
        return self.mlp(torch.cat(z, 1)).squeeze(1)

    def bpr(self, E, g: GPUGraph, n: int, item_lo: int, item_hi: int) -> torch.Tensor:
        k = torch.randint(0, g.ui_src.numel(), (n,), device=E.device)
        u, i = g.ui_src[k], g.ui_dst[k]
        j = torch.randint(item_lo, item_hi, (n,), device=E.device)
        eu = E[u]
        return F.softplus(-((eu * E[i]).sum(1) - (eu * E[j]).sum(1))).mean()
