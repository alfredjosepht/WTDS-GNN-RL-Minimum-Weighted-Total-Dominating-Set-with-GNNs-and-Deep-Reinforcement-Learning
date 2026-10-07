"""GNN encoder + Q head (Section 6), pure PyTorch with sparse edge_index.

Several graphs are batched as one disjoint union (node offsets + `batch` vector),
as in PyG. Per-graph max / argmax use scatter_reduce(..., "amax").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .env import NUM_FEATURES, NUM_GLOBAL, F

NEG_INF = float("-inf")


# ------------------------------------------------------------------ batching
@dataclass
class Batch:
    x: torch.Tensor           # (N, 10)
    edge_index: torch.Tensor  # (2, E) src -> dst
    batch: torch.Tensor       # (N,) graph id of each node
    glob: torch.Tensor        # (B, 3)
    deg: torch.Tensor         # (N,) float, >= 1
    valid: torch.Tensor       # (N,) bool
    ptr: torch.Tensor         # (B+1,) node offsets
    num_graphs: int

    def to(self, device):
        return Batch(self.x.to(device, non_blocking=True), self.edge_index.to(device, non_blocking=True),
                     self.batch.to(device, non_blocking=True), self.glob.to(device, non_blocking=True),
                     self.deg.to(device, non_blocking=True), self.valid.to(device, non_blocking=True),
                     self.ptr.to(device, non_blocking=True), self.num_graphs)


def collate(items, device=None) -> Batch:
    """items: sequence of (graph, X (n,10) np.float32, glob (3,) np.float32)."""
    xs, eis, bs, gl, degs = [], [], [], [], []
    off = 0
    ptr = [0]
    for i, (g, X, glob) in enumerate(items):
        xs.append(X)
        eis.append(g.edge_index + off)
        bs.append(np.full(g.n, i, dtype=np.int64))
        gl.append(glob)
        degs.append(g.degrees)
        off += g.n
        ptr.append(off)
    x = torch.from_numpy(np.concatenate(xs).astype(np.float32, copy=False))
    b = Batch(x=x,
              edge_index=torch.from_numpy(np.concatenate(eis, axis=1)),
              batch=torch.from_numpy(np.concatenate(bs)),
              glob=torch.from_numpy(np.stack(gl).astype(np.float32)),
              deg=torch.from_numpy(np.concatenate(degs).astype(np.float32)).clamp_(min=1.0),
              valid=x[:, F["valid_action"]] > 0.5,
              ptr=torch.tensor(ptr, dtype=torch.long),
              num_graphs=len(items))
    return b.to(device) if device is not None else b


def scatter_sum(src, index, n):
    out = torch.zeros((n,) + src.shape[1:], dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


def scatter_max(src, index, n, fill=NEG_INF):
    shape = (n,) + src.shape[1:]
    out = torch.full(shape, fill, dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)


def graph_argmax(q: torch.Tensor, batch: torch.Tensor, num_graphs: int):
    """Per-graph (max value, argmax node index). Ties -> lowest node index.
    Graphs whose nodes are all -inf return (-inf, some node of that graph)."""
    qmax = scatter_max(q, batch, num_graphs)
    is_max = q == qmax[batch]
    node = torch.arange(q.numel(), device=q.device)
    big = torch.full_like(node, q.numel())
    cand = torch.where(is_max, node, big)
    arg = torch.full((num_graphs,), q.numel(), dtype=torch.long, device=q.device)
    arg = arg.scatter_reduce_(0, batch, cand, reduce="amin", include_self=True)
    return qmax, arg


# ------------------------------------------------------------------ encoders
class MPNNLayer(nn.Module):
    """m_v = Σ_u ReLU(W_msg[h_u ‖ h_v]);  a_v = [m_v ‖ m_v/deg v];
    h_v <- LN(h_v + ReLU(W2 ReLU(W1 [h_v ‖ a_v])))."""

    def __init__(self, d):
        super().__init__()
        self.msg_src = nn.Linear(d, d)               # W_msg split: W_a h_u + W_b h_v
        self.msg_dst = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(3 * d, d)
        self.w2 = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, edge_index, deg):
        src, dst = edge_index
        msg = Fn.relu(self.msg_src(h)[src] + self.msg_dst(h)[dst])
        m = scatter_sum(msg, dst, h.shape[0])
        a = torch.cat([m, m / deg.unsqueeze(1)], dim=1)
        return self.norm(h + Fn.relu(self.w2(Fn.relu(self.w1(torch.cat([h, a], 1))))))


class GINLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, edge_index, deg):
        src, dst = edge_index
        agg = scatter_sum(h[src], dst, h.shape[0])
        return self.norm(h + Fn.relu(self.mlp((1 + self.eps) * h + agg)))


class GATLayer(nn.Module):
    """Single-head GAT: α_uv = softmax_u(LeakyReLU(aᵀ[W h_u ‖ W h_v]))."""

    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.randn(d) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(d) * 0.1)
        self.out = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, edge_index, deg):
        src, dst = edge_index
        n = h.shape[0]
        z = self.W(h)
        e = Fn.leaky_relu((z * self.a_src).sum(1)[src] + (z * self.a_dst).sum(1)[dst], 0.2)
        e = e - scatter_max(e, dst, n, fill=0.0)[dst]
        ex = torch.exp(e)
        alpha = ex / scatter_sum(ex, dst, n)[dst].clamp_min(1e-16)
        m = scatter_sum(alpha.unsqueeze(1) * z[src], dst, n)
        return self.norm(h + Fn.relu(self.out(torch.cat([h, m], 1))))


LAYERS = {"mpnn": MPNNLayer, "gin": GINLayer, "gat": GATLayer}


class QNet(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3, encoder: str = "mpnn",
                 readout: str = "meanmax"):
        super().__init__()
        d = hidden
        self.config = dict(hidden=hidden, layers=layers, encoder=encoder, readout=readout)
        self.inp = nn.Linear(NUM_FEATURES, d)
        self.layers = nn.ModuleList([LAYERS[encoder](d) for _ in range(layers)])
        self.readout = readout
        gdim = (2 * d if readout == "meanmax" else d) + NUM_GLOBAL
        self.w6 = nn.Linear(gdim, d)
        self.w7 = nn.Linear(d, d)
        self.w5 = nn.Linear(2 * d, 1)

    def embed(self, b: Batch):
        h = Fn.relu(self.inp(b.x))
        for layer in self.layers:
            h = layer(h, b.edge_index, b.deg)
        return h

    def forward(self, b: Batch, mask: bool = True) -> torch.Tensor:
        """Q-value per node, (N,). Invalid actions get -inf when mask=True."""
        h = self.embed(b)
        B = b.num_graphs
        if self.readout == "meanmax":
            cnt = (b.ptr[1:] - b.ptr[:-1]).clamp_min(1).to(h.dtype).unsqueeze(1)
            g = torch.cat([scatter_sum(h, b.batch, B) / cnt, scatter_max(h, b.batch, B)], 1)
        elif self.readout == "sum":                      # base-paper readout (ablation)
            g = scatter_sum(h, b.batch, B)
        else:
            raise ValueError(self.readout)
        g = torch.cat([g, b.glob], 1)
        z = torch.cat([self.w6(g)[b.batch], self.w7(h)], 1)
        q = self.w5(Fn.relu(z)).squeeze(1)
        if mask:
            q = q.masked_fill(~b.valid, NEG_INF)
        return q
