"""Build replay mini-batches directly on the GPU.

Equivalent to `ReplayBuffer.sample` + `collate` (tested in tests/test_model.py), but
the static structure of every graph is cached on the device and the state
features of all 2 × batch_size states are computed with a handful of vectorised
torch ops instead of ~128 numpy calls. State s_t and s_{t+n} of a transition live
on the same graph, so both mini-batches share one disjoint-union structure.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .env import F, NUM_FEATURES
from .model import Batch


class GPUGraphCache:
    def __init__(self, pool, device):
        self.pool = pool
        self.device = torch.device(device)
        self.cache: dict = {}

    def get(self, gid):
        c = self.cache.get(gid)
        if c is None:
            g = self.pool.graphs[gid]
            d = self.device
            c = {"n": g.n, "E": g.edge_index.shape[1],
                 "ei": torch.from_numpy(g.edge_index).to(d),
                 "deg": torch.from_numpy(g.degrees.astype(np.float32)).to(d),
                 "forced": torch.from_numpy(self.pool.forced[gid]).to(d),
                 "w": torch.from_numpy(g.w.astype(np.float32)).to(d),
                 "max_deg": float(max(g.max_degree, 1)), "wmax": float(g.w.max()),
                 "logn": math.log(g.n) / 10.0}
            self.cache[gid] = c
        return c

    def prune(self):
        for gid in [k for k in self.cache if k not in self.pool.graphs]:
            del self.cache[gid]


def build_batches(cache: GPUGraphCache, transitions, mask_mode: str = "gain"):
    """transitions: list of (gid, S_t, a, R, S_next, done, gamma_n).
    Returns (Batch for s_t, Batch for s_{t+n}, actions, rewards, dones, gammas) on device."""
    d = cache.device
    cs = [cache.get(tr[0]) for tr in transitions]
    B = len(cs)
    ns = np.array([c["n"] for c in cs], dtype=np.int64)
    Es = np.array([c["E"] for c in cs], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(ns)])
    N = int(offs[-1])
    ptr = torch.from_numpy(offs).to(d)
    batch = torch.repeat_interleave(torch.arange(B, device=d), torch.from_numpy(ns).to(d))
    eoff = torch.repeat_interleave(ptr[:-1], torch.from_numpy(Es).to(d))
    ei = torch.cat([c["ei"] for c in cs], dim=1) + eoff
    deg = torch.cat([c["deg"] for c in cs])
    forced = torch.cat([c["forced"] for c in cs])
    w = torch.cat([c["w"] for c in cs])
    per_g = torch.tensor([[c["max_deg"], c["wmax"], c["logn"]] for c in cs], dtype=torch.float32, device=d)
    md, wm, logn = per_g[:, 0][batch], per_g[:, 1][batch], per_g[:, 2]
    nf = torch.from_numpy(ns.astype(np.float32)).to(d)

    def states(which):
        idx = np.concatenate([tr[which].astype(np.int64) + offs[i] for i, tr in enumerate(transitions)]) \
            if any(len(tr[which]) for tr in transitions) else np.zeros(0, np.int64)
        sel = forced.clone()
        if idx.size:
            sel[torch.from_numpy(idx).to(d)] = True
        src, dst = ei
        sel_nbr = torch.zeros(N, device=d).index_add_(0, dst, sel[src].float())
        cov = sel_nbr > 0
        cov_nbr = torch.zeros(N, device=d).index_add_(0, dst, cov[src].float())
        gain = deg - cov_nbr
        valid = (~sel) & (gain > 0) if mask_mode == "gain" else ~sel
        X = torch.empty((N, NUM_FEATURES), device=d)
        X[:, F["selected"]] = sel.float()
        X[:, F["covered"]] = cov.float()
        X[:, F["forced"]] = forced.float()
        X[:, F["deg_norm"]] = deg / md
        X[:, F["gain_norm"]] = gain / deg
        X[:, F["gain_scaled"]] = gain / md
        X[:, F["covered_nbr_frac"]] = cov_nbr / deg
        X[:, F["selected_nbr_frac"]] = sel_nbr / deg
        X[:, F["valid_action"]] = valid.float()
        X[:, F["weight_norm"]] = w / wm
        n_sel = torch.zeros(B, device=d).index_add_(0, batch, sel.float())
        n_cov = torch.zeros(B, device=d).index_add_(0, batch, cov.float())
        glob = torch.stack([n_cov / nf, n_sel / nf, logn], dim=1)
        return Batch(x=X, edge_index=ei, batch=batch, glob=glob, deg=deg.clamp(min=1.0), valid=valid,
                     ptr=ptr, num_graphs=B)

    cur, nxt = states(1), states(4)
    actions = np.array([tr[2] for tr in transitions])
    rewards = np.array([tr[3] for tr in transitions], dtype=np.float32)
    dones = np.array([tr[5] for tr in transitions], dtype=bool)
    gammas = np.array([tr[6] for tr in transitions], dtype=np.float32)
    return cur, nxt, actions, rewards, dones, gammas


def cat_batches(a: Batch, b: Batch) -> Batch:
    """Disjoint union of two Batches (b's graphs follow a's)."""
    N = a.x.shape[0]
    return Batch(x=torch.cat([a.x, b.x]), edge_index=torch.cat([a.edge_index, b.edge_index + N], 1),
                 batch=torch.cat([a.batch, b.batch + a.num_graphs]), glob=torch.cat([a.glob, b.glob]),
                 deg=torch.cat([a.deg, b.deg]), valid=torch.cat([a.valid, b.valid]),
                 ptr=torch.cat([a.ptr, b.ptr[1:] + N]), num_graphs=a.num_graphs + b.num_graphs)
