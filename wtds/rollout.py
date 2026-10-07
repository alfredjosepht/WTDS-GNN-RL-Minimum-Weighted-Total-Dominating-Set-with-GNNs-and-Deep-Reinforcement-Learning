"""Batched policy rollouts (validation, inference, sampled best-of-k)."""
from __future__ import annotations

import math

import numpy as np
import torch

from .env import TDSEnv
from .model import collate


@torch.no_grad()
def q_values(model, envs, device) -> list:
    """One batched forward pass; returns a numpy Q array (masked, -inf) per env."""
    items = [(e.graph, *e.observation()) for e in envs]
    b = collate(items, device)
    q = model(b).float().cpu().numpy()
    ptr = b.ptr.cpu().numpy()
    return [q[ptr[i]:ptr[i + 1]] for i in range(len(envs))]


def picks_per_forward(n: int, multi_select) -> int:
    """How many vertices to add per forward pass. 1 = exact greedy policy.
    "auto" adds up to ceil(n / 1000) per pass on large graphs (an inference-time
    approximation: the next pick is re-validated against the env before use)."""
    if multi_select == "auto":
        return max(1, math.ceil(n / 1000))
    return max(1, int(multi_select))


def rollout(model, graphs, device, env_kwargs=None, mode: str = "greedy", temperature: float = 1.0,
            rng: np.random.Generator = None, batch_size: int = 32, multi_select=1, trace: bool = False):
    """Run the policy to termination on each graph.

    Returns a list of dicts: solution (all selected incl. forced), order (non-forced
    picks), q (Q-value at selection per picked vertex), forced, steps [, trace].
    """
    env_kwargs = env_kwargs or {}
    rng = rng or np.random.default_rng(0)
    model.eval()
    out = [None] * len(graphs)
    for start in range(0, len(graphs), batch_size):
        idx = list(range(start, min(start + batch_size, len(graphs))))
        envs = {i: TDSEnv(graphs[i], **env_kwargs) for i in idx}
        qsel = {i: {} for i in idx}
        tr = {i: [] for i in idx}
        k = {i: picks_per_forward(graphs[i].n, multi_select) for i in idx}
        active = [i for i in idx if not envs[i].done]
        while active:
            qs = q_values(model, [envs[i] for i in active], device)
            for i, q in zip(active, qs):
                env = envs[i]
                if mode == "sample":
                    valid = np.flatnonzero(np.isfinite(q))
                    z = q[valid] / max(temperature, 1e-6)
                    p = np.exp(z - z.max())
                    p /= p.sum()
                    cands = [int(valid[rng.choice(len(valid), p=p)])]
                else:
                    valid = np.flatnonzero(np.isfinite(q))
                    if k[i] == 1:
                        best = q[valid].max()
                        cands = [int(valid[np.flatnonzero(q[valid] == best)[0]])]
                    else:
                        top = valid[np.argsort(-q[valid], kind="stable")[:k[i]]]
                        cands = top.tolist()
                for v in cands:
                    if env.done or not env.valid_mask()[v]:
                        continue
                    before = env.covered.copy() if trace else None
                    env.step(v)
                    qsel[i][v] = float(q[v])
                    if trace:
                        fin = np.flatnonzero(np.isfinite(q))
                        top = fin[np.argsort(-q[fin], kind="stable")[:5]]
                        tr[i].append({"vertex": v, "q": float(q[v]),
                                      "newly_covered": np.flatnonzero(env.covered & ~before).tolist(),
                                      "top5": [[int(u), float(q[u])] for u in top],
                                      "covered_count": int(env.covered.sum())})
            active = [i for i in active if not envs[i].done]
        for i in idx:
            e = envs[i]
            out[i] = dict(solution=e.solution, order=list(e.order), q=qsel[i],
                          forced=e.forced_list.tolist(), steps=len(e.order))
            if trace:
                out[i]["trace"] = tr[i]
    return out
