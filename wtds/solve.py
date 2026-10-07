"""Inference API and CLI.

    from wtds import solve
    res = solve(G, weights, method="rl", checkpoint="checkpoints/best.pt", rr=True, ls_time=1.0)

    python -m wtds.solve --graph path/to/edges.txt --method rl --out solution.json --plot solution.png

Every returned solution has been verified with the TDS checker (`res.is_valid`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

from .graph import Graph
from .postprocess import local_search, redundancy_removal
from .result import SolveResult

METHODS = ("rl_exact", "rl", "greedy", "greedy_rr", "greedy_rr_ls", "unweighted_greedy_rr", "old_tds_model", "ilp")
DEFAULT_CHECKPOINT = "checkpoints/best.pt"
OLD_TDS_CHECKPOINT = "checkpoints/pretrained/tds_seed0_best.pt"   # unit-reward TDS model
_MODEL_CACHE: dict = {}


def with_weights(graph: Graph, weights) -> Graph:
    """weights: None (keep / default 1), dict {vertex: w}, sequence of n weights, or a 'vertex weight' file."""
    import numpy as np
    from .graph import validate_weights
    if weights is None:
        return graph if graph.weights is not None else Graph(graph.n, graph.indptr, graph.indices,
                                                            np.ones(graph.n), graph.name, dict(graph.meta))
    if isinstance(weights, (str, Path)):
        from .io import attach_weights
        return attach_weights(graph, weights)
    if isinstance(weights, dict):
        labels = graph.meta.get("labels") or list(range(graph.n))
        idx = {l: i for i, l in enumerate(labels)}
        w = np.ones(graph.n)
        for k, x in weights.items():
            w[idx[k]] = float(x)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(graph.n)
    validate_weights(w, graph.meta.get("labels"))
    return Graph(graph.n, graph.indptr, graph.indices, w, graph.name, dict(graph.meta))


def as_graph(G) -> Graph:
    if isinstance(G, Graph):
        return G
    if isinstance(G, (str, Path)):
        from .io import read_graph
        return read_graph(G)
    try:
        import networkx as nx
        if isinstance(G, nx.Graph):
            return Graph.from_networkx(G)
    except ImportError:
        pass
    if isinstance(G, (list, tuple)):
        return Graph.from_adj_list(G)
    raise TypeError(f"cannot interpret {type(G)} as a graph")


def get_model(checkpoint, device):
    from .agent import load_model
    key = (str(Path(checkpoint).resolve()), str(device))
    if key not in _MODEL_CACHE:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"checkpoint {checkpoint} not found; train one with "
                                    "`python -m wtds.train --config configs/default.yaml`")
        _MODEL_CACHE[key] = load_model(checkpoint, device)
    return _MODEL_CACHE[key]


def default_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def tuned_temperature(default: float = 1.0) -> float:
    """Sampling temperature chosen on the VALIDATION set (scripts/tune_inference.py), if available."""
    p = Path(__file__).resolve().parents[1] / "checkpoints" / "inference.json"
    try:
        return float(json.loads(p.read_text())["temperature"])
    except (OSError, KeyError, ValueError):
        return default


def solve_rl(graph: Graph, checkpoint=DEFAULT_CHECKPOINT, rr=True, ls_time=1.0, samples=1,
             temperature=None, seed=0, device=None, multi_select="auto", trace=False,
             env_kwargs=None, method_name="rl") -> SolveResult:
    """samples = 1: greedy policy rollout. samples = k > 1: the greedy rollout PLUS k rollouts sampled from
    softmax(Q / temperature); the cheapest after redundancy removal is kept (never worse than k = 1)."""
    from .rollout import rollout
    graph.require_tds_exists()
    device = device or default_device()
    model = get_model(checkpoint, device)
    ek = dict(getattr(model, "env_kwargs", {}) or {})
    ek.update(env_kwargs or {})
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    runs = rollout(model, [graph], device, env_kwargs=ek, multi_select=multi_select, trace=trace)
    if samples > 1:
        T = tuned_temperature() if temperature is None else temperature
        runs += rollout(model, [graph] * samples, device, env_kwargs=ek, mode="sample",
                        temperature=T, rng=rng, batch_size=min(samples, 16), trace=False)
    t_policy = time.perf_counter() - t0
    best = None
    for r in runs:
        S_raw = r["solution"]
        S_rr = redundancy_removal(graph, S_raw, scores=r["q"], protected=r["forced"]) if rr else S_raw
        key = (float(graph.w[S_rr].sum()), float(graph.w[S_raw].sum()))
        if best is None or key < best[0]:
            best = (key, S_raw, S_rr, r)
    _, S_raw, S_rr, r = best
    t_rr = time.perf_counter() - t0 - t_policy
    S = S_rr
    if ls_time and ls_time > 0:
        S = local_search(graph, S_rr, time_limit=ls_time, seed=seed)
    info = dict(size_raw=len(S_raw), size_rr=len(S_rr), size_ls=len(S), policy_time=t_policy,
                rr_time=t_rr, samples=samples, checkpoint=str(checkpoint), steps=r["steps"],
                forced=len(r["forced"]), rr=rr, ls_time=ls_time, order=r["order"])
    if trace and "trace" in r:
        info["trace"] = r["trace"]
    name = method_name + ("_rr" if rr else "") + ("_ls" if ls_time else "") + (f"_k{samples}" if samples > 1 else "")
    return SolveResult.build(graph, S, time.perf_counter() - t0, name, **info)


def solve_rl_exact(graph: Graph, checkpoint=DEFAULT_CHECKPOINT, samples=16, ls_time=1.0, ilp_time=30.0,
                   seed=0, device=None, temperature=None) -> SolveResult:
    """GNN + RL search, then exact certification.

    1. The GNN policy (greedy rollout + `samples` sampled rollouts, RR, local search) finds a set S_rl.
    2. CP-SAT solves the exact weighted ILP with S_rl as its starting solution (upper bound) for up to
       `ilp_time` seconds. It either proves S_rl optimal, or improves it, or stops at the limit with a
       proven lower bound.
    The returned set is the cheaper of the two; info["optimal"] is True only when optimality is PROVEN.
    """
    from .baselines import ilp
    t0 = time.perf_counter()
    r = solve_rl(graph, checkpoint, rr=True, ls_time=ls_time, samples=samples, temperature=temperature,
                 seed=seed, device=device)
    t_rl = time.perf_counter() - t0
    e = ilp(graph, time_limit=ilp_time, hint=r.vertices, seed=seed)
    S = e.vertices if e.weight < r.weight - 1e-9 else r.vertices
    W = float(graph.w[S].sum())
    proven = bool(e.info["optimal"]) and abs(W - e.weight) < 1e-6
    bound = float(e.info["bound"])
    info = dict(rl_weight=r.weight, rl_size=r.size, rl_time=t_rl, ilp_time_used=e.runtime, optimal=proven,
                lower_bound=bound, certified_gap=(W - bound) / W if W > 0 and bound == bound else float("nan"),
                rl_was_optimal=proven and abs(r.weight - W) < 1e-6, improved_by_ilp=e.weight < r.weight - 1e-9,
                samples=samples, ls_time=ls_time, checkpoint=str(checkpoint))
    return SolveResult.build(graph, S, time.perf_counter() - t0, "rl_exact", **info)


def solve(G, weights=None, method: str = "rl", checkpoint=DEFAULT_CHECKPOINT, rr: bool = True,
          ls_time: float = 1.0, samples: int = 1, temperature=None, seed: int = 0, device=None,
          ilp_time: float = 60.0, multi_select="auto", trace: bool = False) -> SolveResult:
    """Minimum-WEIGHT total dominating set of G (Graph, networkx graph, adjacency list or file path)
    with vertex weights `weights` (see with_weights). The result's weight W(S) is recomputed from S."""
    graph = with_weights(as_graph(G), weights)
    graph.require_tds_exists()
    from . import baselines as B
    if method == "rl":
        return solve_rl(graph, checkpoint, rr, ls_time, samples, temperature, seed, device, multi_select, trace)
    if method == "rl_exact":               # GNN+RL answer, certified (or improved) by the exact solver
        return solve_rl_exact(graph, checkpoint, samples=max(samples, 16), ls_time=ls_time, ilp_time=ilp_time,
                              seed=seed, device=device, temperature=temperature)
    if method == "old_tds_model":          # previous unit-reward TDS model on weighted graphs (+ weighted RR)
        return solve_rl(graph, OLD_TDS_CHECKPOINT, rr, ls_time, 1, temperature, seed, device, multi_select,
                        trace, method_name="old_tds_model")
    if method == "unweighted_greedy_rr":
        return B.unweighted_greedy_rr(graph)
    if method == "greedy":
        return B.greedy(graph)
    if method == "greedy_rr":
        return B.greedy_rr(graph)
    if method == "greedy_rr_ls":
        return B.greedy_rr_ls(graph, ls_time=ls_time, seed=seed)
    if method == "ilp":
        return B.ilp(graph, time_limit=ilp_time)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def plot_solution(graph: Graph, S, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    G = graph.to_networkx()
    if graph.n > 3000:
        raise ValueError("graph too large to draw (> 3000 vertices)")
    pos = nx.kamada_kawai_layout(G) if graph.n <= 300 else nx.spring_layout(G, seed=0)
    Sset = set(S)
    colors = ["#d62728" if v in Sset else "#c7d7e8" for v in G.nodes]
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=max(10, 3000 // max(graph.n, 1)),
                           edgecolors="k", linewidths=0.3)
    ax.set_title(title or f"TDS |S|={len(S)} (red), n={graph.n}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Minimum total dominating set solver")
    ap.add_argument("--graph", required=True, help="edge list / combined v,e file / .col / .clq / .mtx")
    ap.add_argument("--weights", default=None, help="'vertex weight' per line (default: from the graph file, else 1)")
    ap.add_argument("--method", default="rl", choices=METHODS)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--no-rr", action="store_true")
    ap.add_argument("--ls-time", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ilp-time", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="write solution JSON here")
    ap.add_argument("--plot", default=None, help="write a PNG drawing here")
    a = ap.parse_args()
    from .io import read_graph
    g = read_graph(a.graph, a.weights)
    for msg in g.meta.get("weight_warnings", []):
        print("warning:", msg)
    res = solve(g, None, a.method, a.checkpoint, rr=not a.no_rr, ls_time=a.ls_time, samples=a.samples,
                temperature=a.temperature, seed=a.seed, device=a.device, ilp_time=a.ilp_time)
    labels = g.meta.get("labels")
    d = res.to_dict()
    d["info"].pop("order", None)
    d.update(graph=a.graph, n=g.n, m=g.num_edges)
    if labels:
        d["vertex_labels"] = [labels[v] for v in res.vertices]
    print(f"{res.method}: W(S) = {res.weight:g}, |S| = {res.size}, valid = {res.is_valid}, time = {res.runtime:.3f}s")
    if a.out:
        Path(a.out).write_text(json.dumps(d, indent=2))
        print("wrote", a.out)
    if a.plot:
        plot_solution(g, res.vertices, a.plot, title=f"{res.method}: |S|={res.size}")
        print("wrote", a.plot)


class _CallableModule(types.ModuleType):
    """Lets `from wtds import solve` (which yields this module) be called like the function."""

    def __call__(self, *args, **kwargs):
        return solve(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

if __name__ == "__main__":
    main()
