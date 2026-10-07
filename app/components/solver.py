"""Solver calls for the app. Every result is verified with is_total_dominating_set and its
weight W(S) is recomputed from the solution here."""
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

from wtds.checker import is_total_dominating_set

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"Seed 0 (default)": ROOT / "checkpoints" / "best.pt", "Seed 1": ROOT / "checkpoints" / "seed1_best.pt"}
OLD_MODEL = ROOT / "checkpoints" / "pretrained" / "tds_seed0_best.pt"
METHOD_LABELS = {"exact": "GNN + RL → proven minimum", "rl": "GNN + RL (ours, weighted)", "greedy": "Weighted greedy", "greedy_rr": "Weighted greedy + RR",
                 "old": "Unweighted TDS model", "unweighted_greedy_rr": "Unweighted greedy + RR", "ilp": "Exact ILP"}
ILP_MAX_N = 300
from wtds.solve import tuned_temperature
SAMPLE_TEMPERATURE = tuned_temperature()   # chosen on the validation set (scripts/tune_inference.py)


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    from wtds.agent import load_model as _load
    return _load(path, device())


def available_models():
    return {k: v for k, v in MODELS.items() if v.exists()}


def model_info():
    p = ROOT / "checkpoints" / "model_info.json"
    return json.loads(p.read_text()) if p.exists() else None


def _verified(graph, S, method, runtime, **info):
    S = sorted(int(v) for v in set(S))
    ok = is_total_dominating_set(graph.adj, S)
    if not ok:                                     # never show an invalid solution
        raise RuntimeError(f"{method} returned a set that is not a total dominating set")
    return {"method": method, "S": S, "size": len(S), "weight": float(graph.w[S].sum()), "runtime": runtime,
            "valid": ok, **info}


def _rl(graph, path, label, rr=True, ls_time=0.0, samples=1, trace=False):
    from wtds.solve import _MODEL_CACHE, solve_rl
    m = load_model(str(path))
    _MODEL_CACHE[(str(Path(path).resolve()), str(device()))] = m      # share the cached model with wtds.solve
    res = solve_rl(graph, str(path), rr=rr, ls_time=ls_time, samples=samples, temperature=None,
                   device=device(), multi_select="auto", trace=trace and samples == 1)
    info = {k: res.info[k] for k in ("size_raw", "size_rr", "size_ls", "policy_time", "forced", "steps")}
    if trace and "trace" in res.info:
        info["trace"] = res.info["trace"]
    return _verified(graph, res.vertices, label, res.runtime, **info)


def run_rl(graph, model_key, rr=True, ls_time=0.0, samples=1, trace=False):
    return _rl(graph, available_models()[model_key], METHOD_LABELS["rl"], rr, ls_time, samples, trace)


def run_certified(graph, model_key, ls_time=1.0, samples=16, ilp_time=20.0):
    """GNN+RL search (greedy + sampled rollouts, RR, LS) certified by the exact solver (warm-started with it)."""
    from wtds.solve import _MODEL_CACHE, solve_rl_exact
    path = available_models()[model_key]
    _MODEL_CACHE[(str(Path(path).resolve()), str(device()))] = load_model(str(path))
    samples = samples if graph.n <= 1000 else 1          # sampled rollouts pick 1 vertex per pass: too slow when large
    r = solve_rl_exact(graph, str(path), samples=samples, ls_time=ls_time, ilp_time=ilp_time, device=device())
    keep = ("rl_weight", "rl_size", "rl_time", "ilp_time_used", "optimal", "lower_bound", "certified_gap",
            "rl_was_optimal", "improved_by_ilp")
    return _verified(graph, r.vertices, METHOD_LABELS["exact"], r.runtime, samples=samples,
                     **{k: r.info[k] for k in keep})


def run_old_model(graph, rr=True, ls_time=0.0):
    """The previous TDS model (trained with reward -1 per vertex) on the weighted graph."""
    return _rl(graph, OLD_MODEL, METHOD_LABELS["old"], rr, ls_time)


def run_baseline(graph, key, ls_time=0.0, ilp_time=30.0):
    from wtds.baselines import greedy, greedy_rr, greedy_rr_ls, ilp, unweighted_greedy_rr
    if key == "greedy":
        r = greedy(graph)
    elif key == "greedy_rr":
        r = greedy_rr_ls(graph, ls_time=ls_time) if ls_time > 0 else greedy_rr(graph)
    elif key == "unweighted_greedy_rr":
        r = unweighted_greedy_rr(graph)
    elif key == "ilp":
        if graph.n > ILP_MAX_N:
            raise ValueError(f"Exact ILP is limited to {ILP_MAX_N} vertices in the app")
        r = ilp(graph, time_limit=ilp_time)
        return _verified(graph, r.vertices, METHOD_LABELS["ilp"], r.runtime, optimal=bool(r.info["optimal"]),
                         bound=float(r.info["bound"]))
    else:
        raise ValueError(key)
    label = METHOD_LABELS[key] + (" + LS" if key == "greedy_rr" and ls_time > 0 else "")
    return _verified(graph, r.vertices, label, r.runtime)


@st.cache_data(show_spinner=False, max_entries=64)
def small_optimum(edges_key: tuple, weights_key: tuple, n: int):
    """Proven minimum weight by ILP for small graphs (n <= 150, 10 s), else None."""
    from wtds.baselines import ilp
    from wtds.graph import Graph
    g = Graph.from_edges(n, list(edges_key), weights=list(weights_key))
    r = ilp(g, time_limit=10.0)
    return (r.weight, r.size) if r.info["optimal"] else None


def optimum_for(lg):
    """(W*, source) when known: unit-weight grid closed form, or proven by the ILP (n <= 150)."""
    from .graph_input import known_optimum
    k = known_optimum(lg)
    if k:
        return k
    g = lg.graph
    if g.n <= 150 and not len(g.isolated_vertices()):
        v = small_optimum(tuple(map(tuple, g.edges().tolist())), tuple(float(x) for x in g.w), g.n)
        if v is not None:
            return v[0], "proven optimal by the exact ILP"
    return None


def fmt_w(x):
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def solution_json(lg, res):
    g = lg.graph
    return json.dumps({"method": res["method"], "total_weight": res["weight"], "size": res["size"],
                       "valid_total_dominating_set": res["valid"], "runtime_s": round(res["runtime"], 6),
                       "n": g.n, "m": g.num_edges,
                       "vertices": [{"vertex": lg.label(v), "weight": float(g.w[v])} for v in res["S"]]},
                      indent=2, default=str)


def solution_png(lg, S):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from .network import layout
    from .theme import GREY, TEAL
    g = lg.graph
    pos = layout(lg)
    G = g.to_networkx()
    pos = nx.spring_layout(G, seed=0) if pos is None else {k: (x, -y) for k, (x, y) in pos.items()}
    Sset = set(S)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#CFCAC0", width=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[TEAL if v in Sset else GREY for v in G.nodes],
                           node_size=[70 if v in Sset else 26 for v in G.nodes], linewidths=0)
    if g.n <= 150:
        nx.draw_networkx_labels(G, {k: (x, y - 0.035 * (1 if abs(y) < 2 else 30)) for k, (x, y) in pos.items()},
                                labels={v: fmt_w(g.w[v]) for v in G.nodes}, font_size=6, font_color="#6B6F76", ax=ax)
    W = float(np.asarray(g.w)[list(Sset)].sum()) if Sset else 0.0
    ax.set_title(f"Weighted total dominating set: W(S) = {fmt_w(W)}, |S| = {len(Sset)}  (n = {g.n})",
                 fontsize=11, color="#1F2328", loc="left")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()
