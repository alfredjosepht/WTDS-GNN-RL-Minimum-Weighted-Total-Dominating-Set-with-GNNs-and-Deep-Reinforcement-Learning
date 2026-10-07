"""Greedy for minimum TDS = greedy set cover over the open neighbourhoods N(v).

Forced vertices (support vertices of leaves) are added first. Then repeatedly add
the vertex with maximum gain(v) = #uncovered vertices in N(v), ties broken by
higher degree, then lower index. Weighted: maximise gain(v) / w(v).
Guarantee: |S| <= H(Δ) * γ_t(G), H = harmonic number (≈ ln Δ + 1).

Implementation: lazy max-heap. Gains only decrease, so a popped entry whose stored
gain equals the current gain is a true maximum. Total work O(|E| log |V|).
"""
from __future__ import annotations

import heapq
import time

from ..graph import Graph
from ..reductions import forced_vertices
from ..result import SolveResult


def greedy_construct(graph: Graph, use_forced: bool = True, use_weights: bool = True):
    """Returns (S in selection order, score at selection time) without verifying."""
    graph.require_tds_exists()
    adj = graph.adj
    n = graph.n
    deg = graph.degrees.tolist()
    weighted = graph.is_weighted and use_weights
    w = graph.w.tolist()
    covered = [False] * n
    selected = [False] * n
    gain = deg[:]                       # nothing covered yet
    n_uncovered = n
    order, scores = [], []

    def add(v):
        nonlocal n_uncovered
        selected[v] = True
        order.append(v)
        for u in adj[v]:
            if not covered[u]:
                covered[u] = True
                n_uncovered -= 1
                for x in adj[u]:
                    gain[x] -= 1

    from ..reductions import preselected_vertices
    pre = preselected_vertices(graph) if weighted else forced_vertices(graph)
    if use_forced:
        for v in pre.tolist():
            scores.append(float("inf"))  # never removed by RR anyway
            add(v)

    def key(v):
        # weighted: max gain/w, ties lower w, higher degree, lower index; unweighted: max gain, degree, index
        if weighted:
            return (-gain[v] / w[v], w[v], -deg[v], v)
        return (-gain[v], -deg[v], v)

    heap = [key(v) for v in range(n) if not selected[v] and gain[v] > 0]
    heapq.heapify(heap)
    while n_uncovered > 0:
        k = heapq.heappop(heap)
        v = k[-1]
        if selected[v] or gain[v] == 0:
            continue
        cur = key(v)
        if cur != k:                     # stale entry: re-insert with fresh gain
            heapq.heappush(heap, cur)
            continue
        scores.append(float(gain[v]))
        add(v)
    return order, scores


def greedy(graph: Graph, use_forced: bool = True) -> SolveResult:
    t0 = time.perf_counter()
    order, scores = greedy_construct(graph, use_forced)
    return SolveResult.build(graph, order, time.perf_counter() - t0, "greedy",
                             order=order, scores=scores)


def greedy_rr(graph: Graph) -> SolveResult:
    from ..postprocess import redundancy_removal
    t0 = time.perf_counter()
    order, scores = greedy_construct(graph)
    S = redundancy_removal(graph, order, scores=dict(zip(order, scores)))
    return SolveResult.build(graph, S, time.perf_counter() - t0, "greedy_rr",
                             size_before_rr=len(order))


def greedy_rr_ls(graph: Graph, ls_time: float = 1.0, seed: int = 0) -> SolveResult:
    from ..postprocess import local_search, redundancy_removal
    t0 = time.perf_counter()
    order, scores = greedy_construct(graph)
    S = redundancy_removal(graph, order, scores=dict(zip(order, scores)))
    S2 = local_search(graph, S, time_limit=ls_time, seed=seed)
    return SolveResult.build(graph, S2, time.perf_counter() - t0, "greedy_rr_ls",
                             size_before_rr=len(order), size_before_ls=len(S))


def unweighted_greedy_rr(graph: Graph) -> SolveResult:
    """The previous (size-minimising) greedy + RR, ignoring weights; W(S) is still reported."""
    from ..postprocess import redundancy_removal
    t0 = time.perf_counter()
    order, scores = greedy_construct(graph, use_weights=False)
    S = redundancy_removal(graph, order, scores=dict(zip(order, scores)), by_weight=False)
    return SolveResult.build(graph, S, time.perf_counter() - t0, "unweighted_greedy_rr")
