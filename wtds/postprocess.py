"""Post-processing (change C7): redundancy removal and local search.

Both operate on cover_count[u] = |N(u) ∩ S|. S \\ {v} is still a TDS iff every
u in N(v) has cover_count[u] >= 2, an O(deg v) test.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .checker import assert_tds, cover_counts
from .graph import Graph


def redundancy_removal(graph: Graph, S, scores: Optional[dict] = None,
                       protected=None, check: bool = True, by_weight: bool = True) -> list:
    """Remove redundant vertices until S is a minimal TDS.

    Candidates are tried in ascending (score, degree, index), where score is the
    Q-value (RL) or gain (greedy) at selection time; weighted graphs try heavier
    vertices first. `protected` vertices (e.g. forced) are never removed; forced
    vertices are never removable anyway because their leaf has cover_count 1.

    One pass already yields a minimal set (removals only lower cover counts, so a
    vertex that was not removable never becomes removable); the loop re-checks.
    """
    S = list(dict.fromkeys(int(v) for v in S))
    if check:
        assert_tds(graph, S, "redundancy_removal input")
    adj = graph.adj
    deg = graph.degrees
    w = graph.w
    cc = cover_counts(graph, S).tolist()
    in_s = set(S)
    prot = set(int(v) for v in protected) if protected is not None else set()
    scores = scores or {}

    def key(v):
        return (-w[v] if (graph.is_weighted and by_weight) else 0.0, scores.get(v, 0.0), int(deg[v]), v)

    order = sorted(S, key=key)
    changed = True
    while changed:
        changed = False
        for v in order:
            if v not in in_s or v in prot:
                continue
            nb = adj[v]
            if all(cc[u] >= 2 for u in nb):
                in_s.discard(v)
                for u in nb:
                    cc[u] -= 1
                changed = True
    out = [v for v in S if v in in_s]
    if check:
        assert_tds(graph, out, "redundancy_removal output")
    return out


def is_minimal(graph: Graph, S) -> bool:
    """True iff no single vertex can be removed from S keeping it a TDS."""
    cc = cover_counts(graph, S)
    return all(not (cc[graph.neighbors(v)] >= 2).all() for v in S)


def local_search(graph: Graph, S, time_limit: float = 1.0, seed: int = 0, tabu_tenure: int = 7,
                 debug: bool = False, max_iters: Optional[int] = None) -> list:
    """Time-limited local search on a feasible S (Section 5). Never returns a worse
    or infeasible set.

    Weight-aware; every accepted move strictly decreases W(S):
    "2-for-1": add c ∉ S and remove two non-preselected a, b ∈ S that became removable,
      accepted iff w(c) < w(a) + w(b).
    "1-for-1": replace a non-preselected a ∈ S by the cheapest c ∉ S covering all of a's
      private neighbours, accepted iff w(c) < w(a).
    Redundancy removal (most expensive first) runs around c after each move.
    """
    import random
    import time as _time

    from .reductions import forced_vertices

    t_end = _time.perf_counter() + time_limit
    rnd = random.Random(seed)
    adj = graph.adj
    n = graph.n
    w = graph.w.tolist()
    deg = graph.degrees.tolist()
    from .reductions import preselected_vertices
    forced = set(preselected_vertices(graph).tolist())        # forced + free (weight 0) vertices
    S = list(dict.fromkeys(int(v) for v in S))
    assert_tds(graph, S, "local_search input")
    in_s = [False] * n
    for v in S:
        in_s[v] = True
    cc = cover_counts(graph, S).tolist()
    cur_w = sum(w[v] for v in S)
    best_w, best = cur_w, set(S)
    tabu = {}                       # vertex -> swap iteration until which it may not be added
    swap_it = 0

    def add(c):
        in_s[c] = True
        for u in adj[c]:
            cc[u] += 1

    def remove(a):
        in_s[a] = False
        for u in adj[a]:
            cc[u] -= 1

    def removable(a):
        return all(cc[u] >= 2 for u in adj[a])

    def near_s(c):
        """Non-forced selected vertices sharing a neighbour with c (the only ones
        whose removability can change when c is added)."""
        out = set()
        for u in adj[c]:
            for a in adj[u]:
                if in_s[a] and a != c and a not in forced:
                    out.add(a)
        return out

    def local_rr(c):
        nonlocal cur_w
        for a in sorted(near_s(c), key=lambda a: (-w[a], deg[a], a)):
            if in_s[a] and removable(a):
                remove(a)
                cur_w -= w[a]

    def check(ctx):
        if debug:
            assert_tds(graph, [v for v in range(n) if in_s[v]], f"local_search {ctx}")

    def try_two_for_one():
        nonlocal cur_w
        cand = [c for c in range(n) if not in_s[c]]
        rnd.shuffle(cand)
        for c in cand:
            if _time.perf_counter() > t_end:
                return False
            add(c)
            A1 = [a for a in near_s(c) if removable(a)]
            if len(A1) >= 2:
                A1.sort(key=lambda a: (-w[a], deg[a], a))
                for i, a in enumerate(A1):
                    remove(a)
                    for b in A1[i + 1:]:
                        if w[c] < w[a] + w[b] - 1e-12 and removable(b):
                            remove(b)
                            cur_w += w[c] - w[a] - w[b]
                            local_rr(c)
                            check("2-for-1")
                            return True
                    add(a)
            remove(c)
        return False

    def swap_move():
        nonlocal cur_w, swap_it
        sel = sorted((a for a in range(n) if in_s[a] and a not in forced), key=lambda a: (-w[a], a))
        for a in sel:
            if _time.perf_counter() > t_end:
                return False
            priv = [u for u in adj[a] if cc[u] == 1]
            if not priv:                      # redundant: plain removal
                remove(a)
                cur_w -= w[a]
                return True
            cands = set(adj[priv[0]])
            for u in priv[1:]:
                cands &= set(adj[u])
            cands = [c for c in cands if not in_s[c] and c != a and w[c] < w[a] - 1e-12]
            if not cands:
                continue
            c = min(cands, key=lambda c: (w[c], c))
            remove(a)
            add(c)
            cur_w += w[c] - w[a]
            local_rr(c)
            check("swap")
            return True
        return False

    iters = 0
    while _time.perf_counter() < t_end and (max_iters is None or iters < max_iters):
        iters += 1
        moved = try_two_for_one() or swap_move()
        if cur_w < best_w - 1e-12:
            best_w, best = cur_w, {v for v in range(n) if in_s[v]}
        if not moved:
            break
    out = sorted(best)
    assert_tds(graph, out, "local_search output")
    return out
