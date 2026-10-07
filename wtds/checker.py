"""The single source of truth for feasibility: is S a total dominating set?

S is a TDS of G  <=>  N(v) ∩ S != ∅ for every v in V  (open neighbourhood).
"""
from __future__ import annotations

import numpy as np

from .graph import Graph


class InvalidSolutionError(AssertionError):
    pass


def is_total_dominating_set(adj_list, S) -> bool:
    """Reference checker from the specification (pure Python)."""
    if isinstance(adj_list, Graph):
        adj_list = adj_list.adj
    S = set(int(x) for x in S)
    return all(any(u in S for u in adj_list[v]) for v in range(len(adj_list)))


def cover_counts(graph: Graph, S) -> np.ndarray:
    """cover_count[u] = |N(u) ∩ S|, vectorised."""
    in_s = np.zeros(graph.n, dtype=bool)
    S = np.asarray(list(S), dtype=np.int64)
    if S.size:
        in_s[S] = True
    src, dst = graph.edge_index
    return np.bincount(dst[in_s[src]], minlength=graph.n)


def is_tds_fast(graph: Graph, S) -> bool:
    """Vectorised equivalent of is_total_dominating_set (used on large graphs)."""
    S = list(S)
    if any((int(v) < 0 or int(v) >= graph.n) for v in S):
        return False
    return bool((cover_counts(graph, S) > 0).all()) if graph.n else False


def verify(graph: Graph, S) -> bool:
    """Checks with both implementations on small graphs, the fast one on large."""
    fast = is_tds_fast(graph, S)
    if graph.n <= 5000:
        ref = is_total_dominating_set(graph.adj, S)
        if ref != fast:  # pragma: no cover - would be a bug in the checker itself
            raise RuntimeError("checker implementations disagree")
    return fast


def assert_tds(graph: Graph, S, context: str = "") -> None:
    if not verify(graph, S):
        unc = np.flatnonzero(cover_counts(graph, S) == 0)
        raise InvalidSolutionError(
            f"{context}: not a total dominating set; {len(unc)} vertices have no "
            f"neighbour in S (e.g. {unc[:10].tolist()})")
