"""Safe reduction rule (change C6).

If deg(u) = 1, the unique neighbour of u is the only vertex that can cover u,
so it belongs to every total dominating set. Those vertices are pre-selected
and marked `forced`.
"""
from __future__ import annotations

import numpy as np

from .graph import Graph


def forced_vertices(graph: Graph) -> np.ndarray:
    """Sorted array of vertices that are in every TDS (support vertices of leaves).

    Raises NoTotalDominatingSetError if the graph has an isolated vertex.
    """
    graph.require_tds_exists()
    leaves = np.flatnonzero(graph.degrees == 1)
    if leaves.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(graph.indices[graph.indptr[leaves]])


def zero_weight_vertices(graph: Graph) -> np.ndarray:
    """Vertices of weight 0: adding them never increases W(S), so they are pre-selected."""
    if graph.weights is None:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(graph.weights == 0)


def preselected_vertices(graph: Graph) -> np.ndarray:
    """Forced (degree-1 rule) vertices plus free (weight 0) vertices."""
    return np.union1d(forced_vertices(graph), zero_weight_vertices(graph)).astype(np.int64)


def forced_mask(graph: Graph) -> np.ndarray:
    m = np.zeros(graph.n, dtype=bool)
    m[preselected_vertices(graph)] = True
    return m
