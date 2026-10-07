"""Simple undirected graph in CSR form.

Every module in the package works on this class. Self-loops and duplicate edges
are dropped on construction, because total domination is defined with the
*open* neighbourhood N(v), which never contains v.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


class NoTotalDominatingSetError(ValueError):
    """Raised when the graph has an isolated vertex, so no TDS exists."""


class InvalidWeightError(ValueError):
    """Raised for a negative, NaN or infinite vertex weight."""


def validate_weights(w, labels=None) -> None:
    """Weights must be finite and >= 0 (0 = free vertex, pre-selected)."""
    w = np.asarray(w, dtype=np.float64)
    bad = np.flatnonzero(~np.isfinite(w) | (w < 0))
    if bad.size:
        v = int(bad[0])
        name = labels[v] if labels is not None else v
        raise InvalidWeightError(f"vertex {name} has weight {w[v]!r}; weights must be finite and >= 0 "
                                 f"({bad.size} invalid weight(s) in total)")


@dataclass(eq=False)
class Graph:
    n: int
    indptr: np.ndarray            # (n+1,) int64, CSR row pointers
    indices: np.ndarray           # (2|E|,) int64, sorted neighbours of each vertex
    weights: Optional[np.ndarray] = None   # (n,) float64, None = unweighted
    name: str = ""
    meta: dict = field(default_factory=dict)  # generator spec / provenance

    # ------------------------------------------------------------------ build
    @classmethod
    def from_edges(cls, n: int, edges: Iterable, weights=None, name: str = "",
                   meta: Optional[dict] = None) -> "Graph":
        n = int(n)
        e = np.asarray(list(edges) if not isinstance(edges, np.ndarray) else edges,
                       dtype=np.int64).reshape(-1, 2)
        if e.size and (e.min() < 0 or e.max() >= n):
            raise ValueError("edge endpoint out of range [0, n)")
        e = e[e[:, 0] != e[:, 1]]                       # no self-loops
        e = np.sort(e, axis=1)
        e = np.unique(e, axis=0) if len(e) else e       # no multi-edges
        src = np.concatenate([e[:, 0], e[:, 1]])
        dst = np.concatenate([e[:, 1], e[:, 0]])
        order = np.lexsort((dst, src))
        src, dst = src[order], dst[order]
        deg = np.bincount(src, minlength=n).astype(np.int64)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(deg, out=indptr[1:])
        w = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(n)
        if w is not None:
            validate_weights(w)
        return cls(n=n, indptr=indptr, indices=dst.astype(np.int64), weights=w,
                   name=name, meta=dict(meta or {}))

    @classmethod
    def from_networkx(cls, G, name: str = "", meta: Optional[dict] = None,
                      weight_attr: Optional[str] = None) -> "Graph":
        nodes = list(G.nodes())
        idx = {u: i for i, u in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in G.edges()]
        w = None
        if weight_attr is not None:
            w = [G.nodes[u].get(weight_attr, 1.0) for u in nodes]
        return cls.from_edges(len(nodes), edges, weights=w, name=name, meta=meta)

    @classmethod
    def from_adj_list(cls, adj_list, **kw) -> "Graph":
        edges = [(v, u) for v, nb in enumerate(adj_list) for u in nb]
        return cls.from_edges(len(adj_list), edges, **kw)

    # ------------------------------------------------------------- properties
    @property
    def degrees(self) -> np.ndarray:
        return np.diff(self.indptr)

    @property
    def num_edges(self) -> int:
        return int(len(self.indices) // 2)

    @property
    def max_degree(self) -> int:
        return int(self.degrees.max()) if self.n else 0

    @property
    def is_weighted(self) -> bool:
        return self.weights is not None

    @property
    def w(self) -> np.ndarray:
        """Vertex weights (all ones when unweighted)."""
        return self.weights if self.weights is not None else np.ones(self.n)

    @property
    def edge_index(self) -> np.ndarray:
        """(2, 2|E|) int64, both directions, no self-loops. Row 0 = source."""
        if "_edge_index" not in self.__dict__:
            src = np.repeat(np.arange(self.n, dtype=np.int64), self.degrees)
            self.__dict__["_edge_index"] = np.stack([src, self.indices])
        return self.__dict__["_edge_index"]

    @property
    def adj(self) -> list:
        """Adjacency as a list of Python lists (fast for scalar Python loops)."""
        if "_adj" not in self.__dict__:
            ind = self.indices.tolist()
            ptr = self.indptr.tolist()
            self.__dict__["_adj"] = [ind[ptr[v]:ptr[v + 1]] for v in range(self.n)]
        return self.__dict__["_adj"]

    def neighbors(self, v: int) -> np.ndarray:
        return self.indices[self.indptr[v]:self.indptr[v + 1]]

    def edges(self) -> np.ndarray:
        """(|E|, 2) array with u < v."""
        ei = self.edge_index
        m = ei[0] < ei[1]
        return np.stack([ei[0][m], ei[1][m]], axis=1)

    def isolated_vertices(self) -> np.ndarray:
        return np.flatnonzero(self.degrees == 0)

    def require_tds_exists(self) -> None:
        iso = self.isolated_vertices()
        if self.n == 0:
            raise NoTotalDominatingSetError("empty graph")
        if len(iso):
            raise NoTotalDominatingSetError(
                f"graph has {len(iso)} isolated vertex/vertices (e.g. {iso[:5].tolist()}); "
                "a total dominating set exists iff there is no isolated vertex")

    def to_networkx(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(self.edges().tolist())
        if self.weights is not None:
            nx.set_node_attributes(G, {i: float(x) for i, x in enumerate(self.weights)}, "weight")
        return G

    def __repr__(self) -> str:
        return (f"Graph(name={self.name!r}, n={self.n}, m={self.num_edges}, "
                f"weighted={self.is_weighted})")
