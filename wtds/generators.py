"""Graph generators.

Every generated graph is described by a *spec* dict (family, parameters, seed).
`make_graph(spec)` is deterministic, so any graph can be regenerated from its
spec, which is stored in `graph.meta`.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import networkx as nx
import numpy as np

from .graph import Graph

FAMILIES = ("er", "ba", "ws", "rgg", "grid", "tri", "hex", "tree", "path", "cycle",
            "complete", "star", "bipartite", "petersen")


# ---------------------------------------------------------------- primitives
def _er(n, avg_deg, seed):
    p = min(1.0, avg_deg / max(n - 1, 1))
    return nx.fast_gnp_random_graph(n, p, seed=seed) if p < 0.2 else nx.gnp_random_graph(n, p, seed=seed)


def _rgg(n, avg_deg, seed):
    r = math.sqrt(avg_deg / (math.pi * max(n - 1, 1)))
    G = nx.random_geometric_graph(n, r, seed=seed)
    for u in G.nodes:
        G.nodes[u].pop("pos", None)
    return G


def _sorted_ints(G):
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def tri_lattice(rows: int, cols: int):
    return _sorted_ints(nx.triangular_lattice_graph(rows, cols, with_positions=False))


def hex_lattice(rows: int, cols: int):
    return _sorted_ints(nx.hexagonal_lattice_graph(rows, cols, with_positions=False))


def _lattice_count(kind: str, a: int, b: int) -> int:
    """Vertex count of networkx's triangular/hexagonal_lattice_graph(a, b)
    (formulas checked against networkx for a, b in 1..14)."""
    if kind == "tri":
        return (a + 1) * ((b + 1) // 2 + 1) - (b % 2) * ((a + 1) // 2)
    return 2 * (a + 1) * (b + 1) - 2


@lru_cache(maxsize=None)
def lattice_dims(kind: str, n_target: int) -> tuple:
    """Near-square lattice parameters whose vertex count is closest to n_target."""
    if kind == "grid":
        r = max(2, int(round(math.sqrt(n_target))))
        c = max(2, int(round(n_target / r)))
        return r, c
    if kind not in ("tri", "hex"):
        raise ValueError(kind)
    # b counts half-cells for "tri", so a square triangular patch has b ~ 2a.
    ratio = 2.0 if kind == "tri" else 1.0
    tol = max(2, int(0.03 * n_target))
    best = None
    s = max(2, int(math.sqrt(n_target)))
    for a in range(1, 2 * s + 3):
        for b in range(1, 3 * s + 3):
            err = abs(_lattice_count(kind, a, b) - n_target)
            key = (max(err, tol), abs(math.log(ratio * a / b)), err)
            if best is None or key < best[0]:
                best = (key, (a, b))
    return best[1]


def _build_nx(spec: dict):
    f = spec["family"]
    seed = spec.get("seed")
    n = spec.get("n")
    if f == "er":
        return _er(n, spec["avg_deg"], seed)
    if f == "ba":
        return nx.barabasi_albert_graph(n, min(spec["m"], n - 1), seed=seed)
    if f == "ws":
        k = min(spec["k"], n - 1 - ((n - 1) % 2))
        return nx.watts_strogatz_graph(n, k, spec["p"], seed=seed)
    if f == "rgg":
        return _rgg(n, spec["avg_deg"], seed)
    if f == "grid":
        return _sorted_ints(nx.grid_2d_graph(spec["rows"], spec["cols"]))
    if f == "tri":
        return tri_lattice(spec["rows"], spec["cols"])
    if f == "hex":
        return hex_lattice(spec["rows"], spec["cols"])
    if f == "tree":
        return nx.random_labeled_tree(n, seed=seed)
    if f == "path":
        return nx.path_graph(n)
    if f == "cycle":
        return nx.cycle_graph(n)
    if f == "complete":
        return nx.complete_graph(n)
    if f == "star":
        return nx.star_graph(spec["m"])            # m leaves, m+1 vertices
    if f == "bipartite":
        return nx.complete_bipartite_graph(spec["a"], spec["b"])
    if f == "petersen":
        return nx.petersen_graph()
    raise ValueError(f"unknown family {f!r}")


def repair_isolated(G, rng: np.random.Generator):
    """Attach every isolated vertex to a uniformly random other vertex."""
    nodes = list(G.nodes())
    if len(nodes) < 2:
        raise ValueError("need at least 2 vertices")
    for u in [u for u in nodes if G.degree(u) == 0]:
        if G.degree(u) == 0:
            v = u
            while v == u:
                v = nodes[int(rng.integers(len(nodes)))]
            G.add_edge(u, v)
    return G


WEIGHT_DISTS = ("uniform_int", "uniform_small", "degree_correlated", "unit")
TRAIN_WEIGHT_MIX = {"uniform_int": 0.5, "uniform_small": 0.2, "degree_correlated": 0.3}


def sample_weights(dist: str, degrees: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """uniform_int: U{1..100}; uniform_small: U{1..10}; degree_correlated: deg + U{0..10}; unit: all 1."""
    n = len(degrees)
    if dist == "uniform_int":
        return rng.integers(1, 101, size=n).astype(float)
    if dist == "uniform_small":
        return rng.integers(1, 11, size=n).astype(float)
    if dist == "degree_correlated":
        return (degrees + rng.integers(0, 11, size=n)).astype(float)
    if dist == "unit":
        return np.ones(n)
    raise ValueError(f"unknown weight distribution {dist!r}; choose from {WEIGHT_DISTS}")


def make_graph(spec: dict, repair: bool = True) -> Graph:
    """Deterministically build the graph described by `spec`."""
    spec = dict(spec)
    G = _build_nx(spec)
    seed = spec.get("seed")
    rng = np.random.default_rng(None if seed is None else seed + 7919)
    if repair:
        repair_isolated(G, rng)
    weights = None
    if spec.get("weights"):
        dist = spec["weights"] if isinstance(spec["weights"], str) else "uniform_int"
        weights = sample_weights(dist, np.array([d for _, d in sorted(G.degree())]), rng)
    name = spec.get("name") or "_".join(
        str(v) if k == "family" else f"{k}{round(v, 3) if isinstance(v, float) else v}"
        for k, v in spec.items() if k not in ("weights", "weight_range", "name"))
    return Graph.from_networkx(G, name=name, meta=spec) if weights is None else \
        Graph.from_edges(G.number_of_nodes(), list(G.edges()), weights=weights, name=name, meta=spec)


# ------------------------------------------------------------ convenience API
def er(n, avg_deg, seed=None, **kw):
    return make_graph(dict(family="er", n=n, avg_deg=avg_deg, seed=seed, **kw))


def ba(n, m, seed=None, **kw):
    return make_graph(dict(family="ba", n=n, m=m, seed=seed, **kw))


def ws(n, k, p, seed=None, **kw):
    return make_graph(dict(family="ws", n=n, k=k, p=p, seed=seed, **kw))


def rgg(n, avg_deg, seed=None, **kw):
    return make_graph(dict(family="rgg", n=n, avg_deg=avg_deg, seed=seed, **kw))


def grid(rows, cols):
    return make_graph(dict(family="grid", rows=rows, cols=cols))


def lattice(kind: str, n_target: int) -> Graph:
    a, b = lattice_dims(kind, n_target)
    return make_graph(dict(family=kind, rows=a, cols=b))


# ------------------------------------------------------- training distribution
class GraphSampler:
    """Mixed-family random graph distribution (Section 7).

    Each call draws a fresh spec from a fixed-seed RNG, so the whole sequence of
    training graphs is reproducible from `seed`.
    """

    def __init__(self, seed: int, families: Optional[dict] = None, weighted: bool = False):
        self.rng = np.random.default_rng(seed)
        self.families = families or {"er": 0.30, "ba": 0.25, "ws": 0.15, "rgg": 0.15, "lattice": 0.15}
        self.weighted = weighted

    def sample_spec(self, n_min: int, n_max: int) -> dict:
        rng = self.rng
        names = list(self.families)
        probs = np.array([self.families[k] for k in names], dtype=float)
        fam = names[int(rng.choice(len(names), p=probs / probs.sum()))]
        n = int(rng.integers(n_min, n_max + 1))
        seed = int(rng.integers(2**31 - 1))
        if fam == "er":
            spec = dict(family="er", n=n, avg_deg=float(rng.uniform(3, 15)), seed=seed)
        elif fam == "ba":
            spec = dict(family="ba", n=n, m=int(rng.integers(1, 9)), seed=seed)
        elif fam == "ws":
            spec = dict(family="ws", n=n, k=int(rng.choice([4, 6])), p=float(rng.uniform(0.05, 0.3)), seed=seed)
        elif fam == "rgg":
            spec = dict(family="rgg", n=n, avg_deg=float(rng.uniform(3, 12)), seed=seed)
        elif fam == "lattice":
            kind = ["grid", "tri", "hex"][int(rng.integers(3))]
            a, b = lattice_dims(kind, n)
            spec = dict(family=kind, rows=a, cols=b, seed=seed)
        else:
            raise ValueError(fam)
        if self.weighted:
            mix = self.weighted if isinstance(self.weighted, dict) else TRAIN_WEIGHT_MIX
            names = list(mix)
            pw = np.array([mix[k] for k in names], dtype=float)
            spec["weights"] = names[int(rng.choice(len(names), p=pw / pw.sum()))]
        return spec

    def sample(self, n_min: int, n_max: int) -> Graph:
        return make_graph(self.sample_spec(n_min, n_max))
