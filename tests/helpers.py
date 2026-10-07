"""Shared random-graph pools for tests (deterministic)."""
import networkx as nx
import numpy as np

from wtds.generators import GraphSampler, make_graph, repair_isolated
from wtds.graph import Graph


def disjoint_union(g1: Graph, g2: Graph) -> Graph:
    e = np.concatenate([g1.edges(), g2.edges() + g1.n])
    return Graph.from_edges(g1.n + g2.n, e, name=f"{g1.name}+{g2.name}")


def small_random_graphs(count: int, seed: int, n_min: int = 3, n_max: int = 10):
    """Mixed tiny graphs: ER of varying density, trees, unions (disconnected)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        n = int(rng.integers(n_min, n_max + 1))
        kind = int(rng.integers(4))
        s = int(rng.integers(1 << 30))
        if kind == 0:
            g = make_graph(dict(family="er", n=n, avg_deg=float(rng.uniform(1, n - 1)), seed=s))
        elif kind == 1:
            g = make_graph(dict(family="tree", n=n, seed=s))
        elif kind == 2 and n >= 4:
            a = int(rng.integers(2, n - 1))
            g1 = make_graph(dict(family="er", n=a, avg_deg=float(rng.uniform(1, a)), seed=s))
            g2 = make_graph(dict(family="tree", n=n - a, seed=s + 1))
            g = disjoint_union(g1, g2)
        else:
            G = nx.gnp_random_graph(n, float(rng.uniform(0.1, 0.9)), seed=s)
            repair_isolated(G, rng)
            g = Graph.from_networkx(G, name=f"gnp{n}_{s}")
        out.append(g)
    return out


def medium_random_graphs(count: int, seed: int, n_min: int = 10, n_max: int = 120):
    """Training-distribution graphs + trees + disconnected unions."""
    rng = np.random.default_rng(seed)
    sampler = GraphSampler(seed)
    out = []
    for i in range(count):
        r = i % 5
        if r < 3:
            out.append(sampler.sample(n_min, n_max))
        elif r == 3:
            out.append(make_graph(dict(family="tree", n=int(rng.integers(n_min, n_max)),
                                       seed=int(rng.integers(1 << 30)))))
        else:
            out.append(disjoint_union(sampler.sample(n_min, n_max // 2),
                                      make_graph(dict(family="tree", n=int(rng.integers(2, 30)),
                                                      seed=int(rng.integers(1 << 30))))))
    return out
