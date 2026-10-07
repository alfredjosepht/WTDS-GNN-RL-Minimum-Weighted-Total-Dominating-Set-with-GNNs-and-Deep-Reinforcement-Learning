import numpy as np
import pytest

from wtds.baselines import greedy_construct
from wtds.checker import InvalidSolutionError, verify
from wtds.postprocess import is_minimal, redundancy_removal
from wtds.reductions import forced_vertices

from helpers import medium_random_graphs, small_random_graphs


def test_rr_on_full_vertex_set_is_feasible_and_minimal():
    for g in medium_random_graphs(150, seed=5):
        S = redundancy_removal(g, range(g.n))
        assert verify(g, S) and is_minimal(g, S)
        assert set(forced_vertices(g).tolist()) <= set(S)


def test_rr_on_random_supersets():
    rng = np.random.default_rng(0)
    for g in small_random_graphs(300, seed=6, n_max=30):
        base, _ = greedy_construct(g)
        extra = np.flatnonzero(rng.random(g.n) < 0.3).tolist()
        S0 = list(dict.fromkeys(base + extra))
        scores = {v: float(rng.normal()) for v in S0}
        S = redundancy_removal(g, S0, scores=scores)
        assert verify(g, S) and is_minimal(g, S) and set(S) <= set(S0)


def test_rr_respects_protected():
    for g in medium_random_graphs(30, seed=7):
        prot = list(range(0, g.n, 3))
        S = redundancy_removal(g, range(g.n), protected=prot)
        assert set(prot) <= set(S) and verify(g, S)


def test_rr_rejects_infeasible_input():
    g = medium_random_graphs(1, seed=8)[0]
    with pytest.raises(InvalidSolutionError):
        redundancy_removal(g, [0])


def test_local_search_feasible_never_worse_debug():
    from wtds.postprocess import local_search
    from wtds.baselines import greedy_rr
    improved = 0
    for g in medium_random_graphs(60, seed=9):
        S0 = greedy_rr(g).vertices
        S = local_search(g, S0, time_limit=0.05, seed=1, debug=True)
        assert verify(g, S) and len(S) <= len(S0)
        assert set(forced_vertices(g).tolist()) <= set(S)
        improved += len(S) < len(S0)
    assert improved > 0


def test_local_search_weighted():
    from wtds.generators import make_graph
    from wtds.postprocess import local_search
    from wtds.baselines import greedy_rr
    for s in range(10):
        g = make_graph(dict(family="er", n=60, avg_deg=5, seed=s, weights=True))
        S0 = greedy_rr(g).vertices
        S = local_search(g, S0, time_limit=0.05, debug=True)
        assert verify(g, S) and g.w[S].sum() <= g.w[S0].sum() + 1e-9
