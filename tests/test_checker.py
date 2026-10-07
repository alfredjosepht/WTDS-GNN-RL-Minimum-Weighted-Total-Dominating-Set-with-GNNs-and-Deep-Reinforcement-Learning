import networkx as nx
import numpy as np
import pytest

from wtds import Graph, NoTotalDominatingSetError
from wtds.baselines import brute_force, greedy, greedy_rr, ilp
from wtds.checker import is_tds_fast, is_total_dominating_set, verify
from wtds.reductions import forced_vertices

from helpers import small_random_graphs


def test_open_neighbourhood_semantics():
    # Star K_{1,3}: centre 0. {0} dominates (closed nbhd) but is NOT a TDS:
    # vertex 0 has no neighbour in S.
    g = Graph.from_networkx(nx.star_graph(3))
    assert not is_total_dominating_set(g.adj, [0])
    assert is_total_dominating_set(g.adj, [0, 1])
    # K2: both endpoints are needed.
    k2 = Graph.from_edges(2, [(0, 1)])
    assert not is_total_dominating_set(k2.adj, [0])
    assert is_total_dominating_set(k2.adj, [0, 1])


def test_spec_reference_checker_signature():
    adj = [[1], [0, 2], [1]]          # path P3
    assert is_total_dominating_set(adj, {0, 1}) and is_total_dominating_set(adj, [1, 2])
    assert not is_total_dominating_set(adj, [1])


def test_self_loops_and_multiedges_dropped():
    g = Graph.from_edges(3, [(0, 0), (0, 1), (1, 0), (1, 2), (2, 2)])
    assert g.num_edges == 2 and g.adj == [[1], [0, 2], [1]]
    assert (g.edge_index[0] != g.edge_index[1]).all()


def test_fast_checker_matches_reference():
    rng = np.random.default_rng(0)
    for g in small_random_graphs(300, seed=1, n_max=14):
        for _ in range(5):
            S = np.flatnonzero(rng.random(g.n) < rng.uniform(0.2, 0.9)).tolist()
            assert is_tds_fast(g, S) == is_total_dominating_set(g.adj, S)
    g = Graph.from_edges(3, [(0, 1), (1, 2)])
    assert not is_tds_fast(g, [5])     # out of range


def test_isolated_vertex_raises():
    g = Graph.from_edges(4, [(0, 1), (1, 2)])   # vertex 3 isolated
    for fn in (forced_vertices, greedy, greedy_rr, ilp, brute_force):
        with pytest.raises(NoTotalDominatingSetError):
            fn(g)
    with pytest.raises(NoTotalDominatingSetError):
        forced_vertices(Graph.from_edges(1, []))


def _all_min_tds(g):
    from itertools import combinations
    k, _ = brute_force(g)
    return [set(S) for S in combinations(range(g.n), k) if is_total_dominating_set(g.adj, S)]


def test_forced_rule_in_every_minimum_tds():
    count = 0
    for g in small_random_graphs(150, seed=3, n_max=11):
        F = set(forced_vertices(g).tolist())
        leaves = np.flatnonzero(g.degrees == 1)
        assert F == {int(g.neighbors(u)[0]) for u in leaves}
        if F:
            count += 1
            for S in _all_min_tds(g):
                assert F <= S
    assert count > 30    # the pool really exercised the rule


def test_verify_used_on_results():
    g = Graph.from_networkx(nx.petersen_graph())
    r = greedy(g)
    assert r.is_valid and verify(g, r.vertices)
