"""Known total domination numbers, checked against the exact ILP; greedy checked
for validity, lower bound γ_t and the set-cover guarantee H(Δ)·γ_t."""
import networkx as nx
import pytest

from wtds import Graph
from wtds.baselines import greedy, greedy_rr, ilp
from wtds.generators import grid


def gt_path_cycle(n):
    return n // 2 + -(-n // 4) - n // 4


def gt_grid(n):
    m, r = divmod(n, 4)
    return (2 * m + 1) * (2 * m + r)


def harmonic(k):
    return sum(1.0 / i for i in range(1, k + 1))


def check(g: Graph, expected: int, time_limit=60):
    r = ilp(g, time_limit=time_limit)
    assert r.is_valid and r.info["optimal"], r.info
    assert r.size == expected, (g.name, r.size, expected)
    for res in (greedy(g), greedy_rr(g)):
        assert res.is_valid
        assert expected <= res.size <= harmonic(g.max_degree) * expected + 1e-9, (g.name, res.method, res.size)


def test_formula_sanity():
    # Values quoted in the specification.
    assert gt_grid(3) == 3 and gt_grid(10) == 30
    assert [gt_path_cycle(n) for n in range(3, 10)] == [2, 2, 3, 4, 4, 4, 5]


@pytest.mark.parametrize("n", range(3, 30))
def test_paths(n):
    check(Graph.from_networkx(nx.path_graph(n), name=f"P{n}"), gt_path_cycle(n))


@pytest.mark.parametrize("n", range(3, 30))
def test_cycles(n):
    check(Graph.from_networkx(nx.cycle_graph(n), name=f"C{n}"), gt_path_cycle(n))


@pytest.mark.parametrize("n", range(2, 12))
def test_complete(n):
    check(Graph.from_networkx(nx.complete_graph(n), name=f"K{n}"), 2)


@pytest.mark.parametrize("m", range(1, 12))
def test_star(m):
    check(Graph.from_networkx(nx.star_graph(m), name=f"K1,{m}"), 2)


@pytest.mark.parametrize("a,b", [(a, b) for a in range(1, 7) for b in range(a, 7)])
def test_complete_bipartite(a, b):
    check(Graph.from_networkx(nx.complete_bipartite_graph(a, b), name=f"K{a},{b}"), 2)


def test_petersen():
    check(Graph.from_networkx(nx.petersen_graph(), name="petersen"), 4)


@pytest.mark.parametrize("n", range(2, 13))
def test_square_grid(n):
    check(grid(n, n), gt_grid(n), time_limit=120)


def test_greedy_tie_break():
    # P4 = 0-1-2-3: forced = {1, 2} (supports of leaves 0 and 3) -> exactly {1,2}.
    assert greedy(Graph.from_networkx(nx.path_graph(4))).vertices == [1, 2]
    # C6 no leaves: all gains 2, all degrees 2 -> pick lowest index first.
    r = greedy(Graph.from_networkx(nx.cycle_graph(6)))
    assert r.info["order"][0] == 0
