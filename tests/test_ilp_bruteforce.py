import pytest

from wtds.baselines import brute_force, ilp

from helpers import small_random_graphs

GRAPHS = small_random_graphs(200, seed=2024, n_min=3, n_max=10)


def test_pool_is_diverse():
    assert len(GRAPHS) == 200
    assert any(g.max_degree == g.n - 1 for g in GRAPHS)
    assert any((g.degrees == 1).any() for g in GRAPHS)


@pytest.mark.parametrize("idx", range(200))
def test_cpsat_equals_bruteforce(idx):
    g = GRAPHS[idx]
    k, S = brute_force(g)
    r = ilp(g, time_limit=30, backend="cpsat")
    assert r.is_valid and r.info["optimal"]
    assert r.size == k, (g.name, r.size, k)


@pytest.mark.parametrize("idx", range(0, 200, 5))
def test_highs_equals_bruteforce(idx):
    g = GRAPHS[idx]
    k, _ = brute_force(g)
    r = ilp(g, time_limit=30, backend="highs")
    assert r.is_valid and r.info["optimal"] and r.size == k
