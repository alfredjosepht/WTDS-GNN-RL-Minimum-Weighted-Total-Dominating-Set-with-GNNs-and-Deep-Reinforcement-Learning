"""Weighted total domination: hand examples, exact solvers, solver validity, invariants."""
import numpy as np
import pytest

from wtds import Graph, NoTotalDominatingSetError
from wtds.baselines import (brute_force_weighted, greedy, greedy_rr, greedy_rr_ls, ilp,
                            unweighted_greedy_rr)
from wtds.checker import is_total_dominating_set, verify
from wtds.env import TDSEnv
from wtds.generators import make_graph
from wtds.graph import InvalidWeightError
from wtds.postprocess import is_minimal, local_search, redundancy_removal

from helpers import disjoint_union


def labelled(edges, weights, one_based=True):
    """Build a graph from labelled edges; returns (graph, label->index)."""
    labels = sorted({x for e in edges for x in e} | set(weights))
    idx = {l: i for i, l in enumerate(labels)}
    w = [float(weights.get(l, 1)) for l in labels]
    return Graph.from_edges(len(labels), [(idx[a], idx[b]) for a, b in edges], weights=w), idx


GRID3 = [(1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9), (1, 4), (4, 7), (2, 5), (5, 8), (3, 6), (6, 9)]
HAND = [
    ("star", [(0, i) for i in range(1, 6)], {0: 10, 1: 3, 2: 1, 3: 4, 4: 1, 5: 5}, 11, [{0, 2}, {0, 4}]),
    ("triangle", [(1, 2), (2, 3), (3, 1)], {1: 5, 2: 1, 3: 2}, 3, [{2, 3}]),
    ("square", [(1, 2), (2, 3), (3, 4), (4, 1)], {1: 4, 2: 1, 3: 1, 4: 4}, 2, [{2, 3}]),
    ("path", [(1, 2), (2, 3), (3, 4)], {1: 1, 2: 7, 3: 2, 4: 1}, 9, [{2, 3}]),
    ("two_triangles_unit", [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (3, 4)], {}, 2, [{3, 4}]),
    ("two_triangles_bridge", [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (3, 4)], {3: 10, 4: 10}, 4,
     [{1, 2, 5, 6}]),
    ("hexagon", [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)], {1: 1, 2: 1, 3: 5, 4: 5, 5: 1, 6: 1}, 4,
     [{1, 2, 5, 6}]),
    ("grid3_centre", GRID3, {5: 20}, 4, [{1, 2, 8, 9}, {1, 4, 6, 9}, {2, 3, 7, 8}, {3, 4, 6, 7}]),
]


@pytest.mark.parametrize("name,edges,weights,opt,opt_sets", HAND, ids=[h[0] for h in HAND])
def test_hand_examples(name, edges, weights, opt, opt_sets):
    g, idx = labelled(edges, weights)
    inv = {i: l for l, i in idx.items()}
    best, S, all_sets = brute_force_weighted(g)
    assert best == opt
    assert {frozenset(inv[v] for v in s) for s in all_sets} == {frozenset(s) for s in opt_sets}
    r = ilp(g, time_limit=10)
    assert r.info["optimal"] and r.weight == opt and is_total_dominating_set(g.adj, r.vertices)
    for res in (greedy(g), greedy_rr(g), greedy_rr_ls(g, ls_time=0.1), unweighted_greedy_rr(g)):
        assert res.is_valid and res.weight >= opt


def test_min_weight_can_need_more_vertices():
    g, _ = labelled(HAND[5][1], HAND[5][2])
    best_w, S_w, _ = brute_force_weighted(g)
    gu = Graph.from_edges(g.n, g.edges())
    best_size, S_u, _ = brute_force_weighted(gu)
    assert len(S_w) == 4 and best_size == 2 and best_w == 4 and g.w[S_u].sum() == 20


def random_weighted(count, seed, n_max=10):
    rng = np.random.default_rng(seed)
    out = []
    dists = ["uniform_int", "uniform_small", "degree_correlated", "unit"]
    while len(out) < count:
        n = int(rng.integers(3, n_max + 1))
        kind = int(rng.integers(3))
        s = int(rng.integers(1 << 30))
        dist = dists[len(out) % 4]
        if kind == 0:
            g = make_graph(dict(family="er", n=n, avg_deg=float(rng.uniform(1, n - 1)), seed=s, weights=dist))
        elif kind == 1:
            g = make_graph(dict(family="tree", n=n, seed=s, weights=dist))
        elif n >= 5:
            a = int(rng.integers(2, n - 2))
            g1 = make_graph(dict(family="er", n=a, avg_deg=float(rng.uniform(1, a)), seed=s, weights=dist))
            g2 = make_graph(dict(family="tree", n=n - a, seed=s + 1, weights=dist))
            u = disjoint_union(g1, g2)
            g = Graph.from_edges(u.n, u.edges(), weights=np.concatenate([g1.w, g2.w]))
        else:
            continue
        if len(out) % 7 == 0:                       # some zero (free) weights
            w = g.w.copy()
            w[rng.integers(g.n)] = 0.0
            g = Graph.from_edges(g.n, g.edges(), weights=w)
        out.append(g)
    return out


SMALL = random_weighted(200, seed=7)


@pytest.mark.parametrize("i", range(200))
def test_weighted_ilp_equals_bruteforce(i):
    g = SMALL[i]
    best, _, _ = brute_force_weighted(g)
    r = ilp(g, time_limit=20)
    assert r.info["optimal"] and abs(r.weight - best) < 1e-6
    assert abs(r.weight - float(g.w[r.vertices].sum())) < 1e-9


def test_all_solvers_valid_on_300_weighted_graphs(random_checkpoint):
    from wtds.solve import solve
    pool = random_weighted(150, seed=11, n_max=12) + random_weighted(150, seed=12, n_max=60)
    assert sum((g.w == 0).any() for g in pool) > 20 and sum((g.degrees == 1).any() for g in pool) > 50
    for i, g in enumerate(pool):
        results = [greedy(g), greedy_rr(g), unweighted_greedy_rr(g), greedy_rr_ls(g, ls_time=0.01)]
        results.append(solve(g, None, method="rl", checkpoint=random_checkpoint, device="cpu", ls_time=0))
        if i % 5 == 0:
            results.append(solve(g, None, method="rl", checkpoint=random_checkpoint, device="cpu", ls_time=0.01))
        for r in results:
            assert r.is_valid and is_total_dominating_set(g.adj, r.vertices), (r.method, i)
            assert abs(r.weight - float(g.w[r.vertices].sum())) < 1e-9


def test_isolated_and_bad_weights_raise():
    g = Graph.from_edges(3, [(0, 1)], weights=[1, 1, 1])
    with pytest.raises(NoTotalDominatingSetError):
        greedy(g)
    with pytest.raises(InvalidWeightError, match="vertex 1"):
        Graph.from_edges(2, [(0, 1)], weights=[1, -2])
    with pytest.raises(InvalidWeightError):
        Graph.from_edges(2, [(0, 1)], weights=[float("nan"), 1])


def test_zero_weight_preselected():
    g, idx = labelled([(1, 2), (2, 3), (3, 4), (4, 1)], {1: 0, 2: 5, 3: 5, 4: 5})
    env = TDSEnv(g)
    assert env.selected[idx[1]]
    r = greedy_rr(g)
    assert r.is_valid and idx[1] in r.vertices


def test_rr_and_ls_never_break_or_worsen():
    for g in random_weighted(120, seed=21, n_max=40):
        S0 = greedy(g).vertices
        S = redundancy_removal(g, S0)
        assert verify(g, S) and g.w[S].sum() <= g.w[S0].sum() + 1e-9 and is_minimal(g, S)
        S2 = local_search(g, S, time_limit=0.02, debug=True)
        assert verify(g, S2) and g.w[S2].sum() <= g.w[S].sum() + 1e-9


def test_env_reward_sum_is_minus_weight_over_mean():
    rng = np.random.default_rng(3)
    for g in random_weighted(60, seed=31, n_max=40):
        env = TDSEnv(g)                                  # default reward_mode = "weighted"
        pre = set(env.solution)
        total = 0.0
        while not env.done:
            _, r, _, _ = env.step(int(rng.choice(np.flatnonzero(env.valid_mask()))))
            total += r
        added = [v for v in env.solution if v not in pre]
        wbar = float(g.w.mean()) or 1.0
        assert total == pytest.approx(-g.w[added].sum() / wbar)
        assert is_total_dominating_set(g.adj, env.solution)


def test_weight_feature_carries_weights():
    g = make_graph(dict(family="er", n=30, avg_deg=5, seed=1, weights="uniform_int"))
    env = TDSEnv(g)
    np.testing.assert_allclose(env.X[:, 9], g.w / g.w.max(), rtol=1e-6)


def test_weighted_formats(tmp_path):
    from wtds.io import read_graph
    (tmp_path / "e.txt").write_text("a b\nb c\nc a\n")
    (tmp_path / "w.txt").write_text("a 5\nb 1\n")
    g = read_graph(tmp_path / "e.txt", tmp_path / "w.txt")
    assert list(g.w) == [5, 1, 1] and any("default" in m for m in g.meta["weight_warnings"])
    (tmp_path / "c.txt").write_text("v x 2\nv y 3\nv z 4\ne x y\ne y z\n")
    g = read_graph(tmp_path / "c.txt")
    assert list(g.w) == [2, 3, 4] and g.num_edges == 2
    (tmp_path / "d.col").write_text("p edge 3 2\nn 1 7\nn 2 8\nn 3 9\ne 1 2\ne 2 3\n")
    g = read_graph(tmp_path / "d.col")
    assert list(g.w) == [7, 8, 9]
    g = read_graph(tmp_path / "e.txt")
    assert list(g.w) == [1, 1, 1] and g.meta["weight_warnings"]


def test_explicit_nan_weight_in_file_rejected(tmp_path):
    from wtds.io import read_graph
    (tmp_path / "e.txt").write_text("a b\nb c\n")
    (tmp_path / "w.txt").write_text("a nan\n")
    with pytest.raises(InvalidWeightError):
        read_graph(tmp_path / "e.txt", tmp_path / "w.txt")


def test_best_of_k_never_worse_than_greedy_rollout(random_checkpoint):
    from wtds.solve import solve_rl
    for i, g in enumerate(random_weighted(40, seed=41, n_max=40)):
        r1 = solve_rl(g, random_checkpoint, rr=True, ls_time=0, samples=1, device="cpu", seed=i)
        rk = solve_rl(g, random_checkpoint, rr=True, ls_time=0, samples=8, temperature=0.3, device="cpu", seed=i)
        assert rk.is_valid and is_total_dominating_set(g.adj, rk.vertices)
        assert rk.weight <= r1.weight + 1e-9


def test_rl_exact_finds_proven_minimum(random_checkpoint):
    """GNN+RL + exact certification returns the brute-force minimum and says it is proven."""
    from wtds.solve import solve
    for i, g in enumerate(SMALL[:60]):
        best, _, _ = brute_force_weighted(g)
        r = solve(g, None, method="rl_exact", checkpoint=random_checkpoint, device="cpu", ls_time=0.05,
                  ilp_time=10, seed=i)
        assert r.is_valid and r.info["optimal"] and abs(r.weight - best) < 1e-6
        assert r.weight <= r.info["rl_weight"] + 1e-9
    for name, edges, weights, opt, _ in HAND:
        g, _ = labelled(edges, weights)
        r = solve(g, None, method="rl_exact", checkpoint=random_checkpoint, device="cpu", ls_time=0.05, ilp_time=10)
        assert r.info["optimal"] and r.weight == opt
