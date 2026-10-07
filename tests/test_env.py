import networkx as nx
import numpy as np
import pytest

from wtds import Graph
from wtds.checker import is_total_dominating_set
from wtds.env import F, TDSEnv, compute_features
from wtds.generators import make_graph
from wtds.reductions import forced_vertices

from helpers import medium_random_graphs, small_random_graphs

POOL = small_random_graphs(150, seed=21, n_max=14) + medium_random_graphs(150, seed=22)


def run_random_episode(env, rng, check_every_step=True):
    total, steps = 0.0, 0
    while not env.done:
        valid = np.flatnonzero(env.valid_mask())
        assert valid.size > 0, "no legal action before termination"
        # v ∉ S for every legal action, and gain(v) > 0 under the default mask
        assert not env.selected[valid].any()
        if env.mask_mode == "gain":
            assert (env.gain[valid] > 0).all()
        v = int(rng.choice(valid))
        _, r, done, _ = env.step(v)
        total += r
        steps += 1
        if check_every_step:
            assert_matches_scratch(env)
    return total, steps


def assert_matches_scratch(env):
    X, glob, cov = compute_features(env.graph, env.selected, env.forced, env.mask_mode)
    np.testing.assert_array_equal(cov, env.covered)
    np.testing.assert_allclose(env.X, X, rtol=0, atol=1e-6)
    np.testing.assert_allclose(env.global_features(), glob, atol=1e-6)


def test_incremental_equals_scratch_and_valid_tds():
    rng = np.random.default_rng(0)
    for g in POOL:
        env = TDSEnv(g)
        assert_matches_scratch(env)
        total, steps = run_random_episode(env, rng)
        S = env.solution
        assert is_total_dominating_set(g.adj, S)
        nf = len(forced_vertices(g))
        assert steps == len(S) - nf
        assert total == pytest.approx(-(len(S) - nf))      # unit reward: return = -|S \ forced|


def test_selection_covers_only_open_neighbourhood():
    g = Graph.from_networkx(nx.cycle_graph(6))
    env = TDSEnv(g)
    env.step(0)
    assert not env.covered[0] and env.covered[1] and env.covered[5]
    assert env.covered.sum() == 2
    assert env.selected[0] and env.X[0, F["selected"]] == 1 and env.X[0, F["covered"]] == 0


def test_gain_zero_never_legal():
    # K_{1,3} centre 0 is forced (supports leaves). After reset leaves 1..3 are covered,
    # centre is uncovered. Leaves have gain 1 (their nbr 0 uncovered); 0 has gain 0.
    g = Graph.from_networkx(nx.star_graph(3))
    env = TDSEnv(g)
    assert env.selected[0] and env.forced[0]
    assert env.gain[0] == 0 and not env.valid_mask()[0]
    assert env.valid_mask()[1:].all()
    env.step(2)
    assert env.done and sorted(env.solution) == [0, 2]
    with pytest.raises(RuntimeError):
        env.step(1)


def test_illegal_action_raises():
    g = Graph.from_networkx(nx.cycle_graph(8))
    env = TDSEnv(g)
    env.step(0)
    with pytest.raises(ValueError):
        env.step(0)                     # already selected
    env.step(2); env.step(4)            # covers 1,3,5,7 and ... check gain-0 vertices are rejected
    for v in np.flatnonzero((env.gain == 0) & ~env.selected):
        with pytest.raises(ValueError):
            env.step(int(v))


def test_forced_only_zero_step_episode():
    # P4 = 0-1-2-3: forced {1,2} cover everything; zero steps.
    env = TDSEnv(Graph.from_networkx(nx.path_graph(4)))
    assert env.done and env.solution == [1, 2] and not env.valid_mask().any()
    # Two disjoint edges: every vertex forced.
    env = TDSEnv(Graph.from_edges(4, [(0, 1), (2, 3)]))
    assert env.done and env.solution == [0, 1, 2, 3]


def test_forced_flag_and_toggle():
    g = make_graph(dict(family="ba", n=60, m=1, seed=3))   # tree, many leaves
    env = TDSEnv(g)
    F_ = set(forced_vertices(g).tolist())
    assert set(np.flatnonzero(env.forced).tolist()) == F_ and len(F_) > 5
    assert set(np.flatnonzero(env.selected).tolist()) == F_
    env2 = TDSEnv(g, use_forced=False)
    assert not env2.selected.any() and not env2.forced.any()
    total, steps = run_random_episode(env2, np.random.default_rng(0), check_every_step=False)
    assert is_total_dominating_set(g.adj, env2.solution)


def test_reward_modes():
    rng = np.random.default_rng(1)
    for g in POOL[::10]:
        for mode in ("shaped", "paper"):
            env = TDSEnv(g, reward_mode=mode)
            while not env.done:
                valid = np.flatnonzero(env.valid_mask())
                v = int(rng.choice(valid))
                nb = g.neighbors(v)
                already = int(env.covered[nb].sum())
                unc_before = env.n_uncovered
                _, r, _, info = env.step(v)
                newly = unc_before - env.n_uncovered
                if mode == "shaped":
                    # potential-based: Φ(s) = -#uncovered/max_deg
                    phi_diff = (-env.n_uncovered + unc_before) / g.max_degree
                    assert r == pytest.approx(-1 + 0.5 * phi_diff)
                else:
                    assert r == pytest.approx(newly - already)
            assert is_total_dominating_set(g.adj, env.solution)


def test_weighted_unit_reward():
    g = make_graph(dict(family="er", n=40, avg_deg=5, seed=2, weights=True))
    env = TDSEnv(g)
    rng = np.random.default_rng(0)
    total, _ = run_random_episode(env, rng, check_every_step=False)
    S_nf = [v for v in env.solution if not env.forced[v]]
    assert total == pytest.approx(-g.w[S_nf].sum() / g.w.mean())
    assert env.X[:, F["weight_norm"]].max() == pytest.approx(1.0)


def test_notin_mask_ablation_terminates_valid():
    rng = np.random.default_rng(2)
    for g in POOL[::15]:
        env = TDSEnv(g, mask_mode="notin")
        run_random_episode(env, rng)
        assert is_total_dominating_set(g.adj, env.solution)


def test_set_state_matches_replay():
    rng = np.random.default_rng(3)
    for g in POOL[::7]:
        env = TDSEnv(g)
        run_random_episode(env, rng, check_every_step=False)
        S = env.order[: len(env.order) // 2]
        env2 = TDSEnv(g)
        env2.set_state(S)
        assert_matches_scratch(env2)


def test_mask_preserves_an_optimum():
    """Section 4.3: inserting forced vertices then any minimum TDS in any order
    only ever uses legal (gain > 0) actions."""
    from wtds.baselines import ilp
    rng = np.random.default_rng(4)
    for g in small_random_graphs(80, seed=23, n_max=12):
        S_opt = ilp(g, time_limit=30).vertices
        env = TDSEnv(g)
        rest = [v for v in S_opt if not env.selected[v]]
        for _ in range(3):
            env.reset()
            for v in rng.permutation(rest).tolist():
                assert env.valid_mask()[v], (g.name, v)
                env.step(v)
            assert env.done and len(env.solution) == len(S_opt)
