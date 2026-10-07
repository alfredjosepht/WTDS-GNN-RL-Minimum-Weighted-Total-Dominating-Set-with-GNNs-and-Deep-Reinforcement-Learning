import numpy as np
import pytest
import torch

from wtds.agent import DDQNAgent
from wtds.env import TDSEnv
from wtds.generators import GraphSampler
from wtds.graph import Graph
from wtds.model import QNet, collate, graph_argmax
from wtds.replay import GraphPool, ReplayBuffer
from wtds.rollout import rollout
from wtds.checker import is_total_dominating_set

torch.manual_seed(0)
GRAPHS = [GraphSampler(5).sample(10, 60) for _ in range(12)]


def partial_env(g, k, rng):
    env = TDSEnv(g)
    for _ in range(k):
        if env.done:
            break
        env.step(int(rng.choice(np.flatnonzero(env.valid_mask()))))
    return env


@pytest.mark.parametrize("encoder", ["mpnn", "gin", "gat"])
@pytest.mark.parametrize("readout", ["meanmax", "sum"])
def test_batching_equals_single(encoder, readout):
    m = QNet(encoder=encoder, readout=readout).eval()
    rng = np.random.default_rng(0)
    envs = [partial_env(g, 3, rng) for g in GRAPHS]
    items = [(e.graph, *e.observation()) for e in envs]
    with torch.no_grad():
        qb = m(collate(items))
        ptr = collate(items).ptr
        for i, it in enumerate(items):
            qs = m(collate([it]))
            np.testing.assert_allclose(qb[ptr[i]:ptr[i + 1]].numpy(), qs.numpy(), rtol=1e-4, atol=1e-5)


def test_mask_and_permutation_equivariance():
    m = QNet().eval()
    g = GRAPHS[0]
    env = partial_env(g, 2, np.random.default_rng(1))
    X, glob = env.observation()
    with torch.no_grad():
        q = m(collate([(g, X, glob)])).numpy()
    assert np.all(np.isneginf(q[~env.valid_mask()])) and np.all(np.isfinite(q[env.valid_mask()]))
    perm = np.random.default_rng(2).permutation(g.n)        # new label of old vertex v = perm[v]
    inv = np.argsort(perm)
    gp = Graph.from_edges(g.n, perm[g.edges()])
    with torch.no_grad():
        qp = m(collate([(gp, X[inv], glob)])).numpy()
    np.testing.assert_allclose(qp[perm], q, rtol=1e-4, atol=1e-5)


def test_graph_argmax():
    q = torch.tensor([1.0, 3.0, 3.0, -np.inf, 0.5, 2.0, -np.inf, -np.inf])
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
    mx, arg = graph_argmax(q, batch, 3)
    assert mx[:2].tolist() == [3.0, 2.0] and torch.isneginf(mx[2])
    assert arg[:2].tolist() == [1, 5]          # ties -> lowest index


def test_replay_state_matches_env():
    pool = GraphPool()
    buf = ReplayBuffer(100, pool)
    rng = np.random.default_rng(3)
    for g in GRAPHS:
        gid = pool.add(g)
        env = TDSEnv(g)
        while not env.done:
            S = list(env.order)
            X_before, glob_before = (a.copy() for a in env.observation())
            gg, X, glob = buf.state(gid, S, "gain")
            np.testing.assert_allclose(X, X_before, atol=1e-6)
            np.testing.assert_allclose(glob, glob_before, atol=1e-6)
            env.step(int(rng.choice(np.flatnonzero(env.valid_mask()))))


def test_ddqn_update_and_target_masking():
    pool = GraphPool()
    buf = ReplayBuffer(1000, pool, seed=0)
    rng = np.random.default_rng(4)
    for g in GRAPHS:
        gid = pool.add(g)
        env = TDSEnv(g)
        while not env.done:
            S = list(env.order)
            a = int(rng.choice(np.flatnonzero(env.valid_mask())))
            _, r, d, _ = env.step(a)
            buf.add(gid, S, a, r, list(env.order), d, 1.0)
    agent = DDQNAgent({"hidden": 32, "layers": 2}, lr=1e-3)
    cur, nxt, a, r, d, gm = buf.sample(32)
    # all sampled actions are legal in their state
    for (g, X, _), act in zip(cur, a):
        assert X[act, 8] == 1.0
    first = agent.update(cur, nxt, a, r, d, gm)
    assert np.isfinite(first["loss"]) and np.isfinite(first["y_mean"])
    # The target for a terminal transition is just r.
    for _ in range(50):
        st = agent.update(*buf.sample(32))
    assert np.isfinite(st["loss"])


def test_rollout_produces_valid_tds():
    m = QNet().eval()
    res = rollout(m, GRAPHS, "cpu", batch_size=5)
    res_s = rollout(m, GRAPHS, "cpu", mode="sample", temperature=0.5, rng=np.random.default_rng(0))
    res_m = rollout(m, GRAPHS, "cpu", multi_select=4)
    for g, a, b, c in zip(GRAPHS, res, res_s, res_m):
        for r in (a, b, c):
            assert is_total_dominating_set(g.adj, r["solution"])


def _filled_buffer(mask_mode="gain", use_forced=True):
    pool = GraphPool(use_forced=use_forced)
    buf = ReplayBuffer(5000, pool, seed=0)
    rng = np.random.default_rng(7)
    for g in GRAPHS + [GraphSampler(9).sample(20, 80) for _ in range(10)]:
        gid = pool.add(g)
        env = TDSEnv(g, mask_mode=mask_mode, use_forced=use_forced)
        while not env.done:
            S = list(env.order)
            a = int(rng.choice(np.flatnonzero(env.valid_mask())))
            _, r, d, _ = env.step(a)
            buf.add(gid, S, a, r, list(env.order), d, 1.0)
    return pool, buf


@pytest.mark.parametrize("mask_mode,use_forced", [("gain", True), ("notin", False)])
def test_gpu_batch_equals_cpu_collate(mask_mode, use_forced):
    from wtds.gpu_batch import GPUGraphCache, build_batches
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pool, buf = _filled_buffer(mask_mode, use_forced)
    trs = buf.sample_transitions(64)
    cache = GPUGraphCache(pool, dev)
    gc, gn, a, r, d, gm = build_batches(cache, trs, mask_mode)
    for which, gb in ((1, gc), (4, gn)):
        items = [buf.state(tr[0], tr[which], mask_mode) for tr in trs]
        cb = collate(items)
        np.testing.assert_allclose(gb.x.cpu().numpy(), cb.x.numpy(), atol=1e-6)
        np.testing.assert_allclose(gb.glob.cpu().numpy(), cb.glob.numpy(), atol=1e-6)
        assert torch.equal(gb.edge_index.cpu(), cb.edge_index) and torch.equal(gb.batch.cpu(), cb.batch)
        assert torch.equal(gb.valid.cpu(), cb.valid) and torch.equal(gb.ptr.cpu(), cb.ptr)
        np.testing.assert_allclose(gb.deg.cpu().numpy(), cb.deg.numpy())


def test_fused_and_gpu_update_equal_reference():
    import copy
    from wtds.gpu_batch import GPUGraphCache, build_batches
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pool, buf = _filled_buffer()
    trs = buf.sample_transitions(64)
    torch.manual_seed(0)
    ref = DDQNAgent({"hidden": 32, "layers": 3}, lr=1e-3, device=dev)
    fused_gpu = copy.deepcopy(ref)
    cur = [buf.state(tr[0], tr[1], "gain") for tr in trs]
    nxt = [buf.state(tr[0], tr[4], "gain") for tr in trs]
    acts = np.array([tr[2] for tr in trs]); rew = np.array([tr[3] for tr in trs], np.float32)
    dn = np.array([tr[5] for tr in trs]); gm = np.array([tr[6] for tr in trs], np.float32)
    s1 = ref.update_batches(collate(cur, dev), collate(nxt, dev), acts, rew, dn, gm, fused=False)
    s2 = fused_gpu.update_batches(*build_batches(GPUGraphCache(pool, dev), trs), fused=True)
    assert s1["loss"] == pytest.approx(s2["loss"], rel=1e-4, abs=1e-6)
    for p1, p2 in zip(ref.online.parameters(), fused_gpu.online.parameters()):
        torch.testing.assert_close(p1, p2, rtol=1e-4, atol=1e-6)
