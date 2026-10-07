"""Compact replay buffer (Section 7).

A transition is (graph_id, S_t, action, R, S_{t+n}, done), with S stored as a
sorted int32 array of the non-forced selected vertices. Features are rebuilt from
(graph, S) when sampling, so 100k transitions take a few MB. Graphs are kept in
a reference-counted pool and dropped when no transition refers to them.
"""
from __future__ import annotations

import numpy as np

from .env import compute_features
from .graph import Graph
from .reductions import forced_mask


class GraphPool:
    def __init__(self, use_forced: bool = True):
        self.use_forced = use_forced
        self.graphs: dict = {}
        self.forced: dict = {}
        self.refs: dict = {}
        self._next = 0

    def add(self, g: Graph) -> int:
        gid = self._next
        self._next += 1
        self.graphs[gid] = g
        self.forced[gid] = forced_mask(g) if self.use_forced else np.zeros(g.n, dtype=bool)
        self.refs[gid] = 0
        return gid

    def incref(self, gid):
        self.refs[gid] += 1

    def decref(self, gid):
        self.refs[gid] -= 1

    def collect(self, keep=()):
        for gid in [g for g, r in self.refs.items() if r <= 0 and g not in keep]:
            del self.graphs[gid], self.forced[gid], self.refs[gid]


class ReplayBuffer:
    def __init__(self, capacity: int, pool: GraphPool, seed: int = 0):
        self.capacity = capacity
        self.pool = pool
        self.data = [None] * capacity
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, gid, S_t, action, reward, S_next, done, gamma_n):
        old = self.data[self.pos]
        if old is not None:
            self.pool.decref(old[0])
        self.pool.incref(gid)
        self.data[self.pos] = (gid, np.asarray(sorted(S_t), dtype=np.int32), int(action),
                               float(reward), np.asarray(sorted(S_next), dtype=np.int32),
                               bool(done), float(gamma_n))
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def state(self, gid, S, mask_mode):
        g = self.pool.graphs[gid]
        forced = self.pool.forced[gid]
        sel = forced.copy()
        sel[S] = True
        X, glob, _ = compute_features(g, sel, forced, mask_mode)
        return g, X, glob

    def sample_transitions(self, batch_size: int) -> list:
        """Raw transitions (same RNG draws as `sample`), for wtds.gpu_batch."""
        idx = self.rng.integers(0, self.size, size=batch_size)
        return [self.data[i] for i in idx]

    def sample(self, batch_size: int, mask_mode: str = "gain"):
        cur, nxt, actions, rewards, dones, gammas = [], [], [], [], [], []
        for gid, S, a, r, S2, d, gn in self.sample_transitions(batch_size):
            cur.append(self.state(gid, S, mask_mode))
            nxt.append(self.state(gid, S2, mask_mode))
            actions.append(a)
            rewards.append(r)
            dones.append(d)
            gammas.append(gn)
        return (cur, nxt, np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=bool), np.array(gammas, dtype=np.float32))
