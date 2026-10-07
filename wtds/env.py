"""The TDS construction MDP (Section 4).

State   : static graph + per-vertex features (FEATURES) + global features.
Action  : a vertex v with valid_action[v] = 1  <=>  v ∉ S and gain(v) > 0
          (mask_mode="gain", default). mask_mode="notin" (ablation) allows any v ∉ S.
Step    : S <- S ∪ {v}; every u ∈ N(v) becomes covered. v itself is NOT covered
          by its own selection (total domination, change C1).
Reward  : "unit" r = -1 (weighted: -w(v)/mean(w)), so the return is -|S \\ forced|;
          "shaped" r = -1 + β(Φ(s') - Φ(s)), Φ(s) = -#uncovered(s)/max_deg;
          "paper" r = α(#newly covered in N(v) - #already covered in N(v)).
Done    : every vertex has a selected neighbour.

Forced vertices (supports of leaves, change C6) are selected at reset and are not
counted as steps. Features are maintained incrementally: a step touches only v,
N(v) and N(u) for the newly covered u ∈ N(v).
"""
from __future__ import annotations

import math

import numpy as np

from .graph import Graph
from .reductions import forced_vertices

FEATURES = ("selected", "covered", "forced", "deg_norm", "gain_norm", "gain_scaled",
            "covered_nbr_frac", "selected_nbr_frac", "valid_action", "weight_norm")
NUM_FEATURES = len(FEATURES)
NUM_GLOBAL = 3
F = {name: i for i, name in enumerate(FEATURES)}


def _neighbors_of_set(graph: Graph, vs: np.ndarray) -> np.ndarray:
    """Concatenated neighbour lists of the vertices in `vs` (with repetitions)."""
    if vs.size == 0:
        return np.zeros(0, dtype=np.int64)
    starts, ends = graph.indptr[vs], graph.indptr[vs + 1]
    lens = ends - starts
    offs = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    return graph.indices[np.arange(lens.sum()) + offs]


def compute_features(graph: Graph, selected: np.ndarray, forced: np.ndarray,
                     mask_mode: str = "gain"):
    """Features recomputed from scratch from the selection mask (vectorised, O(|E|)).

    Returns (X float32 (n, 10), global float32 (3,), covered bool (n,)).
    Used by the replay buffer to rebuild states, and as the reference the
    incremental environment is tested against.
    """
    n = graph.n
    src, dst = graph.edge_index
    sel_nbr = np.bincount(dst[selected[src]], minlength=n)
    covered = sel_nbr > 0
    cov_nbr = np.bincount(dst[covered[src]], minlength=n)
    return _assemble(graph, selected, covered, forced, sel_nbr, cov_nbr, mask_mode), \
        _global(graph, selected.sum(), covered.sum()), covered


def _assemble(graph, selected, covered, forced, sel_nbr, cov_nbr, mask_mode, rows=None):
    deg = graph.degrees if rows is None else graph.degrees[rows]
    md = max(graph.max_degree, 1)
    w = graph.w if rows is None else graph.w[rows]
    wmax = graph.w.max()
    gain = deg - cov_nbr
    X = np.empty((len(deg), NUM_FEATURES), dtype=np.float32)
    X[:, 0] = selected
    X[:, 1] = covered
    X[:, 2] = forced
    X[:, 3] = deg / md
    X[:, 4] = gain / deg
    X[:, 5] = gain / md
    X[:, 6] = cov_nbr / deg
    X[:, 7] = sel_nbr / deg
    X[:, 8] = valid_from(selected, gain, mask_mode)
    X[:, 9] = w / wmax
    return X


def valid_from(selected, gain, mask_mode):
    if mask_mode == "gain":
        return (~selected) & (gain > 0)
    if mask_mode == "notin":
        return ~selected
    raise ValueError(mask_mode)


def _global(graph, n_sel, n_cov):
    n = graph.n
    return np.array([n_cov / n, n_sel / n, math.log(n) / 10.0], dtype=np.float32)


class TDSEnv:
    def __init__(self, graph: Graph, reward_mode: str = "weighted", shaped_beta: float = 0.5,
                 paper_alpha: float = 1.0, use_forced: bool = True, mask_mode: str = "gain"):
        graph.require_tds_exists()
        if reward_mode not in ("weighted", "unit", "shaped", "paper"):
            raise ValueError(reward_mode)
        self.graph = graph
        self.reward_mode = reward_mode
        self.beta = shaped_beta
        self.alpha = paper_alpha
        self.use_forced = use_forced
        self.mask_mode = mask_mode
        from .reductions import zero_weight_vertices
        base = forced_vertices(graph) if use_forced else np.zeros(0, np.int64)
        self.forced_list = np.union1d(base, zero_weight_vertices(graph)).astype(np.int64)   # forced + free
        self.mean_w = float(graph.w.mean()) or 1.0      # w̄_G (all-zero weights: everything is free)
        self.reset()

    # -------------------------------------------------------------- core API
    def reset(self):
        g = self.graph
        n = g.n
        self.selected = np.zeros(n, dtype=bool)
        self.covered = np.zeros(n, dtype=bool)
        self.forced = np.zeros(n, dtype=bool)
        self.forced[self.forced_list] = True
        self.sel_nbr = np.zeros(n, dtype=np.int64)
        self.cov_nbr = np.zeros(n, dtype=np.int64)
        self.n_uncovered = n
        self.order: list = []          # non-forced selections, in order
        self.X = _assemble(g, self.selected, self.covered, self.forced, self.sel_nbr,
                           self.cov_nbr, self.mask_mode)
        for v in self.forced_list.tolist():
            self._apply(v)
        return self.observation()

    @property
    def done(self) -> bool:
        return self.n_uncovered == 0

    @property
    def solution(self) -> list:
        return np.flatnonzero(self.selected).tolist()

    @property
    def gain(self) -> np.ndarray:
        return self.graph.degrees - self.cov_nbr

    def valid_mask(self) -> np.ndarray:
        return self.X[:, F["valid_action"]] > 0.5

    def global_features(self) -> np.ndarray:
        return _global(self.graph, self.selected.sum(), self.graph.n - self.n_uncovered)

    def observation(self):
        return self.X, self.global_features()

    def step(self, v: int):
        v = int(v)
        if self.done:
            raise RuntimeError("episode already finished")
        if not self.X[v, F["valid_action"]]:
            raise ValueError(f"illegal action {v} (selected={self.selected[v]}, gain={self.gain[v]})")
        g = self.graph
        nb = g.neighbors(v)
        already = int(self.covered[nb].sum())
        n_unc_before = self.n_uncovered
        self._apply(v)
        self.order.append(v)
        newly = n_unc_before - self.n_uncovered
        if self.reward_mode == "weighted":            # r = -w(v)/w̄_G: return = -W(S \ pre)/w̄_G
            r = -float(g.w[v]) / self.mean_w
        elif self.reward_mode == "unit":              # previous TDS reward: -1 per vertex
            r = -1.0
        elif self.reward_mode == "shaped":
            r = -float(g.w[v]) / self.mean_w + self.beta * newly / max(g.max_degree, 1)
        else:
            r = self.alpha * (newly - already)
        return self.observation(), r, self.done, {"newly_covered": newly}

    # ---------------------------------------------------------- internals
    def _apply(self, v: int):
        g = self.graph
        nb = g.neighbors(v)
        self.selected[v] = True
        self.sel_nbr[nb] += 1                      # nb has no repeats
        newly = nb[~self.covered[nb]]
        self.covered[newly] = True
        self.n_uncovered -= len(newly)
        second = _neighbors_of_set(g, newly)
        np.add.at(self.cov_nbr, second, 1)
        rows = np.unique(np.concatenate([[v], nb, second]))
        self.X[rows] = _assemble(g, self.selected[rows], self.covered[rows], self.forced[rows],
                                 self.sel_nbr[rows], self.cov_nbr[rows], self.mask_mode, rows=rows)

    # ---------------------------------------------------------- utilities
    def set_state(self, S) -> None:
        """Reset, then select S (forced vertices are added first automatically)."""
        self.reset()
        for v in S:
            if not self.selected[int(v)]:
                self._apply(int(v))
                self.order.append(int(v))
