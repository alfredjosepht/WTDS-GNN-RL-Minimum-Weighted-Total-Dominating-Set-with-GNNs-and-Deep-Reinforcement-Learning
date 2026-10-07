"""Exact MTDS via integer programming (Section 1):

    min  Σ_v w_v x_v     s.t.  Σ_{u ∈ N(v)} x_u >= 1  ∀v,   x ∈ {0,1}^n

Backends: "cpsat" (OR-Tools, default), "highs" (SciPy milp), "gurobi" (if gurobipy
is installed and licensed), "auto" = gurobi if usable else cpsat.
The result records whether optimality was proven, plus the best bound and gap.
"""
from __future__ import annotations

import time

import numpy as np

from ..graph import Graph
from ..reductions import forced_vertices
from ..result import SolveResult


def _int_weights(graph: Graph):
    """CP-SAT needs integer objective coefficients; scale non-integer weights."""
    w = graph.w
    if np.allclose(w, np.round(w)):
        return np.round(w).astype(np.int64), 1.0
    scale = 1000.0
    return np.round(w * scale).astype(np.int64), scale


def _cpsat(graph, time_limit, threads, seed, hint):
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{v}") for v in range(graph.n)]
    for v in range(graph.n):
        m.AddBoolOr([x[u] for u in graph.adj[v]])
    for f in forced_vertices(graph).tolist():
        m.Add(x[f] == 1)                      # valid: forced vertices are in every TDS
    wi, scale = _int_weights(graph)
    m.Minimize(sum(int(wi[v]) * x[v] for v in range(graph.n)))
    if hint is not None:
        hs = set(hint)
        for v in range(graph.n):
            m.AddHint(x[v], 1 if v in hs else 0)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(time_limit)
    s.parameters.num_workers = int(threads)
    s.parameters.random_seed = int(seed)
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, False, float("nan"), s.StatusName(st)
    S = [v for v in range(graph.n) if s.Value(x[v])]
    return S, st == cp_model.OPTIMAL, s.BestObjectiveBound() / scale, s.StatusName(st)


def _highs(graph, time_limit, threads, seed, hint):
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix
    n = graph.n
    A = csr_matrix((np.ones(len(graph.indices)), graph.indices, graph.indptr), shape=(n, n))
    lb = np.zeros(n)
    lb[forced_vertices(graph)] = 1
    res = milp(c=graph.w, constraints=LinearConstraint(A, lb=1, ub=np.inf),
               integrality=np.ones(n), bounds=Bounds(lb, 1),
               options={"time_limit": float(time_limit), "disp": False})
    if res.x is None:
        return None, False, float("nan"), res.message
    S = np.flatnonzero(res.x > 0.5).tolist()
    bound = getattr(res, "mip_dual_bound", float("nan"))
    return S, res.status == 0, float(bound), res.message


def _gurobi(graph, time_limit, threads, seed, hint):
    import gurobipy as gp
    from gurobipy import GRB
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as m:
            m.Params.TimeLimit = float(time_limit)
            m.Params.Threads = int(threads)
            m.Params.Seed = int(seed)
            x = m.addVars(graph.n, vtype=GRB.BINARY)
            for v in range(graph.n):
                m.addConstr(gp.quicksum(x[u] for u in graph.adj[v]) >= 1)
            for f in forced_vertices(graph).tolist():
                x[f].LB = 1
            m.setObjective(gp.quicksum(float(graph.w[v]) * x[v] for v in range(graph.n)), GRB.MINIMIZE)
            if hint is not None:
                hs = set(hint)
                for v in range(graph.n):
                    x[v].Start = 1.0 if v in hs else 0.0
            m.optimize()
            if m.SolCount == 0:
                return None, False, float("nan"), str(m.Status)
            S = [v for v in range(graph.n) if x[v].X > 0.5]
            return S, m.Status == GRB.OPTIMAL, float(m.ObjBound), str(m.Status)


def gurobi_available() -> bool:
    try:
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
        return True
    except Exception:
        return False


def ilp(graph: Graph, time_limit: float = 60.0, backend: str = "cpsat", threads: int = 8,
        seed: int = 0, warm_start: bool = True, hint=None) -> SolveResult:
    """Exact weighted ILP. `hint`: a feasible TDS used as the starting solution (e.g. the GNN+RL answer);
    otherwise weighted greedy + RR when warm_start=True."""
    graph.require_tds_exists()
    t0 = time.perf_counter()
    if backend == "auto":
        backend = "gurobi" if gurobi_available() else "cpsat"
    if hint is not None:
        hint = [int(v) for v in hint]
    elif warm_start:
        from .greedy import greedy_construct
        from ..postprocess import redundancy_removal
        hint = redundancy_removal(graph, greedy_construct(graph)[0], check=False)
    fn = {"cpsat": _cpsat, "highs": _highs, "gurobi": _gurobi}[backend]
    S, optimal, bound, status = fn(graph, time_limit, threads, seed, hint)
    if S is None:
        if hint is None:
            raise RuntimeError(f"ILP ({backend}) found no solution: {status}")
        S, optimal = hint, False      # fall back to the warm start (still verified)
    rt = time.perf_counter() - t0
    obj = float(graph.w[S].sum())
    gap = (obj - bound) / obj if obj > 0 and np.isfinite(bound) else float("nan")
    return SolveResult.build(graph, S, rt, f"ilp_{backend}", optimal=bool(optimal),
                             bound=float(bound), gap=float(gap), status=str(status),
                             time_limit=float(time_limit))


def brute_force(graph: Graph):
    """Exact γ_t by enumerating subsets in increasing size (n <= ~20). Returns (size, S)."""
    from itertools import combinations
    graph.require_tds_exists()
    n = graph.n
    nb_mask = [sum(1 << u for u in graph.adj[v]) for v in range(n)]
    for k in range(2, n + 1):
        for S in combinations(range(n), k):
            sm = 0
            for v in S:
                sm |= 1 << v
            if all(nb_mask[v] & sm for v in range(n)):
                return k, list(S)
    raise AssertionError("unreachable: V is a TDS when there is no isolated vertex")


def brute_force_weighted(graph: Graph):
    """Exact minimum-weight TDS by enumerating all subsets (n <= ~16). Returns (weight, S, all_optimal_sets)."""
    graph.require_tds_exists()
    n = graph.n
    w = graph.w
    nb_mask = [sum(1 << u for u in graph.adj[v]) for v in range(n)]
    best, sets = float("inf"), []
    for sm in range(1, 1 << n):
        if all(nb_mask[v] & sm for v in range(n)):
            S = [v for v in range(n) if sm >> v & 1]
            wt = float(w[S].sum())
            if wt < best - 1e-9:
                best, sets = wt, [S]
            elif abs(wt - best) <= 1e-9:
                sets.append(S)
    return best, sets[0], sets
