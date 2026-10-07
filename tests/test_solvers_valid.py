"""Every solver returns a valid TDS on 1,000 random graphs (trees, disconnected
graphs, graphs with leaves included). RL solvers are added in later phases."""
import pytest

from wtds.baselines import greedy, greedy_rr
from wtds.checker import is_total_dominating_set
from wtds.postprocess import is_minimal

from helpers import medium_random_graphs, small_random_graphs

POOL = small_random_graphs(500, seed=11, n_max=12) + medium_random_graphs(500, seed=12)


def test_pool_contents():
    assert len(POOL) == 1000
    assert sum((g.degrees == 1).any() for g in POOL) > 200
    import networkx as nx
    assert sum(not nx.is_connected(g.to_networkx()) for g in POOL) > 50


@pytest.mark.parametrize("solver", [greedy, greedy_rr], ids=["greedy", "greedy_rr"])
def test_baselines_valid(solver):
    for g in POOL:
        r = solver(g)
        assert r.is_valid and is_total_dominating_set(g.adj, r.vertices), g.name
        if solver is greedy_rr:
            assert is_minimal(g, r.vertices)


def test_rl_solvers_valid(random_checkpoint):
    from wtds.solve import solve
    for i, g in enumerate(POOL):
        variants = [dict(rr=False, ls_time=0), dict(rr=True, ls_time=0)]
        if i % 4 == 0:
            variants.append(dict(rr=True, ls_time=0.01))
        if i % 25 == 0:
            variants.append(dict(rr=True, ls_time=0, samples=4, temperature=0.5))
        for kw in variants:
            r = solve(g, None, "rl", checkpoint=random_checkpoint, device="cpu", **kw)
            assert r.is_valid and is_total_dominating_set(g.adj, r.vertices), (g.name, kw)
            if kw["rr"]:
                assert is_minimal(g, r.vertices)    # RR output is minimal; LS output is RR-minimal too


def test_solve_api_inputs_and_errors(random_checkpoint, tmp_path):
    import networkx as nx
    from wtds import NoTotalDominatingSetError, solve
    from wtds.io import write_edge_list
    r = solve(nx.petersen_graph(), method="rl", checkpoint=random_checkpoint, device="cpu", ls_time=0.05)
    assert r.is_valid and r.size >= 4
    r = solve(nx.petersen_graph(), method="ilp")
    assert r.size == 4 and r.info["optimal"]
    G = nx.path_graph(5)
    G.add_node(99)
    with pytest.raises(NoTotalDominatingSetError):
        solve(G, method="greedy")
    from wtds.graph import Graph
    p = tmp_path / "g.txt"
    write_edge_list(Graph.from_networkx(nx.cycle_graph(9)), p)
    assert solve(str(p), method="greedy_rr_ls", ls_time=0.05).size == 5
