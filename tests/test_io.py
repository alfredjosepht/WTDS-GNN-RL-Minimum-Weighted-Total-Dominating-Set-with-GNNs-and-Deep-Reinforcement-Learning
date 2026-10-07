import numpy as np

from wtds.generators import GraphSampler, make_graph
from wtds.io import load_graphs, read_dimacs, read_edge_list, read_graph, read_mtx, save_graphs, write_edge_list


def test_npz_roundtrip_and_regeneration(tmp_path):
    s = GraphSampler(0)
    gs = [s.sample(20, 60) for _ in range(10)] + [make_graph(dict(family="er", n=30, avg_deg=4, seed=1, weights=True))]
    save_graphs(gs, tmp_path / "set.npz")
    back = load_graphs(tmp_path / "set.npz")
    for a, b in zip(gs, back):
        assert a.n == b.n and np.array_equal(a.edges(), b.edges()) and a.meta == b.meta
        if a.weights is not None:
            assert np.array_equal(a.weights, b.weights)
        regen = make_graph(b.meta)          # spec alone reproduces the graph
        assert np.array_equal(regen.edges(), a.edges())


def test_edge_list_dimacs_mtx(tmp_path):
    g = make_graph(dict(family="er", n=25, avg_deg=5, seed=3))
    write_edge_list(g, tmp_path / "g.txt")
    assert np.array_equal(read_edge_list(tmp_path / "g.txt").edges(), g.edges())
    with open(tmp_path / "g.col", "w") as f:
        f.write(f"c test\np edge {g.n} {g.num_edges}\n")
        for u, v in g.edges().tolist():
            f.write(f"e {u + 1} {v + 1}\n")
    assert np.array_equal(read_dimacs(tmp_path / "g.col").edges(), g.edges())
    with open(tmp_path / "g.mtx", "w") as f:
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write(f"{g.n} {g.n} {g.num_edges}\n")
        for u, v in g.edges().tolist():
            f.write(f"{v + 1} {u + 1}\n")
    assert np.array_equal(read_mtx(tmp_path / "g.mtx").edges(), g.edges())
    assert read_graph(tmp_path / "g.col").n == g.n


def test_string_labels(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("# comment\na b\nb c\nc,a\n")
    g = read_edge_list(p)
    assert g.n == 3 and g.num_edges == 3
