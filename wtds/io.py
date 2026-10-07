"""Graph input/output.

Readers: plain edge lists (`u v [w]` per line, '#'/'%' comments, arbitrary integer
or string labels), DIMACS `.clq/.col` (`p edge n m`, `e u v`, 1-based) and
Matrix Market `.mtx` (pattern/real, symmetric or general; treated as undirected).
Test sets are stored as `.npz` (edges + weights + JSON spec per graph).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .graph import Graph, InvalidWeightError


def read_edge_list(path, name: str = None) -> Graph:
    labels, edges = {}, []

    def lab(t):
        if t not in labels:
            labels[t] = len(labels)
        return labels[t]

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "#%":
                continue
            parts = line.replace(",", " ").split()
            if len(parts) == 1:          # a lone vertex
                lab(parts[0])
                continue
            edges.append((lab(parts[0]), lab(parts[1])))
    # Relabel so integer labels keep their natural order.
    keys = list(labels)
    try:
        order = sorted(keys, key=lambda k: int(k))
    except ValueError:
        order = keys
    remap = {labels[k]: i for i, k in enumerate(order)}
    e = [(remap[a], remap[b]) for a, b in edges]
    return Graph.from_edges(len(order), e, name=name or Path(path).stem,
                            meta={"source": str(path), "labels": order if len(order) <= 10000 else None})


def read_dimacs(path, name: str = None) -> Graph:
    """DIMACS .col/.clq; optional vertex weights as 'n <vertex> <weight>' lines (1-based)."""
    n, edges, wts = None, [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if not p or p[0] == "c":
                continue
            if p[0] == "p":
                n = int(p[2])
            elif p[0] in ("e", "a"):
                edges.append((int(p[1]) - 1, int(p[2]) - 1))
            elif p[0] == "n" and len(p) >= 3:
                wts[int(p[1]) - 1] = float(p[2])
    if n is None:
        raise ValueError(f"{path}: missing 'p edge n m' line")
    meta = {"source": str(path)}
    w = None
    if wts:
        w = np.ones(n)
        for v, x in wts.items():
            w[v] = x
        if len(wts) < n:
            meta["weight_warnings"] = [f"{n - len(wts)} vertices have no 'n' weight line; their weight defaults to 1"]
    else:
        meta["weight_warnings"] = ["no vertex weights in the file; all weights default to 1"]
        w = np.ones(n)
    return Graph.from_edges(n, edges, weights=w, name=name or Path(path).stem, meta=meta)


def parse_weight_lines(lines, label_to_index: dict, n: int):
    """'vertex weight' per line (labels as in the edge list). Returns (weights, warnings)."""
    w = np.ones(n)
    given = np.zeros(n, dtype=bool)          # an explicit NaN is invalid, not "missing"
    warns, unknown = [], []
    for ln, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line[0] in "#%":
            continue
        p = line.replace(",", " ").split()
        if len(p) < 2:
            warns.append(f"weights line {ln} skipped (expected 'vertex weight')")
            continue
        if p[0] not in label_to_index:
            unknown.append(p[0])
            continue
        try:
            w[label_to_index[p[0]]] = float(p[1])
            given[label_to_index[p[0]]] = True
        except ValueError:
            raise InvalidWeightError(f"vertex {p[0]} has non-numeric weight {p[1]!r}")
    if unknown:
        warns.append(f"{len(unknown)} weight line(s) name vertices not in the graph (e.g. {unknown[0]}); ignored")
    missing = int((~given).sum())
    if missing:
        warns.append(f"{missing} vertex/vertices have no weight; defaulting to 1")
    return w, warns


def read_combined(path, name: str = None) -> Graph:
    """Combined format: 'v <vertex> <weight>' and 'e <u> <v>' lines (any labels)."""
    labels, edges, wl = {}, [], []

    def lab(t):
        if t not in labels:
            labels[t] = len(labels)
        return labels[t]

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            p = raw.split()
            if not p or p[0][0] in "#%c":
                continue
            if p[0] == "v" and len(p) >= 3:
                lab(p[1])
                wl.append(f"{p[1]} {p[2]}")
            elif p[0] == "e" and len(p) >= 3:
                edges.append((lab(p[1]), lab(p[2])))
    order = list(labels)
    w, warns = parse_weight_lines(wl, labels, len(order))
    return Graph.from_edges(len(order), edges, weights=w, name=name or Path(path).stem,
                            meta={"source": str(path), "labels": order, "weight_warnings": warns})


def attach_weights(g: Graph, weights_path=None) -> Graph:
    """Add weights from a 'vertex weight' file (labels as in the graph file); default 1 with a warning."""
    labels = g.meta.get("labels") or list(range(g.n))
    l2i = {str(t): i for i, t in enumerate(labels)}
    if weights_path is None:
        if g.weights is not None:
            return g
        w, warns = np.ones(g.n), ["no weights given; all weights default to 1"]
    else:
        with open(weights_path, encoding="utf-8", errors="replace") as f:
            w, warns = parse_weight_lines(f.read().splitlines(), l2i, g.n)
    meta = dict(g.meta)
    meta["weight_warnings"] = meta.get("weight_warnings", []) + warns
    return Graph(n=g.n, indptr=g.indptr, indices=g.indices, weights=_checked(w, labels), name=g.name, meta=meta)


def _checked(w, labels):
    from .graph import validate_weights
    validate_weights(w, labels)
    return np.asarray(w, dtype=np.float64)


def read_mtx(path, name: str = None) -> Graph:
    from scipy.io import mmread
    from scipy.sparse import coo_matrix
    try:
        A = mmread(str(path), spmatrix=False)
    except TypeError:                     # older SciPy
        A = mmread(str(path))
    A = coo_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{path}: adjacency matrix must be square, got {A.shape}")
    return Graph.from_edges(A.shape[0], np.stack([A.row, A.col], 1), name=name or Path(path).stem,
                            meta={"source": str(path)})


def _is_combined(path) -> bool:
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            p = raw.split()
            if p and p[0] not in ("#", "%"):
                if p[0] in ("v", "e") and len(p) >= 3:
                    return True
                if p[0][0] not in "#%":
                    return False
    return False


def read_graph(path, weights=None) -> Graph:
    """Read a graph and (optionally) a weights file. Without weights: all 1, with a warning."""
    ext = Path(path).suffix.lower()
    if ext in (".clq", ".col", ".dimacs"):
        g = read_dimacs(path)
    elif ext == ".mtx":
        g = read_mtx(path)
    elif _is_combined(path):
        g = read_combined(path)
    else:
        g = read_edge_list(path)
    return attach_weights(g, weights) if (weights is not None or g.weights is None) else g


def write_edge_list(graph: Graph, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# n={graph.n} m={graph.num_edges} name={graph.name}\n")
        for u, v in graph.edges().tolist():
            f.write(f"{u} {v}\n")


def save_graphs(graphs, path) -> None:
    """Save a list of Graphs (with their generator specs) to a single .npz."""
    arrs = {}
    metas = []
    for i, g in enumerate(graphs):
        arrs[f"e{i}"] = g.edges().astype(np.int32)
        if g.weights is not None:
            arrs[f"w{i}"] = g.weights
        metas.append({"n": g.n, "name": g.name, "meta": g.meta})
    arrs["index"] = np.array(json.dumps(metas))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrs)


def load_graphs(path) -> list:
    z = np.load(path, allow_pickle=False)
    metas = json.loads(str(z["index"]))
    out = []
    for i, m in enumerate(metas):
        w = z[f"w{i}"] if f"w{i}" in z.files else None
        out.append(Graph.from_edges(m["n"], z[f"e{i}"], weights=w, name=m["name"], meta=m["meta"]))
    return out
