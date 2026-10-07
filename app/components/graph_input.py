"""Graph + weight input for the app: tolerant parsing (with warnings), generation and hand examples."""
from dataclasses import dataclass, field
from pathlib import Path
import tempfile

import numpy as np
import streamlit as st

from wtds.generators import lattice_dims, make_graph
from wtds.graph import Graph, InvalidWeightError, validate_weights

SAMPLE_EDGES = """# Two triangles joined by a bridge (edges "u v", one per line)
1 2
2 3
3 1
4 5
5 6
6 4
3 4
"""
SAMPLE_WEIGHTS = """# vertex weight
1 1
2 1
3 10
4 10
5 1
6 1
"""
SAMPLE_COMBINED = """# combined format: 'v <vertex> <weight>' and 'e <u> <v>'
v 1 1
v 2 1
v 3 10
v 4 10
v 5 1
v 6 1
e 1 2
e 2 3
e 3 1
e 4 5
e 5 6
e 6 4
e 3 4
"""

GRID3 = [(1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9), (1, 4), (4, 7), (2, 5), (5, 8), (3, 6), (6, 9)]
EXAMPLES = {   # name: (edges, weights (unlisted = 1), note)
    "Two triangles, expensive bridge": ([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (3, 4)], {3: 10, 4: 10},
                                        "Optimum W = 4 with 4 vertices {1,2,5,6}; the smallest TDS {3,4} weighs 20."),
    "3×3 grid, expensive centre": (GRID3, {5: 20}, "Optimum W = 4 while avoiding the centre (weight 20)."),
    "Star with heavy centre": ([(0, i) for i in range(1, 6)], {0: 10, 1: 3, 2: 1, 3: 4, 4: 1, 5: 5},
                               "The centre is forced by the leaves; optimum W = 11."),
    "Triangle": ([(1, 2), (2, 3), (3, 1)], {1: 5, 2: 1, 3: 2}, "Optimum W = 3, S = {2,3}."),
    "Square": ([(1, 2), (2, 3), (3, 4), (4, 1)], {1: 4, 2: 1, 3: 1, 4: 4}, "Optimum W = 2, S = {2,3}."),
    "Path (forced)": ([(1, 2), (2, 3), (3, 4)], {1: 1, 2: 7, 3: 2, 4: 1}, "Both inner vertices are forced: W = 9."),
    "Two triangles, unit weights": ([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (3, 4)], {},
                                    "Optimum W = 2, S = {3,4}."),
    "Hexagon": ([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)], {1: 1, 2: 1, 3: 5, 4: 5, 5: 1, 6: 1},
                "Optimum W = 4, S = {1,2,5,6}."),
}


@dataclass
class LoadedGraph:
    graph: Graph
    labels: list                       # original label of internal vertex i
    source: str
    warnings: list = field(default_factory=list)
    spec: dict = None                  # generator spec (for generated graphs)
    note: str = ""

    def label(self, v):
        return self.labels[v] if self.labels is not None else v


class GraphInputError(ValueError):
    pass


def _parse_edges(text):
    if not text or not text.strip():
        raise GraphInputError("The edge input is empty. Provide at least one edge such as `0 1`.")
    labels, edges, bad, loops, dup, wl = {}, set(), [], 0, 0, []

    def lab(tok):
        if tok not in labels:
            labels[tok] = len(labels)
        return labels[tok]

    lines = text.splitlines()
    dimacs = any(l.split()[:2] in (["p", "edge"], ["p", "col"]) for l in lines if l.strip())
    combined = not dimacs and any(l.split()[:1] == ["v"] and len(l.split()) >= 3 for l in lines)
    for ln, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line[0] in "#%":
            continue
        parts = line.replace(",", " ").replace("\t", " ").split()
        if dimacs:
            if parts[0] in ("c", "p"):
                continue
            if parts[0] == "n" and len(parts) >= 3:
                lab(parts[1])
                wl.append((parts[1], parts[2]))
                continue
            if parts[0] == "e":
                parts = parts[1:]
        elif combined:
            if parts[0] == "v" and len(parts) >= 3:
                lab(parts[1])
                wl.append((parts[1], parts[2]))
                continue
            if parts[0] == "e":
                parts = parts[1:]
        if len(parts) == 1:
            lab(parts[0])
            continue
        if len(parts) != 2:
            bad.append(ln)
            continue
        u, v = lab(parts[0]), lab(parts[1])
        if u == v:
            loops += 1
            continue
        e = (min(u, v), max(u, v))
        if e in edges:
            dup += 1
        edges.add(e)
    if not labels:
        raise GraphInputError("No vertices were found. Each line should look like `u v`.")
    warns = []
    if bad:
        warns.append(f"Skipped {len(bad)} malformed line(s) (e.g. line {bad[0]}).")
    if loops:
        warns.append(f"Dropped {loops} self-loop(s): total domination uses open neighbourhoods.")
    if dup:
        warns.append(f"Merged {dup} duplicate edge(s).")
    return labels, edges, wl, warns


def _parse_weights(text, l2i, n):
    """'vertex weight' lines -> array; missing weights default to 1 (warning)."""
    w = np.ones(n)
    given = np.zeros(n, dtype=bool)          # explicit NaN must be rejected, not treated as missing
    warns, unknown, bad = [], [], []
    for ln, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#%":
            continue
        p = line.replace(",", " ").split()
        if len(p) != 2:
            bad.append(ln)
            continue
        if p[0] not in l2i:
            unknown.append(p[0])
            continue
        try:
            x = float(p[1])
        except ValueError:
            raise GraphInputError(f"Vertex {p[0]} has a non-numeric weight `{p[1]}`.")
        w[l2i[p[0]]] = x
        given[l2i[p[0]]] = True
    if bad:
        warns.append(f"Skipped {len(bad)} malformed weight line(s) (e.g. line {bad[0]}).")
    if unknown:
        warns.append(f"{len(unknown)} weight line(s) name vertices that are not in the graph (e.g. {unknown[0]}).")
    missing = int((~given).sum())
    if missing:
        warns.append(f"{missing} of {n} vertices have no weight; their weight defaults to 1." if missing < n
                     else "No weights were given: every weight defaults to 1.")
    return w, warns


def parse_text(edges_text: str, weights_text: str = "", source: str = "pasted text") -> LoadedGraph:
    labels, edges, wl, warns = _parse_edges(edges_text)
    order = list(labels)
    try:
        order = sorted(order, key=lambda t: int(t))
    except ValueError:
        pass
    l2i = {t: i for i, t in enumerate(order)}
    remap = {labels[t]: l2i[t] for t in order}
    wtext = "\n".join(f"{a} {b}" for a, b in wl) + "\n" + (weights_text or "")
    w, wwarn = _parse_weights(wtext, l2i, len(order))
    as_int = all(t.lstrip("-").isdigit() for t in order)
    shown = [int(t) if as_int else t for t in order]
    try:
        validate_weights(w, shown)
    except InvalidWeightError as ex:
        raise GraphInputError(f"Invalid weight: {ex}")
    g = Graph.from_edges(len(order), [(remap[a], remap[b]) for a, b in edges], weights=w, name=Path(source).stem)
    if not as_int or [int(t) for t in order] != list(range(len(order))):
        warns.append("Vertex labels were relabelled internally; results show your original labels.")
    return LoadedGraph(g, shown, source, warns + wwarn)


def parse_upload(name: str, data: bytes, weights_data: bytes = None) -> LoadedGraph:
    if not data:
        raise GraphInputError(f"`{name}` is empty.")
    wtext = weights_data.decode("utf-8", errors="replace") if weights_data else ""
    if Path(name).suffix.lower() == ".mtx":
        from wtds.io import read_mtx
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mtx") as f:
            f.write(data)
        try:
            g = read_mtx(f.name)
        except Exception as ex:
            raise GraphInputError(f"Could not read `{name}` as Matrix Market ({type(ex).__name__}: {ex}).")
        text = "\n".join(f"{u + 1} {v + 1}" for u, v in g.edges().tolist()) + "".join(f"\n{v + 1}" for v in range(g.n))
        return parse_text(text, wtext, name)
    return parse_text(data.decode("utf-8", errors="replace"), wtext, name)


GEN_TYPES = {"Erdős–Rényi": "er", "Barabási–Albert": "ba", "Watts–Strogatz": "ws",
             "Grid": "grid", "Triangular lattice": "tri", "Hexagonal lattice": "hex"}
WEIGHT_LABELS = {"Uniform 1–100": "uniform_int", "Uniform 1–10": "uniform_small",
                 "Degree-correlated": "degree_correlated", "Unit (all 1)": "unit"}


@st.cache_data(show_spinner=False, max_entries=32)
def generate(kind: str, n: int, density: float, seed: int, side: int = 10, weights: str = "uniform_int") -> LoadedGraph:
    if kind == "er":
        spec = dict(family="er", n=n, avg_deg=float(density), seed=seed)
    elif kind == "ba":
        spec = dict(family="ba", n=n, m=int(density), seed=seed)
    elif kind == "ws":
        spec = dict(family="ws", n=n, k=int(density), p=0.1, seed=seed)
    elif kind == "grid":
        spec = dict(family="grid", rows=side, cols=side, seed=seed)
    else:
        a, b = lattice_dims(kind, n)
        spec = dict(family=kind, rows=a, cols=b, seed=seed)
    spec["weights"] = weights
    g = make_graph(spec)
    return LoadedGraph(g, None, g.name, [], spec)


def example(name: str) -> LoadedGraph:
    edges, weights, note = EXAMPLES[name]
    all_v = sorted({x for e in edges for x in e})
    lg = parse_text("\n".join(f"{a} {b}" for a, b in edges),
                    "\n".join(f"{v} {weights.get(v, 1)}" for v in all_v), name)
    lg.note = note
    return lg


def known_optimum(lg: LoadedGraph):
    """(value, source) when the optimum is known in closed form (unit-weight square grids only)."""
    sp = lg.spec or {}
    if sp.get("family") == "grid" and sp.get("rows") == sp.get("cols") and np.all(lg.graph.w == 1):
        k = sp["rows"]
        m, r = divmod(k, 4)
        return float((2 * m + 1) * (2 * m + r)), f"closed form, {k}×{k} grid, unit weights"
    return None
