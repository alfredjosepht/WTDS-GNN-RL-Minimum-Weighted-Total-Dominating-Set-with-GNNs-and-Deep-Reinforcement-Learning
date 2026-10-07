"""The current graph, shared by all pages."""
import html

import streamlit as st

from .graph_input import example
from .theme import notice


def current():
    if "lg" not in st.session_state:
        st.session_state.lg = example("Two triangles, expensive bridge")      # default demo graph
        st.session_state.lg_key = ("ex", "Two triangles, expensive bridge")
    return st.session_state.lg


def set_current(lg, key):
    if st.session_state.get("lg_key") != key:
        st.session_state.lg = lg
        st.session_state.lg_key = key
        for k in ("result", "compare", "replay", "png"):
            st.session_state.pop(k, None)


def isolated_card(lg):
    iso = lg.graph.isolated_vertices().tolist()
    if not iso:
        return False
    shown = ", ".join(html.escape(str(lg.label(v))) for v in iso[:20]) + (" …" if len(iso) > 20 else "")
    notice(f"<b>No total dominating set can exist for this graph.</b><br>"
           f"A total dominating set needs every vertex to have a <i>neighbour</i> in the set, and "
           f"{len(iso)} vertex{'es have' if len(iso) > 1 else ' has'} no neighbours at all: "
           f"<span class='mono'>{shown}</span>.<br>Remove the isolated vertices or connect them to the graph, "
           f"then try again.", "err")
    return True


def graph_pills(lg):
    g = lg.graph
    import networkx as nx
    comps = nx.number_connected_components(g.to_networkx()) if g.n <= 20000 else None
    w = g.w
    pills = [f"n = {g.n}", f"m = {g.num_edges}", f"weights {w.min():g}–{w.max():g}, total {w.sum():g}",
             f"{comps} component{'s' if comps != 1 else ''}" if comps is not None else None,
             f"source: {lg.source}"]
    st.markdown("".join(f'<span class="pill">{html.escape(p)}</span>' for p in pills if p), unsafe_allow_html=True)
