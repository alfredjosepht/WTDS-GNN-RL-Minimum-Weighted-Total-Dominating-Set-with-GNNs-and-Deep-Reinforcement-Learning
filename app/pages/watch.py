import streamlit as st

from components import solver
from components.network import replay_html
from components.state import current, graph_pills, isolated_card
from components.theme import card_title, notice, stat_tiles

MAX_REPLAY = 600


def render():
    lg = current()
    g = lg.graph
    left, right = st.columns([1, 3], gap="large")
    with left:
        with st.container(border=True):
            card_title("Replay")
            st.markdown('<div class="caption">Replays how the fine-tuned agent builds a minimum-weight total dominating set on the '
                        'current graph (change the graph on the <b>Solve</b> page). The agent picks one vertex per '
                        'step among legal actions (not yet chosen, gain &gt; 0) until every vertex has a chosen '
                        'neighbour; redundancy removal then drops vertices that are not needed.</div>',
                        unsafe_allow_html=True)
            models = solver.available_models()
            model = st.radio("Model", list(models) or ["—"], key="watch_model")
            go = st.button("Record construction", type="primary", width="stretch",
                           disabled=not models or g.n > MAX_REPLAY or len(g.isolated_vertices()) > 0)
            if g.n > MAX_REPLAY:
                notice(f"Replay is limited to {MAX_REPLAY} vertices; this graph has {g.n}.", "warn")
    with right:
        st.markdown('<div class="section-title">Watch the agent</div>', unsafe_allow_html=True)
        graph_pills(lg)
        if isolated_card(lg):
            return
        if go:
            with st.spinner("Running the policy and recording every step…"):
                res = solver.run_rl(g, model, rr=True, ls_time=0.0, samples=1, trace=True)
            from wtds.reductions import preselected_vertices
            forced = preselected_vertices(g).tolist()          # forced (degree-1) + free (weight 0)
            picked = set(forced) | {s["vertex"] for s in res["trace"]}
            removed = sorted(picked - set(res["S"]))
            st.session_state.replay = dict(key=st.session_state.get("lg_key"), trace=res["trace"], forced=forced,
                                           removed=removed, final=res["S"], res=res)
        rp = st.session_state.get("replay")
        if not rp or rp.get("key") != st.session_state.get("lg_key"):
            st.markdown('<div class="caption" style="margin-top:12px;">Press <b>Record construction</b> to run the '
                        'policy. Then use Play, Pause, Step or the slider.</div>', unsafe_allow_html=True)
            return
        res = rp["res"]
        stat_tiles([("Final W(S)", solver.fmt_w(res["weight"]), f"|S| = {res['size']}, valid TDS", "hero"),
                    ("Steps", len(rp["trace"]), "vertices picked by the policy", ""),
                    ("Forced", len(rp["forced"]), "degree-1 rule / weight 0", ""),
                    ("Removed by RR", len(rp["removed"]), "most expensive first", "amber")])
        st.iframe(replay_html(lg, rp["trace"], rp["forced"], rp["removed"], rp["final"], height=500),
                        height=640)
