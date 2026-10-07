import streamlit as st

from components import solver
from components.graph_input import (EXAMPLES, GEN_TYPES, SAMPLE_COMBINED, SAMPLE_EDGES, SAMPLE_WEIGHTS,
                                    WEIGHT_LABELS, GraphInputError, example, generate, parse_text, parse_upload)
from components.network import MAX_DRAW, solution_html
from components.state import current, graph_pills, isolated_card, set_current
from components.theme import card_title, notice, stat_tiles

METHODS = ["GNN + RL → proven minimum", "GNN + RL (ours, weighted)", "Weighted greedy", "Weighted greedy + RR", "Unweighted TDS model", "Exact ILP"]


def input_card():
    card_title("1 · Weighted graph")
    t_ex, t_gen, t_up, t_paste = st.tabs(["Examples", "Generate", "Upload", "Paste"])
    with t_ex:
        st.caption("Small hand-checked graphs where the cheapest set is not always the smallest.")
        for i, name in enumerate(EXAMPLES):
            if st.button(name, key=f"ex_{i}", width="stretch",
                         type="primary" if st.session_state.get("lg_key") == ("ex", name) else "secondary"):
                set_current(example(name), ("ex", name))
                st.rerun()
    with t_gen:
        kind = GEN_TYPES[st.selectbox("Graph type", list(GEN_TYPES), index=0, key="gen_kind")]
        side, n, dens = 10, 100, 0
        if kind == "grid":
            side = st.slider("Side length (side × side grid)", 2, 60, 10, key="gen_side")
        else:
            n = st.slider("Vertices", 10, 5000, 100, step=10, key="gen_n")
            if kind == "er":
                dens = st.slider("Average degree", 2.0, 20.0, 6.0, 0.5, key="gen_er")
            elif kind == "ba":
                dens = st.slider("Edges per new vertex (m)", 1, 8, 3, key="gen_ba")
            elif kind == "ws":
                dens = st.select_slider("Ring neighbours (k)", [2, 4, 6, 8], 4, key="gen_ws")
        wd = WEIGHT_LABELS[st.selectbox("Weight distribution", list(WEIGHT_LABELS), key="gen_w")]
        seed = st.number_input("Seed", 0, 10**6, 0, key="gen_seed")
        if st.button("Use this graph", key="gen_go", width="stretch"):
            set_current(generate(kind, int(n), float(dens), int(seed), side=int(side), weights=wd),
                        ("gen", kind, n, dens, seed, side, wd))
    with t_up:
        st.caption("Edges: `u v` per line (any labels), the combined format (`v <vertex> <weight>`, `e <u> <v>`), "
                   "DIMACS `.col`/`.clq` with optional `n <vertex> <weight>` lines, or `.mtx`. Weights file: "
                   "`vertex weight` per line. Missing weights default to 1.")
        up = st.file_uploader("Graph file", type=["txt", "edges", "el", "col", "clq", "mtx"], key="up_g")
        upw = st.file_uploader("Weights file (optional)", type=["txt", "w", "weights"], key="up_w")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Edges", SAMPLE_EDGES, "sample_edges.txt", "text/plain", width="stretch",
                           help="Sample edge list")
        c2.download_button("Weights", SAMPLE_WEIGHTS, "sample_weights.txt", "text/plain", width="stretch",
                           help="Sample weights file")
        c3.download_button("Combined", SAMPLE_COMBINED, "sample_combined.txt", "text/plain", width="stretch",
                           help="Sample combined v/e file")
        if up is not None:
            key = ("up", up.name, up.size, up.file_id, upw.file_id if upw else None)
            if st.session_state.get("last_upload") != key:
                st.session_state.last_upload = key
                try:
                    set_current(parse_upload(up.name, up.getvalue(), upw.getvalue() if upw else None), key)
                except GraphInputError as ex:
                    st.session_state.input_error = str(ex)
    with t_paste:
        et = st.text_area("Edges (u v per line)", "1 2\n2 3\n3 1\n4 5\n5 6\n6 4\n3 4", height=130)
        wt = st.text_area("Weights (vertex weight per line)", "3 10\n4 10", height=90,
                          help="Vertices without a weight get weight 1 (a warning is shown).")
        if st.button("Use pasted graph", width="stretch"):
            try:
                set_current(parse_text(et, wt, "pasted text"), ("paste", et, wt))
                st.session_state.pop("input_error", None)
            except GraphInputError as ex:
                st.session_state.input_error = str(ex)


def method_card(lg):
    card_title("2 · Method")
    n = lg.graph.n
    method = st.segmented_control("Method", METHODS, default=METHODS[0], label_visibility="collapsed",
                                  key="method") or METHODS[0]
    ilp_off = n > solver.ILP_MAX_N
    if method == "Exact ILP" and ilp_off:
        notice(f"Exact ILP is disabled above {solver.ILP_MAX_N} vertices: the problem is NP-hard and the solver "
               f"could run for minutes. Use the GNN or greedy methods for this graph.", "warn")
    opts = {}
    if method == "GNN + RL → proven minimum":
        models = solver.available_models()
        opts["model"] = st.radio("Model", list(models) or ["—"], horizontal=True, key="model_choice_x")
        opts["ilp_time"] = st.slider("Certification time limit (s)", 5, 120, 20, 5,
                                     help="The GNN+RL answer is given to the exact solver as its starting solution; "
                                          "it proves it minimal or improves it within this time.")
        st.markdown('<div class="caption">The GNN + RL policy searches (greedy + 16 sampled rollouts, redundancy '
                    'removal, local search); the exact solver then <b>certifies</b> the answer, so the result is '
                    'the proven minimum whenever the proof finishes in time.</div>', unsafe_allow_html=True)
    if method == "GNN + RL (ours, weighted)":
        models = solver.available_models()
        if not models:
            notice("No fine-tuned checkpoint found in <span class='mono'>checkpoints/</span>.", "warn")
        opts["model"] = st.radio("Model", list(models) or ["—"], horizontal=True, key="model_choice")
        opts["samples"] = st.select_slider(
            "Sampled rollouts k", [1, 4, 16], 1,
            help=f"k = 1 runs the greedy policy; k > 1 adds k rollouts sampled from softmax(Q / {solver.SAMPLE_TEMPERATURE:g}) "
                 "and keeps the cheapest (temperature chosen on the validation set).")
    if method in ("GNN + RL (ours, weighted)", "Unweighted TDS model"):
        opts["rr"] = st.toggle("Redundancy removal", True, help="Drop vertices (most expensive first) while S stays a TDS.")
    if method in ("GNN + RL (ours, weighted)", "Weighted greedy + RR", "Unweighted TDS model"):
        ls = st.toggle("Local search", False, help="Time-limited 2-for-1 and 1-for-1 moves that strictly lower W(S).")
        opts["ls_time"] = st.slider("Local search time (s)", 0.2, 5.0, 1.0, 0.2) if ls else 0.0
    go = st.button("Find total dominating set", type="primary", width="stretch",
                   disabled=(method == "Exact ILP" and ilp_off),
                   help=f"Exact ILP is limited to {solver.ILP_MAX_N} vertices." if ilp_off and method == "Exact ILP" else None)
    return method, opts, go


def run(lg, method, opts):
    g = lg.graph
    if method == "GNN + RL → proven minimum":
        return solver.run_certified(g, opts["model"], ls_time=1.0, samples=16, ilp_time=opts["ilp_time"])
    if method == "GNN + RL (ours, weighted)":
        return solver.run_rl(g, opts["model"], rr=opts["rr"], ls_time=opts.get("ls_time", 0.0), samples=opts["samples"])
    if method == "Unweighted TDS model":
        return solver.run_old_model(g, rr=opts["rr"], ls_time=opts.get("ls_time", 0.0))
    key = {"Weighted greedy": "greedy", "Weighted greedy + RR": "greedy_rr", "Exact ILP": "ilp"}[method]
    return solver.run_baseline(g, key, ls_time=opts.get("ls_time", 0.0))


def result_view(lg, res):
    g = lg.graph
    opt = solver.optimum_for(lg)
    items = [("Total weight W(S)", solver.fmt_w(res["weight"]), res["method"], "hero"),
             ("|S|", res["size"], "vertices chosen", "")]
    if "rl_weight" in res:                                   # certified GNN + RL
        if res["optimal"]:
            items.append(("Proven minimum", "yes", "the exact solver proved no cheaper TDS exists", "teal"))
        else:
            items.append(("Certified gap", f"{100 * res['certified_gap']:.1f}%",
                          f"lower bound {res['lower_bound']:.1f} (time limit reached)", "amber"))
        opt = None
    if opt:
        val, src = opt
        gap = 100.0 * (res["weight"] - val) / val if val > 0 else 0.0
        items.append(("Gap", f"{gap:.1f}%", f"vs W* = {solver.fmt_w(val)}, {src}", "amber" if gap > 1e-9 else "teal"))
    elif res.get("optimal") is not None and "bound" in res:        # plain exact ILP
        items.append(("ILP status", "optimal" if res["optimal"] else "time limit", f"bound {res['bound']:.1f}", ""))
    items.append(("Runtime", f"{res['runtime']:.3f}s", "wall clock", ""))
    items.append(("Valid", "✓", "is_total_dominating_set", "teal"))
    stat_tiles(items)
    notice("<b>Valid total dominating set:</b> every vertex, including every vertex of S, has a neighbour in S. "
           "W(S) is recomputed from the chosen vertices.")
    if "rl_weight" in res:
        how = ("the exact solver proved it optimal" if res["rl_was_optimal"] else
               ("the exact solver then found a cheaper set and proved it optimal" if res["optimal"] and res["improved_by_ilp"]
                else ("the exact solver improved it" if res["improved_by_ilp"] else "the exact solver could not finish the proof in time")))
        roll = "greedy + 16 sampled rollouts" if res.get("samples", 1) > 1 else "greedy rollout (large graph)"
        st.markdown(f'<div class="caption">GNN + RL alone ({roll}, RR, LS) found W = {solver.fmt_w(res["rl_weight"])} '
                    f'(|S| = {res["rl_size"]}, {res["rl_time"]:.2f}s); {how} ({res["ilp_time_used"]:.2f}s).</div>',
                    unsafe_allow_html=True)
    if "size_raw" in res:
        st.markdown(f'<div class="caption">Construction picked {res["size_raw"]} vertices ({res["forced"]} forced or '
                    f'free) → {res["size_rr"]} after redundancy removal → {res["size_ls"]} after local search.</div>',
                    unsafe_allow_html=True)
    if g.n <= MAX_DRAW:
        st.iframe(solution_html(lg, res["S"], height=540), height=580)
    else:
        notice(f"The graph has {g.n} vertices; drawing is skipped above {MAX_DRAW} so the page stays responsive. "
               "Download the solution below.")
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("solution.json", solver.solution_json(lg, res), "solution.json", "application/json",
                       width="stretch")
    if g.n <= MAX_DRAW:
        if c2.button("Prepare PNG", width="stretch"):
            st.session_state.png = solver.solution_png(lg, res["S"])
        if st.session_state.get("png"):
            c2.download_button("solution.png", st.session_state.png, "solution.png", "image/png", width="stretch")


def render():
    left, right = st.columns([1, 2.5], gap="large")
    with left:
        with st.container(border=True):
            input_card()
        lg = current()
        with st.container(border=True):
            method, opts, go = method_card(lg)
    with right:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
        err = st.session_state.pop("input_error", None)
        if err:
            notice(f"<b>Could not read the input.</b> {err}", "err")
        graph_pills(lg)
        if lg.note:
            st.markdown(f'<div class="caption">{lg.note}</div>', unsafe_allow_html=True)
        for w in lg.warnings:
            notice(w, "warn")
        if isolated_card(lg):
            return
        if go:
            try:
                with st.spinner("Finding a minimum-weight total dominating set…"):
                    st.session_state.result = run(lg, method, opts)
                    st.session_state.pop("png", None)
            except Exception as ex:
                notice(f"<b>Could not solve:</b> {ex}", "err")
        res = st.session_state.get("result")
        if res is None:
            st.markdown('<div class="caption" style="margin-top:14px;">Choose a method and press <b>Find total '
                        'dominating set</b>. Numbers on the vertices are their weights.</div>', unsafe_allow_html=True)
            if lg.graph.n <= MAX_DRAW:
                st.iframe(solution_html(lg, [], height=500), height=540)
        else:
            result_view(lg, res)
