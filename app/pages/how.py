import streamlit as st

from components.theme import table

PIPE = [
    ("01", "Weighted graph", "edges + vertex weights w(v) ≥ 0; isolated vertices and bad weights rejected"),
    ("02", "Reduction", "the neighbour of every leaf is forced; weight-0 vertices are free"),
    ("03", "GNN encoder", "same 3-layer message-passing network; feature 9 now carries w(v)/max w"),
    ("04", "Q-values", "graph summary (mean ‖ max) joined with each vertex embedding"),
    ("05", "Pick a vertex", "highest Q among legal actions: not in S and gain > 0"),
    ("06", "Best of 1 + 16", "greedy rollout + 16 sampled rollouts; RR (most expensive first) + local search"),
    ("07", "Certify", "exact solver starts from this set and proves it minimal (or finds a cheaper one)"),
    ("08", "Verified TDS", "is_total_dominating_set; W(S) recomputed; 'proven minimum' only with a proof"),
]

ROWS = [
    {"Aspect": "Reward", "Previous TDS system": "r = −1 per chosen vertex (return = −|S|)",
     "This weighted system": "r = −w(v) / w̄_G (return = −W(S) / w̄_G, the objective scaled by a per-graph constant)"},
    {"Aspect": "Weight feature (index 9)", "Previous TDS system": "always 1",
     "This weighted system": "carries the real weight w(v) / max w — same input size (10), no architecture change"},
    {"Aspect": "Redundancy removal", "Previous TDS system": "by Q-value and degree",
     "This weighted system": "most expensive vertices first"},
    {"Aspect": "Local search", "Previous TDS system": "2-for-1 + plateau swaps",
     "This weighted system": "2-for-1 if w(c) < w(a) + w(b); 1-for-1 if w(c) < w(a); only strictly improving moves"},
    {"Aspect": "Baselines", "Previous TDS system": "greedy by gain",
     "This weighted system": "weighted greedy by gain / w(v); exact weighted ILP"},
    {"Aspect": "Inputs", "Previous TDS system": "edges only",
     "This weighted system": "edges + weights file, combined v/e format, DIMACS n-lines"},
    {"Aspect": "Algorithm", "Previous TDS system": "3-layer MPNN + Double DQN, mask gain > 0, degree-1 rule",
     "This weighted system": "unchanged (fine-tuned from the previous checkpoints)"},
]


def render():
    st.markdown('<div class="section-title" style="font-size:1.6rem;">How it works</div>', unsafe_allow_html=True)
    st.markdown("""<p>Every vertex v has a weight w(v). A set S is a <b>total dominating set</b> when every vertex,
including the vertices of S, has a neighbour in S. We look for the one with the smallest <b>total weight</b>
W(S) = Σ<sub>v∈S</sub> w(v). The cheapest set can contain <i>more</i> vertices than the smallest one: in two triangles
joined by a bridge whose two endpoints weigh 10, the smallest set is the two bridge vertices (weight 20), while the
cheapest uses four light vertices (weight 4).</p>
<p>The method is the same as the previous TDS system; only the reward changes. The agent receives
<span class="mono">r = −w(v) / w̄<sub>G</sub></span> for each vertex it adds (w̄<sub>G</sub> is the mean weight of the
graph, γ = 1), so its return is −W(S)/w̄<sub>G</sub>: maximising the return minimises the total weight.</p>""",
                unsafe_allow_html=True)
    cells = []
    for i, (n, t, s) in enumerate(PIPE):
        cls = "step loop" if n == "05" else "step"
        cells.append(f'<div class="{cls}"><div class="n">{n}</div><div class="t">{t}</div><div class="s">{s}</div></div>')
        if i < len(PIPE) - 1:
            cells.append('<div class="arrow">→</div>')
    st.markdown(f'<div class="pipe">{"".join(cells)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">Steps 04–05 repeat (dashed) until every vertex has a chosen neighbour; '
                'step 06 runs that construction 17 times.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">Finding the minimum: search, then certify</div>', unsafe_allow_html=True)
    st.markdown("""<p>Minimum-weight total domination is NP-hard, so no neural policy can <i>guarantee</i> the optimum
by itself. The default method therefore combines the two: the GNN + RL policy does the searching (one greedy rollout
plus 16 rollouts sampled from softmax(Q / T), with T chosen on the validation set, then redundancy removal and local
search), and its answer becomes the starting solution and upper bound of the exact CP-SAT solver. The solver either
proves no cheaper total dominating set exists, or finds a cheaper one and proves that. The app says "proven minimum"
only when the proof finishes; otherwise it shows the best set and a certified gap.</p>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Previous TDS system vs. this weighted system</div>', unsafe_allow_html=True)
    table(ROWS, ["Aspect", "Previous TDS system", "This weighted system"])

    st.markdown('<div class="section-title">Why the gain &gt; 0 mask never loses the optimum</div>', unsafe_allow_html=True)
    st.markdown("""<p>With positive weights, an optimal weighted total dominating set S* is minimal (dropping a
redundant vertex would lower the weight), so every v in S* has a <i>private neighbour</i> that only v covers. Insert the
forced vertices first (they belong to every total dominating set), then the rest of S* in any order: when v is inserted,
its private neighbour is still uncovered, so gain(v) ≥ 1 and v is a legal action. The optimal set is therefore always
reachable under the mask. Weight-0 vertices are pre-selected: they never increase W(S).</p>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">References</div>', unsafe_allow_html=True)
    st.markdown("""<p class="caption">Base paper: M. Chen, S. Liu, W. He (2024). <i>Learn to solve dominating set problem
with GNN and reinforcement learning.</i> Applied Mathematics and Computation 474:128717.<br>
Starting point: our previous minimum total dominating set system (GNN + Double DQN), whose checkpoints are fine-tuned
here with the weighted reward.</p>""", unsafe_allow_html=True)
