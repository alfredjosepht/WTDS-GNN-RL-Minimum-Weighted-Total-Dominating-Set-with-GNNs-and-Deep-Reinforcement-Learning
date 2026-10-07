import json
from pathlib import Path

import streamlit as st

from components import solver
from components.theme import notice, stat_tiles, table

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "results" / "final"


def pending(what):
    notice(f"<b>Experiment pending.</b> {what} has not been produced yet; it appears here automatically once "
           f"<span class='mono'>scripts/run_final.py</span> has run.", "warn")


def model_card():
    info = solver.model_info()
    st.markdown('<div class="section-title">Model card</div>', unsafe_allow_html=True)
    if not info:
        pending("checkpoints/model_info.json")
        return
    tiles = []
    for s in info["seeds"]:
        tiles.append((f"Seed {s['seed']} · val gap", f"{100 * s['best_val_gap_rr']:.2f}%",
                      f"after RR; {100 * s['val_gap_rr_step0']:.1f}% before fine-tuning; "
                      f"{s['finetune_steps']:,} steps", "teal"))
    tiles.append(("Weighted greedy + RR", f"{100 * info['seeds'][0]['val_gap_weighted_greedy_rr']:.2f}%",
                  "same 100 validation graphs", ""))
    tiles.append(("Parameters", f"{info['parameters']:,}", f"{info['encoder'].upper()}, {info['layers']} layers, "
                                                          f"hidden {info['hidden']}", ""))
    stat_tiles(tiles)
    st.markdown(f'<div class="caption">{info["validation_note"]}</div>', unsafe_allow_html=True)


def render():
    st.markdown('<div class="section-title" style="font-size:1.6rem;">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">Every number on this page is read from files written by runs in this repository '
                '(<span class="mono">checkpoints/model_info.json</span>, <span class="mono">results/summary.json</span>). '
                'Every reported solution was verified with <span class="mono">is_total_dominating_set</span> and its '
                'weight recomputed from the solution.</div>', unsafe_allow_html=True)
    model_card()
    sp = ROOT / "results" / "summary.json"
    st.markdown('<div class="section-title">Experiments</div>', unsafe_allow_html=True)
    if not sp.exists():
        pending("results/summary.json")
        return
    s = json.loads(sp.read_text())
    v, h, ov = s["validity"], s["headline"], s.get("seed_overlap", {})
    stat_tiles([("Solutions checked", f"{v['solutions_checked']:,}", "all methods, all experiments", ""),
                ("Valid", f"{100 * v['valid_fraction']:.0f}%", "is_total_dominating_set", "teal"),
                ("W1 ILP proven", f"{h['ilp_proven']} / {h['instances']}", f"{h['ilp_not_proven']} not proven (excluded)", ""),
                ("Seed overlap", f"{ov.get('train_val_overlap', '?')}/{ov.get('train_test_overlap', '?')}/"
                                 f"{ov.get('val_test_overlap', '?')}", "train–val / train–test / val–test", "")])
    c = h.get("certified")
    if c:
        stat_tiles([("Proven minimum (GNN + RL → exact)", f"{100 * c['proven_fraction']:.1f}%",
                     f"of {c['instances_x_seeds']} runs (test instances × 2 seeds)", "hero"),
                    ("GNN + RL alone optimal", f"{100 * c['rl_alone_optimal_fraction']:.1f}%",
                     "the policy's own set was the proven minimum", "teal")])
    names = list(s["tables"])
    for tab, k in zip(st.tabs([s["captions"].get(k, k).split(":")[0] for k in names]), names):
        with tab:
            rows = s["tables"][k]
            if rows:
                cols = list(rows[0].keys())
                table(rows, cols, numeric=[c for c in cols if c not in ("family", "graph", "lattice", "weights", "method")])
            st.markdown(f'<div class="caption">{s["captions"].get(k, "")}</div>', unsafe_allow_html=True)
            if k == "W1":
                for f in ("plot_gap_vs_n.png", "plot_runtime_vs_n.png"):
                    if (FINAL / f).exists():
                        st.image(str(FINAL / f), width="stretch")
