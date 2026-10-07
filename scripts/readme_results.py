"""Write the README 'Results' section from results/summary.json and checkpoints/model_info.json.
No number is typed by hand."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"weighted_greedy": "Weighted greedy", "weighted_greedy_rr": "Weighted greedy + RR",
         "weighted_greedy_rr_ls": "Weighted greedy + RR + LS", "unweighted_greedy_rr": "Unweighted greedy + RR",
         "old_tds_model_rr": "Old TDS model (unit reward) + RR", "rl_rr": "**Ours (weighted reward) + RR**",
         "rl_rr_ls": "**Ours + RR + LS**", "rl_rr_k16": "**Ours + RR, best of 1+16 rollouts**",
         "rl_rr_ls_k16": "**Ours + RR + LS, best of 1+16 rollouts**"}


def md(rows):
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])


def pct(m, s=None):
    return "--" if m is None else f"{100 * m:.2f}%" + ("" if s is None else f" ± {100 * s:.2f}")


def main():
    info = json.loads((ROOT / "checkpoints/model_info.json").read_text())
    s = json.loads((ROOT / "results/summary.json").read_text())
    L = [f"Ours and the old model: mean ± std over the 2 seeds. Greedy methods are deterministic. "
         f"**{s['validity']['solutions_checked']:,} reported solutions were checked with `is_total_dominating_set`: "
         f"{100 * s['validity']['valid_fraction']:.0f}% valid**, and every weight was recomputed from the solution."]
    L.append("\n### Fine-tuning\n")
    L.append(md([{"seed": x["seed"], "fine-tuning steps": f"{x['finetune_steps']:,}",
                  "val. gap before (old model + RR)": pct(x["val_gap_rr_step0"]),
                  "best val. gap (ours + RR)": pct(x["best_val_gap_rr"]), "at step": f"{x['best_step']:,}",
                  "val. gap, weighted greedy + RR": pct(x["val_gap_weighted_greedy_rr"])} for x in info["seeds"]]))
    L.append(f"\n{info['encoder'].upper()}, {info['layers']} layers, hidden {info['hidden']}, {info['parameters']:,} "
             f"parameters. {info['validation_note']}")
    h = s["headline"]
    L.append(f"\n### Headline (W1: ER + BA, n = 20–200, weights 1–100)\n\nMean weight gap to the proven optimum on the "
             f"{h['ilp_proven']} of {h['instances']} W1 instances where the ILP proved optimality "
             f"({h['ilp_not_proven']} not proven, excluded):\n")
    L.append(md([{"method": NAMES[k], "mean weight gap": pct(v["gap_mean"], v["gap_std_across_seeds"])}
                 for k, v in h["gap_on_proven"].items()]))
    c = h.get("certified")
    if c:
        L.append(f"\n### Finding the proven minimum\n\nWith exact certification (`method=\"rl_exact\"`, the app's "
                 f"default), the returned set was **proven to be the minimum-weight TDS in "
                 f"{100 * c['proven_fraction']:.1f}%** of {c['instances_x_seeds']} runs (all test instances × 2 seeds); "
                 f"in {100 * c['rl_alone_optimal_fraction']:.1f}% the GNN + RL policy's own set was already that "
                 f"minimum. See the 'Certified' table below.")
    for k in [x for x in ("CERT", "W1", "W2", "W3", "W4") if x in s["tables"]]:
        L.append(f"\n### {s['captions'][k]}\n")
        L.append(md(s["tables"][k]))
        if k == "W1":
            L.append("\n![weight gap vs n](results/final/plot_gap_vs_n.png)\n\n"
                     "![runtime vs n](results/final/plot_runtime_vs_n.png)")
    ov = s["seed_overlap"]
    L.append(f"\n**Data separation:** {ov['n_train_seeds_checked']:,} training-graph seeds checked; "
             f"{ov['n_val_seeds']} validation seeds; {ov['n_test_seeds']} test seeds. Overlaps: train–val "
             f"{ov['train_val_overlap']}, train–test {ov['train_test_overlap']}, val–test {ov['val_test_overlap']}.")
    p = ROOT / "README.md"
    txt = p.read_text(encoding="utf-8")
    txt = re.sub(r"(<!-- RESULTS:BEGIN -->)(.*?)(<!-- RESULTS:END -->)",
                 lambda m: m.group(1) + "\n" + "\n".join(L) + "\n" + m.group(3), txt, flags=re.S)
    p.write_text(txt, encoding="utf-8")
    print("README results written")


if __name__ == "__main__":
    main()
