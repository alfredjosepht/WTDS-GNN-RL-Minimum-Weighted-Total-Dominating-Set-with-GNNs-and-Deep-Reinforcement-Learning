"""Weighted experiments W1-W4.

    python scripts/run_final.py sets      # generate the fixed test graphs (data/test/)
    python scripts/run_final.py ilp       # exact weighted ILP references (20 s), 3 parallel workers
    python scripts/run_final.py timed     # greedy / RL rows, sequential (run on an idle machine)
    python scripts/run_final.py report    # results/summary.json, CSV/LaTeX tables, plots

W1: ER (avg deg 8) and BA (m = 4), n in {20, 50, 100, 200}, 15 graphs per cell, weights uniform 1-100.
W2: n = 100, ER and BA, each weight distribution, 10 graphs each.
W3: grid and triangular lattices, n ~ 100 and ~ 400, weights uniform 1-100.
W4: one BA graph (m = 3), n = 5,000, weights uniform 1-100.
Every solution is verified with is_total_dominating_set; W(S) is recomputed from S.
"""
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wtds.baselines import greedy, greedy_rr, greedy_rr_ls, ilp, unweighted_greedy_rr  # noqa: E402
from wtds.checker import is_total_dominating_set  # noqa: E402
from wtds.generators import lattice_dims, make_graph  # noqa: E402
from wtds.io import load_graphs, save_graphs  # noqa: E402

OUT = ROOT / "results" / "final"
TEST = ROOT / "data" / "test"
OURS = {0: ROOT / "checkpoints/best.pt", 1: ROOT / "checkpoints/seed1_best.pt"}
OLD = {0: ROOT / "checkpoints/pretrained/tds_seed0_best.pt", 1: ROOT / "checkpoints/pretrained/tds_seed1_best.pt"}
LS_TIME, ILP_TIME = 1.0, 20.0
BASE = 6_000_000
TEAL, AMBER, GREY, INK, MUTED = "#0F766E", "#D97706", "#9CA3AF", "#1F2328", "#6B6F76"


def seed_for(*p):
    return BASE + zlib.crc32("|".join(map(str, p)).encode()) % 1_000_000_000


def stage_sets():
    for fam, par in (("er", dict(family="er", avg_deg=8)), ("ba", dict(family="ba", m=4))):
        for n in (20, 50, 100, 200):
            save_graphs([make_graph({**par, "n": n, "seed": seed_for("W1", fam, n, i), "weights": "uniform_int"})
                         for i in range(15)], TEST / "W1" / f"{fam}_n{n}.npz")
        for d in ("uniform_small", "uniform_int", "degree_correlated", "unit"):
            save_graphs([make_graph({**par, "n": 100, "seed": seed_for("W2", fam, d, i), "weights": d})
                         for i in range(10)], TEST / "W2" / f"{fam}_{d}.npz")
    lat = []
    for kind in ("grid", "tri"):
        for t in (100, 400):
            a, b = lattice_dims(kind, t)
            lat.append(make_graph(dict(family=kind, rows=a, cols=b, seed=seed_for("W3", kind, t),
                                       weights="uniform_int")))
    save_graphs(lat, TEST / "W3" / "lattices.npz")
    save_graphs([make_graph(dict(family="ba", n=5000, m=3, seed=seed_for("W4", 0), weights="uniform_int"))],
                TEST / "W4" / "ba_n5000.npz")
    print("test sets written", flush=True)


def instances():
    out = []
    for exp in ("W1", "W2", "W3", "W4"):
        for p in sorted((TEST / exp).glob("*.npz")):
            for i, g in enumerate(load_graphs(p)):
                out.append((exp, p.stem, i, g))
    return out


def _ilp_job(a):
    exp, s, i, g = a
    r = ilp(g, time_limit=ILP_TIME, threads=4)
    assert is_total_dominating_set(g.adj, r.vertices)
    return dict(exp=exp, set=s, idx=i, n=g.n, opt_w=float(g.w[r.vertices].sum()), optimal=bool(r.info["optimal"]),
                bound=float(r.info["bound"]), runtime=r.runtime, vertices=[int(v) for v in r.vertices])


def stage_ilp():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [x for x in instances() if x[0] != "W4"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(_ilp_job, jobs))
    with open(OUT / "solutions_ilp.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("exp", "set", "idx", "vertices")}) + "\n")
    pd.DataFrame([{k: v for k, v in r.items() if k != "vertices"} for r in rows]).to_csv(OUT / "ilp.csv", index=False)
    print(f"ILP: {len(rows)} instances, {sum(r['optimal'] for r in rows)} proven, {time.time() - t0:.0f}s", flush=True)


def stage_timed():
    import torch
    from wtds.evaluate import rl_rows
    from wtds.solve import get_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ck in list(OURS.values()) + list(OLD.values()):
        get_model(str(ck), device)                       # load once, outside the timings
    rows, t0 = [], time.time()
    sol = open(OUT / "solutions_timed.jsonl", "w")

    def rec(exp, s, i, g, method, seed, S, rt):
        S = [int(v) for v in S]
        ok = is_total_dominating_set(g.adj, S)
        assert ok, (exp, s, i, method, seed)
        rows.append(dict(exp=exp, set=s, idx=i, n=g.n, method=method, seed=seed, weight=float(g.w[S].sum()),
                         size=len(S), runtime=rt, valid=ok))
        sol.write(json.dumps(dict(exp=exp, set=s, idx=i, method=method, seed=seed, vertices=S)) + "\n")

    for k, (exp, s, i, g) in enumerate(instances()):
        big = exp == "W4"
        rec(exp, s, i, g, "weighted_greedy", None, *(lambda r: (r.vertices, r.runtime))(greedy(g)))
        rec(exp, s, i, g, "weighted_greedy_rr", None, *(lambda r: (r.vertices, r.runtime))(greedy_rr(g)))
        if exp in ("W1", "W3"):
            r = greedy_rr_ls(g, ls_time=LS_TIME)
            rec(exp, s, i, g, "weighted_greedy_rr_ls", None, r.vertices, r.runtime)
            r = unweighted_greedy_rr(g)
            rec(exp, s, i, g, "unweighted_greedy_rr", None, r.vertices, r.runtime)
        for seed in (0, 1):
            mc = {"ls_time": 0 if (big or exp == "W2") else LS_TIME, "multi_select": "auto"}
            for name, S, rt, _ in rl_rows(g, str(OURS[seed]), "ours", seed, mc, device):
                if name != "rl":
                    rec(exp, s, i, g, name, seed, S, rt)
            if not big:                         # best of greedy rollout + 16 sampled rollouts (T tuned on validation)
                from wtds.solve import solve_rl
                r = solve_rl(g, str(OURS[seed]), rr=True, ls_time=0, samples=16, seed=seed, device=device)
                rec(exp, s, i, g, "rl_rr_k16", seed, r.vertices, r.runtime)
                if exp != "W2":
                    r = solve_rl(g, str(OURS[seed]), rr=True, ls_time=LS_TIME, samples=16, seed=seed, device=device)
                    rec(exp, s, i, g, "rl_rr_ls_k16", seed, r.vertices, r.runtime)
                for name, S, rt, _ in rl_rows(g, str(OLD[seed]), "old", seed, {"ls_time": 0, "multi_select": "auto"},
                                              device):
                    if name == "rl_rr":
                        rec(exp, s, i, g, "old_tds_model_rr", seed, S, rt)
        if (k + 1) % 40 == 0:
            print(f"  {k + 1} instances ({time.time() - t0:.0f}s)", flush=True)
    sol.close()
    pd.DataFrame(rows).to_csv(OUT / "timed.csv", index=False)
    print(f"timed stage: {len(rows)} rows, all valid, {time.time() - t0:.0f}s", flush=True)


def stage_exact():
    """GNN + RL certified by the exact solver (warm-started with the GNN+RL answer), both seeds."""
    import torch
    from wtds.solve import get_model, solve_rl_exact
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ck in OURS.values():
        get_model(str(ck), device)
    rows, t0 = [], time.time()
    with open(OUT / "solutions_exact.jsonl", "w") as sol:
        for k, (exp, s, i, g) in enumerate(instances()):
            for seed, ck in OURS.items():
                r = solve_rl_exact(g, str(ck), samples=16, ls_time=LS_TIME, ilp_time=60.0 if exp == "W4" else ILP_TIME,
                                   seed=seed, device=device)
                assert is_total_dominating_set(g.adj, r.vertices)
                rows.append(dict(exp=exp, set=s, idx=i, n=g.n, method="rl_exact", seed=seed,
                                 weight=float(g.w[r.vertices].sum()), size=r.size, runtime=r.runtime, valid=True,
                                 proven=r.info["optimal"], rl_weight=r.info["rl_weight"],
                                 rl_was_optimal=r.info["rl_was_optimal"], improved_by_ilp=r.info["improved_by_ilp"],
                                 lower_bound=r.info["lower_bound"], rl_time=r.info["rl_time"],
                                 cert_time=r.info["ilp_time_used"]))
                sol.write(json.dumps(dict(exp=exp, set=s, idx=i, method="rl_exact", seed=seed,
                                          vertices=[int(v) for v in r.vertices])) + "\n")
            if (k + 1) % 20 == 0:
                print(f"  {k + 1} instances ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "exact.csv", index=False)
    print(f"exact stage: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ report
PRETTY = {"weighted_greedy": "W-greedy", "weighted_greedy_rr": "W-greedy+RR", "weighted_greedy_rr_ls": "W-greedy+RR+LS",
          "unweighted_greedy_rr": "U-greedy+RR", "old_tds_model_rr": "Old TDS model+RR", "rl_rr": "Ours+RR",
          "rl_rr_ls": "Ours+RR+LS", "rl_rr_k16": "Ours+RR k16", "rl_rr_ls_k16": "Ours+RR+LS k16"}


def agg(d, col):
    if d.seed.notna().any():
        per = d.groupby("seed")[col].mean()
        return float(per.mean()), (float(per.std(ddof=1)) if len(per) > 1 else float("nan"))
    return float(d[col].mean()), float("nan")


def fmt(m, s, pct=False, k=2):
    if m is None or np.isnan(m):
        return "--"
    f = 100 if pct else 1
    return f"{f * m:.{k}f}" + ("" if np.isnan(s) else f" ± {f * s:.{k}f}")


def seed_overlap():
    from wtds.config import load_config
    from wtds.generators import GraphSampler
    cfg = load_config(ROOT / "configs/default.yaml")
    train = set()
    for s in (0, 1):
        ep = int(pd.read_csv(ROOT / f"runs/wtds_seed{s}/val.csv").episodes.max())
        smp = GraphSampler(seed=1_000_003 * (s + 1), families=cfg["train"]["families"], weighted=True)
        for _ in range(2 * ep + 1000):           # 2x margin over the episodes actually drawn
            train.add(smp.sample_spec(20, 100)["seed"])
    val = {sp["seed"] for sp in json.loads((ROOT / "data/val_set.json").read_text())["specs"]}
    test = {g.meta["seed"] for p in TEST.glob("*/*.npz") for g in load_graphs(p) if g.meta.get("seed") is not None}
    r = {"n_train_seeds_checked": len(train), "n_val_seeds": len(val), "n_test_seeds": len(test),
         "train_val_overlap": len(train & val), "train_test_overlap": len(train & test),
         "val_test_overlap": len(val & test)}
    print("seed overlap:", r, flush=True)
    return r


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.read_csv(OUT / "timed.csv")
    il = pd.read_csv(OUT / "ilp.csv")
    t = t.merge(il[["exp", "set", "idx", "opt_w", "optimal"]], on=["exp", "set", "idx"], how="left")
    t["gap"] = np.where(t.optimal == True, (t.weight - t.opt_w) / t.opt_w, np.nan)  # noqa: E712
    assert t.valid.all()
    tables, captions = {}, {}
    # W1
    w1 = t[t.exp == "W1"]
    methods = list(PRETTY)
    rows, plot = [], []
    for s in sorted(w1.set.unique(), key=lambda x: (x.split("_")[0], int(x.split("n")[-1]))):
        d = w1[w1.set == s]
        ils = il[(il.exp == "W1") & (il.set == s)]
        r = {"family": s.split("_")[0].upper(), "n": int(d.n.iloc[0]), "ILP proven": f"{int(ils.optimal.sum())}/{len(ils)}"}
        for m in methods:
            dm = d[(d.method == m)]
            gm, gs = agg(dm[dm.optimal == True], "gap")  # noqa: E712
            tm, ts = agg(dm, "runtime")
            r[PRETTY[m] + " gap %"] = fmt(gm, gs, pct=True)
            plot.append(dict(family=r["family"], n=r["n"], method=m, gap=gm, gsd=gs, rt=tm, rsd=ts))
        rows.append(r)
    tables["W1"] = rows
    captions["W1"] = (f"W1: random graphs, weights uniform 1-100, 15 graphs per cell. Mean weight gap to the ILP optimum "
                      f"(%), only on instances the ILP (limit {ILP_TIME:.0f} s) proved optimal. Ours / old model: mean ± std "
                      f"over 2 seeds. LS limit {LS_TIME:.0f} s for greedy and ours.")
    # W2
    w2 = t[t.exp == "W2"]
    rows = []
    for s in sorted(w2.set.unique()):
        d = w2[w2.set == s]
        ils = il[(il.exp == "W2") & (il.set == s)]
        fam, dist = s.split("_", 1)
        r = {"family": fam.upper(), "weights": dist, "ILP proven": f"{int(ils.optimal.sum())}/{len(ils)}"}
        for m in ("weighted_greedy_rr", "old_tds_model_rr", "rl_rr", "rl_rr_k16"):
            dm = d[d.method == m]
            gm, gs = agg(dm[dm.optimal == True], "gap")  # noqa: E712
            r[PRETTY[m] + " gap %"] = fmt(gm, gs, pct=True)
        rows.append(r)
    tables["W2"] = rows
    captions["W2"] = ("W2: n = 100, each weight distribution, 10 graphs each. Weight gap to the proven ILP optimum (%). "
                      "k16 = best of the greedy rollout and 16 sampled rollouts (temperature chosen on validation).")
    # W3
    w3 = t[t.exp == "W3"]
    rows = []
    for i, d in w3.groupby("idx"):
        ilr = il[(il.exp == "W3") & (il.idx == i)].iloc[0]
        r = {"lattice": f"{['grid', 'grid', 'tri', 'tri'][i]}", "n": int(d.n.iloc[0]),
             "ILP W*": f"{ilr.opt_w:.0f}" + ("" if ilr.optimal else f" (not proven, bound {ilr.bound:.0f})")}
        for m in methods:
            mm, ms = agg(d[d.method == m], "weight")
            r[PRETTY[m]] = fmt(mm, ms, k=1)
        rows.append(r)
    tables["W3"] = rows
    captions["W3"] = "W3: lattices with weights uniform 1-100 (zero-shot sizes). Total weight W(S); ours: mean ± std over seeds."
    # W4
    w4 = t[t.exp == "W4"]
    rows = []
    for m in ("weighted_greedy", "weighted_greedy_rr", "rl_rr"):
        d = w4[w4.method == m]
        wm, wsd = agg(d, "weight")
        tm, tsd = agg(d, "runtime")
        sm, ssd = agg(d, "size")
        rows.append({"method": PRETTY[m], "W(S)": fmt(wm, wsd, k=1), "|S|": fmt(sm, ssd, k=1), "time s": fmt(tm, tsd, k=2)})
    tables["W4"] = rows
    captions["W4"] = "W4: one Barabási–Albert graph, n = 5,000, m = 3, weights uniform 1-100."
    # certified GNN + RL (exact stage)
    ex = pd.read_csv(OUT / "exact.csv") if (OUT / "exact.csv").exists() else None
    if ex is not None:
        ex = ex.merge(il[["exp", "set", "idx", "opt_w", "optimal"]], on=["exp", "set", "idx"], how="left")
        rows = []
        for exp, d in ex.groupby("exp"):
            ref_ok = d[d.optimal == True]  # noqa: E712
            agrees = (abs(ref_ok.weight - ref_ok.opt_w) < 1e-6).mean() if len(ref_ok) else float("nan")
            rows.append({"experiment": exp, "instances x seeds": len(d),
                         "proven minimum": f"{100 * d.proven.mean():.1f}%",
                         "GNN+RL alone already optimal": f"{100 * d.rl_was_optimal.mean():.1f}%",
                         "equals reference optimum": "--" if np.isnan(agrees) else f"{100 * agrees:.1f}%",
                         "GNN+RL time s": f"{d.rl_time.mean():.2f}", "certification time s": f"{d.cert_time.mean():.2f}"})
        tables["CERT"] = rows
        captions["CERT"] = ("Certified: GNN + RL search (greedy + 16 sampled rollouts, RR, LS) whose answer warm-starts "
                            "the exact CP-SAT solver, which proves it minimal or improves it (limit 20 s; 60 s for W4). "
                            "'GNN+RL alone already optimal' = the policy's own set was proven minimum. Both seeds.")
    # headline (W1, proven instances)
    il1 = il[il.exp == "W1"]
    pr = w1[w1.optimal == True]  # noqa: E712
    head = {}
    for m in methods:
        gm, gs = agg(pr[pr.method == m], "gap")
        head[m] = {"gap_mean": gm, "gap_std_across_seeds": None if np.isnan(gs) else gs}
    n_sol = sum(1 for f in OUT.glob("solutions_*.jsonl") for _ in open(f))
    n_ex = 0 if ex is None else len(ex)
    summary = {"validity": {"solutions_checked": int(len(t) + len(il) + n_ex), "valid_fraction": 1.0,
                            "solutions_in_jsonl": n_sol},
               "settings": {"ls_time_s": LS_TIME, "ilp_time_limit_s": ILP_TIME, "seeds": [0, 1]},
               "headline": {"instances": int(len(il1)), "ilp_proven": int(il1.optimal.sum()),
                            "ilp_not_proven": int((~il1.optimal.astype(bool)).sum()), "gap_on_proven": head,
                            "certified": None if ex is None else {
                                "proven_fraction": float(ex.proven.mean()),
                                "rl_alone_optimal_fraction": float(ex.rl_was_optimal.mean()),
                                "instances_x_seeds": int(len(ex))}},
               "tables": tables, "captions": captions, "seed_overlap": seed_overlap()}
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for k, rws in tables.items():
        df = pd.DataFrame(rws)
        df.to_csv(OUT / f"table_{k}.csv", index=False)
        (OUT / f"table_{k}.tex").write_text(df.to_latex(index=False, escape=True), encoding="utf-8")
    # plots
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#C9C4BA",
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
    pe = pd.DataFrame(plot)
    style = {"weighted_greedy_rr": (GREY, "o", "-"), "weighted_greedy_rr_ls": (MUTED, "s", "--"),
             "rl_rr_k16": (TEAL, "o", "-"), "rl_rr_ls_k16": (AMBER, "s", "--")}
    for key, ylab, fname in (("gap", "weight gap to ILP optimum (%)", "plot_gap_vs_n.png"),
                             ("rt", "runtime (s)", "plot_runtime_vs_n.png")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, fam in zip(axes, ("ER", "BA")):
            for m, (c, mk, ls) in style.items():
                d = pe[(pe.family == fam) & (pe.method == m)].sort_values("n")
                f = 100 if key == "gap" else 1
                ax.errorbar(d.n, d[key] * f, yerr=d[key[0] + "sd" if key == "gap" else "rsd"].fillna(0) * f,
                            color=c, marker=mk, ls=ls, lw=1.8, ms=5, capsize=3, label=PRETTY[m])
            ax.set_xscale("log")
            ax.set_xticks([20, 50, 100, 200])
            ax.set_xticklabels(["20", "50", "100", "200"])
            if key == "rt":
                ax.set_yscale("log")
            ax.set_title({"ER": "Erdős–Rényi, avg. degree 8", "BA": "Barabási–Albert, m = 4"}[fam], loc="left",
                         fontsize=11, color=INK)
            ax.set_xlabel("vertices n")
            ax.grid(alpha=0.25, color="#C9C4BA")
        axes[0].set_ylabel(ylab)
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=150)
        plt.close(fig)
    print("report written", flush=True)


if __name__ == "__main__":
    {"sets": stage_sets, "ilp": stage_ilp, "timed": stage_timed, "exact": stage_exact,
     "report": report}[sys.argv[1]]()
