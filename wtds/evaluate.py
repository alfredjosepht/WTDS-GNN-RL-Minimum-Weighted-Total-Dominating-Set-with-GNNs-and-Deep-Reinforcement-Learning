"""Experiment driver.

    python -m wtds.evaluate --config configs/eval_e1.yaml [--shard 0 --num-shards 3]

The config lists test sets (glob patterns of .npz files written by
scripts/make_test_sets.py) and methods. One CSV row per (graph, method, label,
seed); rows already present are skipped, so runs can be resumed or sharded.
Every solution is verified by the checker before it is written.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .baselines import greedy, greedy_rr, greedy_rr_ls, ilp
from .checker import assert_tds
from .io import load_graphs
from .postprocess import local_search, redundancy_removal

COLUMNS = ["set", "idx", "name", "family", "n", "m", "method", "label", "seed", "size", "weight",
           "runtime", "valid", "optimal", "bound", "info"]


def _key(row):
    return (row["set"], int(row["idx"]), row["method"], row["label"], str(row["seed"]))


def family_of(g):
    return g.meta.get("family", "file")


def rl_rows(g, ck, label, seed, mcfg, device):
    """One rollout -> rows for rl (raw), rl_rr and rl_rr_ls (and sampled variants)."""
    from .rollout import rollout
    from .solve import get_model
    model = get_model(ck, device)
    ek = dict(getattr(model, "env_kwargs", {}) or {})
    ek.update(mcfg.get("env", {}) or {})
    rows = []
    k = int(mcfg.get("samples", 1))
    t0 = time.perf_counter()
    if k <= 1:
        runs = rollout(model, [g], device, env_kwargs=ek, multi_select=mcfg.get("multi_select", "auto"))
        tag = ""
    else:
        runs = rollout(model, [g] * k, device, env_kwargs=ek, mode="sample",
                       temperature=float(mcfg.get("temperature", 1.0)),
                       rng=np.random.default_rng(int(mcfg.get("sample_seed", 0))), batch_size=min(k, 16))
        tag = f"_k{k}"
    t_pol = time.perf_counter() - t0
    best_raw = min(runs, key=lambda r: g.w[r["solution"]].sum())
    rows.append(("rl" + tag, best_raw["solution"], t_pol, {"steps": best_raw["steps"]}))
    if mcfg.get("rr", True):
        t1 = time.perf_counter()
        cands = [redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"]) for r in runs]
        S_rr = min(cands, key=lambda S: g.w[S].sum())
        t_rr = t_pol + time.perf_counter() - t1
        rows.append(("rl_rr" + tag, S_rr, t_rr, {}))
        ls_time = float(mcfg.get("ls_time", 0) or 0)
        if ls_time > 0:
            t2 = time.perf_counter()
            S_ls = local_search(g, S_rr, time_limit=ls_time, seed=seed if isinstance(seed, int) else 0)
            rows.append(("rl_rr_ls" + tag, S_ls, t_rr + time.perf_counter() - t2, {"ls_time": ls_time}))
    return rows


def graphs_limit(cfg: dict, graphs) -> int:
    """Number of graphs of a set to evaluate. `max_graphs_large: {min_n, count}` caps
    sets whose graphs have >= min_n vertices (used by E1 to bound ILP time)."""
    lim = cfg.get("max_graphs_per_set")
    large = cfg.get("max_graphs_large")
    if large and graphs and graphs[0].n >= large["min_n"]:
        lim = min(lim or large["count"], large["count"])
    return lim


def run(cfg: dict, shard: int = 0, num_shards: int = 1, device=None, limit_methods=None):
    out = Path(cfg["out"])
    if num_shards > 1:
        out = out.with_name(f"{out.stem}.shard{shard}of{num_shards}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {_key(r) for r in csv.DictReader(f)}
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLUMNS)
    if not done and out.stat().st_size == 0:
        w.writeheader()
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = sorted({p for pat in cfg["sets"] for p in glob.glob(pat)})
    if not paths:
        raise FileNotFoundError(f"no test sets match {cfg['sets']}; run scripts/make_test_sets.py")
    methods = cfg["methods"]
    if limit_methods:
        methods = [m for m in methods if m.get("label", m["type"]) in limit_methods or m["type"] in limit_methods]
    t_start, n_rows = time.time(), 0
    for path in paths:
        set_name = str(Path(path).relative_to(Path(path).parents[1])).replace("\\", "/")
        graphs = load_graphs(path)
        max_n = graphs_limit(cfg, graphs)
        for idx, g in enumerate(graphs[:max_n] if max_n else graphs):
            if idx % num_shards != shard:
                continue
            base = dict(set=set_name, idx=idx, name=g.name, family=family_of(g), n=g.n, m=g.num_edges)
            for mc in methods:
                typ, label = mc["type"], mc.get("label", mc["type"])
                if mc.get("max_n") and g.n > mc["max_n"]:
                    continue
                todo = []
                if typ == "rl":
                    for si, ck in enumerate(mc["checkpoints"]):
                        seed = mc.get("seeds", list(range(len(mc["checkpoints"]))))[si]
                        names = ["rl", "rl_rr", "rl_rr_ls"]
                        k = int(mc.get("samples", 1))
                        if k > 1:
                            names = [x + f"_k{k}" for x in names]
                        if all(_key({**base, "method": nm, "label": label, "seed": seed}) in done
                               for nm in names[: (1 + bool(mc.get("rr", True)) + bool(mc.get("ls_time")))]):
                            continue
                        if not Path(ck).exists():
                            print(f"  skip {label}: missing {ck}")
                            continue
                        for nm, S, rt, info in rl_rows(g, ck, label, seed, mc, device):
                            todo.append((nm, seed, S, rt, info, None, None))
                else:
                    seed = mc.get("seed", 0)
                    if _key({**base, "method": typ, "label": label, "seed": seed}) in done:
                        continue
                    if typ == "greedy":
                        r = greedy(g)
                    elif typ == "greedy_rr":
                        r = greedy_rr(g)
                    elif typ == "greedy_rr_ls":
                        r = greedy_rr_ls(g, ls_time=float(mc.get("ls_time", 1.0)), seed=seed)
                    elif typ == "ilp":
                        r = ilp(g, time_limit=float(mc.get("time_limit", 60)), backend=mc.get("backend", "cpsat"),
                                threads=int(mc.get("threads", 8)))
                    else:
                        raise ValueError(typ)
                    todo.append((typ, seed, r.vertices, r.runtime, {k: v for k, v in r.info.items()
                                 if k in ("status", "gap", "time_limit")}, r.info.get("optimal"), r.info.get("bound")))
                for nm, seed, S, rt, info, opt, bound in todo:
                    assert_tds(g, S, f"{label}/{nm} on {set_name}#{idx}")
                    w.writerow({**base, "method": nm, "label": label, "seed": seed, "size": len(S),
                                "weight": float(g.w[S].sum()), "runtime": f"{rt:.6f}", "valid": True,
                                "optimal": "" if opt is None else bool(opt),
                                "bound": "" if bound is None else bound, "info": json.dumps(info)})
                    n_rows += 1
            fh.flush()
        print(f"[{cfg.get('name', '')} shard {shard}/{num_shards}] {set_name} done "
              f"({n_rows} rows, {time.time() - t_start:.0f}s)", flush=True)
    fh.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--methods", default=None, help="comma-separated labels/types to run (default all)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    run(cfg, a.shard, a.num_shards, a.device, a.methods.split(",") if a.methods else None)


if __name__ == "__main__":
    main()
