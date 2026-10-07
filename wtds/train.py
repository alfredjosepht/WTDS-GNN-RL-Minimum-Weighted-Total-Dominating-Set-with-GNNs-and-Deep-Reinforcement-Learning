"""Double DQN training.

    python -m wtds.train --config configs/default.yaml --seed 0
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .agent import DDQNAgent
from .config import load_config, resolve_device
from .env import TDSEnv
from .generators import GraphSampler
from .gpu_batch import GPUGraphCache, build_batches
from .reductions import forced_mask
from .replay import GraphPool, ReplayBuffer
from .rollout import q_values
from .utils.seeding import seed_everything
from .validation import evaluate_policy, get_validation_set


def epsilon(step, cfg):
    t = cfg["train"]
    frac = min(1.0, step / max(1, t["eps_frac"] * t["total_steps"]))
    return t["eps_start"] + frac * (t["eps_end"] - t["eps_start"])


def n_range(step, cfg):
    t = cfg["train"]
    for until, lo, hi in t["curriculum"]:
        if step < until * t["total_steps"]:
            return lo, hi
    return t["curriculum"][-1][1:]


class Slot:
    """One running episode with its n-step buffer."""

    def __init__(self, env, gid):
        self.env, self.gid = env, gid
        self.pending = collections.deque()   # (S_t, a_t, r_t)
        self.ret = 0.0


def _trim_csv(path: Path, max_step: int) -> None:
    """Keep the header and rows with step <= max_step (rows logged after the resume point)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    keep = rows[:1] + [r for r in rows[1:] if r and int(float(r[0])) <= max_step]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(keep)


def resume_path_exists(cfg, seed) -> bool:
    return (Path(cfg["out"]["run_dir"]) / f'{cfg["out"]["name"]}_seed{seed}' / "resume.pt").exists()


def train(cfg: dict, seed: int, resume: bool = False) -> dict:
    """Train one model. With resume=True, continue from runs/<name>/resume.pt if present
    (saved at every validation). Episodes in progress at the save point are dropped."""
    import random
    seed_everything(seed)
    device = resolve_device(cfg["device"])
    t, ec = cfg["train"], cfg["env"]
    name = f'{cfg["out"]["name"]}_seed{seed}'
    ckpt_dir = Path(cfg["out"]["ckpt_dir"]) / name
    run_dir = Path(cfg["out"]["run_dir"]) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**cfg, "seed": seed, "device": device}, indent=2))

    v = cfg["val"]
    val_graphs, val_info = get_validation_set(v["path"], v["count"], v["n_min"], v["n_max"],
                                              ilp_time=v["ilp_time"])
    opt = val_info["opt"]
    o = np.asarray(opt, float)
    ref = {"greedy_rr_gap": float(np.mean((np.asarray(val_info["greedy_rr"]) - o) / o)),
           "random_rr_gap": float(np.mean((np.asarray(val_info["random_rr"]) - o) / o)),
           "ilp_proven_frac": float(np.mean(val_info["proven"]))}
    print(f"[{name}] device={device} val: greedy_rr gap {ref['greedy_rr_gap']:.4f}, "
          f"random+RR gap {ref['random_rr_gap']:.4f}, ILP proven {ref['ilp_proven_frac']:.2%}", flush=True)

    agent = DDQNAgent(cfg["model"], lr=t["lr"], gamma=t["gamma"], tau=t["tau"],
                      target_update=t["target_update"], hard_every=t["hard_every"], loss=t["loss"],
                      grad_clip=t["grad_clip"], device=device)
    init = t.get("init_checkpoint")
    if init and not (resume and resume_path_exists(cfg, seed)):
        init = str(init).format(seed=seed)
        ck = torch.load(init, map_location=device, weights_only=False)
        res = agent.online.load_state_dict(ck["online"], strict=True)       # raises on missing/unexpected keys
        agent.target.load_state_dict(ck["online"], strict=True)
        print(f"[{name}] initialised from {init}: missing keys {list(res.missing_keys)}, "
              f"unexpected keys {list(res.unexpected_keys)}", flush=True)
    pool = GraphPool(use_forced=ec["use_forced"])
    buf = ReplayBuffer(t["buffer_size"], pool, seed=seed)
    sampler = GraphSampler(seed=1_000_003 * (seed + 1), families=t["families"], weighted=t["weighted"])
    rng = np.random.default_rng(seed)
    env_kwargs = dict(ec)

    resume_path = run_dir / "resume.pt"
    state = None
    if resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        print(f"[{name}] resuming from step {state['step']}", flush=True)

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(str(run_dir), purge_step=state["step"] + 1 if state else None)
    if state:
        _trim_csv(run_dir / "val.csv", state["step"])
        _trim_csv(run_dir / "train.csv", state["step"])
    val_csv = open(run_dir / "val.csv", "a" if state else "w", newline="")
    vw = csv.writer(val_csv)
    tr_csv = open(run_dir / "train.csv", "a" if state else "w", newline="")
    tw = csv.writer(tr_csv)
    if not state:
        vw.writerow(["step", "episodes", "updates", "epsilon", "loss", "gap_raw", "gap_rr", "size_rr",
                     "frac_opt_rr", "greedy_rr_gap", "random_rr_gap", "wall_s"])
        tw.writerow(["step", "episodes", "epsilon", "loss", "q_mean", "y_mean", "ep_return", "ep_len",
                     "n_graphs_pool", "wall_s"])

    gamma, nstep = t["gamma"], t["n_step"]

    def new_slot():
        while True:
            lo, hi = n_range(step, cfg)
            g = sampler.sample(lo, hi)
            env = TDSEnv(g, **env_kwargs)
            if not env.done:                  # skip graphs solved by the forced rule alone
                return Slot(env, pool.add(g))

    def emit(slot, final):
        """Turn the oldest pending step(s) into n-step transitions."""
        S_now = list(slot.env.order)
        while slot.pending and (final or len(slot.pending) >= nstep):
            R, gk = 0.0, 1.0
            for (_, _, r) in slot.pending:
                R += gk * r
                gk *= gamma
            S_t, a_t, _ = slot.pending[0]
            buf.add(slot.gid, S_t, a_t, R, S_now, slot.env.done, gk)
            slot.pending.popleft()
            if not final:
                break

    step, episodes, t0 = 0, 0, time.time()
    losses, stats, ep_returns, ep_lens = [], collections.defaultdict(list), [], []
    best_gap, history = float("inf"), []
    pending_updates = 0.0
    if state:
        from .generators import make_graph
        step, episodes, best_gap = state["step"], state["episodes"], state["best_gap"]
        history, pending_updates = state["history"], state["pending_updates"]
        losses, ep_returns, ep_lens = state["losses"], state["ep_returns"], state["ep_lens"]
        for k, val in state["stats"].items():
            stats[k] = val
        t0 = time.time() - state["wall_s"]
        agent.online.load_state_dict(state["agent"]["online"])
        agent.target.load_state_dict(state["agent"]["target"])
        agent.opt.load_state_dict(state["agent"]["opt"])
        agent.updates = state["agent"]["updates"]
        rng.bit_generator.state = state["rng"]
        sampler.rng.bit_generator.state = state["sampler_rng"]
        buf.rng.bit_generator.state = state["buf_rng"]
        random.setstate(state["py_random"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        b = state["buffer"]
        buf.data, buf.pos, buf.size = b["data"], b["pos"], b["size"]
        pool._next = state["pool_next"]
        for gid, spec in state["pool_specs"].items():
            gid = int(gid)
            pool.graphs[gid] = make_graph(spec)
            pool.forced[gid] = (forced_mask(pool.graphs[gid]) if pool.use_forced
                                else np.zeros(pool.graphs[gid].n, dtype=bool))
            pool.refs[gid] = 0
        for tr in buf.data:
            if tr is not None:
                pool.refs[tr[0]] += 1
        del state
    slots = [new_slot() for _ in range(t["num_envs"])]
    gcache = GPUGraphCache(pool, device)          # replay batches are built on the training device

    def save_resume():
        gids = {tr[0] for tr in buf.data if tr is not None}
        st = {"step": step, "episodes": episodes, "best_gap": best_gap, "history": history,
              "pending_updates": pending_updates, "losses": losses[-1000:], "ep_returns": ep_returns[-200:],
              "ep_lens": ep_lens[-200:], "stats": {k: val[-1000:] for k, val in stats.items()},
              "wall_s": time.time() - t0,
              "agent": {"online": agent.online.state_dict(), "target": agent.target.state_dict(),
                        "opt": agent.opt.state_dict(), "updates": agent.updates},
              "rng": rng.bit_generator.state, "sampler_rng": sampler.rng.bit_generator.state,
              "buf_rng": buf.rng.bit_generator.state, "py_random": random.getstate(),
              "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
              "buffer": {"data": buf.data, "pos": buf.pos, "size": buf.size},
              "pool_next": pool._next, "pool_specs": {gid: pool.graphs[gid].meta for gid in gids}}
        tmp = resume_path.with_suffix(".tmp")
        torch.save(st, tmp)
        tmp.replace(resume_path)

    def validate():
        nonlocal best_gap
        r = evaluate_policy(agent.online, val_graphs, opt, device, env_kwargs=env_kwargs)
        lv = float(np.mean(losses[-500:])) if losses else float("nan")
        row = [step, episodes, agent.updates, epsilon(step, cfg), lv, r["gap_raw"], r["gap_rr"],
               r["size_rr"], r["frac_opt_rr"], ref["greedy_rr_gap"], ref["random_rr_gap"], time.time() - t0]
        vw.writerow(row)
        val_csv.flush()
        for k in ("gap_raw", "gap_rr", "frac_opt_rr"):
            tb.add_scalar(f"val/{k}", r[k], step)
        history.append(dict(zip(["step", "gap_raw", "gap_rr"], [step, r["gap_raw"], r["gap_rr"]])))
        ck = {**agent.state_dict(), "env_kwargs": env_kwargs, "step": step, "val": r, "cfg": cfg, "seed": seed}
        torch.save(ck, ckpt_dir / "last.pt")
        tag = ""
        if r["gap_rr"] < best_gap:
            best_gap = r["gap_rr"]
            torch.save(ck, ckpt_dir / "best.pt")
            tag = " *best*"
        save_resume()
        print(f"[{name}] step {step:>7} ep {episodes:>6} eps {epsilon(step, cfg):.3f} loss {lv:.4f} "
              f"val gap raw {r['gap_raw']:.4f} rr {r['gap_rr']:.4f} opt% {r['frac_opt_rr']:.2f} "
              f"({r['time']:.1f}s) wall {time.time() - t0:.0f}s{tag}", flush=True)

    if step == 0:
        validate()
    while step < t["total_steps"]:
        eps = epsilon(step, cfg)
        explore = rng.random(len(slots)) < eps
        greedy_idx = [i for i in range(len(slots)) if not explore[i]]
        qs = dict(zip(greedy_idx, q_values(agent.online, [slots[i].env for i in greedy_idx], device))) \
            if greedy_idx else {}
        agent.online.train()
        for i, slot in enumerate(slots):
            env = slot.env
            valid = np.flatnonzero(env.valid_mask())
            if i in qs:
                q = qs[i]
                a = int(valid[np.argmax(q[valid])])
            else:
                a = int(rng.choice(valid))
            S_t = list(env.order)
            _, r, done, _ = env.step(a)
            slot.pending.append((S_t, a, r))
            slot.ret += r
            step += 1
            pending_updates += 1.0 / t["train_every"]
            emit(slot, final=done)
            if done:
                episodes += 1
                ep_returns.append(slot.ret)
                ep_lens.append(len(env.order))
                slots[i] = new_slot()
            if step % v["every"] == 0:
                validate()
            if step % t["log_every"] == 0:
                lv = float(np.mean(losses[-200:])) if losses else float("nan")
                row = [step, episodes, eps, lv] + [float(np.mean(stats[k][-200:])) if stats[k] else float("nan")
                                                   for k in ("q_mean", "y_mean")] + \
                      [float(np.mean(ep_returns[-100:])) if ep_returns else float("nan"),
                       float(np.mean(ep_lens[-100:])) if ep_lens else float("nan"), len(pool.graphs), time.time() - t0]
                tw.writerow(row)
                tr_csv.flush()
                tb.add_scalar("train/loss", lv, step)
                tb.add_scalar("train/epsilon", eps, step)
                if ep_returns:
                    tb.add_scalar("train/ep_return", np.mean(ep_returns[-100:]), step)
        if len(buf) >= t["warmup"]:
            while pending_updates >= 1.0:
                pending_updates -= 1.0
                trs = buf.sample_transitions(t["batch_size"])
                st = agent.update_batches(*build_batches(gcache, trs, ec["mask_mode"]))
                losses.append(st["loss"])
                for k in ("q_mean", "y_mean"):
                    stats[k].append(st[k])
        else:
            pending_updates = 0.0
        if step % 2000 < len(slots):
            pool.collect(keep={s.gid for s in slots})
            gcache.prune()
    if step % v["every"] != 0:
        validate()
    tb.close()
    val_csv.close()
    tr_csv.close()
    summary = {"name": name, "best_gap_rr": best_gap, "history": history, "ref": ref,
               "wall_s": time.time() - t0, "steps": step, "episodes": episodes}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if resume_path.exists():
        resume_path.unlink()          # finished: the large resume state is no longer needed
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override train.total_steps")
    ap.add_argument("--name", default=None, help="override out.name (run/checkpoint folder prefix)")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/resume.pt if present")
    args = ap.parse_args()
    ov = {}
    if args.name:
        ov["out"] = {"name": args.name}
    if args.device:
        ov["device"] = args.device
    if args.steps:
        ov["train"] = {"total_steps": args.steps}
    cfg = load_config(args.config, ov)
    seed = cfg["seed"] if args.seed is None else args.seed
    train(cfg, seed, resume=args.resume)


if __name__ == "__main__":
    main()
