"""Plot training curves of one or more runs.

    python scripts/plot_training.py runs/quick_seed0 [runs/...] --out results/quick_curves.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for r in a.runs:
        name = Path(r).name
        tr = pd.read_csv(Path(r) / "train.csv")
        va = pd.read_csv(Path(r) / "val.csv")
        ax[0].plot(tr.step, tr.loss, label=name)
        ax[1].plot(tr.step, tr.ep_return, label=name)
        ax[2].plot(tr.step, tr.q_mean, label=f"{name} Q(s,a)")
        ax[2].plot(tr.step, tr.y_mean, "--", label=f"{name} target y")
        ax[3].plot(va.step, va.gap_rr, "o-", label=f"{name} RL+RR")
        ax[3].plot(va.step, va.gap_raw, "x:", label=f"{name} RL raw")
    va = pd.read_csv(Path(a.runs[0]) / "val.csv")
    ax[3].axhline(va.greedy_rr_gap.iloc[0], color="k", ls="--", label="greedy+RR")
    ax[3].axhline(va.random_rr_gap.iloc[0], color="gray", ls=":", label="random+RR")
    for x, t in zip(ax, ["TD loss (Huber)", "episode return (train, last 100)", "mean Q / target", "validation optimality gap"]):
        x.set_title(t)
        x.set_xlabel("env steps")
        x.grid(alpha=.3)
    ax[2].legend(fontsize=7)
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
