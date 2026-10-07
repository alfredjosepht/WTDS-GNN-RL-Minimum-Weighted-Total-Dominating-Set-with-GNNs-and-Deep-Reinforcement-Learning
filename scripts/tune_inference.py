"""Choose the sampling temperature on the VALIDATION set (never on test sets).

    python scripts/tune_inference.py
For each T, both fine-tuned seeds run the greedy rollout + k = 16 sampled rollouts on the 100 validation graphs;
the cheapest after redundancy removal counts. Writes checkpoints/inference.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wtds.solve import default_device, solve_rl  # noqa: E402
from wtds.validation import get_validation_set  # noqa: E402

TEMPS = [0.03, 0.1, 0.3, 1.0]
K = 16


def main():
    graphs, info = get_validation_set(str(ROOT / "data/val_set.npz"))
    opt = np.asarray(info["opt"], float)
    dev = default_device()
    cks = {0: ROOT / "checkpoints/best.pt", 1: ROOT / "checkpoints/seed1_best.pt"}
    res = {}
    for k_ in (1, K):
        for T in (TEMPS if k_ > 1 else [None]):
            per_seed = []
            for s, ck in cks.items():
                w = [solve_rl(g, str(ck), rr=True, ls_time=0, samples=k_, temperature=T, seed=i, device=dev).weight
                     for i, g in enumerate(graphs)]
                per_seed.append(float(np.mean((np.asarray(w) - opt) / opt)))
            key = "greedy_k1" if k_ == 1 else f"T={T}"
            res[key] = {"gap_per_seed": per_seed, "gap_mean": float(np.mean(per_seed))}
            print(key, res[key], flush=True)
    best = min((k for k in res if k.startswith("T=")), key=lambda k: res[k]["gap_mean"])
    out = {"temperature": float(best[2:]), "k": K, "validation_gaps_rr": res,
           "note": "chosen on data/val_set.npz (100 graphs); k sampled rollouts are added to the greedy rollout"}
    (ROOT / "checkpoints/inference.json").write_text(json.dumps(out, indent=2))
    print("chosen temperature", out["temperature"])


if __name__ == "__main__":
    main()
