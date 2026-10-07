"""Copy the fine-tuned best checkpoints and write checkpoints/model_info.json.

    python scripts/finalize_models.py
Every value is read from the fine-tuning runs (runs/wtds_seed*/val.csv) and the checkpoints.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wtds.model import QNet  # noqa: E402

TARGETS = {0: ROOT / "checkpoints" / "best.pt", 1: ROOT / "checkpoints" / "seed1_best.pt"}


def main():
    seeds, cfg = [], None
    for s, dst in TARGETS.items():
        val = pd.read_csv(ROOT / "runs" / f"wtds_seed{s}" / "val.csv")
        best = val.loc[val.gap_rr.idxmin()]
        step0 = val[val.step == 0].iloc[0]
        src = ROOT / "checkpoints" / f"wtds_seed{s}" / "best.pt"
        ck = torch.load(src, map_location="cpu", weights_only=False)
        assert abs(ck["val"]["gap_rr"] - best.gap_rr) < 1e-9, "best.pt does not match the best validation row"
        shutil.copy2(src, dst)
        cfg = ck["model_cfg"]
        seeds.append({"seed": s, "checkpoint": str(dst.relative_to(ROOT)).replace("\\", "/"),
                      "pretrained_from": f"checkpoints/pretrained/tds_seed{s}_best.pt",
                      "finetune_steps": int(val.step.max()), "best_step": int(best.step),
                      "val_gap_rr_step0": float(step0.gap_rr), "val_gap_raw_step0": float(step0.gap_raw),
                      "best_val_gap_rr": float(best.gap_rr), "best_val_gap_raw": float(best.gap_raw),
                      "best_val_frac_optimal_rr": float(best.frac_opt_rr),
                      "val_gap_weighted_greedy_rr": float(best.greedy_rr_gap),
                      "val_gap_random_rr": float(best.random_rr_gap)})
    m = QNet(**cfg)
    info = {"encoder": cfg["encoder"], "hidden": cfg["hidden"], "layers": cfg["layers"], "readout": cfg["readout"],
            "parameters": int(sum(p.numel() for p in m.parameters())), "reward": "weighted: r = -w(v) / mean weight",
            "seeds": seeds,
            "validation_note": "Validation: 100 fixed weighted graphs (n = 50-100; ER, BA, WS, random geometric, "
                               "lattices; weights uniform 1-100 / 1-10 / degree-correlated), all optima proven by the "
                               "exact ILP. Gap = mean (W(S) - W*) / W*. Both seeds were fine-tuned for the same number of steps "
                               "(80,000, then resumed to 130,000); best.pt = lowest gap after redundancy removal."}
    (ROOT / "checkpoints" / "model_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
