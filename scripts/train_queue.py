"""Run a list of training jobs sequentially (one "lane"), skipping finished ones.

    python scripts/train_queue.py --wait runs/default_seed0/summary.json \
        configs/default.yaml:3 configs/ablation_gin.yaml:0

Each job is CONFIG:SEED. A job is finished when runs/<out.name>_seed<SEED>/summary.json exists.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wtds.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="append", default=[], help="file(s) to wait for before starting")
    ap.add_argument("jobs", nargs="+")
    a = ap.parse_args()
    for f in a.wait:
        while not (ROOT / f).exists():
            time.sleep(30)
    for job in a.jobs:
        cfg_path, seed = job.rsplit(":", 1)
        name = load_config(ROOT / cfg_path)["out"]["name"]
        run_dir = ROOT / "runs" / f"{name}_seed{seed}"
        if (run_dir / "summary.json").exists():
            print(f"skip {job} (finished)", flush=True)
            continue
        log = ROOT / "runs" / f"{name}_seed{seed}.log"
        print(f"start {job} -> {log}", flush=True)
        with open(log, "a") as fh:
            code = subprocess.call([sys.executable, "-m", "wtds.train", "--config", cfg_path, "--seed", seed, "--resume"],
                                   cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        print(f"done {job} exit={code}", flush=True)


if __name__ == "__main__":
    main()
