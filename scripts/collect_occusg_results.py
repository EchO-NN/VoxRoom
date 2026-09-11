#!/usr/bin/env python3
"""Bounded, read-only download of this OccuSG evaluation (never restarts jobs)."""
import argparse
import json
from pathlib import Path
import subprocess
import time

DEST = Path(__file__).resolve().parents[1] / "results/occusg_official_saved_voxels_20260910"
REMOTE = "echo@[fe80::630c:c787:c8e7:5fcb%enp129s0]:/media/echo/data/voxroom_occusg_paper_20260910/results/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--destination", type=Path, default=DEST)
    parser.add_argument("--iterations", type=int, default=120)
    args = parser.parse_args()
    dest = args.destination
    dest.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-az", "--timeout=30", "-e", "ssh -6 -o BatchMode=yes -o ConnectTimeout=8",
               "--exclude=known_voxels.bin", "--exclude=octomap.data", args.remote, str(dest)+"/"]
    for _ in range(args.iterations):
        try:
            result = subprocess.run(command, timeout=120, check=False)
        except subprocess.TimeoutExpired:
            time.sleep(30)
            continue
        if result.returncode == 0:
            if (dest / "completed.json").exists():
                print(f"Final tables and predictions saved to {dest}", flush=True)
                return
            for dataset in ("interioragent", "grscene"):
                status = dest / dataset / "status.json"
                if status.exists() and json.loads(status.read_text()).get("status") == "failed":
                    raise RuntimeError(f"{dataset} failed; partial results and logs downloaded")
        time.sleep(30)
    raise TimeoutError("Collector reached configured polling limit; remote run remains untouched")


if __name__ == "__main__":
    main()
