#!/usr/bin/env python3
"""Collect this one tau=1.5 run; never restart or mutate remote experiments."""
import json
from pathlib import Path
import subprocess
import time

DEST = Path(__file__).resolve().parents[1] / "results/dude_tau1p5_vertical_20260910"
REMOTE = "echo@[fe80::630c:c787:c8e7:5fcb%enp129s0]:/media/echo/data/voxroom_dude_tau1p5_vertical_20260910/"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-az", "--timeout=30", "-e", "ssh -6 -o BatchMode=yes -o ConnectTimeout=8",
               "--exclude=code/", "--exclude=tmp/", "--exclude=ros_logs/", "--exclude=native.log", REMOTE, str(DEST)+"/"]
    for _ in range(360):
        result = subprocess.run(command, timeout=120, check=False)
        if result.returncode == 0:
            if (DEST / "completed.json").exists():
                print(f"Complete: {DEST / 'tables/report.md'}", flush=True)
                return
            for dataset in ("interioragent", "grscene"):
                path = DEST / dataset / "status.json"
                if path.exists() and json.loads(path.read_text()).get("status") == "failed":
                    raise RuntimeError(f"{dataset} failed; partial results downloaded")
        time.sleep(30)
    raise TimeoutError("Six-hour collection limit reached; remote experiment untouched")


if __name__ == "__main__":
    main()
