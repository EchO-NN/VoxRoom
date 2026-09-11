#!/usr/bin/env python3
"""Collect this bounded remote run's results over the Ethernet IPv6 link."""
import json
from pathlib import Path
import subprocess
import time

REMOTE = "echo@fe80::630c:c787:c8e7:5fcb%enp129s0"
SOURCE = "/media/echo/data/voxroom_dude_corrected_20260910"
DEST = Path(__file__).resolve().parents[1] / "results/dude_corrected_20260910"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-az", "--timeout=30", "-e", "ssh -6 -o BatchMode=yes -o ConnectTimeout=8",
               "--exclude=code/", "--exclude=ros_logs/", "--exclude=predictions/",
               "echo@[fe80::630c:c787:c8e7:5fcb%enp129s0]:" + SOURCE + "/", str(DEST) + "/"]
    # A bounded collector only: it never restarts training/replay or changes the remote data.
    for _ in range(180):
        result = subprocess.run(command, timeout=120, check=False)
        if result.returncode == 0:
            if (DEST / "completed.json").is_file():
                print("Completed DUDE result tables downloaded to " + str(DEST / "tables"), flush=True)
                return
            for dataset in ("interioragent", "grscene"):
                status_path = DEST / dataset / "status.json"
                if status_path.is_file() and json.loads(status_path.read_text()).get("status") == "failed":
                    raise RuntimeError(dataset + " replay failed; logs downloaded for diagnosis")
        time.sleep(60)
    raise TimeoutError("DUDE collector reached 3-hour limit; remote experiment left untouched")


if __name__ == "__main__":
    main()
