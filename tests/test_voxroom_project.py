from __future__ import annotations

import subprocess
import sys

import numpy as np

from voxroom_online.isaac_runtime.config import load_config
from voxroom_online.segmentation import full_erosion_marker_roomseg


def test_voxroom_config_loads() -> None:
    cfg = load_config("configs/voxroom_online.yaml")
    assert cfg.project.name == "voxroom_online_interioragent"
    assert cfg.voxroom_policy.frontier_selection_mode == "random"


def test_full_erosion_splits_narrow_connection() -> None:
    free = np.zeros((40, 80), dtype=bool)
    free[8:32, 6:30] = True
    free[8:32, 50:74] = True
    free[18:22, 30:50] = True
    labels, seeds = full_erosion_marker_roomseg(free, min_child_area=20, min_child_ratio=0.01)
    assert int(np.max(labels)) >= 2
    assert int(np.count_nonzero(seeds)) > 0


def test_public_runner_help_starts() -> None:
    result = subprocess.run(
        [sys.executable, "voxroom_online/isaac_runtime/scripts/run_one_episode.py", "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert "--voxroom-viz" in result.stdout
    assert "--frontier-selection-mode" in result.stdout
