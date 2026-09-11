#!/usr/bin/env python3
"""Integration check against native ROS DUDE, not a Python fallback."""
import argparse
import json
from pathlib import Path

import numpy as np

from voxroom_online.isaac_runtime.baselines.offline.dude_runner import DudeIncrementalRunner
from voxroom_online.isaac_runtime.baselines.mask_io import (
    SEGMENTATION_INPUT_MODE_KEY, enforce_room_mask_contract,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ext = args.external_root
    free = np.zeros((320, 320), dtype=bool)
    free[25:110, 35:155] = True
    wall = np.zeros_like(free)
    wall[24:111, 34:156] = True
    wall[free] = False
    arrays = {
        "occupancy_map": wall, "observed_free_mask": free,
        "voxel_vertical_free_xy": free, "voxel_wall_xy": wall,
        "step": np.asarray(1), "map_resolution_m": np.asarray(0.05),
        SEGMENTATION_INPUT_MODE_KEY: np.asarray("raw_vertical_free"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for incremental in (True, False):
        runner = DudeIncrementalRunner(
            repo_root=ext / "dude_ws/src/Incremental_DuDe_ROS", dude_ws=ext / "dude_ws",
            ros_setup=str(ext / "ros_noetic_env/setup.bash"),
            ros_python=str(ext / "ros_noetic_env/bin/python"),
            concavity_threshold_m=2.5, use_incremental=incremental, map_resolution_m=0.05,
        )
        try:
            runner.start_scene("asymmetric_native_coordinate_check")
            result = runner.segment_snapshot(Path("roomseg_step_000001.npz"), arrays)
        finally:
            runner.end_scene()
        native = result.debug_arrays["dude_tagged_image_native"]
        old = enforce_room_mask_contract(native.astype(np.int32), arrays, clip_to_eval_domain=True)
        corrected = result.label_map
        assert result.metadata["runner_type"] == "original_ros"
        assert np.count_nonzero(native) > 0
        assert np.count_nonzero(old) == 0
        assert np.count_nonzero(corrected) / np.count_nonzero(free) > 0.75
        assert not np.any((corrected > 0) & ~free)
        np.savez_compressed(args.output / (runner.baseline_name + ".npz"),
                            free=free, native=native, old=old, corrected=corrected)
        records.append({"method": runner.baseline_name, "free_cells": int(free.sum()),
                        "old_kept_cells": int(np.count_nonzero(old)),
                        "corrected_kept_cells": int(np.count_nonzero(corrected)),
                        "metadata": result.metadata})
        print(json.dumps(records[-1]), flush=True)
    (args.output / "native_coordinate_check.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
