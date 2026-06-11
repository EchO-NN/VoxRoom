from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pytest

from voxroom_online.isaac_runtime.baselines.offline.run_saved_snapshots import run
from voxroom_online.isaac_runtime.baselines.offline.rose2_runner import infer_grid_resolution_origin


def test_offline_baseline_runner_smoke(tmp_path):
    source = tmp_path / "source" / "kujiale_smoke" / "roomseg_snapshots"
    source.mkdir(parents=True)
    shape = (24, 36)
    domain = np.zeros(shape, dtype=bool)
    domain[4:20, 3:14] = True
    domain[4:20, 22:33] = True
    np.savez_compressed(
        source / "roomseg_step_000001.npz",
        occupancy_map=~domain,
        observed_free_mask=domain,
        obstacle_mask=~domain,
        unknown_mask=np.zeros(shape, dtype=bool),
        navigation_free_room_domain=domain,
        final_room_label_map=np.zeros(shape, dtype=np.int32),
        map_resolution_m=np.asarray(0.05, dtype=np.float32),
        map_width_cells=np.asarray(shape[1], dtype=np.int32),
        map_height_cells=np.asarray(shape[0], dtype=np.int32),
    )
    args = Namespace(
        source_run_root=str(tmp_path / "source"),
        output_run_root=str(tmp_path / "out"),
        baselines=[
            "morphological",
            "distance_transform",
            "voronoi",
            "voronoi_width_jump",
            "rose2",
            "dude_incremental",
        ],
        scene_glob="kujiale_*",
        snapshot_glob="roomseg_step_*.npz",
        max_snapshots_per_scene=None,
        overwrite=True,
        allow_baseline_failure=False,
        fallback_python=True,
        map_resolution_m=0.05,
        ipa_ros_workspace=None,
        rose2_ros_workspace=None,
        rose2_launch_file="ROSE.launch",
        dude_repo_root=None,
        dude_concavity_threshold_m=3.0,
    )
    manifest = run(args)
    assert manifest["schema_version"] == 2
    assert manifest["strict_native"] is False
    assert len(manifest["rows"]) == 6
    for method in args.baselines:
        out = tmp_path / "out" / "kujiale_smoke" / "baselines" / method / "roomseg_snapshots" / "roomseg_step_000001.npz"
        arrays = np.load(out, allow_pickle=False)
        labels = arrays["final_room_label_map"]
        assert labels.dtype == np.int32
        assert labels.shape == shape
        metadata = json.loads(str(arrays["baseline_metadata_json"]))
        assert metadata["main_experiment_allowed"] is False


def test_rose2_reads_voxroom_map_origin_keys():
    arrays = {
        "occupancy_map": np.zeros((3, 4), dtype=bool),
        "map_resolution_m": np.asarray(0.05, dtype=np.float32),
        "map_origin_xy_m": np.asarray([1.25, -2.5], dtype=np.float32),
    }
    resolution, origin, metadata = infer_grid_resolution_origin(arrays)
    assert resolution == pytest.approx(0.05)
    assert origin == pytest.approx((1.25, -2.5))
    assert metadata["origin_source_key"] == "map_origin_xy_m"
