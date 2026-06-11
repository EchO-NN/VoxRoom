from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from voxroom_online.isaac_runtime.baselines.live_manager import LiveBaselineManager
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo


def test_saved_stream_manifest_schema(tmp_path):
    args = Namespace(
        live_roomseg_baseline="topology_visual_active",
        live_baseline_output_dir=str(tmp_path / "baselines" / "topology_visual_active"),
        live_baseline_save_stream=True,
        live_baseline_door_detector="disabled",
        live_baseline_policy_control="never",
        live_baseline_panorama_views=12,
    )
    manager = LiveBaselineManager(args, scene_id="scene_a", run_dir=tmp_path)
    manager.on_episode_start({"scene_id": "scene_a"})
    shape = (12, 14)
    free = np.zeros(shape, dtype=bool)
    free[2:10, 2:12] = True
    map_state = {
        "occupancy": ~free,
        "free": free,
        "observed": np.ones(shape, dtype=bool),
        "current_grid": (4, 4),
        "map_info": MapInfo(resolution_m=0.05, min_x=0, max_x=0.7, min_y=0, max_y=0.6, width=14, height=12),
    }
    obs = {
        "has_rgb": True,
        "has_depth": True,
        "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
        "depth": np.ones((4, 5), dtype=np.float32),
        "pose_world": np.zeros(4, dtype=np.float32),
        "camera_pose_world": np.eye(4, dtype=np.float32),
    }
    manager.on_step(step=1, obs=obs, map_state=map_state, frontier_map=np.zeros(shape, dtype=bool), selected_frontier_center_rc=(5, 5))
    manifest = tmp_path / "active_stream" / "stream_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[0]["schema_version"] == 2
    assert rows[0]["has_rgb"] is True
    assert rows[0]["has_depth"] is True
    assert rows[0]["has_camera_pose_world"] is True
    frame = tmp_path / "active_stream" / rows[0]["frame_npz"]
    arrays = np.load(frame, allow_pickle=False)
    for key in ("rgb", "depth", "pose_world", "agent_pose_world", "camera_pose_world", "occupancy_map", "observed_free_mask", "agent_rc", "control_step"):
        assert key in arrays.files
