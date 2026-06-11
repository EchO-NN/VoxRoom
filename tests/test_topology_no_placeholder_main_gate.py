from __future__ import annotations

from argparse import Namespace

import numpy as np

from voxroom_online.isaac_runtime.baselines.live_manager import LiveBaselineManager
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo


def test_topology_scaffold_never_passes_main_gate_without_verified_original_adapter(tmp_path) -> None:
    args = Namespace(
        live_roomseg_baseline="topology_visual_active",
        live_baseline_output_dir=str(tmp_path / "baselines" / "topology_visual_active"),
        live_baseline_save_stream=False,
        live_baseline_door_detector="disabled",
        live_baseline_policy_control="never",
        live_baseline_panorama_views=12,
    )
    manager = LiveBaselineManager(args, scene_id="scene_a", run_dir=tmp_path)
    shape = (8, 10)
    free = np.zeros(shape, dtype=bool)
    free[1:7, 1:9] = True
    manager.on_step(
        step=1,
        obs={"has_rgb": False, "has_depth": False},
        map_state={
            "occupancy": ~free,
            "free": free,
            "observed": np.ones(shape, dtype=bool),
            "current_grid": (3, 3),
            "map_info": MapInfo(resolution_m=0.05, min_x=0, max_x=0.5, min_y=0, max_y=0.4, width=10, height=8),
        },
        frontier_map=np.zeros(shape, dtype=bool),
        selected_frontier_center_rc=(4, 4),
    )
    metadata = manager.impl.latest_metadata  # type: ignore[union-attr]
    assert metadata["main_experiment_allowed"] is False
    assert metadata["runner_type"] == "scaffold_or_debug"
    assert metadata["detector_adapter_verified"] is False
    assert metadata["projection_verified"] is False
    assert "smoke" in metadata["allowed_usage"]

