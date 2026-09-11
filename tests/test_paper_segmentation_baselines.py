from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from voxroom_online.isaac_runtime.baselines.mask_io import (
    SEGMENTATION_INPUT_MODE_KEY,
    build_segmentation_domain_from_source,
)
from voxroom_online.isaac_runtime.baselines.offline.gomez_runner import (
    GomezIncrementalRunner,
)
from voxroom_online.isaac_runtime.baselines.offline.fallback_voronoi import (
    build_voronoi_ipa_input_image,
)
from voxroom_online.isaac_runtime.baselines.offline.run_saved_snapshots import (
    make_runner,
)
from voxroom_online.isaac_runtime.baselines.offline.rose2_runner import (
    build_occupancy_grid_values,
)
from voxroom_online.isaac_runtime.comparison.metadata_gate import (
    _require_selected_segmentation_source,
    assert_main_experiment_metadata,
)


def _two_room_snapshot(*, step: int = 10) -> dict[str, np.ndarray]:
    height = width = 180
    free = np.zeros((height, width), dtype=bool)
    free[30:150, 20:160] = True
    wall = np.zeros_like(free)
    wall[29, 19:161] = True
    wall[150, 19:161] = True
    wall[29:151, 19] = True
    wall[29:151, 160] = True
    wall[29:80, 90] = True
    wall[101:151, 90] = True
    free[:, 90] = False
    free[80:101, 90] = True
    z_centers = (np.arange(60, dtype=np.float32) + 0.5) * 0.05
    voxel_state = np.zeros((len(z_centers), height, width), dtype=np.uint8)
    voxel_state[(z_centers >= 0.2) & (z_centers <= 2.2)] = np.where(
        wall[None, :, :],
        np.uint8(2),
        np.uint8(0),
    )
    tvars_raw_seed_points = np.zeros_like(free)
    tvars_raw_seed_points[79, 90] = True
    tvars_raw_seed_points[101, 90] = True
    return {
        "occupancy_map": wall,
        "final_room_label_map": free.astype(np.int32),
        "voxel_vertical_free_xy": free,
        "voxel_wall_xy": wall,
        "voxel_occupancy_state_zyx": voxel_state,
        "voxel_occupancy_z_centers_m": z_centers,
        "tvars_original_hough_door_seed_map": tvars_raw_seed_points,
        "tvars_original_raw_seed_source_snapshot": np.asarray(
            "/saved/tvars/synthetic_checkpoint.npz"
        ),
        "roomseg_eval_reference_explorable_mask": free,
        "roomseg_eval_explored_reference_mask": free,
        "map_resolution_m": np.asarray(0.05, dtype=np.float32),
        "map_width_cells": np.asarray(width, dtype=np.int32),
        "map_height_cells": np.asarray(height, dtype=np.int32),
        "map_origin_x_m": np.asarray(0.0, dtype=np.float32),
        "map_origin_y_m": np.asarray(0.0, dtype=np.float32),
        "agent_rc": np.asarray([90, 55], dtype=np.int32),
        "base_pose_world_xyzyaw": np.asarray([0.0, 0.0, 0.0, 0.0]),
        "step": np.asarray(step, dtype=np.int64),
    }


def test_segmentation_input_prefers_raw_vertical_free_without_reference_clipping() -> None:
    arrays = _two_room_snapshot()
    vertical = np.asarray(arrays["voxel_vertical_free_xy"], dtype=bool)
    explored = vertical.copy()
    explored[:, 120:] = False
    arrays["roomseg_eval_explored_reference_mask"] = explored
    arrays["navigation_free_room_domain"] = np.ones(vertical.shape, dtype=bool)

    actual, source = build_segmentation_domain_from_source(arrays)

    assert source == "voxel_vertical_free_xy"
    assert np.array_equal(actual, vertical)


def test_segmentation_input_can_select_raw_no_clearance_nav_free() -> None:
    arrays = _two_room_snapshot()
    vertical = np.asarray(arrays["voxel_vertical_free_xy"], dtype=bool)
    nav_free = vertical.copy()
    nav_free[70:110, 55:75] = False
    arrays["voxel_nav_free_xy"] = nav_free
    arrays["navigation_free_room_domain"] = nav_free.copy()
    arrays["roomseg_eval_explored_reference_mask"] = np.zeros_like(nav_free)
    arrays[SEGMENTATION_INPUT_MODE_KEY] = np.asarray(
        "raw_nav_free_no_clearance"
    )

    actual, source = build_segmentation_domain_from_source(arrays)

    assert source == "voxel_nav_free_xy"
    assert np.array_equal(actual, nav_free)


def test_metadata_gate_keeps_vertical_and_no_clearance_nav_contracts_distinct() -> None:
    _require_selected_segmentation_source(
        {
            "segmentation_input_mode": "raw_nav_free_no_clearance",
            "segmentation_source": "voxel_nav_free_xy",
        },
        "dude_incremental",
    )
    with pytest.raises(ValueError, match="incompatible"):
        _require_selected_segmentation_source(
            {
                "segmentation_input_mode": "raw_nav_free_no_clearance",
                "segmentation_source": "voxel_vertical_free_xy",
            },
            "dude_incremental",
        )
    with pytest.raises(ValueError, match="incompatible"):
        _require_selected_segmentation_source(
            {
                "segmentation_input_mode": "raw_vertical_free",
                "segmentation_source": "voxel_nav_free_xy",
            },
            "dude_incremental",
        )


def test_gomez_reproduction_detects_and_persists_door_separator() -> None:
    runner = GomezIncrementalRunner(map_resolution_m=0.05)
    runner.start_scene("synthetic")
    first_arrays = _two_room_snapshot(step=10)
    first = runner.segment_snapshot(Path("roomseg_step_000010.npz"), first_arrays)
    second_arrays = _two_room_snapshot(step=20)
    second_arrays["agent_rc"] = np.asarray([90, 125], dtype=np.int32)
    second = runner.segment_snapshot(Path("roomseg_step_000020.npz"), second_arrays)

    assert len([value for value in np.unique(first.label_map) if int(value) > 0]) == 2
    assert second.metadata["door_line_accepted_count_persistent"] >= 1
    assert second.metadata["door_lines_persistent"] is True
    assert second.metadata["raw_seed_source"] == (
        "saved_tvars_original_hough_door_seed_map"
    )
    assert second.metadata["checkpoint_virtual_laser_recomputed"] is False
    assert second.metadata["voxel_hough_confirmation_enabled"] is True
    assert second.metadata["door_line_hough_confirmed_candidate_count_current"] >= 1
    assert_main_experiment_metadata(second.metadata, "gomez_incremental")


def test_gomez_voxel_hough_rejects_candidate_without_two_vertical_jambs() -> None:
    arrays = _two_room_snapshot(step=10)
    voxel_state = np.asarray(arrays["voxel_occupancy_state_zyx"], dtype=np.uint8).copy()
    voxel_state[:, 94:109, 87:94] = 0
    arrays["voxel_occupancy_state_zyx"] = voxel_state

    runner = GomezIncrementalRunner(map_resolution_m=0.05)
    runner.start_scene("synthetic_missing_jamb")
    result = runner.segment_snapshot(Path("roomseg_step_000010.npz"), arrays)

    assert result.metadata["door_line_candidate_count_persistent"] >= 1
    assert result.metadata["door_line_hough_confirmed_candidate_count_current"] == 0
    assert result.metadata["door_line_accepted_count_persistent"] == 0
    assert len([value for value in np.unique(result.label_map) if int(value) > 0]) == 1


def test_gomez_requires_paired_saved_tvars_raw_seed_map() -> None:
    arrays = _two_room_snapshot(step=10)
    del arrays["tvars_original_hough_door_seed_map"]
    runner = GomezIncrementalRunner(map_resolution_m=0.05)
    runner.start_scene("synthetic_missing_tvars_raw_seed")

    with pytest.raises(ValueError, match="paired saved TVARS raw-seed point map"):
        runner.segment_snapshot(Path("roomseg_step_000010.npz"), arrays)


def test_incremental_replays_reuse_identical_step_result() -> None:
    arrays = _two_room_snapshot(step=0)

    gomez = GomezIncrementalRunner(map_resolution_m=0.05)
    gomez.start_scene("synthetic")
    gomez_first = gomez.segment_snapshot(Path("milestone_020_step_000000.npz"), arrays)
    gomez_same = gomez.segment_snapshot(Path("milestone_040_step_000000.npz"), arrays)
    assert np.array_equal(gomez_same.label_map, gomez_first.label_map)
    assert gomez_same.metadata["same_step_result_reused"] is True

    args = argparse.Namespace(
        dude_repo_root="/tmp/dude",
        dude_concavity_threshold_m=3.0,
        fallback_python=True,
        map_resolution_m=0.05,
        ros_baseline_setup=None,
        ros_baseline_python=None,
        dude_ws="/tmp/dude_ws",
    )
    dude = make_runner("dude_incremental", args)
    dude.start_scene("synthetic")
    dude_first = dude.segment_snapshot(Path("milestone_020_step_000000.npz"), arrays)
    dude_same = dude.segment_snapshot(Path("milestone_040_step_000000.npz"), arrays)
    assert np.array_equal(dude_same.label_map, dude_first.label_map)
    assert dude_same.metadata["same_step_result_reused"] is True


def test_rose2_input_uses_vertical_free_not_navigation_free() -> None:
    arrays = _two_room_snapshot()
    vertical = np.asarray(arrays["voxel_vertical_free_xy"], dtype=bool)
    arrays["navigation_free_room_domain"] = np.ones(vertical.shape, dtype=bool)

    grid, metadata = build_occupancy_grid_values(arrays)

    assert metadata["free_source_key"] == "voxel_vertical_free_xy"
    assert np.array_equal(grid == 0, vertical)


def test_ipa_voronoi_input_uses_vertical_free_not_navigation_free() -> None:
    arrays = _two_room_snapshot()
    vertical = np.asarray(arrays["voxel_vertical_free_xy"], dtype=bool)
    arrays["navigation_free_room_domain"] = np.ones(vertical.shape, dtype=bool)

    image, free, metadata = build_voronoi_ipa_input_image(arrays)

    assert metadata["free_source"] == "voxel_vertical_free_xy"
    assert np.array_equal(free, vertical)
    assert np.array_equal(image == 255, vertical)


def test_dude_factory_exposes_incremental_and_offline_state_modes() -> None:
    args = argparse.Namespace(
        dude_repo_root="/tmp/dude",
        dude_concavity_threshold_m=3.0,
        fallback_python=True,
        map_resolution_m=0.05,
        ros_baseline_setup=None,
        ros_baseline_python=None,
        dude_ws="/tmp/dude_ws",
    )

    incremental = make_runner("dude_incremental", args)
    offline = make_runner("dude_offline", args)

    assert incremental.use_incremental is True
    assert incremental.baseline_name == "dude_incremental"
    assert offline.use_incremental is False
    assert offline.baseline_name == "dude_offline"
