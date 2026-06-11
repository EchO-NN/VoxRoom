from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import VOXEL_UNKNOWN, VoxelOccupancyGrid3D
from voxroom_online.isaac_runtime.mapping.voxel_roomseg_evidence import (
    VoxelRoomsegEvidenceConfig,
    classify_voxel_columns_for_roomseg,
)
from voxroom_online.isaac_runtime.sensors.camera_geometry import CameraIntrinsics
from voxroom_online.isaac_runtime.sensors.depth_backproject import distance_to_camera_to_image_plane_depth


def _one_cell_voxel_grid() -> VoxelOccupancyGrid3D:
    return VoxelOccupancyGrid3D.zeros(
        (1, 1),
        MapInfo(resolution_m=0.05, min_x=0.0, max_x=0.05, min_y=0.0, max_y=0.05, width=1, height=1),
        cfg={"z_min_m": 0.0, "z_max_m": 0.05, "z_resolution_m": 0.05},
    )


def test_effective_frustum_marks_sensor_range_without_occupancy_evidence() -> None:
    grid = _one_cell_voxel_grid()
    grid.config.sensor_range_mark_effective_frustum_enabled = True
    updates = grid.mark_sensor_effective_range_rays(
        camera_origin_world=np.array([0.025, 0.025, 0.025], dtype=np.float32),
        range_endpoints_world=np.array([[0.025, 0.025, 0.025]], dtype=np.float32),
        floor_z=0.0,
    )

    assert updates > 0
    assert int(grid.sensor_range_count[0, 0, 0]) > 0
    assert int(grid.state[0, 0, 0]) == int(VOXEL_UNKNOWN)
    assert int(grid.log_odds[0, 0, 0]) == 0


def test_projective_frustum_volume_marks_sensor_range_by_image_plane_depth() -> None:
    grid = VoxelOccupancyGrid3D.zeros(
        (3, 3),
        MapInfo(resolution_m=0.05, min_x=0.0, max_x=0.15, min_y=0.0, max_y=0.15, width=3, height=3),
        cfg={
            "z_min_m": 0.0,
            "z_max_m": 0.15,
            "z_resolution_m": 0.05,
            "sensor_range_projective_frustum_volume_enabled": True,
            "sensor_range_mark_effective_frustum_enabled": False,
        },
    )
    intr = CameraIntrinsics(width=5, height=5, fx=2.0, fy=2.0, cx=2.0, cy=2.0)

    debug = grid.mark_sensor_projective_frustum_volume(
        camera_pose_world=(-0.10, 0.075, 0.075, 0.0),
        intr=intr,
        floor_z=0.0,
        depth_min_m=0.01,
        depth_max_m=0.30,
    )

    assert int(debug["voxel_sensor_projective_frustum_updates"]) > 0
    assert debug["voxel_sensor_range_mark_mode"] == "projective_frustum_volume"
    assert debug["voxel_sensor_projective_frustum_enabled"] is True
    assert debug["voxel_sensor_depth_range_semantics"] == "image_plane_z"
    assert int(debug["voxel_sensor_projective_frustum_candidate_voxels"]) > 0
    assert int(debug["voxel_sensor_projective_frustum_inside_voxels"]) > 0
    assert int(debug["voxel_sensor_range_projective_candidate_voxels"]) == int(debug["voxel_sensor_projective_frustum_candidate_voxels"])
    assert int(debug["voxel_sensor_range_projective_inside_voxels"]) == int(debug["voxel_sensor_projective_frustum_inside_voxels"])
    assert int(debug["voxel_sensor_range_projective_updates"]) == int(debug["voxel_sensor_projective_frustum_updates"])
    assert int(np.count_nonzero(grid.sensor_range_count)) > 0
    assert int(np.count_nonzero(grid.state == VOXEL_UNKNOWN)) == int(grid.state.size)
    assert int(np.count_nonzero(grid.log_odds)) == 0


def test_distance_to_camera_converts_to_image_plane_z_depth() -> None:
    intr = CameraIntrinsics(width=3, height=3, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    distance = np.full((3, 3), 5.0, dtype=np.float32)

    z_depth = distance_to_camera_to_image_plane_depth(distance, intr)

    assert np.isclose(float(z_depth[1, 1]), 5.0)
    assert float(z_depth[0, 0]) < 5.0
    expected_corner = 5.0 / np.sqrt(3.0)
    assert np.isclose(float(z_depth[0, 0]), expected_corner, atol=1.0e-5)


def test_in_range_unknown_requires_sensor_frustum_evidence() -> None:
    state = np.full((1, 1, 1), VOXEL_UNKNOWN, dtype=np.uint8)
    cfg = VoxelRoomsegEvidenceConfig(sensor_range_count_threshold_for_roomseg=1)
    nav_free = np.zeros((1, 1), dtype=bool)
    nav_obstacle = np.zeros((1, 1), dtype=bool)

    no_frustum = classify_voxel_columns_for_roomseg(
        state_active=state,
        cfg=cfg,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=nav_obstacle,
        sensor_range_active=np.zeros_like(state, dtype=np.uint8),
    )
    in_frustum = classify_voxel_columns_for_roomseg(
        state_active=state,
        cfg=cfg,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=nav_obstacle,
        sensor_range_active=np.ones_like(state, dtype=np.uint8),
    )

    assert int(no_frustum["in_range_unknown_count"][0, 0]) == 0
    assert int(no_frustum["outside_range_unknown_count"][0, 0]) == 1
    assert int(in_frustum["in_range_unknown_count"][0, 0]) == 1
    assert int(in_frustum["outside_range_unknown_count"][0, 0]) == 0


def test_generalized_wall_requires_nine_actual_occupied_cells_by_default() -> None:
    cfg = VoxelRoomsegEvidenceConfig.from_mapping({"wall_min_occupied_z_cells_for_xy_wall": 3})

    assert cfg.wall_min_occupied_z_cells_for_xy_wall == 3
    assert cfg.wall_min_actual_occupied_z_cells_for_xy_wall == 9
