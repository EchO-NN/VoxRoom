from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.coordinate_transform import world_xy_to_grid
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_door_wall_roomseg import _apply_outside_boundary_to_evidence
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import VOXEL_OCCUPIED, VoxelOccupancyGrid3D
from voxroom_online.isaac_runtime.mapping.voxel_roomseg_evidence import build_voxel_roomseg_evidence
from voxroom_online.isaac_runtime.sensors.camera_geometry import CameraIntrinsics


def _grid() -> VoxelOccupancyGrid3D:
    return VoxelOccupancyGrid3D.zeros(
        (2, 3),
        MapInfo(resolution_m=0.05, min_x=0.0, max_x=0.15, min_y=0.0, max_y=0.10, width=3, height=2),
        cfg={
            "z_min_m": -0.10,
            "z_max_m": 0.30,
            "z_resolution_m": 0.10,
            "active_z_min_m": -0.10,
            "active_z_max_cap_m": 0.30,
            "outside_score_threshold": 1,
            "outside_occupied_check_z_min_m": -0.10,
            "outside_occupied_check_z_max_m": 0.30,
            "outside_require_disconnected_from_robot_navigation_component": False,
        },
    )


def _robot_gate_grid(shape: tuple[int, int] = (5, 7)) -> VoxelOccupancyGrid3D:
    height, width = int(shape[0]), int(shape[1])
    return VoxelOccupancyGrid3D.zeros(
        (height, width),
        MapInfo(
            resolution_m=0.05,
            min_x=0.0,
            max_x=0.05 * width,
            min_y=0.0,
            max_y=0.05 * height,
            width=width,
            height=height,
        ),
        cfg={
            "z_min_m": -0.10,
            "z_max_m": 0.30,
            "z_resolution_m": 0.10,
            "active_z_min_m": -0.10,
            "active_z_max_cap_m": 0.30,
            "outside_score_threshold": 1,
            "outside_occupied_check_z_min_m": -0.10,
            "outside_occupied_check_z_max_m": 0.30,
            "outside_require_disconnected_from_robot_navigation_component": True,
            "outside_fail_closed_if_robot_component_unavailable": True,
            "outside_hard_clear_robot_component_scores": True,
        },
    )


def test_outside_requires_nav_free_floor_frustum_and_full_height_no_occupied() -> None:
    grid = _grid()
    nav_free = np.zeros(grid.shape, dtype=bool)
    nav_free[0, 0] = True
    nav_free[0, 1] = True
    nav_free[1, 0] = True
    grid.floor_frustum_seen_count_xy[0, 0] = 1
    grid.floor_frustum_seen_count_xy[0, 1] = 1
    grid.floor_frustum_seen_count_xy[1, 0] = 1
    grid.state[0, 1, 0] = VOXEL_OCCUPIED
    forced = np.zeros(grid.shape, dtype=bool)
    forced[0, 1] = True

    outside, debug = grid.update_outside_from_navigation_free(
        nav_free,
        forced_initial_blind_zone_free_xy=forced,
    )

    assert bool(outside[0, 0])
    assert not bool(outside[0, 1])
    assert not bool(outside[1, 0])
    assert debug["voxel_outside_candidate_cells"] == 1
    assert debug["voxel_outside_confirmed_cells"] == 1


def test_outside_holds_out_floor_unseen_cells() -> None:
    grid = _grid()
    nav_free = np.ones(grid.shape, dtype=bool)

    outside, debug = grid.update_outside_from_navigation_free(nav_free)

    assert not np.any(outside)
    assert debug["voxel_outside_candidate_cells"] == 0
    assert debug["voxel_outside_holdout_floor_unseen_cells"] == int(nav_free.size)


def test_outside_hysteresis_requires_two_candidate_frames_by_default() -> None:
    grid = _grid()
    grid.config.outside_score_threshold = 2
    nav_free = np.ones(grid.shape, dtype=bool)
    grid.floor_frustum_seen_count_xy[:, :] = 1

    outside1, debug1 = grid.update_outside_from_navigation_free(nav_free)
    outside2, debug2 = grid.update_outside_from_navigation_free(nav_free)

    assert not np.any(outside1)
    assert np.all(outside2)
    assert debug1["voxel_outside_confirmed_cells"] == 0
    assert debug2["voxel_outside_confirmed_cells"] == int(nav_free.size)


def test_outside_current_footprint_radius_exclusion_suppresses_synthetic_free() -> None:
    grid = _grid()
    nav_free = np.ones(grid.shape, dtype=bool)
    grid.floor_frustum_seen_count_xy[:, :] = 1
    current = np.zeros(grid.shape, dtype=bool)
    current[0, 0] = True
    grid.config.outside_current_footprint_exclusion_radius_m = 0.06

    outside, debug = grid.update_outside_from_navigation_free(
        nav_free,
        forced_current_footprint_free_xy=current,
    )

    assert not bool(outside[0, 0])
    assert debug["voxel_outside_current_footprint_radius_excluded_cells"] > 1
    assert debug["voxel_outside_candidate_cells"] < int(nav_free.size)


def test_outside_occupied_check_uses_configured_full_height_range() -> None:
    grid = VoxelOccupancyGrid3D.zeros(
        (1, 1),
        MapInfo(resolution_m=0.05, min_x=0.0, max_x=0.05, min_y=0.0, max_y=0.05, width=1, height=1),
        cfg={
            "z_min_m": -0.10,
            "z_max_m": 0.50,
            "z_resolution_m": 0.10,
            "active_z_min_m": -0.10,
            "active_z_max_cap_m": 0.50,
            "outside_score_threshold": 1,
            "outside_occupied_check_z_min_m": -0.10,
            "outside_occupied_check_z_max_m": 0.50,
            "outside_require_disconnected_from_robot_navigation_component": False,
        },
    )
    nav_free = np.ones(grid.shape, dtype=bool)
    grid.floor_frustum_seen_count_xy[0, 0] = 1
    grid.state[-1, 0, 0] = VOXEL_OCCUPIED

    outside, debug = grid.update_outside_from_navigation_free(nav_free)

    assert not bool(outside[0, 0])
    assert debug["voxel_outside_has_occupied_xy_cells"] == 1
    assert debug["voxel_outside_occupied_check_z_min_m"] == grid.z_min_m
    assert debug["voxel_outside_occupied_check_z_max_m"] == grid.z_max_m


def test_floor_frustum_visibility_uses_image_plane_z_not_euclidean_range() -> None:
    info = MapInfo(resolution_m=0.50, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, width=20, height=20)
    grid = VoxelOccupancyGrid3D.zeros(
        (20, 20),
        info,
        cfg={"z_min_m": -0.10, "z_max_m": 0.30, "z_resolution_m": 0.10},
    )
    intr = CameraIntrinsics.from_hfov(width=100, height=100, hfov_deg=160.0)
    pose = (0.25, 5.0, 2.0, 0.0)
    row, col = world_xy_to_grid(4.75, 8.75, info)
    euclidean = np.linalg.norm(np.asarray([4.75 - pose[0], 8.75 - pose[1], -0.05 - pose[2]], dtype=np.float64))
    assert euclidean > 5.0

    debug = grid.mark_floor_projective_frustum_visibility(
        camera_pose_world=pose,
        intr=intr,
        floor_z=0.0,
        depth_min_m=0.20,
        depth_max_m=5.0,
        floor_rel_z_m=-0.05,
        count_delta=1,
        count_max=65535,
    )

    assert grid.floor_frustum_seen_count_xy.dtype == np.uint16
    assert int(grid.floor_frustum_seen_count_xy[row, col]) > 0
    assert debug["voxel_floor_frustum_depth_semantics"] == "image_plane_z"
    assert debug["voxel_floor_frustum_floor_world_z"] == -0.05
    assert debug["voxel_floor_frustum_updates"] > 0
    assert debug["voxel_floor_frustum_seen_xy_count"] > 0


def test_outside_rejects_navigation_free_connected_to_robot_component() -> None:
    grid = _robot_gate_grid()
    nav_free = np.zeros(grid.shape, dtype=bool)
    nav_free[2, 1:6] = True
    grid.floor_frustum_seen_count_xy[nav_free] = 1

    outside, debug = grid.update_outside_from_navigation_free(
        nav_free,
        pre_outside_traversible_xy=nav_free.copy(),
        robot_grid=(2, 1),
    )

    assert not np.any(outside)
    assert debug["voxel_outside_candidate_base_cells"] == 5
    assert debug["voxel_outside_connected_to_robot_rejected_cells"] == 5
    assert debug["voxel_outside_disconnected_candidate_cells"] == 0
    assert debug["voxel_outside_robot_component_available"] is True
    assert int(np.count_nonzero(debug["voxel_outside_robot_component_xy"])) == 5
    assert debug["voxel_outside_overlap_robot_component_cells"] == 0


def test_outside_allows_only_disconnected_navigation_free_cluster() -> None:
    grid = _robot_gate_grid()
    nav_free = np.zeros(grid.shape, dtype=bool)
    nav_free[2, 1:3] = True
    nav_free[2, 5:7] = True
    grid.floor_frustum_seen_count_xy[nav_free] = 1

    outside, debug = grid.update_outside_from_navigation_free(
        nav_free,
        pre_outside_traversible_xy=nav_free.copy(),
        robot_grid=(2, 1),
    )

    assert not bool(outside[2, 1])
    assert not bool(outside[2, 2])
    assert bool(outside[2, 5])
    assert bool(outside[2, 6])
    assert debug["voxel_outside_candidate_base_cells"] == 4
    assert debug["voxel_outside_connected_to_robot_rejected_cells"] == 2
    assert debug["voxel_outside_disconnected_candidate_cells"] == 2


def test_outside_score_hard_clears_when_cluster_becomes_connected_to_robot() -> None:
    grid = _robot_gate_grid()
    nav_free = np.zeros(grid.shape, dtype=bool)
    nav_free[2, 1:3] = True
    nav_free[2, 5:7] = True
    grid.floor_frustum_seen_count_xy[nav_free] = 1

    outside1, _debug1 = grid.update_outside_from_navigation_free(
        nav_free,
        pre_outside_traversible_xy=nav_free.copy(),
        robot_grid=(2, 1),
    )
    assert bool(outside1[2, 5])
    assert int(grid.outside_score_xy[2, 5]) > 0

    nav_free[2, 3:5] = True
    grid.floor_frustum_seen_count_xy[nav_free] = 1
    outside2, debug2 = grid.update_outside_from_navigation_free(
        nav_free,
        pre_outside_traversible_xy=nav_free.copy(),
        robot_grid=(2, 1),
    )

    assert not bool(outside2[2, 5])
    assert int(grid.outside_score_xy[2, 5]) == 0
    assert debug2["voxel_outside_connected_to_robot_rejected_cells"] == int(np.count_nonzero(nav_free))


def test_outside_robot_component_snaps_to_nearest_traversible_seed() -> None:
    grid = _robot_gate_grid()
    nav_free = np.zeros(grid.shape, dtype=bool)
    nav_free[2, 2:5] = True
    grid.floor_frustum_seen_count_xy[nav_free] = 1

    outside, debug = grid.update_outside_from_navigation_free(
        nav_free,
        pre_outside_traversible_xy=nav_free.copy(),
        robot_grid=(2, 1),
    )

    assert debug["voxel_outside_robot_component_available"] is True
    assert debug["voxel_outside_robot_seed_grid"] == [2, 1]
    assert debug["voxel_outside_robot_seed_snapped_grid"] == [2, 2]
    assert debug["voxel_outside_robot_seed_snap_distance_cells"] == 1.0
    assert not np.any(outside[2, 2:5])


def test_outside_fail_closed_when_robot_component_unavailable() -> None:
    grid = _robot_gate_grid()
    nav_free = np.ones(grid.shape, dtype=bool)
    grid.floor_frustum_seen_count_xy[:, :] = 1
    pre_traversible = np.zeros(grid.shape, dtype=bool)

    outside, debug = grid.update_outside_from_navigation_free(
        nav_free,
        pre_outside_traversible_xy=pre_traversible,
        robot_grid=(2, 2),
    )

    assert not np.any(outside)
    assert debug["voxel_outside_candidate_base_cells"] == int(nav_free.size)
    assert debug["voxel_outside_robot_component_available"] is False
    assert debug["voxel_outside_disconnected_candidate_cells"] == 0


def test_outside_connectivity_prevents_diagonal_corner_cutting_like_astar() -> None:
    grid = _robot_gate_grid((3, 3))
    nav_free = np.zeros(grid.shape, dtype=bool)
    nav_free[0, 0] = True
    nav_free[1, 1] = True
    grid.floor_frustum_seen_count_xy[nav_free] = 1

    outside, debug = grid.update_outside_from_navigation_free(
        nav_free,
        pre_outside_traversible_xy=nav_free.copy(),
        robot_grid=(0, 0),
    )

    assert not bool(outside[0, 0])
    assert bool(outside[1, 1])
    assert debug["voxel_outside_connected_to_robot_rejected_cells"] == 1
    assert debug["voxel_outside_disconnected_candidate_cells"] == 1


def test_outside_boundary_masks_roomseg_free_wall_and_unknown_without_becoming_wall() -> None:
    grid = _grid()
    grid.outside_xy[0, 0] = True
    grid.last_outside_debug = {"voxel_outside_xy": grid.outside_xy.copy()}
    nav_free = np.ones(grid.shape, dtype=bool)
    nav_obstacle = np.zeros(grid.shape, dtype=bool)
    nav_unknown = np.zeros(grid.shape, dtype=bool)
    grid.state[:, 0, 0] = VOXEL_OCCUPIED
    evidence = build_voxel_roomseg_evidence(
        voxel_grid=grid,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=nav_obstacle,
        unknown_mask_from_navigation=nav_unknown,
        resolution_m=0.05,
    )

    debug = _apply_outside_boundary_to_evidence(evidence, grid.outside_xy)

    assert debug["voxel_outside_roomseg_boundary_cells"] == 1
    assert not bool(evidence.vertical_free_xy[0, 0])
    assert not bool(evidence.wall_xy[0, 0])
    assert not bool(evidence.unknown_xy[0, 0])
    assert bool(evidence.debug["voxel_outside_xy"][0, 0])
