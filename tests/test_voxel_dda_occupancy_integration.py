from __future__ import annotations

import numpy as np
import pytest

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.voxel_cpu_numba_backend import numba_available
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
)


def _map(width: int, height: int = 1) -> MapInfo:
    return MapInfo(
        resolution_m=0.05,
        min_x=0.0,
        max_x=float(width) * 0.05,
        min_y=0.0,
        max_y=float(height) * 0.05,
        width=int(width),
        height=int(height),
    )


def _numba_line_grid(width: int, *, exclude_n: int, splat_xy: int = 0, splat_z: int = 0) -> VoxelOccupancyGrid3D:
    return VoxelOccupancyGrid3D.zeros(
        (1, int(width)),
        _map(int(width), 1),
        cfg={
            "z_min_m": 0.0,
            "z_max_m": 0.05,
            "z_resolution_m": 0.05,
            "integration_backend": "cpu_numba",
            "ray_traversal_mode": "exact_dda",
            "ray_traversal_tie_epsilon": 1.0e-6,
            "free_excludes_endpoint": True,
            "free_excludes_last_n_voxels_before_endpoint": int(exclude_n),
            "endpoint_splat_xy_radius_cells": int(splat_xy),
            "endpoint_splat_z_radius_cells": int(splat_z),
            "free_vote_cap_per_voxel": 1,
            "occupied_vote_cap_per_voxel": 3,
            "occupied_wins_over_free_same_voxel": True,
            "free_logodds_delta": -1,
            "occupied_logodds_delta": 4,
            "free_logodds_threshold": -1,
            "occupied_logodds_threshold": 1,
            "occupied_to_free_logodds_threshold": -6,
            "cpu_numba_threads_mode": "manual",
            "cpu_numba_threads": 1,
            "cpu_numba_fail_if_thread_count_below": 1,
            "cpu_numba_chunk_rays": 64,
            "cpu_numba_max_samples_per_ray": 32,
            "sensor_range_tracking_enabled": False,
        },
    )


def test_python_reference_dda_tie_aware_thin_not_supercover() -> None:
    grid = VoxelOccupancyGrid3D.zeros(
        (3, 3),
        _map(3, 3),
        cfg={
            "z_min_m": 0.0,
            "z_max_m": 0.05,
            "z_resolution_m": 0.05,
            "ray_traversal_tie_epsilon": 1.0e-6,
        },
    )
    cells = grid.ray_voxels_3d(
        [0.025, 0.125, 0.025],
        [0.125, 0.025, 0.025],
        floor_z=0.0,
        include_endpoint=True,
    )

    assert cells == [(0, 0, 0), (0, 1, 1), (0, 2, 2)]


@pytest.mark.skipif(not numba_available(), reason="numba is unavailable")
def test_cpu_numba_inline_refresh_respects_occupied_to_free_hysteresis() -> None:
    from voxroom_online.isaac_runtime.mapping.voxel_cpu_numba_backend import _apply_logodds_blocked_kernel

    log = np.array([0], dtype=np.int16)
    state = np.array([int(VOXEL_OCCUPIED)], dtype=np.uint8)
    free_grouped = np.array([0], dtype=np.int32)
    free_offsets = np.array([0, 1], dtype=np.int64)
    occ_grouped = np.zeros(0, dtype=np.int32)
    occ_offsets = np.array([0, 0], dtype=np.int64)

    for expected_log, expected_state in ((-1, VOXEL_OCCUPIED), (-2, VOXEL_OCCUPIED), (-3, VOXEL_FREE)):
        _run_apply_kernel(
            _apply_logodds_blocked_kernel,
            log,
            state,
            free_grouped,
            free_offsets,
            occ_grouped,
            occ_offsets,
            occupied_to_free_threshold=-3,
        )
        assert int(log[0]) == int(expected_log)
        assert int(state[0]) == int(expected_state)


@pytest.mark.skipif(not numba_available(), reason="numba is unavailable")
def test_cpu_numba_hit_wins_and_free_vote_cap() -> None:
    from voxroom_online.isaac_runtime.mapping.voxel_cpu_numba_backend import _apply_logodds_blocked_kernel

    free_grouped = np.zeros(10, dtype=np.int32)
    free_offsets = np.array([0, 10], dtype=np.int64)
    occ_grouped = np.zeros(1, dtype=np.int32)
    occ_offsets = np.array([0, 1], dtype=np.int64)

    log = np.array([0], dtype=np.int16)
    state = np.array([int(VOXEL_UNKNOWN)], dtype=np.uint8)
    _run_apply_kernel(_apply_logodds_blocked_kernel, log, state, free_grouped, free_offsets, occ_grouped, occ_offsets)
    assert int(log[0]) == 4
    assert int(state[0]) == int(VOXEL_OCCUPIED)

    log = np.array([0], dtype=np.int16)
    state = np.array([int(VOXEL_UNKNOWN)], dtype=np.uint8)
    _run_apply_kernel(
        _apply_logodds_blocked_kernel,
        log,
        state,
        free_grouped,
        free_offsets,
        np.zeros(0, dtype=np.int32),
        np.array([0, 0], dtype=np.int64),
    )
    assert int(log[0]) == -1
    assert int(state[0]) == int(VOXEL_FREE)


@pytest.mark.skipif(not numba_available(), reason="numba is unavailable")
def test_cpu_numba_exact_dda_excludes_endpoint_only_by_default() -> None:
    grid = _numba_line_grid(5, exclude_n=0)

    stats = grid.integrate_depth_points(
        camera_origin_world=np.array([0.025, 0.025, 0.025], dtype=np.float32),
        points_world=np.array([[0.225, 0.025, 0.025]], dtype=np.float32),
        floor_z=0.0,
        valid_mask=np.array([True], dtype=bool),
    )

    assert stats.depth_rays_integrated == 1
    assert int(grid.state[0, 0, 0]) == int(VOXEL_FREE)
    assert int(grid.state[0, 0, 1]) == int(VOXEL_FREE)
    assert int(grid.state[0, 0, 2]) == int(VOXEL_FREE)
    assert int(grid.state[0, 0, 3]) == int(VOXEL_FREE)
    assert int(grid.state[0, 0, 4]) == int(VOXEL_OCCUPIED)


@pytest.mark.skipif(not numba_available(), reason="numba is unavailable")
def test_cpu_numba_free_only_ray_does_not_mark_endpoint_occupied() -> None:
    grid = _numba_line_grid(5, exclude_n=0)

    stats = grid.integrate_depth_points(
        camera_origin_world=np.array([0.025, 0.025, 0.025], dtype=np.float32),
        points_world=np.array([[0.225, 0.025, 0.025]], dtype=np.float32),
        floor_z=0.0,
        valid_mask=np.array([True], dtype=bool),
        endpoint_is_hit=np.array([False], dtype=bool),
    )

    assert stats.depth_rays_integrated == 1
    assert stats.depth_hit_rays_integrated == 0
    assert stats.depth_free_only_rays_integrated == 1
    assert int(grid.state[0, 0, 0]) == int(VOXEL_FREE)
    assert int(grid.state[0, 0, 1]) == int(VOXEL_FREE)
    assert int(grid.state[0, 0, 2]) == int(VOXEL_FREE)
    assert int(grid.state[0, 0, 3]) == int(VOXEL_FREE)
    assert int(grid.state[0, 0, 4]) == int(VOXEL_FREE)


@pytest.mark.skipif(not numba_available(), reason="numba is unavailable")
def test_cpu_numba_endpoint_splat_marks_surface_band_and_hit_wins() -> None:
    grid = _numba_line_grid(6, exclude_n=0, splat_xy=1, splat_z=0)

    stats = grid.integrate_depth_points(
        camera_origin_world=np.array([0.025, 0.025, 0.025], dtype=np.float32),
        points_world=np.array([[0.225, 0.025, 0.025]], dtype=np.float32),
        floor_z=0.0,
        valid_mask=np.array([True], dtype=bool),
    )

    assert stats.depth_rays_integrated == 1
    assert int(grid.state[0, 0, 0]) == int(VOXEL_FREE)
    assert int(grid.state[0, 0, 1]) == int(VOXEL_FREE)
    assert int(grid.state[0, 0, 2]) == int(VOXEL_FREE)
    assert int(grid.state[0, 0, 3]) == int(VOXEL_OCCUPIED)
    assert int(grid.state[0, 0, 4]) == int(VOXEL_OCCUPIED)
    assert int(grid.state[0, 0, 5]) == int(VOXEL_OCCUPIED)
    assert int(grid.log_odds[0, 0, 3]) > 0


def _run_apply_kernel(
    kernel,
    log: np.ndarray,
    state: np.ndarray,
    free_grouped: np.ndarray,
    free_offsets: np.ndarray,
    occ_grouped: np.ndarray,
    occ_offsets: np.ndarray,
    *,
    occupied_to_free_threshold: int = -6,
) -> None:
    counts = np.zeros(1, dtype=np.int64)
    kernel(
        log,
        state,
        free_grouped,
        free_offsets,
        occ_grouped,
        occ_offsets,
        64,
        -1,
        4,
        -20,
        20,
        -1,
        1,
        int(occupied_to_free_threshold),
        1,
        3,
        True,
        np.zeros(0, dtype=np.uint8),
        np.zeros(1, dtype=np.uint8),
        1,
        counts.copy(),
        counts.copy(),
        counts.copy(),
        counts.copy(),
        counts.copy(),
    )
