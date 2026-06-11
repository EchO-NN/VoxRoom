from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.mapping.ceiling_height_estimator import CeilingHeightEstimator
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import VoxelOccupancyGrid3D


def test_ceiling_estimator_uses_cap_when_ceiling_missing_even_with_old_2m_fallback() -> None:
    estimator = CeilingHeightEstimator(
        active_z_min_m=0.10,
        storage_z_max_m=3.20,
        active_z_max_fallback_m=2.00,
        active_z_max_ceiling_ratio=0.85,
        active_z_max_cap_m=2.80,
    )

    assert np.isclose(estimator.active_z_max_for_height(None), 2.80)


def test_ceiling_estimator_uses_occupied_layer_peak_above_18m() -> None:
    estimator = CeilingHeightEstimator(
        {
            "lock_after_stable": False,
            "candidate_min_z_m": 1.80,
            "candidate_max_z_m": 4.00,
        },
        active_z_min_m=0.10,
        storage_z_max_m=4.00,
        active_z_max_fallback_m=2.80,
        active_z_max_ceiling_ratio=0.85,
        active_z_max_cap_m=2.80,
    )
    z_centers = np.asarray([0.0, 1.75, 1.80, 2.00, 2.50, 3.00], dtype=np.float32)
    state = np.zeros((z_centers.size, 4, 4), dtype=np.uint8)
    state[1, :, :] = 2  # Below 1.8m, so it must not win.
    state[3, :2, :2] = 2
    state[4, :3, :3] = 2

    estimate = estimator.update_from_occupied_layers(state, z_centers)

    assert np.isclose(estimate.height_m, 2.50)
    assert estimate.debug["ceiling_height_source"] == "voxel_occupied_layer_peak"
    assert estimate.debug["occupied_layer_peak_count"] == 9
    assert np.isclose(estimate.active_z_max_m, 2.125)


def test_voxel_grid_uses_cap_when_ceiling_missing_even_with_old_2m_fallback() -> None:
    grid = VoxelOccupancyGrid3D.zeros(
        (4, 4),
        MapInfo(resolution_m=0.05, min_x=0.0, max_x=0.2, min_y=0.0, max_y=0.2, width=4, height=4),
        cfg={
            "z_min_m": 0.0,
            "z_max_m": 3.2,
            "z_resolution_m": 0.05,
            "active_z_min_m": 0.10,
            "active_z_max_mode": "ceiling_ratio",
            "active_z_max_fallback_m": 2.00,
            "active_z_max_cap_m": 2.80,
        },
    )

    assert np.isclose(grid.active_z_max_m, 2.80)
    grid.set_active_z_from_ceiling(None, status="fallback")
    assert grid.ceiling_height_m is None
    assert np.isclose(grid.active_z_max_m, 2.80)
