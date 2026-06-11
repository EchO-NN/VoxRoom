from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
)


def _tiny_grid() -> VoxelOccupancyGrid3D:
    return VoxelOccupancyGrid3D.zeros(
        (1, 1),
        MapInfo(resolution_m=0.05, min_x=0.0, max_x=0.05, min_y=0.0, max_y=0.05, width=1, height=1),
        cfg={"z_min_m": 0.0, "z_max_m": 0.05, "z_resolution_m": 0.05},
    )


def test_neutral_log_odds_preserves_previous_known_state() -> None:
    grid = _tiny_grid()
    assert int(grid.state[0, 0, 0]) == int(VOXEL_UNKNOWN)

    grid.log_odds[0, 0, 0] = 3
    grid.refresh_state_indices(np.array([0], dtype=np.int64))
    assert int(grid.state[0, 0, 0]) == int(VOXEL_OCCUPIED)

    grid.log_odds[0, 0, 0] = 0
    grid.refresh_state_indices(np.array([0], dtype=np.int64))
    assert int(grid.state[0, 0, 0]) == int(VOXEL_OCCUPIED)

    grid.log_odds[0, 0, 0] = -1
    grid.refresh_state_indices(np.array([0], dtype=np.int64))
    assert int(grid.state[0, 0, 0]) == int(VOXEL_OCCUPIED)

    occupied_to_free_threshold = int(grid.config.occupied_to_free_logodds_threshold)

    grid.log_odds[0, 0, 0] = occupied_to_free_threshold + 1
    grid.refresh_state()
    assert int(grid.state[0, 0, 0]) == int(VOXEL_OCCUPIED)

    grid.log_odds[0, 0, 0] = occupied_to_free_threshold
    grid.refresh_state_indices(np.array([0], dtype=np.int64))
    assert int(grid.state[0, 0, 0]) == int(VOXEL_FREE)

    grid.log_odds[0, 0, 0] = 0
    grid.refresh_state()
    assert int(grid.state[0, 0, 0]) == int(VOXEL_FREE)


def test_unobserved_neutral_log_odds_stays_unknown() -> None:
    grid = _tiny_grid()
    grid.refresh_state()
    assert int(grid.state[0, 0, 0]) == int(VOXEL_UNKNOWN)


def test_force_free_bypasses_occupied_to_free_hysteresis() -> None:
    grid = _tiny_grid()
    grid.log_odds[0, 0, 0] = 3
    grid.refresh_state_indices(np.array([0], dtype=np.int64))
    assert int(grid.state[0, 0, 0]) == int(VOXEL_OCCUPIED)

    count, changed = grid.force_free_voxels(np.array([[0, 0, 0]], dtype=np.int64))
    assert count == 1
    grid.refresh_state_indices(changed)
    assert int(grid.log_odds[0, 0, 0]) <= int(grid.config.occupied_to_free_logodds_threshold)
    assert int(grid.state[0, 0, 0]) == int(VOXEL_FREE)
