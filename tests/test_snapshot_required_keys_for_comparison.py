from __future__ import annotations

from pathlib import Path

import numpy as np

from voxroom_online.isaac_runtime.baselines.mask_io import build_metric_domain_from_source
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import build_eval_domain


def test_metric_domain_excludes_unknown_obstacle_even_when_navigation_domain_includes_them() -> None:
    nav = np.ones((3, 4), dtype=bool)
    obstacle = np.zeros_like(nav)
    unknown = np.zeros_like(nav)
    obstacle[0, 0] = True
    unknown[1, 1] = True
    arrays = {
        "occupancy_map": np.zeros_like(nav),
        "final_room_label_map": np.ones(nav.shape, dtype=np.int32),
        "navigation_free_room_domain": nav,
        "observed_free_mask": nav,
        "obstacle_mask": obstacle,
        "unknown_mask": unknown,
    }
    domain = build_metric_domain_from_source(arrays)
    eval_domain, key = build_eval_domain(arrays)
    assert key == "navigation_free_room_domain"
    assert not domain[0, 0]
    assert not domain[1, 1]
    assert np.array_equal(domain, eval_domain)


def test_required_comparison_keys_exist_in_synthetic_snapshot(tmp_path: Path) -> None:
    shape = (3, 4)
    p = tmp_path / "roomseg_step_000001.npz"
    np.savez_compressed(
        p,
        occupancy_map=np.zeros(shape, dtype=np.uint8),
        observed_free_mask=np.ones(shape, dtype=bool),
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.zeros(shape, dtype=bool),
        navigation_free_room_domain=np.ones(shape, dtype=bool),
        vertical_free_room_domain=np.ones(shape, dtype=bool),
        voxel_vertical_free_xy=np.ones(shape, dtype=bool),
        frontier_map=np.zeros(shape, dtype=bool),
        selected_frontier_center_rc=np.asarray([1, 2], dtype=np.int32),
        agent_rc=np.asarray([1, 1], dtype=np.int32),
        final_room_label_map=np.ones(shape, dtype=np.int32),
        map_resolution_m=np.asarray(0.05, dtype=np.float32),
        map_origin_xy_m=np.asarray([0.0, 0.0], dtype=np.float32),
        map_width_cells=np.asarray(shape[1], dtype=np.int32),
        map_height_cells=np.asarray(shape[0], dtype=np.int32),
    )
    with np.load(p, allow_pickle=False) as data:
        missing = [key for key in _REQUIRED_KEYS if key not in data.files]
        assert not missing
        label_shape = data["final_room_label_map"].shape
        for key in _REQUIRED_KEYS:
            arr = data[key]
            if arr.ndim == 2:
                assert arr.shape == label_shape


_REQUIRED_KEYS = [
    "occupancy_map",
    "observed_free_mask",
    "obstacle_mask",
    "unknown_mask",
    "navigation_free_room_domain",
    "vertical_free_room_domain",
    "voxel_vertical_free_xy",
    "frontier_map",
    "selected_frontier_center_rc",
    "agent_rc",
    "final_room_label_map",
    "map_resolution_m",
    "map_origin_xy_m",
    "map_width_cells",
    "map_height_cells",
]

