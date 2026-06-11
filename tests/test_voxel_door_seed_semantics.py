from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.voxel_door_detector import (
    DoorMemoryObservationMaps,
    VoxelDoorDetectorConfig,
    VoxelDoorLineCandidate,
    VoxelDoorMemory,
    build_door_partition_cut_v30,
    _door_extension_no_clearance_occupied_reject,
    _door_scan_z_bounds,
    _extract_parallel_pruned_seed_line_segments,
    _extract_spur_pruned_seed_line_segments,
    _primitive_along_stats,
    _primitive_reject_reason,
    _walk_to_wall,
    classify_voxel_door_seeds,
    classify_voxel_door_seeds_vectorized,
    validate_door_line_local_neck,
)
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
)


def test_door_seed_upper_solid_rejects_in_range_unknown_as_solid() -> None:
    z_centers = np.arange(0.125, 2.825, 0.05, dtype=np.float32)
    state = np.full((int(z_centers.size), 1, 1), VOXEL_UNKNOWN, dtype=np.uint8)
    sensor_range_count = np.zeros_like(state, dtype=np.uint8)

    lower_free = z_centers < 1.775
    actual_upper_occupied = (z_centers >= 1.825) & (z_centers < 1.975)
    upper_in_range_unknown = z_centers >= 1.975

    state[lower_free, 0, 0] = VOXEL_FREE
    state[actual_upper_occupied, 0, 0] = VOXEL_OCCUPIED
    state[upper_in_range_unknown, 0, 0] = VOXEL_UNKNOWN
    sensor_range_count[upper_in_range_unknown, 0, 0] = 1

    cfg = VoxelDoorDetectorConfig()
    assert cfg.upper_solid_use_in_range_unknown is False
    assert cfg.upper_solid_use_out_of_range_unknown is True

    seed, reason_map, *_rest, debug = classify_voxel_door_seeds_vectorized(
        state,
        z_centers,
        np.arange(int(z_centers.size), dtype=np.int32),
        cfg,
        shape=(1, 1),
        return_debug=True,
        sensor_range_count=sensor_range_count,
    )

    assert not bool(seed[0, 0])
    assert int(reason_map[0, 0]) != 1
    assert not bool(debug["voxel_door_upper_solid_uses_in_range_unknown"][0, 0])
    assert bool(debug["voxel_door_upper_solid_uses_out_of_range_unknown"][0, 0])
    assert int(debug["voxel_door_upper_solid_count_xy"][0, 0]) == (
        int(debug["voxel_door_upper_actual_occupied_count_xy"][0, 0])
        + int(debug["voxel_door_upper_out_of_range_unknown_count_xy"][0, 0])
    )
    assert float(debug["voxel_door_upper_in_range_unknown_ratio_active_xy"][0, 0]) > 0.25


def test_door_seed_counts_out_of_range_unknown_tail_as_upper_solid() -> None:
    z_centers = np.arange(0.125, 2.825, 0.05, dtype=np.float32)
    state = np.full((int(z_centers.size), 1, 1), VOXEL_UNKNOWN, dtype=np.uint8)
    sensor_range_count = np.zeros_like(state, dtype=np.uint8)

    lower_free = z_centers < 1.775
    actual_upper_occupied = (z_centers >= 1.825) & (z_centers < 1.975)
    upper_out_of_range_unknown = z_centers >= 1.975

    state[lower_free, 0, 0] = VOXEL_FREE
    state[actual_upper_occupied, 0, 0] = VOXEL_OCCUPIED
    state[upper_out_of_range_unknown, 0, 0] = VOXEL_UNKNOWN

    seed, reason_map, *_rest, debug = classify_voxel_door_seeds_vectorized(
        state,
        z_centers,
        np.arange(int(z_centers.size), dtype=np.int32),
        VoxelDoorDetectorConfig(),
        shape=(1, 1),
        return_debug=True,
        sensor_range_count=sensor_range_count,
    )

    assert bool(seed[0, 0])
    assert int(reason_map[0, 0]) == 1
    assert bool(debug["voxel_door_upper_solid_uses_out_of_range_unknown"][0, 0])
    assert int(debug["voxel_door_upper_out_of_range_unknown_count_xy"][0, 0]) > 0
    assert int(debug["voxel_door_upper_solid_count_xy"][0, 0]) > int(debug["voxel_door_upper_actual_occupied_count_xy"][0, 0])


def test_door_seed_counts_out_of_range_unknown_before_actual_occupied_as_upper_solid() -> None:
    z_centers = np.arange(0.125, 2.725, 0.05, dtype=np.float32)
    state = np.full((int(z_centers.size), 1, 1), VOXEL_UNKNOWN, dtype=np.uint8)
    sensor_range_count = np.zeros_like(state, dtype=np.uint8)

    lower_free = z_centers < 1.775
    actual_upper_occupied = z_centers >= 2.525

    state[lower_free, 0, 0] = VOXEL_FREE
    state[actual_upper_occupied, 0, 0] = VOXEL_OCCUPIED

    seed, reason_map, *_rest, debug = classify_voxel_door_seeds_vectorized(
        state,
        z_centers,
        np.arange(int(z_centers.size), dtype=np.int32),
        VoxelDoorDetectorConfig(),
        shape=(1, 1),
        return_debug=True,
        sensor_range_count=sensor_range_count,
    )

    assert bool(seed[0, 0])
    assert int(reason_map[0, 0]) == 1
    assert bool(debug["voxel_door_upper_solid_uses_out_of_range_unknown"][0, 0])
    assert int(debug["voxel_door_upper_solid_count_xy"][0, 0]) == (
        int(debug["voxel_door_upper_actual_occupied_count_xy"][0, 0])
        + int(debug["voxel_door_upper_out_of_range_unknown_count_xy"][0, 0])
    )
    assert float(debug["voxel_door_upper_solid_ratio_active_xy"][0, 0]) >= 0.8


def test_door_seed_accepts_actual_occupied_upper_column() -> None:
    z_centers = np.arange(0.125, 2.825, 0.05, dtype=np.float32)
    state = np.full((int(z_centers.size), 1, 1), VOXEL_UNKNOWN, dtype=np.uint8)

    lower_free = z_centers < 1.775
    upper_occupied = z_centers >= 1.825

    state[lower_free, 0, 0] = VOXEL_FREE
    state[upper_occupied, 0, 0] = VOXEL_OCCUPIED

    seed, reason_map, *_rest, debug = classify_voxel_door_seeds_vectorized(
        state,
        z_centers,
        np.arange(int(z_centers.size), dtype=np.int32),
        VoxelDoorDetectorConfig(),
        shape=(1, 1),
        return_debug=True,
    )

    assert bool(seed[0, 0])
    assert int(reason_map[0, 0]) == 1
    assert int(debug["voxel_door_upper_solid_count_xy"][0, 0]) == (
        int(debug["voxel_door_upper_actual_occupied_count_xy"][0, 0])
        + int(debug["voxel_door_upper_out_of_range_unknown_count_xy"][0, 0])
    )


def test_door_seed_scan_top_uses_point_ninety_five_ceiling_when_available() -> None:
    grid = _door_seed_grid((1, 1), [])
    grid.ceiling_height_m = 2.75
    grid.active_z_max_m = 2.30

    z_min, z_max, source = _door_scan_z_bounds(grid, VoxelDoorDetectorConfig())

    assert z_min == 0.10
    assert np.isclose(z_max, 2.6125)
    assert source == "ceiling_0.95"


def test_door_seed_scan_top_falls_back_to_three_meters_without_ceiling() -> None:
    grid = _door_seed_grid((1, 1), [])
    grid.ceiling_height_m = None
    grid.active_z_max_m = 2.30

    z_min, z_max, source = _door_scan_z_bounds(grid, VoxelDoorDetectorConfig())

    assert z_min == 0.10
    assert np.isclose(z_max, 3.0)
    assert source == "door_seed_scan_fallback"


def _door_seed_grid(shape: tuple[int, int], seed_cells: list[tuple[int, int]]) -> VoxelOccupancyGrid3D:
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        MapInfo(
            resolution_m=0.05,
            min_x=0.0,
            max_x=float(shape[1]) * 0.05,
            min_y=0.0,
            max_y=float(shape[0]) * 0.05,
            width=int(shape[1]),
            height=int(shape[0]),
        ),
        cfg={"z_min_m": 0.0, "z_max_m": 2.8, "z_resolution_m": 0.05, "active_z_max_fallback_m": 2.8},
    )
    z = grid.z_centers_m
    lower_free = z < 1.775
    upper_occupied = z >= 1.825
    for r, c in seed_cells:
        grid.state[lower_free, int(r), int(c)] = VOXEL_FREE
        grid.state[upper_occupied, int(r), int(c)] = VOXEL_OCCUPIED
    return grid


def test_door_seed_rejects_components_with_both_bbox_axes_under_three_cells() -> None:
    grid = _door_seed_grid((4, 4), [(1, 1), (1, 2), (2, 1), (2, 2)])
    result = classify_voxel_door_seeds(voxel_grid=grid, config=VoxelDoorDetectorConfig())

    assert int(np.count_nonzero(result.door_seed_mask)) == 0
    assert int(result.debug["voxel_door_seed_component_bbox_rejected_cells"]) == 4
    assert int(np.count_nonzero(result.debug["voxel_door_seed_component_bbox_too_small_cells"])) == 4
    assert result.debug["voxel_door_seed_component_bbox_rejected_components"][0]["bbox_height_cells"] == 2
    assert result.debug["voxel_door_seed_component_bbox_rejected_components"][0]["bbox_width_cells"] == 2


def test_door_seed_keeps_line_component_with_one_bbox_axis_at_three_cells() -> None:
    grid = _door_seed_grid((4, 4), [(1, 0), (1, 1), (1, 2)])
    result = classify_voxel_door_seeds(voxel_grid=grid, config=VoxelDoorDetectorConfig())

    assert int(np.count_nonzero(result.door_seed_mask)) == 3
    assert int(result.debug["voxel_door_seed_component_bbox_rejected_cells"]) == 0
    assert int(np.max(result.door_seed_component_map)) == 1


def test_door_seed_requires_navigation_free_cells() -> None:
    grid = _door_seed_grid((4, 5), [(1, 0), (1, 1), (1, 2), (1, 3)])
    navigation_free = np.ones((4, 5), dtype=bool)
    navigation_free[1, 3] = False

    result = classify_voxel_door_seeds(
        voxel_grid=grid,
        config=VoxelDoorDetectorConfig(require_navigation_free_seed=True),
        navigation_free_mask=navigation_free,
    )

    assert int(np.count_nonzero(result.door_seed_mask)) == 3
    assert bool(result.door_seed_mask[1, 0])
    assert bool(result.door_seed_mask[1, 1])
    assert bool(result.door_seed_mask[1, 2])
    assert not bool(result.door_seed_mask[1, 3])
    assert int(result.debug["voxel_door_seed_navigation_free_rejected_cells"]) == 1
    assert bool(result.debug["voxel_door_seed_not_navigation_free_cells"][1, 3])
    assert int(result.debug["voxel_door_seed_component_bbox_rejected_cells"]) == 0


def test_door_seed_uses_explicit_no_clearance_navigation_free_mask() -> None:
    grid = _door_seed_grid((4, 5), [(1, 0), (1, 1), (1, 2), (1, 3)])
    clearance_nav_free = np.ones((4, 5), dtype=bool)
    clearance_nav_free[1, 0] = False
    no_clearance_nav_free = np.ones((4, 5), dtype=bool)
    no_clearance_nav_free[1, 3] = False

    result = classify_voxel_door_seeds(
        voxel_grid=grid,
        config=VoxelDoorDetectorConfig(require_navigation_free_seed=True),
        navigation_free_mask=clearance_nav_free,
        no_clearance_navigation_free_mask=no_clearance_nav_free,
    )

    assert int(np.count_nonzero(result.door_seed_mask)) == 3
    assert bool(result.door_seed_mask[1, 0])
    assert not bool(result.door_seed_mask[1, 3])
    assert result.debug["voxel_door_seed_navigation_free_source"] == "no_clearance_navigation_free_mask"
    assert result.debug["voxel_door_seed_uses_no_clearance_navigation_free"] is True


def test_seed_line_spur_pruning_keeps_short_branches_only() -> None:
    cfg = VoxelDoorDetectorConfig(
        primitive_min_contiguous_seed_run_cells=7,
        seed_line_spur_prune_max_branch_length_ratio=0.35,
        seed_line_spur_prune_max_total_branch_ratio=0.45,
    )
    main = [(10, c) for c in range(12)] + [(10, c) for c in range(16, 20)]
    short_branch = [(14, c) for c in range(5)]

    pruned = _extract_spur_pruned_seed_line_segments([*main, *short_branch], cfg=cfg)

    assert pruned
    assert pruned[0][0] == sorted(main)
    assert pruned[0][1]["spur_pruned_longest_branch_cells"] == 5
    assert pruned[0][1]["spur_pruned_same_line_cells_retained"] == len(main)

    long_branch = [(14, c) for c in range(10)]

    assert _extract_spur_pruned_seed_line_segments([*main, *long_branch], cfg=cfg) == []


def test_primitive_contiguous_seed_run_allows_two_cell_internal_gap() -> None:
    cfg = VoxelDoorDetectorConfig(
        primitive_min_contiguous_seed_run_cells=7,
        primitive_contiguous_gap_cells=3.0,
    )
    cells = [(10, c) for c in range(20, 26)] + [(10, c) for c in range(28, 34)]
    center = np.asarray([10.0, 26.5], dtype=np.float32)
    major = np.asarray([0.0, 1.0], dtype=np.float32)

    _along_min, _along_max, max_gap, _segment_count, longest_run = _primitive_along_stats(cells, center, major, cfg)

    assert max_gap == 3.0
    assert longest_run == 12


def test_disabling_seed_line_regression_gate_only_skips_corr_and_variance() -> None:
    base_kwargs = dict(
        seed_count=7,
        length_cells=7.0,
        thickness_cells=1.0,
        residual_cells=0.5,
        elongation=2.0,
        max_gap=0.0,
        longest_contiguous_run_cells=7,
        line_correlation=0.50,
        orthogonal_variance_cells2=99.0,
    )

    assert _primitive_reject_reason(**base_kwargs, cfg=VoxelDoorDetectorConfig()) == "primitive_line_correlation_too_low"
    assert (
        _primitive_reject_reason(
            **base_kwargs,
            cfg=VoxelDoorDetectorConfig(primitive_require_line_fit_quality=False),
        )
        is None
    )
    assert (
        _primitive_reject_reason(
            **{**base_kwargs, "residual_cells": 99.0},
            cfg=VoxelDoorDetectorConfig(primitive_require_line_fit_quality=False),
        )
        == "primitive_residual_too_high"
    )


def test_parallel_seed_line_pruning_keeps_one_of_similar_parallel_runs() -> None:
    cfg = VoxelDoorDetectorConfig(primitive_min_contiguous_seed_run_cells=7)
    upper = [(10, c) for c in range(20, 40)]
    lower = [(16, c) for c in range(21, 39)]
    connector = [(r, 41) for r in range(12, 16)]

    pruned = _extract_parallel_pruned_seed_line_segments([*upper, *lower, *connector], cfg=cfg)

    assert pruned
    kept, debug = pruned[0]
    assert len(kept) == len(lower)
    assert all(r == 16 for r, _c in kept)
    assert len({r for r, _c in kept}) == 1
    assert debug["parallel_pruned_axis"] == "h"
    assert debug["parallel_pruned_distance_cells"] == 6
    assert debug["parallel_pruned_overlap_ratio"] >= 0.9
    assert debug["parallel_pruned_kept_cells"] == len(lower)
    assert debug["parallel_pruned_removed_cells"] == len(upper)


def test_door_line_inner_real_wall_gate_allows_up_to_configured_cells() -> None:
    cfg = VoxelDoorDetectorConfig(door_line_inner_real_wall_cells_max=8)
    line = [(2, c) for c in range(20)]
    seed = np.zeros((5, 24), dtype=bool)
    seed[2, 10] = True
    free = np.ones((5, 24), dtype=bool)
    wall = np.zeros((5, 24), dtype=bool)
    real_wall = np.zeros((5, 24), dtype=bool)
    real_wall[2, 1:9] = True

    ok, reason, debug = validate_door_line_local_neck(line, seed, free, wall, real_wall, 0.05, cfg)

    assert ok
    assert reason is None
    assert debug["door_line_inner_real_wall_cells"] == 8
    assert debug["door_line_inner_real_wall_cells_max"] == 8

    real_wall[2, 12] = True

    ok, reason, debug = validate_door_line_local_neck(line, seed, free, wall, real_wall, 0.05, cfg)

    assert not ok
    assert reason == "door_line_crosses_real_wall"
    assert debug["door_line_inner_real_wall_cells"] == 9


def test_door_line_seed_overlap_allows_two_cell_anchor_snap_offset() -> None:
    cfg = VoxelDoorDetectorConfig(partition_cut_seed_dilation_cells=2)
    line = [(10, c) for c in range(20, 36)]
    seed = np.zeros((16, 48), dtype=bool)
    seed[12, 22:34] = True
    free = np.ones((16, 48), dtype=bool)
    wall = np.zeros((16, 48), dtype=bool)
    real_wall = np.zeros((16, 48), dtype=bool)

    ok, reason, debug = validate_door_line_local_neck(line, seed, free, wall, real_wall, 0.05, cfg)

    assert ok
    assert reason is None
    assert debug["door_line_seed_overlap"] is True


def test_partition_cut_bridges_small_unknown_gap_up_to_ten_cells() -> None:
    shape = (16, 64)
    full_line = [(10, c) for c in range(20, 52)]
    free = np.ones(shape, dtype=bool)
    unknown = np.zeros(shape, dtype=bool)
    real_wall = np.zeros(shape, dtype=bool)
    seed = np.zeros(shape, dtype=bool)
    seed[12, 24:34] = True
    real_wall[10, 20] = True
    real_wall[10, 51] = True
    free[10, 36:46] = False
    unknown[10, 36:46] = True

    result = build_door_partition_cut_v30(
        full_line_cells=full_line,
        seed_mask=seed,
        accepted_seed_mask=seed,
        partition_free=free,
        partition_unknown=unknown,
        real_wall_barrier=real_wall,
        anchor_a=(10, 20),
        anchor_b=(10, 51),
        max_unknown_bridge_gap_cells=10,
        max_nonfree_bridge_gap_cells=1,
        max_endpoint_wall_gap_cells=1,
        seed_dilation_cells=2,
        min_cut_cells=1,
    )

    assert result.cut_cells
    assert result.debug["door_partition_cut_v30_closed_to_wall"] is True
    assert len(result.bridged_cells) == 10
    assert [(10, c) for c in range(36, 46)] == result.bridged_cells


def test_door_extension_rejects_three_no_clearance_occupied_cells_excluding_walls() -> None:
    cfg = VoxelDoorDetectorConfig(
        door_extension_no_clearance_occupied_reject_cells=3,
        door_extension_no_clearance_occupied_surface_tolerance_cells=0,
    )
    occupied = np.zeros((5, 12), dtype=bool)
    real_wall = np.zeros((5, 12), dtype=bool)
    cells = [(2, c) for c in range(1, 10)]
    anchors = [(2, 1), (2, 9)]
    real_wall[2, 1] = True
    real_wall[2, 9] = True

    occupied[2, [1, 3, 5, 9]] = True
    rejected, debug = _door_extension_no_clearance_occupied_reject(
        cells,
        no_clearance_occupied=occupied,
        real_wall=real_wall,
        anchors=anchors,
        cfg=cfg,
    )

    assert not rejected
    assert debug["door_extension_no_clearance_occupied_cells"] == 2

    occupied[2, 7] = True
    rejected, debug = _door_extension_no_clearance_occupied_reject(
        cells,
        no_clearance_occupied=occupied,
        real_wall=real_wall,
        anchors=anchors,
        cfg=cfg,
    )

    assert rejected
    assert debug["door_extension_no_clearance_occupied_cells"] == 3
    assert debug["door_extension_no_clearance_occupied_hit_cells"] == [[2, 3], [2, 5], [2, 7]]


def test_door_extension_ignores_one_cell_no_clearance_occupied_surface_layer() -> None:
    cfg = VoxelDoorDetectorConfig(
        door_extension_no_clearance_occupied_reject_cells=3,
        door_extension_no_clearance_occupied_surface_tolerance_cells=1,
    )
    occupied = np.zeros((7, 14), dtype=bool)
    real_wall = np.zeros((7, 14), dtype=bool)
    cells = [(3, c) for c in range(1, 13)]
    anchors = [(3, 1), (3, 12)]
    real_wall[3, 1] = True
    real_wall[3, 12] = True

    occupied[3, 3:6] = True
    rejected, debug = _door_extension_no_clearance_occupied_reject(
        cells,
        no_clearance_occupied=occupied,
        real_wall=real_wall,
        anchors=anchors,
        cfg=cfg,
    )

    assert not rejected
    assert debug["door_extension_no_clearance_occupied_cells"] == 0
    assert debug["door_extension_no_clearance_occupied_surface_ignored_cells"] == 3

    occupied[2:5, 8:11] = True
    rejected, debug = _door_extension_no_clearance_occupied_reject(
        cells,
        no_clearance_occupied=occupied,
        real_wall=real_wall,
        anchors=anchors,
        cfg=cfg,
    )

    assert not rejected
    assert debug["door_extension_no_clearance_occupied_cells"] == 1
    assert debug["door_extension_no_clearance_occupied_hit_cells"] == [[3, 9]]
    assert debug["door_extension_no_clearance_occupied_surface_ignored_cells"] == 5


def test_wall_walk_uses_first_supported_no_clearance_occupied_boundary_anchor() -> None:
    wall = np.zeros((24, 48), dtype=bool)
    no_clearance_occupied = np.zeros_like(wall)
    no_clearance_occupied[10, 10:20] = True
    wall[10, 10] = True
    same_seed = np.zeros_like(wall)
    other_seed = np.zeros_like(wall)
    unknown = np.zeros_like(wall)

    result = _walk_to_wall(
        start_rc=np.asarray([10.0, 20.0], dtype=np.float32),
        direction=np.asarray([0.0, -1.0], dtype=np.float32),
        wall_anchor_map=wall,
        wall_anchor_source_map=None,
        same_cluster_mask=same_seed,
        other_door_mask=other_seed,
        unknown_map=unknown,
        no_clearance_occupied_map=no_clearance_occupied,
        resolution_m=0.05,
        cfg=VoxelDoorDetectorConfig(door_walk_no_clearance_occupied_anchor_enabled=True),
    )

    assert result.anchor == (10, 19)
    assert result.status == "hit_no_clearance_occupied_anchor"
    assert result.anchor_source == 6


def test_door_memory_freezes_geometry_until_ceiling_height_changes() -> None:
    shape = (16, 16)

    def candidate(candidate_id: int, cells: list[tuple[int, int]]) -> VoxelDoorLineCandidate:
        center = tuple(float(v) for v in np.mean(np.asarray(cells, dtype=np.float32), axis=0))
        return VoxelDoorLineCandidate(
            candidate_id=candidate_id,
            seed_component_id=candidate_id,
            seed_cells=list(cells),
            center_rc=(center[0], center[1]),
            major_dir_rc=(0.0, 1.0),
            minor_dir_rc=(1.0, 0.0),
            seed_projected_centerline_cells=list(cells),
            extended_centerline_cells=list(cells),
            door_cut_cells=list(cells),
            wall_anchor_a=cells[0],
            wall_anchor_b=cells[-1],
            width_m=float(len(cells)) * 0.05,
            accepted=True,
            reject_reason=None,
            debug={"partition_effective_verified": True, "partition_accepted": True},
        )

    def observation(cells: list[tuple[int, int]], ceiling_height_m: float) -> DoorMemoryObservationMaps:
        mask = np.zeros(shape, dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return DoorMemoryObservationMaps(
            observed_xy=np.ones(shape, dtype=bool),
            sensor_range_xy=np.ones(shape, dtype=bool),
            vertical_free_xy=np.ones(shape, dtype=bool),
            wall_xy=np.zeros(shape, dtype=bool),
            raw_seed_mask=mask,
            current_verified_cut_mask=mask,
            ceiling_height_m=ceiling_height_m,
        )

    first_cells = [(5, c) for c in range(2, 9)]
    shifted_cells = [(5, c) for c in range(3, 11)]
    memory = VoxelDoorMemory(
        VoxelDoorDetectorConfig(
            door_memory_freeze_geometry_unless_ceiling_changes=True,
            door_memory_ceiling_change_threshold_m=0.05,
        )
    )

    result = memory.update([candidate(1, first_cells)], step=1, shape=shape, observation=observation(first_cells, 2.70))
    assert set(result.tracks[0].cut_cells) == set(first_cells)

    result = memory.update([candidate(2, shifted_cells)], step=2, shape=shape, observation=observation(shifted_cells, 2.70))
    assert set(result.tracks[0].cut_cells) == set(first_cells)
    assert result.debug["voxel_door_memory_verified_update_replaced_geometry"] == 0
    assert result.debug["voxel_door_memory_geometry_frozen_without_ceiling_change"] == 1

    result = memory.update([candidate(3, shifted_cells)], step=3, shape=shape, observation=observation(shifted_cells, 2.80))
    assert set(result.tracks[0].cut_cells) == set(shifted_cells)
    assert result.debug["voxel_door_memory_verified_update_replaced_geometry"] == 1
