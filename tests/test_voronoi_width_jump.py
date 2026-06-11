from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.baselines.offline.fallback_voronoi_width_jump import (
    CriticalLine,
    _reject_parallel_interval_candidates,
    _reject_parallel_critical_lines,
    _reject_separator_masks_creating_small_regions,
    voronoi_segment,
)


def test_width_jump_voronoi_finds_abrupt_bottleneck():
    free = np.zeros((80, 110), dtype=bool)
    free[10:30, 45:65] = True
    free[30:70, 10:100] = True

    labels, metadata, debug = voronoi_segment(free, resolution_m=0.05)

    assert metadata["method"] == "voronoi_width_jump"
    assert int(np.count_nonzero(debug["voronoi_critical_point_mask"])) > 0
    assert int(np.count_nonzero(debug["voronoi_horizontal_width_jump_candidate_mask"])) > 0
    assert int(np.max(labels)) >= 1


def test_width_jump_voronoi_checks_horizontal_axis_too():
    free = np.zeros((110, 80), dtype=bool)
    free[45:65, 10:30] = True
    free[10:100, 30:70] = True

    labels, metadata, debug = voronoi_segment(free, resolution_m=0.05)

    assert metadata["method"] == "voronoi_width_jump"
    assert int(np.count_nonzero(debug["voronoi_critical_point_mask"])) > 0
    assert int(np.count_nonzero(debug["voronoi_vertical_width_jump_candidate_mask"])) > 0
    assert int(np.max(labels)) >= 1


def test_width_jump_voronoi_interval_merge_does_not_assume_two_segments():
    free = np.zeros((80, 120), dtype=bool)
    free[12:35, 12:32] = True
    free[12:35, 52:72] = True
    free[35:68, 12:52] = True
    free[35:68, 72:112] = True

    labels, metadata, debug = voronoi_segment(free, resolution_m=0.05)

    assert metadata["method"] == "voronoi_width_jump"
    assert int(np.count_nonzero(debug["voronoi_horizontal_width_jump_candidate_mask"])) > 0
    assert int(np.count_nonzero(debug["voronoi_critical_point_mask"])) > 0
    assert int(np.max(labels)) >= 1


def test_width_jump_voronoi_ignores_small_width_fluctuations():
    free = np.zeros((60, 110), dtype=bool)
    free[22:31, 10:92] = True
    free[31:40, 10:100] = True

    labels, metadata, debug = voronoi_segment(free, resolution_m=0.05)

    assert metadata["critical_point_rule"] == "single_cell_scanline_free_interval_width_change_ge_min_drop"
    assert int(np.count_nonzero(debug["voronoi_critical_point_mask"])) == 0
    assert int(np.count_nonzero(debug["voronoi_horizontal_width_jump_candidate_mask"])) == 0
    assert int(np.count_nonzero(debug["voronoi_vertical_width_jump_candidate_mask"])) == 0
    assert int(np.max(labels)) == 1


def test_width_jump_voronoi_fills_small_nonstructural_input_holes_before_scan():
    free = np.zeros((80, 100), dtype=bool)
    free[10:70, 20:80] = True
    free[30:40, 50] = False

    labels, metadata, debug = voronoi_segment(free, resolution_m=0.05)

    assert metadata["stats"]["input_hole_filled_cells"] == 10
    assert int(np.count_nonzero(debug["voronoi_input_hole_filled_mask"])) == 10
    assert int(np.count_nonzero(debug["voronoi_critical_point_mask"])) == 0
    assert int(np.max(labels)) == 1


def test_parallel_interval_reject_uses_line_spacing_not_endpoint_spacing():
    candidates = [
        {"row": 10, "col": 8, "gap_start": 6, "gap_end": 10, "score": 10.0},
        {"row": 20, "col": 90, "gap_start": 88, "gap_end": 92, "score": 9.0},
        {"row": 55, "col": 12, "gap_start": 10, "gap_end": 14, "score": 8.0},
    ]

    accepted = _reject_parallel_interval_candidates(
        candidates,
        resolution_m=0.05,
        min_distance_m=0.20,
        max_distance_m=1.50,
    )

    assert accepted == candidates


def test_parallel_interval_reject_requires_rectangle_like_pair():
    candidates = [
        {"row": 10, "col": 21, "gap_start": 18, "gap_end": 24, "score": 10.0},
        {"row": 20, "col": 22, "gap_start": 19, "gap_end": 25, "score": 9.0},
        {"row": 55, "col": 12, "gap_start": 10, "gap_end": 14, "score": 8.0},
    ]

    accepted = _reject_parallel_interval_candidates(
        candidates,
        resolution_m=0.05,
        min_distance_m=0.20,
        max_distance_m=1.50,
    )

    assert accepted == [candidates[2]]


def test_parallel_interval_reject_is_one_pass_forward_scan():
    candidates = [
        {"row": 10, "col": 21, "gap_start": 18, "gap_end": 24, "score": 10.0},
        {"row": 20, "col": 22, "gap_start": 19, "gap_end": 25, "score": 9.0},
        {"row": 30, "col": 21, "gap_start": 18, "gap_end": 24, "score": 8.0},
    ]

    accepted = _reject_parallel_interval_candidates(
        candidates,
        resolution_m=0.05,
        min_distance_m=0.20,
        max_distance_m=1.50,
    )

    assert accepted == [candidates[2]]


def test_parallel_reject_applies_to_width_jump_critical_lines():
    lines = [
        CriticalLine((10, 20), (10, 10), (10, 30), 120.0, 21.0),
        CriticalLine((20, 21), (20, 11), (20, 31), 120.0, 21.0),
        CriticalLine((55, 20), (55, 10), (55, 30), 120.0, 21.0),
    ]

    accepted = _reject_parallel_critical_lines(
        lines,
        resolution_m=0.05,
        min_distance_m=0.20,
        max_distance_m=1.50,
    )

    assert accepted == [lines[2]]


def test_parallel_reject_keeps_shifted_width_jump_critical_lines():
    lines = [
        CriticalLine((10, 20), (10, 10), (10, 30), 120.0, 21.0),
        CriticalLine((20, 80), (20, 70), (20, 90), 120.0, 21.0),
    ]

    accepted = _reject_parallel_critical_lines(
        lines,
        resolution_m=0.05,
        min_distance_m=0.20,
        max_distance_m=1.50,
    )

    assert accepted == lines


def test_width_jump_voronoi_records_global_area_split_contract():
    free = np.zeros((80, 120), dtype=bool)
    free[12:35, 12:32] = True
    free[12:35, 52:72] = True
    free[35:68, 12:52] = True
    free[35:68, 72:112] = True

    _, metadata, _ = voronoi_segment(free, resolution_m=0.05)

    assert (
        metadata["width_jump_parallel_reject_scan_order"]
        == "top_to_bottom_for_horizontal_scans_left_to_right_for_vertical_scans"
    )
    assert (
        metadata["width_jump_split_validation_domain"]
        == "combined_separator_after_full_scan"
    )
    assert metadata["width_jump_min_created_region_area_m2"] == 2.0
    assert metadata["width_jump_min_drop_m"] == 1.0
    assert metadata["width_jump_input_hole_fill_max_area_m2"] == 1.0


def test_global_separator_area_reject_removes_lines_touching_tiny_created_regions():
    free = np.zeros((50, 80), dtype=bool)
    free[5:45, 5:75] = True
    left_separator = np.zeros_like(free)
    right_separator = np.zeros_like(free)
    left_separator[5:45, 16] = True
    right_separator[5:45, 23] = True

    kept, rejected = _reject_separator_masks_creating_small_regions(
        free,
        [left_separator, right_separator],
        resolution_m=0.05,
        min_created_region_area_m2=2.0,
    )

    assert not bool(np.any(kept))
    assert int(np.count_nonzero(rejected)) == int(np.count_nonzero(left_separator | right_separator))
