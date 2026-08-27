from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from voxroom_online.isaac_runtime.baselines.data_contract import resolve_map_info
from voxroom_online.isaac_runtime.baselines.mask_io import relabel_consecutive

from .fallback_utils import draw_grid_line, label_components, nearest_seed_fill


ORIGINAL_REPO = "https://github.com/ipa320/ipa_coverage_planning"
ORIGINAL_REPO_COMMIT = "986c18384ed884dadd3bc857cd0c47c13b7d4716"
ORIGINAL_PACKAGE = "ipa_room_segmentation"
ORIGINAL_ACTION = "ipa_building_msgs/MapSegmentation.action"
VORONOI_ALGORITHM_ID = 3
STRUCTURE_4 = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@dataclass(frozen=True)
class VoronoiFallbackParameters:
    map_resolution_m: float = 0.05
    room_area_factor_lower_limit_voronoi: float = 0.1
    room_area_factor_upper_limit_voronoi: float = 1000000.0
    voronoi_neighborhood_index: int = 280
    max_iterations: int = 150
    min_critical_point_distance_factor: float = 0.5
    merge_area_threshold_m2: float = 0.0
    force_merge_area_m2: float = 0.0
    min_critical_line_angle_deg: float = 95.0
    skeleton_min_distance_cells: float = 1.0
    prune_iterations: int = 30
    width_smoothing_cells: int = 9
    width_jump_context_cells: int = 18
    width_jump_min_drop_m: float = 1.0
    width_jump_min_drop_ratio: float = 0.25
    width_jump_min_gradient_cells: float = 3.0
    width_jump_min_spacing_cells: int = 12
    width_jump_min_component_length_cells: int = 12
    width_jump_interval_merge_context_cells: int = 9
    width_jump_interval_merge_min_support_lines: int = 3
    width_jump_interval_merge_transition_radius_cells: int = 2
    width_jump_parallel_reject_min_distance_m: float = 0.20
    width_jump_parallel_reject_max_distance_m: float = 1.50
    width_jump_parallel_reject_max_length_delta_ratio: float = 0.20
    width_jump_parallel_reject_max_scanline_shift_ratio: float = 0.20
    width_jump_min_created_region_area_m2: float = 2.0
    width_jump_input_hole_fill_max_area_m2: float = 1.0


@dataclass(frozen=True)
class VoronoiFallbackStats:
    free_cells: int
    ridge_cells: int
    pruned_ridge_cells: int
    critical_candidates: int
    critical_lines_drawn: int
    critical_lines_rejected_parallel: int
    separator_cells_rejected_small_region: int
    input_hole_filled_cells: int
    initial_region_count: int
    merged_region_count: int
    final_room_count: int


@dataclass(frozen=True)
class CriticalLine:
    point_rc: tuple[int, int]
    basis_a_rc: tuple[int, int]
    basis_b_rc: tuple[int, int]
    angle_deg: float
    length_cells: float


def build_voronoi_free_mask(arrays: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the IPA/Voronoi accessible domain; unknown is never marked free."""

    shape = _infer_shape(arrays)
    candidate, candidate_key = _first_bool_array(
        arrays,
        (
            "navigation_free_room_domain",
            "observed_free_mask",
            "voxel_nav_free_xy",
            "vertical_free_room_domain",
            "voxel_vertical_free_xy",
        ),
        shape,
        require_any=True,
    )
    if candidate is None:
        occupancy, occupancy_key = _first_bool_array(arrays, ("occupancy_map", "voxel_nav_occupied_xy"), shape)
        if occupancy is None:
            candidate = np.zeros(shape, dtype=bool)
            candidate_key = "missing_free_source"
        else:
            candidate = ~occupancy
            candidate_key = "not_%s" % occupancy_key

    unknown, unknown_key = _first_bool_array(arrays, ("unknown_mask", "voxel_nav_unknown_xy", "voxel_unknown_xy"), shape)
    obstacle, obstacle_key = _first_bool_array(arrays, ("obstacle_mask", "occupancy_map", "voxel_nav_occupied_xy", "voxel_wall_xy"), shape)
    if unknown is None:
        unknown = np.zeros(shape, dtype=bool)
        unknown_key = "absent"
    if obstacle is None:
        obstacle = np.zeros(shape, dtype=bool)
        obstacle_key = "absent"

    candidate = np.asarray(candidate, dtype=bool)
    unknown = np.asarray(unknown, dtype=bool)
    obstacle = np.asarray(obstacle, dtype=bool)
    free = candidate & ~unknown & ~obstacle
    metadata = {
        "free_source": str(candidate_key),
        "unknown_source": str(unknown_key),
        "obstacle_source": str(obstacle_key),
        "candidate_free_cells": int(np.count_nonzero(candidate)),
        "unknown_cells": int(np.count_nonzero(unknown)),
        "obstacle_cells": int(np.count_nonzero(obstacle)),
        "unknown_excluded_from_free_cells": int(np.count_nonzero(candidate & unknown)),
        "obstacle_excluded_from_free_cells": int(np.count_nonzero(candidate & obstacle)),
        "final_free_cells": int(np.count_nonzero(free)),
        "unknown_treated_as_free": False,
    }
    return free.astype(bool), metadata


def build_voronoi_ipa_input_image(arrays: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    free_mask, metadata = build_voronoi_free_mask(arrays)
    ipa_image = np.zeros(free_mask.shape, dtype=np.uint8)
    ipa_image[free_mask] = np.uint8(255)
    metadata.update(
        {
            "ipa_input_encoding": "mono8",
            "ipa_input_free_value": 255,
            "ipa_input_inaccessible_value": 0,
        }
    )
    return ipa_image, free_mask, metadata


def _scanline_structural_blocker_mask(arrays: Mapping[str, Any], shape: tuple[int, int]) -> np.ndarray:
    blocker = np.zeros(shape, dtype=bool)
    for key in (
        "voxroom_wall_door_no_extension_no_mincc_barrier_mask",
        "voxroom_wall_door_no_extension_no_mincc_despurred_mask",
        "voxroom_wall_door_no_extension_no_mincc_closed_mask",
        "voxroom_wall_door_no_extension_no_mincc_raw_mask",
        "voxroom_step1_completed_wall_mask",
        "voxroom_door_runtime_accepted_cut_mask",
        "voxroom_door_line_mask",
        "voxel_wall_after_step1_map",
    ):
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=bool)
        if arr.shape == shape:
            blocker |= arr
    return blocker.astype(bool)


def segment_snapshot_arrays(
    snapshot_path: Path | str,
    arrays: Mapping[str, Any],
    *,
    default_resolution_m: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Run the smoke-only Python Voronoi fallback on one replay snapshot."""

    snapshot_path = Path(snapshot_path)
    map_info = resolve_map_info(snapshot_arrays=arrays, default_resolution_m=float(default_resolution_m))
    free_mask, domain_metadata = build_voronoi_free_mask(arrays)
    params = VoronoiFallbackParameters(map_resolution_m=float(map_info.resolution_m))
    structural_blocker = _scanline_structural_blocker_mask(arrays, free_mask.shape)
    labels, algorithm_metadata, debug_arrays = voronoi_segment(
        free_mask,
        resolution_m=float(map_info.resolution_m),
        structural_blocker_mask=structural_blocker,
        merge_area_threshold_m2=float(params.merge_area_threshold_m2),
        force_merge_area_m2=float(params.force_merge_area_m2),
        room_area_factor_lower_limit_voronoi=float(params.room_area_factor_lower_limit_voronoi),
        room_area_factor_upper_limit_voronoi=float(params.room_area_factor_upper_limit_voronoi),
        voronoi_neighborhood_index=int(params.voronoi_neighborhood_index),
        max_iterations=int(params.max_iterations),
        min_critical_point_distance_factor=float(params.min_critical_point_distance_factor),
        min_critical_line_angle_deg=float(params.min_critical_line_angle_deg),
        skeleton_min_distance_cells=float(params.skeleton_min_distance_cells),
        prune_iterations=int(params.prune_iterations),
        width_jump_input_hole_fill_max_area_m2=float(params.width_jump_input_hole_fill_max_area_m2),
        width_jump_min_created_region_area_m2=float(params.width_jump_min_created_region_area_m2),
    )
    metadata = build_fallback_metadata(
        source_snapshot=snapshot_path,
        input_free_definition=str(domain_metadata["free_source"]),
        map_resolution_m=float(map_info.resolution_m),
        map_origin_xy_m=(float(map_info.min_x), float(map_info.min_y)),
        parameters=params,
        algorithm_metadata=algorithm_metadata,
        domain_metadata=domain_metadata,
    )
    return np.asarray(labels, dtype=np.int32), metadata, debug_arrays


def voronoi_segment(
    free_mask: np.ndarray,
    *,
    resolution_m: float = 0.05,
    structural_blocker_mask: np.ndarray | None = None,
    merge_area_threshold_m2: float = 0.0,
    force_merge_area_m2: float = 0.0,
    room_area_factor_lower_limit_voronoi: float = 0.1,
    room_area_factor_upper_limit_voronoi: float = 1000000.0,
    voronoi_neighborhood_index: int = 280,
    max_iterations: int = 150,
    min_critical_point_distance_factor: float = 0.5,
    min_critical_line_angle_deg: float = 95.0,
    skeleton_min_distance_cells: float = 1.0,
    prune_iterations: int = 30,
    width_smoothing_cells: int = 9,
    width_jump_context_cells: int = 18,
    width_jump_min_drop_m: float = 1.0,
    width_jump_min_drop_ratio: float = 0.25,
    width_jump_min_gradient_cells: float = 3.0,
    width_jump_min_spacing_cells: int = 12,
    width_jump_min_component_length_cells: int = 12,
    width_jump_interval_merge_context_cells: int = 9,
    width_jump_interval_merge_min_support_lines: int = 3,
    width_jump_interval_merge_transition_radius_cells: int = 2,
    width_jump_parallel_reject_min_distance_m: float = 0.20,
    width_jump_parallel_reject_max_distance_m: float = 1.50,
    width_jump_parallel_reject_max_length_delta_ratio: float = 0.20,
    width_jump_parallel_reject_max_scanline_shift_ratio: float = 0.20,
    width_jump_min_created_region_area_m2: float = 2.0,
    width_jump_input_hole_fill_max_area_m2: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Voronoi variant that only cuts at abrupt medial-axis width changes.

    This preserves the current fallback Voronoi pipeline shape but replaces the
    "many narrow skeleton points" selector with a smoothed width-jump detector.
    Gradual width changes and small local width noise are ignored.
    """

    free = np.asarray(free_mask, dtype=bool)
    if free.ndim != 2:
        raise ValueError(f"free_mask must be 2D, got shape={free.shape}")
    if float(resolution_m) <= 0:
        raise ValueError("resolution_m must be positive")

    params = VoronoiFallbackParameters(
        map_resolution_m=float(resolution_m),
        room_area_factor_lower_limit_voronoi=float(room_area_factor_lower_limit_voronoi),
        room_area_factor_upper_limit_voronoi=float(room_area_factor_upper_limit_voronoi),
        voronoi_neighborhood_index=int(voronoi_neighborhood_index),
        max_iterations=int(max_iterations),
        min_critical_point_distance_factor=float(min_critical_point_distance_factor),
        merge_area_threshold_m2=float(merge_area_threshold_m2),
        force_merge_area_m2=float(force_merge_area_m2),
        min_critical_line_angle_deg=float(min_critical_line_angle_deg),
        skeleton_min_distance_cells=float(skeleton_min_distance_cells),
        prune_iterations=int(prune_iterations),
        width_smoothing_cells=int(width_smoothing_cells),
        width_jump_context_cells=int(width_jump_context_cells),
        width_jump_min_drop_m=float(width_jump_min_drop_m),
        width_jump_min_drop_ratio=float(width_jump_min_drop_ratio),
        width_jump_min_gradient_cells=float(width_jump_min_gradient_cells),
        width_jump_min_spacing_cells=int(width_jump_min_spacing_cells),
        width_jump_min_component_length_cells=int(width_jump_min_component_length_cells),
        width_jump_interval_merge_context_cells=int(width_jump_interval_merge_context_cells),
        width_jump_interval_merge_min_support_lines=int(width_jump_interval_merge_min_support_lines),
        width_jump_interval_merge_transition_radius_cells=int(width_jump_interval_merge_transition_radius_cells),
        width_jump_parallel_reject_min_distance_m=float(width_jump_parallel_reject_min_distance_m),
        width_jump_parallel_reject_max_distance_m=float(width_jump_parallel_reject_max_distance_m),
        width_jump_parallel_reject_max_length_delta_ratio=float(width_jump_parallel_reject_max_length_delta_ratio),
        width_jump_parallel_reject_max_scanline_shift_ratio=float(width_jump_parallel_reject_max_scanline_shift_ratio),
        width_jump_min_created_region_area_m2=float(width_jump_min_created_region_area_m2),
        width_jump_input_hole_fill_max_area_m2=float(width_jump_input_hole_fill_max_area_m2),
    )
    raw_free = np.asarray(free_mask, dtype=bool)
    structural_blocker = (
        np.zeros(raw_free.shape, dtype=bool)
        if structural_blocker_mask is None
        else np.asarray(structural_blocker_mask, dtype=bool)
    )
    if structural_blocker.shape != raw_free.shape:
        structural_blocker = np.zeros(raw_free.shape, dtype=bool)
    free, input_hole_filled_mask = _fill_small_nonstructural_input_holes(
        raw_free,
        structural_blocker,
        resolution_m=float(params.map_resolution_m),
        max_area_m2=float(params.width_jump_input_hole_fill_max_area_m2),
    )
    if not bool(np.any(free)):
        empty = np.zeros(free.shape, dtype=np.int32)
        stats = VoronoiFallbackStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return empty, _algorithm_metadata(params, stats, "empty_free_domain"), {}

    distance_cells = ndimage.distance_transform_edt(free).astype(np.float32)
    ridge = _approximate_voronoi_ridge(
        distance_cells,
        free,
        min_distance_cells=float(params.skeleton_min_distance_cells),
    )
    pruned_ridge = _prune_skeleton_endpoints(ridge, iterations=min(int(params.prune_iterations), int(params.max_iterations)))
    (
        critical_candidates,
        smoothed_width_cells,
        width_jump_drop_cells,
        width_jump_score,
        horizontal_candidates,
        vertical_candidates,
        horizontal_width_cells,
        vertical_width_cells,
        horizontal_score,
        vertical_score,
        separator_masks,
        pre_parallel_reject_line_count,
        rejected_parallel_critical_lines,
    ) = _scanline_width_jump_separators(
        free,
        resolution_m=float(params.map_resolution_m),
        min_drop_m=float(params.width_jump_min_drop_m),
        parallel_reject_min_distance_m=float(params.width_jump_parallel_reject_min_distance_m),
        parallel_reject_max_distance_m=float(params.width_jump_parallel_reject_max_distance_m),
        parallel_reject_max_length_delta_ratio=float(params.width_jump_parallel_reject_max_length_delta_ratio),
        parallel_reject_max_scanline_shift_ratio=float(params.width_jump_parallel_reject_max_scanline_shift_ratio),
    )
    horizontal_interval_merge_candidates = np.zeros(free.shape, dtype=bool)
    vertical_interval_merge_candidates = np.zeros(free.shape, dtype=bool)
    horizontal_interval_merge_score = np.zeros(free.shape, dtype=np.float32)
    vertical_interval_merge_score = np.zeros(free.shape, dtype=np.float32)
    critical_line_candidates = critical_candidates.astype(bool)
    interval_merge_separator = np.zeros(free.shape, dtype=bool)

    separator = np.zeros(free.shape, dtype=bool)
    for mask in separator_masks:
        separator |= np.asarray(mask, dtype=bool) & free
    separator &= free
    separator_before_small_region_reject = separator.copy()
    separator, rejected_small_region_separator = _reject_separator_masks_creating_small_regions(
        free,
        separator_masks,
        resolution_m=float(params.map_resolution_m),
        min_created_region_area_m2=float(params.width_jump_min_created_region_area_m2),
    )

    split_domain = free & ~separator
    component_labels, initial_count = label_components(split_domain, connectivity=4)
    if int(initial_count) == 0:
        initial_labels = np.zeros(free.shape, dtype=np.int32)
    else:
        initial_labels = nearest_seed_fill(component_labels, domain=free)
    initial_labels = _filter_area_limits(
        initial_labels,
        free,
        resolution_m=float(params.map_resolution_m),
        lower_m2=float(params.room_area_factor_lower_limit_voronoi),
        upper_m2=float(params.room_area_factor_upper_limit_voronoi),
    )
    if not bool(np.any(initial_labels > 0)):
        initial_labels, initial_count = label_components(free, connectivity=4)

    merged_labels, merge_count = _merge_small_regions(
        initial_labels,
        free,
        resolution_m=float(params.map_resolution_m),
        merge_area_threshold_m2=float(params.merge_area_threshold_m2),
        force_merge_area_m2=float(params.force_merge_area_m2),
    )
    labels = relabel_consecutive(merged_labels)
    labels[~free] = 0
    stats = VoronoiFallbackStats(
        free_cells=int(np.count_nonzero(free)),
        ridge_cells=int(np.count_nonzero(ridge)),
        pruned_ridge_cells=int(np.count_nonzero(pruned_ridge)),
        critical_candidates=int(np.count_nonzero(critical_candidates)),
        critical_lines_drawn=int(pre_parallel_reject_line_count - rejected_parallel_critical_lines),
        critical_lines_rejected_parallel=int(rejected_parallel_critical_lines),
        separator_cells_rejected_small_region=int(np.count_nonzero(rejected_small_region_separator)),
        input_hole_filled_cells=int(np.count_nonzero(input_hole_filled_mask)),
        initial_region_count=int(initial_count),
        merged_region_count=int(merge_count),
        final_room_count=int(labels.max()),
    )
    metadata = _algorithm_metadata(params, stats, "critical_line_split")
    debug = {
        "voronoi_metric_domain": free.astype(bool),
        "voronoi_raw_metric_domain_before_hole_fill": raw_free.astype(bool),
        "voronoi_input_hole_filled_mask": input_hole_filled_mask.astype(bool),
        "voronoi_scanline_structural_blocker_mask": structural_blocker.astype(bool),
        "voronoi_distance_transform_cells": distance_cells.astype(np.float32),
        "voronoi_width_cells": (distance_cells * 2.0).astype(np.float32),
        "voronoi_smoothed_width_cells": smoothed_width_cells.astype(np.float32),
        "voronoi_width_jump_drop_cells": width_jump_drop_cells.astype(np.float32),
        "voronoi_width_jump_score": width_jump_score.astype(np.float32),
        "voronoi_horizontal_width_cells": horizontal_width_cells.astype(np.float32),
        "voronoi_vertical_width_cells": vertical_width_cells.astype(np.float32),
        "voronoi_horizontal_width_jump_score": horizontal_score.astype(np.float32),
        "voronoi_vertical_width_jump_score": vertical_score.astype(np.float32),
        "voronoi_horizontal_width_jump_candidate_mask": horizontal_candidates.astype(bool),
        "voronoi_vertical_width_jump_candidate_mask": vertical_candidates.astype(bool),
        "voronoi_horizontal_interval_merge_candidate_mask": horizontal_interval_merge_candidates.astype(bool),
        "voronoi_vertical_interval_merge_candidate_mask": vertical_interval_merge_candidates.astype(bool),
        "voronoi_horizontal_interval_merge_score": horizontal_interval_merge_score.astype(np.float32),
        "voronoi_vertical_interval_merge_score": vertical_interval_merge_score.astype(np.float32),
        "voronoi_critical_line_candidate_mask": critical_line_candidates.astype(bool),
        "voronoi_interval_merge_separator_mask": interval_merge_separator.astype(bool),
        "voronoi_separator_before_small_region_reject_mask": separator_before_small_region_reject.astype(bool),
        "voronoi_separator_rejected_small_region_mask": rejected_small_region_separator.astype(bool),
        "voronoi_ridge_mask": ridge.astype(bool),
        "voronoi_pruned_ridge_mask": pruned_ridge.astype(bool),
        "voronoi_critical_point_mask": critical_candidates.astype(bool),
        "voronoi_separator_mask": separator.astype(bool),
        "voronoi_initial_label_map": np.asarray(initial_labels, dtype=np.int32),
    }
    return labels.astype(np.int32), metadata, debug


def build_fallback_metadata(
    *,
    source_snapshot: Path | str | None,
    input_free_definition: str,
    map_resolution_m: float,
    map_origin_xy_m: tuple[float, float],
    parameters: VoronoiFallbackParameters,
    algorithm_metadata: Mapping[str, Any],
    domain_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "method": "voronoi",
        "source_snapshot": "" if source_snapshot is None else str(source_snapshot),
        "input_free_definition": str(input_free_definition),
        "unknown_treated_as": "occupied/inaccessible",
        "unknown_treated_as_free": False,
        "map_resolution_m": float(map_resolution_m),
        "map_origin_xy_m": [float(map_origin_xy_m[0]), float(map_origin_xy_m[1])],
        "uses_rgb": False,
        "uses_depth": False,
        "uses_oracle_semantics": False,
        "runner_type": "python_fallback",
        "fallback_scope": "smoke_only_not_main_experiment",
        "main_experiment_allowed": False,
        "original_repo": ORIGINAL_REPO,
        "original_repo_commit": ORIGINAL_REPO_COMMIT,
        "original_package": ORIGINAL_PACKAGE,
        "original_action": ORIGINAL_ACTION,
        "original_algorithm_id": VORONOI_ALGORITHM_ID,
        "original_algorithm_name": "VoronoiSegmentation",
        "variant_algorithm_name": "VoronoiSegmentationWidthJump",
        "parameters": asdict(parameters),
    }
    metadata.update(dict(domain_metadata))
    metadata.update(dict(algorithm_metadata))
    return metadata


def _algorithm_metadata(
    params: VoronoiFallbackParameters,
    stats: VoronoiFallbackStats,
    reason: str,
) -> dict[str, Any]:
    return {
        "fallback_algorithm": "bormann_voronoi_width_jump_python",
        "method": "voronoi_width_jump",
        "variant_of": "voronoi",
        "fallback_approximates_original": True,
        "fallback_approximation_notes": (
            "Scanline width-jump variant: every row and every column is scanned "
            "one cell at a time. Each contiguous free interval is compared only "
            "with overlapping free intervals on the previous scanline, and a "
            "separator candidate is created only when that interval width changes "
            "by at least width_jump_min_drop_m. Before scanning, small non-structural "
            "holes in the input free mask are filled, but holes touching wall+door "
            "barriers are preserved. Parallel candidates whose scan "
            "lines are 0.20m to 1.50m apart are rejected in a one-pass "
            "top-to-bottom / left-to-right order only when their lengths differ "
            "by at most 20% and their along-scanline centers are nearly unchanged. "
            "All proposed separators are first collected, then the combined split "
            "is evaluated once; separator candidates that create a new region below "
            "the configured minimum area are removed. Gradual width changes and "
            "small local width fluctuations are intentionally ignored."
        ),
        "reason": str(reason),
        "runner_type": "python_fallback",
        "main_experiment_allowed": False,
        "critical_point_rule": "single_cell_scanline_free_interval_width_change_ge_min_drop",
        "critical_line_rule": "draw_axis_aligned_scanline_segment_on_the_narrower_transition_interval",
        "critical_line_min_angle_deg": float(params.min_critical_line_angle_deg),
        "merge_area_threshold_m2": float(params.merge_area_threshold_m2),
        "force_merge_area_m2": float(params.force_merge_area_m2),
        "voronoi_neighborhood_index": int(params.voronoi_neighborhood_index),
        "max_iterations": int(params.max_iterations),
        "min_critical_point_distance_factor": float(params.min_critical_point_distance_factor),
        "width_smoothing_cells": int(params.width_smoothing_cells),
        "width_jump_context_cells": int(params.width_jump_context_cells),
        "width_jump_min_drop_m": float(params.width_jump_min_drop_m),
        "width_jump_min_drop_ratio": float(params.width_jump_min_drop_ratio),
        "width_jump_min_gradient_cells": float(params.width_jump_min_gradient_cells),
        "width_jump_min_spacing_cells": int(params.width_jump_min_spacing_cells),
        "width_jump_min_component_length_cells": int(params.width_jump_min_component_length_cells),
        "width_jump_interval_merge_context_cells": int(params.width_jump_interval_merge_context_cells),
        "width_jump_interval_merge_min_support_lines": int(params.width_jump_interval_merge_min_support_lines),
        "width_jump_interval_merge_transition_radius_cells": int(params.width_jump_interval_merge_transition_radius_cells),
        "width_jump_parallel_reject_min_distance_m": float(params.width_jump_parallel_reject_min_distance_m),
        "width_jump_parallel_reject_max_distance_m": float(params.width_jump_parallel_reject_max_distance_m),
        "width_jump_parallel_reject_max_length_delta_ratio": float(
            params.width_jump_parallel_reject_max_length_delta_ratio
        ),
        "width_jump_parallel_reject_max_scanline_shift_ratio": float(
            params.width_jump_parallel_reject_max_scanline_shift_ratio
        ),
        "width_jump_parallel_reject_scan_order": "top_to_bottom_for_horizontal_scans_left_to_right_for_vertical_scans",
        "width_jump_split_validation_domain": "combined_separator_after_full_scan",
        "width_jump_min_created_region_area_m2": float(params.width_jump_min_created_region_area_m2),
        "width_jump_input_hole_fill_max_area_m2": float(params.width_jump_input_hole_fill_max_area_m2),
        "room_area_factor_lower_limit_voronoi": float(params.room_area_factor_lower_limit_voronoi),
        "room_area_factor_upper_limit_voronoi": float(params.room_area_factor_upper_limit_voronoi),
        "stats": asdict(stats),
    }


def _approximate_voronoi_ridge(dist: np.ndarray, free: np.ndarray, *, min_distance_cells: float) -> np.ndarray:
    maxed = ndimage.maximum_filter(dist, size=3, mode="nearest")
    ridge = np.asarray(free, dtype=bool) & (dist >= maxed - 1.0e-6) & (dist > float(min_distance_cells))
    if bool(np.any(ridge)):
        return ridge
    return np.asarray(free, dtype=bool) & (dist > float(min_distance_cells))


def _prune_skeleton_endpoints(skeleton: np.ndarray, *, iterations: int) -> np.ndarray:
    out = np.asarray(skeleton, dtype=bool).copy()
    if not bool(np.any(out)):
        return out
    kernel = np.ones((3, 3), dtype=np.int16)
    for _ in range(max(0, int(iterations))):
        neighbor_count = ndimage.convolve(out.astype(np.int16), kernel, mode="constant", cval=0) - out.astype(np.int16)
        endpoints = out & (neighbor_count <= 1)
        if not bool(np.any(endpoints)):
            break
        nodes = out & (neighbor_count >= 3)
        remove = endpoints & ~nodes
        if not bool(np.any(remove)):
            break
        out[remove] = False
    return out


def _critical_narrow_points(
    dist: np.ndarray,
    ridge: np.ndarray,
    *,
    neighborhood_index: int,
    max_iterations: int,
) -> np.ndarray:
    if not bool(np.any(ridge)):
        return np.zeros(ridge.shape, dtype=bool)
    components, _ = label_components(ridge, connectivity=8)
    out = np.zeros(ridge.shape, dtype=bool)
    for component_id in [int(v) for v in np.unique(components) if int(v) > 0]:
        coords = np.argwhere(components == component_id)
        if coords.size == 0:
            continue
        component_dist = dist[components == component_id]
        dynamic_eps = max(3, int(min(max_iterations, max(1, neighborhood_index) / max(float(np.median(component_dist)), 1.0))))
        stride = max(1, dynamic_eps // 2)
        order = np.argsort(component_dist)
        selected: list[tuple[int, int]] = []
        for idx in order.tolist():
            r, c = (int(coords[idx, 0]), int(coords[idx, 1]))
            if any((r - sr) * (r - sr) + (c - sc) * (c - sc) <= stride * stride for sr, sc in selected):
                continue
            selected.append((r, c))
            if len(selected) >= max(1, len(coords) // max(dynamic_eps, 1) + 1):
                break
        for r, c in selected:
            out[r, c] = True
    return out


def _fill_small_nonstructural_input_holes(
    free: np.ndarray,
    structural_blocker: np.ndarray,
    *,
    resolution_m: float,
    max_area_m2: float,
) -> tuple[np.ndarray, np.ndarray]:
    free_mask = np.asarray(free, dtype=bool)
    blocker = np.asarray(structural_blocker, dtype=bool)
    if blocker.shape != free_mask.shape:
        blocker = np.zeros(free_mask.shape, dtype=bool)
    filled = free_mask.copy()
    fill_mask = np.zeros(free_mask.shape, dtype=bool)
    if float(max_area_m2) <= 0.0:
        return filled, fill_mask

    cell_area = float(resolution_m) * float(resolution_m)
    max_area_cells = max(1.0, float(max_area_m2) / max(cell_area, 1.0e-12))
    fillable = ~free_mask & ~blocker
    labels, count = label_components(fillable, connectivity=4)
    h, w = free_mask.shape
    for component_id in range(1, int(count) + 1):
        component = labels == component_id
        area_cells = int(np.count_nonzero(component))
        if float(area_cells) > max_area_cells:
            continue
        ys, xs = np.where(component)
        if ys.size == 0:
            continue
        if bool(np.any((ys == 0) | (xs == 0) | (ys == h - 1) | (xs == w - 1))):
            continue
        if bool(np.any(ndimage.binary_dilation(component, structure=STRUCTURE_4) & blocker)):
            continue
        fill_mask |= component
    filled |= fill_mask
    filled[blocker] = False
    return filled.astype(bool), fill_mask.astype(bool)


def _scanline_width_jump_separators(
    free: np.ndarray,
    *,
    resolution_m: float,
    min_drop_m: float,
    parallel_reject_min_distance_m: float,
    parallel_reject_max_distance_m: float,
    parallel_reject_max_length_delta_ratio: float,
    parallel_reject_max_scanline_shift_ratio: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    int,
    int,
]:
    free_mask = np.asarray(free, dtype=bool)
    horizontal_width = _axis_run_width(free_mask, axis=1)
    vertical_width = _axis_run_width(free_mask, axis=0)
    horizontal_items, horizontal_mask, horizontal_score, horizontal_drop = _axis_scanline_width_jump_candidates(
        free_mask,
        axis="horizontal",
        resolution_m=float(resolution_m),
        min_drop_m=float(min_drop_m),
    )
    vertical_items, vertical_mask, vertical_score, vertical_drop = _axis_scanline_width_jump_candidates(
        free_mask,
        axis="vertical",
        resolution_m=float(resolution_m),
        min_drop_m=float(min_drop_m),
    )
    candidates = [*horizontal_items, *vertical_items]
    for candidate_id, item in enumerate(candidates, start=1):
        item["candidate_id"] = int(candidate_id)
    pre_reject_count = int(len(candidates))
    accepted = _reject_parallel_scanline_candidates(
        candidates,
        resolution_m=float(resolution_m),
        min_distance_m=float(parallel_reject_min_distance_m),
        max_distance_m=float(parallel_reject_max_distance_m),
        max_length_delta_ratio=float(parallel_reject_max_length_delta_ratio),
        max_scanline_shift_ratio=float(parallel_reject_max_scanline_shift_ratio),
    )
    accepted_ids = {int(item["candidate_id"]) for item in accepted}
    horizontal_mask &= _candidate_id_mask(horizontal_items, accepted_ids, free_mask.shape)
    vertical_mask &= _candidate_id_mask(vertical_items, accepted_ids, free_mask.shape)
    horizontal_score = np.where(horizontal_mask, horizontal_score, np.float32(0.0)).astype(np.float32)
    vertical_score = np.where(vertical_mask, vertical_score, np.float32(0.0)).astype(np.float32)
    horizontal_drop = np.where(horizontal_mask, horizontal_drop, np.float32(0.0)).astype(np.float32)
    vertical_drop = np.where(vertical_mask, vertical_drop, np.float32(0.0)).astype(np.float32)
    combined_mask = horizontal_mask | vertical_mask
    combined_score = np.maximum(horizontal_score, vertical_score).astype(np.float32)
    combined_drop = np.maximum(horizontal_drop, vertical_drop).astype(np.float32)
    smoothed_width = np.minimum(horizontal_width, vertical_width).astype(np.float32)
    separator_masks = [np.asarray(item["mask"], dtype=bool) & free_mask for item in accepted]
    return (
        combined_mask.astype(bool),
        smoothed_width,
        combined_drop,
        combined_score,
        horizontal_mask.astype(bool),
        vertical_mask.astype(bool),
        horizontal_width.astype(np.float32),
        vertical_width.astype(np.float32),
        horizontal_score.astype(np.float32),
        vertical_score.astype(np.float32),
        separator_masks,
        pre_reject_count,
        int(pre_reject_count - len(accepted)),
    )


def _axis_scanline_width_jump_candidates(
    free: np.ndarray,
    *,
    axis: str,
    resolution_m: float,
    min_drop_m: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    if axis == "vertical":
        items_t, mask_t, score_t, drop_t = _axis_scanline_width_jump_candidates(
            np.asarray(free, dtype=bool).T,
            axis="horizontal",
            resolution_m=float(resolution_m),
            min_drop_m=float(min_drop_m),
        )
        items: list[dict[str, Any]] = []
        for item in items_t:
            out = dict(item)
            out["axis"] = "vertical"
            out["mask"] = np.asarray(item["mask"], dtype=bool).T.copy()
            out["point_rc"] = [int(item["point_rc"][1]), int(item["point_rc"][0])]
            items.append(out)
        return items, mask_t.T.copy(), score_t.T.copy(), drop_t.T.copy()
    if axis != "horizontal":
        raise ValueError(f"unsupported scanline axis: {axis}")

    free_mask = np.asarray(free, dtype=bool)
    candidate_mask = np.zeros(free_mask.shape, dtype=bool)
    score = np.zeros(free_mask.shape, dtype=np.float32)
    drop = np.zeros(free_mask.shape, dtype=np.float32)
    items: list[dict[str, Any]] = []
    min_drop_cells = max(1, int(math.ceil(float(min_drop_m) / max(float(resolution_m), 1.0e-6))))
    intervals_by_row = [_free_intervals_1d(free_mask[row, :]) for row in range(free_mask.shape[0])]
    seen: set[tuple[int, int, int]] = set()
    for row in range(1, free_mask.shape[0]):
        previous_intervals = intervals_by_row[row - 1]
        current_intervals = intervals_by_row[row]
        if not previous_intervals or not current_intervals:
            continue
        for current_start, current_end in current_intervals:
            current_len = int(current_end) - int(current_start) + 1
            for previous_start, previous_end in _overlapping_intervals(previous_intervals, current_start, current_end):
                previous_len = int(previous_end) - int(previous_start) + 1
                delta = abs(int(current_len) - int(previous_len))
                if int(delta) < int(min_drop_cells):
                    continue
                if current_len <= previous_len:
                    cut_row = int(row)
                    cut_start = int(current_start)
                    cut_end = int(current_end)
                else:
                    cut_row = int(row - 1)
                    cut_start = int(previous_start)
                    cut_end = int(previous_end)
                if cut_start > cut_end:
                    continue
                key = (int(cut_row), int(cut_start), int(cut_end))
                if key in seen:
                    continue
                seen.add(key)
                mask = np.zeros(free_mask.shape, dtype=bool)
                mask[cut_row, cut_start : cut_end + 1] = True
                mask &= free_mask
                if int(np.count_nonzero(mask)) < 2:
                    continue
                center_col = int(round((float(cut_start) + float(cut_end)) * 0.5))
                candidate_mask[cut_row, center_col] = True
                score[cut_row, center_col] = np.float32(delta)
                drop[cut_row, center_col] = np.float32(delta)
                items.append(
                    {
                        "candidate_id": int(len(items) + 1),
                        "axis": "horizontal",
                        "mask": mask,
                        "point_rc": [int(cut_row), int(center_col)],
                        "scan_coord": float(cut_row),
                        "along_start": float(cut_start),
                        "along_end": float(cut_end),
                        "along_center": (float(cut_start) + float(cut_end)) * 0.5,
                        "length": float(cut_end - cut_start + 1),
                        "width_delta_cells": float(delta),
                    }
                )
    return items, candidate_mask, score, drop


def _candidate_id_mask(
    items: list[dict[str, Any]],
    accepted_ids: set[int],
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for item in items:
        if int(item["candidate_id"]) not in accepted_ids:
            continue
        row, col = (int(item["point_rc"][0]), int(item["point_rc"][1]))
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            out[row, col] = True
    return out


def _reject_parallel_scanline_candidates(
    candidates: list[dict[str, Any]],
    *,
    resolution_m: float,
    min_distance_m: float,
    max_distance_m: float,
    max_length_delta_ratio: float,
    max_scanline_shift_ratio: float,
) -> list[dict[str, Any]]:
    if len(candidates) < 2:
        return list(candidates)
    min_cells = float(min_distance_m) / max(float(resolution_m), 1.0e-6)
    max_cells = float(max_distance_m) / max(float(resolution_m), 1.0e-6)
    rejected: set[int] = set()
    for axis in ("horizontal", "vertical"):
        axis_items = sorted(
            [(index, item) for index, item in enumerate(candidates) if str(item["axis"]) == axis],
            key=lambda entry: (
                float(entry[1]["scan_coord"]),
                float(entry[1]["along_start"]),
                float(entry[1]["along_end"]),
                int(entry[0]),
            ),
        )
        for sorted_left_idx, (left_idx, left) in enumerate(axis_items):
            if left_idx in rejected:
                continue
            left_scan = float(left["scan_coord"])
            for right_idx, right in axis_items[sorted_left_idx + 1 :]:
                if right_idx in rejected:
                    continue
                scan_distance = float(right["scan_coord"]) - left_scan
                if scan_distance < min_cells:
                    continue
                if scan_distance > max_cells:
                    break
                if _parallel_line_descriptors_are_rectangle_like(
                    left,
                    right,
                    max_length_delta_ratio=float(max_length_delta_ratio),
                    max_scanline_shift_ratio=float(max_scanline_shift_ratio),
                ):
                    rejected.add(left_idx)
                    rejected.add(right_idx)
                    break
    return [item for index, item in enumerate(candidates) if index not in rejected]


def _critical_width_jump_points(
    dist: np.ndarray,
    ridge: np.ndarray,
    *,
    resolution_m: float,
    smoothing_cells: int,
    context_cells: int,
    min_drop_m: float,
    min_drop_ratio: float,
    min_gradient_cells: float,
    min_spacing_cells: int,
    min_component_length_cells: int,
    interval_merge_context_cells: int,
    interval_merge_min_support_lines: int,
    interval_merge_transition_radius_cells: int,
    parallel_reject_min_distance_m: float,
    parallel_reject_max_distance_m: float,
    parallel_reject_max_length_delta_ratio: float,
    parallel_reject_max_scanline_shift_ratio: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
]:
    ridge = np.asarray(ridge, dtype=bool)
    free = np.asarray(dist) > 0.0
    horizontal_width = _axis_run_width(free, axis=1)
    vertical_width = _axis_run_width(free, axis=0)
    if not bool(np.any(ridge)):
        zeros = np.zeros(ridge.shape, dtype=np.float32)
        empty = np.zeros(ridge.shape, dtype=bool)
        return (
            empty,
            zeros,
            zeros,
            zeros,
            empty,
            empty,
            horizontal_width,
            vertical_width,
            zeros,
            zeros,
            empty,
            empty,
            zeros,
            zeros,
            empty,
            empty,
            [],
        )

    horizontal_mask, horizontal_smoothed, horizontal_drop, horizontal_score = _axis_width_jump_points(
        horizontal_width,
        ridge,
        free,
        resolution_m=resolution_m,
        smoothing_cells=smoothing_cells,
        context_cells=context_cells,
        min_drop_m=min_drop_m,
        min_drop_ratio=min_drop_ratio,
        min_gradient_cells=min_gradient_cells,
        min_spacing_cells=min_spacing_cells,
        min_component_length_cells=min_component_length_cells,
    )
    vertical_mask, vertical_smoothed, vertical_drop, vertical_score = _axis_width_jump_points(
        vertical_width,
        ridge,
        free,
        resolution_m=resolution_m,
        smoothing_cells=smoothing_cells,
        context_cells=context_cells,
        min_drop_m=min_drop_m,
        min_drop_ratio=min_drop_ratio,
        min_gradient_cells=min_gradient_cells,
        min_spacing_cells=min_spacing_cells,
        min_component_length_cells=min_component_length_cells,
    )
    horizontal_merge_mask, horizontal_merge_score, horizontal_merge_separator, horizontal_merge_items = _axis_interval_merge_points(
        free,
        ridge,
        axis=1,
        context_cells=interval_merge_context_cells,
        min_drop_m=min_drop_m,
        min_support_lines=interval_merge_min_support_lines,
        transition_radius_cells=interval_merge_transition_radius_cells,
        resolution_m=resolution_m,
        parallel_reject_min_distance_m=parallel_reject_min_distance_m,
        parallel_reject_max_distance_m=parallel_reject_max_distance_m,
        parallel_reject_max_length_delta_ratio=parallel_reject_max_length_delta_ratio,
        parallel_reject_max_scanline_shift_ratio=parallel_reject_max_scanline_shift_ratio,
    )
    vertical_merge_mask, vertical_merge_score, vertical_merge_separator, vertical_merge_items = _axis_interval_merge_points(
        free,
        ridge,
        axis=0,
        context_cells=interval_merge_context_cells,
        min_drop_m=min_drop_m,
        min_support_lines=interval_merge_min_support_lines,
        transition_radius_cells=interval_merge_transition_radius_cells,
        resolution_m=resolution_m,
        parallel_reject_min_distance_m=parallel_reject_min_distance_m,
        parallel_reject_max_distance_m=parallel_reject_max_distance_m,
        parallel_reject_max_length_delta_ratio=parallel_reject_max_length_delta_ratio,
        parallel_reject_max_scanline_shift_ratio=parallel_reject_max_scanline_shift_ratio,
    )
    horizontal_width_mask = horizontal_mask
    vertical_width_mask = vertical_mask
    horizontal_mask = horizontal_width_mask | horizontal_merge_mask
    vertical_mask = vertical_width_mask | vertical_merge_mask
    horizontal_score = np.maximum(horizontal_score, horizontal_merge_score).astype(np.float32)
    vertical_score = np.maximum(vertical_score, vertical_merge_score).astype(np.float32)
    score = np.maximum(horizontal_score, vertical_score).astype(np.float32)
    critical_line_score = np.maximum(
        np.where(horizontal_width_mask, horizontal_score, np.float32(0.0)),
        np.where(vertical_width_mask, vertical_score, np.float32(0.0)),
    ).astype(np.float32)
    critical_line_candidates = _nms_boolean_candidates(
        horizontal_width_mask | vertical_width_mask,
        critical_line_score,
        min_spacing_cells=max(1, int(min_spacing_cells)),
    )
    interval_merge_candidates = _nms_boolean_candidates(
        horizontal_merge_mask | vertical_merge_mask,
        np.maximum(horizontal_merge_score, vertical_merge_score).astype(np.float32),
        min_spacing_cells=max(1, int(min_spacing_cells)),
    )
    selected = critical_line_candidates | interval_merge_candidates
    smoothed = np.minimum(horizontal_smoothed, vertical_smoothed).astype(np.float32)
    drop = np.maximum(horizontal_drop, vertical_drop).astype(np.float32)
    score = np.where(selected, score, np.float32(0.0)).astype(np.float32)
    interval_separator = (horizontal_merge_separator | vertical_merge_separator) & free
    interval_separator_items = [
        np.asarray(mask, dtype=bool) & free for mask in [*horizontal_merge_items, *vertical_merge_items]
    ]
    return (
        selected,
        smoothed,
        drop,
        score,
        horizontal_mask,
        vertical_mask,
        horizontal_width,
        vertical_width,
        horizontal_score,
        vertical_score,
        horizontal_merge_mask,
        vertical_merge_mask,
        horizontal_merge_score,
        vertical_merge_score,
        critical_line_candidates,
        interval_separator,
        interval_separator_items,
    )


def _axis_width_jump_points(
    width_cells: np.ndarray,
    ridge: np.ndarray,
    free: np.ndarray,
    *,
    resolution_m: float,
    smoothing_cells: int,
    context_cells: int,
    min_drop_m: float,
    min_drop_ratio: float,
    min_gradient_cells: float,
    min_spacing_cells: int,
    min_component_length_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    smoothing = _odd_window(smoothing_cells)
    width_cells = np.asarray(width_cells, dtype=np.float32)
    if smoothing > 1:
        smoothed_width = ndimage.median_filter(width_cells, size=smoothing, mode="nearest").astype(np.float32)
    else:
        smoothed_width = width_cells.astype(np.float32, copy=True)

    context = max(1, int(context_cells))
    local_min_radius = max(1, smoothing // 2)
    min_drop_cells = max(0.0, float(min_drop_m) / max(float(resolution_m), 1.0e-6))
    min_gradient = max(0.0, float(min_gradient_cells))
    spacing = max(1, int(min_spacing_cells))

    selected_mask = np.zeros(ridge.shape, dtype=bool)
    drop_map = np.zeros(ridge.shape, dtype=np.float32)
    score_map = np.zeros(ridge.shape, dtype=np.float32)
    components, _ = label_components(ridge, connectivity=8)
    free_width_input = np.full(ridge.shape, -np.inf, dtype=np.float32)
    free_width_input[np.asarray(free, dtype=bool)] = smoothed_width[np.asarray(free, dtype=bool)]
    for component_id in [int(v) for v in np.unique(components) if int(v) > 0]:
        component = components == component_id
        if int(np.count_nonzero(component)) < int(min_component_length_cells):
            continue

        local_max = ndimage.maximum_filter(free_width_input, size=context * 2 + 1, mode="constant", cval=-np.inf)

        min_input = np.full(ridge.shape, np.inf, dtype=np.float32)
        min_input[component] = smoothed_width[component]
        local_min = ndimage.minimum_filter(min_input, size=local_min_radius * 2 + 1, mode="constant", cval=np.inf)

        transition_radius = max(2, min(context, smoothing))
        transition_max = ndimage.maximum_filter(
            free_width_input,
            size=transition_radius * 2 + 1,
            mode="constant",
            cval=-np.inf,
        )
        transition_drop = transition_max - smoothed_width

        drop = local_max - smoothed_width
        ratio = drop / np.maximum(local_max, np.float32(1.0))
        abrupt_drop_threshold = max(min_gradient, min_drop_cells * 0.5)
        candidate = (
            component
            & np.isfinite(local_max)
            & (smoothed_width <= local_min + np.float32(1.0e-6))
            & (drop >= np.float32(min_drop_cells))
            & (ratio >= np.float32(max(0.0, float(min_drop_ratio))))
            & (transition_drop >= np.float32(abrupt_drop_threshold))
        )
        if not bool(np.any(candidate)):
            continue

        candidate_coords = np.argwhere(candidate)
        candidate_scores = ratio[candidate] * np.float32(100.0) + drop[candidate] + transition_drop[candidate]
        order = np.argsort(-candidate_scores)
        selected: list[tuple[int, int]] = []
        for idx in order.tolist():
            r, c = (int(candidate_coords[idx, 0]), int(candidate_coords[idx, 1]))
            if any((r - sr) * (r - sr) + (c - sc) * (c - sc) <= spacing * spacing for sr, sc in selected):
                continue
            selected.append((r, c))
            selected_mask[r, c] = True
            drop_map[r, c] = np.float32(drop[r, c])
            score_map[r, c] = np.float32(candidate_scores[idx])

    return selected_mask, smoothed_width, drop_map, score_map


def _axis_run_width(free: np.ndarray, *, axis: int) -> np.ndarray:
    mask = np.asarray(free, dtype=bool)
    out = np.zeros(mask.shape, dtype=np.float32)
    if axis == 1:
        for r in range(mask.shape[0]):
            out[r, :] = _run_lengths_1d(mask[r, :])
        return out
    if axis == 0:
        for c in range(mask.shape[1]):
            out[:, c] = _run_lengths_1d(mask[:, c])
        return out
    raise ValueError(f"unsupported axis: {axis}")


def _axis_interval_merge_points(
    free: np.ndarray,
    ridge: np.ndarray,
    *,
    axis: int,
    context_cells: int,
    min_drop_m: float,
    min_support_lines: int,
    transition_radius_cells: int,
    resolution_m: float,
    parallel_reject_min_distance_m: float,
    parallel_reject_max_distance_m: float,
    parallel_reject_max_length_delta_ratio: float,
    parallel_reject_max_scanline_shift_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    if axis == 0:
        mask_t, score_t, separator_t, item_masks_t = _axis_interval_merge_points(
            free.T,
            ridge.T,
            axis=1,
            context_cells=context_cells,
            min_drop_m=min_drop_m,
            min_support_lines=min_support_lines,
            transition_radius_cells=transition_radius_cells,
            resolution_m=resolution_m,
            parallel_reject_min_distance_m=parallel_reject_min_distance_m,
            parallel_reject_max_distance_m=parallel_reject_max_distance_m,
            parallel_reject_max_length_delta_ratio=parallel_reject_max_length_delta_ratio,
            parallel_reject_max_scanline_shift_ratio=parallel_reject_max_scanline_shift_ratio,
        )
        return mask_t.T.copy(), score_t.T.copy(), separator_t.T.copy(), [mask.T.copy() for mask in item_masks_t]
    if axis != 1:
        raise ValueError(f"unsupported axis: {axis}")

    free_mask = np.asarray(free, dtype=bool)
    ridge_mask = np.asarray(ridge, dtype=bool)
    out = np.zeros(free_mask.shape, dtype=bool)
    score = np.zeros(free_mask.shape, dtype=np.float32)
    separator = np.zeros(free_mask.shape, dtype=bool)
    separator_items: list[np.ndarray] = []
    candidates: list[dict[str, int | float]] = []
    radius = max(1, int(context_cells))
    min_gap_cells = max(1, int(math.ceil(float(min_drop_m) / max(float(resolution_m), 1.0e-6))))
    transition_radius = max(1, int(transition_radius_cells))
    min_support = max(1, int(min_support_lines))
    intervals_by_row = [_free_intervals_1d(free_mask[r, :]) for r in range(free_mask.shape[0])]
    for row, current_intervals in enumerate(intervals_by_row):
        if not current_intervals:
            continue
        neighbor_start = max(0, row - radius)
        neighbor_stop = min(free_mask.shape[0], row + radius + 1)
        transition_start = max(0, row - transition_radius)
        transition_stop = min(free_mask.shape[0], row + transition_radius + 1)
        for current_start, current_end in current_intervals:
            support = 0
            for support_row in range(neighbor_start, neighbor_stop):
                if support_row == row:
                    continue
                support_overlaps = _overlapping_intervals(intervals_by_row[support_row], current_start, current_end)
                if len(support_overlaps) >= 2:
                    support += 1
            if support < min_support:
                continue
            for neighbor_row in range(transition_start, transition_stop):
                if neighbor_row == row:
                    continue
                overlapping = _overlapping_intervals(intervals_by_row[neighbor_row], current_start, current_end)
                if len(overlapping) < 2:
                    continue
                row_distance = abs(neighbor_row - row)
                overlapping.sort()
                for left, right in zip(overlapping, overlapping[1:]):
                    gap_start = max(current_start, int(left[1]) + 1)
                    gap_end = min(current_end, int(right[0]) - 1)
                    if gap_start > gap_end:
                        continue
                    gap_width = max(1, gap_end - gap_start + 1)
                    if int(gap_width) < int(min_gap_cells):
                        continue
                    candidate_col = _nearest_ridge_col_on_row(
                        ridge_mask,
                        row=row,
                        preferred_start=gap_start,
                        preferred_end=gap_end,
                        fallback_start=current_start,
                        fallback_end=current_end,
                    )
                    if candidate_col is None:
                        continue
                    merge_score = float(len(overlapping) - 1) * 100.0 + float(gap_width) - float(row_distance)
                    candidates.append(
                        {
                            "row": int(row),
                            "col": int(candidate_col),
                            "gap_start": int(gap_start),
                            "gap_end": int(gap_end),
                            "score": float(merge_score),
                        }
                    )
    accepted = _reject_parallel_interval_candidates(
        candidates,
        resolution_m=float(resolution_m),
        min_distance_m=float(parallel_reject_min_distance_m),
        max_distance_m=float(parallel_reject_max_distance_m),
        max_length_delta_ratio=float(parallel_reject_max_length_delta_ratio),
        max_scanline_shift_ratio=float(parallel_reject_max_scanline_shift_ratio),
    )
    for candidate in accepted:
        row = int(candidate["row"])
        col = int(candidate["col"])
        gap_start = int(candidate["gap_start"])
        gap_end = int(candidate["gap_end"])
        merge_score = float(candidate["score"])
        local_separator = _local_gap_bridge_separator(free_mask, row=row, gap_start=gap_start, gap_end=gap_end)
        out[row, col] = True
        score[row, col] = max(score[row, col], np.float32(merge_score))
        separator |= local_separator
        separator_items.append(local_separator & free_mask)
    return out, score, separator & free_mask, separator_items


def _local_gap_bridge_separator(free: np.ndarray, *, row: int, gap_start: int, gap_end: int) -> np.ndarray:
    free_mask = np.asarray(free, dtype=bool)
    separator = np.zeros(free_mask.shape, dtype=bool)
    h = int(free_mask.shape[0])
    c0 = max(0, int(gap_start))
    c1 = min(int(free_mask.shape[1]), int(gap_end) + 1)
    if c0 >= c1:
        return separator
    for step in (-1, 1):
        r = int(row)
        while 0 <= r < h:
            band = free_mask[r, c0:c1]
            if not bool(np.any(band)):
                break
            separator[r, c0:c1] |= band
            r += step
    return separator


def _reject_separator_masks_creating_small_regions(
    free: np.ndarray,
    separator_masks: list[np.ndarray],
    *,
    resolution_m: float,
    min_created_region_area_m2: float,
) -> tuple[np.ndarray, np.ndarray]:
    free_mask = np.asarray(free, dtype=bool)
    candidate_masks = [np.asarray(mask, dtype=bool) & free_mask for mask in separator_masks if bool(np.any(mask))]
    separator_mask = np.zeros(free_mask.shape, dtype=bool)
    for mask in candidate_masks:
        separator_mask |= mask
    rejected = np.zeros(free_mask.shape, dtype=bool)
    if float(min_created_region_area_m2) <= 0.0 or not bool(np.any(separator_mask)):
        return separator_mask, rejected

    cell_area = float(resolution_m) * float(resolution_m)
    min_area_cells = max(1.0, float(min_created_region_area_m2) / max(cell_area, 1.0e-12))
    original_components, _ = label_components(free_mask, connectivity=4)
    original_areas = {
        int(label): int(np.count_nonzero(original_components == label))
        for label in np.unique(original_components)
        if int(label) > 0
    }
    split_components, _ = label_components(free_mask & ~separator_mask, connectivity=4)
    small_component_mask = np.zeros(free_mask.shape, dtype=bool)
    for label in (int(v) for v in np.unique(split_components) if int(v) > 0):
        component = split_components == label
        area_cells = int(np.count_nonzero(component))
        if float(area_cells) >= min_area_cells:
            continue
        original_ids = [int(v) for v in np.unique(original_components[component]) if int(v) > 0]
        original_area = max((int(original_areas.get(v, 0)) for v in original_ids), default=0)
        if float(original_area) <= min_area_cells:
            continue
        small_component_mask |= component

    if not bool(np.any(small_component_mask)):
        return separator_mask, rejected

    small_touch_mask = ndimage.binary_dilation(small_component_mask, structure=STRUCTURE_4) & separator_mask
    kept = np.zeros(free_mask.shape, dtype=bool)
    for mask in candidate_masks:
        if bool(np.any(mask & small_touch_mask)):
            rejected |= mask
        else:
            kept |= mask
    return kept, rejected


def _separator_splits_free_component(
    component_labels: np.ndarray,
    separator: np.ndarray,
    *,
    row: int,
    col: int,
) -> bool:
    component_id = int(component_labels[int(row), int(col)])
    if component_id <= 0:
        return False
    component = np.asarray(component_labels == component_id, dtype=bool)
    if not bool(np.any(separator & component)):
        return False
    _, after_count = label_components(component & ~np.asarray(separator, dtype=bool), connectivity=4)
    return int(after_count) >= 2


def _reject_parallel_interval_candidates(
    candidates: list[dict[str, int | float]],
    *,
    resolution_m: float,
    min_distance_m: float,
    max_distance_m: float,
    max_length_delta_ratio: float = 0.20,
    max_scanline_shift_ratio: float = 0.20,
) -> list[dict[str, int | float]]:
    """Reject rectangle-like paired interval-merge cuts in scan order.

    Candidates arrive in scan coordinates. For the horizontal scan pass this is
    top-to-bottom rows; the vertical pass calls this helper on transposed masks,
    which makes the same logic left-to-right in the original map. A candidate
    only looks forward. If the next matching candidate is 0.2-1.5m away, has
    nearly the same interval length, and has barely shifted along the scanline,
    both candidates are rejected and the scan continues forward.
    """

    if len(candidates) < 2:
        return list(candidates)
    min_cells = float(min_distance_m) / max(float(resolution_m), 1.0e-6)
    max_cells = float(max_distance_m) / max(float(resolution_m), 1.0e-6)
    indexed = sorted(
        enumerate(candidates),
        key=lambda item: (
            int(item[1]["row"]),
            int(item[1].get("gap_start", item[1].get("col", 0))),
            int(item[1].get("gap_end", item[1].get("col", 0))),
            int(item[0]),
        ),
    )
    rejected: set[int] = set()
    for sorted_left_idx, (left_original_idx, left) in enumerate(indexed):
        if left_original_idx in rejected:
            continue
        left_row = int(left["row"])
        for right_original_idx, right in indexed[sorted_left_idx + 1 :]:
            if right_original_idx in rejected:
                continue
            row_distance_cells = int(right["row"]) - left_row
            if float(row_distance_cells) < float(min_cells):
                continue
            if float(row_distance_cells) > float(max_cells):
                break
            if _parallel_interval_pair_is_rectangle_like(
                left,
                right,
                max_length_delta_ratio=float(max_length_delta_ratio),
                max_scanline_shift_ratio=float(max_scanline_shift_ratio),
            ):
                rejected.add(left_original_idx)
                rejected.add(right_original_idx)
                break
    return [candidate for index, candidate in enumerate(candidates) if index not in rejected]


def _parallel_interval_pair_is_rectangle_like(
    first: Mapping[str, int | float],
    second: Mapping[str, int | float],
    *,
    max_length_delta_ratio: float,
    max_scanline_shift_ratio: float,
) -> bool:
    first_start = int(first["gap_start"])
    first_end = int(first["gap_end"])
    second_start = int(second["gap_start"])
    second_end = int(second["gap_end"])
    first_length = max(1, first_end - first_start + 1)
    second_length = max(1, second_end - second_start + 1)
    max_length = max(first_length, second_length)
    length_delta_ratio = abs(first_length - second_length) / float(max_length)
    if length_delta_ratio > max(0.0, float(max_length_delta_ratio)) + 1.0e-9:
        return False
    first_center = (float(first_start) + float(first_end)) * 0.5
    second_center = (float(second_start) + float(second_end)) * 0.5
    max_shift = max(1.0, float(max_length) * max(0.0, float(max_scanline_shift_ratio)))
    return abs(first_center - second_center) <= max_shift + 1.0e-9


def _nearest_ridge_col_on_row(
    ridge: np.ndarray,
    *,
    row: int,
    preferred_start: int,
    preferred_end: int,
    fallback_start: int,
    fallback_end: int,
) -> int | None:
    preferred = np.flatnonzero(ridge[row, preferred_start : preferred_end + 1])
    center = (float(preferred_start) + float(preferred_end)) * 0.5
    if preferred.size > 0:
        cols = preferred + int(preferred_start)
        return int(cols[int(np.argmin(np.abs(cols.astype(np.float32) - center)))])
    fallback = np.flatnonzero(ridge[row, fallback_start : fallback_end + 1])
    if fallback.size == 0:
        return int(round(center))
    cols = fallback + int(fallback_start)
    return int(cols[int(np.argmin(np.abs(cols.astype(np.float32) - center)))])


def _free_intervals_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    line = np.asarray(mask, dtype=bool)
    intervals: list[tuple[int, int]] = []
    index = 0
    length = int(line.shape[0])
    while index < length:
        if not bool(line[index]):
            index += 1
            continue
        start = index
        while index < length and bool(line[index]):
            index += 1
        intervals.append((int(start), int(index - 1)))
    return intervals


def _overlapping_intervals(
    intervals: list[tuple[int, int]],
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    return [item for item in intervals if int(item[1]) >= int(start) and int(item[0]) <= int(end)]


def _run_lengths_1d(mask: np.ndarray) -> np.ndarray:
    line = np.asarray(mask, dtype=bool)
    out = np.zeros(line.shape, dtype=np.float32)
    index = 0
    length = int(line.shape[0])
    while index < length:
        if not bool(line[index]):
            index += 1
            continue
        start = index
        while index < length and bool(line[index]):
            index += 1
        out[start:index] = np.float32(index - start)
    return out


def _nms_boolean_candidates(candidates: np.ndarray, score: np.ndarray, *, min_spacing_cells: int) -> np.ndarray:
    mask = np.asarray(candidates, dtype=bool)
    out = np.zeros(mask.shape, dtype=bool)
    if not bool(np.any(mask)):
        return out
    coords = np.argwhere(mask)
    scores = np.asarray(score, dtype=np.float32)[mask]
    order = np.argsort(-scores)
    spacing = max(1, int(min_spacing_cells))
    selected: list[tuple[int, int]] = []
    for idx in order.tolist():
        r, c = (int(coords[idx, 0]), int(coords[idx, 1]))
        if any((r - sr) * (r - sr) + (c - sc) * (c - sc) <= spacing * spacing for sr, sc in selected):
            continue
        selected.append((r, c))
        out[r, c] = True
    return out


def _odd_window(value: int) -> int:
    window = max(1, int(value))
    return window if window % 2 == 1 else window + 1


def _select_critical_lines(
    free: np.ndarray,
    dist: np.ndarray,
    critical_points: np.ndarray,
    *,
    min_angle_deg: float,
    min_distance_factor: float,
) -> list[CriticalLine]:
    candidates: list[CriticalLine] = []
    for r, c in np.argwhere(critical_points):
        basis = _nearest_two_obstacle_pixels(free, int(r), int(c), distance_cells=float(dist[int(r), int(c)]))
        if basis is None:
            continue
        a, b = basis
        angle = _angle_deg((int(r), int(c)), a, b)
        if angle < float(min_angle_deg):
            continue
        length = float(np.hypot(r - a[0], c - a[1]) + np.hypot(r - b[0], c - b[1]))
        candidates.append(CriticalLine((int(r), int(c)), a, b, float(angle), length))

    selected: list[CriticalLine] = []
    for line in sorted(candidates, key=lambda item: (-item.angle_deg, item.length_cells)):
        min_sep = max(1.0, float(dist[line.point_rc]) * float(min_distance_factor))
        if any(_point_distance(line.point_rc, other.point_rc) < min_sep for other in selected):
            continue
        selected.append(line)
    return selected


def _reject_parallel_critical_lines(
    lines: list[CriticalLine],
    *,
    resolution_m: float,
    min_distance_m: float,
    max_distance_m: float,
    max_length_delta_ratio: float = 0.20,
    max_scanline_shift_ratio: float = 0.20,
) -> list[CriticalLine]:
    if len(lines) < 2:
        return list(lines)
    min_cells = float(min_distance_m) / max(float(resolution_m), 1.0e-6)
    max_cells = float(max_distance_m) / max(float(resolution_m), 1.0e-6)
    descriptors = [
        (index, line, _critical_line_scan_descriptor(line))
        for index, line in enumerate(lines)
    ]
    rejected: set[int] = set()
    for axis in ("horizontal", "vertical"):
        axis_items = sorted(
            [item for item in descriptors if item[2]["axis"] == axis],
            key=lambda item: (
                float(item[2]["scan_coord"]),
                float(item[2]["along_start"]),
                float(item[2]["along_end"]),
                int(item[0]),
            ),
        )
        for sorted_left_idx, (left_idx, _left_line, left_desc) in enumerate(axis_items):
            if left_idx in rejected:
                continue
            left_scan = float(left_desc["scan_coord"])
            for right_idx, _right_line, right_desc in axis_items[sorted_left_idx + 1 :]:
                if right_idx in rejected:
                    continue
                scan_distance = float(right_desc["scan_coord"]) - left_scan
                if scan_distance < min_cells:
                    continue
                if scan_distance > max_cells:
                    break
                if _parallel_line_descriptors_are_rectangle_like(
                    left_desc,
                    right_desc,
                    max_length_delta_ratio=float(max_length_delta_ratio),
                    max_scanline_shift_ratio=float(max_scanline_shift_ratio),
                ):
                    rejected.add(left_idx)
                    rejected.add(right_idx)
                    break
    return [line for index, line in enumerate(lines) if index not in rejected]


def _critical_line_scan_descriptor(line: CriticalLine) -> dict[str, float | str]:
    r0, c0 = line.basis_a_rc
    r1, c1 = line.basis_b_rc
    row_span = abs(float(r1) - float(r0))
    col_span = abs(float(c1) - float(c0))
    if col_span >= row_span:
        along_start = min(float(c0), float(c1))
        along_end = max(float(c0), float(c1))
        return {
            "axis": "horizontal",
            "scan_coord": (float(r0) + float(r1)) * 0.5,
            "along_start": along_start,
            "along_end": along_end,
            "along_center": (along_start + along_end) * 0.5,
            "length": max(1.0, along_end - along_start + 1.0),
        }
    along_start = min(float(r0), float(r1))
    along_end = max(float(r0), float(r1))
    return {
        "axis": "vertical",
        "scan_coord": (float(c0) + float(c1)) * 0.5,
        "along_start": along_start,
        "along_end": along_end,
        "along_center": (along_start + along_end) * 0.5,
        "length": max(1.0, along_end - along_start + 1.0),
    }


def _parallel_line_descriptors_are_rectangle_like(
    first: Mapping[str, float | str],
    second: Mapping[str, float | str],
    *,
    max_length_delta_ratio: float,
    max_scanline_shift_ratio: float,
) -> bool:
    if str(first["axis"]) != str(second["axis"]):
        return False
    first_length = max(1.0, float(first["length"]))
    second_length = max(1.0, float(second["length"]))
    max_length = max(first_length, second_length)
    length_delta_ratio = abs(first_length - second_length) / float(max_length)
    if length_delta_ratio > max(0.0, float(max_length_delta_ratio)) + 1.0e-9:
        return False
    max_shift = max(1.0, float(max_length) * max(0.0, float(max_scanline_shift_ratio)))
    return abs(float(first["along_center"]) - float(second["along_center"])) <= max_shift + 1.0e-9


def _nearest_two_obstacle_pixels(
    free: np.ndarray,
    r: int,
    c: int,
    *,
    distance_cells: float,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    obstacle = ~np.asarray(free, dtype=bool)
    h, w = obstacle.shape
    base_radius = max(3, int(np.ceil(float(distance_cells))) + 3)
    max_radius = max(h, w)
    for radius in (base_radius, base_radius * 2, base_radius * 4, max_radius):
        r0 = max(0, r - radius)
        r1 = min(h, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(w, c + radius + 1)
        coords = np.argwhere(obstacle[r0:r1, c0:c1])
        if coords.shape[0] < 2:
            continue
        coords[:, 0] += r0
        coords[:, 1] += c0
        deltas = coords.astype(np.float32) - np.asarray([[r, c]], dtype=np.float32)
        distances = np.sum(deltas * deltas, axis=1)
        order = np.argsort(distances)
        first = (int(coords[int(order[0]), 0]), int(coords[int(order[0]), 1]))
        min_basis_distance = max(1.0, float(distance_cells))
        for idx in order[1:].tolist():
            second = (int(coords[int(idx), 0]), int(coords[int(idx), 1]))
            if _point_distance(first, second) >= min_basis_distance:
                return first, second
    return None


def _filter_area_limits(
    labels: np.ndarray,
    domain: np.ndarray,
    *,
    resolution_m: float,
    lower_m2: float,
    upper_m2: float,
) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    out[~np.asarray(domain, dtype=bool)] = 0
    cell_area = float(resolution_m) * float(resolution_m)
    for label in [int(v) for v in np.unique(out) if int(v) > 0]:
        area_m2 = float(np.count_nonzero(out == label)) * cell_area
        if area_m2 < float(lower_m2) or area_m2 > float(upper_m2):
            out[out == label] = 0
    if bool(np.any(out > 0)):
        out = nearest_seed_fill(out, domain=domain)
    return relabel_consecutive(out)


def _merge_small_regions(
    labels: np.ndarray,
    domain: np.ndarray,
    *,
    resolution_m: float,
    merge_area_threshold_m2: float,
    force_merge_area_m2: float,
) -> tuple[np.ndarray, int]:
    out = relabel_consecutive(labels)
    out[~np.asarray(domain, dtype=bool)] = 0
    merges = 0
    cell_area = float(resolution_m) * float(resolution_m)
    for _ in range(1000):
        changed = False
        adjacency = _label_adjacency(out)
        perimeters = _label_perimeters(out)
        areas = {int(label): float(np.count_nonzero(out == label)) * cell_area for label in np.unique(out) if int(label) > 0}
        for label, area_m2 in sorted(areas.items(), key=lambda item: item[1]):
            all_neighbors = adjacency.get(int(label), {})
            neighbors = {int(k): int(v) for k, v in all_neighbors.items() if int(k) > 0}
            if not neighbors:
                continue
            perimeter = max(1, int(perimeters.get(int(label), 1)))
            target = max(neighbors, key=lambda neighbor: int(neighbors[neighbor]))
            shared_ratio = float(neighbors[target]) / float(perimeter)
            wall_ratio = float(all_neighbors.get(0, 0)) / float(perimeter)
            force_merge = area_m2 < float(force_merge_area_m2)
            heuristic_merge = (
                area_m2 < float(merge_area_threshold_m2)
                and (shared_ratio > 0.2 or (len(neighbors) <= 1 and wall_ratio <= 0.75))
            )
            if force_merge or heuristic_merge:
                out[out == int(label)] = int(target)
                out = relabel_consecutive(out)
                merges += 1
                changed = True
                break
        if not changed:
            break
    out[~np.asarray(domain, dtype=bool)] = 0
    return relabel_consecutive(out), int(merges)


def _label_adjacency(labels: np.ndarray) -> dict[int, dict[int, int]]:
    arr = np.asarray(labels, dtype=np.int32)
    adjacency: dict[int, dict[int, int]] = {}
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            label = int(arr[r, c])
            if label <= 0:
                continue
            adjacency.setdefault(label, {})
            for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                nr = r + dr
                nc = c + dc
                neighbor = 0 if nr < 0 or nc < 0 or nr >= h or nc >= w else int(arr[nr, nc])
                if neighbor == label:
                    continue
                adjacency[label][neighbor] = adjacency[label].get(neighbor, 0) + 1
    return adjacency


def _label_perimeters(labels: np.ndarray) -> dict[int, int]:
    return {label: int(sum(counts.values())) for label, counts in _label_adjacency(labels).items()}


def _angle_deg(origin: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> float:
    va = np.asarray([a[0] - origin[0], a[1] - origin[1]], dtype=np.float64)
    vb = np.asarray([b[0] - origin[0], b[1] - origin[1]], dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0:
        return 0.0
    cosine = float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _point_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _infer_shape(arrays: Mapping[str, Any]) -> tuple[int, int]:
    for key in (
        "final_room_label_map",
        "navigation_free_room_domain",
        "observed_free_mask",
        "occupancy_map",
        "unknown_mask",
        "voxel_nav_free_xy",
        "voxel_vertical_free_xy",
    ):
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim >= 2:
            return int(arr.shape[0]), int(arr.shape[1])
    raise KeyError("cannot infer snapshot map shape")


def _first_bool_array(
    arrays: Mapping[str, Any],
    keys: tuple[str, ...],
    shape: tuple[int, int],
    *,
    require_any: bool = False,
) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.shape != shape:
            continue
        out = arr.astype(bool)
        if require_any and not bool(np.any(out)):
            continue
        return out, key
    return None, None


def _smoke_arrays() -> dict[str, np.ndarray]:
    shape = (36, 64)
    observed = np.zeros(shape, dtype=bool)
    observed[6:30, 6:25] = True
    observed[6:30, 39:58] = True
    observed[15:21, 25:39] = True
    unknown = np.zeros(shape, dtype=bool)
    unknown[15:21, 30:34] = True
    obstacle = ~observed
    return {
        "occupancy_map": obstacle,
        "observed_free_mask": observed,
        "obstacle_mask": obstacle,
        "unknown_mask": unknown,
        "final_room_label_map": np.zeros(shape, dtype=np.int32),
        "map_resolution_m": np.asarray(0.05, dtype=np.float32),
        "map_origin_xy_m": np.asarray([0.0, 0.0], dtype=np.float32),
        "map_width_cells": np.asarray(shape[1], dtype=np.int32),
        "map_height_cells": np.asarray(shape[0], dtype=np.int32),
    }


def _run_smoke() -> int:
    arrays = _smoke_arrays()
    labels, metadata, debug = segment_snapshot_arrays("smoke/roomseg_step_000001.npz", arrays)
    unknown = np.asarray(arrays["unknown_mask"], dtype=bool)
    free, _ = build_voronoi_free_mask(arrays)
    assert labels.shape == free.shape
    assert labels.dtype == np.int32
    assert not np.any(labels[unknown] > 0)
    assert not np.any(labels[~free] > 0)
    assert int(metadata["original_algorithm_id"]) == VORONOI_ALGORITHM_ID
    assert metadata["runner_type"] == "python_fallback"
    assert metadata["main_experiment_allowed"] is False
    assert "voronoi_separator_mask" in debug
    print(
        json.dumps(
            {
                "labels": sorted(int(v) for v in np.unique(labels) if int(v) > 0),
                "metadata_runner_type": metadata["runner_type"],
                "metadata_fallback_scope": metadata["fallback_scope"],
                "main_experiment_allowed": metadata["main_experiment_allowed"],
                "critical_line_min_angle_deg": metadata["critical_line_min_angle_deg"],
                "merge_area_threshold_m2": metadata["merge_area_threshold_m2"],
                "force_merge_area_m2": metadata["force_merge_area_m2"],
                "unknown_labeled_cells": int(np.count_nonzero(labels[unknown] > 0)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Width-jump Python variant for Bormann Voronoi segmentation.")
    parser.add_argument("--smoke", action="store_true", help="Run a toy-map smoke test.")
    args = parser.parse_args(argv)
    if args.smoke:
        return _run_smoke()
    parser.error("fallback_voronoi_width_jump.py is smoke-only; pass --smoke or call voronoi_segment() from tests.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
