from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from voxroom_online.isaac_runtime.mapping.online_roomseg.utils import conn, dilate, disk, rasterize_line
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import (
    VOXEL_CONFLICT,
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
)


DOOR_ANCHOR_NONE = 0
DOOR_ANCHOR_STRICT_RAW = 1
DOOR_ANCHOR_PROJECTED = 2
DOOR_ANCHOR_PROJECTED_ANCHOR = 3
DOOR_ANCHOR_STEP1 = 4
DOOR_ANCHOR_FILTERED_LINE = 5
DOOR_ANCHOR_NO_CLEARANCE_OCCUPIED = 6

DOOR_ANCHOR_SOURCE_NAMES = {
    DOOR_ANCHOR_NONE: "none",
    DOOR_ANCHOR_STRICT_RAW: "strict_raw",
    DOOR_ANCHOR_PROJECTED: "projected",
    DOOR_ANCHOR_PROJECTED_ANCHOR: "projected_anchor",
    DOOR_ANCHOR_STEP1: "step1",
    DOOR_ANCHOR_FILTERED_LINE: "filtered_line",
    DOOR_ANCHOR_NO_CLEARANCE_OCCUPIED: "no_clearance_occupied",
}

DOOR_COMPLETION_MIDDLE_SEED_TWO_WALL = "middle_seed_two_wall"
DOOR_COMPLETION_SEED_PAIR_BRIDGE = "seed_pair_bridge"
DOOR_COMPLETION_ONE_SEED_ONE_WALL = "one_seed_one_wall"
DOOR_COMPLETION_STRONG_SEED_CENTERLINE = "strong_seed_centerline"
DOOR_COMPLETION_VISUAL_ONLY = "visual_only"
DOOR_COMPLETION_REJECTED = "rejected"


@dataclass
class VoxelDoorDetectorConfig:
    enabled: bool = True
    seed_method: str = "centroid_ratio"
    require_navigation_free_seed: bool = True
    z_scan_min_m: float = 0.10
    z_scan_max_mode: str = "active_z_max"
    z_scan_top_mode: str = "ceiling_ratio_or_fallback"
    z_scan_max_m: float = 3.00
    door_seed_scan_max_m: float = 3.00
    door_seed_scan_fallback_m: float = 3.00
    ceiling_scan_ratio: float = 0.95
    door_turn_z_min_m: float = 1.65
    door_turn_z_max_m: float = 0.0
    first_upper_occupied_z_max_m: float = 0.0
    ceiling_occupied_reject_band_m: float = 0.0
    reject_upper_occupied_only_in_ceiling_band: bool = False
    centroid_turn_extent_scale: float = 0.50
    min_lower_free_cells: int = 2
    lower_free_ratio_min: float = 0.90
    top_occupied_min_z_m: float = 1.80
    upper_occupied_ratio_min_observed: float = 0.80
    min_upper_observed_cells: int = 3
    min_upper_occupied_cells: int = 3
    door_seed_sensor_range_count_threshold: int = 0
    upper_solid_use_in_range_unknown: bool = False
    upper_solid_use_out_of_range_unknown: bool = True
    upper_generalized_solid_ratio_min_active: float = 0.80
    upper_in_range_unknown_ratio_max: float = 0.25
    upper_free_ratio_max: float = 0.15
    upper_actual_occupied_min_cells: int = 3
    require_occupied_centroid_above_free_centroid: bool = True
    min_top_occupied_cells: int = 3
    allow_unknown_tail_after_top_occupied: bool = True
    allow_end_of_active_range_after_top_occupied: bool = True
    reject_unknown_before_top_occupied: bool = True
    reject_occupied_below_top_threshold: bool = True
    reject_free_after_unknown_tail: bool = True
    reject_occupied_after_unknown_tail: bool = True
    reject_conflict_in_seed_pattern: bool = True
    seed_connectivity: int = 8
    min_seed_component_cells: int = 1
    min_seed_component_bbox_extent_cells: int = 3
    max_seed_component_cells: int = 160
    max_component_thickness_m: float = 0.45
    line_fit_max_residual_cells: float = 1.25
    single_seed_infer_from_wall_enabled: bool = True
    single_seed_wall_search_radius_m: float = 1.20
    door_width_min_m: float = 0.35
    door_width_max_m: float = 1.60
    enforce_seed_door_width_limits: bool = False
    extend_max_m: float = 1.60
    partition_cut_max_total_extension_m: float = 1.60
    wall_anchor_radius_cells: int = 3
    wall_anchor_min_cells: int = 1
    cut_thickness_cells: int = 1
    inner_unknown_ratio_max: float = 0.20
    inner_wall_ratio_max: float = 0.10
    inner_free_or_seed_ratio_min: float = 0.40
    reject_if_intersects_other_door: bool = True
    reject_if_endpoint_is_other_door: bool = True
    conflict_dilation_cells: int = 1
    vectorized_seed_classification: bool = True
    seed_cluster_morph_close_radius_cells: int = 1
    seed_cluster_merge_distance_cells: int = 4
    seed_cluster_collinear_angle_deg: float = 25.0
    seed_cluster_max_perpendicular_gap_cells: int = 2
    seed_cluster_max_along_gap_cells: int = 12
    seed_cluster_max_width_m: float = 1.80
    seed_cluster_min_shared_anchor_score: float = 0.0
    seed_component_connectivity: int = 8
    enable_seed_same_line_gap_bridge: bool = True
    seed_same_line_gap_bridge_max_gap_cells: int = 28
    seed_same_line_gap_bridge_max_perpendicular_cells: int = 1
    seed_same_line_gap_bridge_max_angle_deg: float = 10.0
    seed_same_line_gap_bridge_max_length_m: float = 4.0
    raw_seed_min_component_cells_for_display: int = 1
    raw_seed_requires_wall_context: bool = True
    raw_seed_wall_context_radius_cells: int = 6
    raw_seed_min_wall_context_cells: int = 2
    raw_seed_reject_inside_large_open_ceiling_area: bool = True
    primitive_min_seed_cells: int = 3
    primitive_min_length_cells: int = 3
    primitive_min_contiguous_seed_run_cells: int = 7
    primitive_contiguous_gap_cells: float = 3.00
    primitive_max_thickness_cells: int = 3
    primitive_max_residual_cells: float = 1.75
    primitive_min_elongation: float = 1.4
    primitive_max_along_gap_cells: float = 0.0
    primitive_min_line_correlation: float = 0.95
    primitive_max_orthogonal_variance_cells2: float = 0.65
    primitive_require_line_fit_quality: bool = True
    enable_seed_line_spur_pruning: bool = True
    seed_line_spur_prune_max_branch_length_ratio: float = 0.35
    seed_line_spur_prune_max_total_branch_ratio: float = 0.45
    enable_seed_parallel_line_pruning: bool = True
    seed_parallel_line_max_distance_cells: int = 8
    seed_parallel_line_min_overlap_ratio: float = 0.70
    seed_parallel_line_max_length_delta_ratio: float = 0.25
    enable_seed_blob_line_decomposition: bool = False
    primitive_extraction_method: str = "ransac_then_split"
    ransac_num_trials: int = 64
    ransac_inlier_residual_cells: float = 1.5
    ransac_min_inliers: int = 3
    max_primitives_per_cluster: int = 4
    remove_inliers_after_primitive: bool = True
    seed_pair_max_along_gap_cells: int = 12
    seed_pair_max_axis_angle_deg: float = 25.0
    completion_orientation_mode: str = "pca_plus_axis_plus_wall_pair"
    accepted_orientation_source: str = "seed_primitive_only"
    allow_axis_hv_for_accepted: bool = False
    allow_wall_pair_axis_for_accepted: bool = False
    allow_local_free_neck_axis_for_accepted: bool = False
    allow_axis_hv_for_debug_trials: bool = True
    allow_wall_pair_axis_for_debug_trials: bool = True
    allow_local_free_neck_axis_for_debug_trials: bool = True
    raw_seed_blocks_step2: bool = False
    visual_only_door_blocks_step2: bool = False
    geometry_warning_door_blocks_step2: bool = False
    topology_effective_door_blocks_step2: bool = True
    show_raw_seed: bool = True
    show_seed_line_primitives: bool = True
    show_rejected_primitives_in_diagnostic: bool = True
    show_visual_only_door_in_diagnostic: bool = True
    default_green_only_topology_effective: bool = True
    min_seed_cells_for_accepted_extension: int = 3
    min_seed_line_length_cells_for_accepted_extension: int = 3
    min_seed_elongation_for_direction: float = 1.6
    max_seed_line_residual_cells_for_direction: float = 1.25
    accepted_orientation_mode: str = "seed_major_only"
    allow_axis_orientation_if_aligned_with_seed: bool = True
    axis_orientation_max_angle_to_seed_deg: float = 15.0
    allow_wall_pair_orientation_if_aligned_with_seed: bool = True
    wall_pair_orientation_max_angle_to_seed_deg: float = 20.0
    local_free_neck_orientation_debug_only: bool = True
    single_seed_completion_debug_only: bool = True
    allow_diagonal_orientation_candidates: bool = False
    infer_orientation_from_wall_pairs: bool = True
    infer_orientation_from_local_free_neck: bool = True
    visual_walk_ignore_other_seed_clusters: bool = True
    visual_walk_continue_through_unknown: bool = True
    visual_walk_unknown_bridge_max_cells: int = 2
    visual_width_min_m: float = 0.15
    visual_width_max_m: float = 1.80
    door_line_inner_real_wall_cells_max: int = 8
    door_extension_no_clearance_occupied_reject_cells: int = 3
    door_extension_no_clearance_occupied_surface_tolerance_cells: int = 1
    reject_large_open_free_crossing: bool = False
    one_seed_one_wall_visual_width_max_m: float = 1.60
    seed_pair_bridge_visual_width_max_m: float = 1.80
    partition_cut_bridge_unknown_max_cells: int = 10
    partition_cut_bridge_nonfree_max_cells: int = 1
    partition_cut_seed_dilation_cells: int = 2
    partition_cut_require_seed_overlap: bool = True
    partition_cut_min_cells: int = 1
    partition_inner_unknown_ratio_max: float = 0.80
    partition_inner_wall_ratio_max: float = 0.40
    partition_inner_free_or_seed_ratio_min: float = 0.10
    partition_topology_enabled: bool = True
    partition_topology_local_radius_cells: int = 18
    partition_topology_allow_anchor_closure_without_global_gain: bool = True
    partition_topology_allow_neck_cut_without_global_gain: bool = True
    partition_topology_min_side_area_cells: int = 3
    partition_topology_min_side_width_cells: int = 1
    partition_reject_small_known_side_enabled: bool = True
    partition_small_known_side_area_m2: float = 2.00
    partition_small_known_side_unknown_ratio_max: float = 0.20
    partition_small_known_side_boundary_dilation_cells: int = 1
    enable_one_seed_one_wall_completion: bool = True
    enable_seed_pair_bridge_completion: bool = True
    seed_pair_max_center_distance_m: float = 1.40
    seed_pair_max_perpendicular_gap_cells: int = 2
    seed_pair_max_angle_deg: float = 25.0
    seed_pair_bridge_gap_tolerance_cells: int = 2
    door_walk_strip_half_width_cells: int = 1
    door_walk_anchor_snap_radius_cells: int = 3
    door_walk_unknown_effective_ignores_anchor: bool = True
    door_walk_no_clearance_occupied_anchor_enabled: bool = False
    door_walk_no_clearance_occupied_anchor_support_radius_cells: int = 24
    door_partition_accept_mode: str = "geometry_first"
    partition_topology_reject_mode: str = "warn_only"
    min_geometry_cut_cells: int = 1
    door_wall_attachment_max_endpoint_gap_cells: int = 1
    enable_strong_seed_centerline_fallback: bool = True
    strong_seed_min_cells: int = 6
    strong_seed_max_thickness_cells: int = 5
    strong_seed_allow_topology_test_without_two_anchors: bool = True
    strong_seed_centerline_min_cells: int = 4
    strong_seed_centerline_min_length_cells: int = 3
    strong_seed_centerline_min_elongation: float = 1.6
    strong_seed_centerline_max_residual_cells: float = 1.25
    door_memory_enabled: bool = True
    door_memory_initial_confidence: float = 1.0
    door_memory_match_iou_min: float = 0.05
    door_memory_match_distance_cells: int = 6
    door_memory_match_angle_deg: float = 20.0
    door_memory_observation_dilation_cells: int = 2
    door_memory_min_observed_cells_for_decay: int = 2
    door_memory_contradiction_band_cells: int = 2
    door_memory_min_observed_cells_for_contradiction: int = 4
    door_memory_contradict_wall_ratio: float = 0.75
    door_memory_contradictions_to_prune: int = 5
    door_memory_match_dilation_cells: int = 2
    door_memory_dilated_iou_min: float = 0.10
    door_memory_seed_overlap_min: float = 0.20
    door_memory_anchor_match_distance_cells: int = 4
    door_memory_weak_refresh_updates_visual: bool = False
    door_memory_replace_quality_margin: float = 0.20
    door_memory_allow_verified_geometry_replace: bool = True
    door_memory_freeze_geometry_unless_ceiling_changes: bool = True
    door_memory_ceiling_change_threshold_m: float = 0.05
    door_memory_prevent_shrinking_stable_cut: bool = True
    door_memory_min_length_ratio_to_replace: float = 0.80
    door_memory_decay_per_update: float = 0.02
    door_memory_confirm_increment: float = 0.35
    door_memory_weak_refresh_increment: float = 0.12
    door_memory_min_confidence_to_keep: float = 0.15
    door_memory_ttl_updates: int = 30
    show_candidate_lines_in_debug: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "VoxelDoorDetectorConfig":
        if isinstance(data, cls):
            return data
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class VoxelDoorSeedEvidence:
    row: int
    col: int
    first_occupied_z_m: float | None
    lower_free_cells: int
    top_occupied_cells: int
    unknown_tail_cells: int
    accepted: bool
    reject_reason: str | None
    turn_z_m: float | None = None
    free_centroid_z_m: float | None = None
    occupied_centroid_z_m: float | None = None
    lower_free_ratio: float | None = None
    upper_occupied_ratio_observed: float | None = None
    upper_observed_cells: int = 0
    upper_occupied_cells: int = 0

    def to_dict(self) -> dict:
        return {
            "row": int(self.row),
            "col": int(self.col),
            "first_occupied_z_m": None if self.first_occupied_z_m is None else float(self.first_occupied_z_m),
            "lower_free_cells": int(self.lower_free_cells),
            "top_occupied_cells": int(self.top_occupied_cells),
            "unknown_tail_cells": int(self.unknown_tail_cells),
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
            "turn_z_m": None if self.turn_z_m is None else float(self.turn_z_m),
            "free_centroid_z_m": None if self.free_centroid_z_m is None else float(self.free_centroid_z_m),
            "occupied_centroid_z_m": None if self.occupied_centroid_z_m is None else float(self.occupied_centroid_z_m),
            "lower_free_ratio": None if self.lower_free_ratio is None else float(self.lower_free_ratio),
            "upper_occupied_ratio_observed": None if self.upper_occupied_ratio_observed is None else float(self.upper_occupied_ratio_observed),
            "upper_observed_cells": int(self.upper_observed_cells),
            "upper_occupied_cells": int(self.upper_occupied_cells),
        }


@dataclass
class VoxelDoorLineCandidate:
    candidate_id: int
    seed_component_id: int
    seed_cells: list[tuple[int, int]]
    center_rc: tuple[float, float]
    major_dir_rc: tuple[float, float]
    minor_dir_rc: tuple[float, float]
    seed_projected_centerline_cells: list[tuple[int, int]]
    extended_centerline_cells: list[tuple[int, int]]
    door_cut_cells: list[tuple[int, int]]
    wall_anchor_a: tuple[int, int] | None
    wall_anchor_b: tuple[int, int] | None
    width_m: float
    accepted: bool
    reject_reason: str | None
    debug: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "candidate_id": int(self.candidate_id),
            "seed_component_id": int(self.seed_component_id),
            "seed_cells": [[int(r), int(c)] for r, c in self.seed_cells],
            "center_rc": [float(self.center_rc[0]), float(self.center_rc[1])],
            "major_dir_rc": [float(self.major_dir_rc[0]), float(self.major_dir_rc[1])],
            "minor_dir_rc": [float(self.minor_dir_rc[0]), float(self.minor_dir_rc[1])],
            "seed_projected_centerline_cells": [[int(r), int(c)] for r, c in self.seed_projected_centerline_cells],
            "extended_centerline_cells": [[int(r), int(c)] for r, c in self.extended_centerline_cells],
            "door_cut_cells": [[int(r), int(c)] for r, c in self.door_cut_cells],
            "wall_anchor_a": None if self.wall_anchor_a is None else [int(self.wall_anchor_a[0]), int(self.wall_anchor_a[1])],
            "wall_anchor_b": None if self.wall_anchor_b is None else [int(self.wall_anchor_b[0]), int(self.wall_anchor_b[1])],
            "width_m": float(self.width_m),
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
            "debug": _jsonable(self.debug),
        }
        for key in (
            "cluster_id",
            "seed_group_id",
            "component_ids",
            "completion_mode",
            "orientation_source",
            "anchor_a_source",
            "anchor_b_source",
            "visual_line_cells",
            "partition_cut_cells",
            "partition_geometry_accepted",
            "partition_topology_effective",
            "visual_accepted",
            "partition_accepted",
            "partition_topology_accepted",
            "reject_reason_topology",
            "door_wall_attached",
            "door_wall_attachment_reject_reason",
            "door_topology_accepted",
            "door_topology_reject_reason",
            "door_topology_before_components",
            "door_topology_after_components",
            "door_topology_touched_labels",
            "door_topology_side_areas",
            "door_topology_side_widths_cells",
            "reject_reason_visual",
            "reject_reason_partition",
            "inner_unknown_ratio",
            "inner_wall_ratio",
            "inner_free_or_seed_ratio",
            "score",
        ):
            if key in self.debug:
                out[key] = _jsonable(self.debug[key])
        return out


@dataclass
class DoorExtensionTrial:
    trial_id: int
    cluster_id: int
    seed_group_id: int
    component_ids: list[int]
    completion_mode: str
    orientation_source: str
    direction_rc: tuple[float, float]
    seed_cells: list[tuple[int, int]]
    seed_centerline_cells: list[tuple[int, int]]
    walk_a_cells: list[tuple[int, int]]
    walk_b_cells: list[tuple[int, int]]
    anchor_a: tuple[int, int] | None
    anchor_b: tuple[int, int] | None
    anchor_a_source: str | None
    anchor_b_source: str | None
    anchor_a_reject_reason: str | None
    anchor_b_reject_reason: str | None
    visual_line_cells: list[tuple[int, int]]
    visual_status: str
    visual_accepted: bool
    partition_cut_candidate_cells: list[tuple[int, int]]
    partition_cut_accepted_cells: list[tuple[int, int]]
    partition_status: str
    partition_accepted: bool
    topology_status: str
    topology_accepted: bool
    topology_before_components: int
    topology_after_components: int
    topology_touched_labels: list[int]
    topology_side_areas: list[int]
    topology_side_widths_cells: list[float]
    score: float
    reject_reason: str | None
    debug: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_candidate(cls, candidate: VoxelDoorLineCandidate) -> "DoorExtensionTrial":
        debug = dict(candidate.debug)
        return cls(
            trial_id=int(candidate.candidate_id),
            cluster_id=int(debug.get("cluster_id", candidate.seed_component_id)),
            seed_group_id=int(debug.get("seed_group_id", debug.get("cluster_id", candidate.seed_component_id))),
            component_ids=[int(v) for v in debug.get("component_ids", [candidate.seed_component_id])],
            completion_mode=str(debug.get("completion_mode", DOOR_COMPLETION_REJECTED if not candidate.accepted else DOOR_COMPLETION_MIDDLE_SEED_TWO_WALL)),
            orientation_source=str(debug.get("orientation_source", "unknown")),
            direction_rc=(float(candidate.major_dir_rc[0]), float(candidate.major_dir_rc[1])),
            seed_cells=list(candidate.seed_cells),
            seed_centerline_cells=list(candidate.seed_projected_centerline_cells),
            walk_a_cells=[tuple(v) for v in debug.get("walk_a_cells", [])],  # type: ignore[arg-type]
            walk_b_cells=[tuple(v) for v in debug.get("walk_b_cells", [])],  # type: ignore[arg-type]
            anchor_a=candidate.wall_anchor_a,
            anchor_b=candidate.wall_anchor_b,
            anchor_a_source=None if debug.get("anchor_a_source") is None else str(debug.get("anchor_a_source")),
            anchor_b_source=None if debug.get("anchor_b_source") is None else str(debug.get("anchor_b_source")),
            anchor_a_reject_reason=None if debug.get("anchor_a_reject_reason") is None else str(debug.get("anchor_a_reject_reason")),
            anchor_b_reject_reason=None if debug.get("anchor_b_reject_reason") is None else str(debug.get("anchor_b_reject_reason")),
            visual_line_cells=list(candidate.extended_centerline_cells),
            visual_status=str(debug.get("visual_status", "completed_two_wall_anchors" if candidate.accepted else "rejected")),
            visual_accepted=bool(debug.get("visual_accepted", candidate.accepted)),
            partition_cut_candidate_cells=[tuple(v) for v in debug.get("partition_cut_candidate_cells", [])],  # type: ignore[arg-type]
            partition_cut_accepted_cells=list(candidate.door_cut_cells),
            partition_status=str(debug.get("partition_status", "accepted" if debug.get("partition_accepted") else debug.get("reject_reason_partition", "rejected"))),
            partition_accepted=bool(debug.get("partition_accepted", False)),
            topology_status=str(debug.get("reject_reason_topology") or debug.get("door_topology_reject_reason") or ("accepted" if debug.get("partition_topology_effective") else "not_run")),
            topology_accepted=bool(debug.get("partition_topology_effective", debug.get("door_topology_accepted", False))),
            topology_before_components=int(debug.get("door_topology_before_components", 0) or 0),
            topology_after_components=int(debug.get("door_topology_after_components", 0) or 0),
            topology_touched_labels=[int(v) for v in debug.get("door_topology_touched_labels", [])],
            topology_side_areas=[int(v) for v in debug.get("door_topology_side_areas", [])],
            topology_side_widths_cells=[float(v) for v in debug.get("door_topology_side_widths_cells", [])],
            score=float(debug.get("score", 0.0) or 0.0),
            reject_reason=candidate.reject_reason,
            debug=debug,
        )

    def to_dict(self) -> dict[str, object]:
        return _jsonable(self.__dict__)


@dataclass
class VoxelDoorDetectionResult:
    door_seed_mask: np.ndarray
    door_seed_component_map: np.ndarray
    door_centerline_candidate_mask: np.ndarray
    accepted_door_centerline_mask: np.ndarray
    rejected_door_centerline_mask: np.ndarray
    door_cut_mask: np.ndarray
    door_seed_reject_reason_map: np.ndarray
    candidates: list[VoxelDoorLineCandidate]
    debug: dict[str, object]


@dataclass
class VoxelDoorSeedResult:
    door_seed_mask: np.ndarray
    door_seed_component_map: np.ndarray
    door_seed_reject_reason_map: np.ndarray
    lower_free_cells_xy: np.ndarray
    top_occupied_cells_xy: np.ndarray
    first_occupied_z_xy: np.ndarray
    unknown_tail_cells_xy: np.ndarray
    seed_evidence: list[VoxelDoorSeedEvidence]
    debug: dict[str, object]


@dataclass
class DoorSeedCluster:
    cluster_id: int
    component_ids: list[int]
    seed_cells: list[tuple[int, int]]
    mask: np.ndarray
    bbox_rc: tuple[int, int, int, int]
    center_rc: tuple[float, float]
    major_dir_rc: tuple[float, float]
    minor_dir_rc: tuple[float, float]
    line_fit_residual_cells: float
    thickness_m: float
    length_m: float
    accepted_for_completion: bool
    reject_reason: str | None
    debug: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": int(self.cluster_id),
            "component_ids": [int(v) for v in self.component_ids],
            "seed_cells": [[int(r), int(c)] for r, c in self.seed_cells],
            "bbox_rc": [int(v) for v in self.bbox_rc],
            "center_rc": [float(self.center_rc[0]), float(self.center_rc[1])],
            "major_dir_rc": [float(self.major_dir_rc[0]), float(self.major_dir_rc[1])],
            "minor_dir_rc": [float(self.minor_dir_rc[0]), float(self.minor_dir_rc[1])],
            "line_fit_residual_cells": float(self.line_fit_residual_cells),
            "thickness_m": float(self.thickness_m),
            "length_m": float(self.length_m),
            "accepted_for_completion": bool(self.accepted_for_completion),
            "reject_reason": self.reject_reason,
            "debug": _jsonable(self.debug),
        }


@dataclass
class DoorSeedGroup:
    group_id: int
    group_kind: str
    source_cluster_ids: list[int]
    component_ids: list[int]
    seed_cells: list[tuple[int, int]]
    mask: np.ndarray
    bbox_rc: tuple[int, int, int, int]
    center_rc: tuple[float, float]
    major_dir_rc: tuple[float, float]
    minor_dir_rc: tuple[float, float]
    line_fit_residual_cells: float
    thickness_m: float
    length_m: float
    accepted_for_completion: bool
    reject_reason: str | None
    debug: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_cluster(cls, cluster: DoorSeedCluster) -> "DoorSeedGroup":
        return cls(
            group_id=int(cluster.cluster_id),
            group_kind="single_cluster",
            source_cluster_ids=[int(cluster.cluster_id)],
            component_ids=[int(v) for v in cluster.component_ids],
            seed_cells=list(cluster.seed_cells),
            mask=np.asarray(cluster.mask, dtype=bool),
            bbox_rc=cluster.bbox_rc,
            center_rc=(float(cluster.center_rc[0]), float(cluster.center_rc[1])),
            major_dir_rc=(float(cluster.major_dir_rc[0]), float(cluster.major_dir_rc[1])),
            minor_dir_rc=(float(cluster.minor_dir_rc[0]), float(cluster.minor_dir_rc[1])),
            line_fit_residual_cells=float(cluster.line_fit_residual_cells),
            thickness_m=float(cluster.thickness_m),
            length_m=float(cluster.length_m),
            accepted_for_completion=bool(cluster.accepted_for_completion),
            reject_reason=cluster.reject_reason,
            debug={"source": "cluster", **dict(cluster.debug)},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": int(self.group_id),
            "group_kind": str(self.group_kind),
            "source_cluster_ids": [int(v) for v in self.source_cluster_ids],
            "component_ids": [int(v) for v in self.component_ids],
            "seed_cells": [[int(r), int(c)] for r, c in self.seed_cells],
            "bbox_rc": [int(v) for v in self.bbox_rc],
            "center_rc": [float(self.center_rc[0]), float(self.center_rc[1])],
            "major_dir_rc": [float(self.major_dir_rc[0]), float(self.major_dir_rc[1])],
            "minor_dir_rc": [float(self.minor_dir_rc[0]), float(self.minor_dir_rc[1])],
            "line_fit_residual_cells": float(self.line_fit_residual_cells),
            "thickness_m": float(self.thickness_m),
            "length_m": float(self.length_m),
            "accepted_for_completion": bool(self.accepted_for_completion),
            "reject_reason": self.reject_reason,
            "debug": _jsonable(self.debug),
        }


@dataclass
class DoorSeedLinePrimitive:
    primitive_id: int
    source_cluster_id: int
    source_group_id: int
    source_component_ids: list[int]
    cells: list[tuple[int, int]]
    center_rc: tuple[float, float]
    major_dir_rc: tuple[float, float]
    minor_dir_rc: tuple[float, float]
    length_cells: float
    thickness_cells: float
    residual_cells: float
    elongation: float
    seed_count: int
    bbox_rc: tuple[int, int, int, int]
    along_min: float
    along_max: float
    max_along_gap_cells: float
    contiguous_segment_count: int
    accepted_for_extension: bool
    reject_reason: str | None
    extraction_method: str
    debug: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "primitive_id": int(self.primitive_id),
            "source_cluster_id": int(self.source_cluster_id),
            "source_group_id": int(self.source_group_id),
            "source_component_ids": [int(v) for v in self.source_component_ids],
            "seed_count": int(self.seed_count),
            "bbox": [int(v) for v in self.bbox_rc],
            "center_rc": [float(self.center_rc[0]), float(self.center_rc[1])],
            "major_dir_rc": [float(self.major_dir_rc[0]), float(self.major_dir_rc[1])],
            "minor_dir_rc": [float(self.minor_dir_rc[0]), float(self.minor_dir_rc[1])],
            "length_cells": float(self.length_cells),
            "thickness_cells": float(self.thickness_cells),
            "residual_cells": float(self.residual_cells),
            "elongation": float(self.elongation),
            "along_min": float(self.along_min),
            "along_max": float(self.along_max),
            "max_along_gap_cells": float(self.max_along_gap_cells),
            "contiguous_segment_count": int(self.contiguous_segment_count),
            "accepted_for_extension": bool(self.accepted_for_extension),
            "reject_reason": self.reject_reason,
            "extraction_method": str(self.extraction_method),
            "debug": _jsonable(self.debug),
        }


@dataclass
class DoorCompletionStageMaps:
    seed_mask: np.ndarray
    seed_cluster_map: np.ndarray
    candidate_line_mask: np.ndarray
    provisional_accepted_visual_mask: np.ndarray
    final_accepted_visual_mask: np.ndarray
    partition_cut_mask: np.ndarray
    rejected_line_mask: np.ndarray
    rejected_by_reason_map: np.ndarray


@dataclass
class DoorTrialCandidateGroup:
    cluster_id: int
    component_ids: list[int]
    trials: list[VoxelDoorLineCandidate]
    selected_candidate_id: int | None
    selected_reason: str
    debug: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": int(self.cluster_id),
            "component_ids": [int(v) for v in self.component_ids],
            "trials": [item.to_dict() for item in self.trials],
            "selected_candidate_id": None if self.selected_candidate_id is None else int(self.selected_candidate_id),
            "selected_reason": str(self.selected_reason),
            "debug": _jsonable(self.debug),
        }


@dataclass
class DoorPartitionCutResult:
    mask: np.ndarray
    ordered_cells: list[tuple[int, int]]
    cut_cells: list[tuple[int, int]]
    bridged_cells: list[tuple[int, int]]
    debug: dict[str, object] = field(default_factory=dict)


@dataclass
class DoorTopologyValidationResult:
    topology_accepted: bool
    reject_reason: str | None
    before_components: int
    after_components: int
    touched_labels: list[int]
    new_component_count: int
    side_component_areas: list[int]
    side_component_widths_cells: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "door_topology_accepted": bool(self.topology_accepted),
            "door_topology_reject_reason": self.reject_reason,
            "door_topology_before_components": int(self.before_components),
            "door_topology_after_components": int(self.after_components),
            "door_topology_touched_labels": [int(v) for v in self.touched_labels],
            "door_topology_new_component_count": int(self.new_component_count),
            "door_topology_side_areas": [int(v) for v in self.side_component_areas],
            "door_topology_side_widths_cells": [float(v) for v in self.side_component_widths_cells],
        }


@dataclass
class DoorAttachmentValidation:
    attached: bool
    left_attached: bool
    right_attached: bool
    left_gap_cells: int
    right_gap_cells: int
    reject_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "door_wall_attached": bool(self.attached),
            "door_wall_left_attached": bool(self.left_attached),
            "door_wall_right_attached": bool(self.right_attached),
            "door_wall_left_gap_cells": int(self.left_gap_cells),
            "door_wall_right_gap_cells": int(self.right_gap_cells),
            "door_wall_attachment_reject_reason": self.reject_reason,
        }


@dataclass
class VoxelDoorCompletionResult:
    door_seed_mask: np.ndarray
    door_extension_attempt_all_mask: np.ndarray
    door_extension_attempt_rejected_mask: np.ndarray
    door_centerline_visual_mask: np.ndarray
    door_visual_only_mask: np.ndarray
    door_geometry_warning_cut_mask: np.ndarray
    door_topology_effective_cut_mask: np.ndarray
    door_partition_cut_candidate_mask: np.ndarray
    door_cut_mask_for_partition: np.ndarray
    door_centerline_candidate_mask: np.ndarray
    rejected_door_centerline_mask: np.ndarray
    door_reject_reason_map: np.ndarray
    candidates: list[VoxelDoorLineCandidate]
    debug: dict[str, object]


@dataclass
class DoorAnchorWalkResult:
    anchor: tuple[int, int] | None
    path_cells: list[tuple[int, int]]
    status: str
    anchor_source: int
    hit_other_seed_cells: list[tuple[int, int]]
    hit_unknown_cells: list[tuple[int, int]]
    hit_real_wall_cells: list[tuple[int, int]]
    stopped_reason: str | None


@dataclass
class DoorMemoryObservationMaps:
    observed_xy: np.ndarray
    sensor_range_xy: np.ndarray
    vertical_free_xy: np.ndarray
    wall_xy: np.ndarray
    raw_seed_mask: np.ndarray
    current_verified_cut_mask: np.ndarray
    ceiling_height_m: float | None = None


@dataclass
class StableDoorTrack:
    track_id: int
    first_seen_step: int
    last_seen_step: int
    confidence: float
    center_rc: tuple[float, float]
    major_dir_rc: tuple[float, float]
    cut_cells: list[tuple[int, int]]
    visual_cells: list[tuple[int, int]]
    stable_seed_cells: list[tuple[int, int]] = field(default_factory=list)
    anchor_a_rc: tuple[int, int] | None = None
    anchor_b_rc: tuple[int, int] | None = None
    best_score: float = 0.0
    last_verified_step: int = -1
    last_weak_refresh_step: int = -1
    last_observed_step: int = -1
    not_observed_updates: int = 0
    geometry_locked: bool = True
    source_candidate_ids: list[int] = field(default_factory=list)
    update_count: int = 1
    missed_updates: int = 0
    contradiction_count: int = 0
    ceiling_height_m: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "first_seen_step": int(self.first_seen_step),
            "last_seen_step": int(self.last_seen_step),
            "confidence": float(self.confidence),
            "center_rc": [float(self.center_rc[0]), float(self.center_rc[1])],
            "major_dir_rc": [float(self.major_dir_rc[0]), float(self.major_dir_rc[1])],
            "cut_cells": [[int(r), int(c)] for r, c in self.cut_cells],
            "visual_cells": [[int(r), int(c)] for r, c in self.visual_cells],
            "stable_seed_cells": [[int(r), int(c)] for r, c in self.stable_seed_cells],
            "anchor_a_rc": None if self.anchor_a_rc is None else [int(self.anchor_a_rc[0]), int(self.anchor_a_rc[1])],
            "anchor_b_rc": None if self.anchor_b_rc is None else [int(self.anchor_b_rc[0]), int(self.anchor_b_rc[1])],
            "best_score": float(self.best_score),
            "last_verified_step": int(self.last_verified_step),
            "last_weak_refresh_step": int(self.last_weak_refresh_step),
            "last_observed_step": int(self.last_observed_step),
            "not_observed_updates": int(self.not_observed_updates),
            "geometry_locked": bool(self.geometry_locked),
            "source_candidate_ids": [int(v) for v in self.source_candidate_ids],
            "update_count": int(self.update_count),
            "missed_updates": int(self.missed_updates),
            "contradiction_count": int(self.contradiction_count),
            "ceiling_height_m": None if self.ceiling_height_m is None else float(self.ceiling_height_m),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "StableDoorTrack":
        def rc_float(value: object, default: tuple[float, float]) -> tuple[float, float]:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
                return (float(value[0]), float(value[1]))  # type: ignore[index]
            return default

        def rc_int_or_none(value: object) -> tuple[int, int] | None:
            if value is None:
                return None
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
                return (int(value[0]), int(value[1]))  # type: ignore[index]
            return None

        def cells(value: object) -> list[tuple[int, int]]:
            out: list[tuple[int, int]] = []
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
                        out.append((int(item[0]), int(item[1])))  # type: ignore[index]
            return out

        return cls(
            track_id=int(data.get("track_id", 0) or 0),
            first_seen_step=int(data.get("first_seen_step", -1) or -1),
            last_seen_step=int(data.get("last_seen_step", -1) or -1),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            center_rc=rc_float(data.get("center_rc"), (0.0, 0.0)),
            major_dir_rc=rc_float(data.get("major_dir_rc"), (0.0, 1.0)),
            cut_cells=cells(data.get("cut_cells", [])),
            visual_cells=cells(data.get("visual_cells", [])),
            stable_seed_cells=cells(data.get("stable_seed_cells", [])),
            anchor_a_rc=rc_int_or_none(data.get("anchor_a_rc")),
            anchor_b_rc=rc_int_or_none(data.get("anchor_b_rc")),
            best_score=float(data.get("best_score", 0.0) or 0.0),
            last_verified_step=int(data.get("last_verified_step", -1) or -1),
            last_weak_refresh_step=int(data.get("last_weak_refresh_step", -1) or -1),
            last_observed_step=int(data.get("last_observed_step", -1) or -1),
            not_observed_updates=int(data.get("not_observed_updates", 0) or 0),
            geometry_locked=bool(data.get("geometry_locked", True)),
            source_candidate_ids=[int(v) for v in (data.get("source_candidate_ids", []) or [])],  # type: ignore[union-attr]
            update_count=int(data.get("update_count", 1) or 1),
            missed_updates=int(data.get("missed_updates", 0) or 0),
            contradiction_count=int(data.get("contradiction_count", 0) or 0),
            ceiling_height_m=_finite_float_or_none(data.get("ceiling_height_m")),
        )


@dataclass
class StableDoorMemoryResult:
    stable_door_cut_mask: np.ndarray
    stable_door_visual_mask: np.ndarray
    current_matched_cut_mask: np.ndarray
    tracks: list[StableDoorTrack]
    debug: dict[str, object]


class VoxelDoorMemory:
    def __init__(self, config: VoxelDoorDetectorConfig | Mapping[str, object] | None = None):
        self.config = config if isinstance(config, VoxelDoorDetectorConfig) else VoxelDoorDetectorConfig.from_mapping(config)
        self._tracks: list[StableDoorTrack] = []
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def to_state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "next_track_id": int(self._next_track_id),
            "config": _jsonable(dict(getattr(self.config, "__dict__", {}) or {})),
            "tracks": [track.to_dict() for track in self._tracks],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object] | None,
        config: VoxelDoorDetectorConfig | Mapping[str, object] | None = None,
    ) -> "VoxelDoorMemory":
        mem = cls(config=config)
        raw = dict(state or {})
        mem._next_track_id = int(raw.get("next_track_id", 1) or 1)
        tracks = raw.get("tracks", []) or []
        mem._tracks = [
            StableDoorTrack.from_dict(item)
            for item in tracks
            if isinstance(item, Mapping)
        ]
        if mem._tracks:
            mem._next_track_id = max(int(mem._next_track_id), 1 + max(int(track.track_id) for track in mem._tracks))
        return mem

    def load_state_dict(self, state: Mapping[str, object] | None) -> None:
        restored = VoxelDoorMemory.from_state_dict(state, config=self.config)
        self._next_track_id = restored._next_track_id
        self._tracks = restored._tracks

    def update(
        self,
        candidates: Sequence[VoxelDoorLineCandidate],
        *,
        step: int,
        shape: tuple[int, int],
        observation: DoorMemoryObservationMaps | None = None,
    ) -> StableDoorMemoryResult:
        cfg = self.config
        stable_empty = np.zeros(shape, dtype=bool)
        if not bool(getattr(cfg, "door_memory_enabled", True)):
            return StableDoorMemoryResult(
                stable_door_cut_mask=stable_empty.copy(),
                stable_door_visual_mask=stable_empty.copy(),
                current_matched_cut_mask=stable_empty.copy(),
                tracks=[],
                debug={"voxel_door_memory_enabled": False, "voxel_door_memory_track_count": 0},
            )

        observation = self._validated_observation(observation, shape)
        observed_decay_band = np.zeros(shape, dtype=bool)
        unobserved_track_mask = np.zeros(shape, dtype=bool)
        contradiction_mask = np.zeros(shape, dtype=bool)
        tracks_unobserved = 0
        tracks_decayed = 0
        contradiction_updates = 0
        ceiling_changed_track_count = 0
        geometry_frozen_without_ceiling_change = 0
        observation_ceiling_height_m = self._observation_ceiling_height(observation)
        for track in self._tracks:
            track_mask = _cells_to_mask(track.cut_cells, shape) | _cells_to_mask(getattr(track, "stable_seed_cells", []), shape)
            if not np.any(track_mask):
                track_mask = _cells_to_mask(track.visual_cells, shape)
            if observation is not None:
                band = dilate(track_mask, max(0, int(getattr(cfg, "door_memory_observation_dilation_cells", 2))))
                observed = band & observation.sensor_range_xy
                if int(np.count_nonzero(observed)) < int(getattr(cfg, "door_memory_min_observed_cells_for_decay", 2)):
                    track.not_observed_updates = int(getattr(track, "not_observed_updates", 0)) + 1
                    tracks_unobserved += 1
                    unobserved_track_mask |= track_mask
                    continue
                observed_decay_band |= observed
                track.last_observed_step = int(step)
                ceiling_changed = self._track_ceiling_changed(track, observation)
                if ceiling_changed:
                    ceiling_changed_track_count += 1
                if ceiling_changed and self._track_contradicted(track, observation, shape):
                    track.contradiction_count = int(getattr(track, "contradiction_count", 0)) + 1
                    contradiction_updates += 1
                    contradiction_mask |= dilate(track_mask, max(0, int(getattr(cfg, "door_memory_contradiction_band_cells", 2)))) & observation.sensor_range_xy
                elif int(getattr(track, "contradiction_count", 0)) > 0:
                    track.contradiction_count = max(0, int(track.contradiction_count) - 1)
                track.not_observed_updates = 0
                if getattr(track, "ceiling_height_m", None) is None and observation_ceiling_height_m is not None:
                    track.ceiling_height_m = float(observation_ceiling_height_m)
            track.missed_updates = int(getattr(track, "missed_updates", 0)) + 1
            track.confidence = max(0.0, float(track.confidence) - float(getattr(cfg, "door_memory_decay_per_update", 0.10)))
            tracks_decayed += 1

        for track in self._tracks:
            if not hasattr(track, "stable_seed_cells") or track.stable_seed_cells is None:
                track.stable_seed_cells = []
            if int(getattr(track, "last_observed_step", -1)) < 0:
                track.last_observed_step = int(track.last_seen_step)
            if int(getattr(track, "last_verified_step", -1)) < 0:
                track.last_verified_step = int(track.last_seen_step)
            if float(getattr(track, "best_score", 0.0)) <= 0.0:
                track.best_score = float(len(track.cut_cells))

        verified_current = [
            candidate
            for candidate in candidates
            if bool(candidate.accepted)
            and bool(candidate.debug.get("partition_effective_verified", candidate.debug.get("partition_accepted", False)))
            and bool(candidate.door_cut_cells)
        ]
        refresh_evidence = [
            candidate
            for candidate in candidates
            if bool(candidate.accepted)
            or bool(candidate.debug.get("partition_geometry_accepted", False))
            or bool(candidate.debug.get("stable_memory_refresh_eligible", False))
        ]
        rejected_visual_only = int(
            sum(1 for candidate in candidates if bool(candidate.accepted) and not bool(candidate.debug.get("partition_accepted", False)))
        )
        rejected_partition_false = int(
            sum(1 for candidate in candidates if bool(candidate.accepted) and not bool(candidate.door_cut_cells))
        )
        matched_ids: set[int] = set()
        matched_current_mask = np.zeros(shape, dtype=bool)
        created = 0
        updated = 0
        verified_replaced_geometry = 0
        verified_kept_geometry = 0
        weak_refreshed = 0
        weak_refreshed_no_geometry_update = 0
        for candidate in verified_current:
            cut_cells = [tuple(v) for v in candidate.door_cut_cells]
            visual_cells = [tuple(v) for v in (candidate.extended_centerline_cells or candidate.door_cut_cells)]
            stable_seed_cells = self._candidate_seed_cells(candidate)
            cut_mask = _cells_to_mask(cut_cells, shape)
            matched_current_mask |= cut_mask
            best_track = self._best_matching_track(candidate, cut_mask, shape, matched_ids)
            candidate_quality = self._candidate_quality(candidate, cut_mask, shape, best_track)
            if best_track is None:
                self._tracks.append(
                    StableDoorTrack(
                        track_id=int(self._next_track_id),
                        first_seen_step=int(step),
                        last_seen_step=int(step),
                        confidence=float(getattr(cfg, "door_memory_initial_confidence", 1.0)),
                        center_rc=(float(candidate.center_rc[0]), float(candidate.center_rc[1])),
                        major_dir_rc=(float(candidate.major_dir_rc[0]), float(candidate.major_dir_rc[1])),
                        cut_cells=cut_cells,
                        visual_cells=visual_cells,
                        stable_seed_cells=stable_seed_cells,
                        anchor_a_rc=None if candidate.wall_anchor_a is None else (int(candidate.wall_anchor_a[0]), int(candidate.wall_anchor_a[1])),
                        anchor_b_rc=None if candidate.wall_anchor_b is None else (int(candidate.wall_anchor_b[0]), int(candidate.wall_anchor_b[1])),
                        best_score=float(candidate_quality),
                        last_verified_step=int(step),
                        last_weak_refresh_step=-1,
                        last_observed_step=int(step),
                        not_observed_updates=0,
                        geometry_locked=True,
                        source_candidate_ids=[int(candidate.candidate_id)],
                        update_count=1,
                        missed_updates=0,
                        contradiction_count=0,
                        ceiling_height_m=observation_ceiling_height_m,
                    )
                )
                matched_ids.add(int(self._next_track_id))
                self._next_track_id += 1
                created += 1
                continue
            best_track.last_seen_step = int(step)
            best_track.confidence = min(
                1.0,
                float(best_track.confidence) + float(getattr(cfg, "door_memory_confirm_increment", 0.35)),
            )
            ceiling_changed = self._track_ceiling_changed(best_track, observation)
            if bool(getattr(cfg, "door_memory_freeze_geometry_unless_ceiling_changes", True)) and not ceiling_changed:
                geometry_frozen_without_ceiling_change += 1
            replace_geometry = self._should_replace_verified_geometry(
                best_track,
                candidate,
                cut_mask,
                shape,
                float(candidate_quality),
                ceiling_changed=ceiling_changed,
            )
            if replace_geometry:
                best_track.center_rc = (float(candidate.center_rc[0]), float(candidate.center_rc[1]))
                best_track.major_dir_rc = (float(candidate.major_dir_rc[0]), float(candidate.major_dir_rc[1]))
                best_track.cut_cells = cut_cells
                best_track.visual_cells = visual_cells
                best_track.stable_seed_cells = stable_seed_cells
                best_track.anchor_a_rc = None if candidate.wall_anchor_a is None else (int(candidate.wall_anchor_a[0]), int(candidate.wall_anchor_a[1]))
                best_track.anchor_b_rc = None if candidate.wall_anchor_b is None else (int(candidate.wall_anchor_b[0]), int(candidate.wall_anchor_b[1]))
                best_track.best_score = max(float(getattr(best_track, "best_score", 0.0)), float(candidate_quality))
                best_track.ceiling_height_m = observation_ceiling_height_m
                verified_replaced_geometry += 1
            else:
                best_track.best_score = max(float(getattr(best_track, "best_score", 0.0)), float(candidate_quality))
                verified_kept_geometry += 1
                if ceiling_changed and observation_ceiling_height_m is not None:
                    best_track.ceiling_height_m = float(observation_ceiling_height_m)
            best_track.source_candidate_ids.append(int(candidate.candidate_id))
            best_track.source_candidate_ids = best_track.source_candidate_ids[-16:]
            best_track.update_count += 1
            best_track.missed_updates = 0
            best_track.contradiction_count = 0
            best_track.not_observed_updates = 0
            best_track.last_verified_step = int(step)
            best_track.last_observed_step = int(step)
            matched_ids.add(int(best_track.track_id))
            updated += 1

        for candidate in refresh_evidence:
            if bool(candidate.debug.get("partition_effective_verified", candidate.debug.get("partition_accepted", False))):
                continue
            evidence_cells = [tuple(v) for v in (candidate.door_cut_cells or candidate.extended_centerline_cells or candidate.seed_projected_centerline_cells)]
            evidence_mask = _cells_to_mask(evidence_cells, shape)
            if not np.any(evidence_mask):
                continue
            best_track = self._best_matching_track(candidate, evidence_mask, shape, matched_ids)
            if best_track is None:
                continue
            best_track.last_seen_step = int(step)
            best_track.confidence = min(
                1.0,
                float(best_track.confidence) + float(getattr(cfg, "door_memory_weak_refresh_increment", 0.12)),
            )
            best_track.source_candidate_ids.append(int(candidate.candidate_id))
            best_track.source_candidate_ids = best_track.source_candidate_ids[-16:]
            best_track.update_count += 1
            best_track.missed_updates = 0
            best_track.not_observed_updates = 0
            best_track.last_weak_refresh_step = int(step)
            best_track.last_seen_step = int(step)
            if bool(getattr(cfg, "door_memory_weak_refresh_updates_visual", False)) and self._weak_visual_can_replace(best_track, candidate, evidence_mask, shape):
                best_track.visual_cells = evidence_cells
            else:
                weak_refreshed_no_geometry_update += 1
            matched_ids.add(int(best_track.track_id))
            weak_refreshed += 1

        ttl = int(getattr(cfg, "door_memory_ttl_updates", 8))
        min_conf = float(getattr(cfg, "door_memory_min_confidence_to_keep", 0.25))
        contradictions_to_prune = int(getattr(cfg, "door_memory_contradictions_to_prune", 5))
        confidence_prune_enabled = observation is None
        before_prune = len(self._tracks)
        self._tracks = [
            track
            for track in self._tracks
            if not (
                (
                    confidence_prune_enabled
                    and float(track.confidence) < min_conf
                    and int(getattr(track, "missed_updates", 0)) > ttl
                )
                or int(getattr(track, "contradiction_count", 0)) >= contradictions_to_prune
            )
        ]
        stable_cut = np.zeros(shape, dtype=bool)
        stable_visual = np.zeros(shape, dtype=bool)
        for track in self._tracks:
            stable_cut |= _cells_to_mask(track.cut_cells, shape)
            stable_visual |= _cells_to_mask(track.visual_cells, shape)
        debug = {
            "voxel_door_memory_enabled": True,
            "voxel_door_memory_track_count": int(len(self._tracks)),
            "voxel_door_memory_current_candidate_count": int(len(verified_current)),
            "voxel_door_memory_verified_candidate_count": int(len(verified_current)),
            "voxel_door_memory_refresh_evidence_count": int(len(refresh_evidence)),
            "voxel_door_memory_rejected_visual_only_count": int(rejected_visual_only),
            "voxel_door_memory_rejected_partition_false_count": int(rejected_partition_false),
            "voxel_door_memory_created_count": int(created),
            "voxel_door_memory_verified_create_count": int(created),
            "voxel_door_memory_updated_count": int(updated),
            "voxel_door_memory_verified_update_count": int(updated),
            "voxel_door_memory_weak_refresh_count": int(weak_refreshed),
            "voxel_door_memory_observation_aware_decay": bool(observation is not None),
            "voxel_door_memory_tracks_unobserved_count": int(tracks_unobserved),
            "voxel_door_memory_tracks_decayed_count": int(tracks_decayed),
            "voxel_door_memory_tracks_weak_refreshed_no_geometry_update": int(weak_refreshed_no_geometry_update),
            "voxel_door_memory_verified_update_kept_old_geometry": int(verified_kept_geometry),
            "voxel_door_memory_verified_update_replaced_geometry": int(verified_replaced_geometry),
            "voxel_door_memory_geometry_frozen_without_ceiling_change": int(geometry_frozen_without_ceiling_change),
            "voxel_door_memory_freeze_geometry_unless_ceiling_changes": bool(getattr(cfg, "door_memory_freeze_geometry_unless_ceiling_changes", True)),
            "voxel_door_memory_ceiling_height_m": observation_ceiling_height_m,
            "voxel_door_memory_ceiling_changed_track_count": int(ceiling_changed_track_count),
            "voxel_door_memory_contradiction_count_total": int(sum(int(getattr(track, "contradiction_count", 0)) for track in self._tracks)),
            "voxel_door_memory_contradiction_updates": int(contradiction_updates),
            "voxel_door_memory_confidence_prune_enabled": bool(confidence_prune_enabled),
            "voxel_door_memory_observed_decay_band_mask": observed_decay_band.astype(bool),
            "voxel_door_memory_unobserved_track_mask": unobserved_track_mask.astype(bool),
            "voxel_door_memory_contradiction_mask": contradiction_mask.astype(bool),
            "voxel_door_memory_pruned_count": int(before_prune - len(self._tracks)),
            "voxel_door_memory_stable_cut_cells": int(np.count_nonzero(stable_cut)),
            "voxel_door_memory_stable_visual_cells": int(np.count_nonzero(stable_visual)),
            "voxel_door_memory_tracks": [track.to_dict() for track in self._tracks],
        }
        return StableDoorMemoryResult(
            stable_door_cut_mask=stable_cut.astype(bool),
            stable_door_visual_mask=stable_visual.astype(bool),
            current_matched_cut_mask=matched_current_mask.astype(bool),
            tracks=list(self._tracks),
            debug=debug,
        )

    def _best_matching_track(
        self,
        candidate: VoxelDoorLineCandidate,
        cut_mask: np.ndarray,
        shape: tuple[int, int],
        matched_ids: set[int],
    ) -> StableDoorTrack | None:
        cfg = self.config
        best: tuple[float, StableDoorTrack] | None = None
        cand_mask = np.asarray(cut_mask, dtype=bool)
        cand_d = dilate(cand_mask, max(0, int(getattr(cfg, "door_memory_match_dilation_cells", 2))))
        cand_count = int(np.count_nonzero(cand_d))
        candidate_center = np.asarray(candidate.center_rc, dtype=np.float32)
        candidate_major = self._normalized_vector(candidate.major_dir_rc)
        candidate_anchors = [
            None if candidate.wall_anchor_a is None else (int(candidate.wall_anchor_a[0]), int(candidate.wall_anchor_a[1])),
            None if candidate.wall_anchor_b is None else (int(candidate.wall_anchor_b[0]), int(candidate.wall_anchor_b[1])),
        ]
        for track in self._tracks:
            if int(track.track_id) in matched_ids:
                continue
            track_mask = _cells_to_mask(track.cut_cells, shape)
            track_d = dilate(track_mask, max(0, int(getattr(cfg, "door_memory_match_dilation_cells", 2))))
            union = int(np.count_nonzero(track_d | cand_d))
            inter = int(np.count_nonzero(track_d & cand_d))
            dilated_iou = 0.0 if union <= 0 else float(inter) / float(union)
            track_center = np.asarray(track.center_rc, dtype=np.float32)
            dist = float(np.linalg.norm(candidate_center - track_center))
            track_major = self._normalized_vector(track.major_dir_rc)
            angle = float(np.degrees(np.arccos(min(1.0, max(-1.0, abs(float(np.dot(candidate_major, track_major))))))))
            stable_seed_mask = _cells_to_mask(getattr(track, "stable_seed_cells", []), shape)
            seed_overlap = 0.0
            if cand_count > 0:
                seed_overlap = float(np.count_nonzero(cand_d & stable_seed_mask)) / float(cand_count)
            anchor_close = self._anchors_close(
                candidate_anchors,
                [getattr(track, "anchor_a_rc", None), getattr(track, "anchor_b_rc", None)],
                max_distance_cells=float(getattr(cfg, "door_memory_anchor_match_distance_cells", 4)),
            )
            center_angle_match = dist <= float(getattr(cfg, "door_memory_match_distance_cells", 3)) and angle <= float(getattr(cfg, "door_memory_match_angle_deg", 20.0))
            if not (
                dilated_iou >= float(getattr(cfg, "door_memory_dilated_iou_min", getattr(cfg, "door_memory_match_iou_min", 0.20)))
                or center_angle_match
                or seed_overlap >= float(getattr(cfg, "door_memory_seed_overlap_min", 0.20))
                or anchor_close
            ):
                continue
            score = float(
                3.0 * dilated_iou
                + 1.5 * seed_overlap
                + (0.5 if center_angle_match else 0.0)
                + (0.5 if anchor_close else 0.0)
                + 0.05 * max(0.0, float(getattr(cfg, "door_memory_match_distance_cells", 3)) - dist)
            )
            if best is None or score > best[0]:
                best = (score, track)
        return None if best is None else best[1]

    def _validated_observation(
        self,
        observation: DoorMemoryObservationMaps | None,
        shape: tuple[int, int],
    ) -> DoorMemoryObservationMaps | None:
        if observation is None:
            return None

        def as_bool(name: str) -> np.ndarray:
            arr = np.asarray(getattr(observation, name), dtype=bool)
            if arr.shape != tuple(shape):
                raise ValueError("door memory observation %s must match roomseg shape" % name)
            return arr

        return DoorMemoryObservationMaps(
            observed_xy=as_bool("observed_xy"),
            sensor_range_xy=as_bool("sensor_range_xy"),
            vertical_free_xy=as_bool("vertical_free_xy"),
            wall_xy=as_bool("wall_xy"),
            raw_seed_mask=as_bool("raw_seed_mask"),
            current_verified_cut_mask=as_bool("current_verified_cut_mask"),
            ceiling_height_m=self._observation_ceiling_height(observation),
        )

    def _observation_ceiling_height(self, observation: DoorMemoryObservationMaps | None) -> float | None:
        if observation is None:
            return None
        return _finite_float_or_none(getattr(observation, "ceiling_height_m", None))

    def _track_ceiling_changed(
        self,
        track: StableDoorTrack,
        observation: DoorMemoryObservationMaps | None,
    ) -> bool:
        if not bool(getattr(self.config, "door_memory_freeze_geometry_unless_ceiling_changes", True)):
            return True
        observed_ceiling = self._observation_ceiling_height(observation)
        track_ceiling = _finite_float_or_none(getattr(track, "ceiling_height_m", None))
        if observed_ceiling is None or track_ceiling is None:
            return False
        threshold = max(0.0, float(getattr(self.config, "door_memory_ceiling_change_threshold_m", 0.05)))
        return abs(float(observed_ceiling) - float(track_ceiling)) > threshold

    def _candidate_seed_cells(self, candidate: VoxelDoorLineCandidate) -> list[tuple[int, int]]:
        cells = candidate.seed_cells or candidate.seed_projected_centerline_cells
        return sorted({(int(r), int(c)) for r, c in cells})

    def _candidate_quality(
        self,
        candidate: VoxelDoorLineCandidate,
        cut_mask: np.ndarray,
        shape: tuple[int, int],
        track: StableDoorTrack | None,
    ) -> float:
        debug = candidate.debug
        score = 0.0
        score += 2.0 if bool(debug.get("partition_effective_verified", debug.get("partition_accepted", False))) else 0.0
        score += 1.0 if candidate.wall_anchor_a is not None and candidate.wall_anchor_b is not None else 0.0
        cut_len = int(np.count_nonzero(cut_mask))
        score += 0.3 * min(1.0, float(cut_len) / 12.0)
        if track is not None:
            seed_mask = _cells_to_mask(getattr(track, "stable_seed_cells", []), shape)
            cand_d = dilate(np.asarray(cut_mask, dtype=bool), max(0, int(getattr(self.config, "door_memory_match_dilation_cells", 2))))
            denom = max(1, int(np.count_nonzero(cand_d)))
            score += 0.5 * float(np.count_nonzero(cand_d & seed_mask)) / float(denom)
        unknown_ratio = float(debug.get("partition_inner_unknown_ratio", debug.get("inner_unknown_ratio", 0.0)) or 0.0)
        wall_ratio = float(debug.get("partition_inner_wall_ratio", debug.get("inner_wall_ratio", 0.0)) or 0.0)
        score -= 0.5 * max(0.0, min(1.0, unknown_ratio))
        score -= 0.5 * max(0.0, min(1.0, wall_ratio))
        return float(score)

    def _should_replace_verified_geometry(
        self,
        track: StableDoorTrack,
        candidate: VoxelDoorLineCandidate,
        cut_mask: np.ndarray,
        shape: tuple[int, int],
        candidate_quality: float,
        *,
        ceiling_changed: bool = False,
    ) -> bool:
        cfg = self.config
        if not bool(getattr(cfg, "door_memory_allow_verified_geometry_replace", True)):
            return False
        if bool(getattr(cfg, "door_memory_freeze_geometry_unless_ceiling_changes", True)) and not bool(ceiling_changed):
            return False
        old_mask = _cells_to_mask(track.cut_cells, shape)
        old_len = max(1, int(np.count_nonzero(old_mask)))
        new_len = int(np.count_nonzero(cut_mask))
        if bool(getattr(cfg, "door_memory_prevent_shrinking_stable_cut", True)):
            min_ratio = float(getattr(cfg, "door_memory_min_length_ratio_to_replace", 0.80))
            if float(new_len) < float(old_len) * min_ratio:
                return False
        old_quality = float(getattr(track, "best_score", 0.0))
        if float(candidate_quality) >= old_quality + float(getattr(cfg, "door_memory_replace_quality_margin", 0.20)):
            return True
        dilated_iou = self._dilated_iou(old_mask, np.asarray(cut_mask, dtype=bool), int(getattr(cfg, "door_memory_match_dilation_cells", 2)))
        if (
            dilated_iou >= float(getattr(cfg, "door_memory_dilated_iou_min", 0.10))
            and new_len >= old_len
            and float(candidate_quality) >= old_quality - 1e-6
        ):
            return True
        return False

    def _weak_visual_can_replace(
        self,
        track: StableDoorTrack,
        candidate: VoxelDoorLineCandidate,
        evidence_mask: np.ndarray,
        shape: tuple[int, int],
    ) -> bool:
        stable_cut = _cells_to_mask(track.cut_cells, shape)
        stable_visual_len = max(1, len(track.visual_cells))
        new_len = int(np.count_nonzero(evidence_mask))
        if float(new_len) < 0.80 * float(stable_visual_len):
            return False
        overlap = self._dilated_iou(stable_cut, np.asarray(evidence_mask, dtype=bool), int(getattr(self.config, "door_memory_match_dilation_cells", 2)))
        if overlap < float(getattr(self.config, "door_memory_dilated_iou_min", 0.10)):
            return False
        cand_major = self._normalized_vector(candidate.major_dir_rc)
        track_major = self._normalized_vector(track.major_dir_rc)
        angle = float(np.degrees(np.arccos(min(1.0, max(-1.0, abs(float(np.dot(cand_major, track_major))))))))
        return angle <= 10.0

    def _track_contradicted(
        self,
        track: StableDoorTrack,
        observation: DoorMemoryObservationMaps,
        shape: tuple[int, int],
    ) -> bool:
        cfg = self.config
        cut_mask = _cells_to_mask(track.cut_cells, shape)
        seed_mask = _cells_to_mask(getattr(track, "stable_seed_cells", []), shape)
        band = dilate(cut_mask | seed_mask, max(0, int(getattr(cfg, "door_memory_contradiction_band_cells", 2))))
        observed = band & observation.sensor_range_xy
        observed_count = int(np.count_nonzero(observed))
        if observed_count < int(getattr(cfg, "door_memory_min_observed_cells_for_contradiction", 4)):
            return False
        wall_ratio = float(np.count_nonzero(observed & observation.wall_xy)) / float(max(1, observed_count))
        seed_absent = int(np.count_nonzero(observed & observation.raw_seed_mask)) == 0
        verified_absent = int(np.count_nonzero(observed & observation.current_verified_cut_mask)) == 0
        return (
            wall_ratio >= float(getattr(cfg, "door_memory_contradict_wall_ratio", 0.75))
            and seed_absent
            and verified_absent
        )

    def _dilated_iou(self, a: np.ndarray, b: np.ndarray, dilation_cells: int) -> float:
        aa = dilate(np.asarray(a, dtype=bool), max(0, int(dilation_cells)))
        bb = dilate(np.asarray(b, dtype=bool), max(0, int(dilation_cells)))
        union = int(np.count_nonzero(aa | bb))
        if union <= 0:
            return 0.0
        return float(np.count_nonzero(aa & bb)) / float(union)

    def _normalized_vector(self, vec: Sequence[float]) -> np.ndarray:
        out = np.asarray(vec, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(out))
        if norm <= 1e-6:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return out / norm

    def _anchors_close(
        self,
        candidate_anchors: Sequence[tuple[int, int] | None],
        track_anchors: Sequence[tuple[int, int] | None],
        *,
        max_distance_cells: float,
    ) -> bool:
        for ca in candidate_anchors:
            if ca is None:
                continue
            cvec = np.asarray(ca, dtype=np.float32)
            for ta in track_anchors:
                if ta is None:
                    continue
                if float(np.linalg.norm(cvec - np.asarray(ta, dtype=np.float32))) <= float(max_distance_cells):
                    return True
        return False


def _finite_float_or_none(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _door_scan_z_bounds(voxel_grid: VoxelOccupancyGrid3D, cfg: VoxelDoorDetectorConfig) -> tuple[float, float, str]:
    z_min = float(getattr(cfg, "z_scan_min_m", 0.10))
    active = _finite_float_or_none(getattr(voxel_grid, "active_z_max_m", None))
    if active is None:
        grid_cfg = getattr(voxel_grid, "config", None)
        cap = _finite_float_or_none(getattr(grid_cfg, "active_z_max_cap_m", None))
        active = cap if cap is not None and cap > 0.0 else float(getattr(voxel_grid, "z_max_m", getattr(cfg, "z_scan_max_m", 2.40)))
        source = "active_z_cap_without_ceiling"
    else:
        source = "active_z_max"
    ceiling = _finite_float_or_none(getattr(voxel_grid, "ceiling_height_m", None))
    ceiling_ratio = _finite_float_or_none(getattr(cfg, "ceiling_scan_ratio", None))
    ceiling_ratio = 0.95 if ceiling_ratio is None or ceiling_ratio <= 0.0 else float(ceiling_ratio)
    ceiling_scan_top = None if ceiling is None else float(ceiling_ratio) * float(ceiling)
    configured = float(getattr(cfg, "z_scan_max_m", active))
    seed_configured = _finite_float_or_none(getattr(cfg, "door_seed_scan_max_m", None))
    seed_configured = float(configured) if seed_configured is None else float(seed_configured)
    seed_fallback = _finite_float_or_none(getattr(cfg, "door_seed_scan_fallback_m", None))
    seed_fallback = float(seed_configured) if seed_fallback is None else float(seed_fallback)
    top_mode = str(getattr(cfg, "z_scan_top_mode", "min_active_or_ceiling_or_config") or "").strip().lower()
    legacy_mode = str(getattr(cfg, "z_scan_max_mode", "active_z_max") or "").strip().lower()
    if top_mode in {"ceiling_ratio_or_fallback", "ceiling_or_fallback", "ceiling_ratio_fallback"}:
        if ceiling_scan_top is not None:
            top = float(ceiling_scan_top)
            source = "ceiling_%.2f" % float(ceiling_ratio)
        else:
            top = float(seed_fallback)
            source = "door_seed_scan_fallback"
    elif top_mode in {"min_seed_or_ceiling_ratio", "min_seed_or_ceiling", "min_config_or_ceiling_ratio"}:
        top = float(seed_configured)
        source_parts = ["door_seed_scan_max"]
        if ceiling_scan_top is not None:
            top = min(float(top), float(ceiling_scan_top))
            source_parts.append("ceiling_%.2f" % float(ceiling_ratio))
        source = "min(" + ",".join(source_parts) + ")"
    elif top_mode == "min_active_or_ceiling_or_config":
        top = min(float(active), float(configured))
        source_parts = [source, "config"]
        if ceiling_scan_top is not None:
            top = min(float(top), float(ceiling_scan_top))
            source_parts.append("ceiling_%.2f" % float(ceiling_ratio))
        source = "min(" + ",".join(source_parts) + ")"
    elif top_mode == "config" or legacy_mode == "config":
        top = float(configured)
        source = "config"
    else:
        top = float(active)
        if ceiling_scan_top is not None and top_mode in {"min_active_or_ceiling", "ceiling"}:
            top = min(float(top), float(ceiling_scan_top))
            source = "min(active_z_max,ceiling_%.2f)" % float(ceiling_ratio)
    if top_mode not in {
        "ceiling_ratio_or_fallback",
        "ceiling_or_fallback",
        "ceiling_ratio_fallback",
        "min_seed_or_ceiling_ratio",
        "min_seed_or_ceiling",
        "min_config_or_ceiling_ratio",
    }:
        seed_scan_cap = _finite_float_or_none(getattr(cfg, "door_seed_scan_max_m", None))
        if seed_scan_cap is not None:
            top = min(float(top), float(seed_scan_cap))
            source = "min(%s,door_seed_scan_max)" % str(source)
    if not np.isfinite(top):
        top = float(getattr(voxel_grid, "z_max_m", getattr(cfg, "z_scan_max_m", 2.80)))
        source = "storage_z_max_nonfinite"
    top = max(float(z_min), float(top))
    return float(z_min), float(top), str(source)


def classify_voxel_door_seeds(
    *,
    voxel_grid: VoxelOccupancyGrid3D,
    config: VoxelDoorDetectorConfig | Mapping[str, object] | None = None,
    sensor_range_count: np.ndarray | None = None,
    navigation_free_mask: np.ndarray | None = None,
    no_clearance_navigation_free_mask: np.ndarray | None = None,
) -> VoxelDoorSeedResult:
    cfg = config if isinstance(config, VoxelDoorDetectorConfig) else VoxelDoorDetectorConfig.from_mapping(config)
    shape = tuple(voxel_grid.shape)
    if not bool(cfg.enabled):
        zero = np.zeros(shape, dtype=bool)
        labels = np.zeros(shape, dtype=np.int32)
        return VoxelDoorSeedResult(
            door_seed_mask=zero.copy(),
            door_seed_component_map=labels,
            door_seed_reject_reason_map=np.zeros(shape, dtype=np.uint8),
            lower_free_cells_xy=np.zeros(shape, dtype=np.uint16),
            top_occupied_cells_xy=np.zeros(shape, dtype=np.uint16),
            first_occupied_z_xy=np.full(shape, np.nan, dtype=np.float32),
            unknown_tail_cells_xy=np.zeros(shape, dtype=np.uint16),
            seed_evidence=[],
            debug=_empty_result(shape, enabled=False).debug,
        )

    door_scan_z_min_m, door_scan_z_max_m, door_scan_top_source = _door_scan_z_bounds(voxel_grid, cfg)
    active_idx = voxel_grid.active_z_indices(
        z_min_m=float(door_scan_z_min_m),
        z_max_m=float(door_scan_z_max_m),
    )
    z_centers = voxel_grid.z_centers_m
    used_z = np.asarray(z_centers, dtype=np.float32)[np.asarray(active_idx, dtype=np.int32)] if np.asarray(active_idx).size else np.asarray([], dtype=np.float32)
    z_used_max_m = float(np.max(used_z)) if used_z.size else float("nan")
    if used_z.size and z_used_max_m > float(door_scan_z_max_m) + 1e-5:
        raise AssertionError(
            "voxel_door_seed_active_z_exceeds_scan_max used=%.4f scan_max=%.4f"
            % (float(z_used_max_m), float(door_scan_z_max_m))
        )
    z_state = np.asarray(voxel_grid.state, dtype=np.uint8)
    z_sensor_range_count = sensor_range_count
    if z_sensor_range_count is None:
        z_sensor_range_count = getattr(voxel_grid, "sensor_range_count", None)
    if z_sensor_range_count is not None:
        z_sensor_range_count = np.asarray(z_sensor_range_count)
        if z_sensor_range_count.shape != z_state.shape:
            z_sensor_range_count = None
    seed_started_at = time.perf_counter()
    if bool(cfg.vectorized_seed_classification):
        (
            seed,
            reason_map,
            lower_free_xy,
            top_occ_xy,
            unknown_tail_xy,
            first_occ_z_xy,
            rejected_seed_reasons,
            accepted_seed_evidence,
            seed_debug_maps,
        ) = classify_voxel_door_seeds_vectorized(
            z_state,
            z_centers,
            active_idx,
            cfg,
            shape=shape,
            return_debug=True,
            sensor_range_count=z_sensor_range_count,
        )
    else:
        seed = np.zeros(shape, dtype=bool)
        reason_map = np.zeros(shape, dtype=np.uint8)
        lower_free_xy = np.zeros(shape, dtype=np.uint16)
        top_occ_xy = np.zeros(shape, dtype=np.uint16)
        unknown_tail_xy = np.zeros(shape, dtype=np.uint16)
        first_occ_z_xy = np.full(shape, np.nan, dtype=np.float32)
        seed_debug_maps = _empty_seed_debug_maps(shape)
        accepted_seed_evidence = []
        rejected_seed_reasons = Counter()
        for r in range(shape[0]):
            for c in range(shape[1]):
                sensor_col = None if z_sensor_range_count is None else z_sensor_range_count[:, r, c]
                ev = classify_voxel_door_seed_column(z_state[:, r, c], z_centers, active_idx, cfg, row=r, col=c, sensor_range_count_col=sensor_col)
                lower_free_xy[r, c] = int(ev.lower_free_cells)
                top_occ_xy[r, c] = int(ev.top_occupied_cells)
                unknown_tail_xy[r, c] = int(ev.unknown_tail_cells)
                if ev.first_occupied_z_m is not None:
                    first_occ_z_xy[r, c] = float(ev.first_occupied_z_m)
                if ev.accepted:
                    seed[r, c] = True
                    reason_map[r, c] = 1
                    accepted_seed_evidence.append(ev)
                else:
                    rejected_seed_reasons[str(ev.reject_reason)] += 1
                    reason_map[r, c] = _seed_reject_code(str(ev.reject_reason))
    seed_filter_started_at = time.perf_counter()
    seed_free_mask = no_clearance_navigation_free_mask if no_clearance_navigation_free_mask is not None else navigation_free_mask
    seed_free_source = (
        "no_clearance_navigation_free_mask"
        if no_clearance_navigation_free_mask is not None
        else ("navigation_free_mask" if navigation_free_mask is not None else "none")
    )
    seed, reason_map, navigation_free_filter_debug = _filter_seed_by_navigation_free(seed, reason_map, seed_free_mask, cfg)
    navigation_free_filter_debug["voxel_door_seed_navigation_free_source"] = seed_free_source
    navigation_free_filter_debug["voxel_door_seed_uses_no_clearance_navigation_free"] = bool(
        no_clearance_navigation_free_mask is not None
    )
    if bool(navigation_free_filter_debug.get("voxel_door_seed_navigation_free_filter_applied", False)):
        removed_mask = np.asarray(navigation_free_filter_debug["voxel_door_seed_navigation_free_reject_map"], dtype=bool)
        accepted_seed_evidence = [item for item in accepted_seed_evidence if not bool(removed_mask[int(item.row), int(item.col)])]
        rejected_seed_reasons["not_navigation_free"] += int(
            navigation_free_filter_debug.get("voxel_door_seed_navigation_free_rejected_cells", 0) or 0
        )
    seed, reason_map, labels, component_bbox_filter_debug = _filter_seed_components_by_bbox_extent(seed, reason_map, cfg)
    if bool(component_bbox_filter_debug.get("voxel_door_seed_component_bbox_filter_applied", False)):
        removed_mask = np.asarray(component_bbox_filter_debug["voxel_door_seed_component_bbox_reject_map"], dtype=bool)
        accepted_seed_evidence = [item for item in accepted_seed_evidence if not bool(removed_mask[int(item.row), int(item.col)])]
        rejected_seed_reasons["seed_component_bbox_too_small"] += int(
            component_bbox_filter_debug.get("voxel_door_seed_component_bbox_rejected_cells", 0) or 0
        )
    seed_ms = float((time.perf_counter() - seed_started_at) * 1000.0)
    debug = {
        "voxel_door_enabled": bool(cfg.enabled),
        "voxel_door_seed_only": True,
        "voxel_door_seed_ceiling_filter_version": "v43_false_door_seed_ceiling_bug_fix",
        "voxel_door_seed_method": str(cfg.seed_method),
        "voxel_door_turn_z_estimate_mode": "free_centroid_plus_scaled_free_extent",
        "voxel_door_z_scan_min_m": float(door_scan_z_min_m),
        "voxel_door_z_scan_max_m": float(door_scan_z_max_m),
        "voxel_door_z_scan_top_source": str(door_scan_top_source),
        "voxel_door_z_used_max_m": float(z_used_max_m),
        "voxel_door_z_used_bin_count": int(used_z.size),
        "voxel_door_centroid_turn_extent_scale": float(cfg.centroid_turn_extent_scale),
        "voxel_door_vectorized_seed_classification": bool(cfg.vectorized_seed_classification),
        "voxel_door_sensor_aware_seed_classification": bool(z_sensor_range_count is not None),
        "voxel_door_seed_ms": float(seed_ms),
        "voxel_door_seed_component_bbox_filter_ms": float((time.perf_counter() - seed_filter_started_at) * 1000.0),
        "voxel_door_seed_mask": seed.astype(bool),
        "voxel_door_seed_component_map": labels.astype(np.int32),
        "voxel_door_seed_reject_reason_map": reason_map.astype(np.uint8),
        "voxel_door_seed_reject_reason_counts": dict(rejected_seed_reasons),
        "voxel_door_seed_lower_free_cells_xy": lower_free_xy.astype(np.uint16),
        "voxel_door_seed_top_occupied_cells_xy": top_occ_xy.astype(np.uint16),
        "voxel_door_seed_first_occupied_z_xy": first_occ_z_xy.astype(np.float32),
        "voxel_door_seed_unknown_tail_cells_xy": unknown_tail_xy.astype(np.uint16),
        "voxel_door_seed_cells": int(np.count_nonzero(seed)),
        "voxel_door_seed_evidence": [item.to_dict() for item in accepted_seed_evidence[:2048]],
        **navigation_free_filter_debug,
        **component_bbox_filter_debug,
        **seed_debug_maps,
    }
    debug.update(_seed_reject_reason_maps(reason_map))
    return VoxelDoorSeedResult(
        door_seed_mask=seed.astype(bool),
        door_seed_component_map=labels.astype(np.int32),
        door_seed_reject_reason_map=reason_map.astype(np.uint8),
        lower_free_cells_xy=lower_free_xy.astype(np.uint16),
        top_occupied_cells_xy=top_occ_xy.astype(np.uint16),
        first_occupied_z_xy=first_occ_z_xy.astype(np.float32),
        unknown_tail_cells_xy=unknown_tail_xy.astype(np.uint16),
        seed_evidence=accepted_seed_evidence,
        debug=debug,
    )


def complete_voxel_doors_from_seeds(
    *,
    seed_result: VoxelDoorSeedResult,
    free_map: np.ndarray,
    anchor_wall_map: np.ndarray,
    unknown_map: np.ndarray,
    resolution_m: float,
    config: VoxelDoorDetectorConfig | Mapping[str, object] | None = None,
    anchor_source_map: np.ndarray | None = None,
    free_map_for_visual_validation: np.ndarray | None = None,
    base_partition_free: np.ndarray | None = None,
    door_conflict_free_mask: np.ndarray | None = None,
    door_extension_no_clearance_occupied_mask: np.ndarray | None = None,
    real_wall_barrier_map: np.ndarray | None = None,
    seed_cluster_barrier_map: np.ndarray | None = None,
) -> VoxelDoorCompletionResult:
    cfg = config if isinstance(config, VoxelDoorDetectorConfig) else VoxelDoorDetectorConfig.from_mapping(config)
    seed = np.asarray(seed_result.door_seed_mask, dtype=bool)
    labels = np.asarray(seed_result.door_seed_component_map, dtype=np.int32)
    free = np.asarray(free_map_for_visual_validation if free_map_for_visual_validation is not None else free_map, dtype=bool)
    partition_free = np.asarray(base_partition_free if base_partition_free is not None else free_map, dtype=bool)
    conflict_free = np.asarray(door_conflict_free_mask if door_conflict_free_mask is not None else partition_free, dtype=bool)
    extension_occupied = (
        np.asarray(door_extension_no_clearance_occupied_mask, dtype=bool)
        if door_extension_no_clearance_occupied_mask is not None
        else np.zeros_like(conflict_free, dtype=bool)
    )
    wall = np.asarray(anchor_wall_map, dtype=bool)
    real_wall = np.asarray(real_wall_barrier_map if real_wall_barrier_map is not None else anchor_wall_map, dtype=bool)
    unknown = np.asarray(unknown_map, dtype=bool)
    shape = free.shape
    if (
        wall.shape != shape
        or unknown.shape != shape
        or seed.shape != shape
        or partition_free.shape != shape
        or conflict_free.shape != shape
        or extension_occupied.shape != shape
        or real_wall.shape != shape
    ):
        raise ValueError("voxel door completion maps must share one HxW shape")
    if anchor_source_map is None:
        source_map = np.where(wall, DOOR_ANCHOR_STRICT_RAW, DOOR_ANCHOR_NONE).astype(np.uint8)
    else:
        source_map = np.asarray(anchor_source_map, dtype=np.uint8)
        if source_map.shape != shape:
            raise ValueError("anchor_source_map must match anchor_wall_map shape")

    started_at = time.perf_counter()
    cluster_barrier = np.asarray(seed_cluster_barrier_map if seed_cluster_barrier_map is not None else real_wall, dtype=bool)
    clusters, cluster_map, cluster_debug = _build_door_seed_clusters(seed, labels, resolution_m=float(resolution_m), cfg=cfg, barrier_map=cluster_barrier)
    seed_groups, seed_group_debug = _build_door_seed_groups_v17(
        clusters,
        shape=shape,
        resolution_m=float(resolution_m),
        cfg=cfg,
        barrier_map=cluster_barrier,
    )
    selected_candidates: list[VoxelDoorLineCandidate] = []
    all_trial_candidates: list[VoxelDoorLineCandidate] = []
    trial_groups: list[DoorTrialCandidateGroup] = []
    all_primitives: list[DoorSeedLinePrimitive] = []
    primitives_by_group: dict[int, list[DoorSeedLinePrimitive]] = {}
    primitive_id = 1
    for group in seed_groups:
        primitives = extract_seed_line_primitives_from_group(
            group,
            primitive_id_start=int(primitive_id),
            shape=shape,
            resolution_m=float(resolution_m),
            cfg=cfg,
        )
        primitives_by_group[int(group.group_id)] = primitives
        all_primitives.extend(primitives)
        primitive_id += int(len(primitives))
    cid = 1
    for group in seed_groups:
        group_primitives = primitives_by_group.get(int(group.group_id), [])
        accepted_primitives = [primitive for primitive in group_primitives if bool(primitive.accepted_for_extension)]
        _extensible_ok, extensible_reason, extensible_debug = is_seed_group_extensible(group, cfg, resolution_m=float(resolution_m))
        if not accepted_primitives:
            primitive_reason_counts = Counter(str(primitive.reject_reason or "accepted") for primitive in group_primitives if not bool(primitive.accepted_for_extension))
            selected_reason = str(
                group.reject_reason
                or extensible_reason
                or next((primitive.reject_reason for primitive in group_primitives if primitive.reject_reason), None)
                or "no_seed_line_primitive"
            )
            candidate = _rejected_candidate(cid, int(group.group_id), list(group.seed_cells), selected_reason)
            candidate.debug.update(
                {
                    "cluster_id": int(group.group_id),
                    "seed_group_id": int(group.group_id),
                    "seed_group_kind": str(group.group_kind),
                    "source_cluster_ids": [int(v) for v in group.source_cluster_ids],
                    "component_ids": [int(v) for v in group.component_ids],
                    "completion_mode": DOOR_COMPLETION_REJECTED,
                    "visual_accepted": False,
                    "partition_accepted": False,
                    "partition_topology_accepted": False,
                    "reject_reason_visual": selected_reason,
                    "reject_reason_partition": "visual_rejected",
                    "reject_reason_topology": "visual_rejected",
                    "primitive_extraction_attempted_before_reject": True,
                    "primitive_count_for_group": int(len(group_primitives)),
                    "accepted_primitive_count_for_group": 0,
                    "primitive_reject_reason_counts": dict(primitive_reason_counts),
                    "voxel_v32_seed_line_primitive_completion": True,
                    **extensible_debug,
                }
            )
            selected_candidates.append(candidate)
            all_trial_candidates.append(candidate)
            trial_groups.append(
                DoorTrialCandidateGroup(
                    cluster_id=int(group.group_id),
                    component_ids=[int(v) for v in group.component_ids],
                    trials=[candidate],
                    selected_candidate_id=int(candidate.candidate_id),
                    selected_reason=selected_reason,
                    debug={
                        "trial_count": 1,
                        "selected_score": 0.0,
                        "seed_group_id": int(group.group_id),
                        "seed_group_kind": str(group.group_kind),
                        "source_cluster_ids": [int(v) for v in group.source_cluster_ids],
                        "primitive_count_for_group": int(len(group_primitives)),
                        "accepted_primitive_count_for_group": 0,
                        "primitive_reject_reason_counts": dict(primitive_reason_counts),
                        **extensible_debug,
                    },
                )
            )
            cid += 1
            continue

        trial_candidates = []
        for primitive in accepted_primitives:
            primitive_mask = _cells_to_mask(primitive.cells, shape)
            mode_hint = DOOR_COMPLETION_SEED_PAIR_BRIDGE if str(primitive.extraction_method) == "seed_pair_bridge" else None
            orientation_source = (
                "seed_major"
                if str(primitive.extraction_method) in {"direct_pca", "fallback_centerline"}
                else "seed_primitive_%s" % str(primitive.extraction_method)
            )
            candidate = _candidate_from_seed_component(
                candidate_id=cid,
                component_id=int(primitive.primitive_id),
                seed_cells=list(primitive.cells),
                seed_component_mask=primitive_mask,
                all_seed_mask=seed,
                free_clean=free,
                partition_free_clean=partition_free,
                wall_clean=wall,
                real_wall_barrier_map=real_wall,
                no_clearance_occupied_map=extension_occupied,
                unknown_clean=unknown,
                resolution_m=float(resolution_m),
                cfg=cfg,
                wall_anchor_source_map=source_map,
                same_cluster_mask=group.mask,
                forced_major=np.asarray(primitive.major_dir_rc, dtype=np.float32),
                orientation_source=orientation_source,
                cluster_id=int(primitive.source_cluster_id),
                seed_group_id=int(group.group_id),
                seed_group_kind=str(group.group_kind),
                source_cluster_ids=list(group.source_cluster_ids),
                component_ids=list(primitive.source_component_ids),
                mode_hint=mode_hint,
            )
            candidate.debug.update(
                {
                    "primitive_id": int(primitive.primitive_id),
                    "source_primitive_id": int(primitive.primitive_id),
                    "primitive_extraction_method": str(primitive.extraction_method),
                    "seed_line_primitive": primitive.to_dict(),
                    "accepted_orientation_source": str(getattr(cfg, "accepted_orientation_source", "seed_primitive_only")),
                    "axis_hv_for_accepted_disabled": not bool(getattr(cfg, "allow_axis_hv_for_accepted", False)),
                    "wall_pair_axis_for_accepted_disabled": not bool(getattr(cfg, "allow_wall_pair_axis_for_accepted", False)),
                    "local_free_neck_axis_for_accepted_disabled": not bool(getattr(cfg, "allow_local_free_neck_axis_for_accepted", False)),
                    "voxel_v32_seed_line_primitive_completion": True,
                }
            )
            trial_candidates.append(candidate)
            all_trial_candidates.append(candidate)
            cid += 1
        if trial_candidates:
            selected = _select_best_cluster_candidate(trial_candidates)
            selected_reason = (
                "verified_partition"
                if bool(selected.debug.get("partition_effective_verified", False))
                else (
                    "geometry_candidate"
                    if bool(selected.debug.get("partition_geometry_accepted", False))
                    else ("visual_candidate" if bool(selected.accepted) else "best_rejected_candidate")
                )
            )
            selected_candidates.append(selected)
            trial_groups.append(
                DoorTrialCandidateGroup(
                    cluster_id=int(group.group_id),
                    component_ids=[int(v) for v in group.component_ids],
                    trials=list(trial_candidates),
                    selected_candidate_id=int(selected.candidate_id),
                    selected_reason=selected_reason,
                    debug={
                        "trial_count": int(len(trial_candidates)),
                        "selected_score": float(selected.debug.get("score", 0.0)),
                        "selected_partition_accepted": bool(selected.debug.get("partition_accepted", False)),
                        "selected_partition_effective_verified": bool(selected.debug.get("partition_effective_verified", False)),
                        "selected_visual_accepted": bool(selected.accepted),
                        "selected_reason": selected_reason,
                        "seed_group_id": int(group.group_id),
                        "seed_group_kind": str(group.group_kind),
                        "source_cluster_ids": [int(v) for v in group.source_cluster_ids],
                        "primitive_count_for_group": int(len(group_primitives)),
                        "accepted_primitive_count_for_group": int(len(accepted_primitives)),
                        "selected_primitive_id": int(selected.debug.get("primitive_id", 0) or 0),
                    },
                )
            )
        else:
            candidate = _rejected_candidate(cid, int(group.group_id), list(group.seed_cells), "no_orientation_candidates")
            candidate.debug.update(
                {
                    "cluster_id": int(group.group_id),
                    "seed_group_id": int(group.group_id),
                    "seed_group_kind": str(group.group_kind),
                    "source_cluster_ids": [int(v) for v in group.source_cluster_ids],
                    "component_ids": [int(v) for v in group.component_ids],
                    "completion_mode": DOOR_COMPLETION_REJECTED,
                    "visual_accepted": False,
                    "partition_accepted": False,
                    "partition_topology_accepted": False,
                    "reject_reason_visual": "no_orientation_candidates",
                    "reject_reason_partition": "visual_rejected",
                    "reject_reason_topology": "visual_rejected",
                }
            )
            selected_candidates.append(candidate)
            all_trial_candidates.append(candidate)
            trial_groups.append(
                DoorTrialCandidateGroup(
                    cluster_id=int(group.group_id),
                    component_ids=[int(v) for v in group.component_ids],
                    trials=[candidate],
                    selected_candidate_id=int(candidate.candidate_id),
                    selected_reason="no_orientation_candidates",
                    debug={
                        "trial_count": 1,
                        "selected_score": 0.0,
                        "seed_group_id": int(group.group_id),
                        "seed_group_kind": str(group.group_kind),
                        "source_cluster_ids": [int(v) for v in group.source_cluster_ids],
                    },
                )
            )
            cid += 1

    conflict_debug = _batch_reject_conflicting_doors(selected_candidates, seed, shape, cfg, door_conflict_free_mask=conflict_free)
    primitive_id_map = np.zeros(shape, dtype=np.int32)
    primitive_mask = np.zeros(shape, dtype=bool)
    extensible_primitive_mask = np.zeros(shape, dtype=bool)
    rejected_primitive_mask = np.zeros(shape, dtype=bool)
    primitive_reject_reason_map = np.zeros(shape, dtype=np.uint8)
    for primitive in all_primitives:
        mask = _cells_to_mask(primitive.cells, shape)
        primitive_mask |= mask
        primitive_id_map[(mask) & (primitive_id_map == 0)] = int(primitive.primitive_id)
        if bool(primitive.accepted_for_extension):
            extensible_primitive_mask |= mask
        else:
            rejected_primitive_mask |= mask
            primitive_reject_reason_map[mask] = _door_reject_code(str(primitive.reject_reason or "primitive_rejected"))
    trial_candidate_mask = np.zeros(shape, dtype=bool)
    trial_rejected_mask = np.zeros(shape, dtype=bool)
    selected_candidate_mask = np.zeros(shape, dtype=bool)
    partition_candidate_mask = np.zeros(shape, dtype=bool)
    partition_rejected_mask = np.zeros(shape, dtype=bool)
    partition_reject_reason_map = np.zeros(shape, dtype=np.uint8)
    visual_all_mask = np.zeros(shape, dtype=bool)
    visual_partition_mask = np.zeros(shape, dtype=bool)
    visual_only_mask = np.zeros(shape, dtype=bool)
    geometry_warning_cut_mask = np.zeros(shape, dtype=bool)
    geometry_only_cut_mask = np.zeros(shape, dtype=bool)
    attachment_only_cut_mask = np.zeros(shape, dtype=bool)
    cut_not_closed_to_wall_mask = np.zeros(shape, dtype=bool)
    partition_mask = np.zeros(shape, dtype=bool)
    effective_verified_mask = np.zeros(shape, dtype=bool)
    topology_accepted_cut_mask = np.zeros(shape, dtype=bool)
    topology_warning_cut_mask = np.zeros(shape, dtype=bool)
    wall_attachment_reject_map = np.zeros(shape, dtype=np.uint8)
    rejected_mask = np.zeros(shape, dtype=bool)
    rejected_reason_map = np.zeros(shape, dtype=np.uint8)
    for candidate in all_trial_candidates:
        visual_cells = candidate.extended_centerline_cells or candidate.door_cut_cells or candidate.seed_projected_centerline_cells
        visual = _cells_to_mask(visual_cells, shape)
        if int(cfg.cut_thickness_cells) > 1:
            visual = dilate(visual, int(cfg.cut_thickness_cells) - 1)
        trial_candidate_mask |= visual
        if not bool(candidate.accepted):
            trial_rejected_mask |= visual
        cand_cut = _cells_to_mask([tuple(v) for v in candidate.debug.get("partition_cut_candidate_cells", [])], shape)  # type: ignore[arg-type]
        partition_candidate_mask |= cand_cut
    for candidate in selected_candidates:
        visual_cells = candidate.extended_centerline_cells or candidate.door_cut_cells or candidate.seed_projected_centerline_cells
        cut_cells = candidate.door_cut_cells or candidate.seed_projected_centerline_cells
        visual = _cells_to_mask(visual_cells, shape)
        cut = _cells_to_mask(cut_cells, shape)
        if int(cfg.cut_thickness_cells) > 1:
            visual = dilate(visual, int(cfg.cut_thickness_cells) - 1)
            cut = dilate(cut, int(cfg.cut_thickness_cells) - 1)
        selected_candidate_mask |= visual
        if candidate.accepted:
            visual_all_mask |= visual
            if bool(candidate.debug.get("partition_accepted", bool(candidate.door_cut_cells))):
                visual_partition_mask |= visual
                partition_mask |= cut
                if bool(candidate.debug.get("partition_effective_verified", candidate.debug.get("partition_topology_effective", candidate.debug.get("partition_topology_accepted", False)))):
                    effective_verified_mask |= cut
                    topology_accepted_cut_mask |= cut
                elif bool(candidate.debug.get("door_topology_warning", False)):
                    topology_warning_cut_mask |= cut
            else:
                visual_only_mask |= visual
                cand_cut = _cells_to_mask([tuple(v) for v in candidate.debug.get("partition_cut_candidate_cells", [])], shape)  # type: ignore[arg-type]
                if bool(candidate.debug.get("partition_geometry_accepted", False)):
                    geometry_cells = _cells_to_mask(candidate.door_cut_cells or [tuple(v) for v in candidate.debug.get("partition_cut_candidate_cells", [])], shape)  # type: ignore[arg-type]
                    geometry_warning_cut_mask |= geometry_cells
                    if bool(candidate.debug.get("partition_closure_attached", False)) and not bool(candidate.debug.get("partition_topology_gain", False)):
                        attachment_only_cut_mask |= geometry_cells
                    else:
                        geometry_only_cut_mask |= geometry_cells
                    if str(candidate.debug.get("reject_reason_partition") or "") in {"door_cut_not_closed_to_wall", "door_cut_not_strict_wall_to_wall_closure"}:
                        cut_not_closed_to_wall_mask |= geometry_cells
                    topology_warning_cut_mask |= geometry_cells
                    wall_attachment_reject_map[geometry_cells] = _door_reject_code(str(candidate.debug.get("door_wall_attachment_reject_reason") or "door_cut_not_wall_attached"))
                partition_rejected_mask |= cand_cut
                partition_reject_reason_map[cand_cut] = _door_reject_code(str(candidate.debug.get("reject_reason_partition") or candidate.debug.get("reject_reason_topology") or "partition_rejected"))
        else:
            rejected_mask |= visual
            rejected_reason_map[visual] = _door_reject_code(str(candidate.reject_reason or candidate.debug.get("reject_reason_visual") or "rejected"))
    completion_ms = float((time.perf_counter() - started_at) * 1000.0)
    reason_counts = Counter(str(candidate.reject_reason) for candidate in selected_candidates if not candidate.accepted)
    partition_reason_counts = Counter(
        str(candidate.debug.get("reject_reason_partition"))
        for candidate in selected_candidates
        if candidate.accepted and not bool(candidate.debug.get("partition_accepted", False))
    )
    topology_reason_counts = Counter(
        str(candidate.debug.get("door_topology_reject_reason") or candidate.debug.get("reject_reason_topology"))
        for candidate in selected_candidates
        if candidate.accepted and (candidate.debug.get("door_topology_reject_reason") is not None or candidate.debug.get("reject_reason_topology") is not None)
    )
    trial_reason_counts = Counter(str(candidate.debug.get("reject_reason_visual") or candidate.reject_reason) for candidate in all_trial_candidates if not candidate.accepted)
    source_counts = _anchor_source_counts(source_map, wall)
    completion_mode_counts = Counter(str(candidate.debug.get("completion_mode", DOOR_COMPLETION_REJECTED)) for candidate in selected_candidates)
    trial_mode_counts = Counter(str(candidate.debug.get("completion_mode", DOOR_COMPLETION_REJECTED)) for candidate in all_trial_candidates)
    orientation_counts = Counter(str(candidate.debug.get("orientation_source", "unknown")) for candidate in all_trial_candidates)
    selected_reason_counts = Counter(str(group.selected_reason) for group in trial_groups)
    visual_count = int(sum(1 for candidate in selected_candidates if candidate.accepted))
    partition_count = int(
        sum(
            1
            for candidate in selected_candidates
            if candidate.accepted and bool(candidate.debug.get("partition_accepted", False))
        )
    )
    extensible_seed_group_mask = np.zeros(shape, dtype=bool)
    for primitive in all_primitives:
        if bool(primitive.accepted_for_extension):
            extensible_seed_group_mask |= _cells_to_mask(primitive.cells, shape)
    cluster_reason_counts = Counter(str(cluster.reject_reason) for cluster in clusters if not bool(cluster.accepted_for_completion) and cluster.reject_reason is not None)
    primitive_reason_counts = Counter(str(primitive.reject_reason) for primitive in all_primitives if not bool(primitive.accepted_for_extension) and primitive.reject_reason is not None)
    extension_reason_counts = Counter(str(candidate.debug.get("reject_reason_visual") or candidate.reject_reason) for candidate in all_trial_candidates if not bool(candidate.accepted))
    debug = {
        "voxel_door_completion_ms": float(completion_ms),
        "voxel_v32_seed_line_primitive_completion": True,
        "voxel_door_extension_attempt_all_mask": trial_candidate_mask.astype(bool),
        "voxel_door_extension_trials_map": trial_candidate_mask.astype(bool),
        "voxel_door_extension_attempt_selected_mask": selected_candidate_mask.astype(bool),
        "voxel_door_extension_attempt_rejected_mask": trial_rejected_mask.astype(bool),
        "voxel_door_extension_attempt_reason_map": rejected_reason_map.astype(np.uint8),
        "voxel_door_trial_candidate_lines_map": trial_candidate_mask.astype(bool),
        "voxel_door_trial_rejected_lines_map": trial_rejected_mask.astype(bool),
        "voxel_door_selected_candidate_lines_map": selected_candidate_mask.astype(bool),
        "voxel_door_candidate_lines_map": trial_candidate_mask.astype(bool),
        "voxel_door_centerline_visual_mask": visual_partition_mask.astype(bool),
        "voxel_door_visual_only_mask": visual_only_mask.astype(bool),
        "voxel_door_partition_cut_candidate_mask": partition_candidate_mask.astype(bool),
        "voxel_door_partition_cut_mask": partition_mask.astype(bool),
        "voxel_door_partition_cut_rejected_mask": partition_rejected_mask.astype(bool),
        "voxel_door_partition_reject_reason_map": partition_reject_reason_map.astype(np.uint8),
        "voxel_door_partition_reject_reason_id_map": partition_reject_reason_map.astype(np.uint8),
        "voxel_door_rejected_lines_map": rejected_mask.astype(bool),
        "voxel_door_rejected_by_reason_map": rejected_reason_map.astype(np.uint8),
        "voxel_door_extension_reject_reason_id_map": rejected_reason_map.astype(np.uint8),
        "voxel_door_centerline_mask": partition_mask.astype(bool),
        "voxel_door_cut_mask": partition_mask.astype(bool),
        "voxel_door_partition_cut_accepted_mask": partition_mask.astype(bool),
        "voxel_door_geometry_accepted_cut_mask": (partition_mask | geometry_warning_cut_mask).astype(bool),
        "voxel_door_geometry_warning_cut_mask": geometry_warning_cut_mask.astype(bool),
        "voxel_door_geometry_only_mask": geometry_only_cut_mask.astype(bool),
        "voxel_door_attachment_only_mask": attachment_only_cut_mask.astype(bool),
        "voxel_door_cut_not_closed_to_wall_mask": cut_not_closed_to_wall_mask.astype(bool),
        "voxel_door_partition_effective_verified_mask": effective_verified_mask.astype(bool),
        "voxel_door_topology_effective_cut_mask": topology_accepted_cut_mask.astype(bool),
        "voxel_door_final_cut_mask": partition_mask.astype(bool),
        "voxel_door_topology_accepted_cut_mask": topology_accepted_cut_mask.astype(bool),
        "voxel_door_topology_warning_cut_mask": topology_warning_cut_mask.astype(bool),
        "voxel_door_wall_attachment_reject_map": wall_attachment_reject_map.astype(np.uint8),
        "voxel_accepted_door_centerline_mask": visual_partition_mask.astype(bool),
        "voxel_rejected_door_centerline_mask": rejected_mask.astype(bool),
        "voxel_door_candidates": [candidate.to_dict() for candidate in selected_candidates],
        "voxel_door_trial_candidates": [candidate.to_dict() for candidate in all_trial_candidates],
        "voxel_door_extension_trials": [DoorExtensionTrial.from_candidate(candidate).to_dict() for candidate in all_trial_candidates],
        "voxel_door_trial_candidate_groups": [group.to_dict() for group in trial_groups],
        "voxel_door_reject_reason_counts": dict(reason_counts),
        "voxel_door_trial_reject_reason_counts": dict(trial_reason_counts),
        "voxel_door_extension_reject_reason_counts": dict(extension_reason_counts),
        "voxel_door_completion_mode_counts": dict(completion_mode_counts),
        "voxel_door_candidate_mode_counts": dict(trial_mode_counts),
        "voxel_door_candidate_orientation_counts": dict(orientation_counts),
        "voxel_door_candidate_selected_reason_counts": dict(selected_reason_counts),
        "voxel_door_rejected_long_line_count": int(
            sum(
                1
                for candidate in all_trial_candidates
                if str(candidate.reject_reason or candidate.debug.get("reject_reason_visual")) in {"door_line_too_long", "door_one_seed_line_too_long", "door_seed_pair_line_too_long", "door_visual_width_out_of_range"}
            )
        ),
        "voxel_door_rejected_open_free_crossing_count": int(
            sum(
                1
                for candidate in all_trial_candidates
                if str(candidate.reject_reason or candidate.debug.get("reject_reason_visual")) == "door_line_crosses_large_open_free"
            )
        ),
        "voxel_door_cluster_reject_reason_counts": dict(cluster_reason_counts),
        "voxel_door_seed_cluster_reject_reason_counts": dict(cluster_reason_counts),
        "voxel_door_primitive_reject_reason_counts": dict(primitive_reason_counts),
        "voxel_door_primitive_require_line_fit_quality": bool(getattr(cfg, "primitive_require_line_fit_quality", True)),
        "voxel_door_visual_reject_reason_counts": dict(reason_counts),
        "voxel_door_partition_reject_reason_counts": dict(partition_reason_counts),
        "voxel_door_topology_reject_reason_counts": dict(topology_reason_counts),
        "voxel_door_topology_debug_per_candidate": [DoorExtensionTrial.from_candidate(candidate).to_dict() for candidate in selected_candidates],
        "voxel_door_candidate_count": int(len(selected_candidates)),
        "voxel_door_trial_candidate_count": int(len(all_trial_candidates)),
        "voxel_door_extension_trial_count": int(len(all_trial_candidates)),
        "voxel_door_selected_candidate_count": int(len(selected_candidates)),
        "voxel_door_visual_accepted_count": int(visual_count),
        "voxel_door_partition_accepted_count": int(partition_count),
        "voxel_door_accepted_count": int(partition_count),
        "voxel_door_topology_effective_cells": int(np.count_nonzero(topology_accepted_cut_mask)),
        "voxel_door_partition_effective_verified_cells": int(np.count_nonzero(effective_verified_mask)),
        "voxel_door_final_cut_cells": int(np.count_nonzero(partition_mask)),
        "voxel_door_geometry_warning_cells": int(np.count_nonzero(geometry_warning_cut_mask)),
        "voxel_door_rejected_count": int(sum(1 for candidate in selected_candidates if not candidate.accepted)),
        "voxel_door_visual_only_cells": int(np.count_nonzero(visual_only_mask)),
        "voxel_door_green_default_cells": int(np.count_nonzero(visual_partition_mask | partition_mask)),
        "voxel_door_anchor_wall_union_map": wall.astype(bool),
        "voxel_door_anchor_source_map": source_map.astype(np.uint8),
        "voxel_door_anchor_source_counts": source_counts,
        "voxel_door_seed_mask": seed.astype(bool),
        "voxel_door_raw_seed_mask": seed.astype(bool),
        "voxel_door_raw_seed_cells": int(np.count_nonzero(seed)),
        "voxel_door_seed_component_id_map": labels.astype(np.int32),
        "voxel_door_seed_cluster_id_map": cluster_map.astype(np.int32),
        "voxel_door_seed_reject_reason_id_map": np.asarray(seed_result.door_seed_reject_reason_map, dtype=np.uint8),
        "voxel_door_seed_line_primitive_id_map": primitive_id_map.astype(np.int32),
        "voxel_door_seed_line_primitive_mask": primitive_mask.astype(bool),
        "voxel_door_extensible_primitive_mask": extensible_primitive_mask.astype(bool),
        "voxel_door_rejected_primitive_mask": rejected_primitive_mask.astype(bool),
        "voxel_door_primitive_reject_reason_map": primitive_reject_reason_map.astype(np.uint8),
        "voxel_door_line_primitives": [primitive.to_dict() for primitive in all_primitives],
        "voxel_door_line_primitive_count": int(len(all_primitives)),
        "voxel_door_extensible_primitive_count": int(sum(1 for primitive in all_primitives if primitive.accepted_for_extension)),
        "voxel_door_extensible_seed_group_mask": extensible_seed_group_mask.astype(bool),
        "voxel_door_seed_cluster_map": cluster_map.astype(np.int32),
        "voxel_door_seed_clusters": [cluster.to_dict() for cluster in clusters],
        "voxel_door_seed_cluster_count": int(len(clusters)),
        **seed_group_debug,
        "voxel_door_seed_component_count": int(labels.max()) if labels.size else 0,
        "voxel_door_provisional_accepted_visual_mask": visual_all_mask.astype(bool),
        **conflict_debug,
        **cluster_debug,
    }
    return VoxelDoorCompletionResult(
        door_seed_mask=seed.astype(bool),
        door_extension_attempt_all_mask=trial_candidate_mask.astype(bool),
        door_extension_attempt_rejected_mask=trial_rejected_mask.astype(bool),
        door_centerline_visual_mask=visual_partition_mask.astype(bool),
        door_visual_only_mask=visual_only_mask.astype(bool),
        door_geometry_warning_cut_mask=geometry_warning_cut_mask.astype(bool),
        door_topology_effective_cut_mask=topology_accepted_cut_mask.astype(bool),
        door_partition_cut_candidate_mask=partition_candidate_mask.astype(bool),
        door_cut_mask_for_partition=partition_mask.astype(bool),
        door_centerline_candidate_mask=trial_candidate_mask.astype(bool),
        rejected_door_centerline_mask=rejected_mask.astype(bool),
        door_reject_reason_map=rejected_reason_map.astype(np.uint8),
        candidates=selected_candidates,
        debug=debug,
    )


def detect_voxel_doors(
    *,
    voxel_grid: VoxelOccupancyGrid3D,
    free_map: np.ndarray,
    wall_map: np.ndarray,
    unknown_map: np.ndarray,
    resolution_m: float,
    config: VoxelDoorDetectorConfig | Mapping[str, object] | None = None,
    navigation_free_mask: np.ndarray | None = None,
    door_seed_no_clearance_free_mask: np.ndarray | None = None,
) -> VoxelDoorDetectionResult:
    cfg = config if isinstance(config, VoxelDoorDetectorConfig) else VoxelDoorDetectorConfig.from_mapping(config)
    free = np.asarray(free_map, dtype=bool)
    wall = np.asarray(wall_map, dtype=bool)
    unknown = np.asarray(unknown_map, dtype=bool)
    shape = free.shape
    if wall.shape != shape or unknown.shape != shape or tuple(voxel_grid.shape) != tuple(shape):
        raise ValueError("voxel door maps must share one HxW shape")
    if not bool(cfg.enabled):
        return _empty_result(shape, enabled=False)

    seed_gate_free = (
        door_seed_no_clearance_free_mask
        if door_seed_no_clearance_free_mask is not None
        else (navigation_free_mask if navigation_free_mask is not None else free)
    )
    seed_result = classify_voxel_door_seeds(
        voxel_grid=voxel_grid,
        config=cfg,
        sensor_range_count=getattr(voxel_grid, "sensor_range_count", None),
        navigation_free_mask=free if navigation_free_mask is None else navigation_free_mask,
        no_clearance_navigation_free_mask=seed_gate_free,
    )
    completion = complete_voxel_doors_from_seeds(
        seed_result=seed_result,
        free_map=free,
        anchor_wall_map=wall,
        unknown_map=unknown,
        resolution_m=float(resolution_m),
        config=cfg,
        door_conflict_free_mask=seed_gate_free,
    )
    debug = {
        **seed_result.debug,
        **completion.debug,
        "voxel_door_component_ms": float(completion.debug.get("voxel_door_completion_ms", 0.0)),
    }
    return VoxelDoorDetectionResult(
        door_seed_mask=seed_result.door_seed_mask.astype(bool),
        door_seed_component_map=seed_result.door_seed_component_map.astype(np.int32),
        door_centerline_candidate_mask=completion.door_centerline_candidate_mask.astype(bool),
        accepted_door_centerline_mask=completion.door_centerline_visual_mask.astype(bool),
        rejected_door_centerline_mask=completion.rejected_door_centerline_mask.astype(bool),
        door_cut_mask=completion.door_cut_mask_for_partition.astype(bool),
        door_seed_reject_reason_map=seed_result.door_seed_reject_reason_map.astype(np.uint8),
        candidates=completion.candidates,
        debug=debug,
    )

def classify_voxel_door_seeds_vectorized(
    z_state: np.ndarray,
    z_centers_m: np.ndarray,
    active_z_indices: np.ndarray,
    cfg: VoxelDoorDetectorConfig,
    *,
    shape: tuple[int, int],
    return_debug: bool = False,
    sensor_range_count: np.ndarray | None = None,
):
    method = str(getattr(cfg, "seed_method", "centroid_ratio") or "centroid_ratio").strip().lower()
    if method in {"centroid_ratio", "centroid", "ratio"}:
        return classify_voxel_door_seeds_centroid_ratio_vectorized(
            z_state,
            z_centers_m,
            active_z_indices,
            cfg,
            shape=shape,
            return_debug=return_debug,
            sensor_range_count=sensor_range_count,
        )
    return classify_voxel_door_seeds_strict_contiguous_vectorized(
        z_state,
        z_centers_m,
        active_z_indices,
        cfg,
        shape=shape,
        return_debug=return_debug,
    )


def classify_voxel_door_seeds_strict_contiguous_vectorized(
    z_state: np.ndarray,
    z_centers_m: np.ndarray,
    active_z_indices: np.ndarray,
    cfg: VoxelDoorDetectorConfig,
    *,
    shape: tuple[int, int],
    return_debug: bool = False,
):
    states_all = np.asarray(z_state, dtype=np.uint8)
    centers = np.asarray(z_centers_m, dtype=np.float32).reshape(-1)
    idxs = np.asarray(active_z_indices, dtype=np.int32).reshape(-1)
    idxs = idxs[(idxs >= 0) & (idxs < states_all.shape[0]) & (centers[idxs] >= float(cfg.z_scan_min_m))]
    height, width = int(shape[0]), int(shape[1])
    seed = np.zeros(shape, dtype=bool)
    reason_map = np.zeros(shape, dtype=np.uint8)
    lower_free_xy = np.zeros(shape, dtype=np.uint16)
    top_occ_xy = np.zeros(shape, dtype=np.uint16)
    unknown_tail_xy = np.zeros(shape, dtype=np.uint16)
    first_occ_z_xy = np.full(shape, np.nan, dtype=np.float32)
    accepted_seed_evidence: list[VoxelDoorSeedEvidence] = []
    rejected_seed_reasons: Counter[str] = Counter()
    if idxs.size == 0:
        reason_map[:, :] = _seed_reject_code("no_active_z_bins")
        rejected_seed_reasons["no_active_z_bins"] = int(height * width)
        result = (seed, reason_map, lower_free_xy, top_occ_xy, unknown_tail_xy, first_occ_z_xy, rejected_seed_reasons, accepted_seed_evidence)
        return (*result, _empty_seed_debug_maps(shape)) if bool(return_debug) else result

    state = states_all[idxs].reshape(int(idxs.size), height * width)
    z_count, column_count = state.shape
    is_free = state == int(VOXEL_FREE)
    is_occ = state == int(VOXEL_OCCUPIED)
    is_unknown = state == int(VOXEL_UNKNOWN)
    is_conflict = state == int(VOXEL_CONFLICT)

    non_free = ~is_free
    has_non_free = np.any(non_free, axis=0)
    first_non_free = np.argmax(non_free, axis=0).astype(np.int32)
    lower_free = np.where(has_non_free, first_non_free, z_count).astype(np.uint16)
    reason = np.full(column_count, "accepted", dtype=object)
    accepted = np.zeros(column_count, dtype=bool)
    first_occ_z = np.full(column_count, np.nan, dtype=np.float32)
    top_occ = np.zeros(column_count, dtype=np.uint16)
    unknown_tail = np.zeros(column_count, dtype=np.uint16)

    lower_bad = lower_free.astype(np.int32) < int(cfg.min_lower_free_cells)
    reason[lower_bad] = "lower_free_cells_too_few"
    no_top = (~lower_bad) & ~has_non_free
    reason[no_top] = "no_top_occupied"

    cand = (~lower_bad) & has_non_free
    cand_cols = np.flatnonzero(cand)
    if cand_cols.size:
        first_idx = first_non_free[cand_cols]
        first_state = state[first_idx, cand_cols]
        unknown_first = first_state == int(VOXEL_UNKNOWN)
        conflict_first = first_state == int(VOXEL_CONFLICT)
        occ_first = first_state == int(VOXEL_OCCUPIED)
        reason[cand_cols[unknown_first]] = "unknown_before_top_occupied"
        reason[cand_cols[conflict_first]] = "conflict_before_top_occupied"
        reason[cand_cols[~unknown_first & ~conflict_first & ~occ_first]] = "expected_top_occupied"

        occ_cols = cand_cols[occ_first]
        if occ_cols.size:
            occ_first_idx = first_non_free[occ_cols]
            occ_first_z = centers[idxs[occ_first_idx]]
            first_occ_z[occ_cols] = occ_first_z.astype(np.float32)
            top_low = occ_first_z < float(cfg.top_occupied_min_z_m)
            reason[occ_cols[top_low]] = "top_occupied_too_low"
            top_cols = occ_cols[~top_low]
            if top_cols.size:
                still = np.zeros(column_count, dtype=bool)
                still[top_cols] = True
                for k in range(z_count):
                    active = still & (k >= first_non_free)
                    if not np.any(active):
                        continue
                    occ_here = active & is_occ[k]
                    top_occ[occ_here] = np.minimum(top_occ[occ_here].astype(np.uint32) + 1, np.iinfo(np.uint16).max).astype(np.uint16)
                    still[active & ~is_occ[k]] = False
                top_short = top_cols[top_occ[top_cols].astype(np.int32) < int(cfg.min_top_occupied_cells)]
                reason[top_short] = "top_occupied_run_too_short"
                tail_cols = top_cols[top_occ[top_cols].astype(np.int32) >= int(cfg.min_top_occupied_cells)]
                if tail_cols.size:
                    tail_start = first_non_free[tail_cols].astype(np.int32) + top_occ[tail_cols].astype(np.int32)
                    at_end = tail_start >= z_count
                    if np.any(at_end):
                        if bool(cfg.allow_end_of_active_range_after_top_occupied):
                            accepted[tail_cols[at_end]] = True
                        else:
                            reason[tail_cols[at_end]] = "missing_unknown_tail"
                    rest_cols = tail_cols[~at_end]
                    rest_start = tail_start[~at_end]
                    if rest_cols.size:
                        first_tail_state = state[rest_start, rest_cols]
                        non_unknown_tail = first_tail_state != int(VOXEL_UNKNOWN)
                        reason[rest_cols[non_unknown_tail]] = "non_unknown_after_top_occupied"
                        unknown_tail_cols = rest_cols[~non_unknown_tail]
                        if unknown_tail_cols.size:
                            if not bool(cfg.allow_unknown_tail_after_top_occupied):
                                reason[unknown_tail_cols] = "unknown_tail_not_allowed"
                            else:
                                still_tail = np.zeros(column_count, dtype=bool)
                                still_tail[unknown_tail_cols] = True
                                tail_start_all = np.zeros(column_count, dtype=np.int32)
                                tail_start_all[rest_cols] = rest_start
                                for k in range(z_count):
                                    active = still_tail & (k >= tail_start_all)
                                    if not np.any(active):
                                        continue
                                    unknown_here = active & is_unknown[k]
                                    unknown_tail[unknown_here] = np.minimum(
                                        unknown_tail[unknown_here].astype(np.uint32) + 1,
                                        np.iinfo(np.uint16).max,
                                    ).astype(np.uint16)
                                    still_tail[active & ~is_unknown[k]] = False
                                post_start = tail_start_all[unknown_tail_cols].astype(np.int32) + unknown_tail[unknown_tail_cols].astype(np.int32)
                                tail_ended = post_start >= z_count
                                accepted[unknown_tail_cols[tail_ended]] = True
                                post_cols = unknown_tail_cols[~tail_ended]
                                post_idx = post_start[~tail_ended]
                                if post_cols.size:
                                    post_state = state[post_idx, post_cols]
                                    free_after = post_state == int(VOXEL_FREE)
                                    occ_after = post_state == int(VOXEL_OCCUPIED)
                                    conflict_after = post_state == int(VOXEL_CONFLICT)
                                    if bool(cfg.reject_free_after_unknown_tail):
                                        reason[post_cols[free_after]] = "free_after_unknown_tail"
                                    if bool(cfg.reject_occupied_after_unknown_tail):
                                        reason[post_cols[occ_after]] = "occupied_after_unknown_tail"
                                    if bool(cfg.reject_conflict_in_seed_pattern):
                                        reason[post_cols[conflict_after]] = "conflict_after_unknown_tail"
                                    other = ~(free_after & bool(cfg.reject_free_after_unknown_tail)) & ~(occ_after & bool(cfg.reject_occupied_after_unknown_tail)) & ~(conflict_after & bool(cfg.reject_conflict_in_seed_pattern))
                                    reason[post_cols[other]] = "state_after_unknown_tail"

    seed_flat = accepted
    reason_flat = np.zeros(column_count, dtype=np.uint8)
    reason_flat[seed_flat] = 1
    rejected = ~seed_flat
    for item in reason[rejected]:
        rejected_seed_reasons[str(item)] += 1
    for idx_col in np.flatnonzero(rejected):
        reason_flat[idx_col] = _seed_reject_code(str(reason[idx_col]))
    lower_free_xy[:, :] = lower_free.reshape(shape).astype(np.uint16)
    top_occ_xy[:, :] = top_occ.reshape(shape).astype(np.uint16)
    unknown_tail_xy[:, :] = unknown_tail.reshape(shape).astype(np.uint16)
    first_occ_z_xy[:, :] = first_occ_z.reshape(shape).astype(np.float32)
    seed[:, :] = seed_flat.reshape(shape)
    reason_map[:, :] = reason_flat.reshape(shape)
    for idx_col in np.flatnonzero(seed_flat)[:2048]:
        row = int(idx_col // width)
        col = int(idx_col % width)
        accepted_seed_evidence.append(
            _seed_ev(
                row,
                col,
                None if not np.isfinite(first_occ_z[idx_col]) else float(first_occ_z[idx_col]),
                int(lower_free[idx_col]),
                int(top_occ[idx_col]),
                int(unknown_tail[idx_col]),
                True,
                None,
            )
        )
    result = (seed, reason_map, lower_free_xy, top_occ_xy, unknown_tail_xy, first_occ_z_xy, rejected_seed_reasons, accepted_seed_evidence)
    return (*result, _empty_seed_debug_maps(shape)) if bool(return_debug) else result


def classify_voxel_door_seeds_centroid_ratio_vectorized(
    z_state: np.ndarray,
    z_centers_m: np.ndarray,
    active_z_indices: np.ndarray,
    cfg: VoxelDoorDetectorConfig,
    *,
    shape: tuple[int, int],
    return_debug: bool = False,
    sensor_range_count: np.ndarray | None = None,
):
    states_all = np.asarray(z_state, dtype=np.uint8)
    sensor_all = None if sensor_range_count is None else np.asarray(sensor_range_count)
    if sensor_all is not None and sensor_all.shape != states_all.shape:
        sensor_all = None
    centers = np.asarray(z_centers_m, dtype=np.float32).reshape(-1)
    idxs = np.asarray(active_z_indices, dtype=np.int32).reshape(-1)
    idxs = idxs[(idxs >= 0) & (idxs < states_all.shape[0]) & (centers[idxs] >= float(cfg.z_scan_min_m))]
    height, width = int(shape[0]), int(shape[1])
    seed = np.zeros(shape, dtype=bool)
    reason_map = np.zeros(shape, dtype=np.uint8)
    lower_free_xy = np.zeros(shape, dtype=np.uint16)
    top_occ_xy = np.zeros(shape, dtype=np.uint16)
    unknown_tail_xy = np.zeros(shape, dtype=np.uint16)
    first_occ_z_xy = np.full(shape, np.nan, dtype=np.float32)
    accepted_seed_evidence: list[VoxelDoorSeedEvidence] = []
    rejected_seed_reasons: Counter[str] = Counter()
    empty_debug = _empty_seed_debug_maps(shape)
    if idxs.size == 0:
        reason_map[:, :] = _seed_reject_code("no_active_z_bins")
        rejected_seed_reasons["no_active_z_bins"] = int(height * width)
        result = (seed, reason_map, lower_free_xy, top_occ_xy, unknown_tail_xy, first_occ_z_xy, rejected_seed_reasons, accepted_seed_evidence)
        return (*result, empty_debug) if bool(return_debug) else result

    state = states_all[idxs].reshape(int(idxs.size), height * width)
    z = centers[idxs].astype(np.float32)
    z_column = z[:, None]
    z_count, column_count = state.shape
    z_resolution = _estimate_z_resolution_m(z)
    is_free = state == int(VOXEL_FREE)
    is_occ = state == int(VOXEL_OCCUPIED)
    is_unknown = state == int(VOXEL_UNKNOWN)
    if sensor_all is None:
        sensor_state = np.zeros_like(state, dtype=np.int16)
    else:
        sensor_state = sensor_all[idxs].reshape(int(idxs.size), height * width)
    sensor_threshold = int(getattr(cfg, "door_seed_sensor_range_count_threshold", 0))
    is_in_range = sensor_state > sensor_threshold
    is_in_range_unknown = is_unknown & is_in_range
    is_out_of_range_unknown = is_unknown & ~is_in_range
    is_effective_observed = is_free | is_occ | is_in_range_unknown

    free_count = np.sum(is_free, axis=0).astype(np.int32)
    occ_count = np.sum(is_occ, axis=0).astype(np.int32)
    obs_count = np.sum(is_effective_observed, axis=0).astype(np.int32)
    in_range_unknown_count = np.sum(is_in_range_unknown, axis=0).astype(np.int32)
    out_of_range_unknown_count = np.sum(is_out_of_range_unknown, axis=0).astype(np.int32)
    free_sum_z = np.sum(is_free.astype(np.float32) * z_column, axis=0)
    occ_sum_z = np.sum(is_occ.astype(np.float32) * z_column, axis=0)
    free_centroid = np.full(column_count, np.nan, dtype=np.float32)
    occ_centroid = np.full(column_count, np.nan, dtype=np.float32)
    np.divide(free_sum_z, np.maximum(free_count, 1), out=free_centroid, where=free_count > 0)
    np.divide(occ_sum_z, np.maximum(occ_count, 1), out=occ_centroid, where=occ_count > 0)
    first_free_z = _first_true_z(is_free, z)
    last_free_z = _last_true_z(is_free, z)
    free_span_cells = np.zeros(column_count, dtype=np.int32)
    valid_span = np.isfinite(first_free_z) & np.isfinite(last_free_z)
    free_span_cells[valid_span] = np.maximum(
        0,
        np.rint((last_free_z[valid_span] - first_free_z[valid_span]) / max(float(z_resolution), 1e-9)).astype(np.int32) + 1,
    )
    free_extent_cells = np.maximum(free_count, free_span_cells).astype(np.float32)
    turn_z = free_centroid + float(cfg.centroid_turn_extent_scale) * free_extent_cells * float(z_resolution)
    turn_z = np.round(turn_z.astype(np.float64), 6).astype(np.float32)

    finite_turn = np.isfinite(turn_z)
    turn_pos = np.searchsorted(z, np.where(finite_turn, turn_z, z[-1] + 1.0), side="left").astype(np.int32)
    turn_inside_active = finite_turn & (turn_z >= float(z[0]) - 1e-6) & (turn_z <= float(z[-1]) + 0.5 * float(z_resolution) + 1e-6)

    z_rows = np.arange(z_count, dtype=np.int32)[:, None]
    upper_position = z_rows >= np.clip(turn_pos, 0, z_count)[None, :]
    upper_occ_mask = is_occ & upper_position
    has_upper_occ = np.any(upper_occ_mask, axis=0)
    first_upper_occ_idx = np.full(column_count, z_count, dtype=np.int32)
    if np.any(has_upper_occ):
        first_upper_occ_idx[has_upper_occ] = np.argmax(upper_occ_mask[:, has_upper_occ], axis=0).astype(np.int32)
    first_upper_occ_z = np.full(column_count, np.nan, dtype=np.float32)
    first_upper_occ_z[has_upper_occ] = z[first_upper_occ_idx[has_upper_occ]]

    ceiling_band_m = max(0.0, float(getattr(cfg, "ceiling_occupied_reject_band_m", 0.0)))
    ceiling_band_enabled = ceiling_band_m > 0.0 or bool(getattr(cfg, "reject_upper_occupied_only_in_ceiling_band", False))
    turn_max_for_ceiling_band = _finite_float_or_none(getattr(cfg, "door_turn_z_max_m", None))
    if turn_max_for_ceiling_band is not None and float(turn_max_for_ceiling_band) > float(z[0]):
        ceiling_band_reference_z = min(float(z[-1]), float(turn_max_for_ceiling_band))
    else:
        ceiling_band_reference_z = float(z[-1])
    ceiling_band_start_z = float(ceiling_band_reference_z) - ceiling_band_m
    ceiling_band_start_z = max(float(z[0]), float(ceiling_band_start_z))
    first_upper_limit_z = _finite_float_or_none(getattr(cfg, "first_upper_occupied_z_max_m", None))
    if first_upper_limit_z is not None and float(first_upper_limit_z) > 0.0:
        upper_non_ceiling_limit_z = float(first_upper_limit_z)
    elif ceiling_band_enabled:
        upper_non_ceiling_limit_z = float(ceiling_band_start_z) - 1e-6
    else:
        upper_non_ceiling_limit_z = float(z[-1]) + 1e-6
    if ceiling_band_enabled:
        upper_non_ceiling_limit_z = min(float(upper_non_ceiling_limit_z), float(ceiling_band_start_z) - 1e-6)
    z_at_or_above_turn = z_column >= turn_z[None, :]
    non_ceiling_occ_mask = is_occ & z_at_or_above_turn & (z_column <= float(upper_non_ceiling_limit_z) + 1e-6)
    ceiling_occ_mask = is_occ & z_at_or_above_turn & (z_column >= float(ceiling_band_start_z) - 1e-6) if ceiling_band_enabled else np.zeros_like(is_occ, dtype=bool)
    upper_non_ceiling_occ = np.sum(non_ceiling_occ_mask, axis=0).astype(np.int32)
    upper_ceiling_occ = np.sum(ceiling_occ_mask, axis=0).astype(np.int32)

    is_upper_solid = is_occ | is_out_of_range_unknown
    solid_count = np.sum(is_upper_solid, axis=0).astype(np.int32)

    free_cum = np.cumsum(is_free, axis=0, dtype=np.int32)
    occ_cum = np.cumsum(is_occ, axis=0, dtype=np.int32)
    obs_cum = np.cumsum(is_effective_observed, axis=0, dtype=np.int32)
    solid_cum = np.cumsum(is_upper_solid, axis=0, dtype=np.int32)
    in_range_unknown_cum = np.cumsum(is_in_range_unknown, axis=0, dtype=np.int32)
    out_of_range_unknown_cum = np.cumsum(is_out_of_range_unknown, axis=0, dtype=np.int32)
    active_cum = np.cumsum(np.ones_like(is_free, dtype=np.int8), axis=0, dtype=np.int32)
    lower_free = _gather_cum_before(free_cum, turn_pos)
    lower_occ = _gather_cum_before(occ_cum, turn_pos)
    lower_obs = _gather_cum_before(obs_cum, turn_pos)
    lower_solid = _gather_cum_before(solid_cum, turn_pos)
    lower_in_range_unknown = _gather_cum_before(in_range_unknown_cum, turn_pos)
    lower_out_of_range_unknown = _gather_cum_before(out_of_range_unknown_cum, turn_pos)
    lower_total = _gather_cum_before(active_cum, turn_pos)
    upper_occ = (occ_count - lower_occ).astype(np.int32)
    upper_solid = (solid_count - lower_solid).astype(np.int32)
    upper_obs = (obs_count - lower_obs).astype(np.int32)
    upper_total_active = np.maximum(z_count - lower_total, 0).astype(np.int32)
    upper_in_range_unknown = (in_range_unknown_count - lower_in_range_unknown).astype(np.int32)
    upper_out_of_range_unknown = (out_of_range_unknown_count - lower_out_of_range_unknown).astype(np.int32)
    upper_free = (free_count - lower_free).astype(np.int32)

    lower_free_ratio = np.zeros(column_count, dtype=np.float32)
    np.divide(lower_free.astype(np.float32), np.maximum(lower_total, 1), out=lower_free_ratio, where=lower_total > 0)
    upper_occ_ratio = np.zeros(column_count, dtype=np.float32)
    np.divide(upper_solid.astype(np.float32), np.maximum(upper_total_active, 1), out=upper_occ_ratio, where=upper_total_active > 0)
    upper_solid_ratio_active = upper_occ_ratio.copy()
    upper_actual_occupied_ratio_active = np.zeros(column_count, dtype=np.float32)
    upper_in_range_unknown_ratio_active = np.zeros(column_count, dtype=np.float32)
    upper_free_ratio_active = np.zeros(column_count, dtype=np.float32)
    np.divide(upper_occ.astype(np.float32), np.maximum(upper_total_active, 1), out=upper_actual_occupied_ratio_active, where=upper_total_active > 0)
    np.divide(upper_in_range_unknown.astype(np.float32), np.maximum(upper_total_active, 1), out=upper_in_range_unknown_ratio_active, where=upper_total_active > 0)
    np.divide(upper_free.astype(np.float32), np.maximum(upper_total_active, 1), out=upper_free_ratio_active, where=upper_total_active > 0)

    reason = np.full(column_count, "accepted", dtype=object)

    def reject(mask: np.ndarray, name: str) -> None:
        reason[(reason == "accepted") & np.asarray(mask, dtype=bool)] = str(name)

    door_turn_min_z = float(getattr(cfg, "door_turn_z_min_m", getattr(cfg, "top_occupied_min_z_m", 1.80)))
    door_turn_max_z = _finite_float_or_none(getattr(cfg, "door_turn_z_max_m", None))
    first_upper_occ_max_z = _finite_float_or_none(getattr(cfg, "first_upper_occupied_z_max_m", None))
    min_upper_actual_occ = int(getattr(cfg, "upper_actual_occupied_min_cells", cfg.min_upper_occupied_cells))
    reject(free_count < int(cfg.min_lower_free_cells), "lower_free_cells_too_few")
    reject(occ_count < int(cfg.min_upper_occupied_cells), "upper_occupied_cells_too_few")
    reject(~turn_inside_active, "turn_outside_active_range")
    reject(turn_z < door_turn_min_z, "turn_z_below_door_lintel_min")
    if door_turn_max_z is not None and float(door_turn_max_z) > 0.0:
        reject(turn_z > float(door_turn_max_z), "turn_z_above_door_lintel_max")
    reject(lower_total < int(cfg.min_lower_free_cells), "lower_free_cells_too_few")
    reject(lower_free_ratio < float(cfg.lower_free_ratio_min), "lower_free_ratio_too_low")
    reject(upper_total_active < int(cfg.min_upper_observed_cells), "upper_observed_cells_too_few")
    reject(~has_upper_occ, "upper_actual_occupied_cells_too_few")
    if first_upper_occ_max_z is not None and float(first_upper_occ_max_z) > 0.0:
        reject(first_upper_occ_z > float(first_upper_occ_max_z), "first_upper_occupied_too_high_ceiling_like")
    if bool(getattr(cfg, "reject_upper_occupied_only_in_ceiling_band", True)):
        reject((upper_non_ceiling_occ <= 0) & (upper_ceiling_occ >= min_upper_actual_occ), "upper_occupied_only_in_ceiling_band")
    reject(upper_non_ceiling_occ < min_upper_actual_occ, "upper_non_ceiling_actual_occupied_too_few")
    reject(upper_occ < min_upper_actual_occ, "upper_actual_occupied_cells_too_few")
    reject(upper_solid_ratio_active < float(getattr(cfg, "upper_generalized_solid_ratio_min_active", 0.80)), "upper_generalized_solid_ratio_too_low")
    legacy_upper_ratio = float(getattr(cfg, "upper_occupied_ratio_min_observed", 0.80))
    if legacy_upper_ratio > 0.80:
        reject(upper_solid_ratio_active < legacy_upper_ratio, "upper_occupied_ratio_too_low")
    reject(upper_in_range_unknown_ratio_active > float(getattr(cfg, "upper_in_range_unknown_ratio_max", 0.25)), "upper_in_range_unknown_ratio_too_high")
    reject(upper_free_ratio_active > float(getattr(cfg, "upper_free_ratio_max", 0.15)), "upper_free_ratio_too_high")
    if bool(cfg.require_occupied_centroid_above_free_centroid):
        reject(~(occ_centroid > free_centroid), "occupied_centroid_not_above_free_centroid")

    accepted = reason == "accepted"
    reason_flat = np.zeros(column_count, dtype=np.uint8)
    reason_flat[accepted] = 1
    for item in reason[~accepted]:
        rejected_seed_reasons[str(item)] += 1
    for idx_col in np.flatnonzero(~accepted):
        reason_flat[idx_col] = _seed_reject_code(str(reason[idx_col]))

    first_occ_z = first_upper_occ_z
    lower_free_xy[:, :] = np.clip(lower_free, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16)
    top_occ_xy[:, :] = np.clip(upper_occ, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16)
    first_occ_z_xy[:, :] = first_occ_z.reshape(shape).astype(np.float32)
    seed[:, :] = accepted.reshape(shape)
    reason_map[:, :] = reason_flat.reshape(shape)
    debug_maps = {
        "voxel_door_turn_z_estimate_xy": turn_z.reshape(shape).astype(np.float32),
        "voxel_door_free_centroid_z_xy": free_centroid.reshape(shape).astype(np.float32),
        "voxel_door_occupied_centroid_z_xy": occ_centroid.reshape(shape).astype(np.float32),
        "voxel_door_lower_free_ratio_xy": lower_free_ratio.reshape(shape).astype(np.float32),
        "voxel_door_upper_occupied_ratio_observed_xy": upper_occ_ratio.reshape(shape).astype(np.float32),
        "voxel_door_upper_solid_ratio_active_xy": upper_solid_ratio_active.reshape(shape).astype(np.float32),
        "voxel_door_upper_actual_occupied_ratio_active_xy": upper_actual_occupied_ratio_active.reshape(shape).astype(np.float32),
        "voxel_door_upper_in_range_unknown_ratio_active_xy": upper_in_range_unknown_ratio_active.reshape(shape).astype(np.float32),
        "voxel_door_upper_free_ratio_active_xy": upper_free_ratio_active.reshape(shape).astype(np.float32),
        "voxel_door_first_upper_occupied_z_xy": first_upper_occ_z.reshape(shape).astype(np.float32),
        "voxel_door_upper_non_ceiling_actual_occupied_count_xy": np.clip(upper_non_ceiling_occ, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_upper_ceiling_occupied_count_xy": np.clip(upper_ceiling_occ, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_ceiling_like_reject_mask": np.isin(reason.reshape(shape), ["first_upper_occupied_too_high_ceiling_like", "upper_occupied_only_in_ceiling_band"]).astype(bool),
        "voxel_door_turn_z_above_lintel_max_mask": (reason.reshape(shape) == "turn_z_above_door_lintel_max").astype(bool),
        "voxel_door_first_occ_too_high_mask": (reason.reshape(shape) == "first_upper_occupied_too_high_ceiling_like").astype(bool),
        "voxel_door_upper_observed_count_xy": np.clip(upper_obs, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_upper_total_active_count_xy": np.clip(upper_total_active, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_upper_occupied_count_xy": np.clip(upper_occ, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_upper_solid_count_xy": np.clip(upper_solid, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_upper_actual_occupied_count_xy": np.clip(upper_occ, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_upper_in_range_unknown_count_xy": np.clip(upper_in_range_unknown, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_upper_out_of_range_unknown_count_xy": np.clip(upper_out_of_range_unknown, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_upper_free_count_xy": np.clip(upper_free, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_effective_observed_count_xy": np.clip(obs_count, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_in_range_unknown_count_xy": np.clip(in_range_unknown_count, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_out_of_range_unknown_count_xy": np.clip(out_of_range_unknown_count, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_free_count_xy": np.clip(free_count, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_occupied_count_xy": np.clip(occ_count, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_observed_count_xy": np.clip(obs_count, 0, np.iinfo(np.uint16).max).reshape(shape).astype(np.uint16),
        "voxel_door_upper_solid_uses_in_range_unknown": np.zeros(shape, dtype=bool),
        "voxel_door_upper_solid_uses_out_of_range_unknown": np.ones(shape, dtype=bool),
    }

    for idx_col in np.flatnonzero(accepted)[:2048]:
        row = int(idx_col // width)
        col = int(idx_col % width)
        accepted_seed_evidence.append(
            _seed_ev(
                row,
                col,
                None if not np.isfinite(first_occ_z[idx_col]) else float(first_occ_z[idx_col]),
                int(lower_free[idx_col]),
                int(upper_occ[idx_col]),
                0,
                True,
                None,
                turn_z=None if not np.isfinite(turn_z[idx_col]) else float(turn_z[idx_col]),
                free_centroid=None if not np.isfinite(free_centroid[idx_col]) else float(free_centroid[idx_col]),
                occupied_centroid=None if not np.isfinite(occ_centroid[idx_col]) else float(occ_centroid[idx_col]),
                lower_free_ratio=float(lower_free_ratio[idx_col]),
                upper_occupied_ratio=float(upper_occ_ratio[idx_col]),
                upper_observed=int(upper_obs[idx_col]),
                upper_occupied=int(upper_occ[idx_col]),
            )
        )
    result = (seed, reason_map, lower_free_xy, top_occ_xy, unknown_tail_xy, first_occ_z_xy, rejected_seed_reasons, accepted_seed_evidence)
    return (*result, debug_maps) if bool(return_debug) else result


def _filter_seed_components_by_bbox_extent(
    seed_mask: np.ndarray,
    reason_map: np.ndarray,
    cfg: VoxelDoorDetectorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    seed = np.asarray(seed_mask, dtype=bool)
    reasons = np.asarray(reason_map, dtype=np.uint8)
    min_extent = max(1, int(getattr(cfg, "min_seed_component_bbox_extent_cells", 1)))
    labels, count = ndimage.label(seed, structure=conn(int(cfg.seed_connectivity)))
    reject_map = np.zeros_like(seed, dtype=bool)
    rejected_components: list[dict[str, object]] = []
    if min_extent <= 1 or int(count) <= 0:
        return seed.astype(bool), reasons.astype(np.uint8), labels.astype(np.int32), {
            "voxel_door_seed_component_bbox_filter_applied": False,
            "voxel_door_seed_component_bbox_min_extent_cells": int(min_extent),
            "voxel_door_seed_component_bbox_rejected_cells": 0,
            "voxel_door_seed_component_bbox_rejected_components": [],
            "voxel_door_seed_component_bbox_reject_map": reject_map,
        }
    for component_id in range(1, int(count) + 1):
        component = labels == int(component_id)
        if not np.any(component):
            continue
        rr, cc = np.nonzero(component)
        bbox_h = int(rr.max() - rr.min() + 1)
        bbox_w = int(cc.max() - cc.min() + 1)
        if bbox_h >= min_extent or bbox_w >= min_extent:
            continue
        reject_map |= component
        rejected_components.append(
            {
                "component_id": int(component_id),
                "bbox_rc": [int(rr.min()), int(cc.min()), int(rr.max() + 1), int(cc.max() + 1)],
                "bbox_height_cells": int(bbox_h),
                "bbox_width_cells": int(bbox_w),
                "area_cells": int(rr.size),
            }
        )
    if not np.any(reject_map):
        return seed.astype(bool), reasons.astype(np.uint8), labels.astype(np.int32), {
            "voxel_door_seed_component_bbox_filter_applied": False,
            "voxel_door_seed_component_bbox_min_extent_cells": int(min_extent),
            "voxel_door_seed_component_bbox_rejected_cells": 0,
            "voxel_door_seed_component_bbox_rejected_components": [],
            "voxel_door_seed_component_bbox_reject_map": reject_map,
        }
    filtered_seed = seed.copy()
    filtered_reason = reasons.copy()
    filtered_seed[reject_map] = False
    filtered_reason[reject_map] = _seed_reject_code("seed_component_bbox_too_small")
    filtered_labels, _filtered_count = ndimage.label(filtered_seed, structure=conn(int(cfg.seed_connectivity)))
    return filtered_seed.astype(bool), filtered_reason.astype(np.uint8), filtered_labels.astype(np.int32), {
        "voxel_door_seed_component_bbox_filter_applied": True,
        "voxel_door_seed_component_bbox_min_extent_cells": int(min_extent),
        "voxel_door_seed_component_bbox_rejected_cells": int(np.count_nonzero(reject_map)),
        "voxel_door_seed_component_bbox_rejected_components": rejected_components,
        "voxel_door_seed_component_bbox_reject_map": reject_map.astype(bool),
    }


def _filter_seed_by_navigation_free(
    seed_mask: np.ndarray,
    reason_map: np.ndarray,
    navigation_free_mask: np.ndarray | None,
    cfg: VoxelDoorDetectorConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    seed = np.asarray(seed_mask, dtype=bool)
    reasons = np.asarray(reason_map, dtype=np.uint8)
    reject_map = np.zeros_like(seed, dtype=bool)
    required = bool(getattr(cfg, "require_navigation_free_seed", True))
    if not required:
        return seed.astype(bool), reasons.astype(np.uint8), {
            "voxel_door_seed_requires_navigation_free": False,
            "voxel_door_seed_navigation_free_available": bool(navigation_free_mask is not None),
            "voxel_door_seed_navigation_free_filter_applied": False,
            "voxel_door_seed_navigation_free_rejected_cells": 0,
            "voxel_door_seed_navigation_free_reject_map": reject_map,
        }
    if navigation_free_mask is None:
        return seed.astype(bool), reasons.astype(np.uint8), {
            "voxel_door_seed_requires_navigation_free": True,
            "voxel_door_seed_navigation_free_available": False,
            "voxel_door_seed_navigation_free_filter_applied": False,
            "voxel_door_seed_navigation_free_rejected_cells": 0,
            "voxel_door_seed_navigation_free_reject_map": reject_map,
        }
    nav_free = np.asarray(navigation_free_mask, dtype=bool)
    if nav_free.shape != seed.shape:
        raise ValueError("navigation_free_mask must match voxel door seed shape")
    reject_map = seed & ~nav_free
    if not np.any(reject_map):
        return seed.astype(bool), reasons.astype(np.uint8), {
            "voxel_door_seed_requires_navigation_free": True,
            "voxel_door_seed_navigation_free_available": True,
            "voxel_door_seed_navigation_free_filter_applied": False,
            "voxel_door_seed_navigation_free_rejected_cells": 0,
            "voxel_door_seed_navigation_free_reject_map": reject_map,
        }
    filtered_seed = seed.copy()
    filtered_reasons = reasons.copy()
    filtered_seed[reject_map] = False
    filtered_reasons[reject_map] = _seed_reject_code("not_navigation_free")
    return filtered_seed.astype(bool), filtered_reasons.astype(np.uint8), {
        "voxel_door_seed_requires_navigation_free": True,
        "voxel_door_seed_navigation_free_available": True,
        "voxel_door_seed_navigation_free_filter_applied": True,
        "voxel_door_seed_navigation_free_rejected_cells": int(np.count_nonzero(reject_map)),
        "voxel_door_seed_navigation_free_reject_map": reject_map.astype(bool),
    }


def classify_voxel_door_seed_column(
    z_state_col: np.ndarray,
    z_centers_m: np.ndarray,
    active_z_indices: np.ndarray,
    cfg: VoxelDoorDetectorConfig,
    *,
    row: int = 0,
    col: int = 0,
    sensor_range_count_col: np.ndarray | None = None,
) -> VoxelDoorSeedEvidence:
    method = str(getattr(cfg, "seed_method", "centroid_ratio") or "centroid_ratio").strip().lower()
    if method in {"centroid_ratio", "centroid", "ratio"}:
        return _classify_centroid_ratio_seed_column(
            z_state_col,
            z_centers_m,
            active_z_indices,
            cfg,
            row=row,
            col=col,
            sensor_range_count_col=sensor_range_count_col,
        )
    states = np.asarray(z_state_col, dtype=np.uint8).reshape(-1)
    centers = np.asarray(z_centers_m, dtype=np.float32).reshape(-1)
    idxs = np.asarray(active_z_indices, dtype=np.int32).reshape(-1)
    idxs = idxs[(idxs >= 0) & (idxs < states.size) & (centers[idxs] >= float(cfg.z_scan_min_m))]
    if idxs.size == 0:
        return _seed_ev(row, col, None, 0, 0, 0, False, "no_active_z_bins")
    pos = 0
    lower_free = 0
    first_occ_z = None
    while pos < len(idxs) and int(states[int(idxs[pos])]) == int(VOXEL_FREE):
        lower_free += 1
        pos += 1
    if lower_free < int(cfg.min_lower_free_cells):
        return _seed_ev(row, col, None, lower_free, 0, 0, False, "lower_free_cells_too_few")
    if pos >= len(idxs):
        return _seed_ev(row, col, None, lower_free, 0, 0, False, "no_top_occupied")

    state = int(states[int(idxs[pos])])
    if state == int(VOXEL_UNKNOWN):
        return _seed_ev(row, col, None, lower_free, 0, 0, False, "unknown_before_top_occupied")
    if state == int(VOXEL_CONFLICT):
        return _seed_ev(row, col, None, lower_free, 0, 0, False, "conflict_before_top_occupied")
    if state != int(VOXEL_OCCUPIED):
        return _seed_ev(row, col, None, lower_free, 0, 0, False, "expected_top_occupied")
    first_occ_z = float(centers[int(idxs[pos])])
    if first_occ_z < float(cfg.top_occupied_min_z_m):
        return _seed_ev(row, col, first_occ_z, lower_free, 0, 0, False, "top_occupied_too_low")

    top_occ = 0
    while pos < len(idxs) and int(states[int(idxs[pos])]) == int(VOXEL_OCCUPIED):
        top_occ += 1
        pos += 1
    if top_occ < int(cfg.min_top_occupied_cells):
        return _seed_ev(row, col, first_occ_z, lower_free, top_occ, 0, False, "top_occupied_run_too_short")
    if pos >= len(idxs):
        if bool(cfg.allow_end_of_active_range_after_top_occupied):
            return _seed_ev(row, col, first_occ_z, lower_free, top_occ, 0, True, None)
        return _seed_ev(row, col, first_occ_z, lower_free, top_occ, 0, False, "missing_unknown_tail")

    if int(states[int(idxs[pos])]) != int(VOXEL_UNKNOWN):
        return _seed_ev(row, col, first_occ_z, lower_free, top_occ, 0, False, "non_unknown_after_top_occupied")
    if not bool(cfg.allow_unknown_tail_after_top_occupied):
        return _seed_ev(row, col, first_occ_z, lower_free, top_occ, 0, False, "unknown_tail_not_allowed")
    unknown_tail = 0
    while pos < len(idxs) and int(states[int(idxs[pos])]) == int(VOXEL_UNKNOWN):
        unknown_tail += 1
        pos += 1
    if pos >= len(idxs):
        return _seed_ev(row, col, first_occ_z, lower_free, top_occ, unknown_tail, True, None)
    state = int(states[int(idxs[pos])])
    if state == int(VOXEL_FREE) and bool(cfg.reject_free_after_unknown_tail):
        return _seed_ev(row, col, first_occ_z, lower_free, top_occ, unknown_tail, False, "free_after_unknown_tail")
    if state == int(VOXEL_OCCUPIED) and bool(cfg.reject_occupied_after_unknown_tail):
        return _seed_ev(row, col, first_occ_z, lower_free, top_occ, unknown_tail, False, "occupied_after_unknown_tail")
    if state == int(VOXEL_CONFLICT) and bool(cfg.reject_conflict_in_seed_pattern):
        return _seed_ev(row, col, first_occ_z, lower_free, top_occ, unknown_tail, False, "conflict_after_unknown_tail")
    return _seed_ev(row, col, first_occ_z, lower_free, top_occ, unknown_tail, False, "state_after_unknown_tail")


def _classify_centroid_ratio_seed_column(
    z_state_col: np.ndarray,
    z_centers_m: np.ndarray,
    active_z_indices: np.ndarray,
    cfg: VoxelDoorDetectorConfig,
    *,
    row: int = 0,
    col: int = 0,
    sensor_range_count_col: np.ndarray | None = None,
) -> VoxelDoorSeedEvidence:
    states = np.asarray(z_state_col, dtype=np.uint8).reshape(-1, 1, 1)
    sensor = None if sensor_range_count_col is None else np.asarray(sensor_range_count_col).reshape(-1, 1, 1)
    (
        seed,
        reason_map,
        lower_free,
        upper_occ,
        unknown_tail,
        first_occ_z,
        _counts,
        _accepted,
        maps,
    ) = classify_voxel_door_seeds_centroid_ratio_vectorized(
        states,
        z_centers_m,
        active_z_indices,
        cfg,
        shape=(1, 1),
        return_debug=True,
        sensor_range_count=sensor,
    )
    accepted = bool(seed[0, 0])
    code = int(reason_map[0, 0])
    reason = None if accepted else _seed_reject_reason_from_code(code)
    first = float(first_occ_z[0, 0]) if np.isfinite(first_occ_z[0, 0]) else None
    turn = float(maps["voxel_door_turn_z_estimate_xy"][0, 0]) if np.isfinite(maps["voxel_door_turn_z_estimate_xy"][0, 0]) else None
    free_centroid = float(maps["voxel_door_free_centroid_z_xy"][0, 0]) if np.isfinite(maps["voxel_door_free_centroid_z_xy"][0, 0]) else None
    occ_centroid = float(maps["voxel_door_occupied_centroid_z_xy"][0, 0]) if np.isfinite(maps["voxel_door_occupied_centroid_z_xy"][0, 0]) else None
    return _seed_ev(
        int(row),
        int(col),
        first,
        int(lower_free[0, 0]),
        int(upper_occ[0, 0]),
        int(unknown_tail[0, 0]),
        accepted,
        reason,
        turn_z=turn,
        free_centroid=free_centroid,
        occupied_centroid=occ_centroid,
        lower_free_ratio=float(maps["voxel_door_lower_free_ratio_xy"][0, 0]),
        upper_occupied_ratio=float(maps["voxel_door_upper_occupied_ratio_observed_xy"][0, 0]),
        upper_observed=int(maps["voxel_door_upper_observed_count_xy"][0, 0]),
        upper_occupied=int(maps["voxel_door_upper_occupied_count_xy"][0, 0]),
    )


def _build_door_seed_clusters(
    seed_mask: np.ndarray,
    component_map: np.ndarray,
    *,
    resolution_m: float,
    cfg: VoxelDoorDetectorConfig,
    barrier_map: np.ndarray | None = None,
) -> tuple[list[DoorSeedCluster], np.ndarray, dict[str, object]]:
    seed = np.asarray(seed_mask, dtype=bool)
    labels = np.asarray(component_map, dtype=np.int32)
    barrier = np.zeros_like(seed, dtype=bool) if barrier_map is None else np.asarray(barrier_map, dtype=bool)
    shape = seed.shape
    component_ids = [int(v) for v in np.unique(labels) if int(v) > 0]
    if not component_ids:
        empty = np.zeros(shape, dtype=np.int32)
        return [], empty, {
            "voxel_door_seed_component_count": 0,
            "voxel_door_seed_cluster_count": 0,
            "voxel_door_seed_cluster_map": empty,
            "voxel_door_seed_cluster_merge_edges": [],
            "voxel_door_seed_cluster_merge_reason_counts": {},
        }

    geoms = {cid: _seed_component_geom(labels == cid, cid, resolution_m=float(resolution_m)) for cid in component_ids}
    parent = {cid: cid for cid in component_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edges: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    close_radius = max(0, int(getattr(cfg, "seed_cluster_morph_close_radius_cells", 0)))
    if close_radius > 0:
        closed = ndimage.binary_closing(seed, structure=disk(close_radius)).astype(bool)
        closed_labels, _closed_count = ndimage.label(closed, structure=conn(8))
        closed_component: dict[int, int] = {}
        for cid in component_ids:
            values = closed_labels[labels == int(cid)]
            values = values[values > 0]
            if values.size:
                counts = np.bincount(values.astype(np.int32))
                closed_component[int(cid)] = int(np.argmax(counts))
        for idx, a in enumerate(component_ids):
            for b in component_ids[idx + 1 :]:
                if int(closed_component.get(int(a), -1)) <= 0 or closed_component.get(int(a)) != closed_component.get(int(b)):
                    continue
                if not _seed_components_fit_merge_limits(geoms[a], geoms[b], cfg, barrier):
                    continue
                union(a, b)
                edges.append({"a": int(a), "b": int(b), "reason": "morph_close"})
                reason_counts["morph_close"] += 1
    for idx, a in enumerate(component_ids):
        for b in component_ids[idx + 1 :]:
            reason = _seed_components_merge_reason(geoms[a], geoms[b], cfg, barrier)
            if reason is None:
                continue
            union(a, b)
            edges.append({"a": int(a), "b": int(b), "reason": str(reason)})
            reason_counts[str(reason)] += 1

    groups: dict[int, list[int]] = {}
    for cid in component_ids:
        groups.setdefault(find(cid), []).append(cid)

    cluster_map = np.zeros(shape, dtype=np.int32)
    clusters: list[DoorSeedCluster] = []
    for cluster_id, cids in enumerate(sorted(groups.values(), key=lambda ids: min(ids)), start=1):
        cells: list[tuple[int, int]] = []
        mask = np.zeros(shape, dtype=bool)
        for cid in sorted(cids):
            rr, cc = np.nonzero(labels == cid)
            for r, c in zip(rr, cc):
                cell = (int(r), int(c))
                cells.append(cell)
                mask[cell] = True
        cells = sorted(set(cells))
        cluster_map[mask] = int(cluster_id)
        center, major, minor, residual, thickness_m, length_m, bbox = _fit_seed_cells(cells, shape, float(resolution_m))
        reject_reason = None
        if len(cells) < int(cfg.min_seed_component_cells):
            reject_reason = "seed_cluster_too_small"
        elif len(cells) > int(cfg.max_seed_component_cells):
            reject_reason = "seed_cluster_too_large"
        elif thickness_m > float(cfg.max_component_thickness_m) + 1e-6:
            reject_reason = "seed_cluster_too_thick"
        clusters.append(
            DoorSeedCluster(
                cluster_id=int(cluster_id),
                component_ids=[int(v) for v in sorted(cids)],
                seed_cells=cells,
                mask=mask,
                bbox_rc=bbox,
                center_rc=(float(center[0]), float(center[1])),
                major_dir_rc=(float(major[0]), float(major[1])),
                minor_dir_rc=(float(minor[0]), float(minor[1])),
                line_fit_residual_cells=float(residual),
                thickness_m=float(thickness_m),
                length_m=float(length_m),
                accepted_for_completion=reject_reason is None,
                reject_reason=reject_reason,
                debug={"component_count": int(len(cids))},
            )
        )

    return clusters, cluster_map, {
        "voxel_door_seed_component_count": int(len(component_ids)),
        "voxel_door_seed_cluster_count": int(len(clusters)),
        "voxel_door_seed_cluster_map": cluster_map.astype(np.int32),
        "voxel_door_seed_clusters": [cluster.to_dict() for cluster in clusters],
        "voxel_door_seed_cluster_merge_edges": edges,
        "voxel_door_seed_cluster_merge_reason_counts": dict(reason_counts),
    }


def _build_door_seed_groups_v17(
    clusters: Sequence[DoorSeedCluster],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    cfg: VoxelDoorDetectorConfig,
    barrier_map: np.ndarray | None = None,
) -> tuple[list[DoorSeedGroup], dict[str, object]]:
    groups: list[DoorSeedGroup] = [DoorSeedGroup.from_cluster(cluster) for cluster in clusters]
    pair_debug: list[dict[str, object]] = []
    pair_reason_counts: Counter[str] = Counter()
    if not bool(getattr(cfg, "enable_seed_pair_bridge_completion", True)):
        return groups, {
            "voxel_door_seed_group_count": int(len(groups)),
            "voxel_door_seed_groups": [group.to_dict() for group in groups],
            "voxel_door_seed_pair_group_count": 0,
            "voxel_door_seed_pair_group_edges": [],
            "voxel_door_seed_pair_group_reason_counts": {"disabled": 1},
        }

    barrier = np.zeros(shape, dtype=bool) if barrier_map is None else np.asarray(barrier_map, dtype=bool)
    accepted_clusters = [cluster for cluster in clusters if bool(cluster.accepted_for_completion)]
    max_dist_cells = float(getattr(cfg, "seed_pair_max_center_distance_m", 1.80)) / max(float(resolution_m), 1e-9)
    max_perp = float(getattr(cfg, "seed_pair_max_perpendicular_gap_cells", 2))
    max_angle = float(getattr(cfg, "seed_pair_max_angle_deg", 25.0))
    next_group_id = max([int(cluster.cluster_id) for cluster in clusters] + [0]) + 1
    pair_candidates: list[tuple[float, DoorSeedGroup, dict[str, object], np.ndarray]] = []
    all_seed_mask = _cells_to_mask(
        [cell for cluster in accepted_clusters for cell in cluster.seed_cells],
        shape,
    )
    if bool(getattr(cfg, "enable_seed_same_line_gap_bridge", True)):
        for idx, a in enumerate(accepted_clusters):
            for b in accepted_clusters[idx + 1 :]:
                candidate = _same_line_gap_bridge_group(
                    a,
                    b,
                    group_id=int(next_group_id),
                    shape=shape,
                    resolution_m=float(resolution_m),
                    cfg=cfg,
                    barrier=barrier,
                )
                if candidate is None:
                    continue
                score, group, edge, line = candidate
                pair_candidates.append((score, group, edge, line))
                pair_reason_counts["same_line_gap_bridge_candidate"] += 1
                next_group_id += 1
    for idx, a in enumerate(accepted_clusters):
        ca = np.asarray(a.center_rc, dtype=np.float32)
        major_a = np.asarray(a.major_dir_rc, dtype=np.float32)
        for b in accepted_clusters[idx + 1 :]:
            cb = np.asarray(b.center_rc, dtype=np.float32)
            delta = cb - ca
            dist = float(np.linalg.norm(delta))
            edge = {"a": int(a.cluster_id), "b": int(b.cluster_id), "distance_cells": float(dist)}
            if dist <= 1e-6 or dist > max_dist_cells:
                pair_reason_counts["distance"] += 1
                edge["reject_reason"] = "distance"
                pair_debug.append(edge)
                continue
            major = (delta / dist).astype(np.float32)
            minor = np.asarray([-major[1], major[0]], dtype=np.float32)
            major_b = np.asarray(b.major_dir_rc, dtype=np.float32)
            angle_a = float(np.degrees(np.arccos(min(1.0, max(-1.0, abs(float(np.dot(major, major_a))))))))
            angle_b = float(np.degrees(np.arccos(min(1.0, max(-1.0, abs(float(np.dot(major, major_b))))))))
            if len(a.seed_cells) > 1 and angle_a > max_angle:
                pair_reason_counts["angle"] += 1
                edge.update({"reject_reason": "angle", "angle_a_deg": angle_a, "angle_b_deg": angle_b})
                pair_debug.append(edge)
                continue
            if len(b.seed_cells) > 1 and angle_b > max_angle:
                pair_reason_counts["angle"] += 1
                edge.update({"reject_reason": "angle", "angle_a_deg": angle_a, "angle_b_deg": angle_b})
                pair_debug.append(edge)
                continue
            cells = sorted(set([*a.seed_cells, *b.seed_cells]))
            pts = np.asarray(cells, dtype=np.float32)
            residual = np.abs(np.dot(pts - ((ca + cb) * 0.5)[None, :], minor)) if pts.size else np.asarray([], dtype=np.float32)
            max_residual = float(np.max(residual)) if residual.size else 0.0
            if max_residual > max_perp + 0.5:
                pair_reason_counts["perpendicular_gap"] += 1
                edge.update({"reject_reason": "perpendicular_gap", "max_residual_cells": max_residual})
                pair_debug.append(edge)
                continue
            line = rasterize_line(ca, cb, shape)
            seed_band = dilate(_cells_to_mask(cells, shape), max(0, int(getattr(cfg, "seed_pair_bridge_gap_tolerance_cells", 2))))
            if np.any(line & barrier & ~seed_band):
                pair_reason_counts["barrier_between_seed_pair"] += 1
                edge["reject_reason"] = "barrier_between_seed_pair"
                pair_debug.append(edge)
                continue
            mask = _cells_to_mask(cells, shape)
            bbox = _bbox_from_cells(cells, shape)
            length_m = float((dist + 1.0) * float(resolution_m))
            thickness_m = float((2.0 * max_residual + 1.0) * float(resolution_m))
            pair_seed_mask = _cells_to_mask(cells, shape)
            pair_seed_band = dilate(pair_seed_mask, max(0, int(getattr(cfg, "seed_pair_bridge_gap_tolerance_cells", 2))))
            crosses_other_seed = bool(np.any(line & all_seed_mask & ~pair_seed_band))
            collinearity = max(0.0, 1.0 - (float(angle_a) + float(angle_b)) / max(1e-6, 2.0 * max(max_angle, 1.0)))
            seed_density = float(len(cells)) / float(max(1, int(np.count_nonzero(line))))
            length_penalty = float(dist) / float(max(max_dist_cells, 1e-6))
            score = float(
                2.0 * collinearity
                + 1.5 * min(1.0, seed_density)
                - 2.0 * (1.0 if crosses_other_seed else 0.0)
                - 1.0 * length_penalty
            )
            group = DoorSeedGroup(
                group_id=int(next_group_id),
                group_kind="seed_pair_bridge",
                source_cluster_ids=[int(a.cluster_id), int(b.cluster_id)],
                component_ids=sorted({int(v) for v in [*a.component_ids, *b.component_ids]}),
                seed_cells=cells,
                mask=mask,
                bbox_rc=bbox,
                center_rc=(float(((ca + cb) * 0.5)[0]), float(((ca + cb) * 0.5)[1])),
                major_dir_rc=(float(major[0]), float(major[1])),
                minor_dir_rc=(float(minor[0]), float(minor[1])),
                line_fit_residual_cells=max_residual,
                thickness_m=thickness_m,
                length_m=length_m,
                accepted_for_completion=True,
                reject_reason=None,
                debug={
                    "source": "seed_pair_bridge",
                    "source_cluster_ids": [int(a.cluster_id), int(b.cluster_id)],
                    "distance_cells": float(dist),
                    "angle_a_deg": float(angle_a),
                    "angle_b_deg": float(angle_b),
                    "max_residual_cells": float(max_residual),
                    "seed_pair_bridge_score": float(score),
                    "seed_pair_crosses_other_seed": bool(crosses_other_seed),
                },
            )
            edge.update(
                {
                    "candidate_score": float(score),
                    "seed_density": float(seed_density),
                    "collinearity": float(collinearity),
                    "length_penalty": float(length_penalty),
                    "crosses_other_seed": bool(crosses_other_seed),
                }
            )
            pair_candidates.append((score, group, edge, line.astype(bool)))
            pair_reason_counts["candidate"] += 1
            next_group_id += 1
    used_clusters: set[int] = set()
    accepted_pair_lines: list[np.ndarray] = []
    for score, group, edge, line in sorted(pair_candidates, key=lambda item: item[0], reverse=True):
        source_ids = [int(v) for v in group.source_cluster_ids]
        if any(cluster_id in used_clusters for cluster_id in source_ids):
            edge["reject_reason"] = "seed_pair_cluster_already_matched"
            edge["accepted"] = False
            pair_debug.append(edge)
            pair_reason_counts["seed_pair_cluster_already_matched"] += 1
            continue
        if any(bool(np.any(line & old_line)) for old_line in accepted_pair_lines):
            edge["reject_reason"] = "seed_pair_line_intersection"
            edge["accepted"] = False
            pair_debug.append(edge)
            pair_reason_counts["seed_pair_line_intersection"] += 1
            continue
        groups.append(group)
        used_clusters.update(source_ids)
        accepted_pair_lines.append(line)
        edge.update({"accepted": True, "group_id": int(group.group_id), "selected_score": float(score)})
        pair_debug.append(edge)
        pair_reason_counts["accepted"] += 1
    return groups, {
        "voxel_door_seed_group_count": int(len(groups)),
        "voxel_door_seed_groups": [group.to_dict() for group in groups],
        "voxel_door_seed_pair_group_count": int(sum(1 for group in groups if group.group_kind == "seed_pair_bridge")),
        "voxel_door_seed_same_line_gap_bridge_count": int(sum(1 for group in groups if group.group_kind == "same_line_gap_bridge")),
        "voxel_door_seed_pair_group_edges": pair_debug,
        "voxel_door_seed_pair_group_reason_counts": dict(pair_reason_counts),
    }


def _same_line_gap_bridge_group(
    a: DoorSeedCluster,
    b: DoorSeedCluster,
    *,
    group_id: int,
    shape: tuple[int, int],
    resolution_m: float,
    cfg: VoxelDoorDetectorConfig,
    barrier: np.ndarray,
) -> tuple[float, DoorSeedGroup, dict[str, object], np.ndarray] | None:
    major_a = _unit(np.asarray(a.major_dir_rc, dtype=np.float32))
    major_b = _unit(np.asarray(b.major_dir_rc, dtype=np.float32))
    if major_a is None or major_b is None:
        return None
    angle = _unsigned_angle_deg(major_a, major_b)
    max_angle = float(getattr(cfg, "seed_same_line_gap_bridge_max_angle_deg", 10.0))
    if angle > max_angle + 1e-6:
        return None
    if float(np.dot(major_a, major_b)) < 0.0:
        major_b = -major_b
    major = _unit(major_a + major_b)
    if major is None:
        major = major_a
    minor = np.asarray([-major[1], major[0]], dtype=np.float32)
    cells = sorted({(int(r), int(c)) for r, c in [*a.seed_cells, *b.seed_cells]})
    pts = np.asarray(cells, dtype=np.float32)
    if pts.size == 0:
        return None
    center = np.mean(pts, axis=0).astype(np.float32)
    rel = pts - center[None, :]
    along = np.dot(rel, major)
    orth = np.dot(rel, minor)
    max_perp = float(getattr(cfg, "seed_same_line_gap_bridge_max_perpendicular_cells", 1))
    if orth.size and float(np.max(orth) - np.min(orth)) > max_perp + 1.0 + 1e-6:
        return None
    a_pts = np.asarray(a.seed_cells, dtype=np.float32)
    b_pts = np.asarray(b.seed_cells, dtype=np.float32)
    a_along = np.dot(a_pts - center[None, :], major)
    b_along = np.dot(b_pts - center[None, :], major)
    if not a_along.size or not b_along.size:
        return None
    if float(np.mean(a_along)) <= float(np.mean(b_along)):
        left_max = float(np.max(a_along))
        right_min = float(np.min(b_along))
    else:
        left_max = float(np.max(b_along))
        right_min = float(np.min(a_along))
    gap_cells = max(0.0, right_min - left_max - 1.0)
    max_gap = float(getattr(cfg, "seed_same_line_gap_bridge_max_gap_cells", 28))
    if gap_cells > max_gap + 1e-6:
        return None
    length_m = float((float(np.max(along)) - float(np.min(along)) + 1.0) * float(resolution_m))
    max_length_m = float(getattr(cfg, "seed_same_line_gap_bridge_max_length_m", 4.0))
    if max_length_m > 0.0 and length_m > max_length_m + 1e-6:
        return None
    end_a = pts[int(np.argmin(along))]
    end_b = pts[int(np.argmax(along))]
    line_mask = rasterize_line(end_a, end_b, shape).astype(bool)
    seed_band = dilate(_cells_to_mask(cells, shape), max(1, int(np.ceil(max_gap))))
    if np.any(line_mask & np.asarray(barrier, dtype=bool) & ~seed_band):
        return None
    mask = _cells_to_mask(cells, shape)
    bbox = _bbox_from_cells(cells, shape)
    thickness_cells = float(np.max(orth) - np.min(orth) + 1.0) if orth.size else 1.0
    score = float(20.0 - gap_cells - 0.2 * angle - 0.1 * thickness_cells)
    group = DoorSeedGroup(
        group_id=int(group_id),
        group_kind="same_line_gap_bridge",
        source_cluster_ids=[int(a.cluster_id), int(b.cluster_id)],
        component_ids=sorted({int(v) for v in [*a.component_ids, *b.component_ids]}),
        seed_cells=cells,
        mask=mask,
        bbox_rc=bbox,
        center_rc=(float(center[0]), float(center[1])),
        major_dir_rc=(float(major[0]), float(major[1])),
        minor_dir_rc=(float(minor[0]), float(minor[1])),
        line_fit_residual_cells=float(max(0.0, 0.5 * (thickness_cells - 1.0))),
        thickness_m=float(thickness_cells * float(resolution_m)),
        length_m=float(length_m),
        accepted_for_completion=True,
        reject_reason=None,
        debug={
            "source": "same_line_gap_bridge",
            "source_cluster_ids": [int(a.cluster_id), int(b.cluster_id)],
            "same_line_gap_cells": float(gap_cells),
            "same_line_angle_deg": float(angle),
            "same_line_thickness_cells": float(thickness_cells),
            "same_line_length_m": float(length_m),
        },
    )
    edge = {
        "a": int(a.cluster_id),
        "b": int(b.cluster_id),
        "reason": "same_line_gap_bridge",
        "candidate_score": float(score),
        "same_line_gap_cells": float(gap_cells),
        "same_line_angle_deg": float(angle),
        "same_line_thickness_cells": float(thickness_cells),
        "same_line_length_m": float(length_m),
    }
    return score, group, edge, line_mask


def _seed_component_geom(mask: np.ndarray, component_id: int, resolution_m: float) -> dict[str, object]:
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(mask))]
    center, major, minor, residual, thickness_m, length_m, bbox = _fit_seed_cells(cells, mask.shape, float(resolution_m))
    return {
        "component_id": int(component_id),
        "cells": cells,
        "center": center,
        "major": major,
        "minor": minor,
        "residual": float(residual),
        "thickness_m": float(thickness_m),
        "length_cells": float(length_m / max(float(resolution_m), 1e-9)),
        "resolution_m": float(resolution_m),
        "bbox": bbox,
    }


def _fit_seed_cells(
    cells: Sequence[tuple[int, int]],
    shape: tuple[int, int],
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, tuple[int, int, int, int]]:
    if not cells:
        center = np.asarray([0.0, 0.0], dtype=np.float32)
        major = np.asarray([0.0, 1.0], dtype=np.float32)
        minor = np.asarray([-1.0, 0.0], dtype=np.float32)
        return center, major, minor, 0.0, 0.0, 0.0, (0, 0, 0, 0)
    pts = np.asarray(cells, dtype=np.float32)
    center = np.mean(pts, axis=0).astype(np.float32)
    if pts.shape[0] >= 2:
        centered = pts - center[None, :]
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        major = vt[0].astype(np.float32)
        norm = float(np.linalg.norm(major))
        major = np.asarray([0.0, 1.0], dtype=np.float32) if norm <= 1e-6 else major / norm
    else:
        major = np.asarray([0.0, 1.0], dtype=np.float32)
    minor = np.asarray([-major[1], major[0]], dtype=np.float32)
    rel = pts - center[None, :]
    residual = np.abs(np.dot(rel, minor))
    along = np.dot(rel, major)
    thickness_m = float((float(np.max(residual)) * 2.0 + 1.0) * float(resolution_m)) if residual.size else float(resolution_m)
    length_m = float((float(np.max(along)) - float(np.min(along)) + 1.0) * float(resolution_m)) if along.size else float(resolution_m)
    rows = pts[:, 0].astype(np.int32)
    cols = pts[:, 1].astype(np.int32)
    bbox = (
        max(0, int(rows.min())),
        min(int(shape[0]) - 1, int(rows.max())),
        max(0, int(cols.min())),
        min(int(shape[1]) - 1, int(cols.max())),
    )
    return center, major, minor, float(np.max(residual)) if residual.size else 0.0, thickness_m, length_m, bbox


def _seed_components_merge_reason(a: Mapping[str, object], b: Mapping[str, object], cfg: VoxelDoorDetectorConfig, barrier: np.ndarray) -> str | None:
    if _expanded_bboxes_intersect(a["bbox"], b["bbox"], int(cfg.seed_cluster_merge_distance_cells)) and _seed_components_fit_merge_limits(a, b, cfg, barrier):  # type: ignore[arg-type]
        return "nearby_bbox"
    center_a = np.asarray(a["center"], dtype=np.float32)
    center_b = np.asarray(b["center"], dtype=np.float32)
    major_a = np.asarray(a["major"], dtype=np.float32)
    major_b = np.asarray(b["major"], dtype=np.float32)
    dot = min(1.0, max(-1.0, abs(float(np.dot(major_a, major_b)))))
    angle = float(np.degrees(np.arccos(dot)))
    if angle > float(cfg.seed_cluster_collinear_angle_deg):
        return None
    delta = center_b - center_a
    minor = np.asarray([-major_a[1], major_a[0]], dtype=np.float32)
    perpendicular = abs(float(np.dot(delta, minor)))
    along_gap = max(0.0, abs(float(np.dot(delta, major_a))) - 0.5 * (float(a["length_cells"]) + float(b["length_cells"])))
    if (
        perpendicular <= float(cfg.seed_cluster_max_perpendicular_gap_cells)
        and along_gap <= float(cfg.seed_cluster_max_along_gap_cells)
        and _seed_components_fit_merge_limits(a, b, cfg, barrier)
    ):
        return "collinear"
    return None


def _seed_components_fit_merge_limits(a: Mapping[str, object], b: Mapping[str, object], cfg: VoxelDoorDetectorConfig, barrier: np.ndarray) -> bool:
    cells = [tuple(v) for v in a.get("cells", [])] + [tuple(v) for v in b.get("cells", [])]  # type: ignore[arg-type]
    if not cells:
        return False
    resolution_m = float(a.get("resolution_m", b.get("resolution_m", 1.0)))
    center, _major, _minor, _residual, thickness_m, length_m, _bbox = _fit_seed_cells(cells, barrier.shape, resolution_m)
    if float(thickness_m) > float(cfg.max_component_thickness_m) + 1e-6:
        return False
    if float(length_m) > float(getattr(cfg, "seed_cluster_max_width_m", cfg.door_width_max_m)) + 1e-6:
        return False
    ca = np.rint(np.asarray(a.get("center", center), dtype=np.float32)).astype(np.int32)
    cb = np.rint(np.asarray(b.get("center", center), dtype=np.float32)).astype(np.int32)
    line = rasterize_line(ca, cb, barrier.shape)
    if np.any(line & np.asarray(barrier, dtype=bool)):
        seed_mask = _cells_to_mask(cells, barrier.shape)
        if np.any(line & np.asarray(barrier, dtype=bool) & ~dilate(seed_mask, 1)):
            return False
    return True


def _expanded_bboxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], radius: int) -> bool:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 + radius < br0 or br1 + radius < ar0 or ac1 + radius < bc0 or bc1 + radius < ac0)


def _door_orientation_candidates(
    cluster: DoorSeedCluster | DoorSeedGroup,
    cfg: VoxelDoorDetectorConfig,
    *,
    wall_clean: np.ndarray | None = None,
    free_clean: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    mode = str(getattr(cfg, "accepted_orientation_mode", "seed_major_only") or "seed_major_only").strip().lower()
    seed_major = _unit(np.asarray(cluster.major_dir_rc, dtype=np.float32))
    if seed_major is not None:
        candidates.append(("seed_major", seed_major.astype(np.float32)))

    if bool(getattr(cfg, "allow_axis_orientation_if_aligned_with_seed", True)) and seed_major is not None:
        axis = _nearest_axis(seed_major)
        angle = _unsigned_angle_deg(axis, seed_major)
        if angle <= float(getattr(cfg, "axis_orientation_max_angle_to_seed_deg", 15.0)):
            candidates.append(("axis_snapped_from_seed_major", axis.astype(np.float32)))

    if (
        bool(getattr(cfg, "allow_wall_pair_orientation_if_aligned_with_seed", True))
        and bool(getattr(cfg, "infer_orientation_from_wall_pairs", True))
        and wall_clean is not None
        and seed_major is not None
    ):
        for source, vec in _infer_directions_from_nearby_wall_pairs(cluster, np.asarray(wall_clean, dtype=bool)):
            unit = _unit(vec)
            if unit is not None and _unsigned_angle_deg(unit, seed_major) <= float(getattr(cfg, "wall_pair_orientation_max_angle_to_seed_deg", 20.0)):
                candidates.append(("wall_pair_aligned_with_seed", unit.astype(np.float32)))

    if (
        not bool(getattr(cfg, "local_free_neck_orientation_debug_only", True))
        and bool(getattr(cfg, "infer_orientation_from_local_free_neck", True))
        and free_clean is not None
        and seed_major is not None
    ):
        for source, vec in _infer_directions_from_local_free_neck(cluster, np.asarray(free_clean, dtype=bool)):
            unit = _unit(vec)
            if unit is not None and _unsigned_angle_deg(unit, seed_major) <= float(getattr(cfg, "wall_pair_orientation_max_angle_to_seed_deg", 20.0)):
                candidates.append((source, unit.astype(np.float32)))

    if "legacy" in mode:
        legacy_mode = str(cfg.completion_orientation_mode or "pca_plus_axis").strip().lower()
        if "axis" in legacy_mode:
            candidates.extend(
                [
                    ("axis_h_legacy", np.asarray([0.0, 1.0], dtype=np.float32)),
                    ("axis_v_legacy", np.asarray([1.0, 0.0], dtype=np.float32)),
                ]
            )
        if bool(cfg.allow_diagonal_orientation_candidates) or "diagonal" in legacy_mode:
            scale = float(1.0 / np.sqrt(2.0))
            candidates.extend(
                [
                    ("diag_down_legacy", np.asarray([scale, scale], dtype=np.float32)),
                    ("diag_up_legacy", np.asarray([scale, -scale], dtype=np.float32)),
                ]
            )
    out: list[tuple[str, np.ndarray]] = []
    seen: list[np.ndarray] = []
    for source, vec in candidates:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            continue
        unit = vec / norm
        if any(abs(float(np.dot(unit, old))) > 0.98 for old in seen):
            continue
        seen.append(unit)
        out.append((source, unit.astype(np.float32)))
    return out


def is_seed_group_extensible(group: DoorSeedGroup, cfg: VoxelDoorDetectorConfig, *, resolution_m: float) -> tuple[bool, str | None, dict[str, object]]:
    seed_count = int(len(group.seed_cells))
    length_cells = float(group.length_m) / max(float(resolution_m), 1e-9)
    thickness_cells = max(1.0, float(group.thickness_m) / max(float(resolution_m), 1e-9))
    elongation = float(length_cells / max(thickness_cells, 1e-6))
    residual = float(group.line_fit_residual_cells)
    debug = {
        "seed_group_extensible_checked": True,
        "seed_group_seed_count": int(seed_count),
        "seed_group_line_length_cells": float(length_cells),
        "seed_group_elongation": float(elongation),
        "seed_group_line_fit_residual_cells": float(residual),
        "min_seed_cells_for_accepted_extension": int(getattr(cfg, "min_seed_cells_for_accepted_extension", 3)),
        "min_seed_line_length_cells_for_accepted_extension": int(getattr(cfg, "min_seed_line_length_cells_for_accepted_extension", 3)),
        "min_seed_elongation_for_direction": float(getattr(cfg, "min_seed_elongation_for_direction", 1.6)),
        "max_seed_line_residual_cells_for_direction": float(getattr(cfg, "max_seed_line_residual_cells_for_direction", 1.25)),
    }
    if seed_count < int(getattr(cfg, "min_seed_cells_for_accepted_extension", 3)):
        debug["seed_group_extensible_reason"] = "seed_group_too_few_cells_for_extension"
        return False, "seed_group_too_few_cells_for_extension", debug
    if length_cells + 1e-6 < float(getattr(cfg, "min_seed_line_length_cells_for_accepted_extension", 3)):
        debug["seed_group_extensible_reason"] = "seed_group_line_too_short_for_extension"
        return False, "seed_group_line_too_short_for_extension", debug
    if elongation + 1e-6 < float(getattr(cfg, "min_seed_elongation_for_direction", 1.6)):
        debug["seed_group_extensible_reason"] = "seed_group_orientation_ambiguous"
        return False, "seed_group_orientation_ambiguous", debug
    if residual > float(getattr(cfg, "max_seed_line_residual_cells_for_direction", 1.25)) + 1e-6:
        debug["seed_group_extensible_reason"] = "seed_group_not_line_like"
        return False, "seed_group_not_line_like", debug
    debug["seed_group_extensible_reason"] = None
    return True, None, debug


def extract_seed_line_primitives_from_group(
    group: DoorSeedGroup,
    *,
    primitive_id_start: int,
    shape: tuple[int, int],
    resolution_m: float,
    cfg: VoxelDoorDetectorConfig,
) -> list[DoorSeedLinePrimitive]:
    primitives: list[DoorSeedLinePrimitive] = []
    pid = int(primitive_id_start)
    if str(group.group_kind) == "seed_pair_bridge":
        primitives.append(
            _build_seed_line_primitive(
                pid,
                group,
                list(group.seed_cells),
                shape=shape,
                resolution_m=float(resolution_m),
                cfg=cfg,
                extraction_method="seed_pair_bridge",
                forced_major=np.asarray(group.major_dir_rc, dtype=np.float32),
                force_accept=len(group.seed_cells)
                >= min(int(getattr(cfg, "primitive_min_seed_cells", 3)), int(getattr(cfg, "min_seed_cells_for_accepted_extension", 3))),
            )
        )
        return primitives

    direct = _build_seed_line_primitive(
        pid,
        group,
        list(group.seed_cells),
        shape=shape,
        resolution_m=float(resolution_m),
        cfg=cfg,
        extraction_method="direct_pca",
    )
    primitives.append(direct)
    pid += 1
    if bool(direct.accepted_for_extension):
        return primitives
    if len(group.component_ids) >= 2 and str(direct.reject_reason) == "primitive_along_gap_too_large":
        primitives.append(
            _build_seed_line_primitive(
                pid,
                group,
                list(group.seed_cells),
                shape=shape,
                resolution_m=float(resolution_m),
                cfg=cfg,
                extraction_method="seed_pair_bridge",
                forced_major=np.asarray(group.major_dir_rc, dtype=np.float32),
                force_accept=len(group.seed_cells)
                >= min(int(getattr(cfg, "primitive_min_seed_cells", 3)), int(getattr(cfg, "min_seed_cells_for_accepted_extension", 3))),
            )
        )
        pid += 1
        if bool(primitives[-1].accepted_for_extension):
            return primitives

    if bool(getattr(cfg, "enable_seed_line_spur_pruning", True)):
        if bool(getattr(cfg, "enable_seed_parallel_line_pruning", True)):
            for cells, prune_debug in _extract_parallel_pruned_seed_line_segments(group.seed_cells, cfg=cfg):
                primitive = _build_seed_line_primitive(
                    pid,
                    group,
                    cells,
                    shape=shape,
                    resolution_m=float(resolution_m),
                    cfg=cfg,
                    extraction_method="parallel_pruned_main_run",
                )
                primitive.debug.update(prune_debug)
                primitives.append(primitive)
                pid += 1
                if bool(primitive.accepted_for_extension):
                    return primitives
        for cells, prune_debug in _extract_spur_pruned_seed_line_segments(group.seed_cells, cfg=cfg):
            primitive = _build_seed_line_primitive(
                pid,
                group,
                cells,
                shape=shape,
                resolution_m=float(resolution_m),
                cfg=cfg,
                extraction_method="spur_pruned_main_run",
            )
            primitive.debug.update(prune_debug)
            primitives.append(primitive)
            pid += 1
            if bool(primitive.accepted_for_extension):
                return primitives

    if bool(getattr(cfg, "enable_seed_blob_line_decomposition", True)):
        for cells in _extract_ransac_seed_line_segments(group.seed_cells, group_id=int(group.group_id), cfg=cfg):
            primitive = _build_seed_line_primitive(
                pid,
                group,
                cells,
                shape=shape,
                resolution_m=float(resolution_m),
                cfg=cfg,
                extraction_method="ransac_blob",
            )
            primitives.append(primitive)
            pid += 1
            if len([item for item in primitives if item.extraction_method == "ransac_blob"]) >= int(getattr(cfg, "max_primitives_per_cluster", 4)):
                break

    if not any(bool(item.accepted_for_extension) for item in primitives):
        fallback = _build_seed_line_primitive(
            pid,
            group,
            list(group.seed_cells),
            shape=shape,
            resolution_m=float(resolution_m),
            cfg=cfg,
            extraction_method="fallback_centerline",
            force_accept=_strong_seed_group_fallback_ok(group, cfg, resolution_m=float(resolution_m)),
        )
        if bool(fallback.accepted_for_extension) or len(group.seed_cells) >= int(getattr(cfg, "strong_seed_min_cells", 6)):
            primitives.append(fallback)
    return primitives


def _build_seed_line_primitive(
    primitive_id: int,
    group: DoorSeedGroup,
    cells: Sequence[tuple[int, int]],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    cfg: VoxelDoorDetectorConfig,
    extraction_method: str,
    forced_major: np.ndarray | None = None,
    force_accept: bool = False,
) -> DoorSeedLinePrimitive:
    clean_cells = sorted({(int(r), int(c)) for r, c in cells if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]})
    center, major, minor, residual, thickness_m, length_m, bbox = _fit_seed_cells(clean_cells, shape, float(resolution_m))
    if forced_major is not None:
        unit = _unit(np.asarray(forced_major, dtype=np.float32))
        if unit is not None:
            major = unit.astype(np.float32)
            minor = np.asarray([-major[1], major[0]], dtype=np.float32)
            pts = np.asarray(clean_cells, dtype=np.float32)
            if pts.size:
                rel = pts - center[None, :]
                residual_values = np.abs(np.dot(rel, minor))
                along_values = np.dot(rel, major)
                residual = float(np.max(residual_values)) if residual_values.size else 0.0
                thickness_m = float((2.0 * residual + 1.0) * float(resolution_m))
                length_m = float((float(np.max(along_values)) - float(np.min(along_values)) + 1.0) * float(resolution_m)) if along_values.size else float(resolution_m)
    seed_count = int(len(clean_cells))
    thickness_cells = max(1.0, float(thickness_m) / max(float(resolution_m), 1e-9))
    length_cells = float(length_m) / max(float(resolution_m), 1e-9)
    elongation = float(length_cells / max(thickness_cells, 1e-6))
    line_correlation, orthogonal_variance_cells2 = _primitive_line_fit_quality(clean_cells, center, major, minor)
    along_min, along_max, max_gap, segment_count, longest_run = _primitive_along_stats(clean_cells, center, major, cfg)
    reject_reason = _primitive_reject_reason(
        seed_count=seed_count,
        length_cells=length_cells,
        thickness_cells=thickness_cells,
        residual_cells=float(residual),
        elongation=elongation,
        max_gap=max_gap,
        longest_contiguous_run_cells=int(longest_run),
        line_correlation=float(line_correlation),
        orthogonal_variance_cells2=float(orthogonal_variance_cells2),
        cfg=cfg,
    )
    accepted = reject_reason is None
    return DoorSeedLinePrimitive(
        primitive_id=int(primitive_id),
        source_cluster_id=int(group.source_cluster_ids[0] if group.source_cluster_ids else group.group_id),
        source_group_id=int(group.group_id),
        source_component_ids=[int(v) for v in group.component_ids],
        cells=clean_cells,
        center_rc=(float(center[0]), float(center[1])),
        major_dir_rc=(float(major[0]), float(major[1])),
        minor_dir_rc=(float(minor[0]), float(minor[1])),
        length_cells=float(length_cells),
        thickness_cells=float(thickness_cells),
        residual_cells=float(residual),
        elongation=float(elongation),
        seed_count=int(seed_count),
        bbox_rc=bbox,
        along_min=float(along_min),
        along_max=float(along_max),
        max_along_gap_cells=float(max_gap),
        contiguous_segment_count=int(segment_count),
        accepted_for_extension=accepted,
        reject_reason=reject_reason,
        extraction_method=str(extraction_method),
        debug={
            "source_group_kind": str(group.group_kind),
            "source_group_reject_reason": group.reject_reason,
            "force_accept": bool(force_accept),
            "force_accept_overridden_by_strict_line_filter": bool(force_accept and reject_reason is not None),
            "longest_contiguous_seed_run_cells": int(longest_run),
            "primitive_line_correlation": float(line_correlation),
            "primitive_orthogonal_variance_cells2": float(orthogonal_variance_cells2),
        },
    )


def _primitive_reject_reason(
    *,
    seed_count: int,
    length_cells: float,
    thickness_cells: float,
    residual_cells: float,
    elongation: float,
    max_gap: float,
    longest_contiguous_run_cells: int,
    line_correlation: float,
    orthogonal_variance_cells2: float,
    cfg: VoxelDoorDetectorConfig,
) -> str | None:
    min_seed_cells = min(int(getattr(cfg, "primitive_min_seed_cells", 3)), int(getattr(cfg, "min_seed_cells_for_accepted_extension", 3)))
    min_length_cells = min(float(getattr(cfg, "primitive_min_length_cells", 3)), float(getattr(cfg, "min_seed_line_length_cells_for_accepted_extension", 3)))
    min_elongation = min(float(getattr(cfg, "primitive_min_elongation", 1.4)), float(getattr(cfg, "min_seed_elongation_for_direction", 1.6)))
    max_residual = max(float(getattr(cfg, "primitive_max_residual_cells", 1.75)), float(getattr(cfg, "max_seed_line_residual_cells_for_direction", 1.25)))
    if int(seed_count) < int(min_seed_cells):
        return "primitive_too_few_seed_cells"
    if int(longest_contiguous_run_cells) < int(getattr(cfg, "primitive_min_contiguous_seed_run_cells", 7)):
        return "primitive_contiguous_seed_run_too_short"
    if float(length_cells) + 1e-6 < float(min_length_cells):
        return "primitive_line_too_short"
    if float(thickness_cells) > float(getattr(cfg, "primitive_max_thickness_cells", 3)) + 1e-6:
        return "primitive_too_thick"
    if float(residual_cells) > float(max_residual) + 1e-6:
        return "primitive_residual_too_high"
    if bool(getattr(cfg, "primitive_require_line_fit_quality", True)):
        if float(line_correlation) + 1e-6 < float(getattr(cfg, "primitive_min_line_correlation", 0.93)):
            return "primitive_line_correlation_too_low"
        if float(orthogonal_variance_cells2) > float(getattr(cfg, "primitive_max_orthogonal_variance_cells2", 0.75)) + 1e-6:
            return "primitive_orthogonal_variance_too_high"
    if float(elongation) + 1e-6 < float(min_elongation):
        return "primitive_elongation_too_low"
    max_allowed_gap = float(getattr(cfg, "primitive_max_along_gap_cells", 0.0) or 0.0)
    if max_allowed_gap > 0.0 and float(max_gap) > max_allowed_gap + 1e-6:
        return "primitive_along_gap_too_large"
    return None


def _primitive_along_stats(
    cells: Sequence[tuple[int, int]],
    center: np.ndarray,
    major: np.ndarray,
    cfg: VoxelDoorDetectorConfig,
) -> tuple[float, float, float, int, int]:
    if not cells:
        return 0.0, 0.0, 0.0, 0, 0
    pts = np.asarray(cells, dtype=np.float32)
    along = np.sort(np.dot(pts - np.asarray(center, dtype=np.float32)[None, :], np.asarray(major, dtype=np.float32)))
    if along.size <= 1:
        return float(along[0]), float(along[0]), 0.0, 1, int(along.size)
    gaps = np.diff(along)
    max_gap = float(np.max(gaps)) if gaps.size else 0.0
    max_allowed_gap = float(getattr(cfg, "primitive_max_along_gap_cells", 0.0) or 0.0)
    segment_count = 1 if max_allowed_gap <= 0.0 else int(1 + np.count_nonzero(gaps > max_allowed_gap + 1e-6))
    contiguous_gap = float(getattr(cfg, "primitive_contiguous_gap_cells", 1.50))
    longest_run = 1
    current_run = 1
    for gap in gaps:
        if float(gap) <= contiguous_gap + 1e-6:
            current_run += 1
        else:
            current_run = 1
        longest_run = max(longest_run, current_run)
    return float(along[0]), float(along[-1]), max_gap, segment_count, int(longest_run)


def _primitive_line_fit_quality(
    cells: Sequence[tuple[int, int]],
    center: np.ndarray,
    major: np.ndarray,
    minor: np.ndarray,
) -> tuple[float, float]:
    if len(cells) <= 1:
        return 1.0, 0.0
    pts = np.asarray(cells, dtype=np.float32)
    rel = pts - np.asarray(center, dtype=np.float32)[None, :]
    along = np.dot(rel, np.asarray(major, dtype=np.float32))
    orth = np.dot(rel, np.asarray(minor, dtype=np.float32))
    along_var = float(np.var(along))
    orth_var = float(np.var(orth))
    total = along_var + orth_var
    if total <= 1e-9:
        return 1.0, orth_var
    line_correlation = float(np.sqrt(max(0.0, min(1.0, along_var / total))))
    return line_correlation, orth_var


def _extract_parallel_pruned_seed_line_segments(
    cells: Sequence[tuple[int, int]],
    *,
    cfg: VoxelDoorDetectorConfig,
) -> list[tuple[list[tuple[int, int]], dict[str, object]]]:
    clean = sorted({(int(r), int(c)) for r, c in cells})
    if not clean:
        return []
    min_run = max(1, int(getattr(cfg, "primitive_min_contiguous_seed_run_cells", 7)))
    run_gap = max(1, int(round(float(getattr(cfg, "primitive_contiguous_gap_cells", 1.50)))))
    max_distance = max(1, int(getattr(cfg, "seed_parallel_line_max_distance_cells", 8)))
    min_overlap = max(0.0, min(1.0, float(getattr(cfg, "seed_parallel_line_min_overlap_ratio", 0.70))))
    max_length_delta = max(0.0, float(getattr(cfg, "seed_parallel_line_max_length_delta_ratio", 0.25)))

    def contiguous_runs(values: Sequence[int]) -> list[list[int]]:
        ordered = sorted({int(v) for v in values})
        if not ordered:
            return []
        runs: list[list[int]] = [[ordered[0]]]
        for value in ordered[1:]:
            if int(value) - int(runs[-1][-1]) <= run_gap:
                runs[-1].append(int(value))
            else:
                runs.append([int(value)])
        return runs

    line_runs: list[dict[str, object]] = []
    for axis in ("h", "v"):
        buckets: dict[int, list[int]] = {}
        for r, c in clean:
            fixed, moving = (r, c) if axis == "h" else (c, r)
            buckets.setdefault(int(fixed), []).append(int(moving))
        for fixed, values in buckets.items():
            for run in contiguous_runs(values):
                if len(run) < min_run:
                    continue
                cells_for_run = [(int(fixed), int(v)) if axis == "h" else (int(v), int(fixed)) for v in run]
                line_runs.append(
                    {
                        "axis": axis,
                        "fixed": int(fixed),
                        "start": int(run[0]),
                        "end": int(run[-1]),
                        "length": int(len(run)),
                        "span": int(run[-1] - run[0] + 1),
                        "cells": cells_for_run,
                    }
                )

    if len(line_runs) < 2:
        return []

    def overlap_ratio(a: Mapping[str, object], b: Mapping[str, object]) -> float:
        start = max(int(a["start"]), int(b["start"]))
        end = min(int(a["end"]), int(b["end"]))
        overlap = max(0, end - start + 1)
        denom = max(1, min(int(a["span"]), int(b["span"])))
        return float(overlap) / float(denom)

    pairs: list[tuple[float, dict[str, object], dict[str, object], float, float, int]] = []
    for i, a in enumerate(line_runs):
        for b in line_runs[i + 1 :]:
            if str(a["axis"]) != str(b["axis"]):
                continue
            distance = abs(int(a["fixed"]) - int(b["fixed"]))
            if distance <= 0 or distance > max_distance:
                continue
            length_a = int(a["length"])
            length_b = int(b["length"])
            length_delta_ratio = abs(float(length_a - length_b)) / float(max(length_a, length_b, 1))
            if length_delta_ratio > max_length_delta + 1e-6:
                continue
            overlap = overlap_ratio(a, b)
            if overlap + 1e-6 < min_overlap:
                continue
            score = float(overlap) + 0.01 * float(min(length_a, length_b)) - 0.001 * float(distance)
            pairs.append((score, a, b, overlap, length_delta_ratio, distance))

    if not pairs:
        return []

    all_rows = [r for r, _c in clean]
    all_cols = [c for _r, c in clean]
    center_fixed_by_axis = {
        "h": float(np.mean(all_rows)) if all_rows else 0.0,
        "v": float(np.mean(all_cols)) if all_cols else 0.0,
    }

    out: list[tuple[list[tuple[int, int]], dict[str, object]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    max_count = max(1, int(getattr(cfg, "max_primitives_per_cluster", 4)))
    for _score, a, b, overlap, length_delta_ratio, distance in sorted(pairs, key=lambda item: item[0], reverse=True):
        axis = str(a["axis"])
        center_fixed = float(center_fixed_by_axis.get(axis, 0.0))

        def keep_key(item: Mapping[str, object]) -> tuple[int, float, int]:
            return (
                -int(item["length"]),
                -abs(float(item["fixed"]) - center_fixed),
                int(item["fixed"]),
            )

        kept = max((a, b), key=keep_key)
        removed = b if kept is a else a
        kept_cells = sorted({(int(r), int(c)) for r, c in kept["cells"]})  # type: ignore[index]
        key = tuple(kept_cells)
        if key in seen:
            continue
        seen.add(key)
        debug = {
            "parallel_pruned_axis": axis,
            "parallel_pruned_kept_fixed": int(kept["fixed"]),
            "parallel_pruned_removed_fixed": int(removed["fixed"]),
            "parallel_pruned_distance_cells": int(distance),
            "parallel_pruned_kept_cells": int(len(kept_cells)),
            "parallel_pruned_removed_cells": int(removed["length"]),
            "parallel_pruned_overlap_ratio": float(overlap),
            "parallel_pruned_length_delta_ratio": float(length_delta_ratio),
            "parallel_pruned_min_overlap_ratio": float(min_overlap),
            "parallel_pruned_max_length_delta_ratio": float(max_length_delta),
            "parallel_pruned_max_distance_cells": int(max_distance),
            "parallel_pruned_accepted": True,
        }
        out.append((kept_cells, debug))
        if len(out) >= max_count:
            break
    return out


def _extract_spur_pruned_seed_line_segments(
    cells: Sequence[tuple[int, int]],
    *,
    cfg: VoxelDoorDetectorConfig,
) -> list[tuple[list[tuple[int, int]], dict[str, object]]]:
    clean = sorted({(int(r), int(c)) for r, c in cells})
    if not clean:
        return []
    min_run = max(1, int(getattr(cfg, "primitive_min_contiguous_seed_run_cells", 7)))
    max_branch_ratio = max(0.0, float(getattr(cfg, "seed_line_spur_prune_max_branch_length_ratio", 0.35)))
    max_total_ratio = max(0.0, float(getattr(cfg, "seed_line_spur_prune_max_total_branch_ratio", 0.45)))
    run_gap = max(1, int(round(float(getattr(cfg, "primitive_contiguous_gap_cells", 1.50)))))
    candidates: list[tuple[int, int, str, list[tuple[int, int]], dict[str, object]]] = []

    def contiguous_runs(values: Sequence[int]) -> list[list[int]]:
        ordered = sorted({int(v) for v in values})
        if not ordered:
            return []
        runs: list[list[int]] = [[ordered[0]]]
        for value in ordered[1:]:
            if int(value) - int(runs[-1][-1]) <= run_gap:
                runs[-1].append(int(value))
            else:
                runs.append([int(value)])
        return runs

    def best_axis_line(axis: str) -> tuple[list[tuple[int, int]], int]:
        buckets: dict[int, list[int]] = {}
        for r, c in clean:
            fixed, moving = (r, c) if axis == "h" else (c, r)
            buckets.setdefault(int(fixed), []).append(int(moving))
        best_cells: list[tuple[int, int]] = []
        best_span = 0
        for fixed, values in buckets.items():
            ordered = sorted({int(v) for v in values})
            if not ordered:
                continue
            line_cells = [(int(fixed), int(v)) if axis == "h" else (int(v), int(fixed)) for v in ordered]
            span = int(ordered[-1] - ordered[0] + 1)
            if (len(line_cells), span) > (len(best_cells), best_span):
                best_cells = line_cells
                best_span = span
        return best_cells, int(best_span)

    def longest_leftover_axis_run(leftover: Sequence[tuple[int, int]], axis: str) -> int:
        buckets: dict[int, list[int]] = {}
        for r, c in leftover:
            fixed, moving = (r, c) if axis == "h" else (c, r)
            buckets.setdefault(int(fixed), []).append(int(moving))
        longest = 0
        for values in buckets.values():
            for run in contiguous_runs(values):
                longest = max(longest, int(len(run)))
        return int(longest)

    all_cells = set(clean)
    for axis in ("h", "v"):
        main_line, main_span = best_axis_line(axis)
        main_count = int(len(main_line))
        if main_count < min_run:
            continue
        main_set = set(main_line)
        leftover = sorted(all_cells - main_set)
        leftover_count = int(len(leftover))
        if leftover_count <= 0:
            continue
        longest_branch = max(longest_leftover_axis_run(leftover, "h"), longest_leftover_axis_run(leftover, "v"))
        branch_ratio = float(longest_branch) / float(max(main_count, 1))
        total_ratio = float(leftover_count) / float(max(main_count, 1))
        accepted = branch_ratio <= max_branch_ratio + 1e-6 and total_ratio <= max_total_ratio + 1e-6
        debug = {
            "spur_pruned_main_run_axis": axis,
            "spur_pruned_main_run_cells": int(main_count),
            "spur_pruned_main_line_span_cells": int(main_span),
            "spur_pruned_same_line_cells_retained": int(main_count),
            "spur_pruned_removed_cells": int(leftover_count),
            "spur_pruned_longest_branch_cells": int(longest_branch),
            "spur_pruned_branch_length_ratio": float(branch_ratio),
            "spur_pruned_total_branch_ratio": float(total_ratio),
            "spur_pruned_max_branch_length_ratio": float(max_branch_ratio),
            "spur_pruned_max_total_branch_ratio": float(max_total_ratio),
            "spur_pruned_accepted": bool(accepted),
        }
        if not accepted:
            continue
        candidates.append((main_count, -leftover_count, axis, sorted(main_line), debug))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    out: list[tuple[list[tuple[int, int]], dict[str, object]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    max_count = max(1, int(getattr(cfg, "max_primitives_per_cluster", 4)))
    for _main_count, _neg_leftover_count, _axis, main_run, debug in candidates:
        key = tuple(main_run)
        if key in seen:
            continue
        seen.add(key)
        out.append((main_run, debug))
        if len(out) >= max_count:
            break
    return out


def _extract_ransac_seed_line_segments(
    cells: Sequence[tuple[int, int]],
    *,
    group_id: int,
    cfg: VoxelDoorDetectorConfig,
) -> list[list[tuple[int, int]]]:
    remaining = sorted({(int(r), int(c)) for r, c in cells})
    segments_out: list[list[tuple[int, int]]] = []
    max_primitives = max(0, int(getattr(cfg, "max_primitives_per_cluster", 4)))
    for _ in range(max_primitives):
        if len(remaining) < int(getattr(cfg, "ransac_min_inliers", 3)):
            break
        best = _best_ransac_line(remaining, group_id=int(group_id), cfg=cfg)
        if best is None:
            break
        center, major, inliers = best
        if len(inliers) < int(getattr(cfg, "ransac_min_inliers", 3)):
            break
        segments = _split_cells_by_along_gap(inliers, center, major, max_gap=float(getattr(cfg, "primitive_max_along_gap_cells", 2.0)))
        any_segment = False
        for segment in segments:
            if len(segment) >= int(getattr(cfg, "ransac_min_inliers", 3)):
                segments_out.append(segment)
                any_segment = True
        if not any_segment:
            break
        if bool(getattr(cfg, "remove_inliers_after_primitive", True)):
            remove = set(inliers)
            remaining = [cell for cell in remaining if cell not in remove]
        else:
            break
    return segments_out


def _best_ransac_line(
    cells: Sequence[tuple[int, int]],
    *,
    group_id: int,
    cfg: VoxelDoorDetectorConfig,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]] | None:
    pts = np.asarray(cells, dtype=np.float32)
    if pts.shape[0] < 2:
        return None
    rng = np.random.default_rng(1009 + int(group_id))
    pair_count = min(max(1, int(getattr(cfg, "ransac_num_trials", 64))), max(1, pts.shape[0] * (pts.shape[0] - 1) // 2))
    pairs: list[tuple[int, int]] = []
    if pts.shape[0] <= 32:
        for i in range(pts.shape[0]):
            for j in range(i + 1, pts.shape[0]):
                pairs.append((i, j))
        if len(pairs) > pair_count:
            indices = rng.choice(len(pairs), size=pair_count, replace=False)
            pairs = [pairs[int(i)] for i in indices]
    else:
        seen: set[tuple[int, int]] = set()
        while len(pairs) < pair_count:
            i, j = rng.choice(pts.shape[0], size=2, replace=False).tolist()
            pair = (min(int(i), int(j)), max(int(i), int(j)))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    best_score = -1e9
    best_result: tuple[np.ndarray, np.ndarray, list[tuple[int, int]]] | None = None
    residual_limit = float(getattr(cfg, "ransac_inlier_residual_cells", 1.5))
    for i, j in pairs:
        p0 = pts[int(i)]
        p1 = pts[int(j)]
        delta = p1 - p0
        norm = float(np.linalg.norm(delta))
        if norm < 2.0:
            continue
        major = (delta / norm).astype(np.float32)
        minor = np.asarray([-major[1], major[0]], dtype=np.float32)
        residual = np.abs(np.dot(pts - p0[None, :], minor))
        inlier_idx = np.flatnonzero(residual <= residual_limit)
        if inlier_idx.size < int(getattr(cfg, "ransac_min_inliers", 3)):
            continue
        inlier_pts = pts[inlier_idx]
        along = np.dot(inlier_pts - p0[None, :], major)
        length = float(np.max(along) - np.min(along) + 1.0) if along.size else 0.0
        residual_mean = float(np.mean(residual[inlier_idx])) if inlier_idx.size else residual_limit
        score = float(inlier_idx.size + 0.3 * length - 0.5 * residual_mean)
        if score > best_score:
            best_score = score
            center = np.mean(inlier_pts, axis=0).astype(np.float32)
            best_result = (center, major.astype(np.float32), [tuple(map(int, cells[int(idx)])) for idx in inlier_idx])
    return best_result


def _split_cells_by_along_gap(
    cells: Sequence[tuple[int, int]],
    center: np.ndarray,
    major: np.ndarray,
    *,
    max_gap: float,
) -> list[list[tuple[int, int]]]:
    if not cells:
        return []
    if float(max_gap) <= 0.0:
        return [list(cells)]
    pts = np.asarray(cells, dtype=np.float32)
    along = np.dot(pts - np.asarray(center, dtype=np.float32)[None, :], np.asarray(major, dtype=np.float32))
    order = np.argsort(along)
    segments: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = [tuple(map(int, cells[int(order[0])]))]
    for prev_idx, idx in zip(order[:-1], order[1:]):
        if float(along[int(idx)] - along[int(prev_idx)]) > float(max_gap) + 1e-6:
            segments.append(current)
            current = []
        current.append(tuple(map(int, cells[int(idx)])))
    if current:
        segments.append(current)
    return segments


def _strong_seed_group_fallback_ok(group: DoorSeedGroup, cfg: VoxelDoorDetectorConfig, *, resolution_m: float) -> bool:
    if not bool(getattr(cfg, "enable_strong_seed_centerline_fallback", True)):
        return False
    if len(group.seed_cells) < int(getattr(cfg, "strong_seed_min_cells", getattr(cfg, "strong_seed_centerline_min_cells", 4))):
        return False
    thickness_cells = max(1.0, float(group.thickness_m) / max(float(resolution_m), 1e-9))
    if thickness_cells > float(getattr(cfg, "strong_seed_max_thickness_cells", 5)) + 1e-6:
        return False
    return True


def _unit(vec: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-6:
        return None
    return (arr / norm).astype(np.float32)


def _nearest_axis(vec: np.ndarray) -> np.ndarray:
    unit = _unit(vec)
    if unit is None:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    if abs(float(unit[0])) >= abs(float(unit[1])):
        return np.asarray([1.0, 0.0], dtype=np.float32) * (1.0 if float(unit[0]) >= 0.0 else -1.0)
    return np.asarray([0.0, 1.0], dtype=np.float32) * (1.0 if float(unit[1]) >= 0.0 else -1.0)


def _unsigned_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    au = _unit(a)
    bu = _unit(b)
    if au is None or bu is None:
        return 180.0
    dot = min(1.0, max(-1.0, abs(float(np.dot(au, bu)))))
    return float(np.degrees(np.arccos(dot)))


def _infer_directions_from_nearby_wall_pairs(cluster: DoorSeedCluster, wall_clean: np.ndarray) -> list[tuple[str, np.ndarray]]:
    wall = np.asarray(wall_clean, dtype=bool)
    center = np.asarray(cluster.center_rc, dtype=np.float32)
    radius = max(4, int(np.ceil(max(float(cluster.length_m), float(cluster.thickness_m)) / 0.05)))
    r0 = max(0, int(round(float(center[0]))) - radius)
    r1 = min(wall.shape[0], int(round(float(center[0]))) + radius + 1)
    c0 = max(0, int(round(float(center[1]))) - radius)
    c1 = min(wall.shape[1], int(round(float(center[1]))) + radius + 1)
    rr, cc = np.nonzero(wall[r0:r1, c0:c1])
    if rr.size < 2:
        return []
    pts = np.stack([rr + r0, cc + c0], axis=1).astype(np.float32)
    rel = pts - center[None, :]
    dist = np.linalg.norm(rel, axis=1)
    keep = dist > 1.0
    pts = pts[keep]
    rel = rel[keep]
    if pts.shape[0] < 2:
        return []
    best: tuple[float, np.ndarray] | None = None
    for idx in range(min(64, pts.shape[0])):
        for jdx in range(idx + 1, min(64, pts.shape[0])):
            a = rel[idx]
            b = rel[jdx]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na <= 1e-6 or nb <= 1e-6:
                continue
            opposite = -float(np.dot(a / na, b / nb))
            if opposite < 0.70:
                continue
            vec = pts[jdx] - pts[idx]
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-6:
                continue
            score = opposite * norm
            if best is None or score > best[0]:
                best = (score, vec / norm)
    if best is None:
        return []
    return [("nearby_wall_pair_axis", best[1].astype(np.float32))]


def _infer_directions_from_local_free_neck(cluster: DoorSeedCluster, free_clean: np.ndarray) -> list[tuple[str, np.ndarray]]:
    free = np.asarray(free_clean, dtype=bool)
    center = np.asarray(cluster.center_rc, dtype=np.float32)
    radius = 6
    r0 = max(0, int(round(float(center[0]))) - radius)
    r1 = min(free.shape[0], int(round(float(center[0]))) + radius + 1)
    c0 = max(0, int(round(float(center[1]))) - radius)
    c1 = min(free.shape[1], int(round(float(center[1]))) + radius + 1)
    rr, cc = np.nonzero(free[r0:r1, c0:c1])
    if rr.size < 4:
        return []
    pts = np.stack([rr + r0, cc + c0], axis=1).astype(np.float32)
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    major = vt[0].astype(np.float32)
    norm = float(np.linalg.norm(major))
    if norm <= 1e-6:
        return []
    return [("local_free_neck_axis", major / norm)]


def _select_best_cluster_candidate(candidates: Sequence[VoxelDoorLineCandidate]) -> VoxelDoorLineCandidate:
    def key(candidate: VoxelDoorLineCandidate) -> tuple[int, int, int, int, int, int, float, float, int]:
        source = str(candidate.debug.get("orientation_source", ""))
        width_m = float(candidate.width_m)
        two_anchor = candidate.wall_anchor_a is not None and candidate.wall_anchor_b is not None
        seed_overlap = int(candidate.debug.get("door_partition_seed_on_line_cells", 0) or 0)
        visual_cells = int(candidate.debug.get("visual_line_cells", 0) or 0)
        seed_span_m = max(float(len(candidate.seed_cells)) * 0.10, 0.10)
        width_delta = abs(width_m - seed_span_m)
        return (
            1 if bool(candidate.debug.get("partition_effective_verified", candidate.debug.get("partition_accepted", False))) else 0,
            1 if bool(candidate.debug.get("partition_geometry_accepted", False)) else 0,
            1 if candidate.accepted else 0,
            1 if two_anchor else 0,
            int(seed_overlap),
            -1 if "diag" in source else 0,
            -float(width_delta),
            float(candidate.debug.get("score", 0.0)),
            -int(visual_cells),
        )

    return max(candidates, key=key)


def _door_anchor_source_priority(source: int) -> float:
    if int(source) == DOOR_ANCHOR_STRICT_RAW:
        return 1.0
    if int(source) == DOOR_ANCHOR_PROJECTED:
        return 0.85
    if int(source) == DOOR_ANCHOR_STEP1:
        return 0.80
    if int(source) == DOOR_ANCHOR_FILTERED_LINE:
        return 0.75
    if int(source) == DOOR_ANCHOR_PROJECTED_ANCHOR:
        return 0.70
    if int(source) == DOOR_ANCHOR_NO_CLEARANCE_OCCUPIED:
        return 0.65
    return 0.0


def _door_reject_code(reason: str) -> int:
    values = {
        "door_line_does_not_hit_two_walls": 1,
        "door_line_does_not_hit_wall_anchor": 1,
        "extension_hits_other_door_cluster": 2,
        "extension_intersects_other_door_candidate": 3,
        "extension_endpoint_is_other_door": 4,
        "door_visual_width_out_of_range": 5,
        "door_partition_cut_empty": 6,
        "door_inner_unknown_ratio_too_high": 7,
        "door_inner_wall_ratio_too_high": 8,
        "door_inner_free_or_seed_ratio_too_low": 9,
        "door_extension_total_too_long": 10,
        "door_cut_not_wall_attached": 11,
        "door_cut_no_topology_gain": 12,
        "visual_only_not_partition": 13,
        "door_partition_intersects_stable_other_door": 14,
        "raw_seed_conflict_ignored": 15,
        "door_partition_small_known_side_low_unknown": 16,
        "door_extension_crosses_no_clearance_occupied": 17,
    }
    return int(values.get(str(reason), 255))


def _candidate_from_seed_component(
    *,
    candidate_id: int,
    component_id: int,
    seed_cells: list[tuple[int, int]],
    seed_component_mask: np.ndarray,
    all_seed_mask: np.ndarray,
    free_clean: np.ndarray,
    wall_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    cfg: VoxelDoorDetectorConfig,
    wall_anchor_source_map: np.ndarray | None = None,
    partition_free_clean: np.ndarray | None = None,
    real_wall_barrier_map: np.ndarray | None = None,
    no_clearance_occupied_map: np.ndarray | None = None,
    same_cluster_mask: np.ndarray | None,
    forced_major: np.ndarray | None = None,
    orientation_source: str = "pca",
    cluster_id: int | None = None,
    seed_group_id: int | None = None,
    seed_group_kind: str | None = None,
    source_cluster_ids: Sequence[int] | None = None,
    component_ids: Sequence[int] | None = None,
    mode_hint: str | None = None,
) -> VoxelDoorLineCandidate:
    pts = np.asarray(seed_cells, dtype=np.float32)
    if partition_free_clean is None:
        partition_free_clean = free_clean
    real_wall = np.asarray(real_wall_barrier_map if real_wall_barrier_map is not None else wall_clean, dtype=bool)
    no_clearance_occupied = (
        np.asarray(no_clearance_occupied_map, dtype=bool)
        if no_clearance_occupied_map is not None
        else np.zeros_like(free_clean, dtype=bool)
    )
    same_seed_mask = np.asarray(same_cluster_mask if same_cluster_mask is not None else seed_component_mask, dtype=bool)
    if forced_major is not None:
        center = np.mean(pts, axis=0) if pts.size else np.asarray([0.0, 0.0], dtype=np.float32)
        major = np.asarray(forced_major, dtype=np.float32)
        norm = float(np.linalg.norm(major))
        if norm <= 1e-6:
            return _rejected_candidate(candidate_id, component_id, seed_cells, "zero_orientation")
        major = major / norm
        minor_tmp = np.asarray([-major[1], major[0]], dtype=np.float32)
        residual = np.abs(np.dot(pts - center[None, :], minor_tmp)) if pts.size else np.asarray([], dtype=np.float32)
        residual_max = float(np.max(residual)) if residual.size else 0.0
    elif len(seed_cells) == 1:
        inferred = _infer_single_seed_direction(seed_cells[0], wall_clean, resolution_m, cfg) if bool(cfg.single_seed_infer_from_wall_enabled) else None
        if inferred is None:
            return _rejected_candidate(candidate_id, component_id, seed_cells, "single_seed_no_wall_geometry")
        center = pts[0]
        major = inferred
        residual_max = 0.0
    else:
        center = np.mean(pts, axis=0)
        centered = pts - center[None, :]
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        major = vt[0].astype(np.float32)
        norm = float(np.linalg.norm(major))
        if norm <= 1e-6:
            return _rejected_candidate(candidate_id, component_id, seed_cells, "door_seed_not_line_like")
        major = major / norm
        minor_tmp = np.asarray([-major[1], major[0]], dtype=np.float32)
        residual = np.abs(np.dot(centered, minor_tmp))
        residual_max = float(np.max(residual)) if residual.size else 0.0
        if residual_max > float(cfg.line_fit_max_residual_cells):
            return _rejected_candidate(candidate_id, component_id, seed_cells, "door_seed_not_line_like", center, major)
    minor = np.asarray([-major[1], major[0]], dtype=np.float32)
    seed_centerline = _project_seed_component_to_centerline(pts, center, major, minor, free_clean.shape)
    if not seed_centerline:
        return _rejected_candidate(candidate_id, component_id, seed_cells, "door_centerline_projection_empty", center, major)
    thickness_m = _component_thickness_m(pts, center, minor, resolution_m)
    if thickness_m > float(cfg.max_component_thickness_m) + 1e-6:
        return _rejected_candidate(candidate_id, component_id, seed_cells, "door_seed_component_too_thick", center, major)

    ordered = _sort_cells_along_direction(seed_centerline, center, major)
    start = np.asarray(ordered[0], dtype=np.float32)
    end = np.asarray(ordered[-1], dtype=np.float32)
    walk_a = _walk_to_wall(
        start_rc=start,
        direction=-major,
        wall_anchor_map=wall_clean,
        wall_anchor_source_map=wall_anchor_source_map,
        same_cluster_mask=same_seed_mask,
        other_door_mask=all_seed_mask & ~same_seed_mask,
        unknown_map=unknown_clean,
        no_clearance_occupied_map=no_clearance_occupied,
        resolution_m=resolution_m,
        cfg=cfg,
    )
    walk_b = _walk_to_wall(
        start_rc=end,
        direction=major,
        wall_anchor_map=wall_clean,
        wall_anchor_source_map=wall_anchor_source_map,
        same_cluster_mask=same_seed_mask,
        other_door_mask=all_seed_mask & ~same_seed_mask,
        unknown_map=unknown_clean,
        no_clearance_occupied_map=no_clearance_occupied,
        resolution_m=resolution_m,
        cfg=cfg,
    )
    anchor_a, extension_a, reason_a, source_a = walk_a.anchor, walk_a.path_cells, walk_a.stopped_reason, walk_a.anchor_source
    anchor_b, extension_b, reason_b, source_b = walk_b.anchor, walk_b.path_cells, walk_b.stopped_reason, walk_b.anchor_source
    candidate_anchor_barrier = real_wall.copy()
    for anchor, source in ((anchor_a, source_a), (anchor_b, source_b)):
        if anchor is not None and int(source) == DOOR_ANCHOR_NO_CLEARANCE_OCCUPIED:
            rr, cc = int(anchor[0]), int(anchor[1])
            if 0 <= rr < candidate_anchor_barrier.shape[0] and 0 <= cc < candidate_anchor_barrier.shape[1]:
                candidate_anchor_barrier[rr, cc] = True
    group_id = int(component_id if seed_group_id is None else seed_group_id)
    cluster_debug_id = int(component_id if cluster_id is None else cluster_id)
    component_debug_ids = [int(v) for v in (component_ids or [component_id])]
    source_cluster_debug_ids = [int(v) for v in (source_cluster_ids or [cluster_debug_id])]
    seed_group_kind_debug = str(seed_group_kind or "single_cluster")
    full_cells: list[tuple[int, int]]
    visual_status: str
    completion_mode: str
    if anchor_a is not None and anchor_b is not None:
        full_cells = _line_cells(anchor_a, anchor_b, free_clean.shape)
        visual_status = "completed_two_wall_anchors"
        completion_mode = DOOR_COMPLETION_MIDDLE_SEED_TWO_WALL
    elif (anchor_a is not None or anchor_b is not None) and bool(getattr(cfg, "enable_one_seed_one_wall_completion", True)):
        if anchor_a is not None:
            full_cells = _line_cells(anchor_a, ordered[-1], free_clean.shape)
            visual_status = "completed_one_seed_one_wall_anchor_a"
        else:
            full_cells = _line_cells(ordered[0], anchor_b, free_clean.shape)
            visual_status = "completed_one_seed_one_wall_anchor_b"
        completion_mode = DOOR_COMPLETION_ONE_SEED_ONE_WALL
    elif (
        bool(getattr(cfg, "enable_seed_pair_bridge_completion", True))
        and (
            mode_hint == DOOR_COMPLETION_SEED_PAIR_BRIDGE
            or len(component_debug_ids) >= 2
            or seed_group_kind_debug == "seed_pair_bridge"
        )
    ):
        full_cells = _sort_cells_along_direction(seed_centerline, center, major)
        visual_status = "completed_seed_pair_bridge"
        completion_mode = DOOR_COMPLETION_SEED_PAIR_BRIDGE
    elif _strong_seed_centerline_fallback_ok(
        seed_cells=seed_cells,
        seed_centerline=seed_centerline,
        residual_max=float(residual_max),
        thickness_m=float(thickness_m),
        resolution_m=float(resolution_m),
        cfg=cfg,
    ):
        full_cells = _sort_cells_along_direction(seed_cells, center, major)
        visual_status = "completed_strong_seed_centerline"
        completion_mode = DOOR_COMPLETION_STRONG_SEED_CENTERLINE
    else:
        candidate = _rejected_candidate(candidate_id, component_id, seed_cells, "door_line_does_not_hit_wall_anchor", center, major)
        candidate.seed_projected_centerline_cells = seed_centerline
        candidate.extended_centerline_cells = [*list(reversed(extension_a)), *ordered, *extension_b]
        candidate.debug.update(
            {
                "anchor_a_reject_reason": reason_a,
                "anchor_b_reject_reason": reason_b,
                "anchor_a_source": _anchor_source_name(source_a),
                "anchor_b_source": _anchor_source_name(source_b),
                "cluster_id": int(cluster_debug_id),
                "seed_group_id": int(group_id),
                "seed_group_kind": seed_group_kind_debug,
                "source_cluster_ids": source_cluster_debug_ids,
                "component_ids": component_debug_ids,
                "completion_mode": DOOR_COMPLETION_REJECTED,
                "orientation_source": str(orientation_source),
                "walk_a_cells": [[int(r), int(c)] for r, c in extension_a],
                "walk_b_cells": [[int(r), int(c)] for r, c in extension_b],
                "walk_a_status": walk_a.status,
                "walk_b_status": walk_b.status,
                "walk_crossed_other_seed_cluster": bool(walk_a.hit_other_seed_cells or walk_b.hit_other_seed_cells),
                "walk_crossed_other_seed_cluster_cells": [[int(r), int(c)] for r, c in [*walk_a.hit_other_seed_cells, *walk_b.hit_other_seed_cells][:64]],
                "visual_status": "one_sided_anchor_a" if anchor_a is not None else ("one_sided_anchor_b" if anchor_b is not None else "no_wall_anchor"),
                "visual_accepted": False,
                "partition_accepted": False,
                "partition_topology_accepted": False,
                "reject_reason_visual": "door_line_does_not_hit_wall_anchor",
                "reject_reason_partition": "visual_rejected",
                "reject_reason_topology": "visual_rejected",
            }
        )
        return candidate

    width_m = float(len(full_cells) * resolution_m)
    own_seed = _cells_to_mask(seed_cells, free_clean.shape)
    visual_mask = _cells_to_mask(full_cells, free_clean.shape)
    cut_result = build_door_partition_cut_v30(
        full_line_cells=full_cells,
        seed_mask=own_seed,
        accepted_seed_mask=same_seed_mask,
        partition_free=np.asarray(partition_free_clean, dtype=bool),
        partition_unknown=np.asarray(unknown_clean, dtype=bool),
        real_wall_barrier=candidate_anchor_barrier,
        anchor_a=anchor_a,
        anchor_b=anchor_b,
        max_unknown_bridge_gap_cells=int(getattr(cfg, "partition_cut_bridge_unknown_max_cells", 4)),
        max_nonfree_bridge_gap_cells=int(getattr(cfg, "partition_cut_bridge_nonfree_max_cells", 1)),
        max_endpoint_wall_gap_cells=int(getattr(cfg, "door_wall_attachment_max_endpoint_gap_cells", 1)),
        seed_dilation_cells=int(getattr(cfg, "partition_cut_seed_dilation_cells", 1)),
        min_cut_cells=int(getattr(cfg, "partition_cut_min_cells", 1)),
    )
    partition_mask = cut_result.mask
    partition_cells = list(cut_result.cut_cells)
    seed_on_line_mask = same_seed_mask & visual_mask
    partition_seed_on_line_cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(seed_on_line_mask))]
    if partition_seed_on_line_cells:
        partition_mask = np.asarray(partition_mask, dtype=bool) | seed_on_line_mask
        partition_cells = sorted({(int(r), int(c)) for r, c in [*partition_cells, *partition_seed_on_line_cells]})
    inner = full_cells[1:-1] if len(full_cells) > 2 else full_cells
    inner_unknown_ratio = _ratio(inner, unknown_clean)
    inner_wall_ratio = _ratio(inner, candidate_anchor_barrier & ~own_seed)
    inner_free_or_seed_ratio = _ratio(inner, free_clean | own_seed)
    visual_reject_reason = None
    extension_a_limit_cells = _extension_cells_outside_seed(extension_a, same_seed_mask) if anchor_a is not None else []
    extension_b_limit_cells = _extension_cells_outside_seed(extension_b, same_seed_mask) if anchor_b is not None else []
    extension_total_limit_cells = list(dict.fromkeys([*extension_a_limit_cells, *extension_b_limit_cells]))
    extension_occ_reject, extension_occ_debug = _door_extension_no_clearance_occupied_reject(
        extension_total_limit_cells,
        no_clearance_occupied=no_clearance_occupied,
        real_wall=real_wall,
        anchors=[anchor for anchor in (anchor_a, anchor_b) if anchor is not None],
        cfg=cfg,
    )
    extension_a_m = float(len(extension_a_limit_cells) * resolution_m)
    extension_b_m = float(len(extension_b_limit_cells) * resolution_m)
    extension_total_m = float(len(extension_total_limit_cells) * resolution_m)
    max_total_extension_m = float(getattr(cfg, "partition_cut_max_total_extension_m", cfg.extend_max_m))
    if extension_total_m > max_total_extension_m + 1e-9:
        visual_reject_reason = "door_extension_total_too_long"
    elif bool(extension_occ_reject):
        visual_reject_reason = "door_extension_crosses_no_clearance_occupied"
    elif bool(cfg.enforce_seed_door_width_limits) and (width_m < float(cfg.door_width_min_m) or width_m > float(cfg.door_width_max_m)):
        visual_reject_reason = "door_width_out_of_range"
    elif "diag" in str(orientation_source) and not (anchor_a is not None and anchor_b is not None):
        visual_reject_reason = "door_diagonal_without_support"
    if visual_reject_reason is None:
        neck_ok, neck_reason, neck_debug = validate_door_line_local_neck(
            full_cells,
            same_seed_mask,
            free_clean,
            wall_clean,
            candidate_anchor_barrier,
            resolution_m,
            cfg,
            completion_mode=completion_mode,
            orientation_source=str(orientation_source),
        )
        if not bool(neck_ok):
            visual_reject_reason = str(neck_reason or "door_line_local_neck_rejected")
    else:
        neck_debug = {"door_line_local_neck_checked": False, "door_line_local_neck_skip_reason": str(visual_reject_reason)}
    partition_reject_reason = None
    if visual_reject_reason is not None:
        partition_reject_reason = "visual_rejected"
        partition_cells = []
    elif not partition_cells:
        partition_reject_reason = "door_partition_cut_empty"
    elif inner_unknown_ratio > float(getattr(cfg, "partition_inner_unknown_ratio_max", cfg.inner_unknown_ratio_max)):
        partition_reject_reason = "door_inner_unknown_ratio_too_high"
        partition_cells = []
    elif inner_wall_ratio > float(getattr(cfg, "partition_inner_wall_ratio_max", cfg.inner_wall_ratio_max)):
        partition_reject_reason = "door_inner_wall_ratio_too_high"
        partition_cells = []
    elif inner_free_or_seed_ratio < float(getattr(cfg, "partition_inner_free_or_seed_ratio_min", cfg.inner_free_or_seed_ratio_min)):
        partition_reject_reason = "door_inner_free_or_seed_ratio_too_low"
        partition_cells = []
    if partition_reject_reason is None and len(partition_cells) < max(1, int(getattr(cfg, "min_geometry_cut_cells", 1))):
        partition_reject_reason = "door_partition_cut_empty"
        partition_cells = []
        partition_mask = np.zeros_like(partition_mask, dtype=bool)
    geometry_reject_reason = partition_reject_reason
    geometry_partition_accepted = partition_reject_reason is None
    geometry_partition_cells = list(partition_cells)
    geometry_partition_mask = _cells_to_mask(geometry_partition_cells, free_clean.shape)
    topology = DoorTopologyValidationResult(
        topology_accepted=False,
        reject_reason="partition_basic_rejected" if partition_reject_reason is not None else None,
        before_components=0,
        after_components=0,
        touched_labels=[],
        new_component_count=0,
        side_component_areas=[],
        side_component_widths_cells=[],
    )
    if geometry_partition_accepted and bool(getattr(cfg, "partition_topology_enabled", True)):
        topology = validate_door_partition_cut_topology(
            door_cut_mask=geometry_partition_mask,
            base_partition_free=np.asarray(partition_free_clean, dtype=bool),
            accepted_seed_mask=same_seed_mask,
            partition_real_wall_map=candidate_anchor_barrier,
            connectivity=4,
            min_side_area_cells=int(getattr(cfg, "partition_topology_min_side_area_cells", 4)),
            min_side_width_cells=int(getattr(cfg, "partition_topology_min_side_width_cells", 2)),
            local_radius_cells=int(getattr(cfg, "partition_topology_local_radius_cells", 18)),
            allow_anchor_closure_without_global_gain=bool(getattr(cfg, "partition_topology_allow_anchor_closure_without_global_gain", True)),
            allow_neck_cut_without_global_gain=bool(getattr(cfg, "partition_topology_allow_neck_cut_without_global_gain", True)),
            touches_two_anchors=bool(anchor_a is not None and anchor_b is not None),
            seed_overlap=bool(np.any(geometry_partition_mask & same_seed_mask)),
        )
    elif geometry_partition_accepted:
        topology = DoorTopologyValidationResult(
            topology_accepted=True,
            reject_reason=None,
            before_components=0,
            after_components=0,
            touched_labels=[],
            new_component_count=0,
            side_component_areas=[],
            side_component_widths_cells=[],
        )
    attachment = validate_door_cut_wall_attachment(
        cut_mask=geometry_partition_mask,
        full_line_cells=full_cells,
        real_wall_barrier=candidate_anchor_barrier,
        seed_mask=same_seed_mask,
        max_endpoint_gap_cells=int(getattr(cfg, "door_wall_attachment_max_endpoint_gap_cells", 1)),
    )
    topology_gain_effective = bool(topology.topology_accepted) and (
        int(topology.after_components) > int(topology.before_components)
        or int(topology.new_component_count) > 0
        or not bool(getattr(cfg, "partition_topology_enabled", True))
    )
    closure_verified, closure_debug = verify_cut_wall_to_wall_closure_v30(
        cut_mask=geometry_partition_mask,
        full_line_cells=full_cells,
        anchor_a=anchor_a,
        anchor_b=anchor_b,
        real_wall_barrier=candidate_anchor_barrier,
        base_partition_free=np.asarray(partition_free_clean, dtype=bool),
        max_endpoint_gap_cells=int(getattr(cfg, "door_wall_attachment_max_endpoint_gap_cells", 1)),
        min_side_area_cells=int(getattr(cfg, "partition_topology_min_side_area_cells", 2)),
        min_side_width_cells=int(getattr(cfg, "partition_topology_min_side_width_cells", 1)),
    )
    small_known_side_rejected, small_known_side_debug = validate_door_partition_small_known_side(
        cut_mask=geometry_partition_mask,
        base_partition_free=np.asarray(partition_free_clean, dtype=bool),
        partition_unknown=np.asarray(unknown_clean, dtype=bool),
        partition_real_wall_map=candidate_anchor_barrier,
        resolution_m=float(resolution_m),
        cfg=cfg,
    )
    small_known_side_reason = (
        "door_partition_small_known_side_low_unknown" if bool(small_known_side_rejected) else None
    )
    partition_effective_verified = bool(
        geometry_partition_accepted
        and not bool(small_known_side_rejected)
        and (topology_gain_effective or bool(closure_verified))
    )
    partition_topology_effective = bool(partition_effective_verified)
    final_partition_reject_reason = geometry_reject_reason
    if geometry_partition_accepted and small_known_side_reason is not None:
        final_partition_reject_reason = str(small_known_side_reason)
        partition_cells = []
        partition_mask = np.zeros_like(partition_mask, dtype=bool)
    elif geometry_partition_accepted and not partition_effective_verified:
        final_partition_reject_reason = str(
            cut_result.debug.get("door_partition_cut_v30_reject_reason")
            or topology.reject_reason
            or closure_debug.get("partition_closure_reject_reason")
            or attachment.reject_reason
            or "door_cut_no_topology_gain"
        )
        partition_cells = []
        partition_mask = np.zeros_like(partition_mask, dtype=bool)
    elif partition_effective_verified:
        final_partition_reject_reason = None
        partition_cells = list(geometry_partition_cells)
        partition_mask = geometry_partition_mask.copy()
    topology_warning = bool(geometry_partition_accepted and not partition_topology_effective)
    topology_warning_reason = None if not topology_warning else str(final_partition_reject_reason or "door_cut_no_topology_gain")
    anchor_score = 1.0 if anchor_a is not None and anchor_b is not None else (0.80 if anchor_a is not None or anchor_b is not None else 0.65)
    seed_line_mask = dilate(visual_mask, 1)
    seed_coverage_score = _ratio(seed_cells, seed_line_mask)
    source_values = [value for value in (source_a, source_b) if int(value) != DOOR_ANCHOR_NONE]
    source_priority = float(np.mean([_door_anchor_source_priority(value) for value in source_values])) if source_values else 0.50
    length_score = min(1.0, width_m / max(float(cfg.visual_width_min_m), 1e-6))
    score = float(
        0.30 * anchor_score
        + 0.20 * min(1.0, seed_coverage_score)
        + 0.20 * min(1.0, inner_free_or_seed_ratio)
        + 0.15 * (1.0 - min(1.0, inner_unknown_ratio))
        + 0.10 * source_priority
        + 0.05 * length_score
    )
    return VoxelDoorLineCandidate(
        candidate_id=candidate_id,
        seed_component_id=component_id,
        seed_cells=seed_cells,
        center_rc=(float(center[0]), float(center[1])),
        major_dir_rc=(float(major[0]), float(major[1])),
        minor_dir_rc=(float(minor[0]), float(minor[1])),
        seed_projected_centerline_cells=seed_centerline,
        extended_centerline_cells=full_cells,
        door_cut_cells=partition_cells,
        wall_anchor_a=anchor_a,
        wall_anchor_b=anchor_b,
        width_m=width_m,
        accepted=visual_reject_reason is None,
        reject_reason=visual_reject_reason,
        debug={
            "score": float(score),
            "cluster_id": int(cluster_debug_id),
            "seed_group_id": int(group_id),
            "seed_group_kind": seed_group_kind_debug,
            "source_cluster_ids": source_cluster_debug_ids,
            "component_ids": component_debug_ids,
            "completion_mode": str(completion_mode),
            "orientation_source": str(orientation_source),
            "component_thickness_m": float(thickness_m),
            "residual_max_cells": float(residual_max),
            "inner_unknown_ratio": float(inner_unknown_ratio),
            "inner_wall_ratio": float(inner_wall_ratio),
            "inner_free_or_seed_ratio": float(inner_free_or_seed_ratio),
            "anchor_a_source": _anchor_source_name(source_a),
            "anchor_b_source": _anchor_source_name(source_b),
            "anchor_a_source_code": int(source_a),
            "anchor_b_source_code": int(source_b),
            "visual_line_cells": int(len(full_cells)),
            "door_width_max_m": float(cfg.door_width_max_m),
            "door_partition_width_limit_enforced": bool(cfg.enforce_seed_door_width_limits),
            "door_extension_a_m": float(extension_a_m),
            "door_extension_b_m": float(extension_b_m),
            "door_extension_total_m": float(extension_total_m),
            "door_extension_limit_cells": [[int(r), int(c)] for r, c in extension_total_limit_cells[:128]],
            "door_extension_limit_excludes_seed_cells": True,
            **extension_occ_debug,
            "door_partition_cut_max_total_extension_m": float(max_total_extension_m),
            "door_line_local_neck_debug": neck_debug,
            "partition_cut_cells": int(len(partition_cells)),
            "visual_line_mask_cell_count": int(np.count_nonzero(visual_mask)),
            "partition_cut_mask_cell_count": int(np.count_nonzero(partition_mask)),
            "partition_geometry_cut_cells": int(len(geometry_partition_cells)),
            "partition_geometry_cut_mask_cell_count": int(np.count_nonzero(geometry_partition_mask)),
            "partition_cut_candidate_cells": [[int(r), int(c)] for r, c in cut_result.cut_cells],
            "walk_a_cells": [[int(r), int(c)] for r, c in extension_a],
            "walk_b_cells": [[int(r), int(c)] for r, c in extension_b],
            "walk_a_status": walk_a.status,
            "walk_b_status": walk_b.status,
            "walk_crossed_other_seed_cluster": bool(walk_a.hit_other_seed_cells or walk_b.hit_other_seed_cells),
            "walk_crossed_other_seed_cluster_cells": [[int(r), int(c)] for r, c in [*walk_a.hit_other_seed_cells, *walk_b.hit_other_seed_cells][:64]],
            "visual_status": str(visual_status),
            "partition_status": "accepted" if final_partition_reject_reason is None else final_partition_reject_reason,
            "partition_geometry_status": "accepted" if geometry_reject_reason is None else geometry_reject_reason,
            "door_partition_seed_on_line_cells": int(len(partition_seed_on_line_cells)),
            **cut_result.debug,
            **topology.to_dict(),
            **attachment.to_dict(),
            **closure_debug,
            **small_known_side_debug,
            "visual_accepted": visual_reject_reason is None,
            "partition_geometry_accepted": bool(geometry_partition_accepted),
            "partition_closure_attached": bool(attachment.attached),
            "partition_wall_to_wall_closure_verified": bool(closure_verified),
            "partition_topology_gain": bool(topology_gain_effective),
            "partition_effective_verified": bool(partition_effective_verified),
            "stable_memory_refresh_eligible": bool(visual_reject_reason is None or geometry_partition_accepted),
            "partition_topology_effective": bool(partition_topology_effective),
            "partition_accepted": bool(partition_effective_verified and final_partition_reject_reason is None),
            "partition_accepted_by_geometry_first": False,
            "partition_topology_reject_mode": str(getattr(cfg, "partition_topology_reject_mode", "warn_only")),
            "door_topology_warning": bool(topology_warning),
            "door_topology_warning_reason": topology_warning_reason,
            "partition_topology_accepted": bool(partition_topology_effective),
            "reject_reason_visual": visual_reject_reason,
            "reject_reason_partition": final_partition_reject_reason,
            "reject_reason_geometry_partition": geometry_reject_reason,
            "reject_reason_topology": None if bool(partition_topology_effective) else final_partition_reject_reason,
        },
    )


def validate_door_line_local_neck(
    line_cells: Sequence[tuple[int, int]],
    seed_mask: np.ndarray,
    free_map: np.ndarray,
    wall_anchor_map: np.ndarray,
    real_wall_barrier: np.ndarray,
    resolution_m: float,
    cfg: VoxelDoorDetectorConfig,
    *,
    completion_mode: str = "",
    orientation_source: str = "",
) -> tuple[bool, str | None, dict[str, object]]:
    shape = np.asarray(free_map, dtype=bool).shape
    cells = [tuple((int(r), int(c))) for r, c in line_cells]
    line = _cells_to_mask(cells, shape)
    seed = np.asarray(seed_mask, dtype=bool)
    free = np.asarray(free_map, dtype=bool)
    real_wall = np.asarray(real_wall_barrier, dtype=bool)
    wall = np.asarray(wall_anchor_map, dtype=bool)
    width_m = float(len(cells) * float(resolution_m))
    if str(completion_mode) == DOOR_COMPLETION_ONE_SEED_ONE_WALL:
        max_width = float(getattr(cfg, "one_seed_one_wall_visual_width_max_m", cfg.visual_width_max_m))
    elif str(completion_mode) == DOOR_COMPLETION_SEED_PAIR_BRIDGE:
        max_width = float(getattr(cfg, "seed_pair_bridge_visual_width_max_m", cfg.visual_width_max_m))
    else:
        max_width = float(cfg.visual_width_max_m)
    debug: dict[str, object] = {
        "door_line_local_neck_checked": True,
        "door_line_width_m": float(width_m),
        "door_line_max_width_m": float(max_width),
        "door_line_width_limit_enforced": False,
        "completion_mode": str(completion_mode),
        "orientation_source": str(orientation_source),
    }
    if not cells:
        return False, "door_line_empty", debug
    seed_overlap = bool(np.any(line & dilate(seed, max(1, int(getattr(cfg, "partition_cut_seed_dilation_cells", 1))))))
    debug["door_line_seed_overlap"] = bool(seed_overlap)
    if not seed_overlap:
        return False, "door_line_no_seed_overlap", debug
    inner = np.zeros(shape, dtype=bool)
    if len(cells) > 2:
        inner_cells = cells[1:-1]
        inner = _cells_to_mask(inner_cells, shape)
    inner_real_wall_cells = int(np.count_nonzero(inner & real_wall & ~dilate(seed, 1)))
    max_inner_real_wall_cells = max(0, int(getattr(cfg, "door_line_inner_real_wall_cells_max", 8)))
    debug["door_line_inner_real_wall_cells"] = int(inner_real_wall_cells)
    debug["door_line_inner_real_wall_cells_max"] = int(max_inner_real_wall_cells)
    if int(inner_real_wall_cells) > int(max_inner_real_wall_cells):
        return False, "door_line_crosses_real_wall", debug
    if "diag" in str(orientation_source) and np.count_nonzero(line & wall) < 2:
        return False, "door_diagonal_without_support", debug
    if np.any(inner):
        boundary = wall | real_wall | ~free
        dist = ndimage.distance_transform_edt(~boundary)
        max_open_free_m = float(np.max(dist[inner])) * float(resolution_m)
    else:
        max_open_free_m = 0.0
    debug["door_line_max_open_free_distance_m"] = float(max_open_free_m)
    debug["door_line_large_open_free_reject_enabled"] = bool(getattr(cfg, "reject_large_open_free_crossing", False))
    if bool(getattr(cfg, "reject_large_open_free_crossing", False)) and max_open_free_m > max(0.80, float(max_width) * 0.75):
        return False, "door_line_crosses_large_open_free", debug
    return True, None, debug


def _strong_seed_centerline_fallback_ok(
    *,
    seed_cells: Sequence[tuple[int, int]],
    seed_centerline: Sequence[tuple[int, int]],
    residual_max: float,
    thickness_m: float,
    resolution_m: float,
    cfg: VoxelDoorDetectorConfig,
) -> bool:
    if not bool(getattr(cfg, "enable_strong_seed_centerline_fallback", True)):
        return False
    if len(seed_cells) < int(getattr(cfg, "strong_seed_centerline_min_cells", 4)):
        return False
    if len(seed_centerline) < int(getattr(cfg, "strong_seed_centerline_min_length_cells", 3)):
        return False
    thickness_cells = max(1.0, float(thickness_m) / max(float(resolution_m), 1e-9))
    elongation = float(len(seed_centerline)) / float(thickness_cells)
    if elongation + 1e-9 < float(getattr(cfg, "strong_seed_centerline_min_elongation", 1.6)):
        return False
    if float(residual_max) > float(getattr(cfg, "strong_seed_centerline_max_residual_cells", 1.25)):
        return False
    return True


def _walk_to_wall(
    *,
    start_rc: np.ndarray,
    direction: np.ndarray,
    wall_anchor_map: np.ndarray,
    wall_anchor_source_map: np.ndarray | None,
    same_cluster_mask: np.ndarray | None = None,
    other_door_mask: np.ndarray,
    unknown_map: np.ndarray | None = None,
    no_clearance_occupied_map: np.ndarray | None = None,
    resolution_m: float,
    cfg: VoxelDoorDetectorConfig,
) -> DoorAnchorWalkResult:
    max_steps = max(1, int(round(float(cfg.extend_max_m) / max(float(resolution_m), 1e-9))))
    direction = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return DoorAnchorWalkResult(None, [], "zero_direction", DOOR_ANCHOR_NONE, [], [], [], "zero_direction")
    direction = direction / norm
    unknown = np.zeros_like(wall_anchor_map, dtype=bool) if unknown_map is None else np.asarray(unknown_map, dtype=bool)
    no_clearance_occupied = (
        np.zeros_like(wall_anchor_map, dtype=bool)
        if no_clearance_occupied_map is None
        else np.asarray(no_clearance_occupied_map, dtype=bool)
    )
    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    other_seed_hits: list[tuple[int, int]] = []
    unknown_hits: list[tuple[int, int]] = []
    wall_hits: list[tuple[int, int]] = []
    unknown_run = 0
    max_unknown = max(0, int(getattr(cfg, "visual_walk_unknown_bridge_max_cells", 4)))
    strip_half_width = max(0, int(getattr(cfg, "door_walk_strip_half_width_cells", 1)))
    anchor_snap_radius = max(int(cfg.wall_anchor_radius_cells), int(getattr(cfg, "door_walk_anchor_snap_radius_cells", cfg.wall_anchor_radius_cells)))
    if bool(getattr(cfg, "door_walk_unknown_effective_ignores_anchor", True)) and anchor_snap_radius > 0:
        anchor_shadow = dilate(np.asarray(wall_anchor_map, dtype=bool), int(anchor_snap_radius))
    else:
        anchor_shadow = np.zeros_like(wall_anchor_map, dtype=bool)
    for step in range(1, max_steps + 1):
        rr, cc = np.rint(np.asarray(start_rc, dtype=np.float32) + direction * float(step)).astype(np.int32).tolist()
        r, c = int(rr), int(cc)
        if not (0 <= r < wall_anchor_map.shape[0] and 0 <= c < wall_anchor_map.shape[1]):
            return DoorAnchorWalkResult(None, cells, "out_of_bounds", DOOR_ANCHOR_NONE, other_seed_hits, unknown_hits, wall_hits, "out_of_bounds")
        if (r, c) not in seen:
            seen.add((r, c))
            cells.append((r, c))
        if same_cluster_mask is not None and bool(same_cluster_mask[r, c]):
            continue
        if (
            bool(getattr(cfg, "door_walk_no_clearance_occupied_anchor_enabled", False))
            and bool(no_clearance_occupied[r, c])
            and _no_clearance_occupied_anchor_supported_by_wall(
                (r, c),
                no_clearance_occupied,
                wall_anchor_map,
                max_radius_cells=int(getattr(cfg, "door_walk_no_clearance_occupied_anchor_support_radius_cells", 24)),
            )
        ):
            wall_hits.append((r, c))
            return DoorAnchorWalkResult(
                (r, c),
                cells,
                "hit_no_clearance_occupied_anchor",
                DOOR_ANCHOR_NO_CLEARANCE_OCCUPIED,
                other_seed_hits,
                unknown_hits,
                wall_hits,
                None,
            )
        anchor_hit, anchor_source = _wall_anchor_hit_in_strip(
            (r, c),
            direction,
            wall_anchor_map,
            wall_anchor_source_map,
            strip_half_width_cells=strip_half_width,
            anchor_radius_cells=anchor_snap_radius,
            min_cells=int(cfg.wall_anchor_min_cells),
        )
        if anchor_hit is not None:
            wall_hits.append(anchor_hit)
            return DoorAnchorWalkResult(anchor_hit, cells, "hit_wall", int(anchor_source), other_seed_hits, unknown_hits, wall_hits, None)
        if bool(other_door_mask[r, c]):
            other_seed_hits.append((r, c))
            if not bool(getattr(cfg, "visual_walk_ignore_other_seed_clusters", True)):
                return DoorAnchorWalkResult(None, cells, "extension_hits_other_door_cluster", DOOR_ANCHOR_NONE, other_seed_hits, unknown_hits, wall_hits, "extension_hits_other_door_cluster")
            continue
        unknown_effective = bool(unknown[r, c]) and not bool(anchor_shadow[r, c])
        if unknown_effective:
            unknown_hits.append((r, c))
            unknown_run += 1
            if bool(getattr(cfg, "visual_walk_continue_through_unknown", True)) and unknown_run <= max_unknown:
                continue
            return DoorAnchorWalkResult(None, cells, "too_much_unknown", DOOR_ANCHOR_NONE, other_seed_hits, unknown_hits, wall_hits, "too_much_unknown")
        unknown_run = 0
    return DoorAnchorWalkResult(None, cells, "extend_max_without_wall", DOOR_ANCHOR_NONE, other_seed_hits, unknown_hits, wall_hits, "extend_max_without_wall")


def _extension_cells_outside_seed(cells: Sequence[tuple[int, int]], seed_mask: np.ndarray) -> list[tuple[int, int]]:
    seed = np.asarray(seed_mask, dtype=bool)
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for rr, cc in cells:
        r, c = int(rr), int(cc)
        if not (0 <= r < seed.shape[0] and 0 <= c < seed.shape[1]):
            continue
        if bool(seed[r, c]) or (r, c) in seen:
            continue
        seen.add((r, c))
        out.append((r, c))
    return out


def _door_extension_no_clearance_occupied_reject(
    cells: Sequence[tuple[int, int]],
    *,
    no_clearance_occupied: np.ndarray,
    real_wall: np.ndarray,
    anchors: Sequence[tuple[int, int]],
    cfg: VoxelDoorDetectorConfig,
) -> tuple[bool, dict[str, object]]:
    threshold = max(0, int(getattr(cfg, "door_extension_no_clearance_occupied_reject_cells", 3)))
    surface_tolerance = max(
        0,
        int(getattr(cfg, "door_extension_no_clearance_occupied_surface_tolerance_cells", 1)),
    )
    occupied = np.asarray(no_clearance_occupied, dtype=bool)
    wall = np.asarray(real_wall, dtype=bool)
    anchor_set = {(int(r), int(c)) for r, c in anchors}
    counted: list[tuple[int, int]] = []
    surface_ignored: list[tuple[int, int]] = []
    for rr, cc in cells:
        r, c = int(rr), int(cc)
        if not (0 <= r < occupied.shape[0] and 0 <= c < occupied.shape[1]):
            continue
        if (r, c) in anchor_set or bool(wall[r, c]):
            continue
        if bool(occupied[r, c]):
            if _is_no_clearance_occupied_surface_cell(occupied, r, c, surface_tolerance):
                surface_ignored.append((r, c))
                continue
            counted.append((r, c))
    debug = {
        "door_extension_no_clearance_occupied_gate_enabled": bool(threshold > 0),
        "door_extension_no_clearance_occupied_reject_cells": int(threshold),
        "door_extension_no_clearance_occupied_surface_tolerance_cells": int(surface_tolerance),
        "door_extension_no_clearance_occupied_cells": int(len(counted)),
        "door_extension_no_clearance_occupied_hit_cells": [[int(r), int(c)] for r, c in counted[:64]],
        "door_extension_no_clearance_occupied_surface_ignored_cells": int(len(surface_ignored)),
        "door_extension_no_clearance_occupied_surface_ignored_hit_cells": [
            [int(r), int(c)] for r, c in surface_ignored[:64]
        ],
        "door_extension_no_clearance_occupied_excludes_seed": True,
        "door_extension_no_clearance_occupied_excludes_wall_anchor": True,
        "door_extension_no_clearance_occupied_excludes_real_wall": True,
    }
    return bool(threshold > 0 and len(counted) >= threshold), debug


def _is_no_clearance_occupied_surface_cell(
    occupied: np.ndarray,
    r: int,
    c: int,
    tolerance_cells: int,
) -> bool:
    tolerance = max(0, int(tolerance_cells))
    if tolerance <= 0:
        return False
    occ = np.asarray(occupied, dtype=bool)
    r0, r1 = max(0, int(r) - tolerance), min(occ.shape[0], int(r) + tolerance + 1)
    c0, c1 = max(0, int(c) - tolerance), min(occ.shape[1], int(c) + tolerance + 1)
    if r0 > int(r) - tolerance or c0 > int(c) - tolerance:
        return True
    if r1 < int(r) + tolerance + 1 or c1 < int(c) + tolerance + 1:
        return True
    return not bool(np.all(occ[r0:r1, c0:c1]))


def _batch_reject_conflicting_doors(
    candidates: list[VoxelDoorLineCandidate],
    all_seed_mask: np.ndarray,
    shape: tuple[int, int],
    cfg: VoxelDoorDetectorConfig,
    *,
    door_conflict_free_mask: np.ndarray | None = None,
) -> dict[str, object]:
    raw_seed_conflict_ignored_map = np.zeros(shape, dtype=bool)
    conflict_free = (
        np.asarray(door_conflict_free_mask, dtype=bool)
        if door_conflict_free_mask is not None
        else np.ones(shape, dtype=bool)
    )
    if conflict_free.shape != tuple(shape):
        raise ValueError("door_conflict_free_mask must match door candidate shape")
    provisional = [item for item in candidates if item.accepted]
    if not provisional:
        return {
            "voxel_door_conflict_policy": "accepted_doors_only_nav_free_seedline_overlap",
            "voxel_door_raw_seed_conflict_ignored_map": raw_seed_conflict_ignored_map,
            "voxel_door_raw_seed_conflict_ignored_cells": 0,
            "voxel_door_partition_reject_raw_seed_conflict_count": 0,
            "voxel_door_partition_nonfree_intersection_ignored_count": 0,
            "voxel_door_partition_nav_free_intersection_reject_count": 0,
        }
    line_masks = {
        item.candidate_id: (
            _cells_to_mask(item.seed_projected_centerline_cells or item.seed_cells, shape)
            & conflict_free
        )
        for item in provisional
    }
    for item in provisional:
        own_seed = _cells_to_mask(item.seed_cells, shape)
        same_door_seed_band = dilate(own_seed, max(0, int(getattr(cfg, "seed_cluster_merge_distance_cells", 0))))
        visual = _cells_to_mask(item.extended_centerline_cells or item.door_cut_cells or item.seed_projected_centerline_cells, shape)
        ignored = visual & np.asarray(all_seed_mask, dtype=bool) & ~same_door_seed_band
        raw_seed_conflict_ignored_map |= ignored
        item.debug["raw_seed_conflict_ignored_cells"] = int(np.count_nonzero(ignored))
        item.debug["voxel_door_conflict_policy"] = "accepted_doors_only_nav_free_seedline_overlap"
    nonfree_intersection_ignored_count = 0
    nav_free_intersection_reject_count = 0
    def conflict_key(item: VoxelDoorLineCandidate) -> tuple[float, int, int, int, int, int]:
        return (
            float(item.debug.get("score", 0.0)),
            1 if bool(item.debug.get("partition_effective_verified", item.debug.get("partition_accepted", False))) else 0,
            1 if bool(item.debug.get("partition_geometry_accepted", False)) else 0,
            int(len(item.seed_projected_centerline_cells or item.seed_cells)),
            int(len(item.door_cut_cells)),
            -int(item.candidate_id),
        )

    def reject_conflicting_item(item: VoxelDoorLineCandidate) -> None:
        item.door_cut_cells = []
        item.debug["partition_accepted"] = False
        item.debug["partition_effective_verified"] = False
        item.debug["partition_topology_effective"] = False
        item.debug["partition_topology_accepted"] = False
        item.debug["reject_reason_partition"] = "partition_intersects_other_door_candidate"
        item.debug["reject_reason_topology"] = "partition_intersects_other_door_candidate"

    provisional = [item for item in candidates if item.accepted and bool(item.debug.get("partition_accepted", False))]
    for idx, a in enumerate(provisional):
        for b in provisional[idx + 1 :]:
            if not (
                a.accepted
                and b.accepted
                and bool(a.debug.get("partition_accepted", False))
                and bool(b.debug.get("partition_accepted", False))
            ):
                continue
            overlap = line_masks[a.candidate_id] & line_masks[b.candidate_id]
            if not np.any(overlap):
                continue
            nav_free_intersection_reject_count += 1
            overlap_cells = int(np.count_nonzero(overlap))
            a.debug["partition_nav_free_intersection_cells"] = int(a.debug.get("partition_nav_free_intersection_cells", 0) or 0) + overlap_cells
            b.debug["partition_nav_free_intersection_cells"] = int(b.debug.get("partition_nav_free_intersection_cells", 0) or 0) + overlap_cells
            score_a = float(a.debug.get("score", 0.0))
            score_b = float(b.debug.get("score", 0.0))
            if abs(score_a - score_b) < 0.05:
                if conflict_key(a) >= conflict_key(b):
                    reject_conflicting_item(b)
                else:
                    reject_conflicting_item(a)
            elif score_a < score_b:
                reject_conflicting_item(a)
            else:
                reject_conflicting_item(b)
    return {
        "voxel_door_conflict_policy": "accepted_doors_only_nav_free_seedline_overlap",
        "voxel_door_raw_seed_conflict_ignored_map": raw_seed_conflict_ignored_map.astype(bool),
        "voxel_door_raw_seed_conflict_ignored_cells": int(np.count_nonzero(raw_seed_conflict_ignored_map)),
        "voxel_door_partition_reject_raw_seed_conflict_count": 0,
        "voxel_door_partition_nonfree_intersection_ignored_count": int(nonfree_intersection_ignored_count),
        "voxel_door_partition_nav_free_intersection_reject_count": int(nav_free_intersection_reject_count),
    }


def _infer_single_seed_direction(
    seed_rc: tuple[int, int],
    wall_clean: np.ndarray,
    resolution_m: float,
    cfg: VoxelDoorDetectorConfig,
) -> np.ndarray | None:
    r, c = int(seed_rc[0]), int(seed_rc[1])
    max_steps = max(1, int(round(float(cfg.single_seed_wall_search_radius_m) / max(float(resolution_m), 1e-9))))
    axes = [np.asarray([0.0, 1.0], dtype=np.float32), np.asarray([1.0, 0.0], dtype=np.float32)]
    options: list[tuple[float, np.ndarray]] = []
    for axis in axes:
        distances = []
        ok = True
        for sign in (-1.0, 1.0):
            hit = None
            for step in range(1, max_steps + 1):
                rr = int(round(r + sign * float(axis[0]) * step))
                cc = int(round(c + sign * float(axis[1]) * step))
                if not (0 <= rr < wall_clean.shape[0] and 0 <= cc < wall_clean.shape[1]):
                    break
                if _wall_anchor_at((rr, cc), wall_clean, int(cfg.wall_anchor_radius_cells), int(cfg.wall_anchor_min_cells)):
                    hit = step
                    break
            if hit is None:
                ok = False
                break
            distances.append(float(hit))
        if ok:
            width_m = float((distances[0] + distances[1] + 1.0) * resolution_m)
            if not bool(cfg.enforce_seed_door_width_limits) or float(cfg.door_width_min_m) <= width_m <= float(cfg.door_width_max_m):
                options.append((width_m, axis))
    if not options:
        return None
    options.sort(key=lambda item: item[0])
    return options[0][1]


def _project_seed_component_to_centerline(
    pts: np.ndarray,
    center: np.ndarray,
    major: np.ndarray,
    minor: np.ndarray,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    rel = pts - center[None, :]
    t = np.dot(rel, major)
    n = np.dot(rel, minor)
    center_minor = float(np.mean(n)) if n.size else 0.0
    t_min = int(np.floor(float(np.min(t)))) if t.size else 0
    t_max = int(np.ceil(float(np.max(t)))) if t.size else 0
    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for tt in range(t_min, t_max + 1):
        p = center + major * float(tt) + minor * center_minor
        r, c = np.rint(p).astype(np.int32).tolist()
        if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]:
            key = (int(r), int(c))
            if key not in seen:
                seen.add(key)
                cells.append(key)
    if not cells and pts.size:
        r, c = np.rint(center).astype(np.int32).tolist()
        if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]:
            cells.append((int(r), int(c)))
    return cells


def _line_cells(p0: tuple[int, int] | np.ndarray, p1: tuple[int, int] | np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    a = np.rint(np.asarray(p0, dtype=np.float32)).astype(np.int32)
    b = np.rint(np.asarray(p1, dtype=np.float32)).astype(np.int32)
    steps = max(abs(int(b[0] - a[0])), abs(int(b[1] - a[1]))) + 1
    if steps <= 1:
        cells = [(int(a[0]), int(a[1]))]
    else:
        rows = np.rint(np.linspace(int(a[0]), int(b[0]), steps)).astype(np.int32)
        cols = np.rint(np.linspace(int(a[1]), int(b[1]), steps)).astype(np.int32)
        cells = [(int(r), int(c)) for r, c in zip(rows, cols)]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in cells:
        if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1] and (int(r), int(c)) not in seen:
            seen.add((int(r), int(c)))
            out.append((int(r), int(c)))
    return out


def build_door_partition_cut(
    *,
    full_line_cells: Sequence[tuple[int, int]],
    seed_mask: np.ndarray,
    accepted_seed_mask: np.ndarray | None = None,
    partition_free: np.ndarray,
    partition_unknown: np.ndarray | None = None,
    real_wall_barrier: np.ndarray,
    max_bridge_gap_cells: int = 2,
    max_nonfree_bridge_gap_cells: int = 0,
    include_seed_cells_in_cut: bool = True,
    seed_dilation_cells: int = 1,
    min_cut_cells: int = 1,
) -> DoorPartitionCutResult:
    shape = np.asarray(partition_free, dtype=bool).shape
    free = np.asarray(partition_free, dtype=bool)
    unknown = np.zeros(shape, dtype=bool) if partition_unknown is None else np.asarray(partition_unknown, dtype=bool)
    real_wall = np.asarray(real_wall_barrier, dtype=bool)
    seed = np.asarray(seed_mask, dtype=bool)
    accepted_seed = seed if accepted_seed_mask is None else np.asarray(accepted_seed_mask, dtype=bool)
    seed_band = dilate(seed | accepted_seed, int(seed_dilation_cells)) if bool(include_seed_cells_in_cut) else np.zeros_like(seed, dtype=bool)
    cuttable = free | seed_band | accepted_seed
    ordered = [(int(r), int(c)) for r, c in full_line_cells if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]]
    valid = np.asarray([bool(cuttable[r, c] and not real_wall[r, c]) for r, c in ordered], dtype=bool)
    unknown_bridgeable = np.asarray([bool(unknown[r, c] and not real_wall[r, c]) for r, c in ordered], dtype=bool)
    nonfree_bridgeable = np.asarray([bool((not free[r, c]) and (not unknown[r, c]) and (not seed_band[r, c]) and (not accepted_seed[r, c]) and not real_wall[r, c]) for r, c in ordered], dtype=bool)
    bridged, bridged_indices = _bridge_valid_door_line_gaps_v16(
        valid,
        int(max_bridge_gap_cells),
        unknown_bridgeable=unknown_bridgeable,
        max_nonfree_gap_cells=int(max_nonfree_bridge_gap_cells),
        nonfree_bridgeable=nonfree_bridgeable,
    )
    seed_indices = [idx for idx, (r, c) in enumerate(ordered) if bool(seed_band[r, c])]
    if not ordered or not np.any(bridged):
        out = np.zeros(shape, dtype=bool)
        return DoorPartitionCutResult(
            mask=out,
            ordered_cells=ordered,
            cut_cells=[],
            bridged_cells=[],
            debug={
                "door_partition_cut_bridge_gap_count": int(len(bridged_indices)),
                "door_partition_cut_empty_reason": "no_valid_free_or_seed_cells",
            },
        )
    center_idx = int(np.median(seed_indices)) if seed_indices else int(len(ordered) // 2)
    if not bool(bridged[center_idx]):
        valid_indices = np.flatnonzero(bridged)
        center_idx = int(valid_indices[int(np.argmin(np.abs(valid_indices - center_idx)))])
    lo = center_idx
    while lo - 1 >= 0 and bool(bridged[lo - 1]):
        lo -= 1
    hi = center_idx
    while hi + 1 < len(bridged) and bool(bridged[hi + 1]):
        hi += 1
    interval = ordered[lo : hi + 1]
    cut_cells = [
        ordered[idx]
        for idx in range(lo, hi + 1)
        if bool((cuttable[ordered[idx][0], ordered[idx][1]] or unknown[ordered[idx][0], ordered[idx][1]] or nonfree_bridgeable[idx]) and not real_wall[ordered[idx][0], ordered[idx][1]])
    ]
    if len(cut_cells) < max(1, int(min_cut_cells)):
        cut_cells = []
    out = _cells_to_mask(cut_cells, shape)
    bridged_cells = [ordered[idx] for idx in bridged_indices if 0 <= int(idx) < len(ordered)]
    return DoorPartitionCutResult(
        mask=out.astype(bool),
        ordered_cells=ordered,
        cut_cells=cut_cells,
        bridged_cells=bridged_cells,
        debug={
            "door_partition_cut_bridge_gap_count": int(len(bridged_indices)),
            "door_partition_cut_ordered_cells": int(len(ordered)),
            "door_partition_cut_interval_cells": int(len(interval)),
            "door_partition_cut_cells": int(len(cut_cells)),
            "door_partition_cut_seed_cells": int(sum(1 for r, c in cut_cells if bool(accepted_seed[r, c] or seed_band[r, c]))),
            "door_partition_cut_bridged_cells": [[int(r), int(c)] for r, c in bridged_cells[:64]],
        },
    )


def build_door_partition_cut_v30(
    *,
    full_line_cells: Sequence[tuple[int, int]],
    seed_mask: np.ndarray,
    accepted_seed_mask: np.ndarray | None,
    partition_free: np.ndarray,
    partition_unknown: np.ndarray,
    real_wall_barrier: np.ndarray,
    anchor_a: tuple[int, int] | None,
    anchor_b: tuple[int, int] | None,
    max_unknown_bridge_gap_cells: int,
    max_nonfree_bridge_gap_cells: int,
    max_endpoint_wall_gap_cells: int,
    seed_dilation_cells: int,
    min_cut_cells: int,
) -> DoorPartitionCutResult:
    shape = np.asarray(partition_free, dtype=bool).shape
    if anchor_a is None or anchor_b is None:
        result = build_door_partition_cut(
            full_line_cells=full_line_cells,
            seed_mask=seed_mask,
            accepted_seed_mask=accepted_seed_mask,
            partition_free=partition_free,
            partition_unknown=partition_unknown,
            real_wall_barrier=real_wall_barrier,
            max_bridge_gap_cells=int(max_unknown_bridge_gap_cells),
            max_nonfree_bridge_gap_cells=int(max_nonfree_bridge_gap_cells),
            seed_dilation_cells=int(seed_dilation_cells),
            min_cut_cells=int(min_cut_cells),
        )
        result.debug.update(
            {
                "door_partition_cut_v30_mode": "legacy_seed_interval",
                "door_partition_cut_v30_anchor_to_anchor": False,
                "door_partition_cut_v30_closed_to_wall": False,
                "door_partition_cut_v30_reject_reason": "one_anchor_or_seed_only",
            }
        )
        return result

    free = np.asarray(partition_free, dtype=bool)
    unknown = np.asarray(partition_unknown, dtype=bool)
    real_wall = np.asarray(real_wall_barrier, dtype=bool)
    seed = np.asarray(seed_mask, dtype=bool)
    accepted_seed = seed if accepted_seed_mask is None else np.asarray(accepted_seed_mask, dtype=bool)
    seed_band = dilate(seed | accepted_seed, int(seed_dilation_cells))
    ordered = [(int(r), int(c)) for r, c in full_line_cells if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]]
    if len(ordered) < 3:
        out = np.zeros(shape, dtype=bool)
        return DoorPartitionCutResult(
            mask=out,
            ordered_cells=ordered,
            cut_cells=[],
            bridged_cells=[],
            debug={
                "door_partition_cut_v30_mode": "anchor_to_anchor",
                "door_partition_cut_v30_anchor_to_anchor": True,
                "door_partition_cut_v30_closed_to_wall": False,
                "door_partition_cut_v30_reject_reason": "door_partition_cut_empty",
            },
        )
    ia = _nearest_line_index(ordered, anchor_a)
    ib = _nearest_line_index(ordered, anchor_b)
    lo, hi = sorted((int(ia), int(ib)))
    interval_indices = list(range(lo + 1, hi))
    if not interval_indices:
        out = np.zeros(shape, dtype=bool)
        return DoorPartitionCutResult(
            mask=out,
            ordered_cells=ordered,
            cut_cells=[],
            bridged_cells=[],
            debug={
                "door_partition_cut_v30_mode": "anchor_to_anchor",
                "door_partition_cut_v30_anchor_to_anchor": True,
                "door_partition_cut_v30_closed_to_wall": False,
                "door_partition_cut_v30_reject_reason": "door_partition_cut_empty",
                "door_partition_cut_v30_anchor_indices": [int(ia), int(ib)],
            },
        )
    interval = [ordered[idx] for idx in interval_indices]
    cuttable = free | seed_band | accepted_seed
    valid = np.asarray([bool(cuttable[r, c] and not real_wall[r, c]) for r, c in interval], dtype=bool)
    unknown_bridgeable = np.asarray([bool(unknown[r, c] and not real_wall[r, c]) for r, c in interval], dtype=bool)
    nonfree_bridgeable = np.asarray(
        [bool((not free[r, c]) and (not unknown[r, c]) and (not seed_band[r, c]) and (not accepted_seed[r, c]) and not real_wall[r, c]) for r, c in interval],
        dtype=bool,
    )
    bridged, bridged_local_indices = _bridge_valid_door_line_gaps_v16(
        valid,
        int(max_unknown_bridge_gap_cells),
        unknown_bridgeable=unknown_bridgeable,
        max_nonfree_gap_cells=int(max_nonfree_bridge_gap_cells),
        nonfree_bridgeable=nonfree_bridgeable,
    )
    cut_cells = [
        interval[idx]
        for idx in range(len(interval))
        if bool(bridged[idx] and not real_wall[interval[idx][0], interval[idx][1]])
    ]
    if len(cut_cells) < max(1, int(min_cut_cells)):
        cut_cells = []
    out = _cells_to_mask(cut_cells, shape)
    if cut_cells:
        cut_indices = [idx for idx in interval_indices if ordered[idx] in set(cut_cells)]
        left_gap = _line_gap_to_wall(ordered, min(cut_indices), -1, real_wall) if cut_indices else 10**9
        right_gap = _line_gap_to_wall(ordered, max(cut_indices), 1, real_wall) if cut_indices else 10**9
    else:
        left_gap = right_gap = 10**9
    max_gap = max(0, int(max_endpoint_wall_gap_cells))
    closed = bool(cut_cells and int(left_gap) <= max_gap and int(right_gap) <= max_gap)
    bridged_cells = [interval[idx] for idx in bridged_local_indices if 0 <= int(idx) < len(interval)]
    return DoorPartitionCutResult(
        mask=out.astype(bool),
        ordered_cells=ordered,
        cut_cells=cut_cells,
        bridged_cells=bridged_cells,
        debug={
            "door_partition_cut_v30_mode": "anchor_to_anchor",
            "door_partition_cut_v30_anchor_to_anchor": True,
            "door_partition_cut_v30_anchor_indices": [int(ia), int(ib)],
            "door_partition_cut_v30_interval_cells": int(len(interval)),
            "door_partition_cut_v30_closed_to_wall": bool(closed),
            "door_partition_cut_v30_left_gap_cells": int(left_gap),
            "door_partition_cut_v30_right_gap_cells": int(right_gap),
            "door_partition_cut_v30_reject_reason": None if bool(closed) else "door_cut_not_closed_to_wall",
            "door_partition_cut_bridge_gap_count": int(len(bridged_local_indices)),
            "door_partition_cut_ordered_cells": int(len(ordered)),
            "door_partition_cut_interval_cells": int(len(interval)),
            "door_partition_cut_cells": int(len(cut_cells)),
            "door_partition_cut_seed_cells": int(sum(1 for r, c in cut_cells if bool(accepted_seed[r, c] or seed_band[r, c]))),
            "door_partition_cut_bridged_cells": [[int(r), int(c)] for r, c in bridged_cells[:64]],
        },
    )


def verify_cut_wall_to_wall_closure_v30(
    *,
    cut_mask: np.ndarray,
    full_line_cells: Sequence[tuple[int, int]],
    anchor_a: tuple[int, int] | None,
    anchor_b: tuple[int, int] | None,
    real_wall_barrier: np.ndarray,
    base_partition_free: np.ndarray,
    max_endpoint_gap_cells: int,
    connectivity: int = 4,
    min_side_area_cells: int = 2,
    min_side_width_cells: int = 1,
) -> tuple[bool, dict[str, object]]:
    cut = np.asarray(cut_mask, dtype=bool)
    wall = np.asarray(real_wall_barrier, dtype=bool)
    free = np.asarray(base_partition_free, dtype=bool)
    debug: dict[str, object] = {
        "partition_closure_check_v30": True,
        "partition_closure_has_two_anchors": bool(anchor_a is not None and anchor_b is not None),
    }
    if cut.shape != wall.shape or cut.shape != free.shape:
        raise ValueError("door closure maps must share one HxW shape")
    if anchor_a is None or anchor_b is None:
        debug["partition_closure_reject_reason"] = "missing_two_anchors"
        return False, debug
    ordered = [(int(r), int(c)) for r, c in full_line_cells if 0 <= int(r) < cut.shape[0] and 0 <= int(c) < cut.shape[1]]
    cut_indices = [idx for idx, (r, c) in enumerate(ordered) if bool(cut[r, c])]
    if not ordered or not cut_indices:
        debug["partition_closure_reject_reason"] = "door_partition_cut_empty"
        return False, debug
    left_gap = _line_gap_to_wall(ordered, min(cut_indices), -1, wall)
    right_gap = _line_gap_to_wall(ordered, max(cut_indices), 1, wall)
    endpoint_attached = int(left_gap) <= int(max_endpoint_gap_cells) and int(right_gap) <= int(max_endpoint_gap_cells)
    crosses_free = bool(np.any(cut & free))
    before_free = free & ~wall
    roi = _mask_bbox(dilate(cut, 3), cut.shape)
    r0, r1, c0, c1 = roi
    before_local = before_free[r0:r1, c0:c1]
    cut_local = cut[r0:r1, c0:c1]
    before_labels, before_n = ndimage.label(before_local, structure=conn(int(connectivity)))
    after_local = before_local & ~cut_local
    after_labels, after_n = ndimage.label(after_local, structure=conn(int(connectivity)))
    adjacent = dilate(cut_local, 1) & after_local
    touched = sorted({int(v) for v in np.unique(after_labels[adjacent]) if int(v) > 0})
    side_areas: list[int] = []
    side_widths: list[float] = []
    for label in touched:
        area, width = _door_side_area_and_width_cells(after_labels == int(label))
        side_areas.append(int(area))
        side_widths.append(float(width))
    min_area = max(1, int(min_side_area_cells))
    min_width = max(1, int(min_side_width_cells))
    side_ok = len(touched) >= 2 and not any(area < min_area for area in side_areas[:2]) and not any(width < min_width for width in side_widths[:2])
    local_gain = int(after_n) > int(before_n)
    verified = bool(endpoint_attached and crosses_free and (local_gain or side_ok))
    debug.update(
        {
            "partition_closure_left_gap_cells": int(left_gap),
            "partition_closure_right_gap_cells": int(right_gap),
            "partition_closure_endpoint_attached": bool(endpoint_attached),
            "partition_closure_crosses_free": bool(crosses_free),
            "partition_closure_local_gain": bool(local_gain),
            "partition_closure_touched_labels": [int(v) for v in touched],
            "partition_closure_side_areas": [int(v) for v in side_areas],
            "partition_closure_side_widths_cells": [float(v) for v in side_widths],
            "partition_closure_side_ok": bool(side_ok),
            "partition_closure_reject_reason": None if verified else "door_cut_not_strict_wall_to_wall_closure",
        }
    )
    return verified, debug


def validate_door_partition_small_known_side(
    *,
    cut_mask: np.ndarray,
    base_partition_free: np.ndarray,
    partition_unknown: np.ndarray,
    partition_real_wall_map: np.ndarray,
    resolution_m: float,
    cfg: VoxelDoorDetectorConfig,
) -> tuple[bool, dict[str, object]]:
    topology_enabled = bool(getattr(cfg, "partition_topology_enabled", True))
    enabled = bool(getattr(cfg, "partition_reject_small_known_side_enabled", True))
    area_threshold = float(getattr(cfg, "partition_small_known_side_area_m2", 2.0))
    unknown_threshold = float(getattr(cfg, "partition_small_known_side_unknown_ratio_max", 0.20))
    dilation_cells = max(1, int(getattr(cfg, "partition_small_known_side_boundary_dilation_cells", 1)))
    debug: dict[str, object] = {
        "partition_small_known_side_gate_enabled": bool(enabled),
        "partition_small_known_side_requires_topology": False,
        "partition_small_known_side_topology_enabled": bool(topology_enabled),
        "partition_small_known_side_area_threshold_m2": float(area_threshold),
        "partition_small_known_side_unknown_ratio_threshold": float(unknown_threshold),
        "partition_small_known_side_boundary_dilation_cells": int(dilation_cells),
        "partition_small_known_side_components": [],
    }
    cut = np.asarray(cut_mask, dtype=bool)
    free = np.asarray(base_partition_free, dtype=bool)
    unknown = np.asarray(partition_unknown, dtype=bool)
    wall = np.asarray(partition_real_wall_map, dtype=bool)
    if cut.shape != free.shape or cut.shape != unknown.shape or cut.shape != wall.shape:
        raise ValueError("door small-known-side maps must share one HxW shape")
    if not enabled or area_threshold <= 0.0 or not np.any(cut):
        debug["partition_small_known_side_rejected"] = False
        return False, debug

    before = free & ~wall
    after = before & ~cut
    labels, _count = ndimage.label(after, structure=conn(4))
    adjacent = dilate(cut, 1) & after
    touched = sorted({int(v) for v in np.unique(labels[adjacent]) if int(v) > 0})
    debug["partition_small_known_side_touched_labels"] = [int(v) for v in touched]
    if len(touched) < 2:
        debug["partition_small_known_side_rejected"] = False
        return False, debug

    components: list[dict[str, object]] = []
    for label in touched:
        comp = labels == int(label)
        area_cells = int(np.count_nonzero(comp))
        area_m2 = float(area_cells) * float(resolution_m) ** 2
        ring = dilate(comp, dilation_cells) & ~comp
        ring_cells = int(np.count_nonzero(ring))
        unknown_cells = int(np.count_nonzero(ring & unknown))
        unknown_ratio = 0.0 if ring_cells <= 0 else float(unknown_cells) / float(ring_cells)
        components.append(
            {
                "label": int(label),
                "area_cells": int(area_cells),
                "area_m2": float(area_m2),
                "boundary_cells": int(ring_cells),
                "boundary_unknown_cells": int(unknown_cells),
                "boundary_unknown_ratio": float(unknown_ratio),
                "rejected": bool(area_m2 < area_threshold and unknown_ratio < unknown_threshold),
            }
        )
    components.sort(key=lambda item: int(item.get("area_cells", 0)), reverse=True)
    main = components[:2]
    rejected = any(bool(item.get("rejected", False)) for item in main)
    debug["partition_small_known_side_components"] = main
    debug["partition_small_known_side_rejected"] = bool(rejected)
    debug["partition_small_known_side_reject_reason"] = (
        "door_partition_small_known_side_low_unknown" if rejected else None
    )
    return bool(rejected), debug


def _nearest_line_index(cells: Sequence[tuple[int, int]], target: tuple[int, int]) -> int:
    tr, tc = int(target[0]), int(target[1])
    best_idx = 0
    best_dist = 10**9
    for idx, (r, c) in enumerate(cells):
        dist = abs(int(r) - tr) + abs(int(c) - tc)
        if int(dist) < int(best_dist):
            best_idx = int(idx)
            best_dist = int(dist)
    return int(best_idx)


def validate_door_partition_cut_topology(
    *,
    door_cut_mask: np.ndarray,
    base_partition_free: np.ndarray,
    accepted_seed_mask: np.ndarray,
    partition_real_wall_map: np.ndarray,
    connectivity: int = 4,
    min_side_area_cells: int = 4,
    min_side_width_cells: int = 2,
    local_radius_cells: int = 18,
    allow_anchor_closure_without_global_gain: bool = True,
    allow_neck_cut_without_global_gain: bool = True,
    touches_two_anchors: bool = False,
    seed_overlap: bool = False,
) -> DoorTopologyValidationResult:
    cut = np.asarray(door_cut_mask, dtype=bool)
    base = np.asarray(base_partition_free, dtype=bool)
    seed = np.asarray(accepted_seed_mask, dtype=bool)
    wall = np.asarray(partition_real_wall_map, dtype=bool)
    if cut.shape != base.shape or seed.shape != base.shape or wall.shape != base.shape:
        raise ValueError("door topology maps must share one HxW shape")
    if not np.any(cut):
        return DoorTopologyValidationResult(False, "door_partition_cut_empty", 0, 0, [], 0, [], [])
    before_free_global = base & ~wall
    before_labels_global, before_n_global = ndimage.label(before_free_global, structure=conn(int(connectivity)))
    after_free_global = before_free_global & ~cut
    _after_labels_global, after_n_global = ndimage.label(after_free_global, structure=conn(int(connectivity)))
    roi = _mask_bbox(dilate(cut, max(1, int(local_radius_cells))), cut.shape)
    r0, r1, c0, c1 = roi
    before_free = before_free_global[r0:r1, c0:c1]
    cut_local = cut[r0:r1, c0:c1]
    before_labels, before_n = ndimage.label(before_free, structure=conn(int(connectivity)))
    after_free = before_free & ~cut_local
    after_labels, after_n = ndimage.label(after_free, structure=conn(int(connectivity)))
    adjacent = dilate(cut_local, 1) & after_free
    touched = sorted({int(v) for v in np.unique(after_labels[adjacent]) if int(v) > 0})
    side_areas: list[int] = []
    side_widths: list[float] = []
    for label in touched:
        comp = after_labels == int(label)
        area, width = _door_side_area_and_width_cells(comp)
        side_areas.append(int(area))
        side_widths.append(float(width))
    min_area = max(1, int(min_side_area_cells))
    min_width = max(1, int(min_side_width_cells))
    side_ok = len(touched) >= 2 and not any(area < min_area for area in side_areas[:2]) and not any(width < min_width for width in side_widths[:2])
    global_gain = int(after_n_global) > int(before_n_global)
    local_gain = int(after_n) > int(before_n)
    if side_ok and (global_gain or local_gain):
        reason = None
        accepted = True
    elif side_ok and bool(touches_two_anchors) and bool(allow_anchor_closure_without_global_gain):
        reason = None
        accepted = True
    elif side_ok and bool(seed_overlap) and bool(allow_neck_cut_without_global_gain):
        reason = None
        accepted = True
    elif int(after_n) <= int(before_n):
        reason = "door_cut_no_topology_gain"
        accepted = False
    elif len(touched) < 2:
        reason = "door_cut_no_two_sides"
        accepted = False
    elif any(area < min_area for area in side_areas[:2]) or any(width < min_width for width in side_widths[:2]):
        reason = "door_cut_tiny_side"
        accepted = False
    return DoorTopologyValidationResult(
        topology_accepted=bool(accepted),
        reject_reason=reason,
        before_components=int(before_n_global),
        after_components=int(after_n_global),
        touched_labels=[int(v) for v in touched],
        new_component_count=max(0, int(after_n) - int(before_n)),
        side_component_areas=[int(v) for v in side_areas],
        side_component_widths_cells=[float(v) for v in side_widths],
    )


def _door_side_area_and_width_cells(mask: np.ndarray) -> tuple[int, float]:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return 0, 0.0
    row_span = int(rows.max() - rows.min() + 1)
    col_span = int(cols.max() - cols.min() + 1)
    # For topology auditing, a side is tiny only when its overall extent is tiny.
    # Thin but long corridors should not fail door validation just because one
    # cross-section is one cell wide in a synthetic or partially observed map.
    return int(rows.size), float(max(row_span, col_span))


def validate_door_cut_wall_attachment(
    *,
    cut_mask: np.ndarray,
    full_line_cells: Sequence[tuple[int, int]],
    real_wall_barrier: np.ndarray,
    seed_mask: np.ndarray,
    max_endpoint_gap_cells: int = 1,
) -> DoorAttachmentValidation:
    cut = np.asarray(cut_mask, dtype=bool)
    wall = np.asarray(real_wall_barrier, dtype=bool)
    seed = np.asarray(seed_mask, dtype=bool)
    if cut.shape != wall.shape or seed.shape != wall.shape:
        raise ValueError("door attachment masks must share one HxW shape")
    ordered = [(int(r), int(c)) for r, c in full_line_cells if 0 <= int(r) < cut.shape[0] and 0 <= int(c) < cut.shape[1]]
    if not ordered or not np.any(cut):
        return DoorAttachmentValidation(False, False, False, 10**9, 10**9, "door_cut_empty")
    cut_indices = [idx for idx, (r, c) in enumerate(ordered) if bool(cut[r, c])]
    if not cut_indices:
        return DoorAttachmentValidation(False, False, False, 10**9, 10**9, "door_cut_empty")
    lo, hi = int(min(cut_indices)), int(max(cut_indices))
    left_gap = _line_gap_to_wall(ordered, lo, -1, wall)
    right_gap = _line_gap_to_wall(ordered, hi, 1, wall)
    max_gap = max(0, int(max_endpoint_gap_cells))
    left_attached = int(left_gap) <= max_gap
    right_attached = int(right_gap) <= max_gap
    neck_cells = ordered[lo : hi + 1]
    neck_mask = _cells_to_mask(neck_cells, cut.shape)
    neck_has_seed_or_cut = bool(np.any(neck_mask & (seed | cut)))
    attached = bool(left_attached and right_attached and neck_has_seed_or_cut)
    reason = None
    if not neck_has_seed_or_cut:
        reason = "door_cut_no_seed_or_free_neck"
    elif not left_attached or not right_attached:
        reason = "door_cut_not_wall_attached"
    return DoorAttachmentValidation(
        attached=attached,
        left_attached=bool(left_attached),
        right_attached=bool(right_attached),
        left_gap_cells=int(left_gap),
        right_gap_cells=int(right_gap),
        reject_reason=reason,
    )


def _line_gap_to_wall(ordered: Sequence[tuple[int, int]], start_idx: int, direction: int, wall: np.ndarray) -> int:
    gap = 0
    idx = int(start_idx)
    while 0 <= idx < len(ordered):
        r, c = ordered[idx]
        if bool(wall[int(r), int(c)]):
            return int(gap)
        idx += int(direction)
        gap += 1
    return 10**9


def _mask_bbox(mask: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return 0, int(shape[0]), 0, int(shape[1])
    return (
        max(0, int(rows.min())),
        min(int(shape[0]), int(rows.max()) + 1),
        max(0, int(cols.min())),
        min(int(shape[1]), int(cols.max()) + 1),
    )


def _bbox_from_cells(cells: Sequence[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    if not cells:
        return 0, 0, 0, 0
    rows = np.asarray([int(r) for r, _c in cells], dtype=np.int32)
    cols = np.asarray([int(c) for _r, c in cells], dtype=np.int32)
    return (
        max(0, int(rows.min())),
        min(int(shape[0]) - 1, int(rows.max())),
        max(0, int(cols.min())),
        min(int(shape[1]) - 1, int(cols.max())),
    )


def _bridge_valid_door_line_gaps(values: np.ndarray, max_gap_cells: int, *, bridgeable: np.ndarray | None = None) -> tuple[np.ndarray, list[int]]:
    out = np.asarray(values, dtype=bool).copy()
    bridge = np.ones_like(out, dtype=bool) if bridgeable is None else np.asarray(bridgeable, dtype=bool)
    bridged: list[int] = []
    if int(max_gap_cells) <= 0 or out.size <= 2:
        return out, bridged
    idx = 0
    while idx < out.size:
        if bool(out[idx]):
            idx += 1
            continue
        start = idx
        while idx < out.size and not bool(out[idx]):
            idx += 1
        end = idx
        if start == 0 or end >= out.size:
            continue
        if bool(out[start - 1]) and bool(out[end]) and (end - start) <= int(max_gap_cells) and bool(np.all(bridge[start:end])):
            out[start:end] = True
            bridged.extend(range(start, end))
    return out, bridged


def _bridge_valid_door_line_gaps_v16(
    values: np.ndarray,
    max_unknown_gap_cells: int,
    *,
    unknown_bridgeable: np.ndarray,
    max_nonfree_gap_cells: int,
    nonfree_bridgeable: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    out = np.asarray(values, dtype=bool).copy()
    unknown = np.asarray(unknown_bridgeable, dtype=bool)
    nonfree = np.asarray(nonfree_bridgeable, dtype=bool)
    if out.shape != unknown.shape or out.shape != nonfree.shape:
        raise ValueError("door partition bridge arrays must have the same shape")
    bridged: list[int] = []
    if out.size <= 2:
        return out, bridged
    idx = 0
    while idx < out.size:
        if bool(out[idx]):
            idx += 1
            continue
        start = idx
        while idx < out.size and not bool(out[idx]):
            idx += 1
        end = idx
        if start == 0 or end >= out.size:
            continue
        unknown_count = int(np.count_nonzero(unknown[start:end]))
        nonfree_count = int(np.count_nonzero(nonfree[start:end]))
        hard_block_count = int((end - start) - unknown_count - nonfree_count)
        if (
            bool(out[start - 1])
            and bool(out[end])
            and hard_block_count == 0
            and unknown_count <= int(max_unknown_gap_cells)
            and nonfree_count <= int(max_nonfree_gap_cells)
        ):
            out[start:end] = True
            bridged.extend(range(start, end))
    return out, bridged


def _cells_to_mask(cells: Sequence[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in cells:
        if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]:
            out[int(r), int(c)] = True
    return out


def _wall_anchor_at(cell: tuple[int, int], wall: np.ndarray, radius: int, min_cells: int) -> bool:
    r, c = int(cell[0]), int(cell[1])
    r0, r1 = max(0, r - int(radius)), min(wall.shape[0], r + int(radius) + 1)
    c0, c1 = max(0, c - int(radius)), min(wall.shape[1], c + int(radius) + 1)
    return int(np.count_nonzero(wall[r0:r1, c0:c1])) >= max(1, int(min_cells))


def _no_clearance_occupied_anchor_supported_by_wall(
    cell: tuple[int, int],
    no_clearance_occupied: np.ndarray,
    wall_anchor_map: np.ndarray,
    *,
    max_radius_cells: int,
) -> bool:
    occupied = np.asarray(no_clearance_occupied, dtype=bool)
    wall = np.asarray(wall_anchor_map, dtype=bool)
    r0, c0 = int(cell[0]), int(cell[1])
    if not (0 <= r0 < occupied.shape[0] and 0 <= c0 < occupied.shape[1]):
        return False
    if not bool(occupied[r0, c0]):
        return False
    if bool(wall[r0, c0]):
        return True
    max_radius = max(0, int(max_radius_cells))
    queue: deque[tuple[int, int, int]] = deque([(r0, c0, 0)])
    seen: set[tuple[int, int]] = {(r0, c0)}
    while queue:
        r, c, dist = queue.popleft()
        if bool(wall[r, c]):
            return True
        if int(dist) >= max_radius:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1]):
                    continue
                if (nr, nc) in seen or not bool(occupied[nr, nc]):
                    continue
                seen.add((nr, nc))
                queue.append((nr, nc, int(dist) + 1))
    return False


def _wall_anchor_hit_in_strip(
    cell: tuple[int, int],
    direction: np.ndarray,
    wall: np.ndarray,
    source_map: np.ndarray | None,
    *,
    strip_half_width_cells: int,
    anchor_radius_cells: int,
    min_cells: int,
) -> tuple[tuple[int, int] | None, int]:
    wall_map = np.asarray(wall, dtype=bool)
    direction = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return None, DOOR_ANCHOR_NONE
    direction = direction / norm
    minor = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    best_cell: tuple[int, int] | None = None
    best_source = DOOR_ANCHOR_NONE
    best_score: tuple[float, float] | None = None
    r, c = int(cell[0]), int(cell[1])
    centers: list[tuple[int, int]] = []
    for offset in range(-int(strip_half_width_cells), int(strip_half_width_cells) + 1):
        p = np.asarray([float(r), float(c)], dtype=np.float32) + minor * float(offset)
        rr, cc = np.rint(p).astype(np.int32).tolist()
        if 0 <= int(rr) < wall_map.shape[0] and 0 <= int(cc) < wall_map.shape[1]:
            centers.append((int(rr), int(cc)))
    seen_centers: set[tuple[int, int]] = set()
    for center in centers:
        if center in seen_centers:
            continue
        seen_centers.add(center)
        rr, cc = center
        r0, r1 = max(0, rr - int(anchor_radius_cells)), min(wall_map.shape[0], rr + int(anchor_radius_cells) + 1)
        c0, c1 = max(0, cc - int(anchor_radius_cells)), min(wall_map.shape[1], cc + int(anchor_radius_cells) + 1)
        pr, pc = np.nonzero(wall_map[r0:r1, c0:c1])
        if pr.size < max(1, int(min_cells)):
            continue
        for local_r, local_c in zip(pr, pc):
            wr, wc = int(local_r + r0), int(local_c + c0)
            delta = np.asarray([float(wr - r), float(wc - c)], dtype=np.float32)
            perpendicular = abs(float(np.dot(delta, minor)))
            distance = float(np.linalg.norm(delta))
            source = _dominant_anchor_source((wr, wc), source_map, wall_map, 0)
            score = (_door_anchor_source_priority(source), -distance - 0.01 * perpendicular)
            if best_score is None or score > best_score:
                best_score = score
                best_cell = (wr, wc)
                best_source = int(source)
    if best_cell is None:
        return None, DOOR_ANCHOR_NONE
    return best_cell, int(best_source)


def _dominant_anchor_source(cell: tuple[int, int], source_map: np.ndarray | None, wall: np.ndarray, radius: int) -> int:
    if source_map is None:
        return DOOR_ANCHOR_STRICT_RAW
    r, c = int(cell[0]), int(cell[1])
    r0, r1 = max(0, r - int(radius)), min(source_map.shape[0], r + int(radius) + 1)
    c0, c1 = max(0, c - int(radius)), min(source_map.shape[1], c + int(radius) + 1)
    source = np.asarray(source_map[r0:r1, c0:c1], dtype=np.uint8)
    wall_patch = np.asarray(wall[r0:r1, c0:c1], dtype=bool)
    values = source[wall_patch & (source > 0)]
    if values.size == 0:
        return DOOR_ANCHOR_STRICT_RAW
    counts = np.bincount(values.astype(np.int32), minlength=max(DOOR_ANCHOR_SOURCE_NAMES) + 1)
    return int(np.argmax(counts))


def _anchor_source_name(source: int | None) -> str:
    return str(DOOR_ANCHOR_SOURCE_NAMES.get(int(source or 0), "unknown"))


def _anchor_source_counts(source_map: np.ndarray, wall: np.ndarray) -> dict[str, int]:
    source = np.asarray(source_map, dtype=np.uint8)
    mask = np.asarray(wall, dtype=bool) & (source > 0)
    values = source[mask]
    counts: dict[str, int] = {}
    for code, name in DOOR_ANCHOR_SOURCE_NAMES.items():
        if int(code) == DOOR_ANCHOR_NONE:
            continue
        counts[str(name)] = int(np.count_nonzero(values == int(code)))
    return counts


def _sort_cells_along_direction(cells: list[tuple[int, int]], center: np.ndarray, major: np.ndarray) -> list[tuple[int, int]]:
    return sorted(cells, key=lambda cell: float(np.dot(np.asarray(cell, dtype=np.float32) - center, major)))


def _component_thickness_m(pts: np.ndarray, center: np.ndarray, minor: np.ndarray, resolution_m: float) -> float:
    if pts.shape[0] <= 1:
        return float(resolution_m)
    values = np.dot(pts - center[None, :], minor)
    return float((float(np.max(values)) - float(np.min(values)) + 1.0) * resolution_m)


def _ratio(cells: Sequence[tuple[int, int]], mask: np.ndarray) -> float:
    if not cells:
        return 0.0
    return float(sum(1 for r, c in cells if bool(mask[int(r), int(c)])) / max(1, len(cells)))


def _seed_ev(
    row: int,
    col: int,
    first_occ_z: float | None,
    lower_free: int,
    top_occ: int,
    unknown_tail: int,
    accepted: bool,
    reason: str | None,
    *,
    turn_z: float | None = None,
    free_centroid: float | None = None,
    occupied_centroid: float | None = None,
    lower_free_ratio: float | None = None,
    upper_occupied_ratio: float | None = None,
    upper_observed: int = 0,
    upper_occupied: int = 0,
) -> VoxelDoorSeedEvidence:
    return VoxelDoorSeedEvidence(
        row=int(row),
        col=int(col),
        first_occupied_z_m=first_occ_z,
        lower_free_cells=int(lower_free),
        top_occupied_cells=int(top_occ),
        unknown_tail_cells=int(unknown_tail),
        accepted=bool(accepted),
        reject_reason=reason,
        turn_z_m=turn_z,
        free_centroid_z_m=free_centroid,
        occupied_centroid_z_m=occupied_centroid,
        lower_free_ratio=lower_free_ratio,
        upper_occupied_ratio_observed=upper_occupied_ratio,
        upper_observed_cells=int(upper_observed),
        upper_occupied_cells=int(upper_occupied),
    )


_SEED_REJECT_CODES = {
    "lower_free_cells_too_few": 2,
    "top_occupied_too_low": 3,
    "unknown_before_top_occupied": 4,
    "no_top_occupied": 5,
    "top_occupied_run_too_short": 6,
    "conflict_before_top_occupied": 7,
    "expected_top_occupied": 8,
    "non_unknown_after_top_occupied": 9,
    "free_after_unknown_tail": 10,
    "occupied_after_unknown_tail": 11,
    "upper_occupied_cells_too_few": 12,
    "turn_z_below_1p8": 13,
    "turn_outside_active_range": 14,
    "lower_free_ratio_too_low": 15,
    "upper_observed_cells_too_few": 16,
    "upper_occupied_ratio_too_low": 17,
    "occupied_centroid_not_above_free_centroid": 18,
    "no_active_z_bins": 19,
    "upper_actual_occupied_cells_too_few": 20,
    "upper_generalized_solid_ratio_too_low": 21,
    "upper_in_range_unknown_ratio_too_high": 22,
    "upper_free_ratio_too_high": 23,
    "turn_z_below_door_lintel_min": 24,
    "turn_z_above_door_lintel_max": 25,
    "first_upper_occupied_too_high_ceiling_like": 26,
    "upper_occupied_only_in_ceiling_band": 27,
    "upper_non_ceiling_actual_occupied_too_few": 28,
    "seed_component_bbox_too_small": 29,
    "not_navigation_free": 30,
}


def _seed_reject_code(reason: str) -> int:
    return int(_SEED_REJECT_CODES.get(str(reason), 255))


def _seed_reject_reason_from_code(code: int) -> str:
    for reason, value in _SEED_REJECT_CODES.items():
        if int(value) == int(code):
            return str(reason)
    return "unknown_reject_reason"


def _seed_reject_reason_maps(reason_map: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "voxel_door_lower_free_cells_too_few_cells": np.asarray(reason_map == _SEED_REJECT_CODES["lower_free_cells_too_few"], dtype=bool),
        "voxel_door_top_occupied_too_low_cells": np.asarray(reason_map == _SEED_REJECT_CODES["top_occupied_too_low"], dtype=bool),
        "voxel_door_unknown_before_top_occupied_cells": np.asarray(reason_map == _SEED_REJECT_CODES["unknown_before_top_occupied"], dtype=bool),
        "voxel_door_no_top_occupied_cells": np.asarray(reason_map == _SEED_REJECT_CODES["no_top_occupied"], dtype=bool),
        "voxel_door_top_occupied_run_too_short_cells": np.asarray(reason_map == _SEED_REJECT_CODES["top_occupied_run_too_short"], dtype=bool),
        "voxel_door_conflict_before_top_occupied_cells": np.asarray(reason_map == _SEED_REJECT_CODES["conflict_before_top_occupied"], dtype=bool),
        "voxel_door_upper_occupied_cells_too_few_cells": np.asarray(reason_map == _SEED_REJECT_CODES["upper_occupied_cells_too_few"], dtype=bool),
        "voxel_door_turn_z_below_1p8_cells": np.asarray(reason_map == _SEED_REJECT_CODES["turn_z_below_1p8"], dtype=bool),
        "voxel_door_turn_outside_active_range_cells": np.asarray(reason_map == _SEED_REJECT_CODES["turn_outside_active_range"], dtype=bool),
        "voxel_door_lower_free_ratio_too_low_cells": np.asarray(reason_map == _SEED_REJECT_CODES["lower_free_ratio_too_low"], dtype=bool),
        "voxel_door_upper_observed_cells_too_few_cells": np.asarray(reason_map == _SEED_REJECT_CODES["upper_observed_cells_too_few"], dtype=bool),
        "voxel_door_upper_occupied_ratio_too_low_cells": np.asarray(reason_map == _SEED_REJECT_CODES["upper_occupied_ratio_too_low"], dtype=bool),
        "voxel_door_occupied_centroid_not_above_free_centroid_cells": np.asarray(reason_map == _SEED_REJECT_CODES["occupied_centroid_not_above_free_centroid"], dtype=bool),
        "voxel_door_upper_actual_occupied_cells_too_few_cells": np.asarray(reason_map == _SEED_REJECT_CODES["upper_actual_occupied_cells_too_few"], dtype=bool),
        "voxel_door_upper_generalized_solid_ratio_too_low_cells": np.asarray(reason_map == _SEED_REJECT_CODES["upper_generalized_solid_ratio_too_low"], dtype=bool),
        "voxel_door_upper_in_range_unknown_ratio_too_high_cells": np.asarray(reason_map == _SEED_REJECT_CODES["upper_in_range_unknown_ratio_too_high"], dtype=bool),
        "voxel_door_upper_free_ratio_too_high_cells": np.asarray(reason_map == _SEED_REJECT_CODES["upper_free_ratio_too_high"], dtype=bool),
        "voxel_door_turn_z_below_door_lintel_min_cells": np.asarray(reason_map == _SEED_REJECT_CODES["turn_z_below_door_lintel_min"], dtype=bool),
        "voxel_door_turn_z_above_door_lintel_max_cells": np.asarray(reason_map == _SEED_REJECT_CODES["turn_z_above_door_lintel_max"], dtype=bool),
        "voxel_door_first_upper_occupied_too_high_ceiling_like_cells": np.asarray(reason_map == _SEED_REJECT_CODES["first_upper_occupied_too_high_ceiling_like"], dtype=bool),
        "voxel_door_upper_occupied_only_in_ceiling_band_cells": np.asarray(reason_map == _SEED_REJECT_CODES["upper_occupied_only_in_ceiling_band"], dtype=bool),
        "voxel_door_upper_non_ceiling_actual_occupied_too_few_cells": np.asarray(reason_map == _SEED_REJECT_CODES["upper_non_ceiling_actual_occupied_too_few"], dtype=bool),
        "voxel_door_seed_component_bbox_too_small_cells": np.asarray(reason_map == _SEED_REJECT_CODES["seed_component_bbox_too_small"], dtype=bool),
        "voxel_door_seed_not_navigation_free_cells": np.asarray(reason_map == _SEED_REJECT_CODES["not_navigation_free"], dtype=bool),
    }


def _empty_seed_debug_maps(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "voxel_door_turn_z_estimate_xy": np.full(shape, np.nan, dtype=np.float32),
        "voxel_door_free_centroid_z_xy": np.full(shape, np.nan, dtype=np.float32),
        "voxel_door_occupied_centroid_z_xy": np.full(shape, np.nan, dtype=np.float32),
        "voxel_door_lower_free_ratio_xy": np.zeros(shape, dtype=np.float32),
        "voxel_door_upper_occupied_ratio_observed_xy": np.zeros(shape, dtype=np.float32),
        "voxel_door_upper_solid_ratio_active_xy": np.zeros(shape, dtype=np.float32),
        "voxel_door_upper_actual_occupied_ratio_active_xy": np.zeros(shape, dtype=np.float32),
        "voxel_door_upper_in_range_unknown_ratio_active_xy": np.zeros(shape, dtype=np.float32),
        "voxel_door_upper_free_ratio_active_xy": np.zeros(shape, dtype=np.float32),
        "voxel_door_first_upper_occupied_z_xy": np.full(shape, np.nan, dtype=np.float32),
        "voxel_door_upper_non_ceiling_actual_occupied_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_upper_ceiling_occupied_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_ceiling_like_reject_mask": np.zeros(shape, dtype=bool),
        "voxel_door_turn_z_above_lintel_max_mask": np.zeros(shape, dtype=bool),
        "voxel_door_first_occ_too_high_mask": np.zeros(shape, dtype=bool),
        "voxel_door_upper_observed_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_upper_total_active_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_upper_occupied_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_upper_solid_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_upper_actual_occupied_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_upper_in_range_unknown_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_upper_out_of_range_unknown_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_upper_free_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_effective_observed_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_in_range_unknown_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_out_of_range_unknown_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_free_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_occupied_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_observed_count_xy": np.zeros(shape, dtype=np.uint16),
        "voxel_door_upper_solid_uses_in_range_unknown": np.zeros(shape, dtype=bool),
        "voxel_door_upper_solid_uses_out_of_range_unknown": np.zeros(shape, dtype=bool),
    }


def _estimate_z_resolution_m(z_centers: np.ndarray) -> float:
    z = np.asarray(z_centers, dtype=np.float32).reshape(-1)
    if z.size >= 2:
        diffs = np.diff(z)
        finite = diffs[np.isfinite(diffs) & (diffs > 0)]
        if finite.size:
            return float(np.median(finite))
    return 0.05


def _gather_cum_before(cum: np.ndarray, pos: np.ndarray) -> np.ndarray:
    cumsum = np.asarray(cum, dtype=np.int32)
    p = np.asarray(pos, dtype=np.int32).reshape(-1)
    out = np.zeros(p.shape[0], dtype=np.int32)
    valid = p > 0
    if np.any(valid):
        cols = np.nonzero(valid)[0]
        rows = np.clip(p[valid] - 1, 0, cumsum.shape[0] - 1)
        out[valid] = cumsum[rows, cols]
    return out


def _first_true_z(mask: np.ndarray, z_centers: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    z = np.asarray(z_centers, dtype=np.float32).reshape(-1)
    out = np.full(values.shape[1], np.nan, dtype=np.float32)
    has = np.any(values, axis=0)
    if np.any(has):
        first = np.argmax(values[:, has], axis=0)
        out[has] = z[first]
    return out


def _last_true_z(mask: np.ndarray, z_centers: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    z = np.asarray(z_centers, dtype=np.float32).reshape(-1)
    out = np.full(values.shape[1], np.nan, dtype=np.float32)
    has = np.any(values, axis=0)
    if np.any(has):
        reversed_first = np.argmax(values[::-1, has], axis=0)
        last = values.shape[0] - 1 - reversed_first
        out[has] = z[last]
    return out


def _rejected_candidate(
    candidate_id: int,
    component_id: int,
    seed_cells: list[tuple[int, int]],
    reason: str,
    center: np.ndarray | None = None,
    major: np.ndarray | None = None,
) -> VoxelDoorLineCandidate:
    center_arr = np.mean(np.asarray(seed_cells or [(0, 0)], dtype=np.float32), axis=0) if center is None else np.asarray(center, dtype=np.float32)
    major_arr = np.asarray([0.0, 1.0], dtype=np.float32) if major is None else np.asarray(major, dtype=np.float32)
    norm = float(np.linalg.norm(major_arr))
    major_arr = np.asarray([0.0, 1.0], dtype=np.float32) if norm <= 1e-6 else major_arr / norm
    minor = np.asarray([-major_arr[1], major_arr[0]], dtype=np.float32)
    return VoxelDoorLineCandidate(
        candidate_id=int(candidate_id),
        seed_component_id=int(component_id),
        seed_cells=list(seed_cells),
        center_rc=(float(center_arr[0]), float(center_arr[1])),
        major_dir_rc=(float(major_arr[0]), float(major_arr[1])),
        minor_dir_rc=(float(minor[0]), float(minor[1])),
        seed_projected_centerline_cells=list(seed_cells),
        extended_centerline_cells=list(seed_cells),
        door_cut_cells=list(seed_cells),
        wall_anchor_a=None,
        wall_anchor_b=None,
        width_m=0.0,
        accepted=False,
        reject_reason=str(reason),
        debug={},
    )


def _empty_result(shape: tuple[int, int], *, enabled: bool) -> VoxelDoorDetectionResult:
    zero = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    debug = {
        "voxel_door_enabled": bool(enabled),
        "voxel_door_accepted_count": 0,
        "voxel_door_rejected_count": 0,
        "voxel_door_candidates": [],
    }
    return VoxelDoorDetectionResult(
        door_seed_mask=zero.copy(),
        door_seed_component_map=labels,
        door_centerline_candidate_mask=zero.copy(),
        accepted_door_centerline_mask=zero.copy(),
        rejected_door_centerline_mask=zero.copy(),
        door_cut_mask=zero.copy(),
        door_seed_reject_reason_map=np.zeros(shape, dtype=np.uint8),
        candidates=[],
        debug=debug,
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
