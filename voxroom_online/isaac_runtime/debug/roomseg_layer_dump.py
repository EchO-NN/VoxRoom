from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOMSEG_ARRAY_KEYS = (
    "occupancy_map",
    "observed_free_mask",
    "obstacle_mask",
    "unknown_mask",
    "navigation_free_room_domain",
    "vertical_free_room_domain",
    "vertical_occupied_0p2_2p0",
    "vertical_observed_map",
    "vertical_observed_0p2_2p0",
    "vertical_unknown_before_overlay",
    "roomseg_ray_covered_count",
    "roomseg_terminal_wall_count",
    "roomseg_terminal_wall_height_min",
    "roomseg_terminal_wall_height_max",
    "roomseg_terminal_wall_depth_min",
    "roomseg_terminal_wall_splat",
    "ray_valid_wall_inference",
    "initial_roomseg_free_after_ray_wall",
    "initial_roomseg_occupied_after_ray_wall",
    "initial_roomseg_unknown_after_ray_wall",
    "unknown_before_ray_wall",
    "unknown_after_ray_wall",
    "unknown_removed_by_ray_wall",
    "nav_raw_obstacle",
    "roomseg_static_structural_occupied",
    "nav_obstacle_overlay_candidate",
    "nav_obstacle_overlay_accepted",
    "initial_roomseg_free",
    "initial_roomseg_occupied",
    "initial_roomseg_unknown",
    "initial_roomseg_free_after_fusion",
    "initial_roomseg_occupied_after_fusion",
    "initial_roomseg_unknown_after_fusion",
    "walls_rescued_from_unknown",
    "vertical_free_over_nav_obstacle",
    "nav_obstacle_still_unknown_after_fusion",
    "vertical_carved_map",
    "wall_confidence_map",
    "candidate_wall",
    "component_gate",
    "repaired_roomseg_free",
    "repaired_roomseg_occupied",
    "repaired_roomseg_unknown",
    "roomseg_free_raw",
    "roomseg_free_stable",
    "roomseg_free_clean",
    "free_noise_rejected",
    "ray_fan_spur_rejected",
    "structural_wall_clean",
    "terminal_wall_structural_clean",
    "corridor_axis_v4",
    "corridor_junctions_v4",
    "structural_free_mask",
    "boundary_map",
    "wall_boundary_map",
    "candidate_closure_map",
    "accepted_closure_map",
    "rejected_closure_map",
    "virtual_boundary_map",
    "wall_extension_boundary_mask",
    "door_completion_boundary_mask",
    "virtual_boundary_source_map",
    "representative_wall_map",
    "extended_wall_map",
    "accepted_separators",
    "rejected_separators",
    "virtual_separator_label_fill",
    "vertical_free_clipped_outside_navigation_map",
    "free_wall_conflict_map_before_sanitize",
    "roomseg_sanitized_free",
    "roomseg_sanitized_wall",
    "terminal_wall_roomseg_mask",
    "pre_extension_door_detected_map",
    "pre_extension_door_cut_mask",
    "pre_extension_door_pattern_type_map",
    "strict_pre_extension_door_cut_mask",
    "partial_door_seed_mask",
    "partial_door_line_mask",
    "partial_door_extension_cut_mask",
    "rejected_door_extension_mask",
    "original_step1_step2_virtual_boundary_map",
    "pre_extension_partition_free",
    "pre_extension_room_label_map",
    "step1_step2_accepted_closure_map",
    "step1_step2_accepted_separator_map",
    "room_proposal_labels_before_merge",
    "room_labels_after_merge",
    "final_room_label_map",
    "watershed_free_clean",
    "watershed_wall_candidate_clean",
    "watershed_unknown_mask",
    "watershed_dist_struct",
    "watershed_dist_free_extent",
    "watershed_elevation",
    "watershed_corridor_core",
    "watershed_confirmed_room_seeds",
    "watershed_frontier_room_seeds",
    "watershed_corridor_seeds",
    "watershed_raw_labels",
    "watershed_final_labels",
    "watershed_region_types",
    "voxel_nav_free_xy",
    "voxel_nav_occupied_xy",
    "voxel_nav_unknown_xy",
    "voxel_nav_observed_xy",
    "voxel_outside_xy",
    "voxel_outside_candidate_xy",
    "voxel_outside_candidate_base_xy",
    "voxel_outside_disconnected_candidate_xy",
    "voxel_outside_pre_nav_free_xy",
    "voxel_outside_pre_traversible_xy",
    "voxel_outside_robot_component_xy",
    "voxel_outside_connected_to_robot_rejected_xy",
    "voxel_outside_floor_frustum_seen_xy",
    "voxel_outside_source_free_xy",
    "voxel_outside_vertical_free_source_xy",
    "voxel_outside_vertical_free_count_xy",
    "voxel_outside_full_height_no_occupied_xy",
    "voxel_outside_holdout_floor_unseen_xy",
    "voxel_outside_excluded_synthetic_free_xy",
    "voxel_outside_synthetic_free_provenance_xy",
    "voxel_outside_score_xy",
    "voxel_floor_frustum_seen_count_xy",
    "voxel_vertical_free_xy",
    "voxel_wall_xy",
    "voxel_unknown_xy",
    "voxel_vertical_observed_xy",
    "voxel_active_free_count_xy",
    "voxel_active_occupied_count_xy",
    "voxel_active_unknown_count_xy",
    "voxel_sensor_range_count_xy",
    "voxel_sensor_range_ratio_xy",
    "voxel_sensor_in_range_unknown_count_xy",
    "voxel_sensor_outside_range_unknown_count_xy",
    "voxel_in_range_unknown_ratio_xy",
    "voxel_outside_unknown_ratio_xy",
    "voxel_generalized_occupied_count_xy",
    "voxel_generalized_occupied_ratio_xy",
    "voxel_wall_generalized_raw_xy",
    "voxel_wall_actual_ratio_raw_xy",
    "voxel_wall_actual_occupied_requirement_xy",
    "voxel_wall_rejected_by_outside_unknown_xy",
    "voxel_wall_from_in_range_unknown_xy",
    "voxel_roomseg_nav_obstacle_suppressed_by_vertical_free_xy",
    "voxel_outside_unknown_dominant_xy",
    "voxel_active_z_bin_count_xy",
    "voxel_occupied_ratio_active_xy",
    "voxel_free_wall_conflict_xy",
    "voxel_line_supported_wall_map",
    "voxel_filtered_wall_line_mask",
    "voxel_wall_base_map",
    "voxel_door_seed_mask",
    "voxel_door_raw_seed_mask",
    "voxel_door_seed_component_id_map",
    "voxel_door_seed_cluster_id_map",
    "voxel_door_seed_reject_reason_map",
    "voxel_door_seed_reject_reason_id_map",
    "voxel_door_seed_lower_free_cells_xy",
    "voxel_door_seed_top_occupied_cells_xy",
    "voxel_door_seed_first_occupied_z_xy",
    "voxel_door_seed_unknown_tail_cells_xy",
    "voxel_door_first_upper_occupied_z_xy",
    "voxel_door_upper_non_ceiling_actual_occupied_count_xy",
    "voxel_door_upper_ceiling_occupied_count_xy",
    "voxel_door_ceiling_like_reject_mask",
    "voxel_door_turn_z_above_lintel_max_mask",
    "voxel_door_first_occ_too_high_mask",
    "voxel_door_upper_solid_count_xy",
    "voxel_door_upper_actual_occupied_count_xy",
    "voxel_door_effective_observed_count_xy",
    "voxel_door_in_range_unknown_count_xy",
    "voxel_door_seed_line_primitive_id_map",
    "voxel_door_seed_line_primitive_mask",
    "voxel_door_extensible_primitive_mask",
    "voxel_door_rejected_primitive_mask",
    "voxel_door_extension_trials_map",
    "voxel_door_extension_reject_reason_id_map",
    "voxel_door_candidate_lines_map",
    "voxel_door_rejected_lines_map",
    "voxel_door_centerline_mask",
    "voxel_door_cut_mask",
    "voxel_door_visual_only_mask",
    "voxel_door_partition_cut_candidate_mask",
    "voxel_door_partition_cut_mask",
    "voxel_door_partition_cut_rejected_mask",
    "voxel_door_topology_effective_cut_mask",
    "voxel_door_partition_reject_reason_id_map",
    "voxel_current_door_cut_mask",
    "voxel_current_door_topology_effective_mask",
    "voxel_stable_door_cut_mask",
    "voxel_door_stable_cut_mask",
    "voxel_stable_door_visual_mask",
    "voxel_door_memory_observed_decay_band_mask",
    "voxel_door_memory_unobserved_track_mask",
    "voxel_door_memory_contradiction_mask",
    "voxel_step1_wall_gap_fill_map",
    "voxel_wall_after_step1_map",
    "voxel_step2_extension_candidate_map",
    "voxel_step2_extension_separator_map",
    "voxel_step2_rejected_extension_map",
    "voxel_final_separator_map",
    "voxel_final_room_label_map",
    "context_room_label_map",
    "context_room_reliability_map",
    "frontier_map",
    "selected_frontier_mask",
    "selected_frontier_center_rc",
    "agent_rc",
)


ROOMSEG_SNAPSHOT_ARRAY_KEYS = (
    "occupancy_map",
    "observed_free_mask",
    "obstacle_mask",
    "unknown_mask",
    "navigation_free_room_domain",
    "vertical_free_room_domain",
    "vertical_occupied_0p2_2p0",
    "vertical_observed_map",
    "vertical_observed_0p2_2p0",
    "roomseg_ray_covered_count",
    "roomseg_terminal_wall_count",
    "roomseg_terminal_wall_splat",
    "ray_valid_wall_inference",
    "pass2_extension_intersection_targets",
    "pass2_line_extension_completion",
    "wall_target_after_line_extension",
    "completed_wall_after_line_extension",
    "vertical_free_clipped_outside_navigation_map",
    "free_wall_conflict_map_before_sanitize",
    "roomseg_sanitized_free",
    "roomseg_sanitized_wall",
    "terminal_wall_roomseg_mask",
    "pre_extension_door_detected_map",
    "pre_extension_door_cut_mask",
    "pre_extension_door_pattern_type_map",
    "strict_pre_extension_door_cut_mask",
    "partial_door_seed_mask",
    "partial_door_line_mask",
    "partial_door_extension_cut_mask",
    "rejected_door_extension_mask",
    "original_step1_step2_virtual_boundary_map",
    "wall_extension_boundary_mask",
    "door_completion_boundary_mask",
    "virtual_boundary_source_map",
    "pre_extension_partition_free",
    "pre_extension_room_label_map",
    "step1_step2_accepted_closure_map",
    "step1_step2_accepted_separator_map",
    "roomseg_free_raw",
    "roomseg_free_stable",
    "roomseg_free_clean",
    "free_noise_rejected",
    "ray_fan_spur_rejected",
    "structural_wall_clean",
    "terminal_wall_structural_clean",
    "corridor_axis_v4",
    "corridor_junctions_v4",
    "accepted_separators",
    "rejected_separators",
    "virtual_separator_label_fill",
    "initial_roomseg_free_after_fusion",
    "initial_roomseg_occupied_after_fusion",
    "initial_roomseg_unknown_after_fusion",
    "voxel_nav_free_xy",
    "voxel_nav_occupied_xy",
    "voxel_nav_unknown_xy",
    "voxel_nav_observed_xy",
    "voxel_outside_xy",
    "voxel_outside_candidate_xy",
    "voxel_outside_candidate_base_xy",
    "voxel_outside_disconnected_candidate_xy",
    "voxel_outside_pre_nav_free_xy",
    "voxel_outside_pre_traversible_xy",
    "voxel_outside_robot_component_xy",
    "voxel_outside_connected_to_robot_rejected_xy",
    "voxel_outside_floor_frustum_seen_xy",
    "voxel_outside_source_free_xy",
    "voxel_outside_vertical_free_source_xy",
    "voxel_outside_vertical_free_count_xy",
    "voxel_outside_full_height_no_occupied_xy",
    "voxel_outside_holdout_floor_unseen_xy",
    "voxel_outside_excluded_synthetic_free_xy",
    "voxel_outside_synthetic_free_provenance_xy",
    "voxel_outside_score_xy",
    "voxel_floor_frustum_seen_count_xy",
    "voxel_vertical_free_xy",
    "voxel_wall_xy",
    "voxel_unknown_xy",
    "voxel_vertical_observed_xy",
    "voxel_active_free_count_xy",
    "voxel_active_occupied_count_xy",
    "voxel_active_unknown_count_xy",
    "voxel_sensor_range_count_xy",
    "voxel_sensor_range_ratio_xy",
    "voxel_sensor_in_range_unknown_count_xy",
    "voxel_sensor_outside_range_unknown_count_xy",
    "voxel_in_range_unknown_ratio_xy",
    "voxel_outside_unknown_ratio_xy",
    "voxel_generalized_occupied_count_xy",
    "voxel_generalized_occupied_ratio_xy",
    "voxel_wall_generalized_raw_xy",
    "voxel_wall_actual_ratio_raw_xy",
    "voxel_wall_actual_occupied_requirement_xy",
    "voxel_wall_rejected_by_outside_unknown_xy",
    "voxel_wall_from_in_range_unknown_xy",
    "voxel_roomseg_nav_obstacle_suppressed_by_vertical_free_xy",
    "voxel_outside_unknown_dominant_xy",
    "voxel_active_z_bin_count_xy",
    "voxel_occupied_ratio_active_xy",
    "voxel_free_wall_conflict_xy",
    "voxel_line_supported_wall_map",
    "voxel_filtered_wall_line_mask",
    "voxel_wall_base_map",
    "voxel_door_seed_mask",
    "voxel_door_raw_seed_mask",
    "voxel_door_seed_component_id_map",
    "voxel_door_seed_cluster_id_map",
    "voxel_door_seed_reject_reason_map",
    "voxel_door_seed_reject_reason_id_map",
    "voxel_door_seed_lower_free_cells_xy",
    "voxel_door_seed_top_occupied_cells_xy",
    "voxel_door_seed_first_occupied_z_xy",
    "voxel_door_seed_unknown_tail_cells_xy",
    "voxel_door_first_upper_occupied_z_xy",
    "voxel_door_upper_non_ceiling_actual_occupied_count_xy",
    "voxel_door_upper_ceiling_occupied_count_xy",
    "voxel_door_ceiling_like_reject_mask",
    "voxel_door_turn_z_above_lintel_max_mask",
    "voxel_door_first_occ_too_high_mask",
    "voxel_door_upper_solid_count_xy",
    "voxel_door_upper_actual_occupied_count_xy",
    "voxel_door_effective_observed_count_xy",
    "voxel_door_in_range_unknown_count_xy",
    "voxel_door_seed_line_primitive_id_map",
    "voxel_door_seed_line_primitive_mask",
    "voxel_door_extensible_primitive_mask",
    "voxel_door_rejected_primitive_mask",
    "voxel_door_extension_trials_map",
    "voxel_door_extension_reject_reason_id_map",
    "voxel_door_candidate_lines_map",
    "voxel_door_rejected_lines_map",
    "voxel_door_centerline_mask",
    "voxel_door_cut_mask",
    "voxel_door_visual_only_mask",
    "voxel_door_partition_cut_candidate_mask",
    "voxel_door_partition_cut_mask",
    "voxel_door_partition_cut_rejected_mask",
    "voxel_door_topology_effective_cut_mask",
    "voxel_door_partition_reject_reason_id_map",
    "voxel_current_door_cut_mask",
    "voxel_current_door_topology_effective_mask",
    "voxel_stable_door_cut_mask",
    "voxel_door_stable_cut_mask",
    "voxel_stable_door_visual_mask",
    "voxel_door_memory_observed_decay_band_mask",
    "voxel_door_memory_unobserved_track_mask",
    "voxel_door_memory_contradiction_mask",
    "voxel_step1_wall_gap_fill_map",
    "voxel_wall_after_step1_map",
    "voxel_step2_extension_candidate_map",
    "voxel_step2_extension_separator_map",
    "voxel_step2_rejected_extension_map",
    "voxel_final_separator_map",
    "voxel_final_room_label_map",
    "final_room_label_map",
    "context_room_label_map",
    "frontier_map",
    "selected_frontier_mask",
    "selected_frontier_center_rc",
    "agent_rc",
)


def save_roomseg_layer_dump(
    *,
    out_dir: str | Path,
    step: int,
    room_debug: Mapping[str, object],
    occupancy_map: np.ndarray,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    frontier_map: Optional[np.ndarray] = None,
    selected_frontier_members: Optional[Sequence[Sequence[int]]] = None,
    selected_frontier_center_rc: Optional[Sequence[int]] = None,
    agent_rc: Optional[Sequence[int]] = None,
    max_saves: int = 50,
    save_npz: bool = True,
    save_png: bool = True,
    save_summary_json: bool = True,
    save_overlay_png: Optional[bool] = None,
    save_layers_png: Optional[bool] = None,
    save_navigation_room_masks_png: Optional[bool] = None,
    npz_keys: Optional[Sequence[str]] = None,
    extra_npz_arrays: Optional[Mapping[str, object]] = None,
    include_selected_frontier_sector: bool = True,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = build_roomseg_debug_arrays(
        room_debug=room_debug,
        occupancy_map=occupancy_map,
        observed_free_mask=observed_free_mask,
        obstacle_mask=obstacle_mask,
        unknown_mask=unknown_mask,
        frontier_map=frontier_map,
        selected_frontier_members=selected_frontier_members,
        selected_frontier_center_rc=selected_frontier_center_rc,
        agent_rc=agent_rc,
        include_selected_frontier_sector=include_selected_frontier_sector,
    )
    summary = summarize_roomseg_arrays(arrays, room_debug, int(step))
    stem = "roomseg_step_%06d" % int(step)
    paths = {
        "npz": str(out / ("%s.npz" % stem)),
        "summary_json": str(out / ("%s.summary.json" % stem)),
        "overlay_png": str(out / ("%s.overlay.png" % stem)),
        "layers_png": str(out / ("%s.layers.png" % stem)),
        "navigation_room_masks_png": str(out / ("%s.navigation_room_masks.png" % stem)),
    }
    if bool(save_npz):
        if npz_keys is None:
            npz_arrays = arrays
        else:
            npz_arrays = {str(key): arrays[str(key)] for key in npz_keys if str(key) in arrays}
        for key, value in dict(extra_npz_arrays or {}).items():
            npz_arrays[str(key)] = np.asarray(value)
        np.savez_compressed(paths["npz"], **npz_arrays)
        if extra_npz_arrays:
            summary["extra_npz_arrays"] = {
                str(key): {
                    "shape": [int(v) for v in np.asarray(value).shape],
                    "dtype": str(np.asarray(value).dtype),
                }
                for key, value in dict(extra_npz_arrays).items()
            }
    if bool(save_summary_json):
        Path(paths["summary_json"]).write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overlay_enabled = bool(save_png) if save_overlay_png is None else bool(save_overlay_png)
    layers_enabled = bool(save_png) if save_layers_png is None else bool(save_layers_png)
    nav_masks_enabled = bool(save_png) if save_navigation_room_masks_png is None else bool(save_navigation_room_masks_png)
    if overlay_enabled:
        render_roomseg_overlay(arrays, summary).save(paths["overlay_png"])
    if layers_enabled:
        render_roomseg_layers_grid(arrays, summary).save(paths["layers_png"])
    if nav_masks_enabled:
        render_navigation_room_masks(arrays).save(paths["navigation_room_masks_png"])
    _prune_old(out, int(max_saves))
    return {"paths": paths, "summary": summary}


def build_roomseg_debug_arrays(
    *,
    room_debug: Mapping[str, object],
    occupancy_map: np.ndarray,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    frontier_map: Optional[np.ndarray],
    selected_frontier_members: Optional[Sequence[Sequence[int]]],
    selected_frontier_center_rc: Optional[Sequence[int]],
    agent_rc: Optional[Sequence[int]],
    include_selected_frontier_sector: bool,
) -> dict[str, np.ndarray]:
    shape = tuple(np.asarray(occupancy_map).shape[:2])
    arrays: dict[str, np.ndarray] = {
        "occupancy_map": _bool_array(occupancy_map, shape),
        "observed_free_mask": _bool_array(observed_free_mask, shape),
        "obstacle_mask": _bool_array(obstacle_mask, shape),
        "unknown_mask": _bool_array(unknown_mask, shape),
    }
    mapping = {
        "navigation_free_room_domain": "navigation_free_room_domain",
        "vertical_free_room_domain": "vertical_free_room_domain",
        "vertical_occupied_0p2_2p0": "vertical_occupied_0p2_2p0",
        "vertical_observed_map": "vertical_observed_map",
        "vertical_observed_0p2_2p0": "vertical_observed_0p2_2p0",
        "vertical_unknown_before_overlay": "vertical_unknown_before_overlay",
        "roomseg_ray_covered_count": "roomseg_ray_covered_count",
        "roomseg_terminal_wall_count": "roomseg_terminal_wall_count",
        "roomseg_terminal_wall_height_min": "roomseg_terminal_wall_height_min",
        "roomseg_terminal_wall_height_max": "roomseg_terminal_wall_height_max",
        "roomseg_terminal_wall_depth_min": "roomseg_terminal_wall_depth_min",
        "roomseg_terminal_wall_splat": "roomseg_terminal_wall_splat",
        "ray_valid_wall_inference": "ray_valid_wall_inference",
        "initial_roomseg_free_after_ray_wall": "initial_roomseg_free_after_ray_wall",
        "initial_roomseg_occupied_after_ray_wall": "initial_roomseg_occupied_after_ray_wall",
        "initial_roomseg_unknown_after_ray_wall": "initial_roomseg_unknown_after_ray_wall",
        "unknown_before_ray_wall": "unknown_before_ray_wall",
        "unknown_after_ray_wall": "unknown_after_ray_wall",
        "unknown_removed_by_ray_wall": "unknown_removed_by_ray_wall",
        "nav_raw_obstacle": "nav_raw_obstacle",
        "roomseg_static_structural_occupied": "roomseg_static_structural_occupied",
        "nav_obstacle_overlay_candidate": "nav_obstacle_overlay_candidate",
        "nav_obstacle_overlay_accepted": "nav_obstacle_overlay_accepted",
        "initial_roomseg_free": "initial_roomseg_free",
        "initial_roomseg_occupied": "initial_roomseg_occupied",
        "initial_roomseg_unknown": "initial_roomseg_unknown",
        "initial_roomseg_free_after_fusion": "initial_roomseg_free_after_fusion",
        "initial_roomseg_occupied_after_fusion": "initial_roomseg_occupied_after_fusion",
        "initial_roomseg_unknown_after_fusion": "initial_roomseg_unknown_after_fusion",
        "pass2_extension_intersection_targets": "pass2_extension_intersection_targets",
        "pass2_line_extension_completion": "pass2_line_extension_completion",
        "wall_target_after_line_extension": "wall_target_after_line_extension",
        "completed_wall_after_line_extension": "completed_wall_after_line_extension",
        "vertical_free_clipped_outside_navigation_map": "vertical_free_clipped_outside_navigation_map",
        "free_wall_conflict_map_before_sanitize": "free_wall_conflict_map_before_sanitize",
        "roomseg_sanitized_free": "roomseg_sanitized_free",
        "roomseg_sanitized_wall": "roomseg_sanitized_wall",
        "terminal_wall_roomseg_mask": "terminal_wall_roomseg_mask",
        "pre_extension_door_detected_map": "pre_extension_door_detected_map",
        "pre_extension_door_cut_mask": "pre_extension_door_cut_mask",
        "pre_extension_door_pattern_type_map": "pre_extension_door_pattern_type_map",
        "strict_pre_extension_door_cut_mask": "strict_pre_extension_door_cut_mask",
        "partial_door_seed_mask": "partial_door_seed_mask",
        "partial_door_line_mask": "partial_door_line_mask",
        "partial_door_extension_cut_mask": "partial_door_extension_cut_mask",
        "rejected_door_extension_mask": "rejected_door_extension_mask",
        "original_step1_step2_virtual_boundary_map": "original_step1_step2_virtual_boundary_map",
        "pre_extension_partition_free": "pre_extension_partition_free",
        "pre_extension_room_label_map": "pre_extension_room_label_map",
        "step1_step2_accepted_closure_map": "step1_step2_accepted_closure_map",
        "step1_step2_accepted_separator_map": "step1_step2_accepted_separator_map",
        "walls_rescued_from_unknown": "walls_rescued_from_unknown",
        "vertical_free_over_nav_obstacle": "vertical_free_over_nav_obstacle",
        "nav_obstacle_still_unknown_after_fusion": "nav_obstacle_still_unknown_after_fusion",
        "vertical_carved_map": "vertical_carved_map",
        "wall_confidence_map": "wall_confidence_map",
        "candidate_wall": "candidate_wall",
        "component_gate": "component_gate",
        "repaired_roomseg_free": "repaired_roomseg_free",
        "repaired_roomseg_occupied": "repaired_roomseg_occupied",
        "repaired_roomseg_unknown": "repaired_roomseg_unknown",
        "roomseg_free_raw": "roomseg_free_raw",
        "roomseg_free_stable": "roomseg_free_stable",
        "roomseg_free_clean": "roomseg_free_clean",
        "free_noise_rejected": "free_noise_rejected",
        "ray_fan_spur_rejected": "ray_fan_spur_rejected",
        "structural_wall_clean": "structural_wall_clean",
        "terminal_wall_structural_clean": "terminal_wall_structural_clean",
        "corridor_axis_v4": "corridor_axis_v4",
        "corridor_junctions_v4": "corridor_junctions_v4",
        "structural_free_mask": "structural_free_mask",
        "boundary_map": "boundary_map",
        "wall_boundary_map": "wall_boundary_map",
        "candidate_closure_map": "candidate_closure_map",
        "accepted_closure_map": "accepted_closure_map",
        "rejected_closure_map": "rejected_closure_map",
        "virtual_boundary_map": "virtual_boundary_map",
        "wall_extension_boundary_mask": "wall_extension_boundary_mask",
        "door_completion_boundary_mask": "door_completion_boundary_mask",
        "virtual_boundary_source_map": "virtual_boundary_source_map",
        "accepted_separators": "accepted_separators",
        "rejected_separators": "rejected_separators",
        "virtual_separator_label_fill": "virtual_separator_label_fill",
        "room_proposal_labels_before_merge": "room_proposal_labels_before_merge",
        "room_labels_after_merge": "room_labels_after_merge",
        "final_room_label_map": "final_room_label_map",
        "watershed_free_clean": "free_clean",
        "watershed_wall_candidate_clean": "wall_candidate_clean",
        "watershed_unknown_mask": "unknown_mask",
        "watershed_dist_struct": "dist_struct_m",
        "watershed_dist_free_extent": "dist_free_extent_m",
        "watershed_elevation": "elevation",
        "watershed_corridor_core": "corridor_core",
        "watershed_confirmed_room_seeds": "confirmed_room_seeds",
        "watershed_frontier_room_seeds": "frontier_room_seeds",
        "watershed_corridor_seeds": "corridor_seeds",
        "watershed_raw_labels": "raw_labels",
        "watershed_final_labels": "final_labels",
        "watershed_region_types": "region_type_map",
        "context_room_label_map": "context_room_label_map",
        "context_room_reliability_map": "context_room_reliability_map",
    }
    for out_key, debug_key in mapping.items():
        dtype = None
        if (
            "confidence" in out_key
            or "reliability" in out_key
            or "height_" in out_key
            or "depth_" in out_key
            or out_key in {"watershed_dist_struct", "watershed_dist_free_extent", "watershed_elevation"}
        ):
            dtype = np.float32
        elif out_key in {"virtual_boundary_source_map"}:
            dtype = np.uint8
        elif out_key.endswith("_count"):
            dtype = np.uint16
        arrays[out_key] = _array_from_debug(room_debug, debug_key, shape, dtype)
    if not np.any(arrays["boundary_map"]):
        arrays["boundary_map"] = _bool_array(room_debug.get("clean_structure_map"), shape)
    arrays["representative_wall_map"] = _lines_to_mask(room_debug.get("representative_lines", []), shape)
    arrays["extended_wall_map"] = _lines_to_mask(room_debug.get("extended_lines", room_debug.get("representative_lines", [])), shape)
    arrays["frontier_map"] = _bool_array(frontier_map, shape)
    arrays["selected_frontier_mask"] = _frontier_members_to_mask(selected_frontier_members, shape)
    arrays["selected_frontier_center_rc"] = np.asarray(selected_frontier_center_rc if selected_frontier_center_rc is not None else [-1, -1], dtype=np.int32)
    arrays["agent_rc"] = np.asarray(agent_rc if agent_rc is not None else [-1, -1], dtype=np.int32)
    arrays["selected_frontier_sector"] = (
        _selected_frontier_sector(arrays["agent_rc"], selected_frontier_members, shape)
        if bool(include_selected_frontier_sector)
        else np.zeros(shape, dtype=bool)
    )
    arrays["selected_frontier_sector_missing_overlap"] = arrays["selected_frontier_sector"] & arrays["observed_free_mask"] & (arrays["final_room_label_map"].astype(np.int32) <= 0)
    for key in ROOMSEG_ARRAY_KEYS:
        if key in arrays or key not in room_debug:
            continue
        dtype = _roomseg_debug_dtype(key)
        arrays[key] = _array_from_debug(room_debug, key, shape, dtype)
    for key in ROOMSEG_ARRAY_KEYS:
        arrays.setdefault(key, np.zeros(shape, dtype=np.uint8))
    return arrays


def summarize_roomseg_arrays(arrays: Mapping[str, np.ndarray], room_debug: Mapping[str, object], step: int) -> dict:
    nav_free = _bool_array(arrays.get("navigation_free_room_domain"), _shape(arrays))
    vertical_free = _bool_array(arrays.get("vertical_free_room_domain"), _shape(arrays))
    vertical_occupied = _bool_array(arrays.get("vertical_occupied_0p2_2p0"), _shape(arrays))
    vertical_observed_0p2 = _bool_array(arrays.get("vertical_observed_0p2_2p0"), _shape(arrays))
    unknown = _bool_array(arrays.get("unknown_mask"), _shape(arrays))
    final_labels = np.asarray(arrays.get("final_room_label_map"), dtype=np.int32)
    boundary = _bool_array(arrays.get("boundary_map"), _shape(arrays))
    rep_wall = _bool_array(arrays.get("representative_wall_map"), _shape(arrays))
    ext_wall = _bool_array(arrays.get("extended_wall_map"), _shape(arrays))
    initial_occupied = _bool_array(arrays.get("initial_roomseg_occupied"), _shape(arrays))
    vertical_observed = _bool_array(arrays.get("vertical_observed_map"), _shape(arrays))
    vertical_unknown_before_overlay = _bool_array(arrays.get("vertical_unknown_before_overlay"), _shape(arrays))
    nav_raw_obstacle = _bool_array(arrays.get("nav_raw_obstacle"), _shape(arrays))
    static_structural = _bool_array(arrays.get("roomseg_static_structural_occupied"), _shape(arrays))
    overlay_candidate = _bool_array(arrays.get("nav_obstacle_overlay_candidate"), _shape(arrays))
    overlay_accepted = _bool_array(arrays.get("nav_obstacle_overlay_accepted"), _shape(arrays))
    rescued = _bool_array(arrays.get("walls_rescued_from_unknown"), _shape(arrays))
    vertical_free_over_nav_obstacle = _bool_array(arrays.get("vertical_free_over_nav_obstacle"), _shape(arrays))
    nav_obstacle_still_unknown = _bool_array(arrays.get("nav_obstacle_still_unknown_after_fusion"), _shape(arrays))
    initial_unknown_after_fusion = _bool_array(arrays.get("initial_roomseg_unknown_after_fusion"), _shape(arrays))
    repaired_occupied = _bool_array(arrays.get("repaired_roomseg_occupied"), _shape(arrays))
    candidate_wall = _bool_array(arrays.get("candidate_wall"), _shape(arrays))
    component_gate = _bool_array(arrays.get("component_gate"), _shape(arrays))
    ray_covered_count = np.asarray(arrays.get("roomseg_ray_covered_count"), dtype=np.uint32)
    terminal_wall_count = np.asarray(arrays.get("roomseg_terminal_wall_count"), dtype=np.uint32)
    terminal_wall_splat = _bool_array(arrays.get("roomseg_terminal_wall_splat"), _shape(arrays))
    ray_valid_wall = _bool_array(arrays.get("ray_valid_wall_inference"), _shape(arrays))
    unknown_before_ray_wall = _bool_array(arrays.get("unknown_before_ray_wall"), _shape(arrays))
    unknown_after_ray_wall = _bool_array(arrays.get("unknown_after_ray_wall"), _shape(arrays))
    unknown_removed_by_ray_wall = _bool_array(arrays.get("unknown_removed_by_ray_wall"), _shape(arrays))
    clipped_outside_nav = _bool_array(arrays.get("vertical_free_clipped_outside_navigation_map"), _shape(arrays))
    free_wall_conflict = _bool_array(arrays.get("free_wall_conflict_map_before_sanitize"), _shape(arrays))
    sanitized_free = _bool_array(arrays.get("roomseg_sanitized_free"), _shape(arrays))
    sanitized_wall = _bool_array(arrays.get("roomseg_sanitized_wall"), _shape(arrays))
    strict_pre_door_cut = _bool_array(arrays.get("strict_pre_extension_door_cut_mask"), _shape(arrays))
    partial_door_seed = _bool_array(arrays.get("partial_door_seed_mask"), _shape(arrays))
    partial_door_line = _bool_array(arrays.get("partial_door_line_mask"), _shape(arrays))
    partial_door_cut = _bool_array(arrays.get("partial_door_extension_cut_mask"), _shape(arrays))
    rejected_door_extension = _bool_array(arrays.get("rejected_door_extension_mask"), _shape(arrays))
    original_step_boundary = _bool_array(arrays.get("original_step1_step2_virtual_boundary_map"), _shape(arrays))
    wall_extension_boundary = _bool_array(arrays.get("wall_extension_boundary_mask"), _shape(arrays))
    door_completion_boundary = _bool_array(arrays.get("door_completion_boundary_mask"), _shape(arrays))
    nav_not_vertical = nav_free & ~vertical_free
    nav_unlabeled = nav_free & (final_labels <= 0)
    vertical_unlabeled = vertical_free & (final_labels <= 0)
    counts = {
        "navigation_free": int(np.count_nonzero(nav_free)),
        "vertical_free": int(np.count_nonzero(vertical_free)),
        "vertical_occupied_0p2_2p0": int(np.count_nonzero(vertical_occupied)),
        "vertical_observed": int(np.count_nonzero(vertical_observed)),
        "vertical_observed_0p2_2p0": int(np.count_nonzero(vertical_observed_0p2)),
        "roomseg_ray_covered": int(np.count_nonzero(ray_covered_count)),
        "roomseg_ray_covered_count_sum": int(np.sum(ray_covered_count, dtype=np.uint64)),
        "terminal_wall": int(np.count_nonzero(terminal_wall_count)),
        "terminal_wall_count_sum": int(np.sum(terminal_wall_count, dtype=np.uint64)),
        "terminal_wall_splat": int(np.count_nonzero(terminal_wall_splat)),
        "ray_valid_wall_inference": int(np.count_nonzero(ray_valid_wall)),
        "unknown_before_ray_wall": int(np.count_nonzero(unknown_before_ray_wall)),
        "unknown_after_ray_wall": int(np.count_nonzero(unknown_after_ray_wall)),
        "unknown_removed_by_ray_wall": int(np.count_nonzero(unknown_removed_by_ray_wall)),
        "vertical_free_clipped_outside_navigation": int(np.count_nonzero(clipped_outside_nav)),
        "free_wall_conflict_before_sanitize": int(np.count_nonzero(free_wall_conflict)),
        "roomseg_sanitized_free": int(np.count_nonzero(sanitized_free)),
        "roomseg_sanitized_wall": int(np.count_nonzero(sanitized_wall)),
        "strict_pre_extension_door_cut": int(np.count_nonzero(strict_pre_door_cut)),
        "partial_door_seed": int(np.count_nonzero(partial_door_seed)),
        "partial_door_line": int(np.count_nonzero(partial_door_line)),
        "partial_door_extension_cut": int(np.count_nonzero(partial_door_cut)),
        "rejected_door_extension": int(np.count_nonzero(rejected_door_extension)),
        "original_step1_step2_virtual_boundary": int(np.count_nonzero(original_step_boundary)),
        "wall_extension_boundary": int(np.count_nonzero(wall_extension_boundary)),
        "door_completion_boundary": int(np.count_nonzero(door_completion_boundary)),
        "vertical_unknown_before_overlay": int(np.count_nonzero(vertical_unknown_before_overlay)),
        "nav_raw_obstacle": int(np.count_nonzero(nav_raw_obstacle)),
        "static_structural_occupied": int(np.count_nonzero(static_structural)),
        "nav_obstacle_overlay_candidate": int(np.count_nonzero(overlay_candidate)),
        "nav_obstacle_overlay_accepted": int(np.count_nonzero(overlay_accepted)),
        "walls_rescued_from_unknown": int(np.count_nonzero(rescued)),
        "vertical_free_over_nav_obstacle": int(np.count_nonzero(vertical_free_over_nav_obstacle)),
        "nav_obstacle_still_unknown_after_fusion": int(np.count_nonzero(nav_obstacle_still_unknown)),
        "initial_roomseg_unknown_after_fusion": int(np.count_nonzero(initial_unknown_after_fusion)),
        "unknown": int(np.count_nonzero(unknown)),
        "final_labeled": int(np.count_nonzero(final_labels > 0)),
        "nav_free_not_vertical_free": int(np.count_nonzero(nav_not_vertical)),
        "nav_free_unlabeled": int(np.count_nonzero(nav_unlabeled)),
        "vertical_free_unlabeled": int(np.count_nonzero(vertical_unlabeled)),
        "boundary_on_nav_free": int(np.count_nonzero(boundary & nav_free)),
        "representative_wall_on_nav_free": int(np.count_nonzero(rep_wall & nav_free)),
        "extended_wall_on_nav_free": int(np.count_nonzero(ext_wall & nav_free)),
    }
    wall_gating_audit = {
        "wall_confidence_map_present": bool(np.asarray(arrays.get("wall_confidence_map")).size and np.any(np.asarray(arrays.get("wall_confidence_map")) > 0)),
        "candidate_wall_present": bool(np.any(candidate_wall)),
        "component_gate_present": bool(np.any(component_gate)),
        "repaired_occupied_equals_initial_occupied": bool(np.array_equal(repaired_occupied, initial_occupied)),
        "repaired_occupied_outside_candidate_wall_count": int(np.count_nonzero(repaired_occupied & ~candidate_wall)) if np.any(candidate_wall) else int(np.count_nonzero(repaired_occupied)),
        "warning": "",
    }
    sector = _sector_summary(arrays)
    likely = diagnose_roomseg_likely_cause(counts, sector, wall_gating_audit)
    evidence_debug = dict(room_debug.get("evidence_fusion") or {})
    ray_debug = dict(room_debug.get("ray_valid_wall_inference_debug") or {})
    evidence_summary = {
        "evidence_fusion_enabled": bool(evidence_debug.get("evidence_fusion_enabled", room_debug.get("evidence_fusion_enabled", True))),
        "evidence_fusion_mode": str(evidence_debug.get("evidence_fusion_mode", room_debug.get("evidence_fusion_mode", "vertical_free_nav_obstacle_wall"))),
        "vertical_free_cells": int(evidence_debug.get("vertical_free_cells", counts["vertical_free"])),
        "vertical_observed_cells": int(evidence_debug.get("vertical_observed_cells", counts["vertical_observed"])),
        "vertical_unknown_before_overlay_cells": int(evidence_debug.get("vertical_unknown_before_overlay_cells", counts["vertical_unknown_before_overlay"])),
        "nav_raw_obstacle_cells": int(evidence_debug.get("nav_raw_obstacle_cells", counts["nav_raw_obstacle"])),
        "static_structural_occupied_cells": int(evidence_debug.get("static_structural_occupied_cells", counts["static_structural_occupied"])),
        "nav_obstacle_overlay_candidate_cells": int(evidence_debug.get("nav_obstacle_overlay_candidate_cells", counts["nav_obstacle_overlay_candidate"])),
        "nav_obstacle_overlay_accepted_cells": int(evidence_debug.get("nav_obstacle_overlay_accepted_cells", counts["nav_obstacle_overlay_accepted"])),
        "walls_rescued_from_unknown_cells": int(evidence_debug.get("walls_rescued_from_unknown_cells", counts["walls_rescued_from_unknown"])),
        "vertical_free_over_nav_obstacle_cells": int(evidence_debug.get("vertical_free_over_nav_obstacle_cells", counts["vertical_free_over_nav_obstacle"])),
        "nav_obstacle_still_unknown_after_fusion_cells": int(evidence_debug.get("nav_obstacle_still_unknown_after_fusion_cells", counts["nav_obstacle_still_unknown_after_fusion"])),
        "initial_roomseg_free_cells": int(evidence_debug.get("initial_roomseg_free_cells", np.count_nonzero(_bool_array(arrays.get("initial_roomseg_free_after_fusion"), _shape(arrays))))),
        "initial_roomseg_occupied_cells": int(evidence_debug.get("initial_roomseg_occupied_cells", np.count_nonzero(_bool_array(arrays.get("initial_roomseg_occupied_after_fusion"), _shape(arrays))))),
        "initial_roomseg_unknown_cells": int(evidence_debug.get("initial_roomseg_unknown_cells", counts["initial_roomseg_unknown_after_fusion"])),
        "ray_valid_wall_inference_enabled": bool(ray_debug.get("ray_valid_wall_inference_enabled", room_debug.get("ray_valid_wall_inference_enabled", True))),
        "ray_valid_wall_inference_mode": str(ray_debug.get("ray_valid_wall_inference_mode", room_debug.get("ray_valid_wall_inference_mode", "ray_valid_terminal_wall"))),
        "depth_max_m": float(ray_debug.get("depth_max_m", room_debug.get("depth_max_m", 3.0))),
        "vertical_profile_free_min_height_m": float(ray_debug.get("vertical_profile_free_min_height_m", room_debug.get("vertical_profile_free_min_height_m", 0.2))),
        "vertical_profile_free_max_height_m": float(ray_debug.get("vertical_profile_free_max_height_m", room_debug.get("vertical_profile_free_max_height_m", 2.0))),
        "terminal_wall_cells": int(ray_debug.get("terminal_wall_cells", counts["terminal_wall"])),
        "terminal_wall_splat_cells": int(ray_debug.get("terminal_wall_splat_cells", counts["terminal_wall_splat"])),
        "unknown_before_cells": int(ray_debug.get("unknown_before_cells", counts["unknown_before_ray_wall"])),
        "unknown_after_cells": int(ray_debug.get("unknown_after_cells", counts["unknown_after_ray_wall"])),
        "unknown_removed_by_ray_wall_cells": int(ray_debug.get("unknown_removed_by_ray_wall_cells", counts["unknown_removed_by_ray_wall"])),
        "vertical_free_overridden_by_wall_cells": int(ray_debug.get("vertical_free_overridden_by_wall_cells", room_debug.get("vertical_free_overridden_by_wall_cells", 0))),
    }
    voxel_door_summary_keys = (
        "voxel_roomseg_version",
        "voxel_door_seed_ceiling_filter_version",
        "voxel_door_z_scan_min_m",
        "voxel_door_z_scan_max_m",
        "voxel_door_z_scan_top_source",
        "voxel_door_z_used_max_m",
        "voxel_door_z_used_bin_count",
        "voxel_door_raw_seed_cells",
        "voxel_door_seed_component_count",
        "voxel_door_seed_cluster_count",
        "voxel_door_line_primitive_count",
        "voxel_door_extensible_primitive_count",
        "voxel_door_extension_trial_count",
        "voxel_door_visual_accepted_count",
        "voxel_door_partition_accepted_count",
        "voxel_door_seed_reject_reason_counts",
        "voxel_door_extension_reject_reason_counts",
        "voxel_door_partition_reject_reason_counts",
        "voxel_door_primitive_reject_reason_counts",
        "voxel_door_line_primitives",
    )
    voxel_door_summary = {key: room_debug.get(key, {} if key.endswith("_counts") else 0) for key in voxel_door_summary_keys}
    voxel_door_summary["voxel_door_stable_count"] = int(
        room_debug.get("voxel_door_stable_count", room_debug.get("voxel_door_memory_track_count", 0)) or 0
    )
    return {
        "step": int(step),
        "algorithm": str(room_debug.get("algorithm", "upstream_rose2_vertical_or_free")),
        "source_mode": str(room_debug.get("source_mode", "declutter_reconstruct_mit")),
        "snapshot_source": str(room_debug.get("snapshot_source", "unspecified")),
        "roomseg_update_reason": str(room_debug.get("roomseg_update_reason", "")),
        "agent_moved_since_last_snapshot_cells": room_debug.get("agent_moved_since_last_snapshot_cells"),
        "selected_frontier_changed_from_last_snapshot": room_debug.get("selected_frontier_changed_from_last_snapshot"),
        "strict_fallback_used": bool(room_debug.get("strict_fallback_used", False)),
        "fallback_algorithm": room_debug.get("fallback_algorithm"),
        "shape": [int(v) for v in final_labels.shape],
        "counts": counts,
        **evidence_summary,
        "evidence_fusion": evidence_summary,
        "occupied_source_audit": dict(room_debug.get("occupied_source_audit") or evidence_debug.get("occupied_source_audit") or {}),
        "wall_gating_audit": wall_gating_audit,
        "selected_frontier_sector_debug": sector,
        "likely_cause": likely,
        "explanation": _explain_likely_cause(likely),
        **voxel_door_summary,
    }


def diagnose_roomseg_likely_cause(counts: Mapping[str, int], sector: Optional[Mapping[str, object]] = None, wall_gating_audit: Optional[Mapping[str, object]] = None) -> str:
    c = dict(counts)
    if sector and bool(sector.get("available")):
        overlap = dict(sector.get("overlap") or {})
        c.update(overlap)
    final_zero = max(int(c.get("final_label_zero", c.get("nav_free_unlabeled", 0))), 1)
    if int(c.get("unknown", 0)) > 0.5 * final_zero:
        return "unknown_region"
    if int(c.get("nav_free_not_vertical_free", 0)) > 0.4 * final_zero:
        return "navigation_free_not_vertical_free"
    wall_cut = int(c.get("boundary_map", c.get("boundary_on_nav_free", 0))) + int(c.get("representative_wall_map", c.get("representative_wall_on_nav_free", 0))) + int(c.get("extended_wall_map", c.get("extended_wall_on_nav_free", 0)))
    if wall_cut > 0.05 * max(int(c.get("navigation_free", 0)), 1):
        return "boundary_or_wall_line_cut"
    if int(c.get("vertical_free_unlabeled", 0)) > 0.3 * max(int(c.get("vertical_free", 0)), 1):
        return "structural_free_not_absorbed"
    audit = dict(wall_gating_audit or {})
    if bool(audit.get("repaired_occupied_equals_initial_occupied")) and int(audit.get("repaired_occupied_outside_candidate_wall_count", 0)) > 0:
        return "wall_gating_overconservative"
    return "unclear"


def render_roomseg_overlay(arrays: Mapping[str, np.ndarray], summary: Mapping[str, object]) -> Image.Image:
    shape = _shape(arrays)
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    canvas[:, :] = (42, 45, 52)
    canvas[_bool_array(arrays.get("navigation_free_room_domain"), shape)] = (165, 168, 170)
    canvas[_bool_array(arrays.get("vertical_free_room_domain"), shape)] = (220, 245, 220)
    canvas[_bool_array(arrays.get("unknown_mask"), shape)] = (26, 28, 32)
    canvas[_bool_array(arrays.get("initial_roomseg_unknown_after_fusion"), shape)] = (18, 20, 24)
    canvas[_bool_array(arrays.get("initial_roomseg_occupied_after_fusion"), shape)] = (210, 85, 45)
    canvas[_bool_array(arrays.get("ray_valid_wall_inference"), shape)] = (255, 80, 40)
    canvas[_bool_array(arrays.get("roomseg_terminal_wall_splat"), shape)] = (255, 135, 25)
    canvas[_bool_array(arrays.get("terminal_wall_roomseg_mask"), shape)] = (255, 150, 35)
    canvas[_bool_array(arrays.get("vertical_free_clipped_outside_navigation_map"), shape)] = (45, 135, 255)
    canvas[_bool_array(arrays.get("free_wall_conflict_map_before_sanitize"), shape)] = (255, 45, 45)
    canvas[_bool_array(arrays.get("roomseg_sanitized_wall"), shape)] = (10, 10, 10)
    canvas[_bool_array(arrays.get("nav_obstacle_overlay_accepted"), shape)] = (230, 40, 230)
    canvas[_bool_array(arrays.get("walls_rescued_from_unknown"), shape)] = (255, 35, 35)
    canvas[_bool_array(arrays.get("vertical_free_over_nav_obstacle"), shape)] = (45, 135, 255)
    canvas[_bool_array(arrays.get("voxel_outside_pre_traversible_xy"), shape)] = (122, 150, 122)
    canvas[_bool_array(arrays.get("voxel_outside_candidate_base_xy"), shape)] = (245, 216, 118)
    canvas[_bool_array(arrays.get("voxel_outside_robot_component_xy"), shape)] = (20, 190, 120)
    canvas[_bool_array(arrays.get("voxel_outside_connected_to_robot_rejected_xy"), shape)] = (120, 80, 225)
    canvas[_bool_array(arrays.get("voxel_outside_disconnected_candidate_xy"), shape)] = (255, 210, 55)
    canvas[_bool_array(arrays.get("voxel_outside_floor_frustum_seen_xy"), shape)] = (60, 210, 220)
    canvas[_bool_array(arrays.get("voxel_outside_holdout_floor_unseen_xy"), shape)] = (110, 98, 135)
    canvas[_bool_array(arrays.get("voxel_outside_candidate_xy"), shape)] = (245, 216, 118)
    canvas[_bool_array(arrays.get("voxel_outside_xy"), shape)] = (180, 142, 28)
    labels = np.asarray(arrays.get("final_room_label_map"), dtype=np.int32)
    _blend_labels(canvas, labels, 0.42)
    context = np.asarray(arrays.get("context_room_label_map"), dtype=np.int32)
    _blend_labels(canvas, context * ((context > 0) & (labels <= 0)), 0.70)
    canvas[_bool_array(arrays.get("nav_obstacle_overlay_accepted"), shape)] = (230, 40, 230)
    canvas[_bool_array(arrays.get("walls_rescued_from_unknown"), shape)] = (255, 35, 35)
    canvas[_bool_array(arrays.get("vertical_free_over_nav_obstacle"), shape)] = (45, 135, 255)
    canvas[_bool_array(arrays.get("boundary_map"), shape)] = (255, 45, 45)
    canvas[_bool_array(arrays.get("wall_boundary_map"), shape)] = (255, 145, 30)
    canvas[_bool_array(arrays.get("candidate_closure_map"), shape)] = (0, 220, 255)
    canvas[_bool_array(arrays.get("rejected_closure_map"), shape)] = (255, 0, 220)
    canvas[_bool_array(arrays.get("virtual_boundary_map"), shape)] = (245, 245, 245)
    canvas[_bool_array(arrays.get("accepted_closure_map"), shape)] = (80, 170, 255)
    canvas[_bool_array(arrays.get("original_step1_step2_virtual_boundary_map"), shape)] = (80, 170, 255)
    canvas[_bool_array(arrays.get("wall_extension_boundary_mask"), shape)] = (80, 170, 255)
    canvas[_bool_array(arrays.get("door_completion_boundary_mask"), shape)] = (255, 120, 40)
    canvas[_bool_array(arrays.get("pre_extension_door_detected_map"), shape)] = (0, 210, 255)
    canvas[_bool_array(arrays.get("strict_pre_extension_door_cut_mask"), shape)] = (255, 120, 40)
    canvas[_bool_array(arrays.get("partial_door_seed_mask"), shape)] = (135, 245, 255)
    canvas[_bool_array(arrays.get("partial_door_line_mask"), shape)] = (60, 250, 180)
    canvas[_bool_array(arrays.get("partial_door_extension_cut_mask"), shape)] = (255, 120, 40)
    canvas[_bool_array(arrays.get("rejected_door_extension_mask"), shape)] = (255, 60, 180)
    canvas[_bool_array(arrays.get("pre_extension_door_cut_mask"), shape)] = (255, 120, 40)
    canvas[_bool_array(arrays.get("representative_wall_map"), shape)] = (255, 145, 30)
    canvas[_bool_array(arrays.get("extended_wall_map"), shape)] = (245, 60, 210)
    canvas[_bool_array(arrays.get("frontier_map"), shape)] = (0, 220, 220)
    canvas[_bool_array(arrays.get("selected_frontier_mask"), shape)] = (255, 225, 40)
    _mark_cell(canvas, arrays.get("agent_rc"), (255, 40, 40), 3)
    return _upscale_with_title(canvas, "roomseg overlay | likely=%s" % summary.get("likely_cause", "unclear"))


def render_navigation_room_masks(arrays: Mapping[str, np.ndarray]) -> Image.Image:
    shape = _shape(arrays)
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    canvas[:, :] = (18, 20, 24)
    canvas[_bool_array(arrays.get("observed_free_mask"), shape)] = (150, 156, 160)
    canvas[_bool_array(arrays.get("unknown_mask"), shape)] = (34, 37, 44)
    canvas[_bool_array(arrays.get("obstacle_mask"), shape)] = (4, 5, 6)
    labels = np.asarray(arrays.get("final_room_label_map"), dtype=np.int32)
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == int(label)
        color = np.asarray(_label_color(label), dtype=np.float32)
        base = canvas[mask].astype(np.float32)
        canvas[mask] = np.clip(base * 0.32 + color * 0.68, 0, 255).astype(np.uint8)
    accepted_separators = _bool_array(arrays.get("accepted_separators"), shape)
    if np.any(accepted_separators):
        canvas[accepted_separators] = (245, 250, 255)
    canvas[_bool_array(arrays.get("obstacle_mask"), shape)] = (4, 5, 6)
    return Image.fromarray(canvas, mode="RGB")


def render_roomseg_layers_grid(arrays: Mapping[str, np.ndarray], summary: Mapping[str, object]) -> Image.Image:
    shape = _shape(arrays)
    items = [
        ("navigation_free", _bool_array(arrays.get("navigation_free_room_domain"), shape)),
        ("vertical_free", _bool_array(arrays.get("vertical_free_room_domain"), shape)),
        ("vertical_occupied", _bool_array(arrays.get("vertical_occupied_0p2_2p0"), shape)),
        ("vertical_observed", _bool_array(arrays.get("vertical_observed_map"), shape)),
        ("vertical_observed_0p2", _bool_array(arrays.get("vertical_observed_0p2_2p0"), shape)),
        ("ray_covered_count", np.asarray(arrays.get("roomseg_ray_covered_count"), dtype=np.uint16)),
        ("terminal_wall_count", np.asarray(arrays.get("roomseg_terminal_wall_count"), dtype=np.uint16)),
        ("terminal_wall_splat", _bool_array(arrays.get("roomseg_terminal_wall_splat"), shape)),
        ("ray_valid_wall", _bool_array(arrays.get("ray_valid_wall_inference"), shape)),
        ("unknown_before_ray", _bool_array(arrays.get("unknown_before_ray_wall"), shape)),
        ("unknown_after_ray", _bool_array(arrays.get("unknown_after_ray_wall"), shape)),
        ("unknown_removed_ray", _bool_array(arrays.get("unknown_removed_by_ray_wall"), shape)),
        ("vfree_clipped_nav", _bool_array(arrays.get("vertical_free_clipped_outside_navigation_map"), shape)),
        ("free_wall_conflict", _bool_array(arrays.get("free_wall_conflict_map_before_sanitize"), shape)),
        ("sanitized_free", _bool_array(arrays.get("roomseg_sanitized_free"), shape)),
        ("sanitized_wall", _bool_array(arrays.get("roomseg_sanitized_wall"), shape)),
        ("terminal_wall_roomseg", _bool_array(arrays.get("terminal_wall_roomseg_mask"), shape)),
        ("vertical_unknown_pre", _bool_array(arrays.get("vertical_unknown_before_overlay"), shape)),
        ("nav_raw_obstacle", _bool_array(arrays.get("nav_raw_obstacle"), shape)),
        ("static_structural", _bool_array(arrays.get("roomseg_static_structural_occupied"), shape)),
        ("overlay_candidate", _bool_array(arrays.get("nav_obstacle_overlay_candidate"), shape)),
        ("overlay_accepted", _bool_array(arrays.get("nav_obstacle_overlay_accepted"), shape)),
        ("rescued_unknown", _bool_array(arrays.get("walls_rescued_from_unknown"), shape)),
        ("vfree_over_nav_obs", _bool_array(arrays.get("vertical_free_over_nav_obstacle"), shape)),
        ("outside_boundary", _bool_array(arrays.get("voxel_outside_xy"), shape)),
        ("outside_candidate", _bool_array(arrays.get("voxel_outside_candidate_xy"), shape)),
        ("outside_candidate_base", _bool_array(arrays.get("voxel_outside_candidate_base_xy"), shape)),
        ("outside_disconnected_candidate", _bool_array(arrays.get("voxel_outside_disconnected_candidate_xy"), shape)),
        ("outside_robot_component", _bool_array(arrays.get("voxel_outside_robot_component_xy"), shape)),
        ("outside_connected_rejected", _bool_array(arrays.get("voxel_outside_connected_to_robot_rejected_xy"), shape)),
        ("outside_pre_traversible", _bool_array(arrays.get("voxel_outside_pre_traversible_xy"), shape)),
        ("outside_floor_seen", _bool_array(arrays.get("voxel_outside_floor_frustum_seen_xy"), shape)),
        ("outside_full_no_occ", _bool_array(arrays.get("voxel_outside_full_height_no_occupied_xy"), shape)),
        ("outside_holdout_floor", _bool_array(arrays.get("voxel_outside_holdout_floor_unseen_xy"), shape)),
        ("outside_score", np.asarray(arrays.get("voxel_outside_score_xy"), dtype=np.uint8)),
        ("still_unknown_obs", _bool_array(arrays.get("nav_obstacle_still_unknown_after_fusion"), shape)),
        ("initial_free_fused", _bool_array(arrays.get("initial_roomseg_free_after_fusion"), shape)),
        ("initial_occ_fused", _bool_array(arrays.get("initial_roomseg_occupied_after_fusion"), shape)),
        ("initial_unknown_fused", _bool_array(arrays.get("initial_roomseg_unknown_after_fusion"), shape)),
        ("nav_not_vertical", _bool_array(arrays.get("navigation_free_room_domain"), shape) & ~_bool_array(arrays.get("vertical_free_room_domain"), shape)),
        ("unknown", _bool_array(arrays.get("unknown_mask"), shape)),
        ("repaired_free", _bool_array(arrays.get("repaired_roomseg_free"), shape)),
        ("repaired_occupied", _bool_array(arrays.get("repaired_roomseg_occupied"), shape)),
        ("boundary", _bool_array(arrays.get("boundary_map"), shape)),
        ("wall_boundary", _bool_array(arrays.get("wall_boundary_map"), shape)),
        ("accepted_closure", _bool_array(arrays.get("accepted_closure_map"), shape)),
        ("candidate_closure", _bool_array(arrays.get("candidate_closure_map"), shape)),
        ("wall_extension", _bool_array(arrays.get("wall_extension_boundary_mask"), shape)),
        ("door_completion", _bool_array(arrays.get("door_completion_boundary_mask"), shape)),
        ("boundary_source", np.asarray(arrays.get("virtual_boundary_source_map"), dtype=np.uint8)),
        ("pre_door_detected", _bool_array(arrays.get("pre_extension_door_detected_map"), shape)),
        ("pre_door_cut", _bool_array(arrays.get("pre_extension_door_cut_mask"), shape)),
        ("strict_pre_door_cut", _bool_array(arrays.get("strict_pre_extension_door_cut_mask"), shape)),
        ("partial_seed", _bool_array(arrays.get("partial_door_seed_mask"), shape)),
        ("partial_line", _bool_array(arrays.get("partial_door_line_mask"), shape)),
        ("partial_cut", _bool_array(arrays.get("partial_door_extension_cut_mask"), shape)),
        ("partial_rejected", _bool_array(arrays.get("rejected_door_extension_mask"), shape)),
        ("orig_step_boundary", _bool_array(arrays.get("original_step1_step2_virtual_boundary_map"), shape)),
        ("pre_room_labels", np.asarray(arrays.get("pre_extension_room_label_map"), dtype=np.int32)),
        ("virtual_boundary", _bool_array(arrays.get("virtual_boundary_map"), shape)),
        ("representative_wall", _bool_array(arrays.get("representative_wall_map"), shape)),
        ("extended_wall", _bool_array(arrays.get("extended_wall_map"), shape)),
        ("final_labels", np.asarray(arrays.get("final_room_label_map"), dtype=np.int32)),
        ("nav_free_unlabeled", _bool_array(arrays.get("navigation_free_room_domain"), shape) & (np.asarray(arrays.get("final_room_label_map"), dtype=np.int32) <= 0)),
        ("vertical_unlabeled", _bool_array(arrays.get("vertical_free_room_domain"), shape) & (np.asarray(arrays.get("final_room_label_map"), dtype=np.int32) <= 0)),
        ("context_labels", np.asarray(arrays.get("context_room_label_map"), dtype=np.int32)),
        ("context_absorbed", (np.asarray(arrays.get("context_room_label_map"), dtype=np.int32) > 0) & (np.asarray(arrays.get("final_room_label_map"), dtype=np.int32) <= 0)),
        ("selected_sector", _bool_array(arrays.get("selected_frontier_sector"), shape)),
        ("sector_missing", _bool_array(arrays.get("selected_frontier_sector_missing_overlap"), shape)),
    ]
    thumbs = [_render_layer(name, data) for name, data in items]
    tw, th = thumbs[0].size
    rows = int(math.ceil(len(thumbs) / 4.0))
    grid = Image.new("RGB", (tw * 4, th * rows), (15, 17, 21))
    for idx, img in enumerate(thumbs):
        grid.paste(img, ((idx % 4) * tw, (idx // 4) * th))
    return grid


def _sector_summary(arrays: Mapping[str, np.ndarray]) -> dict:
    shape = _shape(arrays)
    sector = _bool_array(arrays.get("selected_frontier_sector"), shape)
    if not np.any(sector):
        return {"available": False, "overlap": {}}
    final_labels = np.asarray(arrays.get("final_room_label_map"), dtype=np.int32)
    overlap = {
        "navigation_free": int(np.count_nonzero(sector & _bool_array(arrays.get("navigation_free_room_domain"), shape))),
        "vertical_free": int(np.count_nonzero(sector & _bool_array(arrays.get("vertical_free_room_domain"), shape))),
        "unknown": int(np.count_nonzero(sector & _bool_array(arrays.get("unknown_mask"), shape))),
        "final_label_zero": int(np.count_nonzero(sector & (final_labels <= 0))),
        "boundary_map": int(np.count_nonzero(sector & _bool_array(arrays.get("boundary_map"), shape))),
        "representative_wall_map": int(np.count_nonzero(sector & _bool_array(arrays.get("representative_wall_map"), shape))),
        "extended_wall_map": int(np.count_nonzero(sector & _bool_array(arrays.get("extended_wall_map"), shape))),
        "nav_free_not_vertical_free": int(np.count_nonzero(sector & _bool_array(arrays.get("navigation_free_room_domain"), shape) & ~_bool_array(arrays.get("vertical_free_room_domain"), shape))),
        "vertical_free_unlabeled": int(np.count_nonzero(sector & _bool_array(arrays.get("vertical_free_room_domain"), shape) & (final_labels <= 0))),
    }
    return {"available": True, "overlap": overlap}


def _array_from_debug(debug: Mapping[str, object], key: str, shape: tuple[int, int], dtype=None) -> np.ndarray:
    raw = debug.get(key)
    if raw is None and key == "component_gate":
        raw = debug.get("structural_component_gate_mask")
    arr_dtype = dtype if dtype is not None else (np.int32 if "label" in key else bool)
    try:
        arr = np.asarray(raw, dtype=arr_dtype)
    except Exception:
        return np.zeros(shape, dtype=arr_dtype)
    if arr.shape != shape:
        return np.zeros(shape, dtype=arr_dtype)
    return arr


def _roomseg_debug_dtype(key: str):
    if key.endswith("_z_xy") or "_z_" in key or key.endswith("_z_estimate_xy"):
        return np.float32
    if key.endswith("_cells_xy") or key.endswith("_count_xy"):
        return np.uint16
    if key.endswith("_reason_id_map") or key.endswith("_reason_map"):
        return np.int32
    if (
        "ratio" in key
        or "weight" in key
        or "confidence" in key
        or "reliability" in key
        or "height_" in key
        or "depth_" in key
        or key in {"watershed_dist_struct", "watershed_dist_free_extent", "watershed_elevation"}
    ):
        return np.float32
    if key.endswith("_count_xy") or key.endswith("_count") or "_count_" in key:
        return np.uint16
    if "label" in key or key.endswith("_source_map") or key.endswith("_reason_map"):
        return np.int32
    if key.endswith("_map") and ("source" in key or "reason" in key):
        return np.int32
    return bool


def _bool_array(value: object, shape: tuple[int, int]) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=bool)
    except Exception:
        return np.zeros(shape, dtype=bool)
    if arr.shape != shape:
        return np.zeros(shape, dtype=bool)
    return arr


def _shape(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    for value in arrays.values():
        arr = np.asarray(value)
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])
    return (1, 1)


def _frontier_members_to_mask(members: Optional[Sequence[Sequence[int]]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if members is None:
        return out
    for cell in members:
        if len(cell) < 2:
            continue
        row, col = int(cell[0]), int(cell[1])
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            out[row, col] = True
    return out


def _selected_frontier_sector(agent_rc: np.ndarray, members: Optional[Sequence[Sequence[int]]], shape: tuple[int, int]) -> np.ndarray:
    agent = np.asarray(agent_rc, dtype=np.int32).ravel()
    if agent.size < 2 or int(agent[0]) < 0 or int(agent[1]) < 0 or not members:
        return np.zeros(shape, dtype=bool)
    pts = np.asarray(members, dtype=np.float32).reshape((-1, 2))
    if pts.shape[0] < 2:
        return np.zeros(shape, dtype=bool)
    center = np.asarray([float(agent[0]), float(agent[1])], dtype=np.float32)
    angles = np.arctan2(pts[:, 0] - center[0], pts[:, 1] - center[1])
    p0 = pts[int(np.argmin(angles))]
    p1 = pts[int(np.argmax(angles))]
    image = Image.new("L", (shape[1], shape[0]), 0)
    poly = [(float(agent[1]), float(agent[0])), (float(p0[1]), float(p0[0])), (float(p1[1]), float(p1[0]))]
    ImageDraw.Draw(image).polygon(poly, fill=1)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def _lines_to_mask(lines: object, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for line in list(lines or [])[:1024]:
        if not isinstance(line, Mapping):
            continue
        p0 = line.get("p0")
        p1 = line.get("p1")
        if p0 is None or p1 is None:
            continue
        for row, col in _bresenham((int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))):
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                out[row, col] = True
    return out


def _bresenham(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    out = []
    while True:
        out.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return out


def _render_layer(name: str, data: np.ndarray) -> Image.Image:
    arr = np.asarray(data)
    if arr.ndim != 2:
        arr = np.zeros((1, 1), dtype=np.uint8)
    if arr.dtype.kind in {"i", "u"} and np.max(arr) > 1:
        rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        _blend_labels(rgb, arr.astype(np.int32), 1.0)
    elif arr.dtype.kind == "f":
        norm = arr.astype(np.float32)
        if np.max(norm) > np.min(norm):
            norm = (norm - np.min(norm)) / (np.max(norm) - np.min(norm))
        rgb = np.repeat((norm * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    else:
        rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        layer_colors = {
            "outside_boundary": (180, 142, 28),
            "outside_candidate": (245, 216, 118),
            "outside_floor_seen": (60, 210, 220),
            "outside_full_no_occ": (245, 216, 118),
            "outside_holdout_floor": (110, 98, 135),
            "outside_candidate_base": (245, 216, 118),
            "outside_disconnected_candidate": (255, 210, 55),
            "outside_robot_component": (20, 190, 120),
            "outside_connected_rejected": (120, 80, 225),
            "outside_pre_traversible": (122, 150, 122),
        }
        rgb[np.asarray(arr, dtype=bool)] = layer_colors.get(str(name), (220, 220, 220))
    return _upscale_with_title(rgb, name, target=220)


def _upscale_with_title(rgb: np.ndarray, title: str, target: int = 800) -> Image.Image:
    h, w = rgb.shape[:2]
    scale = max(1, min(8, int(target / max(h, w, 1))))
    image = Image.fromarray(rgb).resize((w * scale, h * scale), Image.Resampling.NEAREST)
    out = Image.new("RGB", (image.width, image.height + 18), (15, 17, 21))
    out.paste(image, (0, 18))
    draw = ImageDraw.Draw(out)
    draw.text((4, 3), str(title)[:120], fill=(245, 245, 245), font=ImageFont.load_default())
    return out


def _blend_labels(canvas: np.ndarray, labels: np.ndarray, alpha: float) -> None:
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == label
        color = np.asarray(_label_color(label), dtype=np.float32)
        canvas[mask] = np.clip(canvas[mask].astype(np.float32) * (1.0 - float(alpha)) + color[None, :] * float(alpha), 0, 255).astype(np.uint8)


def _label_color(label: int) -> tuple[int, int, int]:
    palette = [(110, 170, 255), (255, 150, 95), (130, 220, 145), (210, 145, 255), (245, 210, 90), (95, 220, 220), (255, 120, 180)]
    return palette[(int(label) - 1) % len(palette)]


def _mark_cell(canvas: np.ndarray, rc: object, color: tuple[int, int, int], radius: int) -> None:
    arr = np.asarray(rc, dtype=np.int32).ravel()
    if arr.size < 2:
        return
    row, col = int(arr[0]), int(arr[1])
    for rr in range(row - radius, row + radius + 1):
        for cc in range(col - radius, col + radius + 1):
            if 0 <= rr < canvas.shape[0] and 0 <= cc < canvas.shape[1] and (rr - row) ** 2 + (cc - col) ** 2 <= radius * radius:
                canvas[rr, cc] = color


def _explain_likely_cause(cause: str) -> str:
    return {
        "unknown_region": "The selected diagnostic area is dominated by unknown cells, so it must stay unlabeled.",
        "navigation_free_not_vertical_free": "Navigation-free cells are missing from the ROSE2 vertical-free room domain; context overlay can safely label them without changing ROSE2.",
        "boundary_or_wall_line_cut": "ROSE2 boundary or wall-line maps overlap the known-free side and may be cutting labels.",
        "structural_free_not_absorbed": "ROSE2 structural-free cells remain unlabeled after final room absorption.",
        "wall_gating_overconservative": "Wall gating appears computed but repaired occupied still keeps many non-candidate cells.",
        "unclear": "No dominant cause was detected from the available masks.",
    }.get(str(cause), "No explanation available.")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.astype(float).tolist() if value.dtype.kind == "f" else value.astype(int).tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prune_old(out_dir: Path, max_saves: int) -> None:
    if int(max_saves) <= 0:
        return
    stem_mtimes = {}
    for path in out_dir.glob("roomseg_step_*.*"):
        stem = path.name.split(".")[0]
        try:
            mtime = float(path.stat().st_mtime)
        except OSError:
            continue
        stem_mtimes[stem] = max(float(stem_mtimes.get(stem, 0.0)), mtime)
    stems = sorted(stem_mtimes, key=lambda stem: (stem_mtimes[stem], stem))
    for stem in stems[: max(0, len(stems) - int(max_saves))]:
        for path in out_dir.glob(stem + ".*"):
            path.unlink(missing_ok=True)
