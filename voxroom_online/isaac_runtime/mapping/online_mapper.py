from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from voxroom_online.isaac_runtime.mapping.ceiling_height_estimator import CeilingHeightEstimator, CeilingHeightEstimatorConfig
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo, grid_to_world_xy, is_inside_grid, world_xy_to_grid
from voxroom_online.isaac_runtime.mapping.grid_map import OnlineGridMap
from voxroom_online.isaac_runtime.mapping.height_column_profile import HeightColumnProfileConfig, HeightColumnProfileMap
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import (
    NavigationProjection,
    VOXEL_OCCUPIED,
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)
from voxroom_online.isaac_runtime.mapping.vertical_profile import VerticalProfileMap
from voxroom_online.isaac_runtime.sensors.camera_geometry import CameraIntrinsics
from voxroom_online.isaac_runtime.sensors.depth_backproject import (
    backproject_pixels_at_depth_values,
    transform_points,
)


class OnlineMapper:
    def __init__(
        self,
        size_m: float,
        resolution_m: float,
        depth_max_m: float = 5.0,
        depth_min_m: float = 0.20,
        depth_stride_px: int = 8,
        ray_step_m: float | None = None,
        obstacle_min_height_m: float = 0.20,
        obstacle_max_height_m: float = 0.90,
        free_min_height_m: float = -1.50,
        free_max_height_m: float = 0.10,
        vertical_profile_free_min_height_m: float = 0.20,
        vertical_profile_free_max_height_m: float = 2.00,
        splat_point_threshold: int = 6,
        free_splat_point_threshold: int | None = None,
        robot_radius_m: float = 0.14,
        inflation_radius_m: float = 0.0,
        height_profile_enabled: bool = True,
        height_profile_z_min_m: float = 0.10,
        height_profile_z_max_m: float = 2.00,
        height_profile_storage_z_max_m: float | None = 3.20,
        height_profile_active_z_max_fallback_m: float = 2.80,
        height_profile_active_z_max_ceiling_ratio: float = 0.85,
        height_profile_active_z_max_cap_m: float = 2.80,
        height_profile_z_bin_size_m: float = 0.05,
        ceiling_height_estimator_config: CeilingHeightEstimatorConfig | dict | None = None,
        voxel_grid_enabled: bool = True,
        voxel_grid_z_min_m: float = -0.10,
        voxel_grid_z_max_m: float = 4.00,
        voxel_grid_z_resolution_m: float = 0.05,
        voxel_grid_active_z_min_m: float = 0.10,
        voxel_grid_active_z_max_fallback_m: float = 2.80,
        voxel_grid_active_z_max_ceiling_ratio: float = 0.85,
        voxel_grid_active_z_max_cap_m: float = 2.80,
        voxel_grid_config: VoxelOccupancyGridConfig | dict | None = None,
        voxel_outside_config: dict | None = None,
        voxel_navigation_projection_config: dict | None = None,
        voxel_navigation_blind_zone_config: dict | None = None,
        voxel_runtime_config: dict | None = None,
    ):
        self.size_m = float(size_m)
        self.resolution_m = float(resolution_m)
        self.depth_max_m = float(depth_max_m)
        self.depth_min_m = float(depth_min_m)
        self.depth_stride_px = max(1, int(depth_stride_px))
        # Kept for old callers. The online depth map now uses grid ray
        # casting, so this is only retained as a compatibility field.
        self.ray_step_m = float(ray_step_m) if ray_step_m is not None else max(self.resolution_m, 0.05)
        self.obstacle_min_height_m = float(obstacle_min_height_m)
        self.obstacle_max_height_m = float(obstacle_max_height_m)
        self.free_min_height_m = float(free_min_height_m)
        self.free_max_height_m = float(free_max_height_m)
        self.vertical_profile_free_min_height_m = float(vertical_profile_free_min_height_m)
        self.vertical_profile_free_max_height_m = float(vertical_profile_free_max_height_m)
        self.splat_point_threshold = max(1, int(splat_point_threshold))
        self.free_splat_point_threshold = (
            max(1, int(free_splat_point_threshold))
            if free_splat_point_threshold is not None
            else 1
        )
        self.robot_radius_m = float(robot_radius_m)
        # Retained for config/CLI compatibility. Runtime traversal inflation is
        # intentionally limited to the robot footprint radius only.
        self.inflation_radius_m = float(inflation_radius_m)
        self.grid = OnlineGridMap.centered(0.0, 0.0, self.size_m, self.resolution_m)
        self.last_debug_stats: dict = {"reason": "not_updated"}
        self.last_nearfield_debug_stats: dict = {"reason": "not_updated"}
        self.last_static_nearfield_debug_stats: dict = {"reason": "not_updated"}
        self.last_timing_stats: dict = {"reason": "not_updated"}
        self.static_nearfield_mask = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.roomseg_static_structural_occupied = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.depth_free_mask = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.depth_obstacle_endpoint_count = np.zeros_like(self.grid.free, dtype=np.uint16)
        self.voxel_nav_occupied_endpoint_count = np.zeros_like(self.grid.free, dtype=np.uint16)
        self.voxel_nav_free_ray_count = np.zeros_like(self.grid.free, dtype=np.uint16)
        self.voxel_nav_endpoint_decay_debug: dict[str, object] = {}
        self.current_pose_navigation_override_mask = np.zeros_like(self.grid.free, dtype=bool)
        self.last_current_footprint_override_debug: dict[str, object] = {}
        self.last_voxel_blind_zone_forced_xy = np.zeros_like(self.grid.free, dtype=bool)
        self.last_voxel_initial_blind_zone_forced_xy = np.zeros_like(self.grid.free, dtype=bool)
        self.last_voxel_current_footprint_forced_xy = np.zeros_like(self.grid.free, dtype=bool)
        self.last_voxel_outside_debug: dict[str, object] = {}
        self._obstacle_endpoint_evidence_increment = 2
        self._free_ray_obstacle_endpoint_decay = 1
        self.last_inflated_occupied = np.zeros_like(self.grid.free, dtype=bool)
        self.vertical_profile = VerticalProfileMap.zeros(self.grid.free.shape)
        voxel_cfg_raw = dict(voxel_grid_config or {}) if not isinstance(voxel_grid_config, VoxelOccupancyGridConfig) else dict(voxel_grid_config.__dict__)
        outside_cfg = dict(voxel_outside_config or {})
        self.voxel_outside_config = {
            "enabled": bool(outside_cfg.get("enabled", voxel_cfg_raw.get("outside_boundary_enabled", True))),
            "source": str(outside_cfg.get("source", voxel_cfg_raw.get("outside_source", "navigation_free_no_occupied_floor_frustum"))),
            "use_navigation_free_evidence": bool(
                outside_cfg.get("use_navigation_free_evidence", voxel_cfg_raw.get("outside_require_navigation_free", True))
            ),
            "occupied_check_z_min_m": float(outside_cfg.get("occupied_check_z_min_m", voxel_cfg_raw.get("outside_occupied_check_z_min_m", -0.10))),
            "occupied_check_z_max_m": float(outside_cfg.get("occupied_check_z_max_m", voxel_cfg_raw.get("outside_occupied_check_z_max_m", 4.00))),
            "require_floor_frustum_seen": bool(
                outside_cfg.get("require_floor_frustum_seen", voxel_cfg_raw.get("outside_require_floor_frustum_seen", True))
            ),
            "floor_visibility_mode": str(outside_cfg.get("floor_visibility_mode", "single_height")),
            "floor_visibility_rel_z_m": float(outside_cfg.get("floor_visibility_rel_z_m", voxel_cfg_raw.get("outside_floor_visibility_rel_z_m", -0.05))),
            "floor_visibility_count_threshold": int(
                outside_cfg.get("floor_visibility_count_threshold", voxel_cfg_raw.get("outside_floor_frustum_seen_count_threshold", 1))
            ),
            "exclude_forced_blind_zone_free": bool(
                outside_cfg.get("exclude_forced_blind_zone_free", voxel_cfg_raw.get("outside_exclude_initial_blind_zone_free", True))
            ),
            "exclude_current_footprint": bool(
                outside_cfg.get("exclude_current_footprint", voxel_cfg_raw.get("outside_exclude_current_footprint_free", True))
            ),
            "current_footprint_exclusion_radius_m": float(
                outside_cfg.get("current_footprint_exclusion_radius_m", voxel_cfg_raw.get("outside_current_footprint_exclusion_radius_m", 0.25))
            ),
            "candidate_increment": int(outside_cfg.get("candidate_increment", voxel_cfg_raw.get("outside_score_increment", 1))),
            "candidate_decay": int(outside_cfg.get("candidate_decay", voxel_cfg_raw.get("outside_score_decay", 1))),
            "score_threshold": int(outside_cfg.get("score_threshold", voxel_cfg_raw.get("outside_score_threshold", 2))),
            "score_max": int(outside_cfg.get("score_max", voxel_cfg_raw.get("outside_score_max", 5))),
            "use_as_navigation_boundary": bool(
                outside_cfg.get("use_as_navigation_boundary", voxel_cfg_raw.get("outside_use_as_navigation_boundary", True))
            ),
            "use_as_roomseg_domain_boundary": bool(
                outside_cfg.get("use_as_roomseg_domain_boundary", voxel_cfg_raw.get("outside_use_as_roomseg_domain_boundary", True))
            ),
            "use_as_frontier_boundary": bool(outside_cfg.get("use_as_frontier_boundary", voxel_cfg_raw.get("outside_use_as_frontier_boundary", True))),
            "use_as_wall_evidence": bool(outside_cfg.get("use_as_wall_evidence", voxel_cfg_raw.get("outside_use_as_wall_evidence", False))),
            "use_as_door_anchor": bool(outside_cfg.get("use_as_door_anchor", voxel_cfg_raw.get("outside_use_as_door_anchor", False))),
            "use_as_separator_anchor": bool(outside_cfg.get("use_as_separator_anchor", voxel_cfg_raw.get("outside_use_as_separator_anchor", False))),
            "use_as_generalized_occupied": bool(
                outside_cfg.get("use_as_generalized_occupied", voxel_cfg_raw.get("outside_use_as_generalized_occupied", False))
            ),
            "use_as_protected_structural_wall": bool(
                outside_cfg.get("use_as_protected_structural_wall", voxel_cfg_raw.get("outside_use_as_protected_structural_wall", False))
            ),
            "require_disconnected_from_robot_navigation_component": bool(
                outside_cfg.get(
                    "require_disconnected_from_robot_navigation_component",
                    voxel_cfg_raw.get("outside_require_disconnected_from_robot_navigation_component", True),
                )
            ),
            "connectivity_source": str(
                outside_cfg.get("connectivity_source", voxel_cfg_raw.get("outside_connectivity_source", "astar_traversible_pre_outside"))
            ),
            "connectivity_allow_diagonal": bool(
                outside_cfg.get("connectivity_allow_diagonal", voxel_cfg_raw.get("outside_connectivity_allow_diagonal", True))
            ),
            "connectivity_prevent_diagonal_corner_cutting": bool(
                outside_cfg.get(
                    "connectivity_prevent_diagonal_corner_cutting",
                    voxel_cfg_raw.get("outside_connectivity_prevent_diagonal_corner_cutting", True),
                )
            ),
            "robot_snap_to_traversible_max_radius_cells": int(
                outside_cfg.get(
                    "robot_snap_to_traversible_max_radius_cells",
                    voxel_cfg_raw.get("outside_robot_snap_to_traversible_max_radius_cells", 20),
                )
            ),
            "hard_clear_robot_component_scores": bool(
                outside_cfg.get("hard_clear_robot_component_scores", voxel_cfg_raw.get("outside_hard_clear_robot_component_scores", True))
            ),
            "fail_closed_if_robot_component_unavailable": bool(
                outside_cfg.get(
                    "fail_closed_if_robot_component_unavailable",
                    voxel_cfg_raw.get("outside_fail_closed_if_robot_component_unavailable", True),
                )
            ),
            "debug_robot_component_enabled": bool(
                outside_cfg.get("debug_robot_component_enabled", voxel_cfg_raw.get("outside_debug_robot_component_enabled", True))
            ),
            "debug_enabled": bool(outside_cfg.get("debug_enabled", voxel_cfg_raw.get("outside_debug_enabled", True))),
            "debug_color_rgb": tuple(outside_cfg.get("debug_color_rgb", (150, 110, 0))),
            "debug_candidate_color_rgb": tuple(outside_cfg.get("debug_candidate_color_rgb", (230, 190, 40))),
        }
        voxel_cfg_raw.update(
            {
                "outside_boundary_enabled": bool(self.voxel_outside_config["enabled"]),
                "outside_source": str(self.voxel_outside_config["source"]),
                "outside_require_navigation_free": bool(self.voxel_outside_config["use_navigation_free_evidence"]),
                "outside_occupied_check_z_min_m": float(self.voxel_outside_config["occupied_check_z_min_m"]),
                "outside_occupied_check_z_max_m": float(self.voxel_outside_config["occupied_check_z_max_m"]),
                "outside_require_floor_frustum_seen": bool(self.voxel_outside_config["require_floor_frustum_seen"]),
                "outside_floor_visibility_rel_z_m": float(self.voxel_outside_config["floor_visibility_rel_z_m"]),
                "outside_floor_frustum_seen_count_threshold": int(self.voxel_outside_config["floor_visibility_count_threshold"]),
                "outside_score_increment": int(self.voxel_outside_config["candidate_increment"]),
                "outside_score_decay": int(self.voxel_outside_config["candidate_decay"]),
                "outside_score_threshold": int(self.voxel_outside_config["score_threshold"]),
                "outside_score_max": int(self.voxel_outside_config["score_max"]),
                "outside_exclude_initial_blind_zone_free": bool(self.voxel_outside_config["exclude_forced_blind_zone_free"]),
                "outside_exclude_current_footprint_free": bool(self.voxel_outside_config["exclude_current_footprint"]),
                "outside_current_footprint_exclusion_radius_m": float(self.voxel_outside_config["current_footprint_exclusion_radius_m"]),
                "outside_use_as_navigation_boundary": bool(self.voxel_outside_config["use_as_navigation_boundary"]),
                "outside_use_as_roomseg_domain_boundary": bool(self.voxel_outside_config["use_as_roomseg_domain_boundary"]),
                "outside_use_as_frontier_boundary": bool(self.voxel_outside_config["use_as_frontier_boundary"]),
                "outside_use_as_wall_evidence": bool(self.voxel_outside_config["use_as_wall_evidence"]),
                "outside_use_as_door_anchor": bool(self.voxel_outside_config["use_as_door_anchor"]),
                "outside_use_as_separator_anchor": bool(self.voxel_outside_config["use_as_separator_anchor"]),
                "outside_use_as_generalized_occupied": bool(self.voxel_outside_config["use_as_generalized_occupied"]),
                "outside_use_as_protected_structural_wall": bool(self.voxel_outside_config["use_as_protected_structural_wall"]),
                "outside_require_disconnected_from_robot_navigation_component": bool(
                    self.voxel_outside_config["require_disconnected_from_robot_navigation_component"]
                ),
                "outside_connectivity_source": str(self.voxel_outside_config["connectivity_source"]),
                "outside_connectivity_allow_diagonal": bool(self.voxel_outside_config["connectivity_allow_diagonal"]),
                "outside_connectivity_prevent_diagonal_corner_cutting": bool(
                    self.voxel_outside_config["connectivity_prevent_diagonal_corner_cutting"]
                ),
                "outside_robot_snap_to_traversible_max_radius_cells": int(
                    self.voxel_outside_config["robot_snap_to_traversible_max_radius_cells"]
                ),
                "outside_hard_clear_robot_component_scores": bool(self.voxel_outside_config["hard_clear_robot_component_scores"]),
                "outside_fail_closed_if_robot_component_unavailable": bool(
                    self.voxel_outside_config["fail_closed_if_robot_component_unavailable"]
                ),
                "outside_debug_robot_component_enabled": bool(self.voxel_outside_config["debug_robot_component_enabled"]),
                "outside_debug_enabled": bool(self.voxel_outside_config["debug_enabled"]),
            }
        )
        voxel_cfg_raw.setdefault("enabled", bool(voxel_grid_enabled))
        voxel_cfg_raw.setdefault("z_min_m", float(voxel_grid_z_min_m))
        voxel_cfg_raw.setdefault("z_max_m", float(voxel_grid_z_max_m))
        voxel_cfg_raw.setdefault("z_resolution_m", float(voxel_grid_z_resolution_m))
        voxel_cfg_raw.setdefault("active_z_min_m", float(voxel_grid_active_z_min_m))
        voxel_cfg_raw.setdefault("active_z_max_fallback_m", float(voxel_grid_active_z_max_fallback_m))
        voxel_cfg_raw.setdefault("active_z_max_ceiling_ratio", float(voxel_grid_active_z_max_ceiling_ratio))
        voxel_cfg_raw.setdefault("active_z_max_cap_m", float(voxel_grid_active_z_max_cap_m))
        self.voxel_grid_config = VoxelOccupancyGridConfig.from_mapping(voxel_cfg_raw)
        nav_cfg = dict(voxel_navigation_projection_config or {})
        self.voxel_navigation_projection_config = {
            "obstacle_z_min_m": float(nav_cfg.get("obstacle_z_min_m", self.obstacle_min_height_m)),
            "obstacle_z_max_m": float(nav_cfg.get("obstacle_z_max_m", self.obstacle_max_height_m)),
            "free_z_min_m": float(nav_cfg.get("free_z_min_m", 0.10)),
            "free_z_max_m": float(nav_cfg.get("free_z_max_m", self.obstacle_max_height_m)),
            "min_free_voxels": int(nav_cfg.get("min_free_voxels", 1)),
            "occupied_any_voxel_wins": bool(nav_cfg.get("occupied_any_voxel_wins", True)),
            "occupied_use_endpoint_hysteresis": bool(nav_cfg.get("occupied_use_endpoint_hysteresis", True)),
            "occupied_endpoint_count_threshold": int(nav_cfg.get("occupied_endpoint_count_threshold", 1)),
            "occupied_endpoint_decay_per_free_ray": int(nav_cfg.get("occupied_endpoint_decay_per_free_ray", 1)),
            "occupied_endpoint_increment": int(nav_cfg.get("occupied_endpoint_increment", 2)),
            "occupied_endpoint_xy_splat_radius_cells": int(nav_cfg.get("occupied_endpoint_xy_splat_radius_cells", 0)),
            "occupied_endpoint_z_splat_radius_cells": int(nav_cfg.get("occupied_endpoint_z_splat_radius_cells", 0)),
            "occupied_close_radius_cells": int(nav_cfg.get("occupied_close_radius_cells", 1)),
            "occupied_fill_small_holes_max_area_cells": int(nav_cfg.get("occupied_fill_small_holes_max_area_cells", 4)),
            "occupied_priority_over_free": bool(nav_cfg.get("occupied_priority_over_free", True)),
            "unknown_preserve_when_no_observation": bool(nav_cfg.get("unknown_preserve_when_no_observation", True)),
            "debug_navigation_projection_layers": bool(nav_cfg.get("debug_navigation_projection_layers", False)),
            "incremental_enabled": bool(nav_cfg.get("incremental_enabled", True)),
            "full_refresh_on_frontier_update": bool(nav_cfg.get("full_refresh_on_frontier_update", True)),
            "full_refresh_interval_steps": int(nav_cfg.get("full_refresh_interval_steps", 30)),
            "dirty_dilation_radius_cells": int(nav_cfg.get("dirty_dilation_radius_cells", 2)),
            "local_morphology_enabled": bool(nav_cfg.get("local_morphology_enabled", True)),
            "full_morphology_only_on_replan": bool(nav_cfg.get("full_morphology_only_on_replan", True)),
        }
        self.voxel_grid_drives_navigation = bool(self.voxel_grid_config.voxel_grid_drives_navigation)
        self.voxel_grid = VoxelOccupancyGrid3D.zeros(self.grid.free.shape, self.grid.map_info, self.voxel_grid_config)
        self._warm_voxel_cpu_numba_backend()
        self.last_voxel_navigation_projection: NavigationProjection | None = None
        self.force_full_navigation_projection_once = True
        blind_cfg = dict(voxel_navigation_blind_zone_config or {})
        self.voxel_navigation_blind_zone_config = {
            "enabled": bool(blind_cfg.get("enabled", True)),
            "force_initial_blind_zone_free": bool(blind_cfg.get("force_initial_blind_zone_free", True)),
            "initial_blind_zone_radius_m": float(blind_cfg.get("initial_blind_zone_radius_m", 0.80)),
            "initial_blind_zone_steps": int(blind_cfg.get("initial_blind_zone_steps", 60)),
            "force_current_footprint_free": bool(blind_cfg.get("force_current_footprint_free", True)),
            "current_footprint_radius_m": float(blind_cfg.get("current_footprint_radius_m", 0.14)),
            "current_footprint_extra_margin_m": float(blind_cfg.get("current_footprint_extra_margin_m", 0.0)),
            "current_footprint_clear_dynamic_occupied": bool(blind_cfg.get("current_footprint_clear_dynamic_occupied", True)),
            "current_footprint_preserve_static_structural": bool(blind_cfg.get("current_footprint_preserve_static_structural", True)),
            "current_footprint_preserve_roomseg_static_wall": bool(blind_cfg.get("current_footprint_preserve_roomseg_static_wall", True)),
            "current_footprint_preserve_voxel_hard_static_wall": bool(blind_cfg.get("current_footprint_preserve_voxel_hard_static_wall", True)),
            "current_footprint_ignore_dynamic_voxel_occupied_columns": bool(blind_cfg.get("current_footprint_ignore_dynamic_voxel_occupied_columns", True)),
            "write_to_voxel_grid": bool(blind_cfg.get("write_to_voxel_grid", True)),
            "write_to_grid": bool(blind_cfg.get("write_to_grid", True)),
            "free_z_min_m": float(blind_cfg.get("free_z_min_m", 0.10)),
            "free_z_max_m": float(blind_cfg.get("free_z_max_m", 0.90)),
            "free_all_bins_in_range": bool(blind_cfg.get("free_all_bins_in_range", True)),
            "preserve_occupied_cells": bool(blind_cfg.get("preserve_occupied_cells", True)),
            "preserve_static_structural_cells": bool(blind_cfg.get("preserve_static_structural_cells", True)),
            "preserve_voxel_occupied_columns": bool(blind_cfg.get("preserve_voxel_occupied_columns", True)),
        }
        self._initial_blind_zone_center_world_xy: tuple[float, float] | None = None
        self._update_step_index = 0
        self.last_voxel_blind_zone_debug: dict[str, object] = {}
        runtime_cfg = dict(voxel_runtime_config or {})
        self.voxel_runtime_debug_level = str(runtime_cfg.get("runtime_debug_level", "counters") or "counters").strip().lower()
        self.skip_legacy_vertical_profile_when_voxel_backend = bool(runtime_cfg.get("skip_legacy_vertical_profile_when_voxel_backend", False))
        self.skip_legacy_height_profile_when_voxel_backend = bool(runtime_cfg.get("skip_legacy_height_profile_when_voxel_backend", False))
        self.skip_legacy_roomseg_ray_evidence_when_voxel_backend = bool(runtime_cfg.get("skip_legacy_roomseg_ray_evidence_when_voxel_backend", False))
        self._last_legacy_vertical_profile_skipped_by_voxel_backend = False
        self._last_legacy_height_profile_skipped_by_voxel_backend = False
        self._last_legacy_roomseg_ray_skipped_by_voxel_backend = False
        storage_z_max = float(height_profile_storage_z_max_m if height_profile_storage_z_max_m is not None else height_profile_z_max_m)
        self.height_profile_config = HeightColumnProfileConfig(
            enabled=bool(height_profile_enabled),
            z_min_m=float(height_profile_z_min_m),
            z_max_m=float(storage_z_max),
            storage_z_max_m=float(storage_z_max),
            active_z_min_m=float(height_profile_z_min_m),
            active_z_max_m=float(height_profile_active_z_max_cap_m if height_profile_active_z_max_cap_m > 0.0 else height_profile_z_max_m),
            active_z_max_fallback_m=float(height_profile_active_z_max_fallback_m),
            active_z_max_ceiling_ratio=float(height_profile_active_z_max_ceiling_ratio),
            active_z_max_cap_m=float(height_profile_active_z_max_cap_m),
            z_bin_size_m=float(height_profile_z_bin_size_m),
        )
        self.ceiling_height_estimator = CeilingHeightEstimator(
            ceiling_height_estimator_config,
            active_z_min_m=float(height_profile_z_min_m),
            storage_z_max_m=float(storage_z_max),
            active_z_max_fallback_m=float(height_profile_active_z_max_fallback_m),
            active_z_max_ceiling_ratio=float(height_profile_active_z_max_ceiling_ratio),
            active_z_max_cap_m=float(height_profile_active_z_max_cap_m),
        )
        self.last_ceiling_height_estimate = self.ceiling_height_estimator.last_estimate
        self.height_profile = HeightColumnProfileMap.zeros(self.grid.free.shape, self.height_profile_config)
        self._reset_roomseg_ray_evidence()

    def _warm_voxel_cpu_numba_backend(self) -> None:
        if not bool(getattr(self.voxel_grid_config, "enabled", True)):
            return
        if str(getattr(self.voxel_grid_config, "integration_backend", "")).strip().lower() != "cpu_numba":
            return
        from voxroom_online.isaac_runtime.mapping.voxel_cpu_numba_backend import VoxelCpuNumbaBackend

        VoxelCpuNumbaBackend.warmup(self.voxel_grid)

    def reset(self, start_xy: Tuple[float, float]) -> None:
        self.grid = OnlineGridMap.centered(start_xy[0], start_xy[1], self.size_m, self.resolution_m)
        self.last_debug_stats = {"reason": "reset"}
        self.last_nearfield_debug_stats = {"reason": "reset"}
        self.last_static_nearfield_debug_stats = {"reason": "reset"}
        self.last_timing_stats = {"reason": "reset"}
        self.static_nearfield_mask = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.roomseg_static_structural_occupied = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.depth_free_mask = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.depth_obstacle_endpoint_count = np.zeros_like(self.grid.free, dtype=np.uint16)
        self.voxel_nav_occupied_endpoint_count = np.zeros_like(self.grid.free, dtype=np.uint16)
        self.voxel_nav_free_ray_count = np.zeros_like(self.grid.free, dtype=np.uint16)
        self.voxel_nav_endpoint_decay_debug = {}
        self.current_pose_navigation_override_mask = np.zeros_like(self.grid.free, dtype=bool)
        self.last_current_footprint_override_debug = {}
        self.last_voxel_blind_zone_forced_xy = np.zeros_like(self.grid.free, dtype=bool)
        self.last_voxel_initial_blind_zone_forced_xy = np.zeros_like(self.grid.free, dtype=bool)
        self.last_voxel_current_footprint_forced_xy = np.zeros_like(self.grid.free, dtype=bool)
        self.last_voxel_outside_debug = {}
        self.last_inflated_occupied = np.zeros_like(self.grid.free, dtype=bool)
        self.vertical_profile = VerticalProfileMap.zeros(self.grid.free.shape)
        self.voxel_grid = VoxelOccupancyGrid3D.zeros(self.grid.free.shape, self.grid.map_info, self.voxel_grid_config)
        self.last_voxel_navigation_projection = None
        self.force_full_navigation_projection_once = True
        self._initial_blind_zone_center_world_xy = (float(start_xy[0]), float(start_xy[1]))
        self._update_step_index = 0
        self.last_voxel_blind_zone_debug = {}
        self.height_profile.reset_shape(self.grid.free.shape)
        self.ceiling_height_estimator.reset()
        self.last_ceiling_height_estimate = self.ceiling_height_estimator.last_estimate
        self._reset_roomseg_ray_evidence()

    def update_simple_radius(self, base_pose_world: Tuple[float, float, float, float], radius_m: float = 1.5) -> OnlineGridMap:
        # Conservative fallback mapping for smoke tests: mark a local disk free.
        rr = int(radius_m / self.grid.map_info.resolution_m)
        center = self.grid.world_to_grid(base_pose_world[0], base_pose_world[1])
        for dr in range(-rr, rr + 1):
            for dc in range(-rr, rr + 1):
                if dr * dr + dc * dc <= rr * rr:
                    self.grid.mark_free(center[0] + dr, center[1] + dc)
        return self.grid

    def _mark_sensor_effective_range(
        self,
        *,
        pixels_uv: np.ndarray,
        intr: CameraIntrinsics,
        camera_pose_world: Tuple[float, float, float, float],
        floor_z: float,
    ) -> dict[str, object]:
        sparse_enabled = bool(
            self.voxel_grid_config.enabled
            and getattr(self.voxel_grid_config, "sensor_range_tracking_enabled", True)
            and getattr(self.voxel_grid_config, "sensor_range_mark_effective_frustum_enabled", True)
        )
        projective_enabled = bool(
            self.voxel_grid_config.enabled
            and getattr(self.voxel_grid_config, "sensor_range_tracking_enabled", True)
            and getattr(self.voxel_grid_config, "sensor_range_projective_frustum_volume_enabled", True)
        )
        debug: dict[str, object] = {
            "voxel_sensor_range_mark_effective_frustum_enabled": bool(sparse_enabled),
            "voxel_sensor_range_projective_frustum_volume_enabled": bool(projective_enabled),
            "sensor_effective_range_rays": 0,
            "sensor_effective_range_updates": 0,
            "sensor_projective_frustum_updates": 0,
            "sensor_effective_range_ms": 0.0,
        }
        if not sparse_enabled and not projective_enabled:
            return debug
        started_at = time.perf_counter()
        projective_updates = 0
        if projective_enabled:
            projective_debug = self.voxel_grid.mark_sensor_projective_frustum_volume(
                camera_pose_world=camera_pose_world,
                intr=intr,
                floor_z=float(floor_z),
                depth_min_m=float(self.depth_min_m),
                depth_max_m=float(self.depth_max_m),
            )
            debug.update(dict(projective_debug or {}))
            projective_updates = int(
                debug.get(
                    "voxel_sensor_projective_frustum_updates",
                    debug.get("voxel_sensor_range_projective_updates", 0),
                )
                or 0
            )
        floor_frustum_debug = self.voxel_grid.mark_floor_projective_frustum_visibility(
            camera_pose_world=camera_pose_world,
            intr=intr,
            floor_z=float(floor_z),
            depth_min_m=float(self.depth_min_m),
            depth_max_m=float(self.depth_max_m),
            floor_rel_z_m=float(self.voxel_outside_config.get("floor_visibility_rel_z_m", -0.05)),
            count_delta=1,
            count_max=65535,
        )
        debug.update(dict(floor_frustum_debug or {}))
        sparse_updates = 0
        pixels = np.asarray(pixels_uv, dtype=np.float32).reshape(-1, 2)
        if sparse_enabled and pixels.size:
            depth_values = np.full((int(pixels.shape[0]),), float(self.depth_max_m), dtype=np.float32)
            points_cam = backproject_pixels_at_depth_values(pixels, depth_values, intr)
            points_world = transform_points(points_cam, camera_pose_world)
            sparse_updates = self.voxel_grid.mark_sensor_effective_range_rays(
                camera_origin_world=np.asarray(camera_pose_world[:3], dtype=np.float32),
                range_endpoints_world=points_world,
                floor_z=float(floor_z),
            )
            debug["sensor_effective_range_rays"] = int(points_world.shape[0])
        debug["sensor_effective_range_updates"] = int(sparse_updates + projective_updates)
        debug["sensor_projective_frustum_updates"] = int(projective_updates)
        debug["sensor_effective_range_ms"] = _elapsed_ms(started_at)
        return debug

    def update(self, depth: np.ndarray, intr: CameraIntrinsics, base_pose_world: Tuple[float, float, float, float], camera_pose_world: Tuple[float, float, float, float]) -> OnlineGridMap:
        total_started_at = time.perf_counter()
        timings: dict[str, float] = {}
        stage_started_at = total_started_at
        self.current_pose_navigation_override_mask[:, :] = False
        self.last_current_footprint_override_debug = {}
        self.last_voxel_blind_zone_forced_xy[:, :] = False
        self.last_voxel_initial_blind_zone_forced_xy[:, :] = False
        self.last_voxel_current_footprint_forced_xy[:, :] = False
        self.last_voxel_outside_debug = {}
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[:, :, 0]
        if depth_arr.ndim != 2 or depth_arr.size == 0:
            timings["depth_prepare_ms"] = _elapsed_ms(stage_started_at)
            stage_started_at = time.perf_counter()
            self.last_debug_stats = {"reason": "invalid_depth_shape", "depth_shape": list(depth_arr.shape)}
            self._mark_robot_footprint_free(base_pose_world)
            timings["robot_footprint_ms"] = _elapsed_ms(stage_started_at)
            self._finish_timing_stats(timings, total_started_at, reason="invalid_depth_shape")
            return self.grid

        stride = self.depth_stride_px
        vs = np.arange(stride // 2, min(depth_arr.shape[0], intr.height), stride, dtype=np.int32)
        us = np.arange(stride // 2, min(depth_arr.shape[1], intr.width), stride, dtype=np.int32)
        if len(vs) == 0 or len(us) == 0:
            timings["depth_prepare_ms"] = _elapsed_ms(stage_started_at)
            stage_started_at = time.perf_counter()
            self.last_debug_stats = {"reason": "no_sample_pixels", "depth_shape": list(depth_arr.shape)}
            self._mark_robot_footprint_free(base_pose_world)
            timings["robot_footprint_ms"] = _elapsed_ms(stage_started_at)
            self._finish_timing_stats(timings, total_started_at, reason="no_sample_pixels")
            return self.grid

        uu, vv = np.meshgrid(us, vs)
        pixels = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1).astype(np.float32)
        sampled_depth = depth_arr[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]
        depth_min = float(self.depth_min_m)
        depth_max = float(self.depth_max_m)
        depth_eps = max(0.0, float(getattr(self.voxel_grid_config, "depth_max_epsilon_m", 0.01)))
        finite_depth = np.isfinite(sampled_depth)
        hit_mask = finite_depth & (sampled_depth > depth_min) & (sampled_depth < depth_max - depth_eps)
        free_only_mask = np.zeros_like(hit_mask, dtype=bool)
        if bool(getattr(self.voxel_grid_config, "max_range_free_only_rays_enabled", True)):
            if bool(getattr(self.voxel_grid_config, "depth_greater_than_max_marks_free_to_max", True)):
                free_only_mask |= finite_depth & (sampled_depth >= depth_max - depth_eps)
            if bool(getattr(self.voxel_grid_config, "depth_inf_marks_free_to_max", True)):
                free_only_mask |= np.isposinf(sampled_depth)
            if bool(getattr(self.voxel_grid_config, "depth_nan_marks_free_to_max", False)):
                free_only_mask |= np.isnan(sampled_depth)
            if bool(getattr(self.voxel_grid_config, "depth_zero_marks_free_to_max", False)):
                free_only_mask |= finite_depth & (sampled_depth <= 0.0)
        ray_mask = hit_mask | free_only_mask
        bad_invalid_mask = ~ray_mask
        floor_z = float(base_pose_world[2])
        sensor_effective_range_debug = self._mark_sensor_effective_range(
            pixels_uv=pixels,
            intr=intr,
            camera_pose_world=camera_pose_world,
            floor_z=floor_z,
        )
        timings["sensor_effective_range_ms"] = float(sensor_effective_range_debug.get("sensor_effective_range_ms", 0.0) or 0.0)
        if int(np.count_nonzero(ray_mask)) == 0:
            timings["depth_prepare_ms"] = _elapsed_ms(stage_started_at)
            stage_started_at = time.perf_counter()
            self.last_debug_stats = {
                "reason": "no_integrable_depth_rays",
                "sampled_pixels": int(len(pixels)),
                "sampled_depth_hit_count": int(np.count_nonzero(hit_mask)),
                "sampled_depth_free_only_to_max_count": int(np.count_nonzero(free_only_mask)),
                "sampled_depth_bad_invalid_count": int(np.count_nonzero(bad_invalid_mask)),
                "sampled_depth_inf_count": int(np.count_nonzero(np.isposinf(sampled_depth))),
                "sampled_depth_ge_max_count": int(np.count_nonzero(finite_depth & (sampled_depth >= depth_max - depth_eps))),
                "sampled_depth_hit_endpoint_in_map_count": 0,
                "sampled_depth_hit_endpoint_outside_map_count": int(np.count_nonzero(hit_mask)),
                "voxel_depth_hit_ray_count": 0,
                "voxel_depth_free_only_ray_count": 0,
                "voxel_depth_endpoint_is_hit_count": 0,
                "voxel_depth_endpoint_is_free_only_count": 0,
                "voxel_depth_endpoint_outside_map_count": 0,
                "voxel_sensor_depth_range_semantics": str(getattr(self.voxel_grid_config, "depth_range_semantics", "image_plane_z")),
                "depth_shape": list(depth_arr.shape),
                **sensor_effective_range_debug,
            }
            self._mark_robot_footprint_free(base_pose_world)
            timings["robot_footprint_ms"] = _elapsed_ms(stage_started_at)
            self._finish_timing_stats(timings, total_started_at, reason="no_integrable_depth_rays")
            return self.grid

        ray_pixels = pixels[ray_mask]
        ray_depth = sampled_depth[ray_mask].astype(np.float32, copy=True)
        ray_endpoint_is_hit = hit_mask[ray_mask].astype(bool, copy=False)
        ray_depth[~ray_endpoint_is_hit] = np.float32(depth_max)
        hit_pixels = pixels[hit_mask]
        hit_depth = sampled_depth[hit_mask].astype(np.float32, copy=False)
        timings["depth_prepare_ms"] = _elapsed_ms(stage_started_at)
        stage_started_at = time.perf_counter()
        points_cam = backproject_pixels_at_depth_values(ray_pixels, ray_depth, intr)
        points_world = transform_points(points_cam, camera_pose_world)
        ray_rows_cols = _world_points_to_grid(points_world, self.grid.map_info) if points_world.size else np.zeros((0, 2), dtype=np.int32)
        ray_endpoint_in_bounds = (
            (ray_rows_cols[:, 0] >= 0)
            & (ray_rows_cols[:, 0] < self.grid.map_info.height)
            & (ray_rows_cols[:, 1] >= 0)
            & (ray_rows_cols[:, 1] < self.grid.map_info.width)
        ) if ray_rows_cols.size else np.zeros((0,), dtype=bool)
        if hit_pixels.size:
            hit_points_cam = backproject_pixels_at_depth_values(hit_pixels, hit_depth, intr)
            hit_points_world = transform_points(hit_points_cam, camera_pose_world)
        else:
            hit_points_world = np.zeros((0, 3), dtype=np.float32)
        if self._initial_blind_zone_center_world_xy is None:
            self._initial_blind_zone_center_world_xy = (float(base_pose_world[0]), float(base_pose_world[1]))

        hit_rel_z = hit_points_world[:, 2].astype(np.float32) - floor_z if hit_points_world.size else np.zeros((0,), dtype=np.float32)
        hit_rows_cols = _world_points_to_grid(hit_points_world, self.grid.map_info) if hit_points_world.size else np.zeros((0, 2), dtype=np.int32)
        hit_in_bounds = (
            (hit_rows_cols[:, 0] >= 0)
            & (hit_rows_cols[:, 0] < self.grid.map_info.height)
            & (hit_rows_cols[:, 1] >= 0)
            & (hit_rows_cols[:, 1] < self.grid.map_info.width)
        ) if hit_rows_cols.size else np.zeros((0,), dtype=bool)
        timings["depth_project_ms"] = _elapsed_ms(stage_started_at)
        if int(points_world.shape[0]) == 0:
            self.last_debug_stats = {
                "reason": "no_ray_points",
                "valid_points": int(len(points_world)),
                "depth_shape": list(depth_arr.shape),
                "depth_m_percentiles": _percentiles(hit_depth),
                "voxel_depth_hit_ray_count": int(np.count_nonzero(ray_endpoint_is_hit)),
                "voxel_depth_free_only_ray_count": int(np.count_nonzero(~ray_endpoint_is_hit)),
                "voxel_depth_endpoint_is_hit_count": int(np.count_nonzero(ray_endpoint_is_hit)),
                "voxel_depth_endpoint_is_free_only_count": int(np.count_nonzero(~ray_endpoint_is_hit)),
                "voxel_depth_endpoint_outside_map_count": int(ray_endpoint_in_bounds.size - np.count_nonzero(ray_endpoint_in_bounds)),
                "voxel_sensor_depth_range_semantics": str(getattr(self.voxel_grid_config, "depth_range_semantics", "image_plane_z")),
                **sensor_effective_range_debug,
            }
            self._finish_timing_stats(timings, total_started_at, reason="no_ray_points")
            return self.grid

        origin_cell = self.grid.world_to_grid(float(camera_pose_world[0]), float(camera_pose_world[1]))
        if not is_inside_grid(origin_cell[0], origin_cell[1], self.grid.map_info):
            origin_cell = self.grid.world_to_grid(float(base_pose_world[0]), float(base_pose_world[1]))
        if not is_inside_grid(origin_cell[0], origin_cell[1], self.grid.map_info):
            self.last_debug_stats = {
                "reason": "ray_origin_out_of_bounds",
                "valid_points": int(np.count_nonzero(hit_in_bounds)),
                "depth_shape": list(depth_arr.shape),
                "depth_m_percentiles": _percentiles(hit_depth[hit_in_bounds] if hit_in_bounds.size else hit_depth),
                "camera_pose_world": [float(v) for v in camera_pose_world],
                "base_pose_world": [float(v) for v in base_pose_world],
                "voxel_depth_hit_ray_count": int(np.count_nonzero(ray_endpoint_is_hit)),
                "voxel_depth_free_only_ray_count": int(np.count_nonzero(~ray_endpoint_is_hit)),
                "voxel_depth_endpoint_is_hit_count": int(np.count_nonzero(ray_endpoint_is_hit)),
                "voxel_depth_endpoint_is_free_only_count": int(np.count_nonzero(~ray_endpoint_is_hit)),
                "voxel_depth_endpoint_outside_map_count": int(ray_endpoint_in_bounds.size - np.count_nonzero(ray_endpoint_in_bounds)),
                "voxel_sensor_depth_range_semantics": str(getattr(self.voxel_grid_config, "depth_range_semantics", "image_plane_z")),
                **sensor_effective_range_debug,
            }
            self._finish_timing_stats(timings, total_started_at, reason="ray_origin_out_of_bounds")
            return self.grid

        in_bounds_pixels = hit_pixels[hit_in_bounds] if hit_in_bounds.size else np.zeros((0, 2), dtype=np.float32)
        in_bounds_depth = hit_depth[hit_in_bounds] if hit_in_bounds.size else np.zeros((0,), dtype=np.float32)
        rows_cols = hit_rows_cols[hit_in_bounds] if hit_in_bounds.size else np.zeros((0, 2), dtype=np.int32)
        rel_z = hit_rel_z[hit_in_bounds] if hit_in_bounds.size else np.zeros((0,), dtype=np.float32)
        voxel_backend_active = bool(self.voxel_grid_config.enabled and self.voxel_grid_drives_navigation)
        skip_legacy_vertical_profile = bool(voxel_backend_active and self.skip_legacy_vertical_profile_when_voxel_backend)
        skip_legacy_height_profile = bool(voxel_backend_active and self.skip_legacy_height_profile_when_voxel_backend)
        skip_legacy_roomseg_ray = bool(voxel_backend_active and self.skip_legacy_roomseg_ray_evidence_when_voxel_backend)
        self._last_legacy_vertical_profile_skipped_by_voxel_backend = bool(skip_legacy_vertical_profile)
        self._last_legacy_height_profile_skipped_by_voxel_backend = bool(skip_legacy_height_profile)
        self._last_legacy_roomseg_ray_skipped_by_voxel_backend = bool(skip_legacy_roomseg_ray)
        if bool(self.voxel_grid_config.enabled):
            self._mark_voxel_navigation_endpoint_evidence(rows_cols, rel_z)
        if bool(self.height_profile_config.enabled) and not bool(self.voxel_grid_config.enabled):
            self.last_ceiling_height_estimate = self.ceiling_height_estimator.update(rel_z, rows_cols)
        if bool(self.height_profile_config.enabled) and not bool(skip_legacy_height_profile):
            self.height_profile_config.active_z_max_m = float(self.last_ceiling_height_estimate.active_z_max_m)
            self.height_profile.active_z_min_m = float(self.height_profile_config.active_z_min_m)
            self.height_profile.active_z_max_m = float(self.height_profile_config.active_z_max_m)
        voxel_nav_projection = None
        if bool(self.voxel_grid_config.enabled):
            status = "locked" if bool(self.last_ceiling_height_estimate.locked) else ("stable" if bool(self.last_ceiling_height_estimate.stable) else "fallback")
            self.voxel_grid.set_active_z_from_ceiling(self.last_ceiling_height_estimate.height_m, status=status)
            voxel_stats = self.voxel_grid.integrate_depth_points(
                camera_origin_world=np.asarray(camera_pose_world[:3], dtype=np.float32),
                points_world=points_world,
                depths_m=ray_depth,
                floor_z=floor_z,
                endpoint_is_hit=ray_endpoint_is_hit,
            )
            if bool(voxel_stats.python_debug_backend_used):
                raise RuntimeError("python_debug voxel backend must not be used in runtime")
            self.last_ceiling_height_estimate = self.ceiling_height_estimator.update_from_occupied_layers(
                self.voxel_grid.state,
                self.voxel_grid.z_centers_m,
            )
            status = "locked" if bool(self.last_ceiling_height_estimate.locked) else ("stable" if bool(self.last_ceiling_height_estimate.stable) else "fallback")
            self.voxel_grid.set_active_z_from_ceiling(self.last_ceiling_height_estimate.height_m, status=status)
            if bool(self.height_profile_config.enabled) and not bool(skip_legacy_height_profile):
                self.height_profile_config.active_z_max_m = float(self.last_ceiling_height_estimate.active_z_max_m)
                self.height_profile.active_z_min_m = float(self.height_profile_config.active_z_min_m)
                self.height_profile.active_z_max_m = float(self.height_profile_config.active_z_max_m)
            floor_endpoint_mask = rel_z <= float(self.free_max_height_m)
            if np.any(floor_endpoint_mask):
                free_z_idx = self.voxel_grid.z_index_for_height(
                    max(float(self.voxel_navigation_projection_config.get("free_z_min_m", 0.10)), float(self.voxel_grid.active_z_min_m))
                )
                if free_z_idx is not None:
                    rc = np.asarray(rows_cols[floor_endpoint_mask], dtype=np.int32)
                    if rc.size:
                        floor_cells = np.empty((int(rc.shape[0]), 3), dtype=np.int32)
                        floor_cells[:, 0] = int(free_z_idx)
                        floor_cells[:, 1] = rc[:, 0]
                        floor_cells[:, 2] = rc[:, 1]
                        _count, changed = self.voxel_grid.mark_free_voxels_array(floor_cells)
                        self.voxel_grid.refresh_state_indices(changed)
                        if changed.size:
                            hw = int(self.grid.map_info.height * self.grid.map_info.width)
                            if getattr(self.voxel_grid, "last_dirty_rc_flags", None) is None or int(np.asarray(self.voxel_grid.last_dirty_rc_flags).size) != hw:
                                self.voxel_grid.last_dirty_rc_flags = np.zeros(hw, dtype=np.uint8)
                            self.voxel_grid.last_dirty_rc_flags[np.asarray(changed % max(1, hw), dtype=np.int64)] = 1
            blind_started_at = time.perf_counter()
            self.last_voxel_blind_zone_debug = self._apply_voxel_navigation_blind_zone(
                base_pose_world,
                write_to_voxel=bool(self.voxel_navigation_blind_zone_config.get("write_to_voxel_grid", True)),
                write_to_grid=False,
            )
            timings["voxel_blind_zone_voxel_ms"] = _elapsed_ms(blind_started_at)
        else:
            self.last_voxel_navigation_projection = None
            self.last_voxel_blind_zone_debug = {}
        stage_started_at = time.perf_counter()
        if not bool(skip_legacy_vertical_profile):
            self.vertical_profile.mark_occupied_points(rows_cols, rel_z)
        if bool(self.height_profile_config.enabled) and not bool(skip_legacy_height_profile):
            self.height_profile.mark_occupied_points(rows_cols, rel_z)
        timings["vertical_profile_occupied_ms"] = _elapsed_ms(stage_started_at)
        timings["legacy_vertical_profile_skipped"] = 1.0 if bool(skip_legacy_vertical_profile) else 0.0
        timings["legacy_height_profile_skipped"] = 1.0 if bool(skip_legacy_height_profile) else 0.0
        obstacle_mask = (rel_z >= self.obstacle_min_height_m) & (rel_z <= self.obstacle_max_height_m)
        free_mask = (rel_z >= self.free_min_height_m) & (rel_z <= self.free_max_height_m)
        ray_clear_mask = (rel_z >= self.free_min_height_m) & (rel_z <= self.obstacle_max_height_m)
        camera_rel_z = float(camera_pose_world[2]) - floor_z
        map_width = int(self.grid.map_info.width)
        stage_started_at = time.perf_counter()
        skip_all_legacy_ray_cast = bool(skip_legacy_roomseg_ray and skip_legacy_vertical_profile and (not bool(self.height_profile_config.enabled) or skip_legacy_height_profile))
        ray_result = _collect_ray_cast_evidence(
            origin_cell=(int(origin_cell[0]), int(origin_cell[1])),
            endpoints=np.zeros((0, 2), dtype=np.int32) if skip_all_legacy_ray_cast else rows_cols,
            endpoint_rel_z=np.zeros((0,), dtype=np.float32) if skip_all_legacy_ray_cast else rel_z,
            endpoint_depth_m=np.zeros((0,), dtype=np.float32) if skip_all_legacy_ray_cast else in_bounds_depth,
            endpoint_is_obstacle=np.zeros((0,), dtype=bool) if skip_all_legacy_ray_cast else obstacle_mask,
            ray_can_clear=np.zeros((0,), dtype=bool) if skip_all_legacy_ray_cast else ray_clear_mask,
            map_width=map_width,
            depth_min_m=float(self.depth_min_m),
            depth_max_m=float(self.depth_max_m),
            camera_rel_z_m=float(camera_rel_z),
            z_min_m=float(self.vertical_profile_free_min_height_m),
            z_max_m=float(self.vertical_profile_free_max_height_m),
            band_ranges_m=self.vertical_profile.band_ranges_m,
        )
        if not bool(skip_legacy_vertical_profile):
            self._mark_vertical_profile_free_weighted_flat(
                ray_result.free_flat_by_band,
                ray_result.free_weight_by_band,
            )
        height_ray_result = None
        if bool(self.height_profile_config.enabled) and not bool(skip_legacy_height_profile):
            height_ray_result = _collect_ray_cast_evidence(
                origin_cell=(int(origin_cell[0]), int(origin_cell[1])),
                endpoints=rows_cols,
                endpoint_rel_z=rel_z,
                endpoint_depth_m=in_bounds_depth,
                endpoint_is_obstacle=obstacle_mask,
                ray_can_clear=ray_clear_mask,
                map_width=map_width,
                depth_min_m=float(self.depth_min_m),
                depth_max_m=float(self.depth_max_m),
                camera_rel_z_m=float(camera_rel_z),
                z_min_m=float(self.height_profile_config.z_min_m),
                z_max_m=float(self.height_profile_config.z_max_m),
                band_ranges_m=self.height_profile.bin_ranges_m,
            )
            self.height_profile.mark_free_ray_cells_by_bin(
                height_ray_result.free_flat_by_band,
                height_ray_result.free_weight_by_band,
            )
        if not bool(skip_legacy_roomseg_ray):
            self._mark_roomseg_ray_covered_weighted_flat(
                ray_result.roomseg_ray_covered_flat,
                ray_result.roomseg_ray_covered_weight,
            )
            self._mark_roomseg_terminal_wall_flat(
                ray_result.terminal_wall_flat,
                ray_result.terminal_wall_depth_m,
                ray_result.terminal_wall_rel_z_m,
            )
            self._refresh_roomseg_terminal_wall_splat(radius_cells=1)
        timings["ray_cast_ms"] = _elapsed_ms(stage_started_at)
        timings["legacy_roomseg_ray_skipped"] = 1.0 if bool(skip_legacy_roomseg_ray) else 0.0

        stage_started_at = time.perf_counter()
        free_unique = _unique_flat_array(ray_result.free_flat)
        if bool(self.voxel_grid_config.enabled):
            self._decay_voxel_navigation_endpoint_evidence(free_unique)
        free_protected_by_obstacle_endpoint = 0
        stale_obstacle_endpoint_cells_cleared = 0
        if free_unique.size:
            free_flat = np.asarray(free_unique, dtype=np.int64)
            endpoint_counts = self.depth_obstacle_endpoint_count.reshape(-1)
            protected = np.asarray(endpoint_counts[free_flat], dtype=np.uint16) > 0
            free_protected_by_obstacle_endpoint = int(np.count_nonzero(protected))
            if np.any(protected):
                protected_flat = free_flat[protected]
                decayed = np.maximum(
                    endpoint_counts[protected_flat].astype(np.int32) - int(self._free_ray_obstacle_endpoint_decay),
                    0,
                ).astype(np.uint16)
                endpoint_counts[protected_flat] = decayed
                newly_clear = protected_flat[decayed == 0]
                stale_obstacle_endpoint_cells_cleared = int(newly_clear.size)
                free_flat = np.concatenate([free_flat[~protected], newly_clear]) if newly_clear.size else free_flat[~protected]
            if free_flat.size:
                rows = free_flat // map_width
                cols = free_flat % map_width
                self.grid.free[rows, cols] = 1
                self.grid.occupied[rows, cols] = 0
                self.grid.observed[rows, cols] = 1
                self.depth_free_mask[rows, cols] = 1
        occupied_unique = _unique_flat_array(ray_result.occupied_flat)
        if occupied_unique.size:
            rows = occupied_unique // map_width
            cols = occupied_unique % map_width
            uint16_max = int(np.iinfo(np.uint16).max)
            flat = np.asarray(occupied_unique, dtype=np.int64)
            current = self.depth_obstacle_endpoint_count.reshape(-1).astype(np.uint32)
            current[flat] = np.minimum(current[flat] + int(self._obstacle_endpoint_evidence_increment), uint16_max)
            self.depth_obstacle_endpoint_count.reshape(-1)[:] = current.astype(np.uint16)
            self.grid.free[rows, cols] = 0
            self.grid.occupied[rows, cols] = 1
            self.grid.observed[rows, cols] = 1
        if bool(self.voxel_grid_config.enabled):
            force_full_nav_projection = bool(getattr(self, "force_full_navigation_projection_once", False))
            voxel_nav_projection = self.voxel_grid.project_navigation(
                **self.voxel_navigation_projection_config,
                nav_endpoint_count_xy=self.voxel_nav_occupied_endpoint_count,
                incremental=True,
                force_full=force_full_nav_projection,
                dirty_rc_flags=getattr(self.voxel_grid, "last_dirty_rc_flags", None),
                projection_step=int(self._update_step_index),
            )
            self.force_full_navigation_projection_once = False
            self.last_voxel_navigation_projection = voxel_nav_projection
            pre_outside_free = np.asarray(voxel_nav_projection.free, dtype=bool).copy()
            pre_outside_occupied = np.asarray(voxel_nav_projection.occupied, dtype=bool).copy()
            pre_outside_inflated_occupied = _dilate_binary(pre_outside_occupied, max(0, int(self._robot_footprint_radius_cells())))
            pre_outside_traversible = pre_outside_free & ~pre_outside_inflated_occupied
            pose_override = np.asarray(
                getattr(self, "current_pose_navigation_override_mask", np.zeros_like(pre_outside_free, dtype=bool)),
                dtype=bool,
            )
            if pose_override.shape == pre_outside_traversible.shape and np.any(pose_override):
                pre_outside_traversible = pre_outside_traversible | pose_override
            robot_grid = self.grid.world_to_grid(float(base_pose_world[0]), float(base_pose_world[1]))
            robot_grid = (int(robot_grid[0]), int(robot_grid[1]))
            outside_xy, outside_debug = self.voxel_grid.update_outside_from_navigation_free(
                pre_outside_free,
                pre_outside_traversible_xy=pre_outside_traversible,
                robot_grid=robot_grid,
                forced_initial_blind_zone_free_xy=self.last_voxel_initial_blind_zone_forced_xy,
                forced_current_footprint_free_xy=self.last_voxel_current_footprint_forced_xy,
                synthetic_free_provenance_xy=getattr(self, "synthetic_navigation_free_provenance_xy", None),
            )
            self.last_voxel_outside_debug = dict(outside_debug)
            if np.any(outside_xy):
                voxel_nav_projection.free[outside_xy] = False
                voxel_nav_projection.occupied[outside_xy] = False
                voxel_nav_projection.observed[outside_xy] = True
                voxel_nav_projection.unknown[outside_xy] = False
        if bool(self.voxel_grid_config.enabled) and bool(self.voxel_grid_drives_navigation) and voxel_nav_projection is not None:
            self.grid.free[:, :] = np.asarray(voxel_nav_projection.free, dtype=np.uint8)
            self.grid.occupied[:, :] = np.asarray(voxel_nav_projection.occupied, dtype=np.uint8)
            self.grid.observed[:, :] = np.asarray(voxel_nav_projection.observed, dtype=np.uint8)
            self.depth_free_mask[:, :] = np.asarray(voxel_nav_projection.free, dtype=np.uint8)
            self._apply_outside_boundary_to_grid()
        blind_2d_started_at = time.perf_counter()
        blind_2d_debug = self._apply_voxel_navigation_blind_zone(
            base_pose_world,
            write_to_voxel=False,
            write_to_grid=True,
        )
        if blind_2d_debug:
            merged = dict(self.last_voxel_blind_zone_debug)
            additive_keys = {
                "voxel_blind_zone_written_voxels",
            }
            for key, value in blind_2d_debug.items():
                if key in additive_keys and isinstance(value, (int, np.integer)) and not isinstance(value, bool) and key in merged:
                    merged[key] = int(merged.get(key, 0) or 0) + int(value)
                else:
                    merged[key] = value
            merged["voxel_blind_zone_write_to_voxel"] = bool(
                self.last_voxel_blind_zone_debug.get("voxel_blind_zone_write_to_voxel", False)
                or blind_2d_debug.get("voxel_blind_zone_write_to_voxel", False)
            )
            merged["voxel_blind_zone_write_to_grid"] = bool(
                self.last_voxel_blind_zone_debug.get("voxel_blind_zone_write_to_grid", False)
                or blind_2d_debug.get("voxel_blind_zone_write_to_grid", False)
            )
            self.last_voxel_blind_zone_debug = merged
        self._apply_outside_boundary_to_grid()
        timings["voxel_blind_zone_2d_ms"] = _elapsed_ms(blind_2d_started_at)
        timings["grid_write_ms"] = _elapsed_ms(stage_started_at)
        stage_started_at = time.perf_counter()
        self._mark_robot_footprint_free(base_pose_world)
        self._apply_outside_boundary_to_grid()
        timings["robot_footprint_ms"] = _elapsed_ms(stage_started_at)
        stage_started_at = time.perf_counter()
        self.last_debug_stats = self._build_debug_stats(
            depth_arr=depth_arr,
            valid_depth=in_bounds_depth,
            valid_pixels=in_bounds_pixels,
            rel_z=rel_z,
            obstacle_mask=obstacle_mask,
            free_mask=free_mask,
            occupied_endpoint_cells=int(occupied_unique.size),
            free_ray_cells=int(free_unique.size),
            free_protected_by_obstacle_endpoint=int(free_protected_by_obstacle_endpoint),
            stale_obstacle_endpoint_cells_cleared=int(stale_obstacle_endpoint_cells_cleared),
            ray_count=int(ray_result.nav_ray_count),
            skipped_height_rays=int(ray_result.nav_skipped_height_rays),
            vertical_profile_ray_count=int(ray_result.vertical_profile_ray_count),
            vertical_profile_skipped_height_rays=int(ray_result.vertical_profile_skipped_height_rays),
            height_profile_ray_count=0 if height_ray_result is None else int(height_ray_result.vertical_profile_ray_count),
            height_profile_skipped_height_rays=0 if height_ray_result is None else int(height_ray_result.vertical_profile_skipped_height_rays),
            ray_cast_backend=str(ray_result.backend),
            ray_unique_endpoint_cells=int(ray_result.unique_endpoint_cells),
            base_pose_world=base_pose_world,
            camera_pose_world=camera_pose_world,
            ray_origin_cell=origin_cell,
        )
        self.last_debug_stats.update(
            {
                "sampled_pixels": int(len(pixels)),
                "sampled_depth_hit_count": int(np.count_nonzero(hit_mask)),
                "sampled_depth_free_only_to_max_count": int(np.count_nonzero(free_only_mask)),
                "sampled_depth_bad_invalid_count": int(np.count_nonzero(bad_invalid_mask)),
                "sampled_depth_inf_count": int(np.count_nonzero(np.isposinf(sampled_depth))),
                "sampled_depth_ge_max_count": int(np.count_nonzero(finite_depth & (sampled_depth >= depth_max - depth_eps))),
                "sampled_depth_hit_endpoint_in_map_count": int(np.count_nonzero(hit_in_bounds)) if hit_in_bounds.size else 0,
                "sampled_depth_hit_endpoint_outside_map_count": int(np.count_nonzero(hit_mask))
                - (int(np.count_nonzero(hit_in_bounds)) if hit_in_bounds.size else 0),
                "voxel_depth_hit_ray_count": int(np.count_nonzero(ray_endpoint_is_hit)),
                "voxel_depth_free_only_ray_count": int(np.count_nonzero(~ray_endpoint_is_hit)),
                "voxel_depth_endpoint_is_hit_count": int(np.count_nonzero(ray_endpoint_is_hit)),
                "voxel_depth_endpoint_is_free_only_count": int(np.count_nonzero(~ray_endpoint_is_hit)),
                "voxel_depth_endpoint_outside_map_count": int(ray_endpoint_in_bounds.size - np.count_nonzero(ray_endpoint_in_bounds)),
                "depth_range_semantics": str(getattr(self.voxel_grid_config, "depth_range_semantics", "image_plane_z")),
                "voxel_sensor_depth_range_semantics": str(getattr(self.voxel_grid_config, "depth_range_semantics", "image_plane_z")),
            }
        )
        self.last_debug_stats.update(sensor_effective_range_debug)
        self.last_debug_stats.update(
            {key: value for key, value in dict(self.last_voxel_outside_debug).items() if not isinstance(value, np.ndarray)}
        )
        timings["debug_stats_ms"] = _elapsed_ms(stage_started_at)
        self._finish_timing_stats(timings, total_started_at, reason="ok")
        self._update_step_index += 1
        return self.grid

    def update_nearfield_topdown(
        self,
        depth: np.ndarray,
        intr: CameraIntrinsics,
        base_pose_world: Tuple[float, float, float, float],
        camera_pose_world: Tuple[float, float, float, float],
        radius_m: float = 0.75,
        ignore_radius_m: float = 0.22,
        depth_stride_px: int = 3,
        floor_tolerance_m: float = 0.12,
        obstacle_min_height_m: float | None = None,
        obstacle_max_height_m: float | None = None,
        splat_point_threshold: int | None = None,
        free_splat_point_threshold: int | None = None,
    ) -> dict:
        """Fuse a downward near-field depth camera into the 2D map.

        The overhead camera is only used to fill the ground blind spot around
        the robot. It is treated as a top-down depth sensor whose optical axis
        points toward -Z; pixels are projected into the robot-local XY plane.
        """
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[:, :, 0]
        if depth_arr.ndim != 2 or depth_arr.size == 0:
            stats = {"reason": "invalid_depth_shape", "depth_shape": list(depth_arr.shape)}
            self.last_nearfield_debug_stats = stats
            return stats

        stride = max(1, int(depth_stride_px))
        vs = np.arange(stride // 2, min(depth_arr.shape[0], intr.height), stride, dtype=np.int32)
        us = np.arange(stride // 2, min(depth_arr.shape[1], intr.width), stride, dtype=np.int32)
        if len(vs) == 0 or len(us) == 0:
            stats = {"reason": "no_sample_pixels", "depth_shape": list(depth_arr.shape)}
            self.last_nearfield_debug_stats = stats
            return stats

        uu, vv = np.meshgrid(us, vs)
        pixels = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1).astype(np.float32)
        sampled_depth = depth_arr[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]
        cam_z = float(camera_pose_world[2])
        floor_z = float(base_pose_world[2])
        camera_height = max(1e-3, cam_z - floor_z)
        max_depth = camera_height + max(0.20, float(floor_tolerance_m) * 2.0)
        valid = np.isfinite(sampled_depth) & (sampled_depth > 0.01) & (sampled_depth < max_depth)
        if int(valid.sum()) == 0:
            stats = {
                "reason": "no_valid_depth",
                "sampled_pixels": int(len(pixels)),
                "camera_height_m": float(camera_height),
                "depth_m_percentiles": _percentiles(sampled_depth),
            }
            self.last_nearfield_debug_stats = stats
            return stats

        pixels = pixels[valid]
        sampled_depth = sampled_depth[valid]
        x_right = (pixels[:, 0] - intr.cx) * sampled_depth / intr.fx
        y_down = (pixels[:, 1] - intr.cy) * sampled_depth / intr.fy
        local_forward = -y_down
        local_left = -x_right
        local_range = np.sqrt(local_forward * local_forward + local_left * local_left)
        radius = max(0.0, float(radius_m))
        ignore_radius = max(0.0, float(ignore_radius_m))
        range_mask = (local_range <= radius) & (local_range >= ignore_radius)
        if int(np.count_nonzero(range_mask)) == 0:
            stats = {
                "reason": "no_points_in_nearfield_radius",
                "valid_points": int(len(sampled_depth)),
                "camera_height_m": float(camera_height),
                "radius_m": float(radius),
                "ignore_radius_m": float(ignore_radius),
            }
            self.last_nearfield_debug_stats = stats
            return stats

        local_forward = local_forward[range_mask]
        local_left = local_left[range_mask]
        local_range = local_range[range_mask]
        sampled_depth = sampled_depth[range_mask]
        rel_height = camera_height - sampled_depth
        yaw = float(base_pose_world[3])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        world_x = float(base_pose_world[0]) + cos_yaw * local_forward - sin_yaw * local_left
        world_y = float(base_pose_world[1]) + sin_yaw * local_forward + cos_yaw * local_left
        rows_cols = np.asarray(
            [self.grid.world_to_grid(float(x), float(y)) for x, y in zip(world_x, world_y)],
            dtype=np.int32,
        )
        in_bounds = (
            (rows_cols[:, 0] >= 0)
            & (rows_cols[:, 0] < self.grid.map_info.height)
            & (rows_cols[:, 1] >= 0)
            & (rows_cols[:, 1] < self.grid.map_info.width)
        )
        if int(np.count_nonzero(in_bounds)) == 0:
            stats = {
                "reason": "no_points_in_map_bounds",
                "valid_points": int(len(sampled_depth)),
                "camera_height_m": float(camera_height),
            }
            self.last_nearfield_debug_stats = stats
            return stats

        rows_cols = rows_cols[in_bounds]
        rel_height = rel_height[in_bounds]
        local_range = local_range[in_bounds]
        floor_tol = max(0.01, float(floor_tolerance_m))
        obs_min = self.obstacle_min_height_m if obstacle_min_height_m is None else float(obstacle_min_height_m)
        obs_max = self.obstacle_max_height_m if obstacle_max_height_m is None else float(obstacle_max_height_m)
        free_mask = np.abs(rel_height) <= floor_tol
        obstacle_mask = (rel_height >= obs_min) & (rel_height <= obs_max)
        obstacle_splats = self._splat_cells(
            rows_cols[obstacle_mask],
            occupied=True,
            point_threshold=self.splat_point_threshold if splat_point_threshold is None else int(splat_point_threshold),
        )
        free_splats = self._splat_cells(
            rows_cols[free_mask],
            occupied=False,
            point_threshold=(
                self.free_splat_point_threshold
                if free_splat_point_threshold is None
                else int(free_splat_point_threshold)
            ),
        )
        self._mark_robot_footprint_free(base_pose_world)
        stats = {
            "reason": "ok",
            "depth_shape": [int(v) for v in depth_arr.shape],
            "camera_pose_world": [float(v) for v in camera_pose_world],
            "camera_height_m": float(camera_height),
            "radius_m": float(radius),
            "ignore_radius_m": float(ignore_radius),
            "valid_points": int(len(rel_height)),
            "free_band_points": int(np.count_nonzero(free_mask)),
            "obstacle_band_points": int(np.count_nonzero(obstacle_mask)),
            "free_splat_cells": int(free_splats),
            "obstacle_splat_cells": int(obstacle_splats),
            "depth_m_percentiles": _percentiles(sampled_depth),
            "rel_height_m_percentiles": _percentiles(rel_height),
            "local_range_m_percentiles": _percentiles(local_range),
            "height_filters_m": {
                "floor_tolerance": float(floor_tol),
                "obstacle_min": float(obs_min),
                "obstacle_max": float(obs_max),
            },
        }
        self.last_nearfield_debug_stats = stats
        return stats

    def update_static_nearfield(
        self,
        static_occupancy: np.ndarray,
        static_navigable: np.ndarray,
        static_map_info: MapInfo,
        base_pose_world: Tuple[float, float, float, float],
        radius_m: float = 1.0,
        static_openings: np.ndarray | None = None,
    ) -> dict:
        """Use the preprocessed map as a local blind-spot fill around the robot."""
        occupancy = np.asarray(static_occupancy).astype(bool)
        navigable = np.asarray(static_navigable).astype(bool)
        openings = None if static_openings is None else np.asarray(static_openings).astype(bool)
        if occupancy.shape != navigable.shape:
            stats = {
                "reason": "shape_mismatch",
                "occupancy_shape": [int(v) for v in occupancy.shape],
                "navigable_shape": [int(v) for v in navigable.shape],
            }
            self.last_static_nearfield_debug_stats = stats
            return stats
        if openings is not None and openings.shape != navigable.shape:
            stats = {
                "reason": "opening_shape_mismatch",
                "opening_shape": [int(v) for v in openings.shape],
                "navigable_shape": [int(v) for v in navigable.shape],
            }
            self.last_static_nearfield_debug_stats = stats
            return stats
        if occupancy.shape != (int(static_map_info.height), int(static_map_info.width)):
            stats = {
                "reason": "map_info_shape_mismatch",
                "map_shape": [int(v) for v in occupancy.shape],
                "map_info_shape": [int(static_map_info.height), int(static_map_info.width)],
            }
            self.last_static_nearfield_debug_stats = stats
            return stats

        radius = max(0.0, float(radius_m))
        radius_cells = int(math.ceil(radius / self.grid.map_info.resolution_m))
        center = self.grid.world_to_grid(float(base_pose_world[0]), float(base_pose_world[1]))
        free_cells = 0
        occupied_cells = 0
        blocked_clearance_cells = 0
        roomseg_static_structural_occupied_cells = 0
        skipped_outside_dynamic = 0
        skipped_outside_static = 0
        sampled_cells = 0
        opening_cells = 0
        for dr, dc in _disk_offsets(radius_cells):
            row, col = int(center[0] + dr), int(center[1] + dc)
            if not is_inside_grid(row, col, self.grid.map_info):
                skipped_outside_dynamic += 1
                continue
            wx, wy = grid_to_world_xy(row, col, self.grid.map_info)
            static_row, static_col = world_xy_to_grid(wx, wy, static_map_info)
            if not is_inside_grid(static_row, static_col, static_map_info):
                skipped_outside_static += 1
                continue
            sampled_cells += 1
            self.static_nearfield_mask[row, col] = 1
            is_opening = openings is not None and bool(openings[static_row, static_col])
            if bool(navigable[static_row, static_col]) or is_opening:
                self.grid.free[row, col] = 1
                self.grid.occupied[row, col] = 0
                self.grid.observed[row, col] = 1
                self.roomseg_static_structural_occupied[row, col] = 0
                free_cells += 1
                if is_opening:
                    opening_cells += 1
            else:
                self.grid.free[row, col] = 0
                self.grid.occupied[row, col] = 0
                if bool(occupancy[static_row, static_col]):
                    occupied_cells += 1
                    self.roomseg_static_structural_occupied[row, col] = 1
                    roomseg_static_structural_occupied_cells += 1
                else:
                    blocked_clearance_cells += 1
                    self.roomseg_static_structural_occupied[row, col] = 0
                self.grid.observed[row, col] = 1

        stats = {
            "reason": "ok",
            "source": "preprocessed_static_map",
            "radius_m": float(radius),
            "radius_cells": int(radius_cells),
            "sampled_cells": int(sampled_cells),
            "free_cells": int(free_cells),
            "occupied_cells": int(occupied_cells),
            "blocked_clearance_cells": int(blocked_clearance_cells),
            "opening_cells": int(opening_cells),
            "roomseg_static_structural_occupied_cells": int(roomseg_static_structural_occupied_cells),
            "skipped_outside_dynamic": int(skipped_outside_dynamic),
            "skipped_outside_static": int(skipped_outside_static),
            "base_pose_world": [float(v) for v in base_pose_world],
        }
        self.last_static_nearfield_debug_stats = stats
        return stats

    def traversible(self, unknown_is_obstacle: bool = True) -> np.ndarray:
        occupied = self.inflated_occupied()
        self.last_inflated_occupied = np.asarray(occupied, dtype=bool)
        if unknown_is_obstacle:
            free = self.grid.free.astype(bool)
        else:
            free = np.ones_like(occupied, dtype=bool)
        if np.any(self.current_pose_navigation_override_mask):
            override = np.asarray(self.current_pose_navigation_override_mask, dtype=bool)
            occupied = np.asarray(occupied, dtype=bool) & ~override
            free = np.asarray(free, dtype=bool) | override
        return free & ~occupied

    def inflated_occupied(self) -> np.ndarray:
        radius_cells = self._robot_footprint_radius_cells()
        return _dilate_binary(self.grid.occupied.astype(bool), radius_cells)

    def _apply_voxel_navigation_blind_zone(
        self,
        base_pose_world: Tuple[float, float, float, float],
        *,
        write_to_voxel: bool,
        write_to_grid: bool,
    ) -> dict[str, object]:
        cfg = self.voxel_navigation_blind_zone_config
        out: dict[str, object] = {
            "voxel_blind_zone_enabled": bool(cfg.get("enabled", True)),
            "voxel_blind_zone_step_index": int(self._update_step_index),
            "voxel_initial_blind_zone_free_cells": 0,
            "voxel_current_footprint_forced_free_cells": 0,
            "voxel_blind_zone_written_voxels": 0,
            "voxel_blind_zone_grid_cells": 0,
            "voxel_blind_zone_write_to_voxel": bool(write_to_voxel),
            "voxel_blind_zone_write_to_grid": bool(write_to_grid),
            "voxel_blind_zone_free_z_min_m": float(cfg.get("free_z_min_m", 0.10)),
            "voxel_blind_zone_free_z_max_m": float(cfg.get("free_z_max_m", 0.90)),
            "voxel_blind_zone_active_this_frame": False,
            "voxel_blind_zone_changed_voxels": 0,
            "voxel_blind_zone_skipped_occupied_cells": 0,
            "current_pose_navigation_override_cells": 0,
            "current_pose_dynamic_occupied_cleared_cells": 0,
            "current_pose_static_wall_preserved_cells": 0,
        }
        if not bool(cfg.get("enabled", True)):
            return out
        if self._initial_blind_zone_center_world_xy is None:
            self._initial_blind_zone_center_world_xy = (float(base_pose_world[0]), float(base_pose_world[1]))

        requests: list[tuple[str, tuple[float, float], float]] = []
        if (
            bool(cfg.get("force_initial_blind_zone_free", True))
            and int(self._update_step_index) < max(0, int(cfg.get("initial_blind_zone_steps", 20)))
            and self._initial_blind_zone_center_world_xy is not None
        ):
            requests.append(
                (
                    "initial",
                    self._initial_blind_zone_center_world_xy,
                    max(0.0, float(cfg.get("initial_blind_zone_radius_m", 0.60))),
                )
            )
        if bool(cfg.get("force_current_footprint_free", False)):
            requests.append(
                (
                    "current",
                    (float(base_pose_world[0]), float(base_pose_world[1])),
                    max(
                        0.0,
                        float(cfg.get("current_footprint_radius_m", 0.25))
                        + float(cfg.get("current_footprint_extra_margin_m", 0.0)),
                    ),
                )
            )

        total_grid_cells = 0
        total_voxels = 0
        changed_chunks: list[np.ndarray] = []
        initial_cells = 0
        current_cells = 0
        skipped_occupied_cells = 0
        current_dynamic_occupied_cleared = 0
        current_static_wall_preserved = 0
        for reason, center_xy, radius_m in requests:
            rows, cols = self._disk_cells_for_world_xy(center_xy, radius_m)
            if rows.size == 0:
                continue
            z_idx = np.zeros(0, dtype=np.int32)
            if write_to_voxel and bool(cfg.get("write_to_voxel_grid", True)) and bool(self.voxel_grid_config.enabled):
                z_idx = self.voxel_grid.active_z_indices(
                    z_min_m=float(cfg.get("free_z_min_m", 0.10)),
                    z_max_m=float(cfg.get("free_z_max_m", 0.35)),
                )
                if not bool(cfg.get("free_all_bins_in_range", True)) and z_idx.size:
                    z_idx = z_idx[:1]
            if reason == "current" and bool(cfg.get("current_footprint_clear_dynamic_occupied", True)):
                keep = np.ones(int(rows.size), dtype=bool)
                static_blocked = np.zeros(int(rows.size), dtype=bool)
                if bool(cfg.get("current_footprint_preserve_static_structural", True)) or bool(
                    cfg.get("current_footprint_preserve_roomseg_static_wall", True)
                ):
                    static_blocked |= self.roomseg_static_structural_occupied[rows, cols].astype(bool)
                if (
                    z_idx.size
                    and bool(cfg.get("current_footprint_preserve_voxel_hard_static_wall", True))
                    and not bool(cfg.get("current_footprint_ignore_dynamic_voxel_occupied_columns", True))
                    and bool(self.voxel_grid_config.enabled)
                ):
                    z = np.asarray(z_idx, dtype=np.int64).reshape(-1)
                    voxel_occupied = np.any(self.voxel_grid.state[z[:, None], rows[None, :], cols[None, :]] == int(VOXEL_OCCUPIED), axis=0)
                    static_blocked |= voxel_occupied
                keep &= ~static_blocked
                current_static_wall_preserved += int(np.count_nonzero(static_blocked))
            else:
                keep = self._free_override_keep_mask(rows, cols, z_idx=z_idx)
            skipped_occupied_cells += int(rows.size - int(np.count_nonzero(keep)))
            if not np.any(keep):
                continue
            rows = rows[keep]
            cols = cols[keep]
            self.last_voxel_blind_zone_forced_xy[rows, cols] = True
            if reason == "initial":
                self.last_voxel_initial_blind_zone_forced_xy[rows, cols] = True
            elif reason == "current":
                self.last_voxel_current_footprint_forced_xy[rows, cols] = True
            if write_to_grid and bool(cfg.get("write_to_grid", True)):
                if reason == "current":
                    occupied_before = self.grid.occupied[rows, cols].astype(bool)
                    current_dynamic_occupied_cleared += int(np.count_nonzero(occupied_before))
                self.grid.free[rows, cols] = 1
                self.grid.occupied[rows, cols] = 0
                self.grid.observed[rows, cols] = 1
                self.depth_free_mask[rows, cols] = 1
                if reason == "current":
                    self.current_pose_navigation_override_mask[rows, cols] = True
            total_grid_cells += int(rows.size)
            if reason == "initial":
                initial_cells += int(rows.size)
            else:
                current_cells += int(rows.size)
            if write_to_voxel and bool(cfg.get("write_to_voxel_grid", True)) and bool(self.voxel_grid_config.enabled):
                if z_idx.size:
                    voxels = np.empty((int(z_idx.size) * int(rows.size), 3), dtype=np.int32)
                    voxels[:, 0] = np.repeat(z_idx.astype(np.int32), int(rows.size))
                    voxels[:, 1] = np.tile(rows.astype(np.int32), int(z_idx.size))
                    voxels[:, 2] = np.tile(cols.astype(np.int32), int(z_idx.size))
                    count, changed = self.voxel_grid.force_free_voxels(voxels)
                    total_voxels += int(count)
                    if changed.size:
                        changed_chunks.append(np.asarray(changed, dtype=np.int64))
        if changed_chunks:
            changed_all = np.unique(np.concatenate(changed_chunks).astype(np.int64, copy=False))
            self.voxel_grid.refresh_state_indices(changed_all)
            hw = int(self.grid.map_info.height * self.grid.map_info.width)
            if getattr(self.voxel_grid, "last_dirty_rc_flags", None) is None or int(np.asarray(self.voxel_grid.last_dirty_rc_flags).size) != hw:
                self.voxel_grid.last_dirty_rc_flags = np.zeros(hw, dtype=np.uint8)
            self.voxel_grid.last_dirty_rc_flags[np.asarray(changed_all % max(1, hw), dtype=np.int64)] = 1
        else:
            changed_all = np.zeros(0, dtype=np.int64)
        out.update(
            {
                "voxel_initial_blind_zone_free_cells": int(initial_cells),
                "voxel_current_footprint_forced_free_cells": int(current_cells),
                "voxel_blind_zone_grid_cells": int(total_grid_cells),
                "voxel_blind_zone_written_voxels": int(total_voxels),
                "voxel_blind_zone_changed_voxels": int(changed_all.size),
                "voxel_blind_zone_skipped_occupied_cells": int(skipped_occupied_cells),
                "voxel_blind_zone_active_this_frame": bool(total_grid_cells > 0 or total_voxels > 0),
                "voxel_initial_blind_zone_radius_m": float(cfg.get("initial_blind_zone_radius_m", 0.80)),
                "voxel_current_footprint_radius_m": float(cfg.get("current_footprint_radius_m", 0.35)),
                "voxel_current_footprint_extra_margin_m": float(cfg.get("current_footprint_extra_margin_m", 0.0)),
                "voxel_initial_blind_zone_steps": int(cfg.get("initial_blind_zone_steps", 60)),
                "current_pose_navigation_override_cells": int(np.count_nonzero(self.current_pose_navigation_override_mask)),
                "current_pose_dynamic_occupied_cleared_cells": int(current_dynamic_occupied_cleared),
                "current_pose_static_wall_preserved_cells": int(current_static_wall_preserved),
            }
        )
        self.last_current_footprint_override_debug = {
            "current_pose_navigation_override_cells": int(np.count_nonzero(self.current_pose_navigation_override_mask)),
            "current_pose_dynamic_occupied_cleared_cells": int(current_dynamic_occupied_cleared),
            "current_pose_static_wall_preserved_cells": int(current_static_wall_preserved),
        }
        return out

    def _free_override_keep_mask(self, rows: np.ndarray, cols: np.ndarray, *, z_idx: np.ndarray | None = None) -> np.ndarray:
        keep = np.ones(int(rows.size), dtype=bool)
        cfg = self.voxel_navigation_blind_zone_config
        if bool(cfg.get("preserve_occupied_cells", True)):
            keep &= self.grid.occupied[rows, cols].astype(bool) == 0
        if bool(cfg.get("preserve_static_structural_cells", True)):
            keep &= self.roomseg_static_structural_occupied[rows, cols].astype(bool) == 0
        if (
            z_idx is not None
            and int(np.asarray(z_idx).size) > 0
            and bool(cfg.get("preserve_voxel_occupied_columns", True))
            and bool(self.voxel_grid_config.enabled)
        ):
            z = np.asarray(z_idx, dtype=np.int64).reshape(-1)
            voxel_occupied = np.any(self.voxel_grid.state[z[:, None], rows[None, :], cols[None, :]] == int(VOXEL_OCCUPIED), axis=0)
            keep &= ~voxel_occupied
        return keep

    def _apply_outside_boundary_to_grid(self) -> dict[str, object]:
        outside = np.asarray(getattr(self.voxel_grid, "outside_xy", np.zeros_like(self.grid.free, dtype=bool)), dtype=bool)
        if outside.shape != self.grid.free.shape:
            outside = np.zeros_like(self.grid.free, dtype=bool)
        enabled = bool(getattr(self.voxel_grid_config, "outside_boundary_enabled", True)) and bool(
            getattr(self.voxel_grid_config, "outside_use_as_navigation_boundary", True)
        )
        if not enabled or not np.any(outside):
            debug = {
                "voxel_outside_navigation_boundary_applied": bool(enabled),
                "voxel_outside_navigation_boundary_cells": 0,
                "voxel_outside_grid_free_cleared_cells": 0,
                "voxel_outside_grid_occupied_cleared_cells": 0,
            }
            self.last_voxel_outside_debug = {**dict(self.last_voxel_outside_debug), **debug}
            return debug
        free_before = np.asarray(self.grid.free, dtype=bool)
        occupied_before = np.asarray(self.grid.occupied, dtype=bool)
        self.grid.free[outside] = 0
        self.grid.occupied[outside] = 0
        self.grid.observed[outside] = 1
        self.depth_free_mask[outside] = 0
        projection = getattr(self, "last_voxel_navigation_projection", None)
        if projection is not None:
            projection.free[outside] = False
            projection.occupied[outside] = False
            projection.observed[outside] = True
            projection.unknown[outside] = False
        debug = {
            "voxel_outside_navigation_boundary_applied": True,
            "voxel_outside_navigation_boundary_cells": int(np.count_nonzero(outside)),
            "voxel_outside_grid_free_cleared_cells": int(np.count_nonzero(free_before & outside)),
            "voxel_outside_grid_occupied_cleared_cells": int(np.count_nonzero(occupied_before & outside)),
        }
        self.last_voxel_outside_debug = {**dict(self.last_voxel_outside_debug), **debug}
        return debug

    def _disk_cells_for_world_xy(self, center_world_xy: tuple[float, float], radius_m: float) -> tuple[np.ndarray, np.ndarray]:
        center = self.grid.world_to_grid(float(center_world_xy[0]), float(center_world_xy[1]))
        radius_cells = max(0, int(math.ceil(max(0.0, float(radius_m)) / max(float(self.grid.map_info.resolution_m), 1e-6))))
        rows: list[int] = []
        cols: list[int] = []
        for dr, dc in _disk_offsets(radius_cells):
            row, col = int(center[0] + dr), int(center[1] + dc)
            if 0 <= row < self.grid.map_info.height and 0 <= col < self.grid.map_info.width:
                rows.append(row)
                cols.append(col)
        if not rows:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)

    def _mark_robot_footprint_free(
        self,
        base_pose_world: Tuple[float, float, float, float],
        *,
        clear_dynamic_occupied: bool = True,
    ) -> None:
        cfg = self.voxel_navigation_blind_zone_config
        clear_dynamic = bool(clear_dynamic_occupied) and bool(cfg.get("current_footprint_clear_dynamic_occupied", True))
        preserve_static = bool(cfg.get("current_footprint_preserve_static_structural", True)) or bool(
            cfg.get("current_footprint_preserve_roomseg_static_wall", True)
        )
        radius_cells = max(1, self._robot_footprint_radius_cells())
        center = self.grid.world_to_grid(float(base_pose_world[0]), float(base_pose_world[1]))
        profile_cells: List[Tuple[int, int]] = []
        dynamic_cleared = 0
        static_preserved = 0
        for dr, dc in _disk_offsets(radius_cells):
            row, col = int(center[0] + dr), int(center[1] + dc)
            if not (0 <= row < self.grid.map_info.height and 0 <= col < self.grid.map_info.width):
                continue
            if preserve_static and bool(self.roomseg_static_structural_occupied[row, col]):
                static_preserved += 1
                continue
            if clear_dynamic and bool(self.grid.occupied[row, col]):
                dynamic_cleared += 1
            marked = self._set_free_cell(
                row,
                col,
                mark_mask=self.depth_free_mask,
                clear_occupied=clear_dynamic,
                preserve_occupied=False,
            )
            if marked and 0 <= row < self.grid.map_info.height and 0 <= col < self.grid.map_info.width:
                self.current_pose_navigation_override_mask[row, col] = True
                self.last_voxel_blind_zone_forced_xy[row, col] = True
                self.last_voxel_current_footprint_forced_xy[row, col] = True
                profile_cells.append((row, col))
        if profile_cells:
            free_z = min(
                max(float(self.vertical_profile_free_min_height_m), 0.80),
                max(float(self.vertical_profile_free_min_height_m), float(self.vertical_profile_free_max_height_m) - 1e-3),
            )
            self.vertical_profile.mark_free_ray_cells(profile_cells, rel_z_m=free_z)
        previous = dict(self.last_current_footprint_override_debug)
        previous.update(
            {
                "current_pose_navigation_override_cells": int(np.count_nonzero(self.current_pose_navigation_override_mask)),
                "current_pose_dynamic_occupied_cleared_cells": int(previous.get("current_pose_dynamic_occupied_cleared_cells", 0) or 0)
                + int(dynamic_cleared),
                "current_pose_static_wall_preserved_cells": int(previous.get("current_pose_static_wall_preserved_cells", 0) or 0)
                + int(static_preserved),
            }
        )
        self.last_current_footprint_override_debug = previous

    def _set_free_cell(
        self,
        row: int,
        col: int,
        mark_mask: np.ndarray | None = None,
        *,
        clear_occupied: bool = True,
        preserve_occupied: bool = False,
    ) -> bool:
        if 0 <= int(row) < self.grid.map_info.height and 0 <= int(col) < self.grid.map_info.width:
            rr, cc = int(row), int(col)
            if preserve_occupied and (
                bool(self.grid.occupied[rr, cc]) or bool(self.roomseg_static_structural_occupied[rr, cc])
            ):
                return False
            self.grid.free[rr, cc] = 1
            if bool(clear_occupied):
                self.grid.occupied[rr, cc] = 0
            self.grid.observed[rr, cc] = 1
            if mark_mask is not None:
                mark_mask[rr, cc] = 1
            return True
        return False

    def _set_occupied_cell(self, row: int, col: int) -> None:
        if 0 <= int(row) < self.grid.map_info.height and 0 <= int(col) < self.grid.map_info.width:
            rr, cc = int(row), int(col)
            self.grid.free[rr, cc] = 0
            self.grid.occupied[rr, cc] = 1
            self.grid.observed[rr, cc] = 1

    def _finish_timing_stats(self, timings: dict[str, float], total_started_at: float, *, reason: str) -> None:
        out = {str(key): float(value) for key, value in timings.items()}
        out["update_total_ms"] = _elapsed_ms(total_started_at)
        out["reason"] = str(reason)
        out["legacy_vertical_profile_skipped_by_voxel_backend"] = bool(getattr(self, "_last_legacy_vertical_profile_skipped_by_voxel_backend", False))
        out["legacy_height_profile_skipped_by_voxel_backend"] = bool(getattr(self, "_last_legacy_height_profile_skipped_by_voxel_backend", False))
        out["legacy_roomseg_ray_skipped_by_voxel_backend"] = bool(getattr(self, "_last_legacy_roomseg_ray_skipped_by_voxel_backend", False))
        self.last_timing_stats = out

    def _reset_roomseg_ray_evidence(self) -> None:
        shape = self.grid.free.shape
        self.roomseg_ray_covered_count = np.zeros(shape, dtype=np.uint16)
        self.roomseg_terminal_wall_count = np.zeros(shape, dtype=np.uint16)
        self.roomseg_terminal_wall_height_min = np.full(shape, np.inf, dtype=np.float32)
        self.roomseg_terminal_wall_height_max = np.full(shape, -np.inf, dtype=np.float32)
        self.roomseg_terminal_wall_depth_min = np.full(shape, np.inf, dtype=np.float32)
        self.roomseg_terminal_wall_splat = np.zeros(shape, dtype=np.uint8)

    def roomseg_ray_evidence(self) -> dict[str, np.ndarray]:
        return {
            "ray_covered_count": np.asarray(self.roomseg_ray_covered_count, dtype=np.uint16),
            "terminal_wall_count": np.asarray(self.roomseg_terminal_wall_count, dtype=np.uint16),
            "terminal_wall_height_min": np.asarray(self.roomseg_terminal_wall_height_min, dtype=np.float32),
            "terminal_wall_height_max": np.asarray(self.roomseg_terminal_wall_height_max, dtype=np.float32),
            "terminal_wall_depth_min": np.asarray(self.roomseg_terminal_wall_depth_min, dtype=np.float32),
            "terminal_wall_splat": np.asarray(self.roomseg_terminal_wall_splat, dtype=np.uint8),
        }

    def roomseg_height_profile_evidence(self) -> dict[str, np.ndarray | float]:
        return {
            "free_ray_count": np.asarray(self.height_profile.free_ray_count, dtype=np.uint16).copy(),
            "occupied_count": np.asarray(self.height_profile.occupied_count, dtype=np.uint16).copy(),
            "observed_count": np.asarray(self.height_profile.observed_count, dtype=np.uint16).copy(),
            "z_min_m": float(self.height_profile.z_min_m),
            "z_max_m": float(self.height_profile.z_max_m),
            "z_bin_size_m": float(self.height_profile.z_bin_size_m),
            "active_z_min_m": float(self.height_profile_config.active_z_min_m),
            "active_z_max_m": float(self.height_profile_config.active_z_max_m or self.height_profile_config.active_z_max_cap_m or self.height_profile_config.z_max_m),
            "ceiling_height_estimate_m": (
                np.nan
                if self.last_ceiling_height_estimate.height_m is None
                else float(self.last_ceiling_height_estimate.height_m)
            ),
        }

    def roomseg_voxel_evidence(self) -> dict[str, np.ndarray | float | None]:
        outside_debug = dict(getattr(self.voxel_grid, "last_outside_debug", {}) or {})
        return {
            "state": np.asarray(self.voxel_grid.state, dtype=np.uint8).copy(),
            "log_odds": np.asarray(self.voxel_grid.log_odds, dtype=np.int16).copy(),
            "sensor_range_count": np.asarray(self.voxel_grid.sensor_range_count, dtype=np.uint8).copy(),
            "floor_frustum_seen_count_xy": np.asarray(self.voxel_grid.floor_frustum_seen_count_xy, dtype=np.uint16).copy(),
            "voxel_floor_frustum_seen_count_xy": np.asarray(self.voxel_grid.floor_frustum_seen_count_xy, dtype=np.uint16).copy(),
            "voxel_outside_score_xy": np.asarray(self.voxel_grid.outside_score_xy, dtype=np.uint8).copy(),
            "voxel_outside_xy": np.asarray(self.voxel_grid.outside_xy, dtype=bool).copy(),
            "voxel_outside_candidate_xy": np.asarray(
                outside_debug.get("voxel_outside_candidate_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_outside_candidate_base_xy": np.asarray(
                outside_debug.get("voxel_outside_candidate_base_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_outside_disconnected_candidate_xy": np.asarray(
                outside_debug.get("voxel_outside_disconnected_candidate_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_outside_pre_nav_free_xy": np.asarray(
                outside_debug.get("voxel_outside_pre_nav_free_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_outside_pre_traversible_xy": np.asarray(
                outside_debug.get("voxel_outside_pre_traversible_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_outside_robot_component_xy": np.asarray(
                outside_debug.get("voxel_outside_robot_component_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_outside_connected_to_robot_rejected_xy": np.asarray(
                outside_debug.get("voxel_outside_connected_to_robot_rejected_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_outside_full_height_no_occupied_xy": np.asarray(
                outside_debug.get("voxel_outside_full_height_no_occupied_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_outside_floor_frustum_seen_xy": np.asarray(
                outside_debug.get(
                    "voxel_outside_floor_frustum_seen_xy",
                    np.asarray(self.voxel_grid.floor_frustum_seen_count_xy, dtype=np.uint16)
                    >= max(1, int(getattr(self.voxel_grid.config, "outside_floor_frustum_seen_count_threshold", 1))),
                ),
                dtype=bool,
            ).copy(),
            "voxel_outside_holdout_floor_unseen_xy": np.asarray(
                outside_debug.get("voxel_outside_holdout_floor_unseen_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_outside_has_occupied_xy": np.asarray(
                outside_debug.get("voxel_outside_has_occupied_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_outside_excluded_synthetic_free_xy": np.asarray(
                outside_debug.get("voxel_outside_excluded_synthetic_free_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_outside_synthetic_free_provenance_xy": np.asarray(
                outside_debug.get("voxel_outside_synthetic_free_provenance_xy", np.zeros(self.grid.free.shape, dtype=bool)),
                dtype=bool,
            ).copy(),
            "voxel_nav_occupied_endpoint_count_xy": np.asarray(self.voxel_nav_occupied_endpoint_count, dtype=np.uint16).copy(),
            "voxel_nav_free_ray_count_xy": np.asarray(self.voxel_nav_free_ray_count, dtype=np.uint16).copy(),
            "z_min_m": float(self.voxel_grid.z_min_m),
            "z_max_m": float(self.voxel_grid.z_max_m),
            "z_resolution_m": float(self.voxel_grid.z_resolution_m),
            "active_z_min_m": float(self.voxel_grid.active_z_min_m),
            "active_z_max_m": None if self.voxel_grid.active_z_max_m is None else float(self.voxel_grid.active_z_max_m),
            "ceiling_height_estimate_m": (
                None
                if self.last_ceiling_height_estimate.height_m is None
                else float(self.last_ceiling_height_estimate.height_m)
            ),
            "ceiling_estimate_status": str(getattr(self.voxel_grid, "ceiling_estimate_status", "snapshot")),
        }

    def navigation_debug_layers(self, *, include_arrays: bool = False) -> dict[str, object]:
        debug: dict[str, object] = {}
        projection_debug = dict(getattr(self.voxel_grid, "last_navigation_debug", {}) or {})
        outside_debug = dict(getattr(self.voxel_grid, "last_outside_debug", {}) or {})
        if bool(include_arrays):
            for key in (
                "voxel_nav_occupied_from_voxel_xy",
                "voxel_nav_occupied_from_endpoint_xy",
                "voxel_nav_occupied_raw_xy",
                "voxel_nav_occupied_closed_xy",
                "voxel_nav_free_raw_xy",
                "voxel_nav_free_suppressed_by_occupied_xy",
                "voxel_nav_final_free_xy",
                "voxel_nav_final_occupied_xy",
                "voxel_nav_final_unknown_xy",
            ):
                value = projection_debug.get(key)
                if isinstance(value, np.ndarray):
                    debug[key] = np.asarray(value).copy()
            projection = getattr(self, "last_voxel_navigation_projection", None)
            if projection is not None:
                if "voxel_nav_final_free_xy" not in debug:
                    debug["voxel_nav_final_free_xy"] = np.asarray(projection.free, dtype=bool).copy()
                if "voxel_nav_final_occupied_xy" not in debug:
                    debug["voxel_nav_final_occupied_xy"] = np.asarray(projection.occupied, dtype=bool).copy()
                if "voxel_nav_final_unknown_xy" not in debug:
                    debug["voxel_nav_final_unknown_xy"] = np.asarray(projection.unknown, dtype=bool).copy()
            cache = getattr(self.voxel_grid, "_navigation_projection_cache", None)
            if cache is not None and bool(getattr(cache, "initialized", False)):
                if "voxel_nav_final_free_xy" not in debug:
                    debug["voxel_nav_final_free_xy"] = np.asarray(cache.free, dtype=bool).copy()
                if "voxel_nav_final_occupied_xy" not in debug:
                    debug["voxel_nav_final_occupied_xy"] = np.asarray(cache.occupied, dtype=bool).copy()
                if "voxel_nav_final_unknown_xy" not in debug:
                    debug["voxel_nav_final_unknown_xy"] = np.asarray(cache.unknown, dtype=bool).copy()
            for key in (
                "voxel_outside_xy",
                "voxel_outside_candidate_xy",
                "voxel_outside_candidate_base_xy",
                "voxel_outside_disconnected_candidate_xy",
                "voxel_outside_pre_nav_free_xy",
                "voxel_outside_pre_traversible_xy",
                "voxel_outside_robot_component_xy",
                "voxel_outside_connected_to_robot_rejected_xy",
                "voxel_outside_full_height_no_occupied_xy",
                "voxel_outside_floor_frustum_seen_xy",
                "voxel_outside_holdout_floor_unseen_xy",
                "voxel_outside_has_occupied_xy",
                "voxel_outside_excluded_synthetic_free_xy",
                "voxel_outside_synthetic_free_provenance_xy",
                "voxel_outside_score_xy",
            ):
                value = outside_debug.get(key)
                if isinstance(value, np.ndarray):
                    debug[key] = np.asarray(value).copy()
            if "voxel_outside_xy" not in debug:
                debug["voxel_outside_xy"] = np.asarray(self.voxel_grid.outside_xy, dtype=bool).copy()
            if "voxel_outside_score_xy" not in debug:
                debug["voxel_outside_score_xy"] = np.asarray(self.voxel_grid.outside_score_xy, dtype=np.uint8).copy()
            debug["voxel_floor_frustum_seen_count_xy"] = np.asarray(self.voxel_grid.floor_frustum_seen_count_xy, dtype=np.uint16).copy()
        for key, value in projection_debug.items():
            if not isinstance(value, np.ndarray):
                debug[key] = value
        for key, value in outside_debug.items():
            if not isinstance(value, np.ndarray):
                debug[key] = value
        debug.update(dict(self.voxel_nav_endpoint_decay_debug))
        debug.update(dict(self.last_current_footprint_override_debug))
        if bool(include_arrays):
            debug["voxel_nav_occupied_endpoint_count_xy"] = np.asarray(self.voxel_nav_occupied_endpoint_count, dtype=np.uint16).copy()
            debug["voxel_nav_free_ray_count_xy"] = np.asarray(self.voxel_nav_free_ray_count, dtype=np.uint16).copy()
            debug["current_pose_navigation_override_mask"] = np.asarray(self.current_pose_navigation_override_mask, dtype=bool).copy()
        debug["current_pose_navigation_override_cells"] = int(np.count_nonzero(self.current_pose_navigation_override_mask))
        return debug

    def _mark_roomseg_ray_covered_flat(self, flat_values: List[int]) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        flat = _valid_flat_array(flat_values, total_cells)
        if flat.size == 0:
            return
        uint16_max = int(np.iinfo(np.uint16).max)
        counts = np.bincount(flat, minlength=total_cells).reshape(h, w).astype(np.uint32)
        updated = np.asarray(self.roomseg_ray_covered_count, dtype=np.uint32) + counts
        self.roomseg_ray_covered_count[:, :] = np.minimum(updated, uint16_max).astype(np.uint16)

    def _mark_roomseg_ray_covered_weighted_flat(self, flat_values: np.ndarray, weights: np.ndarray) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        flat, weight = _valid_flat_weight_arrays(flat_values, weights, total_cells)
        if flat.size == 0:
            return
        uint16_max = int(np.iinfo(np.uint16).max)
        counts = np.bincount(flat, weights=weight, minlength=total_cells).reshape(h, w).astype(np.uint32)
        updated = np.asarray(self.roomseg_ray_covered_count, dtype=np.uint32) + counts
        self.roomseg_ray_covered_count[:, :] = np.minimum(updated, uint16_max).astype(np.uint16)

    def _mark_roomseg_terminal_wall_cell(
        self,
        row: int,
        col: int,
        *,
        endpoint_depth_m: float,
        endpoint_rel_z_m: float,
    ) -> None:
        rr, cc = int(row), int(col)
        if not is_inside_grid(rr, cc, self.grid.map_info):
            return
        depth = float(endpoint_depth_m)
        rel_z = float(endpoint_rel_z_m)
        if not (np.isfinite(depth) and self.depth_min_m < depth < self.depth_max_m):
            return
        if not (
            float(self.vertical_profile_free_min_height_m)
            <= rel_z
            <= float(self.vertical_profile_free_max_height_m)
        ):
            return
        uint16_max = int(np.iinfo(np.uint16).max)
        self.roomseg_terminal_wall_count[rr, cc] = min(
            uint16_max,
            int(self.roomseg_terminal_wall_count[rr, cc]) + 1,
        )
        self.roomseg_terminal_wall_height_min[rr, cc] = min(
            float(self.roomseg_terminal_wall_height_min[rr, cc]),
            rel_z,
        )
        self.roomseg_terminal_wall_height_max[rr, cc] = max(
            float(self.roomseg_terminal_wall_height_max[rr, cc]),
            rel_z,
        )
        self.roomseg_terminal_wall_depth_min[rr, cc] = min(
            float(self.roomseg_terminal_wall_depth_min[rr, cc]),
            depth,
        )

    def _mark_roomseg_terminal_wall_flat(
        self,
        flat_values: np.ndarray,
        endpoint_depth_m: np.ndarray,
        endpoint_rel_z_m: np.ndarray,
    ) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        flat = np.asarray(flat_values, dtype=np.int64).reshape(-1)
        depth = np.asarray(endpoint_depth_m, dtype=np.float32).reshape(-1)
        rel_z = np.asarray(endpoint_rel_z_m, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return
        if depth.shape != flat.shape or rel_z.shape != flat.shape:
            raise ValueError("terminal wall flat/depth/height arrays must have the same shape")
        valid = (
            (flat >= 0)
            & (flat < total_cells)
            & np.isfinite(depth)
            & (depth > float(self.depth_min_m))
            & (depth < float(self.depth_max_m))
            & np.isfinite(rel_z)
            & (rel_z >= float(self.vertical_profile_free_min_height_m))
            & (rel_z <= float(self.vertical_profile_free_max_height_m))
        )
        if int(np.count_nonzero(valid)) == 0:
            return
        flat = flat[valid]
        depth = depth[valid]
        rel_z = rel_z[valid]
        uint16_max = int(np.iinfo(np.uint16).max)
        counts = np.bincount(flat, minlength=total_cells).astype(np.uint32)
        current = self.roomseg_terminal_wall_count.reshape(-1).astype(np.uint32)
        self.roomseg_terminal_wall_count.reshape(-1)[:] = np.minimum(current + counts, uint16_max).astype(np.uint16)
        np.minimum.at(self.roomseg_terminal_wall_height_min.reshape(-1), flat, rel_z)
        np.maximum.at(self.roomseg_terminal_wall_height_max.reshape(-1), flat, rel_z)
        np.minimum.at(self.roomseg_terminal_wall_depth_min.reshape(-1), flat, depth)

    def _refresh_roomseg_terminal_wall_splat(self, radius_cells: int = 1, min_count: int = 1) -> None:
        terminal = np.asarray(self.roomseg_terminal_wall_count, dtype=np.uint32) >= max(1, int(min_count))
        self.roomseg_terminal_wall_splat[:, :] = _dilate_binary(terminal, int(radius_cells)).astype(np.uint8)

    def _mark_vertical_profile_free_flat(self, flat_indices_by_band: List[List[int]]) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        uint16_max = int(np.iinfo(np.uint16).max)
        for band_idx, flat_values in enumerate(flat_indices_by_band):
            flat = _valid_flat_array(flat_values, total_cells)
            if flat.size == 0:
                continue
            counts = np.bincount(flat, minlength=total_cells).reshape(h, w).astype(np.uint32)
            free_updated = np.asarray(self.vertical_profile.free_ray_count[band_idx], dtype=np.uint32) + counts
            observed_updated = np.asarray(self.vertical_profile.observed_count[band_idx], dtype=np.uint32) + counts
            self.vertical_profile.free_ray_count[band_idx][:, :] = np.minimum(free_updated, uint16_max).astype(np.uint16)
            self.vertical_profile.observed_count[band_idx][:, :] = np.minimum(observed_updated, uint16_max).astype(np.uint16)
            touched = np.flatnonzero(counts.reshape(-1) > 0)
            if touched.size:
                self.vertical_profile.unknown_count[band_idx].reshape(-1)[touched] = 0

    def _mark_vertical_profile_free_weighted_flat(
        self,
        flat_indices_by_band: List[np.ndarray],
        weights_by_band: List[np.ndarray],
    ) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        uint16_max = int(np.iinfo(np.uint16).max)
        for band_idx, (flat_values, weights) in enumerate(zip(flat_indices_by_band, weights_by_band)):
            flat, weight = _valid_flat_weight_arrays(flat_values, weights, total_cells)
            if flat.size == 0:
                continue
            counts = np.bincount(flat, weights=weight, minlength=total_cells).reshape(h, w).astype(np.uint32)
            free_updated = np.asarray(self.vertical_profile.free_ray_count[band_idx], dtype=np.uint32) + counts
            observed_updated = np.asarray(self.vertical_profile.observed_count[band_idx], dtype=np.uint32) + counts
            self.vertical_profile.free_ray_count[band_idx][:, :] = np.minimum(free_updated, uint16_max).astype(np.uint16)
            self.vertical_profile.observed_count[band_idx][:, :] = np.minimum(observed_updated, uint16_max).astype(np.uint16)
            touched = np.flatnonzero(counts.reshape(-1) > 0)
            if touched.size:
                self.vertical_profile.unknown_count[band_idx].reshape(-1)[touched] = 0

    def _robot_footprint_radius_cells(self) -> int:
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        return int(math.ceil(max(0.0, self.robot_radius_m) / self.resolution_m))

    def _mark_voxel_navigation_endpoint_evidence(self, rows_cols: np.ndarray, rel_z: np.ndarray) -> None:
        cfg = self.voxel_navigation_projection_config
        if not bool(cfg.get("occupied_use_endpoint_hysteresis", True)):
            self.voxel_nav_endpoint_decay_debug = {
                "voxel_nav_endpoint_hysteresis_enabled": False,
                "voxel_nav_endpoint_incremented_cells": 0,
            }
            return
        rc = np.asarray(rows_cols, dtype=np.int32)
        z = np.asarray(rel_z, dtype=np.float32).reshape(-1)
        if rc.size == 0 or z.size == 0:
            self.voxel_nav_endpoint_decay_debug = {
                "voxel_nav_endpoint_hysteresis_enabled": True,
                "voxel_nav_endpoint_incremented_cells": 0,
            }
            return
        mask = (z >= float(cfg.get("obstacle_z_min_m", self.obstacle_min_height_m))) & (
            z <= float(cfg.get("obstacle_z_max_m", self.obstacle_max_height_m))
        )
        if not np.any(mask):
            self.voxel_nav_endpoint_decay_debug = {
                "voxel_nav_endpoint_hysteresis_enabled": True,
                "voxel_nav_endpoint_incremented_cells": 0,
            }
            return
        splatted = _splat_rows_cols(
            rc[mask],
            radius_cells=int(cfg.get("occupied_endpoint_xy_splat_radius_cells", 0)),
            shape=self.grid.free.shape,
        )
        if splatted.size == 0:
            incremented = 0
        else:
            width = int(self.grid.map_info.width)
            flat = np.unique(splatted[:, 0].astype(np.int64) * width + splatted[:, 1].astype(np.int64))
            counts = self.voxel_nav_occupied_endpoint_count.reshape(-1).astype(np.uint32)
            inc = max(1, int(cfg.get("occupied_endpoint_increment", 2)))
            counts[flat] = np.minimum(counts[flat] + inc, int(np.iinfo(np.uint16).max))
            self.voxel_nav_occupied_endpoint_count.reshape(-1)[:] = counts.astype(np.uint16)
            incremented = int(flat.size)
        self.voxel_nav_endpoint_decay_debug = {
            "voxel_nav_endpoint_hysteresis_enabled": True,
            "voxel_nav_endpoint_incremented_cells": int(incremented),
            "voxel_nav_endpoint_increment": int(cfg.get("occupied_endpoint_increment", 2)),
            "voxel_nav_endpoint_xy_splat_radius_cells": int(cfg.get("occupied_endpoint_xy_splat_radius_cells", 0)),
        }

    def _decay_voxel_navigation_endpoint_evidence(self, free_flat: np.ndarray) -> None:
        cfg = self.voxel_navigation_projection_config
        free_unique = _unique_flat_array(free_flat)
        decayed_cells = 0
        stale_cleared = 0
        if free_unique.size and bool(cfg.get("occupied_use_endpoint_hysteresis", True)):
            total_cells = int(self.grid.map_info.height) * int(self.grid.map_info.width)
            free_unique = free_unique[(free_unique >= 0) & (free_unique < total_cells)]
            if free_unique.size:
                counts = self.voxel_nav_occupied_endpoint_count.reshape(-1)
                free_ray_counts = self.voxel_nav_free_ray_count.reshape(-1).astype(np.uint32)
                free_ray_counts[free_unique] = np.minimum(free_ray_counts[free_unique] + 1, int(np.iinfo(np.uint16).max))
                self.voxel_nav_free_ray_count.reshape(-1)[:] = free_ray_counts.astype(np.uint16)
                protected = counts[free_unique] > 0
                if np.any(protected):
                    protected_flat = free_unique[protected]
                    before = counts[protected_flat].astype(np.int32)
                    decayed = np.maximum(before - max(1, int(cfg.get("occupied_endpoint_decay_per_free_ray", 1))), 0).astype(np.uint16)
                    counts[protected_flat] = decayed
                    decayed_cells = int(protected_flat.size)
                    stale_cleared = int(np.count_nonzero(decayed == 0))
        previous = dict(self.voxel_nav_endpoint_decay_debug)
        previous.update(
            {
                "voxel_nav_endpoint_decayed_cells": int(decayed_cells),
                "voxel_nav_endpoint_stale_cells_cleared": int(stale_cleared),
                "voxel_nav_endpoint_decay_per_free_ray": int(cfg.get("occupied_endpoint_decay_per_free_ray", 1)),
                "voxel_nav_endpoint_count_cells": int(np.count_nonzero(self.voxel_nav_occupied_endpoint_count)),
                "voxel_nav_endpoint_count_sum": int(np.sum(self.voxel_nav_occupied_endpoint_count, dtype=np.uint64)),
                "voxel_nav_free_ray_count_cells": int(np.count_nonzero(self.voxel_nav_free_ray_count)),
            }
        )
        self.voxel_nav_endpoint_decay_debug = previous

    def _splat_cells(
        self,
        rows_cols: np.ndarray,
        occupied: bool,
        point_threshold: int | None = None,
        mark_mask: np.ndarray | None = None,
    ) -> int:
        if rows_cols.size == 0:
            return 0
        threshold = self.splat_point_threshold if point_threshold is None else max(1, int(point_threshold))
        h, w = self.grid.occupied.shape
        flat = rows_cols[:, 0].astype(np.int64) * int(w) + rows_cols[:, 1].astype(np.int64)
        counts = np.bincount(flat, minlength=int(h * w))
        selected = np.flatnonzero(counts >= threshold)
        if len(selected) == 0:
            return 0
        rows = selected // int(w)
        cols = selected % int(w)
        for row, col in zip(rows, cols):
            if occupied:
                self._set_occupied_cell(int(row), int(col))
            else:
                self._set_free_cell(int(row), int(col))
            if mark_mask is not None:
                mark_mask[int(row), int(col)] = 1
        return int(len(selected))

    def _build_debug_stats(
        self,
        *,
        depth_arr: np.ndarray,
        valid_depth: np.ndarray,
        valid_pixels: np.ndarray,
        rel_z: np.ndarray,
        obstacle_mask: np.ndarray,
        free_mask: np.ndarray,
        occupied_endpoint_cells: int,
        free_ray_cells: int,
        free_protected_by_obstacle_endpoint: int,
        stale_obstacle_endpoint_cells_cleared: int,
        ray_count: int,
        skipped_height_rays: int,
        vertical_profile_ray_count: int,
        vertical_profile_skipped_height_rays: int,
        height_profile_ray_count: int,
        height_profile_skipped_height_rays: int,
        ray_cast_backend: str,
        ray_unique_endpoint_cells: int,
        base_pose_world: Tuple[float, float, float, float],
        camera_pose_world: Tuple[float, float, float, float],
        ray_origin_cell: Tuple[int, int],
    ) -> dict:
        height = int(depth_arr.shape[0])
        rows = valid_pixels[:, 1].astype(np.float32) if len(valid_pixels) else np.zeros((0,), dtype=np.float32)
        top = rows < height / 3.0
        middle = (rows >= height / 3.0) & (rows < 2.0 * height / 3.0)
        bottom = rows >= 2.0 * height / 3.0
        return {
            "reason": "ok",
            "mapping_mode": "depth_ray_cast",
            "depth_shape": [int(v) for v in depth_arr.shape],
            "depth_stride_px": int(self.depth_stride_px),
            "vertical_profile_depth_stride_px": int(self.depth_stride_px),
            "ray_cast_backend": str(ray_cast_backend),
            "ray_unique_endpoint_cells": int(ray_unique_endpoint_cells),
            "base_pose_world": [float(v) for v in base_pose_world],
            "camera_pose_world": [float(v) for v in camera_pose_world],
            "ray_origin_cell": [int(ray_origin_cell[0]), int(ray_origin_cell[1])],
            "depth_m_percentiles": _percentiles(valid_depth),
            "rel_z_m_percentiles": _percentiles(rel_z),
            "valid_points": int(len(rel_z)),
            "ray_count": int(ray_count),
            "skipped_height_rays": int(skipped_height_rays),
            "vertical_profile_ray_count": int(vertical_profile_ray_count),
            "vertical_profile_skipped_height_rays": int(vertical_profile_skipped_height_rays),
            "height_profile_enabled": bool(self.height_profile_config.enabled),
            "height_profile_ray_count": int(height_profile_ray_count),
            "height_profile_skipped_height_rays": int(height_profile_skipped_height_rays),
            "height_profile_z_bin_count": int(self.height_profile.z_bin_count),
            "legacy_vertical_profile_skipped_by_voxel_backend": bool(getattr(self, "_last_legacy_vertical_profile_skipped_by_voxel_backend", False)),
            "legacy_height_profile_skipped_by_voxel_backend": bool(getattr(self, "_last_legacy_height_profile_skipped_by_voxel_backend", False)),
            "legacy_roomseg_ray_skipped_by_voxel_backend": bool(getattr(self, "_last_legacy_roomseg_ray_skipped_by_voxel_backend", False)),
            **dict(self.last_ceiling_height_estimate.debug),
            "free_band_points": int(np.count_nonzero(free_mask)),
            "obstacle_band_points": int(np.count_nonzero(obstacle_mask)),
            "below_free_min_points": int(np.count_nonzero(rel_z < self.free_min_height_m)),
            "between_free_and_obstacle_points": int(
                np.count_nonzero((rel_z > self.free_max_height_m) & (rel_z < self.obstacle_min_height_m))
            ),
            "above_obstacle_max_points": int(np.count_nonzero(rel_z > self.obstacle_max_height_m)),
            "ceiling_like_points": int(np.count_nonzero(rel_z > 1.8)),
            "negative_height_points": int(np.count_nonzero(rel_z < -0.05)),
            "occupied_endpoint_cells": int(occupied_endpoint_cells),
            "free_ray_cells": int(free_ray_cells),
            "free_ray_cells_protected_by_obstacle_endpoint": int(free_protected_by_obstacle_endpoint),
            "stale_obstacle_endpoint_cells_cleared_by_free_rays": int(stale_obstacle_endpoint_cells_cleared),
            "obstacle_endpoint_evidence_increment": int(self._obstacle_endpoint_evidence_increment),
            "free_ray_obstacle_endpoint_decay": int(self._free_ray_obstacle_endpoint_decay),
            "depth_obstacle_endpoint_cells": int(np.count_nonzero(self.depth_obstacle_endpoint_count)),
            "depth_obstacle_endpoint_count_sum": int(np.sum(self.depth_obstacle_endpoint_count, dtype=np.uint64)),
            "voxel_nav_endpoint_count_cells": int(np.count_nonzero(self.voxel_nav_occupied_endpoint_count)),
            "voxel_nav_endpoint_count_sum": int(np.sum(self.voxel_nav_occupied_endpoint_count, dtype=np.uint64)),
            "voxel_nav_free_ray_count_cells": int(np.count_nonzero(self.voxel_nav_free_ray_count)),
            **dict(self.voxel_nav_endpoint_decay_debug),
            **dict(self.last_current_footprint_override_debug),
            "obstacle_splat_cells": int(occupied_endpoint_cells),
            "free_splat_cells": int(free_ray_cells),
            "image_bands": {
                "top": self._band_debug(rel_z, obstacle_mask, free_mask, top),
                "middle": self._band_debug(rel_z, obstacle_mask, free_mask, middle),
                "bottom": self._band_debug(rel_z, obstacle_mask, free_mask, bottom),
            },
            "height_filters_m": {
                "free_min": float(self.free_min_height_m),
                "free_max": float(self.free_max_height_m),
                "obstacle_min": float(self.obstacle_min_height_m),
                "obstacle_max": float(self.obstacle_max_height_m),
                "vertical_profile_free_min": float(self.vertical_profile_free_min_height_m),
                "vertical_profile_free_max": float(self.vertical_profile_free_max_height_m),
            },
            "vertical_profile": self.vertical_profile.to_debug_dict(),
            "voxel_grid": self.voxel_grid.to_debug_dict(),
            "voxel_navigation_projection": dict(self.voxel_grid.last_navigation_debug),
            "voxel_navigation_blind_zone": dict(self.last_voxel_blind_zone_debug),
            **dict(self.last_voxel_blind_zone_debug),
            "voxel_grid_drives_navigation": bool(self.voxel_grid_drives_navigation),
            "height_profile": {
                "z_min_m": float(self.height_profile.z_min_m),
                "z_max_m": float(self.height_profile.z_max_m),
                "z_bin_size_m": float(self.height_profile.z_bin_size_m),
                "z_bin_count": int(self.height_profile.z_bin_count),
                "active_z_min_m": float(self.height_profile_config.active_z_min_m),
                "active_z_max_m": float(self.height_profile_config.active_z_max_m or self.height_profile_config.active_z_max_cap_m or self.height_profile_config.z_max_m),
                "active_z_bin_count": int(np.count_nonzero(
                    (self.height_profile.bin_centers_m >= float(self.height_profile_config.active_z_min_m))
                    & (self.height_profile.bin_centers_m <= float(self.height_profile_config.active_z_max_m or self.height_profile_config.active_z_max_cap_m or self.height_profile_config.z_max_m))
                )),
                "ceiling_height_estimate_m": (
                    None
                    if self.last_ceiling_height_estimate.height_m is None
                    else float(self.last_ceiling_height_estimate.height_m)
                ),
                "ceiling_height_stable": bool(self.last_ceiling_height_estimate.stable),
                "ceiling_height_locked": bool(self.last_ceiling_height_estimate.locked),
                "free_ray_cells": int(np.count_nonzero(self.height_profile.free_ray_count)),
                "occupied_cells": int(np.count_nonzero(self.height_profile.occupied_count)),
                "observed_cells": int(np.count_nonzero(self.height_profile.observed_count)),
                "free_ray_count_sum": int(np.sum(self.height_profile.free_ray_count, dtype=np.uint64)),
                "occupied_count_sum": int(np.sum(self.height_profile.occupied_count, dtype=np.uint64)),
            },
            "roomseg_ray_evidence": {
                "ray_covered_cells": int(np.count_nonzero(self.roomseg_ray_covered_count)),
                "ray_covered_count_sum": int(np.sum(self.roomseg_ray_covered_count, dtype=np.uint64)),
                "terminal_wall_cells": int(np.count_nonzero(self.roomseg_terminal_wall_count)),
                "terminal_wall_count_sum": int(np.sum(self.roomseg_terminal_wall_count, dtype=np.uint64)),
                "terminal_wall_splat_cells": int(np.count_nonzero(self.roomseg_terminal_wall_splat)),
                "terminal_wall_depth_min_m_percentiles": _percentiles(
                    self.roomseg_terminal_wall_depth_min[np.isfinite(self.roomseg_terminal_wall_depth_min)]
                ),
                "terminal_wall_height_min_m_percentiles": _percentiles(
                    self.roomseg_terminal_wall_height_min[np.isfinite(self.roomseg_terminal_wall_height_min)]
                ),
                "terminal_wall_height_max_m_percentiles": _percentiles(
                    self.roomseg_terminal_wall_height_max[np.isfinite(self.roomseg_terminal_wall_height_max)]
                ),
            },
            "splat_thresholds": {
                "free": int(self.free_splat_point_threshold),
                "obstacle": int(self.splat_point_threshold),
                "unused_for_mapping_mode": "depth_ray_cast",
            },
        }

    @staticmethod
    def _band_debug(rel_z: np.ndarray, obstacle_mask: np.ndarray, free_mask: np.ndarray, band_mask: np.ndarray) -> dict:
        rel = rel_z[band_mask]
        return {
            "points": int(len(rel)),
            "rel_z_m_percentiles": _percentiles(rel),
            "free_band_points": int(np.count_nonzero(free_mask & band_mask)),
            "obstacle_band_points": int(np.count_nonzero(obstacle_mask & band_mask)),
            "ceiling_like_points": int(np.count_nonzero(rel > 1.8)),
            "negative_height_points": int(np.count_nonzero(rel < -0.05)),
        }


def _percentiles(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {}
    keys = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    vals = np.percentile(arr, keys)
    return {"p%d" % int(key): float(val) for key, val in zip(keys, vals)}


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - float(started_at)) * 1000.0)


@dataclass
class _RayCastEvidence:
    backend: str
    free_flat: np.ndarray
    occupied_flat: np.ndarray
    free_flat_by_band: List[np.ndarray]
    free_weight_by_band: List[np.ndarray]
    roomseg_ray_covered_flat: np.ndarray
    roomseg_ray_covered_weight: np.ndarray
    terminal_wall_flat: np.ndarray
    terminal_wall_depth_m: np.ndarray
    terminal_wall_rel_z_m: np.ndarray
    nav_ray_count: int
    nav_skipped_height_rays: int
    vertical_profile_ray_count: int
    vertical_profile_skipped_height_rays: int
    unique_endpoint_cells: int


def _world_points_to_grid(points_world: np.ndarray, info: MapInfo) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    cols = np.floor((points[:, 0].astype(np.float64) - float(info.min_x)) / float(info.resolution_m)).astype(np.int32)
    rows = np.floor((float(info.max_y) - points[:, 1].astype(np.float64)) / float(info.resolution_m)).astype(np.int32)
    return np.stack([rows, cols], axis=1)


def _collect_ray_cast_evidence(
    *,
    origin_cell: Tuple[int, int],
    endpoints: np.ndarray,
    endpoint_rel_z: np.ndarray,
    endpoint_depth_m: np.ndarray,
    endpoint_is_obstacle: np.ndarray,
    ray_can_clear: np.ndarray,
    map_width: int,
    depth_min_m: float,
    depth_max_m: float,
    camera_rel_z_m: float,
    z_min_m: float,
    z_max_m: float,
    band_ranges_m: Tuple[Tuple[float, float], ...],
) -> _RayCastEvidence:
    cells = np.asarray(endpoints, dtype=np.int32).reshape(-1, 2)
    rel_z = np.asarray(endpoint_rel_z, dtype=np.float32).reshape(-1)
    depth = np.asarray(endpoint_depth_m, dtype=np.float32).reshape(-1)
    obstacle = np.asarray(endpoint_is_obstacle, dtype=bool).reshape(-1)
    clear = np.asarray(ray_can_clear, dtype=bool).reshape(-1)
    n = int(cells.shape[0])
    if not (len(rel_z) == len(depth) == len(obstacle) == len(clear) == n):
        raise ValueError("ray evidence arrays must have matching lengths")
    band_count = len(tuple(band_ranges_m))
    empty_i64 = np.zeros((0,), dtype=np.int64)
    empty_f32 = np.zeros((0,), dtype=np.float32)
    if n == 0:
        return _RayCastEvidence(
            backend="endpoint_grouped_bresenham",
            free_flat=empty_i64,
            occupied_flat=empty_i64,
            free_flat_by_band=[empty_i64.copy() for _ in range(band_count)],
            free_weight_by_band=[empty_i64.copy() for _ in range(band_count)],
            roomseg_ray_covered_flat=empty_i64,
            roomseg_ray_covered_weight=empty_i64,
            terminal_wall_flat=empty_i64,
            terminal_wall_depth_m=empty_f32,
            terminal_wall_rel_z_m=empty_f32,
            nav_ray_count=0,
            nav_skipped_height_rays=0,
            vertical_profile_ray_count=0,
            vertical_profile_skipped_height_rays=0,
            unique_endpoint_cells=0,
        )

    width = int(map_width)
    endpoint_flat = cells[:, 0].astype(np.int64) * width + cells[:, 1].astype(np.int64)
    order = np.argsort(endpoint_flat, kind="stable")
    sorted_flat = endpoint_flat[order]
    unique_flat, starts, counts = np.unique(sorted_flat, return_index=True, return_counts=True)

    lo = np.maximum(float(z_min_m), np.minimum(float(camera_rel_z_m), rel_z.astype(np.float32)))
    hi = np.minimum(float(z_max_m), np.maximum(float(camera_rel_z_m), rel_z.astype(np.float32)))
    valid_interval = hi >= lo
    band_membership = np.zeros((band_count, n), dtype=bool)
    has_band_evidence = np.zeros((n,), dtype=bool)
    for band_idx, (band_lo, band_hi) in enumerate(band_ranges_m):
        band_mask = valid_interval & (float(band_hi) > lo) & (float(band_lo) < hi)
        if np.any(band_mask):
            band_membership[int(band_idx), band_mask] = True
            has_band_evidence[band_mask] = True

    roomseg_ray_valid = np.isfinite(depth) & (depth > float(depth_min_m)) & (depth < float(depth_max_m))
    terminal_valid = roomseg_ray_valid & np.isfinite(rel_z) & (rel_z >= float(z_min_m)) & (rel_z <= float(z_max_m))

    free_chunks: list[np.ndarray] = []
    covered_chunks: list[np.ndarray] = []
    covered_weight_chunks: list[np.ndarray] = []
    band_flat_chunks: list[list[np.ndarray]] = [[] for _ in range(band_count)]
    band_weight_chunks: list[list[np.ndarray]] = [[] for _ in range(band_count)]
    nav_ray_count = 0
    vertical_profile_ray_count = 0

    for endpoint, start, count in zip(unique_flat, starts, counts):
        group_indices = order[int(start) : int(start) + int(count)]
        end_row = int(endpoint // width)
        end_col = int(endpoint % width)
        line = _bresenham_cells(origin_cell, (end_row, end_col))
        if not line:
            continue
        line_flat = np.fromiter((int(row) * width + int(col) for row, col in line), dtype=np.int64)
        group_clear = group_indices[clear[group_indices]]
        if group_clear.size:
            nav_ray_count += int(group_clear.size)
            if np.any(~obstacle[group_clear]):
                free_chunks.append(line_flat)
            elif line_flat.size > 1:
                free_chunks.append(line_flat[:-1])

        profile_line_flat = line_flat[:-1]
        if profile_line_flat.size == 0:
            continue
        valid_group = group_indices[roomseg_ray_valid[group_indices] & has_band_evidence[group_indices]]
        if valid_group.size == 0:
            continue
        valid_count = int(valid_group.size)
        vertical_profile_ray_count += valid_count
        covered_chunks.append(profile_line_flat)
        covered_weight_chunks.append(np.full(profile_line_flat.shape, valid_count, dtype=np.int64))
        for band_idx in range(band_count):
            band_count_for_group = int(np.count_nonzero(band_membership[int(band_idx), valid_group]))
            if band_count_for_group <= 0:
                continue
            band_flat_chunks[band_idx].append(profile_line_flat)
            band_weight_chunks[band_idx].append(np.full(profile_line_flat.shape, band_count_for_group, dtype=np.int64))

    occupied_flat = endpoint_flat[obstacle]
    terminal_flat = endpoint_flat[terminal_valid]
    return _RayCastEvidence(
        backend="endpoint_grouped_bresenham",
        free_flat=_concat_i64(free_chunks),
        occupied_flat=np.asarray(occupied_flat, dtype=np.int64),
        free_flat_by_band=[_concat_i64(chunks) for chunks in band_flat_chunks],
        free_weight_by_band=[_concat_i64(chunks) for chunks in band_weight_chunks],
        roomseg_ray_covered_flat=_concat_i64(covered_chunks),
        roomseg_ray_covered_weight=_concat_i64(covered_weight_chunks),
        terminal_wall_flat=np.asarray(terminal_flat, dtype=np.int64),
        terminal_wall_depth_m=np.asarray(depth[terminal_valid], dtype=np.float32),
        terminal_wall_rel_z_m=np.asarray(rel_z[terminal_valid], dtype=np.float32),
        nav_ray_count=int(nav_ray_count),
        nav_skipped_height_rays=int(n - nav_ray_count),
        vertical_profile_ray_count=int(vertical_profile_ray_count),
        vertical_profile_skipped_height_rays=int(n - vertical_profile_ray_count),
        unique_endpoint_cells=int(len(unique_flat)),
    )


def _concat_i64(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate([np.asarray(chunk, dtype=np.int64).reshape(-1) for chunk in chunks]).astype(np.int64, copy=False)


def _valid_flat_array(values: List[int], total_cells: int) -> np.ndarray:
    if not values:
        return np.zeros((0,), dtype=np.int64)
    arr = np.asarray(values, dtype=np.int64)
    return arr[(arr >= 0) & (arr < int(total_cells))]


def _valid_flat_weight_arrays(values: np.ndarray, weights: np.ndarray, total_cells: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    if arr.shape != weight.shape:
        raise ValueError("flat value and weight arrays must have the same shape")
    valid = (arr >= 0) & (arr < int(total_cells)) & np.isfinite(weight) & (weight > 0.0)
    return arr[valid], weight[valid]


def _unique_flat(values: List[int]) -> np.ndarray:
    if not values:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.asarray(values, dtype=np.int64))


def _unique_flat_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(arr)


def _append_vertical_profile_free_ray_cells(
    flat_indices_by_band: List[List[int]],
    free_cells: List[Tuple[int, int]],
    *,
    map_width: int,
    origin_rel_z_m: float,
    endpoint_rel_z_m: float,
    z_min_m: float,
    z_max_m: float,
    vertical_profile: VerticalProfileMap,
) -> int:
    """Mark vertical free evidence for xy columns crossed by a free ray.

    The room-segmentation vertical profile asks: for this xy column, did any
    ray pass through free space between z_min and z_max? Using only the depth
    endpoint height drops valid rays that hit the floor or a low object, even
    though the same ray crossed 0.2-2.0 m free space before the hit. Therefore
    each crossed xy cell receives the clipped free vertical interval spanned by
    the camera origin and the depth endpoint.
    """

    if not free_cells:
        return 0
    width = int(map_width)
    lo = max(float(z_min_m), min(float(origin_rel_z_m), float(endpoint_rel_z_m)))
    hi = min(float(z_max_m), max(float(origin_rel_z_m), float(endpoint_rel_z_m)))
    if hi < lo:
        return 0
    band_indices: list[int] = []
    for band_idx, (_name, (band_lo, band_hi)) in enumerate(zip(vertical_profile.band_names, vertical_profile.band_ranges_m)):
        if float(band_hi) <= lo or float(band_lo) >= hi:
            continue
        band_indices.append(int(band_idx))
    if not band_indices:
        return 0
    added = 0
    for row, col in free_cells:
        flat = int(row) * width + int(col)
        for band_idx in band_indices:
            flat_indices_by_band[int(band_idx)].append(flat)
            added += 1
    return int(added)


def _disk_offsets(radius_cells: int) -> List[Tuple[int, int]]:
    rr = int(max(0, radius_cells))
    out: List[Tuple[int, int]] = []
    for dr in range(-rr, rr + 1):
        for dc in range(-rr, rr + 1):
            if dr * dr + dc * dc <= rr * rr:
                out.append((dr, dc))
    return out


def _splat_rows_cols(rows_cols: np.ndarray, *, radius_cells: int, shape: tuple[int, int]) -> np.ndarray:
    rc = np.asarray(rows_cols, dtype=np.int32).reshape(-1, 2)
    if rc.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    radius = max(0, int(radius_cells))
    h, w = int(shape[0]), int(shape[1])
    chunks: list[np.ndarray] = []
    for dr, dc in _disk_offsets(radius):
        out = rc.copy()
        out[:, 0] += int(dr)
        out[:, 1] += int(dc)
        valid = (out[:, 0] >= 0) & (out[:, 0] < h) & (out[:, 1] >= 0) & (out[:, 1] < w)
        if np.any(valid):
            chunks.append(out[valid])
    if not chunks:
        return np.zeros((0, 2), dtype=np.int32)
    merged = np.concatenate(chunks, axis=0)
    flat = np.unique(merged[:, 0].astype(np.int64) * w + merged[:, 1].astype(np.int64))
    out = np.empty((int(flat.size), 2), dtype=np.int32)
    out[:, 0] = (flat // w).astype(np.int32)
    out[:, 1] = (flat % w).astype(np.int32)
    return out


def _bresenham_cells(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    row0, col0 = int(start[0]), int(start[1])
    row1, col1 = int(end[0]), int(end[1])
    drow = abs(row1 - row0)
    dcol = abs(col1 - col0)
    srow = 1 if row0 < row1 else -1
    scol = 1 if col0 < col1 else -1
    row, col = row0, col0
    cells: List[Tuple[int, int]] = []
    if dcol > drow:
        err = dcol // 2
        while col != col1:
            cells.append((row, col))
            col += scol
            err -= drow
            if err < 0:
                row += srow
                err += dcol
        cells.append((row, col))
    else:
        err = drow // 2
        while row != row1:
            cells.append((row, col))
            row += srow
            err -= dcol
            if err < 0:
                col += scol
                err += drow
        cells.append((row, col))
    return cells


def _dilate_binary(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask).astype(bool)
    if radius_cells <= 0 or not np.any(src):
        return src
    out = np.array(src, copy=True)
    rows, cols = np.nonzero(src)
    h, w = src.shape
    offsets = _disk_offsets(radius_cells)
    for row, col in zip(rows, cols):
        for dr, dc in offsets:
            rr, cc = int(row + dr), int(col + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out
