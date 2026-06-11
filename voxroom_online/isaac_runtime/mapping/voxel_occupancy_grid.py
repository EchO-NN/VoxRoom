from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo


VOXEL_UNKNOWN = np.uint8(0)
VOXEL_FREE = np.uint8(1)
VOXEL_OCCUPIED = np.uint8(2)
VOXEL_CONFLICT = np.uint8(3)


@dataclass
class VoxelOccupancyGridConfig:
    enabled: bool = True
    z_min_m: float = -0.10
    z_max_m: float = 4.00
    z_resolution_m: float = 0.05
    active_z_min_m: float = 0.10
    active_z_max_mode: str = "ceiling_ratio"
    active_z_max_ceiling_ratio: float = 0.85
    active_z_max_cap_m: float = 2.80
    active_z_max_fallback_m: float = 2.80
    integration_backend: str = "cpu_numba"
    cuda_device: str = "cuda:0"
    cuda_chunk_rays: int = 8192
    cuda_ray_step_voxels: float = 1.00
    cuda_max_samples_per_ray: int = 320
    cuda_keep_logodds_on_device: bool = True
    cpu_numba_threads: int = 28
    cpu_numba_threads_mode: str = "auto"
    cpu_numba_autotune_candidates: tuple[int, ...] = (2, 4, 8, 14, 28)
    cpu_numba_autotune_repeat: int = 2
    cpu_numba_autotune_rays: int = 30000
    cpu_numba_autotune_cache_path: str = "debug/voxel_cpu_numba_autotune.json"
    cpu_numba_autotune_metric: str = "integrate_ms"
    cpu_numba_chunk_rays: int = 131072
    cpu_numba_use_bincount_updates: bool = True
    cpu_numba_max_samples_per_ray: int = 320
    cpu_numba_preallocate_buffers: bool = True
    cpu_numba_skip_per_ray_unique_if_step_voxels_ge_1: bool = True
    cpu_numba_strict_required: bool = True
    cpu_numba_fail_if_thread_count_below: int = 2
    cpu_numba_report_threading_layer: bool = True
    cpu_numba_event_block_size: int = 4096
    cpu_numba_event_chunk_count_multiplier: int = 4
    cpu_numba_inline_state_refresh: bool = True
    cpu_numba_disable_changed_flatnonzero: bool = True
    python_debug_backend_allowed: bool = False
    depth_stride_px: int = 2
    depth_min_m: float = 0.20
    depth_max_m: float = 5.00
    depth_range_semantics: str = "image_plane_z"
    max_range_free_only_rays_enabled: bool = True
    depth_greater_than_max_marks_free_to_max: bool = True
    depth_inf_marks_free_to_max: bool = True
    depth_nan_marks_free_to_max: bool = False
    depth_zero_marks_free_to_max: bool = False
    depth_max_epsilon_m: float = 0.01
    free_logodds_delta: int = -1
    occupied_logodds_delta: int = 4
    free_logodds_threshold: int = -1
    occupied_logodds_threshold: int = 1
    occupied_to_free_logodds_threshold: int = -6
    ray_traversal_mode: str = "exact_dda"
    ray_traversal_tie_epsilon: float = 1.0e-6
    free_vote_cap_per_voxel: int = 1
    occupied_vote_cap_per_voxel: int = 3
    occupied_wins_over_free_same_voxel: bool = True
    logodds_min: int = -20
    logodds_max: int = 20
    mark_endpoint_occupied: bool = True
    free_excludes_endpoint: bool = True
    free_excludes_last_n_voxels_before_endpoint: int = 0
    endpoint_splat_xy_radius_cells: int = 0
    endpoint_splat_z_radius_cells: int = 0
    conflict_enabled: bool = False
    conflict_margin: int = 0
    voxel_grid_drives_navigation: bool = True
    sensor_range_tracking_enabled: bool = True
    sensor_range_count_delta: int = 1
    sensor_range_count_max: int = 255
    sensor_range_count_threshold: int = 1
    sensor_range_projective_frustum_volume_enabled: bool = True
    sensor_range_projective_active_z_only: bool = True
    sensor_range_projective_chunk_voxels: int = 262144
    sensor_range_projective_update_every_steps: int = 1
    sensor_range_mark_effective_frustum_enabled: bool = False
    sensor_range_effective_step_voxels: float = 1.00
    sensor_range_effective_max_samples_per_ray: int = 320
    sensor_range_effective_chunk_rays: int = 4096
    sensor_range_mark_endpoint_column_enabled: bool = False
    sensor_range_endpoint_column_xy_radius_cells: int = 0
    sensor_range_mark_active_z_only: bool = True
    sensor_range_mark_ray_samples_enabled: bool = False
    sensor_range_mark_ray_samples_for_debug: bool = False
    sensor_range_behind_endpoint_margin_m: float = 0.00
    sensor_range_count_decay_per_update: int = 0
    outside_boundary_enabled: bool = True
    outside_source: str = "navigation_free_and_full_height_no_occupied"
    outside_occupied_check_z_min_m: float = -0.10
    outside_occupied_check_z_max_m: float = 4.00
    outside_require_navigation_free: bool = True
    outside_navigation_free_source: str = "voxel_navigation_projection_free"
    outside_require_floor_frustum_seen: bool = True
    outside_floor_visibility_rel_z_m: float = -0.05
    outside_floor_frustum_seen_count_threshold: int = 1
    outside_score_increment: int = 1
    outside_score_decay: int = 1
    outside_score_threshold: int = 2
    outside_score_max: int = 5
    outside_exclude_initial_blind_zone_free: bool = True
    outside_exclude_current_footprint_free: bool = True
    outside_current_footprint_exclusion_radius_m: float = 0.25
    outside_use_as_navigation_boundary: bool = True
    outside_use_as_roomseg_domain_boundary: bool = True
    outside_use_as_frontier_boundary: bool = True
    outside_use_as_wall_evidence: bool = False
    outside_use_as_door_anchor: bool = False
    outside_use_as_separator_anchor: bool = False
    outside_use_as_generalized_occupied: bool = False
    outside_use_as_protected_structural_wall: bool = False
    outside_require_disconnected_from_robot_navigation_component: bool = True
    outside_connectivity_source: str = "astar_traversible_pre_outside"
    outside_connectivity_allow_diagonal: bool = True
    outside_connectivity_prevent_diagonal_corner_cutting: bool = True
    outside_robot_snap_to_traversible_max_radius_cells: int = 20
    outside_hard_clear_robot_component_scores: bool = True
    outside_fail_closed_if_robot_component_unavailable: bool = True
    outside_debug_robot_component_enabled: bool = True
    outside_debug_enabled: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "VoxelOccupancyGridConfig":
        if isinstance(data, cls):
            cfg = data
            for key, value in overrides.items():
                if value is not None and hasattr(cfg, key):
                    setattr(cfg, key, value)
            return cfg
        raw = dict(data or {})
        for key, value in overrides.items():
            if value is not None:
                raw[key] = value
        if "cpu_numba_autotune_candidates" in raw:
            value = raw["cpu_numba_autotune_candidates"]
            if isinstance(value, str):
                raw["cpu_numba_autotune_candidates"] = tuple(int(v.strip()) for v in value.split(",") if v.strip())
            else:
                raw["cpu_numba_autotune_candidates"] = tuple(int(v) for v in value)
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


def _active_z_cap_or_storage_max(config: VoxelOccupancyGridConfig) -> float:
    cap = float(getattr(config, "active_z_max_cap_m", 0.0))
    if np.isfinite(cap) and cap > 0.0:
        return float(cap)
    return float(getattr(config, "z_max_m", 3.20))


@dataclass
class NavigationProjection:
    free: np.ndarray
    occupied: np.ndarray
    observed: np.ndarray
    unknown: np.ndarray
    debug: dict[str, object] = field(default_factory=dict)


@dataclass
class VoxelNavigationProjectionCache:
    free: np.ndarray
    occupied: np.ndarray
    observed: np.ndarray
    unknown: np.ndarray
    occupied_from_voxel: np.ndarray
    occupied_from_endpoint: np.ndarray
    free_raw: np.ndarray
    observed_from_voxel: np.ndarray
    dirty_rc_flags: np.ndarray
    initialized: bool = False
    last_full_refresh_step: int = -1


@dataclass
class NavigationProjectionConfig:
    obstacle_z_min_m: float = 0.20
    obstacle_z_max_m: float = 0.90
    free_z_min_m: float = 0.10
    free_z_max_m: float = 0.90
    min_free_voxels: int = 1
    occupied_any_voxel_wins: bool = True
    occupied_use_endpoint_hysteresis: bool = True
    occupied_endpoint_count_threshold: int = 1
    occupied_endpoint_decay_per_free_ray: int = 1
    occupied_endpoint_increment: int = 2
    occupied_endpoint_xy_splat_radius_cells: int = 0
    occupied_endpoint_z_splat_radius_cells: int = 0
    occupied_close_radius_cells: int = 1
    occupied_fill_small_holes_max_area_cells: int = 4
    occupied_priority_over_free: bool = True
    unknown_preserve_when_no_observation: bool = True
    debug_navigation_projection_layers: bool = False
    incremental_enabled: bool = True
    full_refresh_on_frontier_update: bool = True
    full_refresh_interval_steps: int = 30
    dirty_dilation_radius_cells: int = 2
    local_morphology_enabled: bool = True
    full_morphology_only_on_replan: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "NavigationProjectionConfig":
        raw = dict(data or {})
        for key, value in overrides.items():
            if value is not None:
                raw[key] = value
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class VoxelIntegrationStats:
    depth_rays_integrated: int = 0
    depth_hit_rays_integrated: int = 0
    depth_free_only_rays_integrated: int = 0
    skipped_empty_rays: int = 0
    free_update_count: int = 0
    occupied_update_count: int = 0
    integration_backend: str = "none"
    python_debug_backend_used: bool = False
    integrate_total_ms: float = 0.0
    ray_sample_ms: float = 0.0
    unique_ms: float = 0.0
    scatter_ms: float = 0.0
    refresh_state_ms: float = 0.0
    refresh_mode: str = "none"
    refresh_changed_voxels: int = 0
    cuda_chunk_rays: int = 0
    cuda_max_samples_per_ray: int = 0
    sensor_range_update_count: int = 0
    sensor_range_decay_applied: int = 0
    voxel_integrate_backend_thread_count: int = 0
    voxel_integrate_backend_effective_thread_count: int = 0
    voxel_integrate_numba_requested_thread_count: int = 0
    voxel_integrate_numba_threading_layer: str = "unknown"
    voxel_integrate_numba_threads_mode: str = "manual"
    voxel_integrate_numba_autotune_ms: float = 0.0
    voxel_integrate_sample_kernel_ms: float = 0.0
    voxel_integrate_bincount_free_ms: float = 0.0
    voxel_integrate_bincount_occ_ms: float = 0.0
    voxel_integrate_bincount_sensor_ms: float = 0.0
    voxel_integrate_apply_logodds_ms: float = 0.0
    voxel_integrate_endpoint_column_ms: float = 0.0
    voxel_integrate_buffer_alloc_ms: float = 0.0
    voxel_integrate_total_samples: int = 0
    voxel_integrate_total_unique_free_voxels: int = 0
    voxel_integrate_total_unique_occ_voxels: int = 0
    voxel_integrate_total_unique_sensor_voxels: int = 0
    voxel_integrate_total_sensor_events: int = 0
    voxel_integrate_total_occ_events: int = 0
    voxel_integrate_pass1_ms: float = 0.0
    voxel_integrate_event_prefix_ms: float = 0.0
    voxel_integrate_pass2_ms: float = 0.0
    voxel_integrate_event_bucket_ms: float = 0.0
    voxel_integrate_point_filter_ms: float = 0.0
    voxel_integrate_world_to_voxel_ms: float = 0.0
    voxel_integrate_count_kernel_ms: float = 0.0
    voxel_integrate_prefix_ms: float = 0.0
    voxel_integrate_write_events_ms: float = 0.0
    voxel_integrate_bucket_free_ms: float = 0.0
    voxel_integrate_bucket_occ_ms: float = 0.0
    voxel_integrate_bucket_sensor_ms: float = 0.0
    voxel_integrate_apply_sensor_ms: float = 0.0
    voxel_integrate_changed_extract_ms: float = 0.0
    voxel_integrate_changed_scan_ms: float = 0.0
    voxel_integrate_refresh_state_ms: float = 0.0
    voxel_integrate_projection_ms: float = 0.0
    voxel_integrate_free_event_count: int = 0
    voxel_integrate_occ_event_count: int = 0
    voxel_integrate_sensor_event_count: int = 0
    voxel_integrate_changed_flag_count: int = 0
    voxel_integrate_dirty_rc_count: int = 0
    voxel_integrate_touched_block_count: int = 0
    voxel_integrate_num_blocks: int = 0
    voxel_numba_threading_layer: str = "unknown"
    voxel_numba_requested_unavailable: bool = False
    voxel_sensor_range_projective_frustum_updates: int = 0
    voxel_sensor_range_projective_frustum_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "voxel_depth_rays_integrated": int(self.depth_rays_integrated),
            "voxel_depth_hit_rays_integrated": int(self.depth_hit_rays_integrated),
            "voxel_depth_free_only_rays_integrated": int(self.depth_free_only_rays_integrated),
            "voxel_depth_hit_ray_count": int(self.depth_hit_rays_integrated),
            "voxel_depth_free_only_ray_count": int(self.depth_free_only_rays_integrated),
            "voxel_depth_endpoint_is_hit_count": int(self.depth_hit_rays_integrated),
            "voxel_depth_endpoint_is_free_only_count": int(self.depth_free_only_rays_integrated),
            "voxel_depth_rays_skipped_empty": int(self.skipped_empty_rays),
            "voxel_free_update_count": int(self.free_update_count),
            "voxel_occupied_update_count": int(self.occupied_update_count),
            "voxel_integration_backend": str(self.integration_backend),
            "voxel_python_debug_backend_used": bool(self.python_debug_backend_used),
            "python_debug_backend_used": bool(self.python_debug_backend_used),
            "voxel_integrate_total_ms": float(self.integrate_total_ms),
            "voxel_integrate_ray_sample_ms": float(self.ray_sample_ms),
            "voxel_integrate_unique_ms": float(self.unique_ms),
            "voxel_integrate_scatter_ms": float(self.scatter_ms),
            "voxel_refresh_state_ms": float(self.refresh_state_ms),
            "voxel_refresh_mode": str(self.refresh_mode),
            "voxel_refresh_changed_voxels": int(self.refresh_changed_voxels),
            "voxel_cuda_chunk_rays": int(self.cuda_chunk_rays),
            "voxel_cuda_max_samples_per_ray": int(self.cuda_max_samples_per_ray),
            "voxel_sensor_range_update_count": int(self.sensor_range_update_count),
            "voxel_sensor_range_decay_applied": int(self.sensor_range_decay_applied),
            "voxel_integrate_backend_thread_count": int(self.voxel_integrate_backend_thread_count),
            "voxel_integrate_backend_effective_thread_count": int(self.voxel_integrate_backend_effective_thread_count),
            "voxel_integrate_numba_requested_thread_count": int(self.voxel_integrate_numba_requested_thread_count),
            "voxel_integrate_numba_threading_layer": str(self.voxel_integrate_numba_threading_layer),
            "voxel_integrate_numba_threads_mode": str(self.voxel_integrate_numba_threads_mode),
            "voxel_integrate_numba_autotune_ms": float(self.voxel_integrate_numba_autotune_ms),
            "voxel_integrate_sample_kernel_ms": float(self.voxel_integrate_sample_kernel_ms),
            "voxel_integrate_bincount_free_ms": float(self.voxel_integrate_bincount_free_ms),
            "voxel_integrate_bincount_occ_ms": float(self.voxel_integrate_bincount_occ_ms),
            "voxel_integrate_bincount_sensor_ms": float(self.voxel_integrate_bincount_sensor_ms),
            "voxel_integrate_apply_logodds_ms": float(self.voxel_integrate_apply_logodds_ms),
            "voxel_integrate_endpoint_column_ms": float(self.voxel_integrate_endpoint_column_ms),
            "voxel_integrate_buffer_alloc_ms": float(self.voxel_integrate_buffer_alloc_ms),
            "voxel_integrate_total_samples": int(self.voxel_integrate_total_samples),
            "voxel_integrate_total_unique_free_voxels": int(self.voxel_integrate_total_unique_free_voxels),
            "voxel_integrate_total_unique_occ_voxels": int(self.voxel_integrate_total_unique_occ_voxels),
            "voxel_integrate_total_unique_sensor_voxels": int(self.voxel_integrate_total_unique_sensor_voxels),
            "voxel_integrate_total_sensor_events": int(self.voxel_integrate_total_sensor_events),
            "voxel_integrate_total_occ_events": int(self.voxel_integrate_total_occ_events),
            "voxel_integrate_pass1_ms": float(self.voxel_integrate_pass1_ms),
            "voxel_integrate_event_prefix_ms": float(self.voxel_integrate_event_prefix_ms),
            "voxel_integrate_pass2_ms": float(self.voxel_integrate_pass2_ms),
            "voxel_integrate_event_bucket_ms": float(self.voxel_integrate_event_bucket_ms),
            "voxel_integrate_point_filter_ms": float(self.voxel_integrate_point_filter_ms),
            "voxel_integrate_world_to_voxel_ms": float(self.voxel_integrate_world_to_voxel_ms),
            "voxel_integrate_count_kernel_ms": float(self.voxel_integrate_count_kernel_ms),
            "voxel_integrate_prefix_ms": float(self.voxel_integrate_prefix_ms),
            "voxel_integrate_write_events_ms": float(self.voxel_integrate_write_events_ms),
            "voxel_integrate_bucket_free_ms": float(self.voxel_integrate_bucket_free_ms),
            "voxel_integrate_bucket_occ_ms": float(self.voxel_integrate_bucket_occ_ms),
            "voxel_integrate_bucket_sensor_ms": float(self.voxel_integrate_bucket_sensor_ms),
            "voxel_integrate_apply_sensor_ms": float(self.voxel_integrate_apply_sensor_ms),
            "voxel_integrate_changed_extract_ms": float(self.voxel_integrate_changed_extract_ms),
            "voxel_integrate_changed_scan_ms": float(self.voxel_integrate_changed_scan_ms),
            "voxel_integrate_refresh_state_ms": float(self.voxel_integrate_refresh_state_ms),
            "voxel_integrate_projection_ms": float(self.voxel_integrate_projection_ms),
            "voxel_integrate_free_event_count": int(self.voxel_integrate_free_event_count),
            "voxel_integrate_occ_event_count": int(self.voxel_integrate_occ_event_count),
            "voxel_integrate_sensor_event_count": int(self.voxel_integrate_sensor_event_count),
            "voxel_integrate_changed_flag_count": int(self.voxel_integrate_changed_flag_count),
            "voxel_integrate_dirty_rc_count": int(self.voxel_integrate_dirty_rc_count),
            "voxel_integrate_touched_block_count": int(self.voxel_integrate_touched_block_count),
            "voxel_integrate_num_blocks": int(self.voxel_integrate_num_blocks),
            "voxel_numba_threading_layer": str(self.voxel_numba_threading_layer),
            "voxel_numba_requested_unavailable": bool(self.voxel_numba_requested_unavailable),
            "voxel_sensor_range_projective_frustum_updates": int(self.voxel_sensor_range_projective_frustum_updates),
            "voxel_sensor_range_projective_frustum_ms": float(self.voxel_sensor_range_projective_frustum_ms),
        }


@dataclass
class VoxelOccupancyGrid3D:
    log_odds: np.ndarray
    state: np.ndarray
    sensor_range_count: np.ndarray
    floor_frustum_seen_count_xy: np.ndarray
    outside_score_xy: np.ndarray
    outside_xy: np.ndarray
    z_min_m: float
    z_max_m: float
    z_resolution_m: float
    map_info: MapInfo
    config: VoxelOccupancyGridConfig = field(default_factory=VoxelOccupancyGridConfig)
    active_z_min_m: float = 0.10
    active_z_max_m: float | None = None
    ceiling_height_m: float | None = None
    ceiling_estimate_status: str = "unavailable"
    last_integration_stats: VoxelIntegrationStats = field(default_factory=VoxelIntegrationStats)
    last_navigation_debug: dict[str, object] = field(default_factory=dict)
    last_projective_frustum_debug: dict[str, object] = field(default_factory=dict)
    last_floor_frustum_debug: dict[str, object] = field(default_factory=dict)
    last_outside_debug: dict[str, object] = field(default_factory=dict)
    _navigation_projection_cache: VoxelNavigationProjectionCache | None = None
    last_dirty_rc_flags: np.ndarray | None = None

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        map_info: MapInfo,
        cfg: VoxelOccupancyGridConfig | Mapping[str, object] | None = None,
    ) -> "VoxelOccupancyGrid3D":
        config = cfg if isinstance(cfg, VoxelOccupancyGridConfig) else VoxelOccupancyGridConfig.from_mapping(cfg)
        height, width = int(shape[0]), int(shape[1])
        z_bins = int(math.ceil((float(config.z_max_m) - float(config.z_min_m)) / float(config.z_resolution_m)))
        if z_bins <= 0:
            raise ValueError("voxel grid requires at least one z bin")
        log_odds = np.zeros((z_bins, height, width), dtype=np.int16)
        state = np.zeros((z_bins, height, width), dtype=np.uint8)
        sensor_range_count = np.zeros((z_bins, height, width), dtype=np.uint8)
        floor_frustum_seen_count_xy = np.zeros((height, width), dtype=np.uint16)
        outside_score_xy = np.zeros((height, width), dtype=np.uint8)
        outside_xy = np.zeros((height, width), dtype=bool)
        return cls(
            log_odds=log_odds,
            state=state,
            sensor_range_count=sensor_range_count,
            floor_frustum_seen_count_xy=floor_frustum_seen_count_xy,
            outside_score_xy=outside_score_xy,
            outside_xy=outside_xy,
            z_min_m=float(config.z_min_m),
            z_max_m=float(config.z_max_m),
            z_resolution_m=float(config.z_resolution_m),
            map_info=map_info,
            config=config,
            active_z_min_m=float(config.active_z_min_m),
            active_z_max_m=float(_active_z_cap_or_storage_max(config)),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.state.shape[1]), int(self.state.shape[2])

    @property
    def z_bin_count(self) -> int:
        return int(self.state.shape[0])

    @property
    def z_centers_m(self) -> np.ndarray:
        return self.z_min_m + (np.arange(self.z_bin_count, dtype=np.float32) + 0.5) * self.z_resolution_m

    def reset_shape(self, shape: tuple[int, int], map_info: MapInfo | None = None) -> None:
        if map_info is not None:
            self.map_info = map_info
        height, width = int(shape[0]), int(shape[1])
        self.log_odds = np.zeros((self.z_bin_count, height, width), dtype=np.int16)
        self.state = np.zeros((self.z_bin_count, height, width), dtype=np.uint8)
        self.sensor_range_count = np.zeros((self.z_bin_count, height, width), dtype=np.uint8)
        self.floor_frustum_seen_count_xy = np.zeros((height, width), dtype=np.uint16)
        self.outside_score_xy = np.zeros((height, width), dtype=np.uint8)
        self.outside_xy = np.zeros((height, width), dtype=bool)
        self.last_integration_stats = VoxelIntegrationStats()
        self.last_navigation_debug = {}
        self.last_projective_frustum_debug = {}
        self.last_floor_frustum_debug = {}
        self.last_outside_debug = {}
        self._navigation_projection_cache = None
        self.last_dirty_rc_flags = np.zeros(height * width, dtype=np.uint8)

    def z_index_for_height(self, rel_z_m: float) -> int | None:
        zf = (float(rel_z_m) - self.z_min_m) / self.z_resolution_m
        if not np.isfinite(zf):
            return None
        zi = int(math.floor(zf))
        if zi < 0 or zi >= self.z_bin_count:
            return None
        return zi

    def voxel_center_z(self, z_idx: int) -> float:
        return float(self.z_min_m + (int(z_idx) + 0.5) * self.z_resolution_m)

    def world_to_voxel_float(self, point_world_xyz: Sequence[float], floor_z: float) -> tuple[float, float, float]:
        point = np.asarray(point_world_xyz, dtype=np.float64).reshape(-1)
        if point.size < 3:
            raise ValueError("point_world_xyz must have at least 3 values")
        z_float = (float(point[2]) - float(floor_z) - self.z_min_m) / self.z_resolution_m
        row_float = (float(self.map_info.max_y) - float(point[1])) / float(self.map_info.resolution_m)
        col_float = (float(point[0]) - float(self.map_info.min_x)) / float(self.map_info.resolution_m)
        return float(z_float), float(row_float), float(col_float)

    def ray_voxels_3d(
        self,
        origin_world_xyz: Sequence[float],
        endpoint_world_xyz: Sequence[float],
        *,
        floor_z: float,
        include_endpoint: bool = True,
    ) -> list[tuple[int, int, int]]:
        p0 = np.asarray(self.world_to_voxel_float(origin_world_xyz, floor_z), dtype=np.float64)
        p1 = np.asarray(self.world_to_voxel_float(endpoint_world_xyz, floor_z), dtype=np.float64)
        if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
            return []
        clipped = self._clip_segment_to_volume(p0, p1)
        if clipped is None:
            return []
        p0, p1 = clipped
        cells = self._dda_voxels(p0, p1)
        if not include_endpoint and cells:
            return cells[:-1]
        return cells

    def integrate_depth_points(
        self,
        *,
        camera_origin_world: Sequence[float],
        points_world: np.ndarray,
        depths_m: np.ndarray | None = None,
        floor_z: float,
        valid_mask: np.ndarray | None = None,
        endpoint_is_hit: np.ndarray | None = None,
    ) -> VoxelIntegrationStats:
        started_at = time.perf_counter()
        stats = VoxelIntegrationStats()
        self.last_dirty_rc_flags = np.zeros(int(self.shape[0] * self.shape[1]), dtype=np.uint8)
        if not bool(self.config.enabled):
            stats.integration_backend = "disabled"
            stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
            self.last_integration_stats = stats
            return stats
        sensor_range_decay_applied = self.decay_sensor_range_count()
        stats.sensor_range_decay_applied = int(sensor_range_decay_applied)
        points = np.asarray(points_world, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3 or points.size == 0:
            stats.integration_backend = self._resolve_integration_backend()
            stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
            self.last_integration_stats = stats
            return stats
        if valid_mask is None:
            valid = np.ones(points.shape[0], dtype=bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
            if valid.size != points.shape[0]:
                raise ValueError("valid_mask must match points_world length")
        if endpoint_is_hit is None:
            hit = np.ones(points.shape[0], dtype=bool)
        else:
            hit = np.asarray(endpoint_is_hit, dtype=bool).reshape(-1)
            if hit.size != points.shape[0]:
                raise ValueError("endpoint_is_hit must match points_world length")
        backend = self._resolve_integration_backend()
        if backend == "cuda_torch":
            from voxroom_online.isaac_runtime.mapping.voxel_cuda_backend import VoxelCudaBackend

            stats = VoxelCudaBackend.integrate(
                self,
                camera_origin_world=camera_origin_world,
                points_world=points,
                floor_z=float(floor_z),
                valid_mask=valid,
                endpoint_is_hit=hit,
            )
        elif backend == "cpu_numba":
            from voxroom_online.isaac_runtime.mapping.voxel_cpu_numba_backend import VoxelCpuNumbaBackend

            stats = VoxelCpuNumbaBackend.integrate(
                self,
                camera_origin_world=camera_origin_world,
                points_world=points,
                floor_z=float(floor_z),
                valid_mask=valid,
                endpoint_is_hit=hit,
            )
        elif backend == "cpu_vectorized":
            from voxroom_online.isaac_runtime.mapping.voxel_cpu_fast_backend import VoxelCpuVectorizedBackend

            stats = VoxelCpuVectorizedBackend.integrate(
                self,
                camera_origin_world=camera_origin_world,
                points_world=points,
                floor_z=float(floor_z),
                valid_mask=valid,
                endpoint_is_hit=hit,
                backend_name="cpu_vectorized",
            )
        elif backend == "python_debug":
            if not bool(self.config.python_debug_backend_allowed):
                raise RuntimeError("Python voxel DDA backend is debug-only and disabled for runtime")
            stats = self._integrate_depth_points_python_debug(
                camera_origin_world=camera_origin_world,
                points_world=points,
                depths_m=depths_m,
                floor_z=float(floor_z),
                valid_mask=valid,
                endpoint_is_hit=hit,
            )
        else:
            raise RuntimeError("unsupported voxel integration backend: %s" % backend)
        if str(getattr(stats, "integration_backend", backend)) != "cpu_numba":
            if int(getattr(stats, "free_update_count", 0)) or int(getattr(stats, "occupied_update_count", 0)):
                self.last_dirty_rc_flags = np.ones(int(self.shape[0] * self.shape[1]), dtype=np.uint8)
                self.invalidate_navigation_projection_cache(reason="non_numba_backend_update")
        stats.sensor_range_decay_applied = int(sensor_range_decay_applied)
        stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
        self.last_integration_stats = stats
        return stats

    def _resolve_integration_backend(self) -> str:
        requested = str(getattr(self.config, "integration_backend", "auto") or "auto").strip().lower()
        valid = {"auto", "cuda_torch", "cpu_numba", "cpu_vectorized", "python_debug"}
        if requested not in valid:
            raise ValueError("unsupported voxel integration backend: %s" % requested)
        if requested == "auto":
            try:
                from voxroom_online.isaac_runtime.mapping.voxel_cuda_backend import VoxelCudaBackend

                if VoxelCudaBackend.is_available(str(self.config.cuda_device)):
                    return "cuda_torch"
            except Exception:
                pass
            try:
                from voxroom_online.isaac_runtime.mapping.voxel_cpu_numba_backend import numba_available

                if numba_available():
                    return "cpu_numba"
            except Exception:
                pass
            return "cpu_vectorized"
        if requested == "cuda_torch":
            from voxroom_online.isaac_runtime.mapping.voxel_cuda_backend import VoxelCudaBackend

            if not VoxelCudaBackend.is_available(str(self.config.cuda_device)):
                raise RuntimeError("cuda_torch voxel backend requested but torch CUDA is unavailable")
            return requested
        if requested == "cpu_numba":
            return requested
        return requested

    def _integrate_depth_points_python_debug(
        self,
        *,
        camera_origin_world: Sequence[float],
        points_world: np.ndarray,
        depths_m: np.ndarray | None = None,
        floor_z: float,
        valid_mask: np.ndarray | None = None,
        endpoint_is_hit: np.ndarray | None = None,
    ) -> VoxelIntegrationStats:
        _ = depths_m
        started_at = time.perf_counter()
        stats = VoxelIntegrationStats(integration_backend="python_debug", python_debug_backend_used=True)
        points = np.asarray(points_world, dtype=np.float32)
        if valid_mask is None:
            valid = np.ones(points.shape[0], dtype=bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if endpoint_is_hit is None:
            hit_mask = np.ones(points.shape[0], dtype=bool)
        else:
            hit_mask = np.asarray(endpoint_is_hit, dtype=bool).reshape(-1)
        origin = np.asarray(camera_origin_world, dtype=np.float32).reshape(-1)[:3]
        free_voxels: list[tuple[int, int, int]] = []
        occupied_voxels: list[tuple[int, int, int]] = []
        sensor_range_voxels: list[tuple[int, int, int]] = []
        flush_threshold = 250_000
        exclude_n = max(0, int(self.config.free_excludes_last_n_voxels_before_endpoint))
        sensor_enabled = bool(getattr(self.config, "sensor_range_tracking_enabled", True))
        sensor_ray_enabled = sensor_enabled and bool(getattr(self.config, "sensor_range_mark_ray_samples_enabled", True))
        sensor_endpoint_enabled = sensor_enabled and bool(getattr(self.config, "sensor_range_mark_endpoint_column_enabled", True))
        valid_indices = np.nonzero(valid)[0]
        for ray_index in valid_indices:
            point = points[int(ray_index)]
            ray_is_hit = bool(hit_mask[int(ray_index)])
            voxels = self.ray_voxels_3d(origin, point[:3], floor_z=float(floor_z), include_endpoint=True)
            if not voxels:
                stats.skipped_empty_rays += 1
                continue
            stats.depth_rays_integrated += 1
            if ray_is_hit:
                stats.depth_hit_rays_integrated += 1
            else:
                stats.depth_free_only_rays_integrated += 1
            if sensor_ray_enabled:
                sensor_range_voxels.extend(voxels)
            if ray_is_hit and sensor_endpoint_enabled:
                sensor_range_voxels.extend(self._sensor_endpoint_column_voxels(voxels[-1]))
            if ray_is_hit and bool(self.config.mark_endpoint_occupied):
                occupied_voxels.extend(self._endpoint_splat(voxels[-1]))
            if ray_is_hit and bool(self.config.free_excludes_endpoint):
                free_end = max(0, len(voxels) - 1 - exclude_n)
                free_voxels.extend(voxels[:free_end])
            else:
                free_voxels.extend(voxels)
            if len(free_voxels) >= flush_threshold:
                stats.free_update_count += self.mark_free_voxels(free_voxels)
                free_voxels.clear()
            if len(occupied_voxels) >= flush_threshold:
                stats.occupied_update_count += self.mark_occupied_voxels(occupied_voxels)
                occupied_voxels.clear()
            if len(sensor_range_voxels) >= flush_threshold:
                count, _changed = self.mark_sensor_range_voxels_array(sensor_range_voxels)
                stats.sensor_range_update_count += int(count)
                sensor_range_voxels.clear()
        stats.free_update_count += self.mark_free_voxels(free_voxels)
        stats.occupied_update_count += self.mark_occupied_voxels(occupied_voxels)
        count, _changed = self.mark_sensor_range_voxels_array(sensor_range_voxels)
        stats.sensor_range_update_count += int(count)
        refresh_started_at = time.perf_counter()
        self.refresh_state()
        stats.refresh_state_ms = float((time.perf_counter() - refresh_started_at) * 1000.0)
        stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
        return stats

    def mark_free_voxels(self, voxels: Iterable[Sequence[int]]) -> int:
        count, _changed = self._add_logodds_array(voxels, int(self.config.free_logodds_delta))
        return int(count)

    def mark_occupied_voxels(self, voxels: Iterable[Sequence[int]]) -> int:
        count, _changed = self._add_logodds_array(voxels, int(self.config.occupied_logodds_delta))
        return int(count)

    def mark_sensor_effective_range_rays(
        self,
        *,
        camera_origin_world: Sequence[float],
        range_endpoints_world: np.ndarray,
        floor_z: float,
        step_voxels: float | None = None,
        max_samples_per_ray: int | None = None,
        chunk_rays: int | None = None,
    ) -> int:
        """Mark voxels that have appeared inside the camera frustum.

        This updates only sensor_range_count. It must not create free or occupied
        evidence, because in-range unknown means unknown state plus frustum coverage.
        """
        if not bool(getattr(self.config, "sensor_range_tracking_enabled", True)):
            return 0
        if not bool(getattr(self.config, "sensor_range_mark_effective_frustum_enabled", True)):
            return 0
        endpoints = np.asarray(range_endpoints_world, dtype=np.float32)
        if endpoints.ndim != 2 or endpoints.shape[1] < 3 or endpoints.size == 0:
            return 0
        endpoints = endpoints[:, :3]
        valid_endpoints = np.all(np.isfinite(endpoints), axis=1)
        if not np.any(valid_endpoints):
            return 0
        endpoints = endpoints[valid_endpoints]
        origin = np.asarray(camera_origin_world, dtype=np.float32).reshape(-1)
        if origin.size < 3 or not np.all(np.isfinite(origin[:3])):
            return 0
        p0 = np.asarray(self.world_to_voxel_float(origin[:3], float(floor_z)), dtype=np.float32)
        if not np.all(np.isfinite(p0)):
            return 0
        step = float(
            step_voxels
            if step_voxels is not None
            else getattr(self.config, "sensor_range_effective_step_voxels", 1.0)
        )
        step = max(0.25, step)
        max_samples = int(
            max_samples_per_ray
            if max_samples_per_ray is not None
            else getattr(self.config, "sensor_range_effective_max_samples_per_ray", 320)
        )
        max_samples = max(2, max_samples)
        chunk_n = int(
            chunk_rays
            if chunk_rays is not None
            else getattr(self.config, "sensor_range_effective_chunk_rays", 4096)
        )
        chunk_n = max(1, chunk_n)
        height = int(self.state.shape[1])
        width = int(self.state.shape[2])
        z_bins = int(self.z_bin_count)
        total = 0
        for start in range(0, int(endpoints.shape[0]), chunk_n):
            chunk = endpoints[start : start + chunk_n]
            p1 = np.empty((int(chunk.shape[0]), 3), dtype=np.float32)
            p1[:, 0] = (chunk[:, 2] - float(floor_z) - float(self.z_min_m)) / float(self.z_resolution_m)
            p1[:, 1] = (float(self.map_info.max_y) - chunk[:, 1]) / float(self.map_info.resolution_m)
            p1[:, 2] = (chunk[:, 0] - float(self.map_info.min_x)) / float(self.map_info.resolution_m)
            valid = np.all(np.isfinite(p1), axis=1)
            if not np.any(valid):
                continue
            p1 = p1[valid]
            delta = p1 - p0.reshape(1, 3)
            max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
            samples = int(min(max_samples, max(2, math.ceil(max_delta / step) + 1)))
            t = np.linspace(0.0, 1.0, samples, dtype=np.float32).reshape(samples, 1, 1)
            pts = p0.reshape(1, 1, 3) + t * delta.reshape(1, int(delta.shape[0]), 3)
            vox = np.floor(pts).astype(np.int32).reshape(-1, 3)
            inside = (
                (vox[:, 0] >= 0)
                & (vox[:, 0] < z_bins)
                & (vox[:, 1] >= 0)
                & (vox[:, 1] < height)
                & (vox[:, 2] >= 0)
                & (vox[:, 2] < width)
            )
            if not np.any(inside):
                continue
            vox = vox[inside]
            flat = (
                vox[:, 0].astype(np.int64) * int(height * width)
                + vox[:, 1].astype(np.int64) * int(width)
                + vox[:, 2].astype(np.int64)
            )
            total += int(self.mark_sensor_range_flat_indices(flat))
        return int(total)

    def mark_sensor_projective_frustum_volume(
        self,
        *,
        camera_pose_world: Sequence[float],
        intr: object,
        floor_z: float,
        depth_min_m: float,
        depth_max_m: float,
        active_z_only: bool | None = None,
        chunk_voxels: int | None = None,
    ) -> dict[str, object]:
        """Mark voxel centers inside the projective image-plane z-depth frustum."""
        started_at = time.perf_counter()
        projective_enabled = bool(getattr(self.config, "sensor_range_projective_frustum_volume_enabled", True))
        depth_semantics = str(getattr(self.config, "depth_range_semantics", "image_plane_z"))
        self.last_projective_frustum_debug = {
            "voxel_sensor_range_mark_mode": "projective_frustum_volume",
            "voxel_sensor_projective_frustum_enabled": bool(projective_enabled),
            "voxel_sensor_projective_frustum_candidate_voxels": 0,
            "voxel_sensor_projective_frustum_inside_voxels": 0,
            "voxel_sensor_projective_frustum_updates": 0,
            "voxel_sensor_projective_frustum_ms": 0.0,
            "voxel_sensor_depth_range_semantics": depth_semantics,
            "voxel_sensor_range_projective_frustum_enabled": bool(projective_enabled),
            "voxel_sensor_range_projective_candidate_voxels": 0,
            "voxel_sensor_range_projective_inside_voxels": 0,
            "voxel_sensor_range_projective_updates": 0,
            "voxel_sensor_range_projective_frustum_ms": 0.0,
        }

        def finish(updates_value: int = 0) -> dict[str, object]:
            elapsed_ms = float((time.perf_counter() - started_at) * 1000.0)
            self.last_projective_frustum_debug["voxel_sensor_projective_frustum_updates"] = int(updates_value)
            self.last_projective_frustum_debug["voxel_sensor_projective_frustum_ms"] = elapsed_ms
            self.last_projective_frustum_debug["voxel_sensor_range_projective_updates"] = int(updates_value)
            self.last_projective_frustum_debug["voxel_sensor_range_projective_frustum_ms"] = elapsed_ms
            return dict(self.last_projective_frustum_debug)

        if not bool(getattr(self.config, "sensor_range_tracking_enabled", True)):
            return finish(0)
        if not bool(getattr(self.config, "sensor_range_projective_frustum_volume_enabled", True)):
            return finish(0)
        pose = np.asarray(camera_pose_world, dtype=np.float64).reshape(-1)
        if pose.size < 4 or not np.all(np.isfinite(pose[:4])):
            return finish(0)
        depth_min = max(0.0, float(depth_min_m))
        depth_max = float(depth_max_m)
        if not np.isfinite(depth_max) or depth_max <= depth_min:
            return finish(0)
        use_active_z = (
            bool(getattr(self.config, "sensor_range_projective_active_z_only", True))
            if active_z_only is None
            else bool(active_z_only)
        )
        z_indices = np.asarray(self.active_z_indices() if use_active_z else np.arange(self.z_bin_count), dtype=np.int64)
        if z_indices.size == 0:
            return finish(0)
        fx = float(getattr(intr, "fx"))
        fy = float(getattr(intr, "fy"))
        cx = float(getattr(intr, "cx"))
        cy = float(getattr(intr, "cy"))
        image_w = int(getattr(intr, "width"))
        image_h = int(getattr(intr, "height"))
        if not all(np.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0.0 or fy <= 0.0 or image_w <= 0 or image_h <= 0:
            return finish(0)
        cam_x, cam_y, cam_z, yaw = float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        corners_world: list[tuple[float, float, float]] = [(cam_x, cam_y, cam_z)]
        for depth in (depth_min, depth_max):
            for u in (0.0, float(image_w - 1)):
                for v in (0.0, float(image_h - 1)):
                    forward = float(depth)
                    left = (cx - u) * forward / fx
                    up = (cy - v) * forward / fy
                    world_x = cam_x + cos_yaw * forward - sin_yaw * left
                    world_y = cam_y + sin_yaw * forward + cos_yaw * left
                    world_z = cam_z + up
                    corners_world.append((world_x, world_y, world_z))
        corners = np.asarray(corners_world, dtype=np.float64)
        resolution = float(self.map_info.resolution_m)
        height = int(self.state.shape[1])
        width = int(self.state.shape[2])
        x_min = float(np.min(corners[:, 0])) - resolution
        x_max = float(np.max(corners[:, 0])) + resolution
        y_min = float(np.min(corners[:, 1])) - resolution
        y_max = float(np.max(corners[:, 1])) + resolution
        z_world_min = float(np.min(corners[:, 2])) - float(self.z_resolution_m)
        z_world_max = float(np.max(corners[:, 2])) + float(self.z_resolution_m)
        col_min = max(0, int(math.floor((x_min - float(self.map_info.min_x)) / resolution)))
        col_max = min(width - 1, int(math.floor((x_max - float(self.map_info.min_x)) / resolution)))
        row_min = max(0, int(math.floor((float(self.map_info.max_y) - y_max) / resolution)))
        row_max = min(height - 1, int(math.floor((float(self.map_info.max_y) - y_min) / resolution)))
        z_min_idx = max(0, int(math.floor((z_world_min - float(floor_z) - float(self.z_min_m)) / float(self.z_resolution_m))))
        z_max_idx = min(
            self.z_bin_count - 1,
            int(math.floor((z_world_max - float(floor_z) - float(self.z_min_m)) / float(self.z_resolution_m))),
        )
        if col_max < col_min or row_max < row_min or z_max_idx < z_min_idx:
            return finish(0)
        z_indices = z_indices[(z_indices >= int(z_min_idx)) & (z_indices <= int(z_max_idx))]
        if z_indices.size == 0:
            return finish(0)
        rows = np.arange(row_min, row_max + 1, dtype=np.int64)
        cols = np.arange(col_min, col_max + 1, dtype=np.int64)
        sub_height = int(rows.size)
        sub_width = int(cols.size)
        sub_hw = int(sub_height * sub_width)
        total = int(z_indices.size) * sub_hw
        hw = int(height * width)
        self.last_projective_frustum_debug.update(
            {
                "voxel_sensor_projective_frustum_candidate_voxels": int(total),
                "voxel_sensor_range_projective_candidate_voxels": int(total),
                "voxel_sensor_range_projective_aabb_rows": [int(row_min), int(row_max)],
                "voxel_sensor_range_projective_aabb_cols": [int(col_min), int(col_max)],
                "voxel_sensor_range_projective_aabb_z": [int(z_min_idx), int(z_max_idx)],
                "voxel_sensor_range_projective_active_z_count": int(z_indices.size),
            }
        )
        chunk_n = max(
            1,
            int(
                chunk_voxels
                if chunk_voxels is not None
                else getattr(self.config, "sensor_range_projective_chunk_voxels", 262144)
            ),
        )
        updates = 0
        inside_voxels = 0
        z_centers = (float(floor_z) + self.z_min_m + (z_indices.astype(np.float64) + 0.5) * float(self.z_resolution_m))
        for start in range(0, total, chunk_n):
            end = min(total, start + chunk_n)
            flat = np.arange(start, end, dtype=np.int64)
            z_pos = flat // sub_hw
            rc_flat = flat % sub_hw
            r = rows[rc_flat // sub_width]
            c = cols[rc_flat % sub_width]
            world_x = float(self.map_info.min_x) + (c.astype(np.float64) + 0.5) * float(self.map_info.resolution_m)
            world_y = float(self.map_info.max_y) - (r.astype(np.float64) + 0.5) * float(self.map_info.resolution_m)
            world_z = z_centers[z_pos]
            dx = world_x - cam_x
            dy = world_y - cam_y
            dz = world_z - cam_z
            forward = cos_yaw * dx + sin_yaw * dy
            left = -sin_yaw * dx + cos_yaw * dy
            up = dz
            valid_forward = (forward >= depth_min) & (forward <= depth_max)
            if not np.any(valid_forward):
                continue
            u = cx - left * fx / np.maximum(forward, 1.0e-6)
            v = cy - up * fy / np.maximum(forward, 1.0e-6)
            inside = valid_forward & (u >= 0.0) & (u < float(image_w)) & (v >= 0.0) & (v < float(image_h))
            if not np.any(inside):
                continue
            inside_voxels += int(np.count_nonzero(inside))
            z = z_indices[z_pos[inside]]
            flat_indices = z.astype(np.int64) * hw + r[inside].astype(np.int64) * int(width) + c[inside].astype(np.int64)
            updates += int(self.mark_sensor_range_flat_indices(flat_indices))
        self.last_projective_frustum_debug["voxel_sensor_projective_frustum_inside_voxels"] = int(inside_voxels)
        self.last_projective_frustum_debug["voxel_sensor_range_projective_inside_voxels"] = int(inside_voxels)
        return finish(updates)

    def mark_floor_projective_frustum_visibility(
        self,
        *,
        camera_pose_world: Sequence[float],
        intr: object,
        floor_z: float,
        depth_min_m: float,
        depth_max_m: float,
        floor_rel_z_m: float | None = None,
        count_delta: int = 1,
        count_max: int = 65535,
    ) -> dict[str, object]:
        """Mark XY floor cells that lie inside the camera image-plane z-depth frustum."""
        started_at = time.perf_counter()
        enabled = bool(getattr(self.config, "outside_boundary_enabled", True)) and bool(
            getattr(self.config, "outside_require_floor_frustum_seen", True)
        )
        rel_floor_z = float(
            getattr(self.config, "outside_floor_visibility_rel_z_m", -0.05)
            if floor_rel_z_m is None
            else floor_rel_z_m
        )
        floor_world_z = float(floor_z) + rel_floor_z
        debug: dict[str, object] = {
            "voxel_floor_frustum_enabled": bool(enabled),
            "voxel_floor_frustum_floor_rel_z_m": float(rel_floor_z),
            "voxel_floor_frustum_floor_world_z": float(floor_world_z),
            "voxel_floor_frustum_candidate_xy": 0,
            "voxel_floor_frustum_inside_xy": 0,
            "voxel_floor_frustum_updates": 0,
            "voxel_floor_frustum_seen_xy": int(np.count_nonzero(self.floor_frustum_seen_count_xy)),
            "voxel_floor_frustum_seen_xy_count": int(np.count_nonzero(self.floor_frustum_seen_count_xy)),
            "voxel_floor_frustum_depth_min_m": float(depth_min_m),
            "voxel_floor_frustum_depth_max_m": float(depth_max_m),
            "voxel_floor_frustum_semantics": "image_plane_z",
            "voxel_floor_frustum_depth_semantics": "image_plane_z",
            "voxel_floor_frustum_ms": 0.0,
        }

        def finish() -> dict[str, object]:
            debug["voxel_floor_frustum_seen_xy"] = int(np.count_nonzero(self.floor_frustum_seen_count_xy))
            debug["voxel_floor_frustum_seen_xy_count"] = int(np.count_nonzero(self.floor_frustum_seen_count_xy))
            debug["voxel_floor_frustum_ms"] = float((time.perf_counter() - started_at) * 1000.0)
            if bool(getattr(self.config, "outside_debug_enabled", True)):
                debug["voxel_floor_frustum_seen_xy_map"] = (
                    self.floor_frustum_seen_count_xy
                    >= max(1, int(getattr(self.config, "outside_floor_frustum_seen_count_threshold", 1)))
                )
            self.last_floor_frustum_debug = dict(debug)
            return dict(debug)

        if not enabled:
            return finish()
        pose = np.asarray(camera_pose_world, dtype=np.float64).reshape(-1)
        if pose.size < 4 or not np.all(np.isfinite(pose[:4])):
            return finish()
        depth_min = max(0.0, float(depth_min_m))
        depth_max = float(depth_max_m)
        if not np.isfinite(depth_max) or depth_max <= depth_min:
            return finish()
        fx = float(getattr(intr, "fx"))
        fy = float(getattr(intr, "fy"))
        cx = float(getattr(intr, "cx"))
        cy = float(getattr(intr, "cy"))
        image_w = int(getattr(intr, "width"))
        image_h = int(getattr(intr, "height"))
        if not all(np.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0.0 or fy <= 0.0 or image_w <= 0 or image_h <= 0:
            return finish()

        cam_x, cam_y, cam_z, yaw = float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        corners_world: list[tuple[float, float]] = [(cam_x, cam_y)]
        for depth in (depth_min, depth_max):
            for u in (0.0, float(image_w - 1)):
                for v in (0.0, float(image_h - 1)):
                    forward = float(depth)
                    left = (cx - u) * forward / fx
                    world_x = cam_x + cos_yaw * forward - sin_yaw * left
                    world_y = cam_y + sin_yaw * forward + cos_yaw * left
                    corners_world.append((world_x, world_y))
        corners = np.asarray(corners_world, dtype=np.float64)
        resolution = float(self.map_info.resolution_m)
        height = int(self.state.shape[1])
        width = int(self.state.shape[2])
        x_min = float(np.min(corners[:, 0])) - resolution
        x_max = float(np.max(corners[:, 0])) + resolution
        y_min = float(np.min(corners[:, 1])) - resolution
        y_max = float(np.max(corners[:, 1])) + resolution
        col_min = max(0, int(math.floor((x_min - float(self.map_info.min_x)) / resolution)))
        col_max = min(width - 1, int(math.floor((x_max - float(self.map_info.min_x)) / resolution)))
        row_min = max(0, int(math.floor((float(self.map_info.max_y) - y_max) / resolution)))
        row_max = min(height - 1, int(math.floor((float(self.map_info.max_y) - y_min) / resolution)))
        if col_max < col_min or row_max < row_min:
            return finish()

        rows = np.arange(row_min, row_max + 1, dtype=np.int64)
        cols = np.arange(col_min, col_max + 1, dtype=np.int64)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        world_x = float(self.map_info.min_x) + (cc.astype(np.float64) + 0.5) * resolution
        world_y = float(self.map_info.max_y) - (rr.astype(np.float64) + 0.5) * resolution
        dx = world_x - cam_x
        dy = world_y - cam_y
        dz = float(floor_world_z) - cam_z
        forward = cos_yaw * dx + sin_yaw * dy
        left = -sin_yaw * dx + cos_yaw * dy
        up = dz
        valid_forward = (forward >= depth_min) & (forward <= depth_max)
        u = cx - left * fx / np.maximum(forward, 1.0e-6)
        v = cy - up * fy / np.maximum(forward, 1.0e-6)
        inside = valid_forward & (u >= 0.0) & (u < float(image_w)) & (v >= 0.0) & (v < float(image_h))
        debug["voxel_floor_frustum_candidate_xy"] = int(inside.size)
        debug["voxel_floor_frustum_inside_xy"] = int(np.count_nonzero(inside))
        if np.any(inside):
            target_rows = rr[inside].astype(np.int64)
            target_cols = cc[inside].astype(np.int64)
            current = self.floor_frustum_seen_count_xy[target_rows, target_cols].astype(np.uint32)
            delta = max(1, int(count_delta))
            cap = int(np.clip(int(count_max), 1, 65535))
            self.floor_frustum_seen_count_xy[target_rows, target_cols] = np.minimum(current + int(delta), cap).astype(np.uint16)
            debug["voxel_floor_frustum_updates"] = int(target_rows.size)
        return finish()

    def update_outside_from_navigation_free(
        self,
        navigation_free_xy: np.ndarray,
        *,
        pre_outside_traversible_xy: np.ndarray | None = None,
        robot_grid: tuple[int, int] | None = None,
        forced_initial_blind_zone_free_xy: np.ndarray | None = None,
        forced_current_footprint_free_xy: np.ndarray | None = None,
        synthetic_free_provenance_xy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        cfg = self.config
        shape = self.shape
        nav_free = np.asarray(navigation_free_xy, dtype=bool)
        if nav_free.shape != shape:
            raise ValueError("navigation_free_xy shape %s does not match voxel shape %s" % (nav_free.shape, shape))
        enabled = bool(getattr(cfg, "outside_boundary_enabled", True))
        require_disconnected = bool(getattr(cfg, "outside_require_disconnected_from_robot_navigation_component", True))
        outside_check_z_min_m = float(getattr(cfg, "outside_occupied_check_z_min_m", self.z_min_m))
        outside_check_z_max_m = float(getattr(cfg, "outside_occupied_check_z_max_m", self.z_max_m))
        outside_check_z_min_m = max(float(self.z_min_m), float(outside_check_z_min_m))
        outside_check_z_max_m = min(float(self.z_max_m), float(outside_check_z_max_m))
        z_idx = self.z_indices_for_world_range(outside_check_z_min_m, outside_check_z_max_m)
        if z_idx.size:
            has_occupied_xy = np.any(self.state[z_idx] == int(VOXEL_OCCUPIED), axis=0)
        else:
            has_occupied_xy = np.zeros(shape, dtype=bool)
        full_height_no_occupied_xy = ~has_occupied_xy
        floor_seen_xy = self.floor_frustum_seen_count_xy >= max(
            1,
            int(getattr(cfg, "outside_floor_frustum_seen_count_threshold", 1)),
        )
        initial_forced = (
            np.zeros(shape, dtype=bool)
            if forced_initial_blind_zone_free_xy is None
            else np.asarray(forced_initial_blind_zone_free_xy, dtype=bool)
        )
        current_forced = (
            np.zeros(shape, dtype=bool)
            if forced_current_footprint_free_xy is None
            else np.asarray(forced_current_footprint_free_xy, dtype=bool)
        )
        initial_forced_invalid = bool(initial_forced.shape != shape)
        current_forced_invalid = bool(current_forced.shape != shape)
        if initial_forced.shape != shape:
            initial_forced = np.zeros(shape, dtype=bool)
        if current_forced.shape != shape:
            current_forced = np.zeros(shape, dtype=bool)
        current_forced_raw = current_forced.astype(bool, copy=True)
        if bool(getattr(cfg, "outside_exclude_current_footprint_free", True)):
            radius_cells = int(
                math.ceil(
                    max(0.0, float(getattr(cfg, "outside_current_footprint_exclusion_radius_m", 0.0)))
                    / max(float(self.map_info.resolution_m), 1.0e-6)
                )
            )
            if radius_cells > 0 and np.any(current_forced):
                current_forced = _dilate_bool(current_forced, radius_cells)
        forced_excluded = np.zeros(shape, dtype=bool)
        if bool(getattr(cfg, "outside_exclude_initial_blind_zone_free", True)):
            forced_excluded |= initial_forced
        if bool(getattr(cfg, "outside_exclude_current_footprint_free", True)):
            forced_excluded |= current_forced

        synthetic_excluded = np.zeros(shape, dtype=bool)
        synthetic_invalid = False
        synthetic_input = _validate_bool_xy_mask(synthetic_free_provenance_xy, shape, "synthetic_free_provenance_xy", strict=False)
        if synthetic_input is None and synthetic_free_provenance_xy is not None:
            synthetic_invalid = True
        elif synthetic_input is not None:
            synthetic_excluded = synthetic_input

        candidate_base = np.ones(shape, dtype=bool)
        if bool(getattr(cfg, "outside_require_navigation_free", True)):
            candidate_base &= nav_free
        candidate_base &= full_height_no_occupied_xy
        if bool(getattr(cfg, "outside_require_floor_frustum_seen", True)):
            candidate_base &= floor_seen_xy
        candidate_base &= ~forced_excluded
        candidate_base &= ~synthetic_excluded
        if not enabled:
            candidate_base &= False

        trav: np.ndarray | None = None
        robot_component = np.zeros(shape, dtype=bool)
        connected_to_robot_rejected = np.zeros(shape, dtype=bool)
        robot_component_available = False
        robot_seed_grid: tuple[int, int] | None = None
        robot_seed_snapped_grid: tuple[int, int] | None = None
        robot_seed_snap_distance_cells: float | None = None
        if require_disconnected:
            if pre_outside_traversible_xy is not None:
                trav = _validate_bool_xy_mask(pre_outside_traversible_xy, shape, "pre_outside_traversible_xy", strict=True)
            if trav is not None and robot_grid is not None:
                robot_seed_grid = (int(robot_grid[0]), int(robot_grid[1]))
                max_radius = max(0, int(getattr(cfg, "outside_robot_snap_to_traversible_max_radius_cells", 20)))
                snapped, snap_d2 = _snap_to_true_cell(trav, robot_seed_grid, max_radius)
                if snapped is not None:
                    robot_seed_snapped_grid = (int(snapped[0]), int(snapped[1]))
                    robot_seed_snap_distance_cells = float(math.sqrt(float(snap_d2)))
                    robot_component = _reachable_component_from_seed(
                        trav,
                        robot_seed_snapped_grid,
                        allow_diagonal=bool(getattr(cfg, "outside_connectivity_allow_diagonal", True)),
                        prevent_diagonal_corner_cutting=bool(
                            getattr(cfg, "outside_connectivity_prevent_diagonal_corner_cutting", True)
                        ),
                    )
                    robot_component_available = bool(np.any(robot_component))
            if robot_component_available:
                connected_to_robot_rejected = candidate_base & robot_component
                candidate = candidate_base & ~robot_component
            elif bool(getattr(cfg, "outside_fail_closed_if_robot_component_unavailable", True)):
                candidate = np.zeros(shape, dtype=bool)
            else:
                candidate = candidate_base
        else:
            candidate = candidate_base

        increment = max(0, int(getattr(cfg, "outside_score_increment", 1)))
        decay = max(0, int(getattr(cfg, "outside_score_decay", 1)))
        score_max = max(1, int(getattr(cfg, "outside_score_max", 5)))
        scores = self.outside_score_xy.astype(np.int16)
        if decay:
            scores[~candidate] = np.maximum(scores[~candidate] - int(decay), 0)
        if increment:
            scores[candidate] = np.minimum(scores[candidate] + int(increment), score_max)
        clear_mask = np.ones(shape, dtype=bool) if not enabled else (has_occupied_xy | forced_excluded | synthetic_excluded)
        if require_disconnected and bool(getattr(cfg, "outside_hard_clear_robot_component_scores", True)):
            clear_mask |= robot_component
        scores[clear_mask] = 0
        self.outside_score_xy[:, :] = np.clip(scores, 0, 255).astype(np.uint8)
        threshold = max(1, int(getattr(cfg, "outside_score_threshold", 2)))
        self.outside_xy[:, :] = self.outside_score_xy >= threshold
        holdout_floor_unseen = nav_free & full_height_no_occupied_xy & ~floor_seen_xy
        combined_synthetic_excluded = forced_excluded | synthetic_excluded
        robot_component_cells = int(np.count_nonzero(robot_component))
        outside_overlap_robot_component = int(np.count_nonzero(self.outside_xy & robot_component))
        debug: dict[str, object] = {
            "voxel_outside_enabled": bool(enabled),
            "voxel_outside_candidate_cells": int(np.count_nonzero(candidate)),
            "voxel_outside_candidate_base_cells": int(np.count_nonzero(candidate_base)),
            "voxel_outside_disconnected_candidate_cells": int(np.count_nonzero(candidate)),
            "voxel_outside_confirmed_cells": int(np.count_nonzero(self.outside_xy)),
            "voxel_outside_source": str(getattr(cfg, "outside_source", "navigation_free_and_full_height_no_occupied")),
            "voxel_outside_navigation_free_source_cells": int(np.count_nonzero(nav_free)),
            "voxel_outside_has_occupied_xy_cells": int(np.count_nonzero(has_occupied_xy)),
            "voxel_outside_full_height_no_occupied_cells": int(np.count_nonzero(full_height_no_occupied_xy)),
            "voxel_outside_floor_frustum_seen_cells": int(np.count_nonzero(floor_seen_xy)),
            "voxel_outside_holdout_floor_unseen_cells": int(np.count_nonzero(holdout_floor_unseen)),
            "voxel_outside_forced_initial_blind_excluded_cells": int(np.count_nonzero(initial_forced)),
            "voxel_outside_forced_current_footprint_excluded_cells": int(np.count_nonzero(current_forced_raw)),
            "voxel_outside_current_footprint_radius_excluded_cells": int(np.count_nonzero(current_forced)),
            "voxel_outside_forced_initial_blind_shape_invalid": bool(initial_forced_invalid),
            "voxel_outside_forced_current_footprint_shape_invalid": bool(current_forced_invalid),
            "voxel_outside_synthetic_free_provenance_shape_invalid": bool(synthetic_invalid),
            "voxel_outside_current_footprint_exclusion_radius_m": float(
                getattr(cfg, "outside_current_footprint_exclusion_radius_m", 0.0)
            ),
            "voxel_outside_require_disconnected_from_robot_navigation_component": bool(require_disconnected),
            "voxel_outside_robot_component_available": bool(robot_component_available),
            "voxel_outside_robot_component_cells": int(robot_component_cells),
            "voxel_outside_connected_to_robot_rejected_cells": int(np.count_nonzero(connected_to_robot_rejected)),
            "voxel_outside_overlap_robot_component_cells": int(outside_overlap_robot_component),
            "voxel_outside_robot_seed_grid": list(robot_seed_grid) if robot_seed_grid is not None else None,
            "voxel_outside_robot_seed_snapped_grid": list(robot_seed_snapped_grid) if robot_seed_snapped_grid is not None else None,
            "voxel_outside_robot_seed_snap_distance_cells": robot_seed_snap_distance_cells,
            "voxel_outside_robot_snap_to_traversible_max_radius_cells": int(
                getattr(cfg, "outside_robot_snap_to_traversible_max_radius_cells", 20)
            ),
            "voxel_outside_connectivity_source": str(getattr(cfg, "outside_connectivity_source", "astar_traversible_pre_outside")),
            "voxel_outside_connectivity_allow_diagonal": bool(getattr(cfg, "outside_connectivity_allow_diagonal", True)),
            "voxel_outside_connectivity_prevent_diagonal_corner_cutting": bool(
                getattr(cfg, "outside_connectivity_prevent_diagonal_corner_cutting", True)
            ),
            "voxel_outside_hard_clear_robot_component_scores": bool(getattr(cfg, "outside_hard_clear_robot_component_scores", True)),
            "voxel_outside_fail_closed_if_robot_component_unavailable": bool(
                getattr(cfg, "outside_fail_closed_if_robot_component_unavailable", True)
            ),
            "voxel_outside_score_threshold": int(threshold),
            "voxel_outside_occupied_check_z_min_m": float(outside_check_z_min_m),
            "voxel_outside_occupied_check_z_max_m": float(outside_check_z_max_m),
            "voxel_outside_z_bin_count": int(z_idx.size),
        }
        if bool(getattr(cfg, "outside_debug_enabled", True)):
            debug.update(
                {
                    "voxel_outside_xy": self.outside_xy.astype(bool),
                    "voxel_outside_candidate_xy": candidate.astype(bool),
                    "voxel_outside_candidate_base_xy": candidate_base.astype(bool),
                    "voxel_outside_disconnected_candidate_xy": candidate.astype(bool),
                    "voxel_outside_pre_nav_free_xy": nav_free.astype(bool),
                    "voxel_outside_pre_traversible_xy": (
                        trav.astype(bool) if trav is not None else np.zeros(shape, dtype=bool)
                    ),
                    "voxel_outside_robot_component_xy": robot_component.astype(bool),
                    "voxel_outside_connected_to_robot_rejected_xy": connected_to_robot_rejected.astype(bool),
                    "voxel_outside_full_height_no_occupied_xy": full_height_no_occupied_xy.astype(bool),
                    "voxel_outside_floor_frustum_seen_xy": floor_seen_xy.astype(bool),
                    "voxel_outside_holdout_floor_unseen_xy": holdout_floor_unseen.astype(bool),
                    "voxel_outside_has_occupied_xy": has_occupied_xy.astype(bool),
                    "voxel_outside_score_xy": self.outside_score_xy.astype(np.uint8),
                    "voxel_outside_excluded_synthetic_free_xy": combined_synthetic_excluded.astype(bool),
                    "voxel_outside_synthetic_free_provenance_xy": synthetic_excluded.astype(bool),
                }
            )
        self.last_outside_debug = dict(debug)
        return self.outside_xy.astype(bool), dict(debug)

    def mark_sensor_range_voxels_array(self, voxels: np.ndarray | Iterable[Sequence[int]]) -> tuple[int, np.ndarray]:
        if not bool(getattr(self.config, "sensor_range_tracking_enabled", True)):
            return 0, np.zeros(0, dtype=np.int64)
        if isinstance(voxels, np.ndarray):
            arr = np.asarray(voxels, dtype=np.int64)
        else:
            arr = np.asarray(list(voxels), dtype=np.int64)
        if arr.size == 0:
            return 0, np.zeros(0, dtype=np.int64)
        arr = arr.reshape(-1, 3)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        valid = (z >= 0) & (z < self.z_bin_count) & (r >= 0) & (r < self.state.shape[1]) & (c >= 0) & (c < self.state.shape[2])
        if not np.any(valid):
            return 0, np.zeros(0, dtype=np.int64)
        arr = np.unique(arr[valid], axis=0)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        delta = max(0, int(getattr(self.config, "sensor_range_count_delta", 1)))
        max_value = int(np.clip(int(getattr(self.config, "sensor_range_count_max", 255)), 0, 255))
        if delta <= 0 or max_value <= 0:
            return int(arr.shape[0]), self.flat_indices_from_voxels(arr)
        current = self.sensor_range_count[z, r, c].astype(np.uint16) + int(delta)
        self.sensor_range_count[z, r, c] = np.minimum(current, int(max_value)).astype(np.uint8)
        return int(arr.shape[0]), self.flat_indices_from_voxels(arr)

    def mark_sensor_range_flat_indices(self, flat_indices: np.ndarray | Iterable[int]) -> int:
        if not bool(getattr(self.config, "sensor_range_tracking_enabled", True)):
            return 0
        idx = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return 0
        idx = np.unique(idx[(idx >= 0) & (idx < self.sensor_range_count.size)])
        if idx.size == 0:
            return 0
        delta = max(0, int(getattr(self.config, "sensor_range_count_delta", 1)))
        max_value = int(np.clip(int(getattr(self.config, "sensor_range_count_max", 255)), 0, 255))
        if delta <= 0 or max_value <= 0:
            return int(idx.size)
        flat = self.sensor_range_count.reshape(-1)
        updated = flat[idx].astype(np.uint16) + int(delta)
        flat[idx] = np.minimum(updated, int(max_value)).astype(np.uint8)
        return int(idx.size)

    def decay_sensor_range_count(self) -> int:
        if not bool(getattr(self.config, "sensor_range_tracking_enabled", True)):
            return 0
        decay = max(0, int(getattr(self.config, "sensor_range_count_decay_per_update", 0)))
        if decay <= 0:
            return 0
        before = int(np.count_nonzero(self.sensor_range_count))
        values = self.sensor_range_count.astype(np.int16) - int(decay)
        self.sensor_range_count[:, :, :] = np.maximum(values, 0).astype(np.uint8)
        return int(before - np.count_nonzero(self.sensor_range_count))

    def mark_free_voxels_array(self, voxels: np.ndarray | Iterable[Sequence[int]]) -> tuple[int, np.ndarray]:
        return self._add_logodds_array(voxels, int(self.config.free_logodds_delta))

    def mark_occupied_voxels_array(self, voxels: np.ndarray | Iterable[Sequence[int]]) -> tuple[int, np.ndarray]:
        return self._add_logodds_array(voxels, int(self.config.occupied_logodds_delta))

    def force_free_voxels(self, voxels: np.ndarray | Iterable[Sequence[int]]) -> tuple[int, np.ndarray]:
        if isinstance(voxels, np.ndarray):
            arr = np.asarray(voxels, dtype=np.int64)
        else:
            arr = np.asarray(list(voxels), dtype=np.int64)
        if arr.size == 0:
            return 0, np.zeros(0, dtype=np.int64)
        arr = arr.reshape(-1, 3)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        valid = (z >= 0) & (z < self.z_bin_count) & (r >= 0) & (r < self.state.shape[1]) & (c >= 0) & (c < self.state.shape[2])
        if not np.any(valid):
            return 0, np.zeros(0, dtype=np.int64)
        arr = np.unique(arr[valid], axis=0)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        value = min(
            int(self.config.free_logodds_threshold),
            int(self.config.free_logodds_delta),
            int(getattr(self.config, "occupied_to_free_logodds_threshold", self.config.free_logodds_threshold)),
        )
        self.log_odds[z, r, c] = np.minimum(self.log_odds[z, r, c].astype(np.int32), int(value)).astype(np.int16)
        return int(arr.shape[0]), self.flat_indices_from_voxels(arr)

    def refresh_state(self) -> None:
        # Neutral log-odds preserves the previous known state; unknown means never observed.
        free_threshold = int(self.config.free_logodds_threshold)
        occupied_to_free_threshold = min(
            free_threshold,
            int(getattr(self.config, "occupied_to_free_logodds_threshold", free_threshold)),
        )
        previous_occupied = self.state == int(VOXEL_OCCUPIED)
        free = self.log_odds <= free_threshold
        free_allowed = free & (~previous_occupied | (self.log_odds <= occupied_to_free_threshold))
        self.state[free_allowed] = int(VOXEL_FREE)
        self.state[self.log_odds >= int(self.config.occupied_logodds_threshold)] = int(VOXEL_OCCUPIED)

    def refresh_state_indices(self, flat_indices: np.ndarray | Iterable[int]) -> int:
        idx = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return 0
        idx = np.unique(idx[(idx >= 0) & (idx < self.state.size)])
        if idx.size == 0:
            return 0
        log = self.log_odds.reshape(-1)
        state = self.state.reshape(-1)
        free_threshold = int(self.config.free_logodds_threshold)
        occupied_to_free_threshold = min(
            free_threshold,
            int(getattr(self.config, "occupied_to_free_logodds_threshold", free_threshold)),
        )
        previous_occupied = state[idx] == int(VOXEL_OCCUPIED)
        free = log[idx] <= free_threshold
        free_allowed = free & (~previous_occupied | (log[idx] <= occupied_to_free_threshold))
        occupied = log[idx] >= int(self.config.occupied_logodds_threshold)
        state[idx[free_allowed]] = int(VOXEL_FREE)
        state[idx[occupied]] = int(VOXEL_OCCUPIED)
        return int(idx.size)

    def flat_indices_from_voxels(self, voxels: np.ndarray | Iterable[Sequence[int]]) -> np.ndarray:
        if isinstance(voxels, np.ndarray):
            arr = np.asarray(voxels, dtype=np.int64)
        else:
            arr = np.asarray(list(voxels), dtype=np.int64)
        if arr.size == 0:
            return np.zeros(0, dtype=np.int64)
        arr = arr.reshape(-1, 3)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        valid = (z >= 0) & (z < self.z_bin_count) & (r >= 0) & (r < self.state.shape[1]) & (c >= 0) & (c < self.state.shape[2])
        if not np.any(valid):
            return np.zeros(0, dtype=np.int64)
        z = z[valid]
        r = r[valid]
        c = c[valid]
        return z.astype(np.int64) * int(self.state.shape[1] * self.state.shape[2]) + r.astype(np.int64) * int(self.state.shape[2]) + c.astype(np.int64)

    def set_active_z_from_ceiling(self, ceiling_height_m: float | None, status: str = "unknown") -> None:
        self.ceiling_height_m = None if ceiling_height_m is None or not np.isfinite(ceiling_height_m) else float(ceiling_height_m)
        self.ceiling_estimate_status = str(status or "unknown")
        mode = str(self.config.active_z_max_mode).strip().lower()
        if mode in {"ceiling_ratio", "ceiling_90pct", "ceiling_80pct"} and self.ceiling_height_m is not None:
            active = float(self.ceiling_height_m) * float(self.config.active_z_max_ceiling_ratio)
            cap = float(getattr(self.config, "active_z_max_cap_m", 0.0))
            if np.isfinite(cap) and cap > 0.0:
                active = min(float(active), cap)
        else:
            active = float(_active_z_cap_or_storage_max(self.config))
        self.active_z_min_m = float(self.config.active_z_min_m)
        self.active_z_max_m = float(np.clip(active, self.active_z_min_m, self.z_max_m))

    def active_z_indices(self, *, z_min_m: float | None = None, z_max_m: float | None = None) -> np.ndarray:
        zmin = self.active_z_min_m if z_min_m is None else float(z_min_m)
        zmax = self.active_z_max_m if z_max_m is None else float(z_max_m)
        if zmax is None:
            zmax = float(_active_z_cap_or_storage_max(self.config))
        z = self.z_centers_m
        idx = np.nonzero((z >= float(zmin)) & (z <= float(zmax)))[0].astype(np.int32)
        if idx.size == 0:
            fallback = self.z_index_for_height(float(zmin))
            if fallback is not None:
                idx = np.asarray([fallback], dtype=np.int32)
        return idx

    def z_indices_for_world_range(self, z_min_m: float, z_max_m: float) -> np.ndarray:
        """Return storage z-bin indices in a relative-height range without active-z clamping."""
        zmin = float(z_min_m)
        zmax = float(z_max_m)
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax < zmin:
            return np.zeros(0, dtype=np.int32)
        z = self.z_centers_m
        return np.nonzero((z >= zmin) & (z <= zmax))[0].astype(np.int32)

    def z_indices_for_height_range(self, z_min_m: float, z_max_m: float) -> np.ndarray:
        """Alias for full-storage relative-height z-bin lookup."""
        return self.z_indices_for_world_range(z_min_m, z_max_m)

    def project_navigation(
        self,
        *,
        obstacle_z_min_m: float = 0.20,
        obstacle_z_max_m: float = 0.90,
        free_z_min_m: float = 0.10,
        free_z_max_m: float = 0.90,
        min_free_voxels: int = 1,
        nav_endpoint_count_xy: np.ndarray | None = None,
        config: NavigationProjectionConfig | Mapping[str, object] | None = None,
        occupied_any_voxel_wins: bool = True,
        occupied_use_endpoint_hysteresis: bool = True,
        occupied_endpoint_count_threshold: int = 1,
        occupied_endpoint_decay_per_free_ray: int = 1,
        occupied_endpoint_increment: int = 2,
        occupied_endpoint_xy_splat_radius_cells: int = 0,
        occupied_endpoint_z_splat_radius_cells: int = 0,
        occupied_close_radius_cells: int = 1,
        occupied_fill_small_holes_max_area_cells: int = 4,
        occupied_priority_over_free: bool = True,
        unknown_preserve_when_no_observation: bool = True,
        debug_navigation_projection_layers: bool = True,
        incremental: bool = True,
        force_full: bool = False,
        dirty_rc_flags: np.ndarray | None = None,
        projection_step: int | None = None,
        full_refresh_interval_steps: int | None = None,
        dirty_dilation_radius_cells: int | None = None,
        local_morphology_enabled: bool | None = None,
        full_morphology_only_on_replan: bool | None = None,
        incremental_enabled: bool | None = None,
        full_refresh_on_frontier_update: bool | None = None,
    ) -> NavigationProjection:
        started_at = time.perf_counter()
        cfg = (
            config
            if isinstance(config, NavigationProjectionConfig)
            else NavigationProjectionConfig.from_mapping(
                config,
                obstacle_z_min_m=obstacle_z_min_m,
                obstacle_z_max_m=obstacle_z_max_m,
                free_z_min_m=free_z_min_m,
                free_z_max_m=free_z_max_m,
                min_free_voxels=min_free_voxels,
                occupied_any_voxel_wins=occupied_any_voxel_wins,
                occupied_use_endpoint_hysteresis=occupied_use_endpoint_hysteresis,
                occupied_endpoint_count_threshold=occupied_endpoint_count_threshold,
                occupied_endpoint_decay_per_free_ray=occupied_endpoint_decay_per_free_ray,
                occupied_endpoint_increment=occupied_endpoint_increment,
                occupied_endpoint_xy_splat_radius_cells=occupied_endpoint_xy_splat_radius_cells,
                occupied_endpoint_z_splat_radius_cells=occupied_endpoint_z_splat_radius_cells,
                occupied_close_radius_cells=occupied_close_radius_cells,
                occupied_fill_small_holes_max_area_cells=occupied_fill_small_holes_max_area_cells,
                occupied_priority_over_free=occupied_priority_over_free,
                unknown_preserve_when_no_observation=unknown_preserve_when_no_observation,
                debug_navigation_projection_layers=debug_navigation_projection_layers,
                incremental_enabled=incremental_enabled,
                full_refresh_on_frontier_update=full_refresh_on_frontier_update,
                full_refresh_interval_steps=full_refresh_interval_steps,
                dirty_dilation_radius_cells=dirty_dilation_radius_cells,
                local_morphology_enabled=local_morphology_enabled,
                full_morphology_only_on_replan=full_morphology_only_on_replan,
            )
        )
        occ_idx = self.active_z_indices(z_min_m=float(cfg.obstacle_z_min_m), z_max_m=float(cfg.obstacle_z_max_m))
        free_idx = self.active_z_indices(z_min_m=float(cfg.free_z_min_m), z_max_m=float(cfg.free_z_max_m))
        union_idx = np.unique(np.concatenate([occ_idx, free_idx])).astype(np.int32) if occ_idx.size or free_idx.size else np.asarray([], dtype=np.int32)
        projection_backend = "numpy"
        if nav_endpoint_count_xy is not None:
            endpoint_count = np.asarray(nav_endpoint_count_xy)
            if endpoint_count.shape != self.shape:
                raise ValueError("nav_endpoint_count_xy shape %s does not match grid shape %s" % (endpoint_count.shape, self.shape))
        else:
            endpoint_count = None
        cache = self._navigation_projection_cache
        step_i = -1 if projection_step is None else int(projection_step)
        interval = max(0, int(cfg.full_refresh_interval_steps))
        force_interval = bool(interval > 0 and cache is not None and cache.initialized and step_i >= 0 and (step_i - int(cache.last_full_refresh_step)) >= interval)
        use_incremental = (
            bool(incremental)
            and bool(cfg.incremental_enabled)
            and step_i >= 0
            and not bool(force_full)
            and not bool(force_interval)
            and cache is not None
            and bool(cache.initialized)
        )
        raw_dirty_flags = dirty_rc_flags
        if raw_dirty_flags is None:
            raw_dirty_flags = self.last_dirty_rc_flags
        dirty_indices = np.zeros(0, dtype=np.int64)
        dirty_source_count = 0
        if use_incremental and raw_dirty_flags is not None:
            flags = np.asarray(raw_dirty_flags, dtype=np.uint8).reshape(-1)
            if flags.size == int(self.shape[0] * self.shape[1]):
                dirty_source_count = int(np.count_nonzero(flags))
                if dirty_source_count > 0:
                    flags_2d = flags.reshape(self.shape).astype(bool)
                    radius = max(0, int(cfg.dirty_dilation_radius_cells))
                    if radius > 0:
                        flags_2d = _dilate_bool(flags_2d, radius)
                    dirty_indices = np.flatnonzero(flags_2d.reshape(-1)).astype(np.int64)
        if use_incremental and dirty_indices.size > 0:
            projection_backend = "numba_dirty_column"
            try:
                from voxroom_online.isaac_runtime.mapping.voxel_projection_numba import project_navigation_dirty_columns

                project_navigation_dirty_columns(
                    self.state,
                    endpoint_count,
                    dirty_indices,
                    occ_z_indices=occ_idx,
                    free_z_indices=free_idx,
                    endpoint_threshold=int(cfg.occupied_endpoint_count_threshold),
                    min_free_voxels=int(cfg.min_free_voxels),
                    occupied_any_voxel_wins=bool(cfg.occupied_any_voxel_wins),
                    occupied_use_endpoint_hysteresis=bool(cfg.occupied_use_endpoint_hysteresis),
                    out_occupied_from_voxel_flat=cache.occupied_from_voxel.reshape(-1),
                    out_occupied_from_endpoint_flat=cache.occupied_from_endpoint.reshape(-1),
                    out_free_raw_flat=cache.free_raw.reshape(-1),
                    out_observed_from_voxel_flat=cache.observed_from_voxel.reshape(-1),
                )
            except Exception:
                projection_backend = "numpy_dirty_column"
                state_flat = self.state.reshape(int(self.state.shape[0]), -1)
                for rc in dirty_indices:
                    rc_i = int(rc)
                    if rc_i < 0 or rc_i >= state_flat.shape[1]:
                        continue
                    occ_values = state_flat[occ_idx, rc_i] if occ_idx.size else np.zeros(0, dtype=np.uint8)
                    free_values = state_flat[free_idx, rc_i] if free_idx.size else np.zeros(0, dtype=np.uint8)
                    union_values = state_flat[union_idx, rc_i] if union_idx.size else np.zeros(0, dtype=np.uint8)
                    cache.occupied_from_voxel.reshape(-1)[rc_i] = bool(occ_values.size and np.any(occ_values == int(VOXEL_OCCUPIED)))
                    cache.free_raw.reshape(-1)[rc_i] = bool(free_values.size and int(np.count_nonzero(free_values == int(VOXEL_FREE))) >= max(1, int(cfg.min_free_voxels)))
                    cache.observed_from_voxel.reshape(-1)[rc_i] = bool(union_values.size and np.any(union_values != int(VOXEL_UNKNOWN)))
                    cache.occupied_from_endpoint.reshape(-1)[rc_i] = bool(
                        endpoint_count is not None
                        and bool(cfg.occupied_use_endpoint_hysteresis)
                        and int(np.asarray(endpoint_count).reshape(-1)[rc_i]) >= max(1, int(cfg.occupied_endpoint_count_threshold))
                    )
            occupied_from_voxel = cache.occupied_from_voxel.astype(bool, copy=False)
            occupied_from_endpoint = cache.occupied_from_endpoint.astype(bool, copy=False)
            free_raw = cache.free_raw.astype(bool, copy=False)
            observed_from_voxel = cache.observed_from_voxel.astype(bool, copy=False)
            projection_mode = "incremental"
            full_refresh = False
        elif use_incremental and dirty_indices.size == 0:
            projection_backend = "cache_clean"
            occupied_from_voxel = cache.occupied_from_voxel.astype(bool, copy=False)
            occupied_from_endpoint = cache.occupied_from_endpoint.astype(bool, copy=False)
            free_raw = cache.free_raw.astype(bool, copy=False)
            observed_from_voxel = cache.observed_from_voxel.astype(bool, copy=False)
            projection_mode = "cached_no_dirty"
            full_refresh = False
        else:
            projection_mode = "full"
            full_refresh = True
            projection_backend = "numpy"
            try:
                from voxroom_online.isaac_runtime.mapping.voxel_projection_numba import project_navigation_columns

                occupied_from_voxel, occupied_from_endpoint, free_raw, observed_from_voxel = project_navigation_columns(
                    self.state,
                    endpoint_count,
                    occ_z_indices=occ_idx,
                    free_z_indices=free_idx,
                    endpoint_threshold=int(cfg.occupied_endpoint_count_threshold),
                    min_free_voxels=int(cfg.min_free_voxels),
                    occupied_any_voxel_wins=bool(cfg.occupied_any_voxel_wins),
                    occupied_use_endpoint_hysteresis=bool(cfg.occupied_use_endpoint_hysteresis),
                )
                projection_backend = "numba_column"
            except Exception:
                if occ_idx.size and bool(cfg.occupied_any_voxel_wins):
                    occupied_from_voxel = np.any(self.state[occ_idx] == int(VOXEL_OCCUPIED), axis=0)
                else:
                    occupied_from_voxel = np.zeros(self.shape, dtype=bool)
                if free_idx.size:
                    free_count = np.sum(self.state[free_idx] == int(VOXEL_FREE), axis=0)
                    free_raw = free_count >= max(1, int(cfg.min_free_voxels))
                else:
                    free_raw = np.zeros(self.shape, dtype=bool)
                if endpoint_count is not None and bool(cfg.occupied_use_endpoint_hysteresis):
                    occupied_from_endpoint = endpoint_count.astype(np.uint16) >= max(1, int(cfg.occupied_endpoint_count_threshold))
                else:
                    occupied_from_endpoint = np.zeros(self.shape, dtype=bool)
                if union_idx.size:
                    observed_from_voxel = np.any(self.state[union_idx] != int(VOXEL_UNKNOWN), axis=0)
                else:
                    observed_from_voxel = np.zeros(self.shape, dtype=bool)
        occupied_raw = np.asarray(occupied_from_voxel | occupied_from_endpoint, dtype=bool)
        occupied_closed = occupied_raw
        morphology_enabled = full_refresh or not bool(cfg.full_morphology_only_on_replan)
        if int(cfg.occupied_close_radius_cells) > 0 and bool(morphology_enabled):
            occupied_closed = _binary_close_disk(occupied_closed, int(cfg.occupied_close_radius_cells))
        hole_filled_mask = np.zeros(self.shape, dtype=bool)
        if int(cfg.occupied_fill_small_holes_max_area_cells) > 0 and bool(morphology_enabled):
            occupied_closed, hole_filled_mask = _fill_small_false_holes(
                occupied_closed,
                max_area_cells=int(cfg.occupied_fill_small_holes_max_area_cells),
            )
        occupied = np.asarray(occupied_closed, dtype=bool)
        free_suppressed_by_occupied = np.asarray(free_raw, dtype=bool) & occupied
        if bool(cfg.occupied_priority_over_free):
            free = np.asarray(free_raw, dtype=bool) & ~occupied
        else:
            free = np.asarray(free_raw, dtype=bool)
            occupied = occupied & ~free
        observed = np.asarray(observed_from_voxel | free | occupied, dtype=bool)
        unknown = ~observed
        free &= ~unknown
        occupied &= ~unknown
        cache_dirty = np.zeros(int(self.shape[0] * self.shape[1]), dtype=np.uint8)
        if raw_dirty_flags is not None and np.asarray(raw_dirty_flags).reshape(-1).size == cache_dirty.size:
            cache_dirty[:] = np.asarray(raw_dirty_flags, dtype=np.uint8).reshape(-1)
        self._navigation_projection_cache = VoxelNavigationProjectionCache(
            free=free.astype(bool),
            occupied=occupied.astype(bool),
            observed=observed.astype(bool),
            unknown=unknown.astype(bool),
            occupied_from_voxel=np.asarray(occupied_from_voxel, dtype=np.uint8),
            occupied_from_endpoint=np.asarray(occupied_from_endpoint, dtype=np.uint8),
            free_raw=np.asarray(free_raw, dtype=np.uint8),
            observed_from_voxel=np.asarray(observed_from_voxel, dtype=np.uint8),
            dirty_rc_flags=cache_dirty,
            initialized=True,
            last_full_refresh_step=step_i if full_refresh and step_i >= 0 else (
                int(cache.last_full_refresh_step) if cache is not None and cache.initialized else -1
            ),
        )
        debug = {
            "voxel_nav_obstacle_z_min_m": float(cfg.obstacle_z_min_m),
            "voxel_nav_obstacle_z_max_m": float(cfg.obstacle_z_max_m),
            "voxel_nav_free_z_min_m": float(cfg.free_z_min_m),
            "voxel_nav_free_z_max_m": float(cfg.free_z_max_m),
            "voxel_nav_min_free_voxels": int(cfg.min_free_voxels),
            "voxel_nav_occupied_any_voxel_wins": bool(cfg.occupied_any_voxel_wins),
            "voxel_nav_occupied_use_endpoint_hysteresis": bool(cfg.occupied_use_endpoint_hysteresis),
            "voxel_nav_occupied_endpoint_count_threshold": int(cfg.occupied_endpoint_count_threshold),
            "voxel_nav_occupied_endpoint_decay_per_free_ray": int(cfg.occupied_endpoint_decay_per_free_ray),
            "voxel_nav_occupied_endpoint_increment": int(cfg.occupied_endpoint_increment),
            "voxel_nav_occupied_endpoint_xy_splat_radius_cells": int(cfg.occupied_endpoint_xy_splat_radius_cells),
            "voxel_nav_occupied_close_radius_cells": int(cfg.occupied_close_radius_cells),
            "voxel_nav_occupied_fill_small_holes_max_area_cells": int(cfg.occupied_fill_small_holes_max_area_cells),
            "voxel_nav_obstacle_z_bins": int(occ_idx.size),
            "voxel_nav_free_z_bins": int(free_idx.size),
            "voxel_nav_free_cells": int(np.count_nonzero(free)),
            "voxel_nav_occupied_cells": int(np.count_nonzero(occupied)),
            "voxel_nav_occupied_from_voxel_cells": int(np.count_nonzero(occupied_from_voxel)),
            "voxel_nav_occupied_from_endpoint_cells": int(np.count_nonzero(occupied_from_endpoint)),
            "voxel_nav_occupied_raw_cells": int(np.count_nonzero(occupied_raw)),
            "voxel_nav_occupied_closed_cells": int(np.count_nonzero(occupied_closed)),
            "voxel_nav_free_raw_cells": int(np.count_nonzero(free_raw)),
            "voxel_nav_free_suppressed_by_occupied_cells": int(np.count_nonzero(free_suppressed_by_occupied)),
            "voxel_nav_occupied_hole_filled_cells": int(np.count_nonzero(hole_filled_mask)),
            "voxel_nav_observed_cells": int(np.count_nonzero(observed)),
            "voxel_nav_unknown_cells": int(np.count_nonzero(unknown)),
            "voxel_project_navigation_backend": str(projection_backend),
            "voxel_project_navigation_mode": str(projection_mode),
            "voxel_project_navigation_force_full": bool(force_full),
            "voxel_project_navigation_full_refresh": bool(full_refresh),
            "voxel_project_navigation_incremental_enabled": bool(cfg.incremental_enabled),
            "voxel_project_navigation_dirty_rc_count": int(dirty_source_count),
            "voxel_project_navigation_dirty_projected_rc_count": int(dirty_indices.size),
            "voxel_project_navigation_dirty_dilation_radius_cells": int(cfg.dirty_dilation_radius_cells),
            "voxel_project_navigation_morphology_applied": bool(morphology_enabled),
            "voxel_project_navigation_last_full_refresh_step": int(self._navigation_projection_cache.last_full_refresh_step),
            "voxel_project_navigation_ms": float((time.perf_counter() - started_at) * 1000.0),
        }
        if bool(cfg.debug_navigation_projection_layers):
            debug.update(
                {
                    "voxel_nav_occupied_from_voxel_xy": occupied_from_voxel.astype(bool),
                    "voxel_nav_occupied_from_endpoint_xy": occupied_from_endpoint.astype(bool),
                    "voxel_nav_occupied_raw_xy": occupied_raw.astype(bool),
                    "voxel_nav_occupied_closed_xy": occupied_closed.astype(bool),
                    "voxel_nav_free_raw_xy": np.asarray(free_raw, dtype=bool),
                    "voxel_nav_free_suppressed_by_occupied_xy": free_suppressed_by_occupied.astype(bool),
                    "voxel_nav_final_free_xy": free.astype(bool),
                    "voxel_nav_final_occupied_xy": occupied.astype(bool),
                    "voxel_nav_final_unknown_xy": unknown.astype(bool),
                }
            )
        self.last_navigation_debug = dict(debug)
        return NavigationProjection(
            free=free.astype(bool),
            occupied=occupied.astype(bool),
            observed=observed.astype(bool),
            unknown=unknown.astype(bool),
            debug=debug,
        )

    def invalidate_navigation_projection_cache(self, *, reason: str = "") -> None:
        self._navigation_projection_cache = None
        debug = dict(self.last_navigation_debug)
        debug["voxel_project_navigation_cache_invalidated"] = True
        debug["voxel_project_navigation_cache_invalidated_reason"] = str(reason)
        self.last_navigation_debug = debug

    def to_debug_dict(self) -> dict[str, object]:
        z_count, height, width = self.state.shape
        debug = {
            "voxel_backend": "voxel_occupancy_door_wall_v33",
            "voxel_grid_enabled": bool(self.config.enabled),
            "voxel_grid_shape_zyx": [int(z_count), int(height), int(width)],
            "voxel_z_min_m": float(self.z_min_m),
            "voxel_z_max_m": float(self.z_max_m),
            "voxel_z_resolution_m": float(self.z_resolution_m),
            "voxel_active_z_min_m": float(self.active_z_min_m),
            "voxel_active_z_max_m": None if self.active_z_max_m is None else float(self.active_z_max_m),
            "voxel_ceiling_height_m": None if self.ceiling_height_m is None else float(self.ceiling_height_m),
            "voxel_ceiling_estimate_status": str(self.ceiling_estimate_status),
            "voxel_active_z_bin_count": int(self.active_z_indices().size),
            "voxel_integration_backend_requested": str(self.config.integration_backend),
            "voxel_python_debug_backend_allowed": bool(self.config.python_debug_backend_allowed),
            "voxel_state_free_count_3d": int(np.count_nonzero(self.state == int(VOXEL_FREE))),
            "voxel_state_occupied_count_3d": int(np.count_nonzero(self.state == int(VOXEL_OCCUPIED))),
            "voxel_state_unknown_count_3d": int(np.count_nonzero(self.state == int(VOXEL_UNKNOWN))),
            "voxel_state_conflict_count_3d": int(np.count_nonzero(self.state == int(VOXEL_CONFLICT))),
            "voxel_sensor_range_tracking_enabled": bool(getattr(self.config, "sensor_range_tracking_enabled", True)),
            "voxel_sensor_range_count_nonzero_3d": int(np.count_nonzero(self.sensor_range_count)),
            "voxel_sensor_range_count_threshold": int(getattr(self.config, "sensor_range_count_threshold", 1)),
            "voxel_sensor_range_mark_effective_frustum_enabled": bool(getattr(self.config, "sensor_range_mark_effective_frustum_enabled", True)),
            "voxel_sensor_range_endpoint_column_enabled": bool(getattr(self.config, "sensor_range_mark_endpoint_column_enabled", True)),
            "voxel_sensor_range_mark_ray_samples_enabled": bool(getattr(self.config, "sensor_range_mark_ray_samples_enabled", True)),
            "voxel_sensor_range_behind_endpoint_margin_m": float(getattr(self.config, "sensor_range_behind_endpoint_margin_m", 0.0)),
            "voxel_floor_frustum_seen_cells": int(np.count_nonzero(self.floor_frustum_seen_count_xy)),
            "voxel_outside_confirmed_cells": int(np.count_nonzero(self.outside_xy)),
            "voxel_outside_score_nonzero_cells": int(np.count_nonzero(self.outside_score_xy)),
            "voxel_outside_boundary_enabled": bool(getattr(self.config, "outside_boundary_enabled", True)),
        }
        debug.update(self.last_integration_stats.to_dict())
        debug.update({key: value for key, value in dict(self.last_navigation_debug).items() if not isinstance(value, np.ndarray)})
        debug.update({key: value for key, value in dict(self.last_floor_frustum_debug).items() if not isinstance(value, np.ndarray)})
        debug.update({key: value for key, value in dict(self.last_outside_debug).items() if not isinstance(value, np.ndarray)})
        return debug

    def _add_logodds(self, voxels: Iterable[Sequence[int]], delta: int) -> int:
        count, _changed = self._add_logodds_array(voxels, int(delta))
        return int(count)

    def _add_logodds_array(self, voxels: np.ndarray | Iterable[Sequence[int]], delta: int) -> tuple[int, np.ndarray]:
        if isinstance(voxels, np.ndarray):
            arr = np.asarray(voxels, dtype=np.int64)
        else:
            arr = np.asarray(list(voxels), dtype=np.int64)
        if arr.size == 0:
            return 0, np.zeros(0, dtype=np.int64)
        arr = arr.reshape(-1, 3)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        valid = (z >= 0) & (z < self.z_bin_count) & (r >= 0) & (r < self.state.shape[1]) & (c >= 0) & (c < self.state.shape[2])
        if not np.any(valid):
            return 0, np.zeros(0, dtype=np.int64)
        z = z[valid]
        r = r[valid]
        c = c[valid]
        np.add.at(self.log_odds, (z, r, c), int(delta))
        np.clip(self.log_odds, int(self.config.logodds_min), int(self.config.logodds_max), out=self.log_odds)
        changed = z.astype(np.int64) * int(self.state.shape[1] * self.state.shape[2]) + r.astype(np.int64) * int(self.state.shape[2]) + c.astype(np.int64)
        return int(z.size), np.unique(changed)

    def _endpoint_splat(self, voxel: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        z0, r0, c0 = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        rz = max(0, int(self.config.endpoint_splat_z_radius_cells))
        rxy = max(0, int(self.config.endpoint_splat_xy_radius_cells))
        out: list[tuple[int, int, int]] = []
        for dz in range(-rz, rz + 1):
            for dr in range(-rxy, rxy + 1):
                for dc in range(-rxy, rxy + 1):
                    z, r, c = z0 + dz, r0 + dr, c0 + dc
                    if 0 <= z < self.z_bin_count and 0 <= r < self.state.shape[1] and 0 <= c < self.state.shape[2]:
                        out.append((z, r, c))
        return out

    def _sensor_endpoint_column_voxels(self, voxel: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        _z0, r0, c0 = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        radius = max(0, int(getattr(self.config, "sensor_range_endpoint_column_xy_radius_cells", 0)))
        if bool(getattr(self.config, "sensor_range_mark_active_z_only", True)):
            z_values = [int(v) for v in self.active_z_indices().tolist()]
        else:
            z_values = list(range(int(self.z_bin_count)))
        out: list[tuple[int, int, int]] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < self.state.shape[1] and 0 <= c < self.state.shape[2]):
                    continue
                out.extend((int(z), int(r), int(c)) for z in z_values)
        return out

    def _clip_segment_to_volume(self, p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        bounds = np.asarray([self.z_bin_count, self.state.shape[1], self.state.shape[2]], dtype=np.float64)
        direction = p1 - p0
        t_min = 0.0
        t_max = 1.0
        eps = max(1.0e-12, float(getattr(self.config, "ray_traversal_tie_epsilon", 1.0e-6)))
        for axis in range(3):
            lower = 0.0
            upper = max(0.0, float(bounds[axis]) - eps)
            if abs(float(direction[axis])) < eps:
                if p0[axis] < lower or p0[axis] > upper:
                    return None
                continue
            inv = 1.0 / float(direction[axis])
            ta = (lower - float(p0[axis])) * inv
            tb = (upper - float(p0[axis])) * inv
            if ta > tb:
                ta, tb = tb, ta
            t_min = max(t_min, ta)
            t_max = min(t_max, tb)
            if t_min > t_max:
                return None
        q0 = p0 + direction * t_min
        q1 = p0 + direction * t_max
        q0 = np.minimum(np.maximum(q0, 0.0), bounds - eps)
        q1 = np.minimum(np.maximum(q1, 0.0), bounds - eps)
        return q0, q1

    def _dda_voxels(self, p0: np.ndarray, p1: np.ndarray) -> list[tuple[int, int, int]]:
        bounds = np.asarray([self.z_bin_count, self.state.shape[1], self.state.shape[2]], dtype=np.int32)
        direction = p1 - p0
        cell = np.floor(p0).astype(np.int32)
        target = np.floor(p1).astype(np.int32)
        cell = np.minimum(np.maximum(cell, 0), bounds - 1)
        target = np.minimum(np.maximum(target, 0), bounds - 1)
        step = np.sign(direction).astype(np.int32)
        t_delta = np.full(3, np.inf, dtype=np.float64)
        t_max = np.full(3, np.inf, dtype=np.float64)
        eps = max(1.0e-12, float(getattr(self.config, "ray_traversal_tie_epsilon", 1.0e-6)))
        for axis in range(3):
            d = float(direction[axis])
            if abs(d) < eps:
                continue
            t_delta[axis] = abs(1.0 / d)
            if step[axis] > 0:
                boundary = math.floor(float(p0[axis])) + 1.0
                t_max[axis] = (boundary - float(p0[axis])) / d
            elif step[axis] < 0:
                boundary = math.floor(float(p0[axis]))
                t_max[axis] = (float(p0[axis]) - boundary) / (-d)
        out: list[tuple[int, int, int]] = []
        max_steps = int(np.sum(bounds) * 4 + 16)
        for _ in range(max_steps):
            if np.any(cell < 0) or np.any(cell >= bounds):
                break
            out.append((int(cell[0]), int(cell[1]), int(cell[2])))
            if np.array_equal(cell, target):
                break
            t_next = float(np.min(t_max))
            if not np.isfinite(t_next):
                break
            for axis in range(3):
                if t_max[axis] <= t_next + eps:
                    cell[axis] += int(step[axis])
                    t_max[axis] += t_delta[axis]
        return out


def _disk_structure(radius_cells: int) -> np.ndarray:
    radius = max(0, int(radius_cells))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((yy * yy + xx * xx) <= radius * radius).astype(bool)


def _binary_close_disk(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_cells))
    if radius <= 0 or not np.any(src):
        return src.copy()
    try:
        from scipy import ndimage

        return ndimage.binary_closing(src, structure=_disk_structure(radius)).astype(bool)
    except Exception:
        dilated = _dilate_bool(src, radius)
        return ~_dilate_bool(~dilated, radius)


def _fill_small_false_holes(mask: np.ndarray, *, max_area_cells: int) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(mask, dtype=bool)
    max_area = max(0, int(max_area_cells))
    filled = src.copy()
    filled_mask = np.zeros(src.shape, dtype=bool)
    if max_area <= 0 or not np.any(~src):
        return filled, filled_mask
    try:
        from scipy import ndimage

        labels, count = ndimage.label(~src)
        for label in range(1, int(count) + 1):
            component = labels == label
            area = int(np.count_nonzero(component))
            if area == 0 or area > max_area:
                continue
            rows, cols = np.nonzero(component)
            if int(rows.min()) == 0 or int(cols.min()) == 0 or int(rows.max()) == src.shape[0] - 1 or int(cols.max()) == src.shape[1] - 1:
                continue
            filled[component] = True
            filled_mask[component] = True
        return filled, filled_mask
    except Exception:
        visited = np.zeros(src.shape, dtype=bool)
        h, w = src.shape
        inv = ~src
        for row in range(h):
            for col in range(w):
                if visited[row, col] or not inv[row, col]:
                    continue
                stack = [(row, col)]
                visited[row, col] = True
                cells: list[tuple[int, int]] = []
                touches_border = False
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    touches_border = touches_border or rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and inv[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                if not touches_border and len(cells) <= max_area:
                    for rr, cc in cells:
                        filled[rr, cc] = True
                        filled_mask[rr, cc] = True
        return filled, filled_mask


def _validate_bool_xy_mask(mask: np.ndarray | None, shape: tuple[int, int], name: str, *, strict: bool = True) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        if strict:
            raise ValueError("%s shape %s does not match expected %s" % (str(name), arr.shape, shape))
        return None
    return arr


def _snap_to_true_cell(mask: np.ndarray, seed: tuple[int, int], max_radius: int) -> tuple[tuple[int, int] | None, float]:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    r0, c0 = int(seed[0]), int(seed[1])
    if 0 <= r0 < h and 0 <= c0 < w and bool(arr[r0, c0]):
        return (r0, c0), 0.0
    radius_limit = max(0, int(max_radius))
    for radius in range(1, radius_limit + 1):
        best: tuple[int, int] | None = None
        best_d2 = float("inf")
        for r in range(r0 - radius, r0 + radius + 1):
            if r < 0 or r >= h:
                continue
            for c in range(c0 - radius, c0 + radius + 1):
                if c < 0 or c >= w or not bool(arr[r, c]):
                    continue
                d2 = float((int(r) - r0) ** 2 + (int(c) - c0) ** 2)
                if d2 < best_d2:
                    best = (int(r), int(c))
                    best_d2 = d2
        if best is not None:
            return best, best_d2
    return None, float("inf")


def _reachable_component_from_seed(
    traversible: np.ndarray,
    seed: tuple[int, int],
    *,
    allow_diagonal: bool = True,
    prevent_diagonal_corner_cutting: bool = True,
) -> np.ndarray:
    trav = np.asarray(traversible, dtype=bool)
    h, w = trav.shape
    out = np.zeros((h, w), dtype=bool)
    sr, sc = int(seed[0]), int(seed[1])
    if sr < 0 or sr >= h or sc < 0 or sc >= w or not bool(trav[sr, sc]):
        return out
    steps: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if bool(allow_diagonal):
        steps.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    queue: deque[tuple[int, int]] = deque([(sr, sc)])
    out[sr, sc] = True
    while queue:
        r, c = queue.popleft()
        for dr, dc in steps:
            rr, cc = int(r + dr), int(c + dc)
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if out[rr, cc] or not bool(trav[rr, cc]):
                continue
            if dr != 0 and dc != 0 and bool(prevent_diagonal_corner_cutting):
                if not bool(trav[r + dr, c]):
                    continue
                if not bool(trav[r, c + dc]):
                    continue
            out[rr, cc] = True
            queue.append((rr, cc))
    return out


def _dilate_bool(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_cells))
    if radius <= 0 or not np.any(src):
        return src.copy()
    try:
        from scipy import ndimage

        return ndimage.binary_dilation(src, structure=_disk_structure(radius)).astype(bool)
    except Exception:
        pass
    out = src.copy()
    rows, cols = np.nonzero(src)
    h, w = src.shape
    offsets = [
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if dr * dr + dc * dc <= radius * radius
    ]
    for row, col in zip(rows, cols):
        for dr, dc in offsets:
            rr, cc = int(row + dr), int(col + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out
