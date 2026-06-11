from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.height_profile_door_wall_roomseg import extend_step2_wall_lines
from voxroom_online.isaac_runtime.mapping.online_roomseg.debug_viz import save_online_roomseg_debug
from voxroom_online.isaac_runtime.mapping.online_roomseg.separator_candidates import (
    DoorNeckConfig,
    LineExtensionConfig,
    LineExtensionHit,
    NoiseWallGapFillConfig,
    SeparatorCandidate,
    build_door_neck_candidates_from_extension_intersections,
    build_step2_separator_candidates_from_extensions,
    fill_noise_wall_gaps_from_runs,
)
from voxroom_online.isaac_runtime.mapping.online_roomseg.topology_tests import TopologyTestConfig, greedily_select_separators
from voxroom_online.isaac_runtime.mapping.online_roomseg.utils import component_metrics, conn, dilate, rasterize_line, relabel_compact
from voxroom_online.isaac_runtime.mapping.online_roomseg.wall_lines import (
    FilteredWallLine,
    LineFilteringConfig,
    LineWallsConfig,
    WallRunMergeConfig,
    WallRunSnapConfig,
    extract_line_supported_walls,
    filtered_wall_line_mask,
    filter_and_snap_wall_lines,
    line_supported_wall_mask,
    snap_wall_segments_to_runs,
)
from voxroom_online.isaac_runtime.mapping.room_segmentation import RoomMask, RoomSegmentationConfig
from voxroom_online.isaac_runtime.mapping.room_segmentation import _proposal_masks_debug, _room_from_mask
from voxroom_online.isaac_runtime.mapping.voxel_door_detector import (
    DOOR_ANCHOR_FILTERED_LINE,
    DOOR_ANCHOR_PROJECTED,
    DOOR_ANCHOR_PROJECTED_ANCHOR,
    DOOR_ANCHOR_STEP1,
    DOOR_ANCHOR_STRICT_RAW,
    DoorMemoryObservationMaps,
    VoxelDoorCompletionResult,
    VoxelDoorDetectorConfig,
    VoxelDoorMemory,
    classify_voxel_door_seeds,
    complete_voxel_doors_from_seeds,
)
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import VoxelOccupancyGrid3D
from voxroom_online.isaac_runtime.mapping.voxel_roomseg_evidence import (
    VoxelRoomsegEvidence,
    VoxelRoomsegEvidenceConfig,
    build_voxel_roomseg_evidence,
)
from voxroom_online.isaac_runtime.mapping.wall_projection import ProjectedWallLine, WallProjectionConfig, project_wall_evidence_to_axis_accumulator_lines


VOXEL_OCCUPANCY_ROOMSEG_BACKEND = "voxel_occupancy_door_wall_v33"
VOXEL_OCCUPANCY_ROOMSEG_ALGORITHM = "voxel_occupancy_door_wall_v33"
VOXEL_OCCUPANCY_ROOMSEG_CONTEXT = "voxel_occupancy_door_wall_v33_vlm"
VOXEL_OCCUPANCY_ROOMSEG_LEGACY_BACKENDS = {"voxel_occupancy_door_wall_v32", "voxel_occupancy_door_wall_v29", "voxel_occupancy_door_wall_v9"}
VOXEL_OCCUPANCY_ROOMSEG_LEGACY_CONTEXTS = {"voxel_occupancy_door_wall_v32_vlm", "voxel_occupancy_door_wall_v29_vlm", "voxel_occupancy_door_wall_v9_vlm"}


@dataclass
class Step2StageMaps:
    extension_hits_all_map: np.ndarray
    extension_hits_pre_topology_map: np.ndarray
    separator_candidates_pre_topology_map: np.ndarray
    topology_rejected_separator_map: np.ndarray
    accepted_separator_map: np.ndarray
    accepted_partition_cut_map: np.ndarray


@dataclass
class Step2LinePool:
    source_lines: list[FilteredWallLine]
    source_line_map: np.ndarray
    target_wall_map: np.ndarray
    target_source_map: np.ndarray
    debug: dict[str, object]


@dataclass
class PartitionMapBundle:
    wall_anchor_support_map: np.ndarray
    door_anchor_wall_map: np.ndarray
    step2_target_wall_map: np.ndarray
    partition_real_wall_map: np.ndarray
    base_partition_free: np.ndarray
    partition_unknown: np.ndarray
    removed_by_seed_carve_map: np.ndarray
    debug: dict[str, object]


@dataclass
class DoorAcceptanceMasks:
    raw_seed_mask: np.ndarray
    current_accepted_seed_mask: np.ndarray
    current_accepted_visual_mask: np.ndarray
    current_accepted_cut_mask: np.ndarray
    stable_visual_mask: np.ndarray
    stable_cut_mask: np.ndarray
    step2_block_mask: np.ndarray
    wall_carve_mask: np.ndarray
    projection_hard_forbidden_mask: np.ndarray
    debug: dict[str, object]


def _outside_roomseg_boundary_mask(voxel_grid: VoxelOccupancyGrid3D, shape: tuple[int, int]) -> np.ndarray:
    cfg = getattr(voxel_grid, "config", None)
    enabled = bool(getattr(cfg, "outside_boundary_enabled", True)) and bool(getattr(cfg, "outside_use_as_roomseg_domain_boundary", True))
    if not enabled:
        return np.zeros(shape, dtype=bool)
    outside = np.asarray(getattr(voxel_grid, "outside_xy", np.zeros(shape, dtype=bool)), dtype=bool)
    if outside.shape != tuple(shape):
        return np.zeros(shape, dtype=bool)
    return outside.copy()


def _apply_outside_boundary_to_evidence(evidence: VoxelRoomsegEvidence, outside_xy: np.ndarray) -> dict[str, object]:
    outside = np.asarray(outside_xy, dtype=bool)
    if outside.shape != np.asarray(evidence.vertical_free_xy).shape:
        return {
            "voxel_outside_roomseg_boundary_enabled": True,
            "voxel_outside_roomseg_boundary_cells": 0,
            "voxel_outside_roomseg_boundary_shape_mismatch": True,
        }
    if not np.any(outside):
        evidence.debug["voxel_outside_xy"] = outside.astype(bool)
        return {
            "voxel_outside_roomseg_boundary_enabled": True,
            "voxel_outside_roomseg_boundary_cells": 0,
            "voxel_outside_removed_free_cells": 0,
            "voxel_outside_removed_wall_cells": 0,
            "voxel_outside_removed_unknown_cells": 0,
        }

    removed_free = int(np.count_nonzero(np.asarray(evidence.vertical_free_xy, dtype=bool) & outside))
    removed_wall = int(np.count_nonzero(np.asarray(evidence.wall_xy, dtype=bool) & outside))
    removed_unknown = int(np.count_nonzero(np.asarray(evidence.unknown_xy, dtype=bool) & outside))
    bool_fields = (
        "vertical_free_xy",
        "wall_xy",
        "unknown_xy",
        "occupied_any_xy",
        "raw_occupied_wall_support_xy",
        "strict_raw_wall_xy",
        "wall_suppressed_by_free_xy",
        "unknown_dominant_xy",
        "wall_support_loose_xy",
        "wall_support_unknown_gated_xy",
        "wall_support_rejected_unknown_xy",
        "structural_wall_seed_xy",
        "structural_wall_ratio_xy",
        "wall_rejected_by_free_xy",
        "wall_rejected_by_unknown_xy",
        "nonstructural_occupied_xy",
        "small_unknown_hole_filled_xy",
        "wall_line_support_xy",
        "wall_line_support_raw_xy",
        "wall_line_support_rejected_by_free_xy",
        "wall_line_support_rejected_by_unknown_xy",
        "wall_line_support_rejected_by_observed_xy",
        "wall_line_support_rejected_by_nav_edge_xy",
        "ratio_wall_debug_xy",
        "free_wall_conflict_xy",
        "wall_line_support_strong_xy",
        "wall_line_support_conflict_xy",
        "wall_line_support_near_free_boundary_xy",
        "wall_line_support_rejected_furniture_xy",
        "wall_line_support_weight_xy",
        "wall_support_raw_occupied_xy",
        "wall_support_known_xy",
        "wall_support_unknown_rejected_xy",
        "wall_support_nav_unknown_rejected_xy",
        "wall_support_frontier_band_rejected_xy",
        "wall_support_free_conflict_xy",
        "wall_support_strong_xy",
        "wall_support_for_projection_xy",
        "frontier_unknown_band_xy",
        "strong_structural_support_xy",
        "bridge_only_support_xy",
        "forbidden_frontier_residual_support_xy",
        "forbidden_unknown_boundary_support_xy",
        "free_conflict_support_xy",
        "protected_structural_wall_band_xy",
        "support_seed_for_projection_xy",
        "support_bridge_for_projection_xy",
    )
    for name in bool_fields:
        if not hasattr(evidence, name):
            continue
        value = getattr(evidence, name)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.shape == outside.shape:
            if arr.dtype == bool:
                setattr(evidence, name, (arr & ~outside).astype(bool))
            elif np.issubdtype(arr.dtype, np.floating):
                updated = arr.copy()
                updated[outside] = 0.0
                setattr(evidence, name, updated.astype(arr.dtype, copy=False))
    evidence.debug["voxel_outside_xy"] = outside.astype(bool)
    evidence.debug["voxel_outside_roomseg_boundary_masked_free_xy"] = (np.asarray(evidence.vertical_free_xy, dtype=bool) & outside).astype(bool)
    debug = {
        "voxel_outside_roomseg_boundary_enabled": True,
        "voxel_outside_roomseg_boundary_cells": int(np.count_nonzero(outside)),
        "voxel_outside_removed_free_cells": int(removed_free),
        "voxel_outside_removed_wall_cells": int(removed_wall),
        "voxel_outside_removed_unknown_cells": int(removed_unknown),
        "voxel_outside_use_as_wall_evidence": False,
        "voxel_outside_use_as_door_anchor": False,
        "voxel_outside_use_as_separator_anchor": False,
    }
    evidence.debug.update(debug)
    return debug


@dataclass
class StableSeparatorTrack:
    track_id: int
    confidence: float
    first_seen_step: int
    last_seen_step: int
    line_cells: list[tuple[int, int]]
    p0_rc: tuple[float, float]
    p1_rc: tuple[float, float]
    source_candidate_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "kind": "step2_corridor",
            "confidence": float(self.confidence),
            "first_seen_step": int(self.first_seen_step),
            "last_seen_step": int(self.last_seen_step),
            "missed_update_count": 0,
            "contradiction_count": 0,
            "line_cells": [[int(r), int(c)] for r, c in self.line_cells],
            "p0_rc": [float(self.p0_rc[0]), float(self.p0_rc[1])],
            "p1_rc": [float(self.p1_rc[0]), float(self.p1_rc[1])],
            "source_candidate_ids": [int(v) for v in self.source_candidate_ids],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "StableSeparatorTrack":
        def rc_float(value: object, default: tuple[float, float]) -> tuple[float, float]:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
                return (float(value[0]), float(value[1]))  # type: ignore[index]
            return default

        def cells(value: object) -> list[tuple[int, int]]:
            out: list[tuple[int, int]] = []
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
                        out.append((int(item[0]), int(item[1])))  # type: ignore[index]
            return out

        return cls(
            track_id=int(data.get("track_id", 0) or 0),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            first_seen_step=int(data.get("first_seen_step", -1) or -1),
            last_seen_step=int(data.get("last_seen_step", -1) or -1),
            line_cells=cells(data.get("line_cells", [])),
            p0_rc=rc_float(data.get("p0_rc"), (0.0, 0.0)),
            p1_rc=rc_float(data.get("p1_rc"), (0.0, 0.0)),
            source_candidate_ids=[int(v) for v in (data.get("source_candidate_ids", []) or [])],  # type: ignore[union-attr]
        )


class StableSeparatorMemory:
    def __init__(self, ttl_updates: int = 30, decay_per_update: float = 0.02, min_confidence_to_keep: float = 0.15):
        self.ttl_updates = int(ttl_updates)
        self.decay_per_update = float(decay_per_update)
        self.min_confidence_to_keep = float(min_confidence_to_keep)
        self._tracks: list[StableSeparatorTrack] = []
        self._next_track_id = 1

    def to_state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "next_track_id": int(self._next_track_id),
            "ttl_updates": int(self.ttl_updates),
            "decay_per_update": float(self.decay_per_update),
            "min_confidence_to_keep": float(self.min_confidence_to_keep),
            "tracks": [track.to_dict() for track in self._tracks],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object] | None,
        *,
        ttl_updates: int = 30,
        decay_per_update: float = 0.02,
        min_confidence_to_keep: float = 0.15,
    ) -> "StableSeparatorMemory":
        raw = dict(state or {})
        mem = cls(
            ttl_updates=int(raw.get("ttl_updates", ttl_updates) or ttl_updates),
            decay_per_update=float(raw.get("decay_per_update", decay_per_update) or decay_per_update),
            min_confidence_to_keep=float(raw.get("min_confidence_to_keep", min_confidence_to_keep) or min_confidence_to_keep),
        )
        mem._next_track_id = int(raw.get("next_track_id", 1) or 1)
        tracks = raw.get("tracks", []) or []
        mem._tracks = [
            StableSeparatorTrack.from_dict(item)
            for item in tracks
            if isinstance(item, Mapping)
        ]
        if mem._tracks:
            mem._next_track_id = max(int(mem._next_track_id), 1 + max(int(track.track_id) for track in mem._tracks))
        return mem

    def load_state_dict(self, state: Mapping[str, object] | None) -> None:
        restored = StableSeparatorMemory.from_state_dict(
            state,
            ttl_updates=self.ttl_updates,
            decay_per_update=self.decay_per_update,
            min_confidence_to_keep=self.min_confidence_to_keep,
        )
        self._next_track_id = restored._next_track_id
        self._tracks = restored._tracks

    def update(self, candidates: Sequence[SeparatorCandidate], *, step: int, shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, object]]:
        for track in self._tracks:
            if int(track.last_seen_step) != int(step):
                track.confidence = max(0.0, float(track.confidence) - self.decay_per_update)
        updated = 0
        created = 0
        for candidate in candidates:
            if not bool(candidate.accepted):
                continue
            cells = _mask_cells(candidate.mask(shape))
            if not cells:
                continue
            match = self._best_match(candidate, cells, shape)
            if match is None:
                self._tracks.append(
                    StableSeparatorTrack(
                        track_id=int(self._next_track_id),
                        confidence=1.0,
                        first_seen_step=int(step),
                        last_seen_step=int(step),
                        line_cells=cells,
                        p0_rc=(float(candidate.p0_rc[0]), float(candidate.p0_rc[1])),
                        p1_rc=(float(candidate.p1_rc[0]), float(candidate.p1_rc[1])),
                        source_candidate_ids=[int(candidate.candidate_id)],
                    )
                )
                self._next_track_id += 1
                created += 1
            else:
                match.confidence = min(1.0, float(match.confidence) + 0.25)
                match.last_seen_step = int(step)
                match.line_cells = cells
                match.p0_rc = (float(candidate.p0_rc[0]), float(candidate.p0_rc[1]))
                match.p1_rc = (float(candidate.p1_rc[0]), float(candidate.p1_rc[1]))
                match.source_candidate_ids.append(int(candidate.candidate_id))
                match.source_candidate_ids = match.source_candidate_ids[-16:]
                updated += 1
        before = len(self._tracks)
        self._tracks = [
            track
            for track in self._tracks
            if float(track.confidence) >= self.min_confidence_to_keep and int(step) - int(track.last_seen_step) <= self.ttl_updates
        ]
        stable = np.zeros(shape, dtype=bool)
        for track in self._tracks:
            stable |= _cells_to_mask(track.line_cells, shape)
        return stable.astype(bool), {
            "voxel_separator_memory_enabled": True,
            "voxel_separator_memory_track_count": int(len(self._tracks)),
            "voxel_separator_memory_created_count": int(created),
            "voxel_separator_memory_updated_count": int(updated),
            "voxel_separator_memory_pruned_count": int(before + created - len(self._tracks)),
            "voxel_stable_step2_separator_cells": int(np.count_nonzero(stable)),
            "voxel_separator_memory_tracks": [track.to_dict() for track in self._tracks],
        }

    def _best_match(self, candidate: SeparatorCandidate, cells: list[tuple[int, int]], shape: tuple[int, int]) -> StableSeparatorTrack | None:
        cand = _cells_to_mask(cells, shape)
        best: tuple[float, StableSeparatorTrack] | None = None
        center = 0.5 * (np.asarray(candidate.p0_rc, dtype=np.float32) + np.asarray(candidate.p1_rc, dtype=np.float32))
        for track in self._tracks:
            old = _cells_to_mask(track.line_cells, shape)
            union = int(np.count_nonzero(cand | old))
            inter = int(np.count_nonzero(cand & old))
            iou = 0.0 if union <= 0 else float(inter) / float(union)
            old_center = 0.5 * (np.asarray(track.p0_rc, dtype=np.float32) + np.asarray(track.p1_rc, dtype=np.float32))
            dist = float(np.linalg.norm(center - old_center))
            if iou <= 0.0 and dist > 6.0:
                continue
            score = float(iou + max(0.0, 6.0 - dist) * 0.02)
            if best is None or score > best[0]:
                best = (score, track)
        return None if best is None else best[1]


@dataclass
class VoxelStep2TopologyConfig:
    corridor_min_split_area_m2: float = 0.05
    corridor_min_new_component_width_m: float = 0.10
    corridor_reject_tiny_side_width_cells_leq: int = 2
    corridor_tiny_side_min_area_m2: float = 0.03
    corridor_tiny_side_min_length_m: float = 0.35
    corridor_accept_long_narrow_side: bool = True
    corridor_local_topology_radius_cells: int = 20
    reject_small_known_side_for_line_extensions: bool = True
    small_known_side_area_m2: float = 1.00
    small_known_side_unknown_ratio_max: float = 0.20
    small_known_side_boundary_dilation_cells: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "VoxelStep2TopologyConfig":
        if isinstance(data, cls):
            return data
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class VoxelOccupancyDoorWallRoomSegConfig:
    enabled: bool = True
    resolution_m: float = 0.05
    map_info: MapInfo | None = None
    voxel_evidence: VoxelRoomsegEvidenceConfig = field(default_factory=VoxelRoomsegEvidenceConfig)
    wall_projection: WallProjectionConfig = field(default_factory=WallProjectionConfig)
    door: VoxelDoorDetectorConfig = field(default_factory=VoxelDoorDetectorConfig)
    line_walls: LineWallsConfig = field(default_factory=LineWallsConfig)
    line_filtering: LineFilteringConfig = field(default_factory=LineFilteringConfig)
    wall_run_snap: WallRunSnapConfig = field(default_factory=WallRunSnapConfig)
    wall_run_merge: WallRunMergeConfig = field(default_factory=WallRunMergeConfig)
    step1_gap_fill_enabled: bool = True
    step1_gap_fill_max_gap_m: float = 0.30
    step1_gap_fill_max_lateral_offset_m: float = 0.15
    step1_gap_fill_min_endpoint_support: float = 0.25
    step1_gap_fill_forbidden_on_door: bool = True
    step2_enabled: bool = False
    line_extension: LineExtensionConfig = field(default_factory=LineExtensionConfig)
    door_neck: DoorNeckConfig = field(default_factory=DoorNeckConfig)
    topology_test: TopologyTestConfig = field(default_factory=TopologyTestConfig)
    reject_step2_if_intersects_door: bool = True
    door_intersection_dilation_cells: int = 1
    enable_extension_intersection_fallback: bool = False
    reject_step2_if_tiny_side_width_cells_leq: int = 3
    separator_memory_enabled: bool = True
    separator_memory_ttl_updates: int = 30
    separator_memory_decay_per_update: float = 0.02
    separator_memory_min_confidence_to_keep: float = 0.15
    final_connectivity: int = 4
    merge_small_components_enabled: bool = False
    min_observed_free_cells: int = 1
    min_room_area_m2: float = 0.05
    use_real_wall_as_partition_barrier: bool = True
    real_wall_barrier_dilation_cells: int = 0
    door_seed_wall_carve_radius_cells: int = 2
    voxel_step2_topology: VoxelStep2TopologyConfig = field(default_factory=VoxelStep2TopologyConfig)
    voxel_show_wall_diagnostics: bool = False
    debug_dump: bool = False
    debug_dir: str = "debug/voxel_occupancy_door_wall_roomseg"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "VoxelOccupancyDoorWallRoomSegConfig":
        raw_root = dict(data or {})
        raw = dict(raw_root)
        online = dict(raw_root.get("online_roomseg", {}) or {})
        step1 = dict(raw_root.get("voxel_step1", {}) or {})
        step2 = dict(raw_root.get("voxel_step2", {}) or {})
        voxel_roomseg = dict(raw_root.get("voxel_roomseg", {}) or {})
        voxel_visualization = dict(raw_root.get("voxel_visualization", {}) or {})
        debug_layers = dict(raw_root.get("debug_layers", {}) or {})

        if "enabled" in debug_layers:
            raw["debug_dump"] = bool(debug_layers.get("enabled"))
        if "output_dir" in debug_layers:
            raw["debug_dir"] = str(debug_layers.get("output_dir"))

        topology_raw = dict(raw_root.get("topology_test", online.get("topology_test", online.get("topology", {}))) or {})
        voxel_step2_topology_raw = dict(raw_root.get("voxel_step2_topology", {}) or {})
        if "compound_separator" in voxel_step2_topology_raw and "compound_separator" not in topology_raw:
            topology_raw["compound_separator"] = voxel_step2_topology_raw.get("compound_separator")
        for key in (
            "reject_small_known_side_for_line_extensions",
            "small_known_side_area_m2",
            "small_known_side_unknown_ratio_max",
            "small_known_side_boundary_dilation_cells",
        ):
            if key in voxel_step2_topology_raw:
                topology_raw[key] = voxel_step2_topology_raw[key]
        if "reject_if_tiny_side_width_cells_leq" in step2:
            topology_raw["reject_if_side_width_cells_leq"] = int(step2["reject_if_tiny_side_width_cells_leq"])
        else:
            topology_raw.setdefault("reject_if_side_width_cells_leq", int(raw.get("reject_step2_if_tiny_side_width_cells_leq", 3)))
        topology_raw.setdefault("min_new_component_width_m", 0.20)

        line_extension_raw = dict(online.get("line_extension", raw_root.get("line_extension", {})) or {})
        line_extension_raw.update({key: step2[key] for key in step2 if key in LineExtensionConfig.__dataclass_fields__})
        line_extension_raw.setdefault("min_extension_m", 0.40)
        line_extension_raw.setdefault("max_extension_m", 1.60)
        line_extension_raw.setdefault("max_probe_m", 1.60)
        line_extension_raw.setdefault("allow_hit_virtual_door_on_pass2", False)
        line_extension_raw.setdefault("require_free_between_start_and_hit", True)
        line_extension_raw.setdefault("min_free_cells_between_start_and_hit", 3)

        nested = {
            "voxel_evidence": VoxelRoomsegEvidenceConfig.from_mapping(raw_root.get("voxel_roomseg_evidence", {})),
            "wall_projection": WallProjectionConfig.from_mapping(raw_root.get("voxel_wall_projection", {})),
            "door": VoxelDoorDetectorConfig.from_mapping(raw_root.get("voxel_door", {})),
            "line_walls": LineWallsConfig.from_mapping(online.get("line_walls", raw_root.get("line_walls"))),
            "line_filtering": LineFilteringConfig.from_mapping(online.get("line_filtering", raw_root.get("line_filtering"))),
            "wall_run_snap": WallRunSnapConfig.from_mapping(online.get("wall_run_snap", raw_root.get("wall_run_snap"))),
            "wall_run_merge": WallRunMergeConfig.from_mapping(online.get("wall_run_merge", raw_root.get("wall_run_merge"))),
            "line_extension": LineExtensionConfig.from_mapping(line_extension_raw),
            "door_neck": DoorNeckConfig.from_mapping(online.get("door_neck", raw_root.get("door_neck"))),
            "topology_test": TopologyTestConfig.from_mapping(topology_raw),
            "voxel_step2_topology": VoxelStep2TopologyConfig.from_mapping(voxel_step2_topology_raw),
        }
        step1_key_map = {
            "gap_fill_enabled": "step1_gap_fill_enabled",
            "gap_fill_max_gap_m": "step1_gap_fill_max_gap_m",
            "gap_fill_max_lateral_offset_m": "step1_gap_fill_max_lateral_offset_m",
            "gap_fill_min_endpoint_support": "step1_gap_fill_min_endpoint_support",
            "gap_fill_forbidden_on_door": "step1_gap_fill_forbidden_on_door",
        }
        for src, dst in step1_key_map.items():
            if src in step1:
                raw[dst] = step1[src]
        step2_key_map = {
            "reject_if_intersects_door": "reject_step2_if_intersects_door",
            "door_intersection_dilation_cells": "door_intersection_dilation_cells",
            "enable_extension_intersection_fallback": "enable_extension_intersection_fallback",
            "reject_if_tiny_side_width_cells_leq": "reject_step2_if_tiny_side_width_cells_leq",
            "separator_memory_enabled": "separator_memory_enabled",
            "separator_memory_ttl_updates": "separator_memory_ttl_updates",
            "separator_memory_decay_per_update": "separator_memory_decay_per_update",
            "separator_memory_min_confidence_to_keep": "separator_memory_min_confidence_to_keep",
        }
        for src, dst in step2_key_map.items():
            if src in step2:
                raw[dst] = step2[src]
        if "enabled" in step2:
            raw["step2_enabled"] = bool(step2["enabled"])
        for key in ("use_real_wall_as_partition_barrier", "real_wall_barrier_dilation_cells", "door_seed_wall_carve_radius_cells"):
            if key in voxel_roomseg:
                raw[key] = voxel_roomseg[key]
        if "voxel_show_wall_diagnostics" in voxel_visualization:
            raw["voxel_show_wall_diagnostics"] = voxel_visualization["voxel_show_wall_diagnostics"]
        for key, value in overrides.items():
            if value is not None:
                raw[key] = value
        fields = {name for name in cls.__dataclass_fields__}
        base = {key: raw[key] for key in raw if key in fields and key not in nested}
        base.update(nested)
        cfg = cls(**base)
        cfg.voxel_evidence.active_z_min_m = float(cfg.voxel_evidence.active_z_min_m)
        return cfg

    def room_config(self) -> RoomSegmentationConfig:
        return RoomSegmentationConfig(
            algorithm=VOXEL_OCCUPANCY_ROOMSEG_ALGORITHM,
            source_grid="voxel_vertical_free",
            proposal_mode=VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
            finalization_mode="no_merge",
            min_observed_free_cells=int(self.min_observed_free_cells),
            min_room_area_m2=float(self.min_room_area_m2),
            resolution_m=float(self.resolution_m),
            map_info=self.map_info,
        )


@dataclass
class VoxelOccupancyDoorWallRoomSegResult:
    room_label_map: np.ndarray
    separator_map: np.ndarray
    wall_map: np.ndarray
    door_cut_map: np.ndarray
    step1_wall_gap_fill_map: np.ndarray
    step2_extension_separator_map: np.ndarray
    layers: dict[str, np.ndarray]
    debug: dict[str, object]


class VoxelOccupancyDoorWallRoomSegmenter:
    context_source = VOXEL_OCCUPANCY_ROOMSEG_CONTEXT

    def __init__(self, config: VoxelOccupancyDoorWallRoomSegConfig | Mapping[str, object] | None = None, map_info: MapInfo | None = None):
        self.config = config if isinstance(config, VoxelOccupancyDoorWallRoomSegConfig) else VoxelOccupancyDoorWallRoomSegConfig.from_mapping(config or {}, map_info=map_info)
        if map_info is not None:
            self.config.map_info = map_info
            self.config.resolution_m = float(map_info.resolution_m)
        self.last_debug: dict[str, object] = {}
        self.last_result: VoxelOccupancyDoorWallRoomSegResult | None = None
        self.door_memory = VoxelDoorMemory(self.config.door)
        self.separator_memory = StableSeparatorMemory(
            ttl_updates=int(self.config.separator_memory_ttl_updates),
            decay_per_update=float(self.config.separator_memory_decay_per_update),
            min_confidence_to_keep=float(self.config.separator_memory_min_confidence_to_keep),
        )

    def export_replay_state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "door_memory": self.door_memory.to_state_dict(),
            "separator_memory": self.separator_memory.to_state_dict(),
        }

    def import_replay_state(self, state: Mapping[str, object] | None) -> None:
        raw = dict(state or {})
        door_state = raw.get("door_memory")
        if isinstance(door_state, Mapping):
            self.door_memory.load_state_dict(door_state)
        separator_state = raw.get("separator_memory")
        if isinstance(separator_state, Mapping):
            self.separator_memory.load_state_dict(separator_state)

    def update(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        *,
        step: int,
        voxel_grid: VoxelOccupancyGrid3D | None = None,
        object_memory: Iterable[object] | None = None,
        **kwargs,
    ) -> list[RoomMask]:
        _ = object_memory
        door_seed_no_clearance_free_mask = kwargs.get("door_seed_no_clearance_free_mask", observed_free_mask)
        if voxel_grid is None:
            self.last_result = None
            self.last_debug = {
                "backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
                "algorithm": VOXEL_OCCUPANCY_ROOMSEG_ALGORITHM,
                "context_source": VOXEL_OCCUPANCY_ROOMSEG_CONTEXT,
                "source": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
                "reason": "voxel_grid_missing",
                "room_count": 0,
                "rooms": [],
            }
            return []
        navigation_free_mask = kwargs.get("navigation_free_mask", observed_free_mask)
        navigation_obstacle_mask = kwargs.get("navigation_obstacle_mask", obstacle_mask)
        memory_before = self.export_replay_state()
        result = run_voxel_occupancy_door_wall_roomseg(
            occupancy_map=occupancy_map,
            observed_free_mask=observed_free_mask,
            obstacle_mask=obstacle_mask,
            unknown_mask=unknown_mask,
            voxel_grid=voxel_grid,
            navigation_free_mask=navigation_free_mask,
            navigation_obstacle_mask=navigation_obstacle_mask,
            door_seed_no_clearance_free_mask=door_seed_no_clearance_free_mask,
            resolution_m=float(self.config.resolution_m),
            config=self.config,
            step=int(step),
            door_memory=self.door_memory,
            separator_memory=self.separator_memory,
        )
        memory_after = self.export_replay_state()
        self.last_result = result
        self.last_debug = dict(result.debug)
        self.last_debug["voxel_roomseg_memory_before_json"] = json.dumps(memory_before, ensure_ascii=False, sort_keys=True)
        self.last_debug["voxel_roomseg_memory_after_json"] = json.dumps(memory_after, ensure_ascii=False, sort_keys=True)
        self.last_debug["voxel_door_memory_before_roomseg_json"] = json.dumps(memory_before.get("door_memory", {}), ensure_ascii=False, sort_keys=True)
        self.last_debug["voxel_door_memory_after_roomseg_json"] = json.dumps(memory_after.get("door_memory", {}), ensure_ascii=False, sort_keys=True)
        self.last_debug["voxel_separator_memory_before_roomseg_json"] = json.dumps(memory_before.get("separator_memory", {}), ensure_ascii=False, sort_keys=True)
        self.last_debug["voxel_separator_memory_after_roomseg_json"] = json.dumps(memory_after.get("separator_memory", {}), ensure_ascii=False, sort_keys=True)
        return _rooms_from_labels(result.room_label_map, np.asarray(result.layers["voxel_unknown_xy"], dtype=bool), self.config.room_config(), int(step), result.debug)


def run_voxel_occupancy_door_wall_roomseg(
    *,
    occupancy_map: np.ndarray,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    voxel_grid: VoxelOccupancyGrid3D,
    navigation_free_mask: np.ndarray,
    navigation_obstacle_mask: np.ndarray,
    resolution_m: float,
    config: VoxelOccupancyDoorWallRoomSegConfig | Mapping[str, object] | None = None,
    step: int = 0,
    door_memory: VoxelDoorMemory | None = None,
    separator_memory: StableSeparatorMemory | None = None,
    door_seed_no_clearance_free_mask: np.ndarray | None = None,
) -> VoxelOccupancyDoorWallRoomSegResult:
    cfg = config if isinstance(config, VoxelOccupancyDoorWallRoomSegConfig) else VoxelOccupancyDoorWallRoomSegConfig.from_mapping(config, resolution_m=resolution_m)
    shape = np.asarray(observed_free_mask, dtype=bool).shape
    for name, arr in {
        "occupancy_map": occupancy_map,
        "obstacle_mask": obstacle_mask,
        "unknown_mask": unknown_mask,
        "navigation_free_mask": navigation_free_mask,
        "navigation_obstacle_mask": navigation_obstacle_mask,
        **({"door_seed_no_clearance_free_mask": door_seed_no_clearance_free_mask} if door_seed_no_clearance_free_mask is not None else {}),
    }.items():
        if np.asarray(arr).shape != shape:
            raise ValueError("%s must match observed_free_mask shape" % name)
    if tuple(voxel_grid.shape) != tuple(shape):
        raise ValueError("voxel_grid shape must match roomseg masks")
    nav_free_mask = np.asarray(navigation_free_mask, dtype=bool).copy()
    door_seed_free_mask = (
        np.asarray(door_seed_no_clearance_free_mask, dtype=bool).copy()
        if door_seed_no_clearance_free_mask is not None
        else nav_free_mask.copy()
    )
    nav_obstacle_mask = np.asarray(navigation_obstacle_mask, dtype=bool).copy()
    nav_unknown_mask = np.asarray(unknown_mask, dtype=bool).copy()
    outside_xy = _outside_roomseg_boundary_mask(voxel_grid, tuple(shape))
    if np.any(outside_xy):
        nav_free_mask[outside_xy] = False
        door_seed_free_mask[outside_xy] = False
        nav_obstacle_mask[outside_xy] = False
        nav_unknown_mask[outside_xy] = False

    evidence = build_voxel_roomseg_evidence(
        voxel_grid=voxel_grid,
        navigation_free_mask=nav_free_mask,
        navigation_obstacle_mask=nav_obstacle_mask,
        unknown_mask_from_navigation=nav_unknown_mask,
        resolution_m=float(resolution_m),
        config=cfg.voxel_evidence,
    )
    outside_grid_debug = dict(getattr(voxel_grid, "last_outside_debug", {}) or {})
    for key, value in outside_grid_debug.items():
        if isinstance(value, np.ndarray) and np.asarray(value).shape == tuple(shape):
            evidence.debug[key] = np.asarray(value).copy()
        elif not isinstance(value, np.ndarray):
            evidence.debug[key] = value
    evidence.debug["voxel_floor_frustum_seen_count_xy"] = np.asarray(
        getattr(voxel_grid, "floor_frustum_seen_count_xy", np.zeros(shape, dtype=np.uint16)),
        dtype=np.uint16,
    ).copy()
    evidence.debug["voxel_outside_score_xy"] = np.asarray(
        getattr(voxel_grid, "outside_score_xy", np.zeros(shape, dtype=np.uint8)),
        dtype=np.uint8,
    ).copy()
    outside_roomseg_debug = _apply_outside_boundary_to_evidence(evidence, outside_xy)
    door_seed_result = classify_voxel_door_seeds(
        voxel_grid=voxel_grid,
        config=cfg.door,
        sensor_range_count=getattr(voxel_grid, "sensor_range_count", None),
        navigation_free_mask=nav_free_mask,
        no_clearance_navigation_free_mask=door_seed_free_mask,
    )
    door_seed_mask = np.asarray(door_seed_result.door_seed_mask, dtype=bool) & ~outside_xy

    wall_line_support_strong_xy = np.asarray(
        evidence.wall_line_support_strong_xy
        if evidence.wall_line_support_strong_xy is not None
        else evidence.wall_line_support_xy,
        dtype=bool,
    )
    wall_line_support_conflict_xy = np.asarray(
        evidence.wall_line_support_conflict_xy
        if evidence.wall_line_support_conflict_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
    wall_line_support_rejected_unknown_xy = np.asarray(
        evidence.wall_line_support_rejected_unknown_xy
        if evidence.wall_line_support_rejected_unknown_xy is not None
        else evidence.wall_line_support_rejected_by_unknown_xy,
        dtype=bool,
    )
    wall_line_support_weight_xy = np.asarray(
        evidence.wall_line_support_weight_xy
        if evidence.wall_line_support_weight_xy is not None
        else wall_line_support_strong_xy.astype(np.float32),
        dtype=np.float32,
    )
    projection_seed = np.asarray(
        evidence.support_seed_for_projection_xy
        if evidence.support_seed_for_projection_xy is not None
        else (
            evidence.wall_support_strong_xy
            if evidence.wall_support_strong_xy is not None
            else wall_line_support_strong_xy
        ),
        dtype=bool,
    )
    projection_bridge = np.asarray(
        evidence.support_bridge_for_projection_xy
        if evidence.support_bridge_for_projection_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
    projection_input = np.asarray(
        evidence.wall_support_for_projection_xy
        if evidence.wall_support_for_projection_xy is not None
        else (projection_seed | projection_bridge),
        dtype=bool,
    )
    projection_weight = np.asarray(
        evidence.wall_support_weight_xy
        if evidence.wall_support_weight_xy is not None
        else wall_line_support_weight_xy,
        dtype=np.float32,
    )
    frontier_unknown_band = np.asarray(
        evidence.frontier_unknown_band_xy
        if evidence.frontier_unknown_band_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
    forbidden_frontier_residual = np.asarray(
        evidence.forbidden_frontier_residual_support_xy
        if evidence.forbidden_frontier_residual_support_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
    forbidden_unknown_boundary = np.asarray(
        evidence.forbidden_unknown_boundary_support_xy
        if evidence.forbidden_unknown_boundary_support_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
    forbidden_residual_support = forbidden_frontier_residual | forbidden_unknown_boundary
    protected_structural_wall_band = np.asarray(
        evidence.protected_structural_wall_band_xy
        if evidence.protected_structural_wall_band_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
    projection_hard_forbidden_mask = np.zeros(shape, dtype=bool)
    wall_projection = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=projection_seed,
        support_bridge_map=projection_bridge,
        forbidden_frontier_residual_map=forbidden_residual_support,
        protected_structural_wall_band=protected_structural_wall_band,
        support_weight=projection_weight,
        door_forbidden_mask=projection_hard_forbidden_mask,
        vertical_free_map=evidence.vertical_free_xy,
        unknown_map=evidence.unknown_xy,
        unknown_ratio_map=evidence.unknown_ratio_active_xy,
        navigation_unknown_map=nav_unknown_mask,
        frontier_unknown_band=frontier_unknown_band,
        structural_side_support_map=np.asarray(evidence.wall_xy, dtype=bool)
        | np.asarray(evidence.structural_wall_ratio_xy, dtype=bool)
        | projection_seed,
        resolution_m=float(resolution_m),
        config=cfg.wall_projection,
    )
    anchor_wall_projection = wall_projection
    projected_wall_map = np.asarray(
        wall_projection.projected_wall_display_map
        if wall_projection.projected_wall_display_map is not None
        else wall_projection.projected_wall_map,
        dtype=bool,
    )
    anchor_projected_wall_map = np.asarray(
        wall_projection.projected_wall_anchor_map
        if wall_projection.projected_wall_anchor_map is not None
        else projected_wall_map,
        dtype=bool,
    )
    projected_step2_lines = [
        projected_wall_line_to_filtered_wall_line(line, float(resolution_m), line_id=100000 + idx)
        for idx, line in enumerate(getattr(wall_projection, "projected_step2_source_lines", []) or [], start=1)
    ]
    wall_for_line_extraction = np.asarray(evidence.wall_xy, dtype=bool) | projected_wall_map
    segments, wall_debug = extract_line_supported_walls(
        wall_for_line_extraction,
        resolution_m=float(resolution_m),
        config=cfg.line_walls,
    )
    raw_line_map = line_supported_wall_mask(segments, shape)
    filtered_lines, filter_debug = filter_and_snap_wall_lines(
        segments,
        wall_candidate_clean=wall_for_line_extraction,
        free_clean=evidence.vertical_free_xy,
        resolution_m=float(resolution_m),
        config=cfg.line_filtering,
    )
    seed_line_filter_cfg = replace(
        cfg.line_filtering,
        min_filtered_line_length_m=min(float(cfg.line_filtering.min_filtered_line_length_m), 0.15),
        min_filtered_support_ratio=min(float(cfg.line_filtering.min_filtered_support_ratio), 0.20),
        endpoint_min_wall_support_m=min(float(cfg.line_filtering.endpoint_min_wall_support_m), 0.10),
        min_confidence=min(float(cfg.line_filtering.min_confidence), 0.20),
    )
    extension_seed_lines, seed_filter_debug = filter_and_snap_wall_lines(
        segments,
        wall_candidate_clean=wall_for_line_extraction | anchor_projected_wall_map,
        free_clean=evidence.vertical_free_xy,
        resolution_m=float(resolution_m),
        config=seed_line_filter_cfg,
    )
    filtered_line_map = filtered_wall_line_mask(filtered_lines, shape)
    extension_seed_line_map = filtered_wall_line_mask(extension_seed_lines, shape)
    filtered_line_map_clean = filtered_line_map.copy()
    validated_extension_seed_line_map = extension_seed_line_map.copy()
    wall_base_pre_step1 = np.asarray(evidence.wall_xy, dtype=bool) | projected_wall_map

    wall_runs, wall_run_debug = snap_wall_segments_to_runs(
        segments,
        wall_base_pre_step1,
        resolution_m=float(resolution_m),
        max_angle_to_axis_deg=float(cfg.wall_run_snap.max_angle_to_axis_deg),
        support_band_cells=int(cfg.wall_run_snap.support_band_cells),
        min_run_length_m=float(cfg.wall_run_snap.min_run_length_m),
        min_support_ratio=float(cfg.wall_run_snap.min_support_ratio),
        close_holes_m=0.0,
    )
    step1_gap_fill_map, step1_gap_debug = fill_noise_wall_gaps_from_runs(
        wall_runs,
        shape=shape,
        resolution_m=float(resolution_m),
        config=NoiseWallGapFillConfig(
            enabled=bool(cfg.step1_gap_fill_enabled),
            max_gap_m=float(cfg.step1_gap_fill_max_gap_m),
            max_lateral_offset_m=float(cfg.step1_gap_fill_max_lateral_offset_m),
            min_endpoint_support=float(cfg.step1_gap_fill_min_endpoint_support),
            thickness_cells=0,
        ),
    )
    step1_gap_fill_map &= np.asarray(evidence.active_observed_xy, dtype=bool)
    if bool(cfg.step1_gap_fill_forbidden_on_door):
        step1_gap_fill_map &= ~projection_hard_forbidden_mask
    step1_gap_fill_map &= ~np.asarray(evidence.vertical_free_xy, dtype=bool)
    step1_completed_wall_map = wall_base_pre_step1 | step1_gap_fill_map
    partition_maps = build_voxel_partition_maps(
        evidence=evidence,
        door_seed_mask=door_seed_mask,
        strict_raw_wall=np.asarray(evidence.wall_xy, dtype=bool),
        projected_wall_map=projected_wall_map,
        anchor_projected_wall_map=anchor_projected_wall_map,
        filtered_line_map=filtered_line_map_clean,
        extension_seed_line_map=validated_extension_seed_line_map,
        step1_gap_fill_map=step1_gap_fill_map,
        cfg=cfg,
    )
    real_wall_barrier_map = np.asarray(partition_maps.partition_real_wall_map, dtype=bool)
    if int(cfg.real_wall_barrier_dilation_cells) > 0:
        real_wall_barrier_for_partition = dilate(real_wall_barrier_map, int(cfg.real_wall_barrier_dilation_cells))
    else:
        real_wall_barrier_for_partition = real_wall_barrier_map.copy()
    base_partition_free = np.asarray(partition_maps.base_partition_free, dtype=bool) & ~real_wall_barrier_for_partition
    free_after_step1 = base_partition_free.copy()
    unknown_after_step1 = np.asarray(partition_maps.partition_unknown, dtype=bool)
    door_anchor_source_map = _door_anchor_source_map(
        shape,
        strict_raw_wall=np.asarray(evidence.wall_xy, dtype=bool),
        projected_wall=projected_wall_map,
        anchor_projected_wall=np.zeros(shape, dtype=bool),
        step1_gap_fill=step1_gap_fill_map,
        filtered_line=np.zeros(shape, dtype=bool),
    )
    door_anchor_source_map = np.where(partition_maps.door_anchor_wall_map, door_anchor_source_map, 0).astype(np.uint8)
    door_anchor_source_map[(partition_maps.door_anchor_wall_map) & (door_anchor_source_map == 0)] = DOOR_ANCHOR_STRICT_RAW
    door_anchor_wall_map = np.asarray(partition_maps.door_anchor_wall_map, dtype=bool)
    door_cluster_barrier = np.asarray(partition_maps.partition_real_wall_map, dtype=bool) & ~dilate(door_seed_mask, 2)
    door_completion = complete_voxel_doors_from_seeds(
        seed_result=door_seed_result,
        free_map=evidence.vertical_free_xy,
        free_map_for_visual_validation=evidence.vertical_free_xy,
        base_partition_free=base_partition_free,
        door_conflict_free_mask=door_seed_free_mask,
        door_extension_no_clearance_occupied_mask=nav_obstacle_mask,
        anchor_wall_map=door_anchor_wall_map,
        unknown_map=unknown_after_step1,
        resolution_m=float(resolution_m),
        config=cfg.door,
        anchor_source_map=door_anchor_source_map,
        real_wall_barrier_map=real_wall_barrier_for_partition,
        seed_cluster_barrier_map=door_cluster_barrier,
    )
    current_door_visual_mask = np.asarray(door_completion.door_centerline_visual_mask, dtype=bool)
    current_door_cut_mask = np.asarray(door_completion.door_cut_mask_for_partition, dtype=bool)
    current_geometry_warning_cut_mask = np.asarray(door_completion.door_geometry_warning_cut_mask, dtype=bool)
    current_topology_effective_cut_mask = np.asarray(
        door_completion.debug.get("voxel_door_partition_effective_verified_mask", door_completion.door_topology_effective_cut_mask),
        dtype=bool,
    )
    sensor_range_count_xy = np.asarray(
        evidence.debug.get("voxel_sensor_range_count_xy", np.zeros(shape, dtype=np.uint16)),
        dtype=np.uint16,
    )
    if sensor_range_count_xy.shape != shape:
        sensor_range_count_xy = np.zeros(shape, dtype=np.uint16)
    sensor_range_xy = sensor_range_count_xy > 0
    if not np.any(sensor_range_xy):
        sensor_range_xy = np.asarray(evidence.active_observed_xy, dtype=bool)
    door_memory_observation = DoorMemoryObservationMaps(
        observed_xy=np.asarray(evidence.active_observed_xy, dtype=bool)
        | np.asarray(evidence.vertical_free_xy, dtype=bool)
        | np.asarray(evidence.wall_xy, dtype=bool),
        sensor_range_xy=np.asarray(sensor_range_xy, dtype=bool),
        vertical_free_xy=np.asarray(evidence.vertical_free_xy, dtype=bool),
        wall_xy=np.asarray(real_wall_barrier_for_partition, dtype=bool),
        raw_seed_mask=np.asarray(door_seed_mask, dtype=bool),
        current_verified_cut_mask=np.asarray(current_topology_effective_cut_mask, dtype=bool),
        ceiling_height_m=getattr(voxel_grid, "ceiling_height_m", None),
    )
    if door_memory is not None:
        door_memory_result = door_memory.update(
            door_completion.candidates,
            step=int(step),
            shape=shape,
            observation=door_memory_observation,
        )
        stable_door_cut_mask = np.asarray(door_memory_result.stable_door_cut_mask, dtype=bool)
        stable_door_visual_mask = np.asarray(door_memory_result.stable_door_visual_mask, dtype=bool)
        door_memory_debug = dict(door_memory_result.debug)
    else:
        stable_door_cut_mask = np.zeros(shape, dtype=bool)
        stable_door_visual_mask = np.zeros(shape, dtype=bool)
        door_memory_debug = {
            "voxel_door_memory_enabled": bool(cfg.door.door_memory_enabled),
            "voxel_door_memory_active": False,
            "voxel_door_memory_track_count": 0,
            "voxel_door_memory_observed_decay_band_mask": np.zeros(shape, dtype=bool),
            "voxel_door_memory_unobserved_track_mask": np.zeros(shape, dtype=bool),
            "voxel_door_memory_contradiction_mask": np.zeros(shape, dtype=bool),
        }
    current_accepted_visual_mask = np.asarray(door_completion.debug.get("voxel_accepted_door_centerline_mask", current_door_cut_mask), dtype=bool)
    current_accepted_cut_mask = current_door_cut_mask.copy()
    door_cut_mask = current_accepted_cut_mask | stable_door_cut_mask
    door_visual_mask = current_door_visual_mask | stable_door_visual_mask
    accepted_door_visual_mask = current_accepted_visual_mask | stable_door_visual_mask
    cluster_map = np.asarray(door_completion.debug.get("voxel_door_seed_cluster_map", np.zeros(shape, dtype=np.int32)), dtype=np.int32)
    current_accepted_seed_mask = accepted_seed_mask_from_candidates(door_completion.candidates, cluster_map, shape) & door_seed_mask
    step2_door_reject_mask = current_topology_effective_cut_mask | stable_door_cut_mask
    wall_carve_mask = current_accepted_cut_mask | stable_door_cut_mask
    door_acceptance = DoorAcceptanceMasks(
        raw_seed_mask=door_seed_mask,
        current_accepted_seed_mask=current_accepted_seed_mask,
        current_accepted_visual_mask=current_accepted_visual_mask,
        current_accepted_cut_mask=current_accepted_cut_mask,
        stable_visual_mask=stable_door_visual_mask,
        stable_cut_mask=stable_door_cut_mask,
        step2_block_mask=step2_door_reject_mask,
        wall_carve_mask=wall_carve_mask,
        projection_hard_forbidden_mask=projection_hard_forbidden_mask,
        debug={
            "voxel_door_acceptance_policy": "v26_raw_seed_debug_only",
            "voxel_door_acceptance_policy_v32": "topology_effective_cut_only",
            "voxel_door_acceptance_policy_v29": "topology_effective_cut_only",
            "voxel_raw_seed_not_step2_block": True,
            "voxel_raw_seed_not_wall_carve": True,
            "voxel_raw_seed_not_projection_hard_forbidden": True,
            "voxel_current_accepted_seed_cells": int(np.count_nonzero(current_accepted_seed_mask)),
            "voxel_step2_block_raw_seed_removed": True,
            "voxel_step2_block_raw_seed_removed_cells": int(np.count_nonzero(door_seed_mask & ~step2_door_reject_mask)),
            "voxel_step2_block_cells": int(np.count_nonzero(step2_door_reject_mask)),
            "voxel_step2_block_topology_effective_door_only": True,
            "voxel_step2_door_block_topology_effective_cells": int(np.count_nonzero(step2_door_reject_mask)),
            "voxel_step2_raw_seed_not_blocking_cells": int(np.count_nonzero(door_seed_mask & ~step2_door_reject_mask)),
            "voxel_step2_raw_seed_would_have_blocked_cells": int(np.count_nonzero(door_seed_mask & ~step2_door_reject_mask)),
            "voxel_step2_raw_seed_not_blocking_mask": (door_seed_mask & ~step2_door_reject_mask).astype(bool),
            "voxel_wall_carve_accepted_door_cells": int(np.count_nonzero(wall_carve_mask)),
        },
    )
    step1_wall_mask = step1_completed_wall_map
    partition_maps = build_voxel_partition_maps(
        evidence=evidence,
        door_seed_mask=door_seed_mask,
        wall_carve_mask=door_acceptance.wall_carve_mask,
        strict_raw_wall=np.asarray(evidence.wall_xy, dtype=bool),
        projected_wall_map=projected_wall_map,
        anchor_projected_wall_map=anchor_projected_wall_map,
        filtered_line_map=filtered_line_map_clean,
        extension_seed_line_map=validated_extension_seed_line_map,
        step1_gap_fill_map=step1_gap_fill_map,
        cfg=cfg,
    )
    real_wall_barrier_map = np.asarray(partition_maps.partition_real_wall_map, dtype=bool)
    if int(cfg.real_wall_barrier_dilation_cells) > 0:
        real_wall_barrier_for_partition = dilate(real_wall_barrier_map, int(cfg.real_wall_barrier_dilation_cells))
    else:
        real_wall_barrier_for_partition = real_wall_barrier_map.copy()
    base_partition_free = np.asarray(partition_maps.base_partition_free, dtype=bool) & ~real_wall_barrier_for_partition
    free_after_step1 = base_partition_free.copy()
    unknown_after_step1 = np.asarray(partition_maps.partition_unknown, dtype=bool)
    step2_line_pool = build_step2_line_pool(
        filtered_lines=filtered_lines,
        extension_seed_lines=extension_seed_lines,
        projected_step2_lines=projected_step2_lines,
        strict_raw_wall=np.asarray(evidence.wall_xy, dtype=bool),
        projected_wall_map=projected_wall_map,
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        step1_completed_wall_map=step1_completed_wall_map,
        filtered_line_map=filtered_line_map_clean,
        extension_seed_line_map=validated_extension_seed_line_map,
        shape=shape,
        resolution_m=float(resolution_m),
        target_wall_override=partition_maps.step2_target_wall_map,
    )
    step2_disabled_debug: dict[str, object] = {}
    if not bool(cfg.step2_enabled):
        step2_line_pool.debug.update(
            {
                "voxel_step2_wall_line_extension_disabled": True,
                "voxel_step2_wall_line_extension_disable_reason": "config.voxel_step2.enabled=false",
                "voxel_step2_source_line_count_before_disable": int(len(step2_line_pool.source_lines)),
                "voxel_step2_source_line_count": 0,
                "voxel_step2_projected_source_line_count": 0,
                "voxel_step2_source_line_count_by_source": {},
                "voxel_step2_filtered_line_count": 0,
                "voxel_step2_extension_seed_line_count": 0,
                "voxel_step2_source_line_dedup_count": 0,
                "voxel_projected_step2_source_line_map": np.zeros(shape, dtype=bool),
                "voxel_step2_projected_source_line_map": np.zeros(shape, dtype=bool),
                "voxel_step2_source_by_kind_map": np.zeros(shape, dtype=np.uint8),
                "voxel_step2_source_lines": [],
            }
        )
        step2_disabled_debug = {
            "voxel_step2_wall_line_extension_disabled": True,
            "voxel_step2_wall_line_extension_disable_reason": "config.voxel_step2.enabled=false",
            "voxel_step2_source_line_count_before_disable": int(len(step2_line_pool.source_lines)),
        }
        step2_line_pool = Step2LinePool(
            source_lines=[],
            source_line_map=np.zeros(shape, dtype=bool),
            target_wall_map=step2_line_pool.target_wall_map,
            target_source_map=step2_line_pool.target_source_map,
            debug=dict(step2_line_pool.debug),
        )
    accepted_seed_for_partition = np.zeros_like(current_accepted_seed_mask, dtype=bool)

    line_cfg = replace(
        cfg.line_extension,
        min_extension_m=0.40 if cfg.line_extension.min_extension_m is None else float(cfg.line_extension.min_extension_m),
        max_extension_m=1.60 if cfg.line_extension.max_extension_m is None else float(cfg.line_extension.max_extension_m),
        max_probe_m=min(float(cfg.line_extension.max_probe_m), 1.60),
        allow_hit_virtual_door_on_pass2=False,
        require_free_between_start_and_hit=True,
        min_free_cells_between_start_and_hit=max(3, int(cfg.line_extension.min_free_cells_between_start_and_hit)),
    )
    step2_extensions, step2_extension_debug = extend_step2_wall_lines(
        filtered_lines=step2_line_pool.source_lines,
        free_after_step1=free_after_step1,
        step1_wall_mask=step2_line_pool.target_wall_map,
        unknown_after_step1=unknown_after_step1,
        door_mask=step2_door_reject_mask,
        resolution_m=float(resolution_m),
        config=line_cfg,
        reject_if_intersects_door=bool(cfg.reject_step2_if_intersects_door),
        door_intersection_dilation_cells=int(cfg.door_intersection_dilation_cells),
    )
    _annotate_step2_extensions(
        step2_extensions,
        source_lines=step2_line_pool.source_lines,
        target_source_map=step2_line_pool.target_source_map,
        raw_seed_mask=door_seed_mask,
        current_accepted_door_mask=current_accepted_visual_mask | current_accepted_cut_mask,
        stable_door_mask=stable_door_visual_mask | stable_door_cut_mask,
        door_block_mask=step2_door_reject_mask,
        shape=shape,
    )
    step2_candidates, step2_candidate_debug = build_step2_separator_candidates_from_extensions(
        step2_extensions,
        accepted_virtual_targets=None,
        resolution_m=float(resolution_m),
        config=cfg.door_neck,
        start_id=1,
    )
    _annotate_step2_candidates(step2_candidates)
    if bool(cfg.enable_extension_intersection_fallback):
        intersection_candidates, intersection_target_map, intersection_debug = build_door_neck_candidates_from_extension_intersections(
            step2_extensions,
            free_clean=free_after_step1,
            unknown_clean=unknown_after_step1,
            resolution_m=float(resolution_m),
            line_config=line_cfg,
            door_config=cfg.door_neck,
            start_id=int(len(step2_candidates) + 1),
        )
        intersection_debug = {"voxel_step2_intersection_fallback_enabled": True, **dict(intersection_debug)}
    else:
        intersection_candidates = []
        intersection_target_map = np.zeros(shape, dtype=bool)
        intersection_debug = {
            "voxel_step2_intersection_fallback_enabled": False,
            "voxel_step2_intersection_candidate_count": 0,
            "voxel_step2_intersection_candidate_map": intersection_target_map.astype(bool),
        }
    for candidate in intersection_candidates:
        candidate.kind = "line_extension_corridor_separator"
        candidate.debug["kind_detail"] = "corridor_separator"
        candidate.debug["candidate_source"] = "step2_extension_intersection"
        candidate.debug["source_line_kind"] = "extension_intersection"
        candidate.debug["target_hit_source"] = "extension_intersection"
        candidate.debug["intersects_raw_seed"] = bool(np.any(candidate.mask(shape) & door_seed_mask))
        candidate.debug["intersects_accepted_door"] = bool(np.any(candidate.mask(shape) & (current_accepted_visual_mask | current_accepted_cut_mask)))
        candidate.debug["intersects_stable_door"] = bool(np.any(candidate.mask(shape) & (stable_door_visual_mask | stable_door_cut_mask)))
        candidate.debug["intersects_step2_door_block"] = bool(np.any(candidate.mask(shape) & step2_door_reject_mask))
        candidate.debug["extension_reject_reason"] = ""
    intersection_candidates, intersection_pre_rejected = _reject_step2_candidates_intersecting_doors(
        intersection_candidates,
        door_block_mask=step2_door_reject_mask,
        shape=shape,
    )
    all_step2_candidates = [*step2_candidates, *intersection_candidates]
    topology_cfg = replace(
        cfg.topology_test,
        reject_if_side_width_cells_leq=int(cfg.reject_step2_if_tiny_side_width_cells_leq),
        corridor_min_split_area_m2=float(cfg.voxel_step2_topology.corridor_min_split_area_m2),
        corridor_min_new_component_width_m=float(cfg.voxel_step2_topology.corridor_min_new_component_width_m),
        corridor_reject_tiny_side_width_cells_leq=int(cfg.voxel_step2_topology.corridor_reject_tiny_side_width_cells_leq),
        corridor_tiny_side_min_area_m2=float(cfg.voxel_step2_topology.corridor_tiny_side_min_area_m2),
        corridor_tiny_side_min_length_m=float(cfg.voxel_step2_topology.corridor_tiny_side_min_length_m),
        corridor_accept_long_narrow_side=bool(cfg.voxel_step2_topology.corridor_accept_long_narrow_side),
        corridor_local_topology_radius_cells=int(cfg.voxel_step2_topology.corridor_local_topology_radius_cells),
        reject_small_known_side_for_line_extensions=bool(cfg.voxel_step2_topology.reject_small_known_side_for_line_extensions),
        small_known_side_area_m2=float(cfg.voxel_step2_topology.small_known_side_area_m2),
        small_known_side_unknown_ratio_max=float(cfg.voxel_step2_topology.small_known_side_unknown_ratio_max),
        small_known_side_boundary_dilation_cells=int(cfg.voxel_step2_topology.small_known_side_boundary_dilation_cells),
    )
    accepted_step2, rejected_step2, accepted_step2_map, _raw_topology_labels, topology_debug = greedily_select_separators(
        all_step2_candidates,
        free_clean=free_after_step1,
        unknown_clean=unknown_after_step1,
        wall_candidate_clean=step2_line_pool.target_wall_map | intersection_target_map,
        corridor_skeleton=np.zeros(shape, dtype=bool),
        resolution_m=float(resolution_m),
        config=topology_cfg,
    )
    accepted_step2, rejected_step2, accepted_step2_map, corridor_local_debug = _apply_corridor_local_acceptance(
        accepted_step2,
        rejected_step2,
        accepted_step2_map,
        free_after_step1=free_after_step1,
        target_wall_map=step2_line_pool.target_wall_map | intersection_target_map,
        door_block_mask=step2_door_reject_mask,
        resolution_m=float(resolution_m),
    )
    rejected_step2 = [*intersection_pre_rejected, *rejected_step2]
    accepted_step2, post_candidate_rejected_step2, accepted_step2_map, step2_post_candidate_small_side_debug = _filter_step2_candidates_small_known_side(
        accepted_step2,
        accepted_topology_map=accepted_step2_map,
        free_clean=base_partition_free,
        unknown_clean=unknown_after_step1,
        real_wall_barrier=real_wall_barrier_for_partition,
        shape=shape,
        resolution_m=float(resolution_m),
        config=topology_cfg,
    )
    rejected_step2 = [*post_candidate_rejected_step2, *rejected_step2]
    _annotate_step2_candidates([*accepted_step2, *rejected_step2])
    if separator_memory is not None and bool(cfg.separator_memory_enabled) and bool(cfg.step2_enabled):
        stable_step2_separator_map, separator_memory_debug = separator_memory.update(accepted_step2, step=int(step), shape=shape)
    else:
        stable_step2_separator_map = np.zeros(shape, dtype=bool)
        separator_memory_debug = {
            "voxel_separator_memory_enabled": False,
            "voxel_separator_memory_track_count": 0,
            **step2_disabled_debug,
        }
    accepted_step2_map = (np.asarray(accepted_step2_map, dtype=bool) | stable_step2_separator_map).astype(bool)
    step2_candidate_map = _rasterize_candidates(all_step2_candidates, shape)
    step2_partition_cut_accepted_map, step2_partition_cut_candidate_from_accepted_map, step2_partition_cut_debug = build_step2_partition_cut_v16(
        accepted_step2,
        accepted_topology_map=accepted_step2_map,
        base_partition_free=base_partition_free,
        partition_unknown=unknown_after_step1,
        real_wall_barrier=real_wall_barrier_for_partition,
        shape=shape,
        max_unknown_bridge_cells=2,
        max_nonfree_bridge_cells=1,
    )
    step2_partition_cut_candidate_map = step2_candidate_map.astype(bool)
    step2_extension_separator_map = (step2_partition_cut_accepted_map & base_partition_free) | (stable_step2_separator_map & base_partition_free)
    step2_extension_separator_map, step2_post_cut_small_side_debug = _filter_separator_map_small_known_side(
        step2_extension_separator_map,
        free_clean=base_partition_free,
        unknown_clean=unknown_after_step1,
        resolution_m=float(resolution_m),
        config=topology_cfg,
    )
    final_virtual_separator_map = door_cut_mask | step2_extension_separator_map
    partition_free_for_label = base_partition_free & ~final_virtual_separator_map
    partition_free = partition_free_for_label.copy()
    labels, _count = ndimage.label(partition_free, structure=conn(int(cfg.final_connectivity)))
    labels = relabel_compact(labels.astype(np.int32))
    labels[unknown_after_step1] = 0
    labels[~partition_free_for_label] = 0
    labels, small_room_label_debug = _filter_small_room_labels(
        labels,
        min_area_m2=float(cfg.min_room_area_m2),
        resolution_m=float(resolution_m),
    )
    final_separator_map = real_wall_barrier_for_partition | door_cut_mask | step2_extension_separator_map

    boundary_source = np.zeros(shape, dtype=np.uint8)
    boundary_source[real_wall_barrier_for_partition] = 1
    boundary_source[step1_gap_fill_map] = 4
    boundary_source[door_cut_mask] = 2
    boundary_source[step2_extension_separator_map] = 3
    step2_layers = _extension_layers(step2_extensions, shape)
    step2_extension_reject_reason_map, step2_extension_reject_reason_legend = _reason_map_for_extensions(step2_extensions, shape)
    step2_topology_rejected_map = _rasterize_candidates(rejected_step2, shape)
    step2_candidate_reject_reason_map, step2_candidate_reject_reason_legend = _reason_map_for_candidates(rejected_step2, shape)
    step2_intersection_candidate_map = _rasterize_candidates(intersection_candidates, shape)
    step2_stage_maps = Step2StageMaps(
        extension_hits_all_map=step2_layers["all"].astype(bool),
        extension_hits_pre_topology_map=step2_layers["accepted"].astype(bool),
        separator_candidates_pre_topology_map=step2_candidate_map.astype(bool),
        topology_rejected_separator_map=step2_topology_rejected_map.astype(bool),
        accepted_separator_map=accepted_step2_map.astype(bool),
        accepted_partition_cut_map=step2_extension_separator_map.astype(bool),
    )
    step2_partition_cut_empty_count = int(len(accepted_step2)) if np.any(accepted_step2_map) and not np.any(step2_extension_separator_map) else 0
    display_wall_map = (
        np.asarray(evidence.wall_xy, dtype=bool)
        | projected_wall_map
        | step1_gap_fill_map
    )
    layers = {
        "voxel_nav_free_xy": nav_free_mask.astype(bool),
        "voxel_door_seed_no_clearance_free_xy": door_seed_free_mask.astype(bool),
        "voxel_nav_occupied_xy": nav_obstacle_mask.astype(bool),
        "voxel_nav_observed_xy": (~nav_unknown_mask).astype(bool),
        "voxel_nav_unknown_xy": nav_unknown_mask.astype(bool),
        "voxel_outside_xy": outside_xy.astype(bool),
        "voxel_outside_candidate_xy": np.asarray(evidence.debug.get("voxel_outside_candidate_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_outside_candidate_base_xy": np.asarray(evidence.debug.get("voxel_outside_candidate_base_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_outside_disconnected_candidate_xy": np.asarray(evidence.debug.get("voxel_outside_disconnected_candidate_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_outside_pre_nav_free_xy": np.asarray(evidence.debug.get("voxel_outside_pre_nav_free_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_outside_pre_traversible_xy": np.asarray(evidence.debug.get("voxel_outside_pre_traversible_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_outside_robot_component_xy": np.asarray(evidence.debug.get("voxel_outside_robot_component_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_outside_connected_to_robot_rejected_xy": np.asarray(
            evidence.debug.get("voxel_outside_connected_to_robot_rejected_xy", np.zeros(shape, dtype=bool)),
            dtype=bool,
        ),
        "voxel_outside_floor_frustum_seen_xy": np.asarray(evidence.debug.get("voxel_outside_floor_frustum_seen_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_outside_full_height_no_occupied_xy": np.asarray(evidence.debug.get("voxel_outside_full_height_no_occupied_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_outside_holdout_floor_unseen_xy": np.asarray(evidence.debug.get("voxel_outside_holdout_floor_unseen_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_outside_excluded_synthetic_free_xy": np.asarray(
            evidence.debug.get("voxel_outside_excluded_synthetic_free_xy", np.zeros(shape, dtype=bool)),
            dtype=bool,
        ),
        "voxel_outside_synthetic_free_provenance_xy": np.asarray(
            evidence.debug.get("voxel_outside_synthetic_free_provenance_xy", np.zeros(shape, dtype=bool)),
            dtype=bool,
        ),
        "voxel_outside_score_xy": np.asarray(evidence.debug.get("voxel_outside_score_xy", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_vertical_free_xy": evidence.vertical_free_xy,
        "voxel_wall_xy": evidence.wall_xy,
        "voxel_display_wall_xy": display_wall_map,
        "voxel_wall_projected_xy": projected_wall_map,
        "voxel_projected_wall_map": projected_wall_map,
        "voxel_projected_structural_wall_map": projected_wall_map,
        "voxel_anchor_projected_wall_map": anchor_projected_wall_map,
        "voxel_unknown_xy": evidence.unknown_xy,
        "voxel_unknown_dominant_xy": np.asarray(evidence.unknown_dominant_xy, dtype=bool),
        "voxel_structural_wall_seed_xy": np.asarray(evidence.structural_wall_seed_xy, dtype=bool),
        "voxel_structural_wall_ratio_xy": np.asarray(evidence.structural_wall_ratio_xy, dtype=bool),
        "voxel_wall_ratio_raw_xy": np.asarray(evidence.structural_wall_ratio_xy, dtype=bool),
        "voxel_wall_line_support_xy": np.asarray(evidence.wall_line_support_xy, dtype=bool),
        "voxel_wall_line_support_raw_xy": np.asarray(evidence.wall_line_support_raw_xy, dtype=bool),
        "voxel_wall_line_support_rejected_by_free_xy": np.asarray(evidence.wall_line_support_rejected_by_free_xy, dtype=bool),
        "voxel_wall_line_support_rejected_by_unknown_xy": np.asarray(evidence.wall_line_support_rejected_by_unknown_xy, dtype=bool),
        "voxel_wall_line_support_rejected_by_observed_xy": np.asarray(evidence.wall_line_support_rejected_by_observed_xy, dtype=bool),
        "voxel_wall_line_support_rejected_by_nav_edge_xy": np.asarray(evidence.wall_line_support_rejected_by_nav_edge_xy, dtype=bool),
        "voxel_wall_projection_support_input_xy": projection_input,
        "voxel_wall_support_raw_occupied_xy": np.asarray(evidence.wall_support_raw_occupied_xy if evidence.wall_support_raw_occupied_xy is not None else evidence.wall_line_support_raw_xy, dtype=bool),
        "voxel_wall_support_known_xy": np.asarray(evidence.wall_support_known_xy if evidence.wall_support_known_xy is not None else wall_line_support_strong_xy, dtype=bool),
        "voxel_wall_support_for_projection_xy": np.asarray(evidence.wall_support_for_projection_xy if evidence.wall_support_for_projection_xy is not None else projection_input, dtype=bool),
        "voxel_wall_support_weight_xy": projection_weight,
        "voxel_wall_support_unknown_rejected_xy": np.asarray(evidence.wall_support_unknown_rejected_xy if evidence.wall_support_unknown_rejected_xy is not None else wall_line_support_rejected_unknown_xy, dtype=bool),
        "voxel_wall_support_nav_unknown_rejected_xy": np.asarray(evidence.wall_support_nav_unknown_rejected_xy if evidence.wall_support_nav_unknown_rejected_xy is not None else np.zeros(shape, dtype=bool), dtype=bool),
        "voxel_wall_support_frontier_band_rejected_xy": np.asarray(evidence.wall_support_frontier_band_rejected_xy if evidence.wall_support_frontier_band_rejected_xy is not None else forbidden_frontier_residual, dtype=bool),
        "voxel_wall_support_free_conflict_xy": np.asarray(evidence.wall_support_free_conflict_xy if evidence.wall_support_free_conflict_xy is not None else np.zeros(shape, dtype=bool), dtype=bool),
        "voxel_frontier_unknown_band_xy": frontier_unknown_band,
        "voxel_strong_structural_support_xy": np.asarray(evidence.strong_structural_support_xy if evidence.strong_structural_support_xy is not None else projection_seed, dtype=bool),
        "voxel_bridge_only_support_xy": np.asarray(evidence.bridge_only_support_xy if evidence.bridge_only_support_xy is not None else projection_bridge, dtype=bool),
        "voxel_forbidden_frontier_residual_support_xy": forbidden_frontier_residual,
        "voxel_forbidden_unknown_boundary_support_xy": forbidden_unknown_boundary,
        "voxel_free_conflict_support_xy": np.asarray(evidence.free_conflict_support_xy if evidence.free_conflict_support_xy is not None else np.zeros(shape, dtype=bool), dtype=bool),
        "voxel_protected_structural_wall_band_xy": protected_structural_wall_band,
        "voxel_support_seed_for_projection_xy": projection_seed,
        "voxel_support_bridge_for_projection_xy": projection_bridge,
        "voxel_support_for_projection_display_xy": projection_input,
        "voxel_projected_wall_display_map": np.asarray(wall_projection.projected_wall_display_map if wall_projection.projected_wall_display_map is not None else projected_wall_map, dtype=bool),
        "voxel_projected_wall_anchor_map": anchor_projected_wall_map,
        "voxel_projected_wall_step2_source_map": np.asarray(wall_projection.projected_wall_step2_source_map if wall_projection.projected_wall_step2_source_map is not None else projected_wall_map, dtype=bool),
        "voxel_wall_projection_accumulator_h_votes": np.asarray(wall_projection.debug.get("voxel_wall_projection_accumulator_h_votes", np.zeros(shape, dtype=np.float32)), dtype=np.float32),
        "voxel_wall_projection_accumulator_v_votes": np.asarray(wall_projection.debug.get("voxel_wall_projection_accumulator_v_votes", np.zeros(shape, dtype=np.float32)), dtype=np.float32),
        "voxel_wall_projection_reject_reason_map": np.asarray(wall_projection.debug.get("voxel_wall_projection_reject_reason_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_wall_projection_step2_source_reject_reason_map": np.asarray(wall_projection.debug.get("voxel_wall_projection_step2_source_reject_reason_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_projection_hard_forbidden_mask": projection_hard_forbidden_mask,
        "voxel_wall_rejected_by_free_xy": np.asarray(evidence.wall_rejected_by_free_xy, dtype=bool),
        "voxel_wall_rejected_by_unknown_xy": np.asarray(evidence.wall_rejected_by_unknown_xy, dtype=bool),
        "voxel_nonstructural_occupied_xy": np.asarray(evidence.nonstructural_occupied_xy, dtype=bool),
        "voxel_small_unknown_hole_filled_map": np.asarray(evidence.small_unknown_hole_filled_xy, dtype=bool),
        "voxel_wall_support_unknown_gated_xy": np.asarray(evidence.wall_support_unknown_gated_xy, dtype=bool),
        "voxel_wall_support_rejected_unknown_xy": np.asarray(evidence.wall_support_rejected_unknown_xy, dtype=bool),
        "voxel_free_raw_xy": np.asarray(evidence.debug.get("voxel_free_raw_xy", evidence.vertical_free_xy), dtype=bool),
        "voxel_wall_raw_xy": np.asarray(evidence.structural_wall_ratio_xy, dtype=bool),
        "voxel_occupied_any_xy": np.asarray(evidence.occupied_any_xy, dtype=bool),
        "voxel_raw_occupied_wall_support_xy": np.asarray(evidence.raw_occupied_wall_support_xy, dtype=bool),
        "voxel_strict_raw_wall_xy": np.asarray(evidence.strict_raw_wall_xy, dtype=bool),
        "voxel_wall_suppressed_by_free_xy": np.asarray(evidence.wall_suppressed_by_free_xy, dtype=bool),
        "voxel_ratio_wall_debug_xy": np.asarray(evidence.ratio_wall_debug_xy, dtype=bool),
        "voxel_door_anchor_wall_xy": door_anchor_wall_map,
        "voxel_door_anchor_wall_union_map": door_anchor_wall_map,
        "voxel_door_anchor_source_map": door_anchor_source_map,
        "voxel_wall_anchor_support_map": partition_maps.wall_anchor_support_map,
        "voxel_door_anchor_wall_map": partition_maps.door_anchor_wall_map,
        "voxel_partition_real_wall_map": partition_maps.partition_real_wall_map,
        "voxel_partition_real_wall_removed_by_seed_carve_map": partition_maps.removed_by_seed_carve_map,
        "voxel_raw_door_seed_mask": door_acceptance.raw_seed_mask,
        "voxel_extensible_door_seed_group_mask": door_acceptance.current_accepted_seed_mask,
        "voxel_nonextensible_door_seed_mask": door_seed_mask & ~door_acceptance.current_accepted_seed_mask,
        "voxel_step2_block_mask": door_acceptance.step2_block_mask,
        "voxel_step2_block_raw_seed_removed_map": door_seed_mask & ~door_acceptance.step2_block_mask,
        "voxel_wall_carve_accepted_door_mask": door_acceptance.wall_carve_mask,
        "voxel_legacy_raw_seed_carve_would_remove_wall_map": np.asarray(partition_maps.debug.get("voxel_legacy_raw_seed_carve_would_remove_wall_map", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_partition_unknown": partition_maps.partition_unknown,
        "voxel_wall_projection_support_map": np.asarray(wall_projection.support_map, dtype=bool),
        "voxel_wall_projection_rejected_support_map": np.asarray(wall_projection.rejected_support_map, dtype=bool),
        "voxel_wall_projection_forbidden_unknown_map": np.asarray(wall_projection.debug.get("voxel_wall_projection_forbidden_unknown_map", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_free_wall_conflict_xy": np.asarray(evidence.free_wall_conflict_xy, dtype=bool),
        "voxel_vertical_observed_xy": evidence.active_observed_xy,
        "voxel_active_observed_xy": evidence.active_observed_xy,
        "voxel_line_supported_wall_map": raw_line_map,
        "voxel_filtered_wall_line_mask": filtered_line_map,
        "voxel_extension_seed_wall_line_mask": extension_seed_line_map,
        "voxel_wall_base_map": wall_base_pre_step1,
        "voxel_door_seed_mask": door_seed_mask,
        "voxel_door_raw_seed_mask": door_seed_mask,
        "voxel_door_seed_component_map": door_seed_result.door_seed_component_map,
        "voxel_door_seed_component_id_map": door_seed_result.door_seed_component_map,
        "voxel_door_seed_reject_reason_id_map": door_seed_result.door_seed_reject_reason_map,
        "voxel_door_seed_cluster_id_map": np.asarray(door_completion.debug.get("voxel_door_seed_cluster_id_map", door_completion.debug.get("voxel_door_seed_cluster_map", np.zeros(shape, dtype=np.int32))), dtype=np.int32),
        "voxel_door_seed_line_primitive_id_map": np.asarray(door_completion.debug.get("voxel_door_seed_line_primitive_id_map", np.zeros(shape, dtype=np.int32)), dtype=np.int32),
        "voxel_door_seed_line_primitive_mask": np.asarray(door_completion.debug.get("voxel_door_seed_line_primitive_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_extensible_primitive_mask": np.asarray(door_completion.debug.get("voxel_door_extensible_primitive_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_rejected_primitive_mask": np.asarray(door_completion.debug.get("voxel_door_rejected_primitive_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_extension_trials_map": np.asarray(door_completion.debug.get("voxel_door_extension_trials_map", door_completion.debug.get("voxel_door_extension_attempt_all_mask", np.zeros(shape, dtype=bool))), dtype=bool),
        "voxel_door_extension_reject_reason_id_map": np.asarray(door_completion.debug.get("voxel_door_extension_reject_reason_id_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_door_partition_reject_reason_id_map": np.asarray(door_completion.debug.get("voxel_door_partition_reject_reason_id_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_door_extensible_seed_group_mask": np.asarray(door_completion.debug.get("voxel_door_extensible_seed_group_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_extension_attempt_all_mask": door_completion.door_extension_attempt_all_mask,
        "voxel_door_extension_attempt_selected_mask": np.asarray(door_completion.debug.get("voxel_door_extension_attempt_selected_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_extension_attempt_rejected_mask": door_completion.door_extension_attempt_rejected_mask,
        "voxel_door_extension_attempt_reason_map": np.asarray(door_completion.debug.get("voxel_door_extension_attempt_reason_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_door_candidate_lines_map": door_completion.door_centerline_candidate_mask,
        "voxel_door_rejected_lines_map": door_completion.rejected_door_centerline_mask,
        "voxel_door_centerline_mask": door_visual_mask,
        "voxel_door_centerline_visual_mask": door_visual_mask,
        "voxel_door_current_centerline_visual_mask": current_door_visual_mask,
        "voxel_door_visual_only_mask": np.asarray(door_completion.debug.get("voxel_door_visual_only_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_geometry_warning_cut_mask": (current_geometry_warning_cut_mask | np.asarray(door_completion.debug.get("voxel_door_geometry_warning_cut_mask", np.zeros(shape, dtype=bool)), dtype=bool)).astype(bool),
        "voxel_door_geometry_only_mask": np.asarray(door_completion.debug.get("voxel_door_geometry_only_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_attachment_only_mask": np.asarray(door_completion.debug.get("voxel_door_attachment_only_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_cut_not_closed_to_wall_mask": np.asarray(door_completion.debug.get("voxel_door_cut_not_closed_to_wall_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_partition_effective_verified_mask": (current_topology_effective_cut_mask | stable_door_cut_mask).astype(bool),
        "voxel_door_topology_effective_cut_mask": (current_topology_effective_cut_mask | stable_door_cut_mask).astype(bool),
        "voxel_door_partition_cut_candidate_mask": door_completion.door_partition_cut_candidate_mask,
        "voxel_door_partition_cut_accepted_mask": door_cut_mask,
        "voxel_door_current_cut_mask": current_door_cut_mask,
        "voxel_current_door_cut_mask": current_accepted_cut_mask,
        "voxel_current_door_topology_effective_mask": current_topology_effective_cut_mask,
        "voxel_stable_door_cut_mask": stable_door_cut_mask,
        "voxel_stable_door_visual_mask": stable_door_visual_mask,
        "voxel_door_stable_cut_mask": stable_door_cut_mask,
        "voxel_door_memory_observed_decay_band_mask": np.asarray(door_memory_debug.get("voxel_door_memory_observed_decay_band_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_memory_unobserved_track_mask": np.asarray(door_memory_debug.get("voxel_door_memory_unobserved_track_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_memory_contradiction_mask": np.asarray(door_memory_debug.get("voxel_door_memory_contradiction_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_final_door_cut_mask": door_cut_mask,
        "voxel_door_final_cut_mask": door_cut_mask,
        "voxel_door_wall_attachment_reject_map": np.asarray(door_completion.debug.get("voxel_door_wall_attachment_reject_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_door_raw_seed_conflict_ignored_map": np.asarray(door_completion.debug.get("voxel_door_raw_seed_conflict_ignored_map", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_partition_cut_rejected_mask": np.asarray(door_completion.debug.get("voxel_door_partition_cut_rejected_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_partition_reject_reason_map": np.asarray(door_completion.debug.get("voxel_door_partition_reject_reason_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_door_reject_reason_map": door_completion.door_reject_reason_map,
        "voxel_door_cut_mask": door_cut_mask,
        "voxel_door_topology_accepted_cut_mask": np.asarray(door_completion.debug.get("voxel_door_topology_accepted_cut_mask", current_door_cut_mask), dtype=bool),
        "voxel_door_topology_warning_cut_mask": np.asarray(door_completion.debug.get("voxel_door_topology_warning_cut_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_accepted_door_centerline_mask": accepted_door_visual_mask,
        "voxel_rejected_door_centerline_mask": door_completion.rejected_door_centerline_mask,
        "voxel_step1_wall_gap_fill_map": step1_gap_fill_map,
        "voxel_wall_after_step1_map": step1_wall_mask,
        "voxel_step1_completed_wall_map": step1_wall_mask,
        "voxel_real_wall_barrier_map": real_wall_barrier_map,
        "voxel_real_wall_barrier_for_partition": real_wall_barrier_for_partition,
        "voxel_base_partition_free": base_partition_free,
        "voxel_step2_source_line_map": step2_line_pool.source_line_map,
        "voxel_projected_step2_source_line_map": np.asarray(step2_line_pool.debug.get("voxel_projected_step2_source_line_map", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_step2_projected_source_line_map": np.asarray(step2_line_pool.debug.get("voxel_step2_projected_source_line_map", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_step2_source_by_kind_map": np.asarray(step2_line_pool.debug.get("voxel_step2_source_by_kind_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_step2_target_wall_map": step2_line_pool.target_wall_map,
        "voxel_step2_target_source_map": step2_line_pool.target_source_map,
        "voxel_step2_door_reject_mask": step2_door_reject_mask,
        "voxel_step2_raw_seed_not_blocking_mask": np.asarray(door_acceptance.debug.get("voxel_step2_raw_seed_not_blocking_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_step2_extension_candidate_map": step2_layers["all"],
        "voxel_step2_extension_hits_all_map": step2_stage_maps.extension_hits_all_map,
        "voxel_step2_extension_hits_pre_topology_map": step2_stage_maps.extension_hits_pre_topology_map,
        "voxel_step2_extension_reject_reason_map": step2_extension_reject_reason_map,
        "voxel_step2_separator_candidates_pre_topology_map": step2_stage_maps.separator_candidates_pre_topology_map,
        "voxel_step2_candidate_reject_reason_map": step2_candidate_reject_reason_map,
        "voxel_step2_partition_cut_candidate_map": step2_partition_cut_candidate_map,
        "voxel_step2_partition_cut_candidate_from_accepted_map": step2_partition_cut_candidate_from_accepted_map,
        "voxel_step2_partition_cut_accepted_map": step2_extension_separator_map,
        "voxel_step2_topology_rejected_separator_map": step2_stage_maps.topology_rejected_separator_map,
        "voxel_step2_accepted_separator_map": step2_stage_maps.accepted_separator_map,
        "voxel_step2_extension_separator_map": step2_extension_separator_map,
        "voxel_stable_step2_separator_mask": stable_step2_separator_map,
        "voxel_step2_intersection_candidate_map": step2_intersection_candidate_map,
        "voxel_step2_intersection_target_map": intersection_target_map,
        "voxel_step2_rejected_extension_map": step2_layers["rejected"] | step2_topology_rejected_map,
        "voxel_boundary_source_map": boundary_source,
        "voxel_partition_free": partition_free,
        "voxel_partition_free_before_label": partition_free_for_label,
        "partition_free_for_label": partition_free_for_label,
        "voxel_final_virtual_separator_map": final_virtual_separator_map,
        "voxel_final_room_label_map": labels,
        "voxel_room_label_map_visual": labels,
        "voxel_final_separator_map": final_separator_map,
        "voxel_final_separator_source_map": boundary_source,
        "final_room_labels": labels,
        "accepted_separators": final_separator_map,
        "rejected_separators": _rasterize_candidates(rejected_step2, shape),
    }
    layers.update(_height_profile_alias_layers(layers))
    report = {
        "step": int(step),
        "backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "algorithm": VOXEL_OCCUPANCY_ROOMSEG_ALGORITHM,
        "voxel_roomseg_version": "v43_false_door_seed_ceiling_bug_fix",
        "stage_order": ["voxel_evidence", "voxel_door_seed", "wall_projection", "step1_real_wall_gap_fill", "voxel_door_completion", "step2_wall_line_extension", "final_4conn_labels"],
        "wall_segment_count": int(len(segments)),
        "filtered_wall_line_count": int(len(filtered_lines)),
        "extension_seed_wall_line_count": int(len(extension_seed_lines)),
        "voxel_step2_source_line_count": int(step2_line_pool.debug.get("voxel_step2_source_line_count", 0)),
        "voxel_step2_projected_source_line_count": int(step2_line_pool.debug.get("voxel_step2_projected_source_line_count", 0)),
        "voxel_step2_source_line_count_by_source": dict(step2_line_pool.debug.get("voxel_step2_source_line_count_by_source", {}) or {}),
        "voxel_step2_filtered_line_count": int(step2_line_pool.debug.get("voxel_step2_filtered_line_count", 0)),
        "voxel_step2_extension_seed_line_count": int(step2_line_pool.debug.get("voxel_step2_extension_seed_line_count", 0)),
        "voxel_step2_source_line_dedup_count": int(step2_line_pool.debug.get("voxel_step2_source_line_dedup_count", 0)),
        "voxel_step2_target_wall_cells": int(step2_line_pool.debug.get("voxel_step2_target_wall_cells", 0)),
        "voxel_step2_target_source_counts": dict(step2_line_pool.debug.get("voxel_step2_target_source_counts", {}) or {}),
        "voxel_wall_raw_cells": int(np.count_nonzero(evidence.structural_wall_ratio_xy)),
        "voxel_wall_ratio_raw_cells": int(np.count_nonzero(evidence.structural_wall_ratio_xy)),
        "voxel_raw_occupied_wall_support_cells": int(np.count_nonzero(evidence.raw_occupied_wall_support_xy)),
        "voxel_wall_line_support_cells": int(np.count_nonzero(evidence.wall_line_support_xy)),
        "voxel_wall_line_support_raw_cells": int(np.count_nonzero(evidence.wall_line_support_raw_xy)),
        "voxel_wall_line_support_strong_cells": int(np.count_nonzero(wall_line_support_strong_xy)),
        "voxel_wall_line_support_conflict_cells": int(np.count_nonzero(wall_line_support_conflict_xy)),
        "voxel_frontier_unknown_band_cells": int(np.count_nonzero(frontier_unknown_band)),
        "voxel_forbidden_frontier_residual_support_cells": int(np.count_nonzero(forbidden_frontier_residual)),
        "voxel_forbidden_unknown_boundary_support_cells": int(np.count_nonzero(forbidden_unknown_boundary)),
        "voxel_support_seed_for_projection_cells": int(np.count_nonzero(projection_seed)),
        "voxel_support_bridge_for_projection_cells": int(np.count_nonzero(projection_bridge)),
        "voxel_strict_raw_wall_cells": int(np.count_nonzero(evidence.strict_raw_wall_xy)),
        "voxel_wall_projected_cells": int(np.count_nonzero(projected_wall_map)),
        "voxel_projected_structural_wall_cells": int(np.count_nonzero(projected_wall_map)),
        "voxel_display_wall_cells": int(np.count_nonzero(display_wall_map)),
        "voxel_anchor_projected_wall_cells": int(np.count_nonzero(anchor_projected_wall_map)),
        "voxel_door_anchor_wall_cells": int(np.count_nonzero(door_anchor_wall_map)),
        **partition_maps.debug,
        "voxel_door_accepted_count": int(door_completion.debug.get("voxel_door_accepted_count", 0)),
        "voxel_door_visual_accepted_count": int(door_completion.debug.get("voxel_door_visual_accepted_count", 0)),
        "voxel_door_partition_accepted_count": int(door_completion.debug.get("voxel_door_partition_accepted_count", 0)),
        "voxel_door_current_cut_cells": int(np.count_nonzero(current_door_cut_mask)),
        "voxel_stable_door_cut_cells": int(np.count_nonzero(stable_door_cut_mask)),
        "voxel_door_stable_count": int(door_memory_debug.get("voxel_door_memory_track_count", 0)),
        "voxel_door_topology_effective_cells": int(np.count_nonzero(current_topology_effective_cut_mask | stable_door_cut_mask)),
        "voxel_door_partition_effective_verified_cells": int(np.count_nonzero(current_topology_effective_cut_mask | stable_door_cut_mask)),
        "voxel_door_geometry_warning_cells": int(np.count_nonzero(current_geometry_warning_cut_mask)),
        "voxel_final_door_cut_cells": int(np.count_nonzero(door_cut_mask)),
        "voxel_v30_door_partition_stability_patch": True,
        "voxel_seed_not_added_to_partition_free": True,
        "voxel_legacy_seed_free_injection_would_add_cells": int(np.count_nonzero(current_accepted_seed_mask)),
        "voxel_legacy_seed_free_injection_overlap_cut_cells": int(np.count_nonzero(current_accepted_seed_mask & door_cut_mask)),
        **door_memory_debug,
        "voxel_door_rejected_count": int(door_completion.debug.get("voxel_door_rejected_count", 0)),
        "voxel_real_wall_barrier_cells": int(np.count_nonzero(real_wall_barrier_for_partition)),
        "voxel_base_partition_free_cells": int(np.count_nonzero(base_partition_free)),
        "voxel_step1_gap_fill_cells": int(np.count_nonzero(step1_gap_fill_map)),
        "voxel_step2_extension_count": int(len(step2_extensions)),
        "voxel_step2_candidate_count": int(len(all_step2_candidates)),
        "voxel_step2_hit_candidate_count": int(len(step2_candidates)),
        "voxel_step2_intersection_candidate_count": int(len(intersection_candidates)),
        "voxel_step2_intersection_pre_rejected_count": int(len(intersection_pre_rejected)),
        "voxel_step2_accepted_count": int(len(accepted_step2)),
        "voxel_step2_rejected_count": int(len(rejected_step2)),
        "voxel_stable_step2_separator_cells": int(np.count_nonzero(stable_step2_separator_map)),
        "voxel_step2_extension_reject_reason_counts": _extension_reason_counts(step2_extensions),
        "voxel_step2_topology_reject_reason_counts": _candidate_reason_counts(rejected_step2),
        "voxel_step2_extension_reject_reason_legend": dict(step2_extension_reject_reason_legend),
        "voxel_step2_candidate_reject_reason_legend": dict(step2_candidate_reject_reason_legend),
        "voxel_step2_partition_cut_empty_count": int(step2_partition_cut_empty_count),
        "voxel_step2_partition_cut_candidate_cells": int(np.count_nonzero(step2_partition_cut_candidate_map)),
        "voxel_step2_partition_cut_accepted_cells": int(np.count_nonzero(step2_extension_separator_map)),
        "voxel_step2_partition_cut_debug": dict(step2_partition_cut_debug),
        "voxel_step2_post_candidate_small_side_debug": dict(step2_post_candidate_small_side_debug),
        "voxel_step2_post_candidate_small_side_rejected_count": int(step2_post_candidate_small_side_debug.get("rejected_candidate_count", 0)),
        "voxel_step2_post_cut_small_side_debug": dict(step2_post_cut_small_side_debug),
        "voxel_step2_post_cut_small_side_rejected_components": int(step2_post_cut_small_side_debug.get("rejected_component_count", 0)),
        "voxel_small_room_label_debug": dict(small_room_label_debug),
        "voxel_small_room_label_removed_count": int(small_room_label_debug.get("removed_label_count", 0)),
        "voxel_step2_fallback_unblocked_by_raw_seed": True,
        "voxel_step2_block_topology_effective_door_only": True,
        "voxel_step2_door_block_topology_effective_cells": int(np.count_nonzero(step2_door_reject_mask)),
        "voxel_step2_raw_seed_not_blocking_cells": int(np.count_nonzero(door_seed_mask & ~step2_door_reject_mask)),
        "voxel_step2_reject_reason_counts": _extension_and_candidate_reasons(step2_extensions, rejected_step2),
        "voxel_step2_candidate_debug_list": [candidate.to_dict() for candidate in [*accepted_step2, *rejected_step2]],
        "voxel_step2_corridor_topology_debug_per_candidate": [
            candidate.to_dict()
            for candidate in [*accepted_step2, *rejected_step2]
            if str(candidate.kind) == "line_extension_corridor_separator"
        ],
        "voxel_room_count": _room_count(labels),
        "final_connectivity": int(cfg.final_connectivity),
        "merge_small_components_enabled": bool(cfg.merge_small_components_enabled),
        "frontier_source": "voxel_vertical_free",
        "frontier_vertical_free_cells": int(np.count_nonzero(evidence.vertical_free_xy)),
        "frontier_cells_from_vertical_free": int(np.count_nonzero(evidence.vertical_free_xy)),
        "use_real_wall_as_partition_barrier": bool(cfg.use_real_wall_as_partition_barrier),
        "real_wall_barrier_dilation_cells": int(cfg.real_wall_barrier_dilation_cells),
        "candidates": [candidate.to_dict() for candidate in [*accepted_step2, *rejected_step2]],
        **door_acceptance.debug,
        **separator_memory_debug,
        **corridor_local_debug,
    }
    debug = {
        "backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "actual_backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "source_backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "roomseg_backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "algorithm": VOXEL_OCCUPANCY_ROOMSEG_ALGORITHM,
        "variant": "voxel_v33_replay_and_cpu28_speed",
        "voxel_roomseg_version": "v43_false_door_seed_ceiling_bug_fix",
        "voxel_v30_door_partition_stability_patch": True,
        "source": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "context_source": VOXEL_OCCUPANCY_ROOMSEG_CONTEXT,
        "room_map_mode": VOXEL_OCCUPANCY_ROOMSEG_CONTEXT,
        "strict_fallback_used": False,
        "silent_fallback_used": False,
        "legacy_style_used": False,
        "navigation_obstacle_written": False,
        "step": int(step),
        "resolution_m": float(resolution_m),
        "voxel_room_count": _room_count(labels),
        "room_count": _room_count(labels),
        "voxel_seed_not_added_to_partition_free": True,
        "voxel_legacy_seed_free_injection_would_add_cells": int(np.count_nonzero(current_accepted_seed_mask)),
        "voxel_legacy_seed_free_injection_overlap_cut_cells": int(np.count_nonzero(current_accepted_seed_mask & door_cut_mask)),
        "voxel_step2_fallback_unblocked_by_raw_seed": True,
        "voxel_step2_block_topology_effective_door_only": True,
        "voxel_step2_door_block_topology_effective_cells": int(np.count_nonzero(step2_door_reject_mask)),
        "voxel_step2_raw_seed_not_blocking_cells": int(np.count_nonzero(door_seed_mask & ~step2_door_reject_mask)),
        "voxel_step2_raw_seed_would_have_blocked_cells": int(np.count_nonzero(door_seed_mask & ~step2_door_reject_mask)),
        "voxel_door_stable_count": int(door_memory_debug.get("voxel_door_memory_track_count", 0)),
        "merge_small_components_enabled": bool(cfg.merge_small_components_enabled),
        "voxel_show_wall_diagnostics": bool(cfg.voxel_show_wall_diagnostics),
        "voxel_step2_reject_reason_counts": report["voxel_step2_reject_reason_counts"],
        "voxel_step2_extension_reject_reason_counts": report["voxel_step2_extension_reject_reason_counts"],
        "voxel_step2_topology_reject_reason_counts": report["voxel_step2_topology_reject_reason_counts"],
        "voxel_step2_partition_cut_empty_count": int(step2_partition_cut_empty_count),
        "voxel_step2_partition_cut_debug": dict(step2_partition_cut_debug),
        "voxel_step2_post_candidate_small_side_debug": dict(step2_post_candidate_small_side_debug),
        "voxel_step2_post_candidate_small_side_rejected_count": int(step2_post_candidate_small_side_debug.get("rejected_candidate_count", 0)),
        "voxel_step2_post_cut_small_side_debug": dict(step2_post_cut_small_side_debug),
        "voxel_step2_post_cut_small_side_rejected_components": int(step2_post_cut_small_side_debug.get("rejected_component_count", 0)),
        "voxel_small_room_label_debug": dict(small_room_label_debug),
        "voxel_small_room_label_removed_count": int(small_room_label_debug.get("removed_label_count", 0)),
        **door_acceptance.debug,
        **separator_memory_debug,
        **corridor_local_debug,
        "voxel_step2_corridor_topology_debug_per_candidate": report["voxel_step2_corridor_topology_debug_per_candidate"],
        **step2_line_pool.debug,
        "separator_report": report,
        "filtered_wall_lines_report": {
            "filtered_wall_lines": [line.to_dict() for line in filtered_lines],
            **filter_debug,
        },
        "extension_seed_wall_lines_report": {
            "filtered_wall_lines": [line.to_dict() for line in extension_seed_lines],
            **seed_filter_debug,
        },
        "line_extension_report": {"pass2": step2_extension_debug},
        "voxel_step1_gap_fill_report": step1_gap_debug,
        "voxel_wall_run_report": wall_run_debug,
        "topology_report": {"final": topology_debug, "step2_candidates": step2_candidate_debug},
        "voxel_step2_intersection_debug": intersection_debug,
        "accepted_separators": [candidate.to_dict() for candidate in accepted_step2],
        "rejected_separators": [candidate.to_dict() for candidate in rejected_step2],
        "proposal_room_masks": _proposal_masks_debug(labels),
        "frontier_source": "voxel_vertical_free",
        "frontier_vertical_free_cells": int(np.count_nonzero(evidence.vertical_free_xy)),
        "frontier_vertical_observed_cells": int(np.count_nonzero(evidence.active_observed_xy)),
        "frontier_vertical_wall_cells": int(np.count_nonzero(display_wall_map)),
        "frontier_cells_from_vertical_free": int(np.count_nonzero(evidence.vertical_free_xy)),
        "frontier_cells_removed_by_navigation_unreachable": 0,
        **voxel_grid.to_debug_dict(),
        **evidence.debug,
        **wall_projection.debug,
        **_prefixed_projection_debug(anchor_wall_projection.debug, "voxel_anchor_wall_projection"),
        **door_seed_result.debug,
        **door_completion.debug,
        **door_memory_debug,
        **separator_memory_debug,
        **door_acceptance.debug,
        **corridor_local_debug,
        **partition_maps.debug,
        **layers,
        "line_walls": wall_debug,
    }
    debug.update(_height_profile_alias_debug(debug))
    if bool(cfg.debug_dump):
        dump = save_online_roomseg_debug(
            out_dir=Path(str(cfg.debug_dir)) / ("voxel_roomseg_step_%06d" % int(step)),
            layers=layers,
            separator_report=report,
            extra_reports={
                "filtered_wall_lines": debug["filtered_wall_lines_report"],
                "line_extension_report": debug["line_extension_report"],
                "topology_report": debug["topology_report"],
            },
            save_layers=True,
            save_candidate_json=True,
        )
        debug["online_roomseg_debug_paths"] = dict(dump.get("paths", {}))
    return VoxelOccupancyDoorWallRoomSegResult(
        room_label_map=labels.astype(np.int32),
        separator_map=final_separator_map.astype(bool),
        wall_map=step1_wall_mask.astype(bool),
        door_cut_map=door_cut_mask.astype(bool),
        step1_wall_gap_fill_map=step1_gap_fill_map.astype(bool),
        step2_extension_separator_map=step2_extension_separator_map.astype(bool),
        layers=layers,
        debug=debug,
    )


def build_step2_partition_cut_v16(
    candidates: Sequence[SeparatorCandidate],
    *,
    accepted_topology_map: np.ndarray,
    base_partition_free: np.ndarray,
    partition_unknown: np.ndarray,
    real_wall_barrier: np.ndarray,
    shape: tuple[int, int],
    max_unknown_bridge_cells: int = 2,
    max_nonfree_bridge_cells: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    free = np.asarray(base_partition_free, dtype=bool)
    unknown = np.asarray(partition_unknown, dtype=bool)
    real_wall = np.asarray(real_wall_barrier, dtype=bool)
    accepted_topology = np.asarray(accepted_topology_map, dtype=bool)
    accepted_cut = accepted_topology & free
    candidate_cut = accepted_topology.copy()
    bridge_gap_count = 0
    bridge_cell_count = 0
    for candidate in candidates:
        cells = _ordered_line_cells_rc(candidate.p0_rc, candidate.p1_rc, shape)
        if len(cells) <= 0:
            continue
        valid = np.asarray([bool(free[int(r), int(c)]) for r, c in cells], dtype=bool)
        unknown_flags = np.asarray([bool(unknown[int(r), int(c)] and not real_wall[int(r), int(c)]) for r, c in cells], dtype=bool)
        nonfree_flags = np.asarray([bool((not free[int(r), int(c)]) and not unknown[int(r), int(c)] and not real_wall[int(r), int(c)]) for r, c in cells], dtype=bool)
        bridged, gap_count, bridged_indices = _bridge_step2_line_gaps_v16(
            valid,
            unknown_flags=unknown_flags,
            nonfree_flags=nonfree_flags,
            max_unknown_bridge_cells=int(max_unknown_bridge_cells),
            max_nonfree_bridge_cells=int(max_nonfree_bridge_cells),
        )
        bridge_gap_count += int(gap_count)
        bridge_cell_count += int(len(bridged_indices))
        if not np.any(bridged):
            continue
        line_mask = np.zeros(shape, dtype=bool)
        bridged_cells = cells[np.flatnonzero(bridged)]
        line_mask[bridged_cells[:, 0], bridged_cells[:, 1]] = True
        candidate_cut |= line_mask
        accepted_cut |= line_mask & free
    return accepted_cut.astype(bool), candidate_cut.astype(bool), {
        "step2_partition_cut_v16": True,
        "step2_partition_cut_bridge_gap_count": int(bridge_gap_count),
        "step2_partition_cut_bridge_cell_count": int(bridge_cell_count),
        "step2_partition_cut_candidate_cells_from_accepted": int(np.count_nonzero(candidate_cut)),
        "step2_partition_cut_accepted_cells": int(np.count_nonzero(accepted_cut)),
    }


def _filter_separator_map_small_known_side(
    separator_map: np.ndarray,
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    config: TopologyTestConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    cfg = config if isinstance(config, TopologyTestConfig) else TopologyTestConfig.from_mapping(config)
    enabled = bool(cfg.reject_small_known_side_for_line_extensions)
    area_threshold = float(cfg.small_known_side_area_m2)
    unknown_threshold = float(cfg.small_known_side_unknown_ratio_max)
    dilation_cells = max(1, int(cfg.small_known_side_boundary_dilation_cells))
    sep = np.asarray(separator_map, dtype=bool)
    free = np.asarray(free_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    debug: dict[str, object] = {
        "enabled": bool(enabled),
        "small_known_side_area_threshold_m2": float(area_threshold),
        "small_known_side_unknown_ratio_threshold": float(unknown_threshold),
        "small_known_side_boundary_dilation_cells": int(dilation_cells),
        "input_separator_cells": int(np.count_nonzero(sep)),
        "kept_separator_cells": int(np.count_nonzero(sep)),
        "rejected_separator_cells": 0,
        "component_count": 0,
        "rejected_component_count": 0,
        "components": [],
    }
    if not enabled or area_threshold <= 0.0 or not np.any(sep):
        return sep.astype(bool), debug

    labels, count = ndimage.label(sep, structure=conn(8))
    debug["component_count"] = int(count)
    kept = np.zeros_like(sep, dtype=bool)
    rejected = np.zeros_like(sep, dtype=bool)
    component_debug: list[dict[str, object]] = []
    before_labels, before_count = ndimage.label(free, structure=conn(4))
    distance_cells = ndimage.distance_transform_edt(free)
    for idx in range(1, int(count) + 1):
        comp = labels == int(idx)
        reject, reject_debug = _separator_component_small_known_side_reject(
            comp,
            free_clean=free,
            unknown_clean=unknown,
            before_component_count=int(before_count),
            distance_cells=distance_cells,
            resolution_m=float(resolution_m),
            area_threshold_m2=area_threshold,
            unknown_threshold=unknown_threshold,
            dilation_cells=dilation_cells,
        )
        entry = {
            "component_id": int(idx),
            "separator_cells": int(np.count_nonzero(comp)),
            **reject_debug,
            "rejected": bool(reject),
        }
        component_debug.append(entry)
        if reject:
            rejected |= comp
        else:
            kept |= comp
    debug["components"] = component_debug[:128]
    debug["rejected_component_count"] = int(sum(1 for item in component_debug if bool(item.get("rejected", False))))
    debug["kept_separator_cells"] = int(np.count_nonzero(kept))
    debug["rejected_separator_cells"] = int(np.count_nonzero(rejected))
    return kept.astype(bool), debug


def _filter_step2_candidates_small_known_side(
    candidates: Sequence[SeparatorCandidate],
    *,
    accepted_topology_map: np.ndarray,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    real_wall_barrier: np.ndarray,
    shape: tuple[int, int],
    resolution_m: float,
    config: TopologyTestConfig,
) -> tuple[list[SeparatorCandidate], list[SeparatorCandidate], np.ndarray, dict[str, object]]:
    cfg = config if isinstance(config, TopologyTestConfig) else TopologyTestConfig.from_mapping(config)
    debug: dict[str, object] = {
        "enabled": bool(cfg.reject_small_known_side_for_line_extensions),
        "input_candidate_count": int(len(candidates)),
        "accepted_candidate_count": int(len(candidates)),
        "rejected_candidate_count": 0,
        "candidate_debug": [],
    }
    if not bool(cfg.reject_small_known_side_for_line_extensions) or float(cfg.small_known_side_area_m2) <= 0.0:
        return list(candidates), [], np.asarray(accepted_topology_map, dtype=bool), debug
    accepted: list[SeparatorCandidate] = []
    rejected: list[SeparatorCandidate] = []
    entries: list[dict[str, object]] = []
    free = np.asarray(free_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    real_wall = np.asarray(real_wall_barrier, dtype=bool)
    before_component_count = int(ndimage.label(free, structure=conn(4))[1])
    distance_cells = ndimage.distance_transform_edt(free)
    for candidate in candidates:
        cand_mask = np.asarray(candidate.mask(shape), dtype=bool)
        cut_mask = _candidate_step2_cut_mask(
            candidate,
            free_clean=free,
            unknown_clean=unknown,
            real_wall_barrier=real_wall,
            shape=shape,
        )
        if not np.any(cut_mask):
            cut_mask = cand_mask & free
        reject, reject_debug = _separator_component_small_known_side_reject(
            cut_mask,
            free_clean=free,
            unknown_clean=unknown,
            before_component_count=before_component_count,
            distance_cells=distance_cells,
            resolution_m=float(resolution_m),
            area_threshold_m2=float(cfg.small_known_side_area_m2),
            unknown_threshold=float(cfg.small_known_side_unknown_ratio_max),
            dilation_cells=max(1, int(cfg.small_known_side_boundary_dilation_cells)),
        )
        entry = {
            "candidate_id": int(candidate.candidate_id),
            "kind": str(candidate.kind),
            "cut_cells": int(np.count_nonzero(cut_mask)),
            "rejected": bool(reject),
            **reject_debug,
        }
        entries.append(entry)
        candidate.debug["step2_post_candidate_small_side_debug"] = entry
        if reject:
            candidate.accepted = False
            candidate.reject_reason = "reject_small_known_side_low_unknown"
            candidate.debug["post_candidate_reject_reason"] = "reject_small_known_side_low_unknown"
            rejected.append(candidate)
        else:
            accepted.append(candidate)
    accepted_map = np.zeros(shape, dtype=bool)
    for candidate in accepted:
        accepted_map |= np.asarray(candidate.mask(shape), dtype=bool)
    debug["accepted_candidate_count"] = int(len(accepted))
    debug["rejected_candidate_count"] = int(len(rejected))
    debug["candidate_debug"] = entries[:256]
    return accepted, rejected, accepted_map.astype(bool), debug


def _candidate_step2_cut_mask(
    candidate: SeparatorCandidate,
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    real_wall_barrier: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    free = np.asarray(free_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    real_wall = np.asarray(real_wall_barrier, dtype=bool)
    cells = _ordered_line_cells_rc(candidate.p0_rc, candidate.p1_rc, shape)
    if len(cells) <= 0:
        return np.zeros(shape, dtype=bool)
    valid = np.asarray([bool(free[int(r), int(c)]) for r, c in cells], dtype=bool)
    unknown_flags = np.asarray([bool(unknown[int(r), int(c)] and not real_wall[int(r), int(c)]) for r, c in cells], dtype=bool)
    nonfree_flags = np.asarray([bool((not free[int(r), int(c)]) and not unknown[int(r), int(c)] and not real_wall[int(r), int(c)]) for r, c in cells], dtype=bool)
    bridged, _gap_count, _bridged_indices = _bridge_step2_line_gaps_v16(
        valid,
        unknown_flags=unknown_flags,
        nonfree_flags=nonfree_flags,
        max_unknown_bridge_cells=2,
        max_nonfree_bridge_cells=1,
    )
    out = np.zeros(shape, dtype=bool)
    if np.any(bridged):
        bridged_cells = cells[np.flatnonzero(bridged)]
        out[bridged_cells[:, 0], bridged_cells[:, 1]] = True
    return out.astype(bool)


def _separator_component_small_known_side_reject(
    cut: np.ndarray,
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    before_component_count: int,
    distance_cells: np.ndarray,
    resolution_m: float,
    area_threshold_m2: float,
    unknown_threshold: float,
    dilation_cells: int,
) -> tuple[bool, dict[str, object]]:
    cut_mask = np.asarray(cut, dtype=bool)
    free = np.asarray(free_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    debug: dict[str, object] = {
        "before_component_count": int(before_component_count),
        "after_component_count": int(before_component_count),
        "topology_touched_labels": [],
        "small_known_side_components": [],
        "small_known_side_rejected": False,
        "small_known_side_reject_reason": None,
    }
    if not np.any(cut_mask & free):
        debug["skip_reason"] = "separator_not_on_free"
        return False, debug
    after = free & ~cut_mask
    after_labels, after_count = ndimage.label(after, structure=conn(4))
    debug["after_component_count"] = int(after_count)
    if int(after_count) <= int(before_component_count):
        debug["skip_reason"] = "no_component_split"
        return False, debug
    adjacent = dilate(cut_mask, 1) & after
    touched = sorted({int(v) for v in np.unique(after_labels[adjacent]) if int(v) > 0})
    debug["topology_touched_labels"] = [int(v) for v in touched]
    if len(touched) < 2:
        debug["skip_reason"] = "less_than_two_sides"
        return False, debug

    component_info: list[dict[str, object]] = []
    for label in touched:
        side = after_labels == int(label)
        info = component_metrics(side, float(resolution_m), distance_cells)
        info["label"] = int(label)
        component_info.append(info)
    component_info.sort(key=lambda item: int(item.get("area_cells", 0)), reverse=True)
    entries: list[dict[str, object]] = []
    rejected = False
    for info in component_info[:2]:
        label = int(info.get("label", 0))
        if label <= 0:
            continue
        side = after_labels == int(label)
        area_m2 = float(info.get("area_m2", float(np.count_nonzero(side)) * float(resolution_m) ** 2))
        ring = dilate(side, dilation_cells) & ~side
        ring_cells = int(np.count_nonzero(ring))
        unknown_cells = int(np.count_nonzero(ring & unknown))
        unknown_ratio = 0.0 if ring_cells <= 0 else float(unknown_cells) / float(ring_cells)
        entry = {
            "label": int(label),
            "area_m2": float(area_m2),
            "area_cells": int(info.get("area_cells", int(np.count_nonzero(side)))),
            "boundary_cells": int(ring_cells),
            "boundary_unknown_cells": int(unknown_cells),
            "boundary_unknown_ratio": float(unknown_ratio),
            "rejected": bool(area_m2 < float(area_threshold_m2) and unknown_ratio < float(unknown_threshold)),
        }
        rejected = bool(rejected or entry["rejected"])
        entries.append(entry)
    debug["small_known_side_components"] = entries
    debug["small_known_side_rejected"] = bool(rejected)
    debug["small_known_side_reject_reason"] = "reject_small_known_side_low_unknown" if rejected else None
    return bool(rejected), debug


def _filter_small_room_labels(
    labels: np.ndarray,
    *,
    min_area_m2: float,
    resolution_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    arr = np.asarray(labels, dtype=np.int32).copy()
    min_area = float(min_area_m2)
    cell_area = max(float(resolution_m) ** 2, 1e-9)
    min_cells = int(np.ceil(min_area / cell_area)) if min_area > 0.0 else 0
    debug: dict[str, object] = {
        "enabled": bool(min_cells > 1),
        "min_area_m2": float(min_area),
        "resolution_m": float(resolution_m),
        "min_cells": int(min_cells),
        "input_label_count": int(_room_count(arr)),
        "removed_label_count": 0,
        "output_label_count": int(_room_count(arr)),
        "removed_labels": [],
    }
    if min_cells <= 1:
        return arr.astype(np.int32), debug
    removed: list[dict[str, object]] = []
    for label in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        count = int(np.count_nonzero(arr == int(label)))
        if count >= min_cells:
            continue
        arr[arr == int(label)] = 0
        removed.append(
            {
                "label": int(label),
                "area_cells": int(count),
                "area_m2": float(count * cell_area),
            }
        )
    out = relabel_compact(arr.astype(np.int32))
    debug["removed_label_count"] = int(len(removed))
    debug["output_label_count"] = int(_room_count(out))
    debug["removed_labels"] = removed[:128]
    return out.astype(np.int32), debug


def _ordered_line_cells_rc(p0_rc: np.ndarray, p1_rc: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    p0 = np.rint(np.asarray(p0_rc, dtype=np.float32)).astype(np.int32)
    p1 = np.rint(np.asarray(p1_rc, dtype=np.float32)).astype(np.int32)
    dr = int(p1[0] - p0[0])
    dc = int(p1[1] - p0[1])
    steps = max(abs(dr), abs(dc))
    if steps <= 0:
        rows = np.asarray([int(p0[0])], dtype=np.int32)
        cols = np.asarray([int(p0[1])], dtype=np.int32)
    else:
        rows = np.rint(np.linspace(int(p0[0]), int(p1[0]), steps + 1)).astype(np.int32)
        cols = np.rint(np.linspace(int(p0[1]), int(p1[1]), steps + 1)).astype(np.int32)
    inside = (rows >= 0) & (rows < int(shape[0])) & (cols >= 0) & (cols < int(shape[1]))
    if not np.any(inside):
        return np.zeros((0, 2), dtype=np.int32)
    coords = np.stack([rows[inside], cols[inside]], axis=1)
    keep = np.ones(len(coords), dtype=bool)
    if len(coords) > 1:
        keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
    return coords[keep].astype(np.int32)


def _bridge_step2_line_gaps_v16(
    valid: np.ndarray,
    *,
    unknown_flags: np.ndarray,
    nonfree_flags: np.ndarray,
    max_unknown_bridge_cells: int,
    max_nonfree_bridge_cells: int,
) -> tuple[np.ndarray, int, list[int]]:
    out = np.asarray(valid, dtype=bool).copy()
    unknown = np.asarray(unknown_flags, dtype=bool)
    nonfree = np.asarray(nonfree_flags, dtype=bool)
    if out.shape != unknown.shape or out.shape != nonfree.shape:
        raise ValueError("step2 bridge arrays must have the same shape")
    bridged: list[int] = []
    gap_count = 0
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
            and unknown_count <= int(max_unknown_bridge_cells)
            and nonfree_count <= int(max_nonfree_bridge_cells)
        ):
            out[start:end] = True
            bridged.extend(range(start, end))
            gap_count += 1
    return out, int(gap_count), bridged


def _rooms_from_labels(labels: np.ndarray, unknown: np.ndarray, config: RoomSegmentationConfig, step: int, debug: Mapping[str, object]) -> list[RoomMask]:
    out: list[RoomMask] = []
    min_cells = max(1, int(config.min_observed_free_cells))
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = np.asarray(labels == label, dtype=bool)
        if int(np.count_nonzero(mask)) < min_cells:
            continue
        room = _room_from_mask("pending", mask, unknown, [], config, int(step))
        room.source = VOXEL_OCCUPANCY_ROOMSEG_BACKEND
        room.metadata["label_id"] = int(label)
        room.metadata["proposal_labels"] = [int(label)]
        room.metadata["source_finalization_mode"] = VOXEL_OCCUPANCY_ROOMSEG_BACKEND
        room.metadata["accepted_separator_count"] = _accepted_separator_count(debug)
        out.append(room)
    return out


def _accepted_separator_count(debug: Mapping[str, object]) -> int:
    report = debug.get("separator_report")
    if isinstance(report, Mapping):
        for key in ("voxel_step2_accepted_count", "accepted_count"):
            if key in report:
                try:
                    return int(report[key])
                except (TypeError, ValueError):
                    pass
    value = debug.get("accepted_separators")
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(np.count_nonzero(value))
    try:
        return int(len(value))  # type: ignore[arg-type]
    except TypeError:
        return 0


def _accepted_door_seed_cluster_mask(cluster_map: np.ndarray, clusters: object) -> np.ndarray:
    labels = np.asarray(cluster_map, dtype=np.int32)
    accepted_ids: set[int] = set()
    if isinstance(clusters, Sequence) and not isinstance(clusters, (str, bytes)):
        for item in clusters:
            if not isinstance(item, Mapping):
                continue
            if bool(item.get("accepted_for_completion", False)):
                try:
                    accepted_ids.add(int(item.get("cluster_id", 0)))
                except (TypeError, ValueError):
                    pass
    out = np.zeros_like(labels, dtype=bool)
    for cluster_id in accepted_ids:
        if int(cluster_id) > 0:
            out |= labels == int(cluster_id)
    return out.astype(bool)


def accepted_seed_mask_from_candidates(candidates: Sequence[VoxelDoorCompletionResult] | Sequence[object], cluster_map: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    labels = np.asarray(cluster_map, dtype=np.int32)
    accepted_ids: set[int] = set()
    for candidate in candidates:
        debug = getattr(candidate, "debug", None)
        if not isinstance(debug, Mapping):
            continue
        if bool(debug.get("partition_accepted", False)) or bool(debug.get("partition_topology_effective", False)):
            raw_id = debug.get("seed_group_id", debug.get("cluster_id", None))
            try:
                cluster_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if cluster_id > 0:
                accepted_ids.add(cluster_id)
    out = np.zeros(shape, dtype=bool)
    for cluster_id in accepted_ids:
        out |= labels == int(cluster_id)
    return out.astype(bool)


def _cells_to_mask(cells: Sequence[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in cells:
        rr, cc = int(r), int(c)
        if 0 <= rr < int(shape[0]) and 0 <= cc < int(shape[1]):
            out[rr, cc] = True
    return out


def _mask_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(arr))]


def projected_wall_line_to_filtered_wall_line(line: ProjectedWallLine, resolution_m: float, line_id: int | None = None) -> FilteredWallLine:
    if str(line.axis) == "h":
        p0 = np.asarray([int(line.line), int(line.start)], dtype=np.float32)
        p1 = np.asarray([int(line.line), int(line.end)], dtype=np.float32)
        theta = 0.0
    elif str(line.axis) == "v":
        p0 = np.asarray([int(line.start), int(line.line)], dtype=np.float32)
        p1 = np.asarray([int(line.end), int(line.line)], dtype=np.float32)
        theta = float(np.pi / 2.0)
    else:
        raise ValueError("projected wall line axis must be h or v")
    length_m = float((np.linalg.norm(p1 - p0) + 1.0) * float(resolution_m))
    support = float(line.support_ratio)
    return FilteredWallLine(
        line_id=int(line_id if line_id is not None else line.line_id),
        p0_rc=p0,
        p1_rc=p1,
        theta=float(theta),
        normal_theta=float(theta + np.pi / 2.0),
        length_m=length_m,
        support_ratio=support,
        mean_wall_score=support,
        thickness_m=float(resolution_m),
        source_segment_ids=[int(line.line_id)],
        endpoint_quality={"p0_support_m": float(resolution_m), "p1_support_m": float(resolution_m)},
        confidence=float(min(1.0, max(0.0, support))),
        debug={"source": "projected_step2_source", "projected_wall_line": line.to_dict()},
    )


def build_voxel_partition_maps(
    *,
    evidence: VoxelRoomsegEvidence,
    door_seed_mask: np.ndarray,
    wall_carve_mask: np.ndarray | None = None,
    strict_raw_wall: np.ndarray,
    projected_wall_map: np.ndarray,
    anchor_projected_wall_map: np.ndarray,
    filtered_line_map: np.ndarray,
    extension_seed_line_map: np.ndarray,
    step1_gap_fill_map: np.ndarray,
    cfg: VoxelOccupancyDoorWallRoomSegConfig,
) -> PartitionMapBundle:
    seed = np.asarray(door_seed_mask, dtype=bool)
    strict = np.asarray(strict_raw_wall, dtype=bool)
    projected = np.asarray(projected_wall_map, dtype=bool)
    anchor_projected = np.asarray(anchor_projected_wall_map, dtype=bool)
    filtered = np.asarray(filtered_line_map, dtype=bool)
    extension_seed = np.asarray(extension_seed_line_map, dtype=bool)
    step1 = np.asarray(step1_gap_fill_map, dtype=bool)
    raw_support = np.asarray(evidence.raw_occupied_wall_support_xy, dtype=bool)
    unknown_dominant = np.asarray(getattr(evidence, "unknown_dominant_xy", np.zeros_like(strict, dtype=bool)), dtype=bool)
    unknown_rejected = np.asarray(getattr(evidence, "wall_rejected_by_unknown_xy", np.zeros_like(strict, dtype=bool)), dtype=bool)
    vertical_free = np.asarray(evidence.vertical_free_xy, dtype=bool)
    shape = vertical_free.shape
    for name, arr in {
        "strict_raw_wall": strict,
        "projected_wall_map": projected,
        "anchor_projected_wall_map": anchor_projected,
        "filtered_line_map": filtered,
        "extension_seed_line_map": extension_seed,
        "step1_gap_fill_map": step1,
        "door_seed_mask": seed,
    }.items():
        if arr.shape != shape:
            raise ValueError("%s must match voxel evidence shape" % name)
    seed_carve_radius = int(getattr(cfg, "door_seed_wall_carve_radius_cells", cfg.door.seed_cluster_morph_close_radius_cells))
    legacy_seed_carve = dilate(seed, max(0, seed_carve_radius))
    carve = np.zeros(shape, dtype=bool) if wall_carve_mask is None else np.asarray(wall_carve_mask, dtype=bool)
    if carve.shape != shape:
        raise ValueError("wall_carve_mask must match voxel evidence shape")
    clean_structural_wall = strict & ~vertical_free
    projected_structural_wall = projected & ~vertical_free
    step1_structural_wall = step1 & ~vertical_free
    validated_extension_seed = extension_seed & ~vertical_free
    wall_anchor_support_map = clean_structural_wall | projected_structural_wall | step1_structural_wall
    door_anchor_wall_map = wall_anchor_support_map.copy()
    step2_target_wall_map = wall_anchor_support_map | validated_extension_seed
    partition_real_wall_raw = clean_structural_wall | projected_structural_wall | step1_structural_wall
    if not bool(cfg.use_real_wall_as_partition_barrier):
        partition_real_wall_raw = np.zeros(shape, dtype=bool)
    partition_real_wall_map = np.asarray(partition_real_wall_raw, dtype=bool).copy()
    legacy_removed_by_seed_carve = partition_real_wall_map & legacy_seed_carve
    removed_by_seed_carve = partition_real_wall_map & carve
    partition_real_wall_map &= ~carve
    base_partition_free = vertical_free & ~partition_real_wall_map
    partition_unknown = ~(base_partition_free | partition_real_wall_map)
    old_raw_support_loss = vertical_free & raw_support & ~partition_real_wall_map
    debug = {
        "voxel_partition_maps_stage": "v19_wall_recovery_partition_sources",
        "voxel_wall_anchor_support_cells": int(np.count_nonzero(wall_anchor_support_map)),
        "voxel_door_anchor_wall_cells": int(np.count_nonzero(door_anchor_wall_map)),
        "voxel_door_anchor_allowed_sources": ["structural_wall", "projected_structural_wall", "step1"],
        "voxel_step2_target_allowed_sources": ["structural_wall", "projected_structural_wall", "step1", "validated_extension_seed"],
        "voxel_door_anchor_unknown_overlap_cells": int(np.count_nonzero(door_anchor_wall_map & partition_unknown)),
        "voxel_step2_target_wall_cells_v18": int(np.count_nonzero(step2_target_wall_map)),
        "voxel_step2_target_wall_cells_v15": int(np.count_nonzero(step2_target_wall_map)),
        "voxel_step2_target_wall_unknown_rejected_cells": int(np.count_nonzero((strict | projected | anchor_projected | filtered | extension_seed | step1) & unknown_dominant)),
        "voxel_step2_false_frontier_wall_rejected_map": unknown_rejected.astype(bool),
        "voxel_partition_extension_seed_excluded_cells": int(np.count_nonzero(extension_seed & ~partition_real_wall_map)),
        "voxel_partition_filtered_line_excluded_cells": int(np.count_nonzero(filtered & ~partition_real_wall_map)),
        "voxel_partition_anchor_projection_excluded_cells": int(np.count_nonzero(anchor_projected & ~partition_real_wall_map)),
        "voxel_partition_real_wall_cells": int(np.count_nonzero(partition_real_wall_map)),
        "voxel_raw_seed_carve_disabled": True,
        "voxel_raw_seed_carve_cells_legacy_would_remove": int(np.count_nonzero(legacy_removed_by_seed_carve)),
        "voxel_partition_real_wall_removed_by_raw_seed_carve_map": np.zeros(shape, dtype=bool),
        "voxel_legacy_raw_seed_carve_would_remove_wall_map": legacy_removed_by_seed_carve.astype(bool),
        "voxel_wall_carve_accepted_door_cells": int(np.count_nonzero(carve)),
        "voxel_wall_carve_accepted_door_mask": carve.astype(bool),
        "voxel_partition_real_wall_removed_by_seed_carve_cells": int(np.count_nonzero(removed_by_seed_carve)),
        "voxel_base_partition_free_cells": int(np.count_nonzero(base_partition_free)),
        "voxel_base_partition_free_lost_to_old_raw_support_cells": int(np.count_nonzero(old_raw_support_loss)),
        "voxel_seed_wall_carve_radius_cells": int(seed_carve_radius),
    }
    return PartitionMapBundle(
        wall_anchor_support_map=wall_anchor_support_map.astype(bool),
        door_anchor_wall_map=door_anchor_wall_map.astype(bool),
        step2_target_wall_map=step2_target_wall_map.astype(bool),
        partition_real_wall_map=partition_real_wall_map.astype(bool),
        base_partition_free=base_partition_free.astype(bool),
        partition_unknown=partition_unknown.astype(bool),
        removed_by_seed_carve_map=removed_by_seed_carve.astype(bool),
        debug=debug,
    )


def build_step2_line_pool(
    *,
    filtered_lines: Sequence[FilteredWallLine],
    extension_seed_lines: Sequence[FilteredWallLine],
    projected_step2_lines: Sequence[FilteredWallLine] | None = None,
    strict_raw_wall: np.ndarray,
    projected_wall_map: np.ndarray,
    anchor_projected_wall_map: np.ndarray,
    step1_completed_wall_map: np.ndarray,
    filtered_line_map: np.ndarray,
    extension_seed_line_map: np.ndarray,
    shape: tuple[int, int],
    resolution_m: float,
    target_wall_override: np.ndarray | None = None,
) -> Step2LinePool:
    filtered_input = _tag_step2_lines(filtered_lines, "filtered")
    projected_lines = _tag_step2_lines(projected_step2_lines or [], "projected_step2_source")
    extension_input = _tag_step2_lines(extension_seed_lines, "extension_seed")
    source_lines, dedup_count = _dedup_step2_lines([*filtered_input, *projected_lines, *extension_input])
    source_line_map = filtered_wall_line_mask(source_lines, shape)
    projected_step2_line_map = filtered_wall_line_mask(projected_lines, shape)
    target_source = np.zeros(shape, dtype=np.uint8)
    target_source[np.asarray(strict_raw_wall, dtype=bool)] = 1
    target_source[np.asarray(projected_wall_map, dtype=bool)] = 2
    target_source[np.asarray(anchor_projected_wall_map, dtype=bool)] = 3
    target_source[np.asarray(step1_completed_wall_map, dtype=bool)] = 4
    target_source[np.asarray(filtered_line_map, dtype=bool)] = 5
    target_source[np.asarray(extension_seed_line_map, dtype=bool)] = 6
    target_source[projected_step2_line_map] = 7
    target_wall = target_source > 0
    if target_wall_override is not None:
        target_wall = np.asarray(target_wall_override, dtype=bool)
        clean_source = np.zeros(shape, dtype=np.uint8)
        clean_source[np.asarray(strict_raw_wall, dtype=bool) & target_wall] = 1
        clean_source[np.asarray(projected_wall_map, dtype=bool) & target_wall] = 2
        clean_source[np.asarray(step1_completed_wall_map, dtype=bool) & target_wall] = 4
        clean_source[np.asarray(extension_seed_line_map, dtype=bool) & target_wall & (clean_source == 0)] = 6
        clean_source[projected_step2_line_map & target_wall & (clean_source == 0)] = 7
        target_source = clean_source
    source_counts = {
        "strict_raw_wall": int(np.count_nonzero(target_source == 1)),
        "projected_wall": int(np.count_nonzero(target_source == 2)),
        "anchor_projected_wall": int(np.count_nonzero(target_source == 3)),
        "step1_completed_wall": int(np.count_nonzero(target_source == 4)),
        "filtered_wall_line": int(np.count_nonzero(target_source == 5)),
        "extension_seed_line": int(np.count_nonzero(target_source == 6)),
        "projected_step2_source": int(np.count_nonzero(target_source == 7)),
    }
    line_source_counts = Counter(str(line.debug.get("step2_source_kind", line.debug.get("source", "unknown"))) for line in source_lines)
    if projected_lines:
        line_source_counts["projected_step2_source"] = int(len(projected_lines))
    source_kind_map = np.zeros(shape, dtype=np.uint8)
    source_kind_map[np.asarray(filtered_line_map, dtype=bool)] = 1
    source_kind_map[projected_step2_line_map] = 2
    source_kind_map[np.asarray(extension_seed_line_map, dtype=bool)] = 3
    debug = {
        "voxel_step2_projected_source_line_count": int(len(projected_lines)),
        "voxel_step2_source_line_count_by_source": dict(line_source_counts),
        "voxel_projected_step2_source_line_map": projected_step2_line_map.astype(bool),
        "voxel_step2_projected_source_line_map": projected_step2_line_map.astype(bool),
        "voxel_step2_source_by_kind_map": source_kind_map.astype(np.uint8),
        "voxel_step2_source_line_count": int(len(source_lines)),
        "voxel_step2_filtered_line_count": int(len(filtered_lines)),
        "voxel_step2_extension_seed_line_count": int(len(extension_seed_lines)),
        "voxel_step2_source_line_dedup_count": int(dedup_count),
        "voxel_step2_target_wall_cells": int(np.count_nonzero(target_wall)),
        "voxel_step2_target_source_counts": source_counts,
        "voxel_step2_source_lines": [line.to_dict() for line in source_lines[:1024]],
        "voxel_step2_line_pool_resolution_m": float(resolution_m),
    }
    return Step2LinePool(
        source_lines=list(source_lines),
        source_line_map=source_line_map.astype(bool),
        target_wall_map=target_wall.astype(bool),
        target_source_map=target_source.astype(np.uint8),
        debug=debug,
    )


def _tag_step2_lines(lines: Sequence[FilteredWallLine], kind: str) -> list[FilteredWallLine]:
    out = list(lines)
    for line in out:
        line.debug = dict(line.debug)
        line.debug["step2_source_kind"] = str(kind)
    return out


_STEP2_TARGET_SOURCE_NAMES = {
    0: "none",
    1: "strict",
    2: "projected",
    3: "anchor_projected",
    4: "step1",
    5: "filtered",
    6: "extension_seed",
    7: "projected_step2_source",
}


def _annotate_step2_extensions(
    extensions: Sequence[LineExtensionHit],
    *,
    source_lines: Sequence[FilteredWallLine],
    target_source_map: np.ndarray,
    raw_seed_mask: np.ndarray,
    current_accepted_door_mask: np.ndarray,
    stable_door_mask: np.ndarray,
    door_block_mask: np.ndarray,
    shape: tuple[int, int],
) -> None:
    source_kind_by_id: dict[int, str] = {}
    for line in source_lines:
        kind = str(line.debug.get("step2_source_kind", line.debug.get("source", "filtered")))
        line_id = int(line.line_id)
        if line_id in source_kind_by_id and source_kind_by_id[line_id] != kind:
            source_kind_by_id[line_id] = "mixed"
        else:
            source_kind_by_id[line_id] = kind
    target_source = np.asarray(target_source_map, dtype=np.uint8)
    raw_seed = np.asarray(raw_seed_mask, dtype=bool)
    current_door = np.asarray(current_accepted_door_mask, dtype=bool)
    stable_door = np.asarray(stable_door_mask, dtype=bool)
    door_block = np.asarray(door_block_mask, dtype=bool)
    for hit in extensions:
        line_mask = rasterize_line(hit.p_start_rc, hit.p_hit_rc, shape)
        target_source_name = _target_source_name_at(target_source, hit.p_hit_rc)
        hit.debug["source_line_kind"] = source_kind_by_id.get(int(hit.source_line_id), "unknown")
        hit.debug["target_hit_source"] = target_source_name
        hit.debug["intersects_raw_seed"] = bool(np.any(line_mask & raw_seed))
        hit.debug["intersects_accepted_door"] = bool(np.any(line_mask & current_door))
        hit.debug["intersects_stable_door"] = bool(np.any(line_mask & stable_door))
        hit.debug["intersects_step2_door_block"] = bool(np.any(line_mask & door_block))
        hit.debug["extension_reject_reason"] = "" if hit.reject_reason is None else str(hit.reject_reason)


def _annotate_step2_candidates(candidates: Sequence[SeparatorCandidate]) -> None:
    for candidate in candidates:
        extension = candidate.debug.get("extension")
        extension_debug = extension.get("debug", {}) if isinstance(extension, Mapping) else {}
        if isinstance(extension_debug, Mapping):
            for key in (
                "source_line_kind",
                "target_hit_source",
                "intersects_raw_seed",
                "intersects_accepted_door",
                "intersects_stable_door",
                "intersects_step2_door_block",
                "extension_reject_reason",
            ):
                if key in extension_debug:
                    candidate.debug[key] = extension_debug[key]
        candidate.debug["topology_reject_reason"] = "" if not candidate.reject_reason else str(candidate.reject_reason)
        candidate.debug["corridor_local_acceptance"] = bool(candidate.debug.get("corridor_local_acceptance_accepted", False))


def _target_source_name_at(target_source_map: np.ndarray, point_rc: np.ndarray) -> str:
    arr = np.asarray(target_source_map, dtype=np.uint8)
    rc = np.rint(np.asarray(point_rc, dtype=np.float32)).astype(np.int32)
    r, c = int(rc[0]), int(rc[1])
    if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
        code = int(arr[r, c])
        if code:
            return _STEP2_TARGET_SOURCE_NAMES.get(code, "unknown")
    r0, r1 = max(0, r - 1), min(arr.shape[0], r + 2)
    c0, c1 = max(0, c - 1), min(arr.shape[1], c + 2)
    if r0 < r1 and c0 < c1:
        values, counts = np.unique(arr[r0:r1, c0:c1], return_counts=True)
        pairs = [(int(v), int(n)) for v, n in zip(values.tolist(), counts.tolist()) if int(v) > 0]
        if pairs:
            code = max(pairs, key=lambda item: item[1])[0]
            return _STEP2_TARGET_SOURCE_NAMES.get(int(code), "unknown")
    return "none"


def _reason_map_for_extensions(extensions: Sequence[LineExtensionHit], shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, int]]:
    out = np.zeros(shape, dtype=np.uint8)
    legend: dict[str, int] = {}
    next_code = 1
    for hit in extensions:
        if hit.reject_reason is None:
            continue
        reason = str(hit.reject_reason)
        if reason not in legend:
            legend[reason] = int(next_code)
            next_code = min(255, next_code + 1)
        out[rasterize_line(hit.p_start_rc, hit.p_hit_rc, shape)] = np.uint8(legend[reason])
    return out, legend


def _reason_map_for_candidates(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, int]]:
    out = np.zeros(shape, dtype=np.uint8)
    legend: dict[str, int] = {}
    next_code = 1
    for candidate in candidates:
        if not candidate.reject_reason:
            continue
        reason = str(candidate.reject_reason)
        if reason not in legend:
            legend[reason] = int(next_code)
            next_code = min(255, next_code + 1)
        out[candidate.mask(shape)] = np.uint8(legend[reason])
    return out, legend


def _reject_step2_candidates_intersecting_doors(
    candidates: Sequence[SeparatorCandidate],
    *,
    door_block_mask: np.ndarray,
    shape: tuple[int, int],
) -> tuple[list[SeparatorCandidate], list[SeparatorCandidate]]:
    door_block = np.asarray(door_block_mask, dtype=bool)
    kept: list[SeparatorCandidate] = []
    rejected: list[SeparatorCandidate] = []
    for candidate in candidates:
        mask = candidate.mask(shape)
        if np.any(mask & door_block):
            candidate.accepted = False
            candidate.reject_reason = "reject_partition_intersects_existing_separator_or_door"
            candidate.debug["pre_topology_reject_reason"] = candidate.reject_reason
            candidate.debug["intersects_step2_door_block"] = True
            rejected.append(candidate)
        else:
            kept.append(candidate)
    return kept, rejected


def _apply_corridor_local_acceptance(
    accepted: Sequence[SeparatorCandidate],
    rejected: Sequence[SeparatorCandidate],
    accepted_map: np.ndarray,
    *,
    free_after_step1: np.ndarray,
    target_wall_map: np.ndarray,
    door_block_mask: np.ndarray,
    resolution_m: float,
) -> tuple[list[SeparatorCandidate], list[SeparatorCandidate], np.ndarray, dict[str, object]]:
    free = np.asarray(free_after_step1, dtype=bool)
    target = np.asarray(target_wall_map, dtype=bool)
    door_block = np.asarray(door_block_mask, dtype=bool)
    out_accepted = list(accepted)
    out_rejected: list[SeparatorCandidate] = []
    out_map = np.asarray(accepted_map, dtype=bool).copy()
    locally_accepted = 0
    checked = 0
    reject_counts: Counter[str] = Counter()
    for candidate in rejected:
        mask = candidate.mask(free.shape)
        candidate.debug["corridor_local_acceptance_checked"] = True
        checked += 1
        reason = str(candidate.reject_reason or "")
        topology_like = reason in {
            "reject_no_topology_gain",
            "reject_no_component_split",
            "reject_tiny_fragment",
            "reject_no_new_component",
            "reject_local_fragment_too_small",
            "reject_split_tiny_side_width_1_to_3_cells",
            "reject_too_many_tiny_fragments",
            "reject_side_area_too_small",
            "reject_no_two_sides",
            "reject_tiny_side",
        }
        neck_ok, neck_debug = validate_step2_wall_to_wall_neck_v30(
            candidate,
            free=free,
            target_wall=target,
            door_block=door_block,
            resolution_m=float(resolution_m),
        )
        candidate.debug.update(neck_debug)
        if topology_like and bool(neck_ok):
            candidate.accepted = True
            candidate.reject_reason = ""
            candidate.debug["corridor_local_acceptance_accepted"] = True
            candidate.debug["corridor_local_acceptance_reason"] = "accepted_wall_to_wall_neck_v30"
            out_accepted.append(candidate)
            out_map |= mask
            locally_accepted += 1
        else:
            candidate.debug["corridor_local_acceptance_accepted"] = False
            candidate.debug["corridor_local_acceptance_reason"] = (
                "not_topology_reject"
                if not topology_like
                else str(neck_debug.get("step2_local_neck_v30_reject_reason", "local_neck_rejected"))
            )
            reject_counts[str(candidate.debug["corridor_local_acceptance_reason"])] += 1
            out_rejected.append(candidate)
    return out_accepted, out_rejected, out_map.astype(bool), {
        "voxel_step2_corridor_local_acceptance_checked_count": int(checked),
        "voxel_step2_corridor_local_acceptance_accepted_count": int(locally_accepted),
        "voxel_step2_local_neck_v30_accepted_count": int(locally_accepted),
        "voxel_step2_local_neck_v30_reject_counts": dict(reject_counts),
    }


def validate_step2_wall_to_wall_neck_v30(
    candidate: SeparatorCandidate,
    *,
    free: np.ndarray,
    target_wall: np.ndarray,
    door_block: np.ndarray,
    resolution_m: float,
) -> tuple[bool, dict[str, object]]:
    mask = candidate.mask(np.asarray(free, dtype=bool).shape)
    free_arr = np.asarray(free, dtype=bool)
    target = np.asarray(target_wall, dtype=bool)
    door = np.asarray(door_block, dtype=bool)
    length_ok = 0.40 <= float(candidate.length_m) <= 1.60
    no_door = not bool(np.any(mask & door))
    crosses_free = bool(np.any(mask & free_arr))
    target_touch_count = int(np.count_nonzero(dilate(mask, 1) & target))
    target_touch = bool(target_touch_count >= 2)
    before = free_arr & ~target
    roi = _mask_bbox_local(dilate(mask, max(2, int(round(0.25 / max(float(resolution_m), 1e-6))))), before.shape)
    r0, r1, c0, c1 = roi
    before_local = before[r0:r1, c0:c1]
    cut_local = mask[r0:r1, c0:c1]
    before_labels, before_n = ndimage.label(before_local, structure=conn(4))
    after_local = before_local & ~cut_local
    after_labels, after_n = ndimage.label(after_local, structure=conn(4))
    adjacent = dilate(cut_local, 1) & after_local
    touched = sorted({int(v) for v in np.unique(after_labels[adjacent]) if int(v) > 0})
    side_areas = [int(np.count_nonzero(after_labels == int(label))) for label in touched]
    side_ok = len(touched) >= 2 and not any(area < 2 for area in side_areas[:2])
    local_gain = int(after_n) > int(before_n)
    ok = bool(length_ok and no_door and crosses_free and target_touch and (local_gain or side_ok))
    reject_reason = None
    if not ok:
        if not length_ok:
            reject_reason = "length_out_of_range"
        elif not no_door:
            reject_reason = "intersects_verified_door"
        elif not crosses_free:
            reject_reason = "does_not_cross_free"
        elif not target_touch:
            reject_reason = "does_not_touch_target_wall_twice"
        else:
            reject_reason = "no_local_wall_to_wall_neck_split"
    return ok, {
        "step2_local_neck_v30_checked": True,
        "step2_local_neck_v30_length_ok": bool(length_ok),
        "step2_local_neck_v30_no_door_intersection": bool(no_door),
        "step2_local_neck_v30_crosses_free": bool(crosses_free),
        "step2_local_neck_v30_target_touch_count": int(target_touch_count),
        "step2_local_neck_v30_local_gain": bool(local_gain),
        "step2_local_neck_v30_side_ok": bool(side_ok),
        "step2_local_neck_v30_touched_labels": [int(v) for v in touched],
        "step2_local_neck_v30_side_areas": [int(v) for v in side_areas],
        "step2_local_neck_v30_reject_reason": reject_reason,
    }


def _mask_bbox_local(mask: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return 0, int(shape[0]), 0, int(shape[1])
    return (
        max(0, int(rows.min())),
        min(int(shape[0]), int(rows.max()) + 1),
        max(0, int(cols.min())),
        min(int(shape[1]), int(cols.max()) + 1),
    )


def _dedup_step2_lines(lines: Sequence[FilteredWallLine]) -> tuple[list[FilteredWallLine], int]:
    kept: list[FilteredWallLine] = []
    dropped = 0
    for line in lines:
        duplicate_index = None
        for idx, old in enumerate(kept):
            if _step2_lines_equivalent(line, old):
                duplicate_index = idx
                break
        if duplicate_index is None:
            kept.append(line)
            continue
        dropped += 1
        old = kept[int(duplicate_index)]
        if (float(line.confidence), float(line.length_m)) > (float(old.confidence), float(old.length_m)):
            kept[int(duplicate_index)] = line
    for idx, line in enumerate(kept, start=1):
        line.debug = dict(line.debug)
        line.debug["step2_line_pool_id"] = int(idx)
    return kept, int(dropped)


def _step2_lines_equivalent(a: FilteredWallLine, b: FilteredWallLine) -> bool:
    a0 = np.rint(np.asarray(a.p0_rc, dtype=np.float32)).astype(np.int32)
    a1 = np.rint(np.asarray(a.p1_rc, dtype=np.float32)).astype(np.int32)
    b0 = np.rint(np.asarray(b.p0_rc, dtype=np.float32)).astype(np.int32)
    b1 = np.rint(np.asarray(b.p1_rc, dtype=np.float32)).astype(np.int32)
    endpoints_same = max(_cheb(a0, b0), _cheb(a1, b1)) <= 1
    endpoints_reversed = max(_cheb(a0, b1), _cheb(a1, b0)) <= 1
    if not (endpoints_same or endpoints_reversed):
        return False
    angle = abs(float(a.theta) - float(b.theta))
    angle = min(angle, abs(float(np.pi) - angle))
    if float(np.degrees(angle)) >= 5.0:
        return False
    if set(int(v) for v in a.source_segment_ids) & set(int(v) for v in b.source_segment_ids):
        return True
    return endpoints_same or endpoints_reversed


def _cheb(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.max(np.abs(np.asarray(a, dtype=np.int32) - np.asarray(b, dtype=np.int32))))


def _extension_layers(extensions: Sequence[LineExtensionHit], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    out = {
        "all": np.zeros(shape, dtype=bool),
        "accepted": np.zeros(shape, dtype=bool),
        "rejected": np.zeros(shape, dtype=bool),
    }
    for hit in extensions:
        line = rasterize_line(hit.p_start_rc, hit.p_hit_rc, shape)
        out["all"] |= line
        if hit.reject_reason is None:
            out["accepted"] |= line
        else:
            out["rejected"] |= line
    return out


def _rasterize_candidates(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        out |= candidate.mask(shape)
    return out.astype(bool)


def _door_anchor_source_map(
    shape: tuple[int, int],
    *,
    strict_raw_wall: np.ndarray,
    projected_wall: np.ndarray,
    anchor_projected_wall: np.ndarray,
    step1_gap_fill: np.ndarray,
    filtered_line: np.ndarray,
) -> np.ndarray:
    source = np.zeros(shape, dtype=np.uint8)
    source[np.asarray(filtered_line, dtype=bool)] = DOOR_ANCHOR_FILTERED_LINE
    source[np.asarray(anchor_projected_wall, dtype=bool)] = DOOR_ANCHOR_PROJECTED_ANCHOR
    source[np.asarray(step1_gap_fill, dtype=bool)] = DOOR_ANCHOR_STEP1
    source[np.asarray(projected_wall, dtype=bool)] = DOOR_ANCHOR_PROJECTED
    source[np.asarray(strict_raw_wall, dtype=bool)] = DOOR_ANCHOR_STRICT_RAW
    return source


def _prefixed_projection_debug(debug: Mapping[str, object], prefix: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in debug.items():
        if str(key).startswith("voxel_wall_projection_"):
            suffix = str(key)[len("voxel_wall_projection_") :]
            out[f"{prefix}_{suffix}"] = value
    return out


def _room_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(labels) if int(v) > 0]))


def _extension_reason_counts(extensions: Sequence[LineExtensionHit]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for hit in extensions:
        if hit.reject_reason is not None:
            counts[str(hit.reject_reason)] += 1
    return dict(counts)


def _extension_and_candidate_reasons(extensions: Sequence[LineExtensionHit], rejected_candidates: Sequence[SeparatorCandidate]) -> dict[str, int]:
    counts: Counter[str] = Counter(_extension_reason_counts(extensions))
    for candidate in rejected_candidates:
        counts[str(candidate.reject_reason)] += 1
    return dict(counts)


def _candidate_reason_counts(candidates: Sequence[SeparatorCandidate]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        counts[str(candidate.reject_reason)] += 1
    return dict(counts)


def _height_profile_alias_layers(layers: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    alias = {
        "height_profile_vertical_free_xy": "voxel_vertical_free_xy",
        "height_profile_wall_xy": "voxel_wall_xy",
        "height_profile_unknown_xy": "voxel_unknown_xy",
        "height_profile_vertical_observed_xy": "voxel_vertical_observed_xy",
        "height_profile_wall_base_mask": "voxel_wall_base_map",
        "height_profile_line_supported_wall_mask": "voxel_line_supported_wall_map",
        "height_profile_filtered_wall_line_mask": "voxel_filtered_wall_line_mask",
        "height_profile_door_seed_mask": "voxel_door_seed_mask",
        "height_profile_door_cut_mask": "voxel_door_cut_mask",
        "height_profile_accepted_door_centerline_mask": "voxel_accepted_door_centerline_mask",
        "height_profile_rejected_door_centerline_mask": "voxel_rejected_door_centerline_mask",
        "height_profile_step1_wall_gap_fill_map": "voxel_step1_wall_gap_fill_map",
        "height_profile_step1_wall_mask": "voxel_wall_after_step1_map",
        "height_profile_step2_line_extensions_all": "voxel_step2_extension_candidate_map",
        "height_profile_step2_line_extensions_rejected": "voxel_step2_rejected_extension_map",
        "height_profile_step2_extension_separator_map": "voxel_step2_extension_separator_map",
        "height_profile_boundary_source_map": "voxel_boundary_source_map",
        "height_profile_partition_free": "voxel_partition_free",
        "height_profile_final_room_label_map": "voxel_final_room_label_map",
        "height_profile_room_label_map_visual": "voxel_room_label_map_visual",
        "height_profile_final_separator_map": "voxel_final_separator_map",
    }
    return {dst: np.asarray(layers[src]) for dst, src in alias.items() if src in layers}


def _height_profile_alias_debug(debug: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for dst, src in {
        "height_profile_room_count": "voxel_room_count",
        "height_profile_step2_reject_reason_counts": "voxel_step2_reject_reason_counts",
        "height_profile_active_z_max_m": "voxel_active_z_max_m",
        "height_profile_ceiling_height_m": "voxel_ceiling_height_m",
    }.items():
        if src in debug:
            out[dst] = debug[src]
    return out
