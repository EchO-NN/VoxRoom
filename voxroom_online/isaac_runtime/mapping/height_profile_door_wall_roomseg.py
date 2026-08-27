from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.height_column_profile import HeightColumnProfileMap
from voxroom_online.isaac_runtime.mapping.height_profile_door_detector import (
    HeightProfileDoorConfig,
    HeightProfileDoorDetectionResult,
    detect_height_profile_doors,
)
from voxroom_online.isaac_runtime.mapping.height_profile_roomseg_evidence import (
    HeightProfileRoomsegEvidenceConfig,
    build_height_profile_roomseg_evidence,
)
from voxroom_online.isaac_runtime.mapping.online_roomseg.debug_viz import save_online_roomseg_debug
from voxroom_online.isaac_runtime.mapping.online_roomseg.separator_candidates import (
    DoorNeckConfig,
    LineExtensionConfig,
    LineExtensionHit,
    NoiseWallGapFillConfig,
    SeparatorCandidate,
    build_door_neck_candidates_from_extensions,
    fill_noise_wall_gaps_from_runs,
)
from voxroom_online.isaac_runtime.mapping.online_roomseg.topology_tests import TopologyTestConfig, greedily_select_separators
from voxroom_online.isaac_runtime.mapping.online_roomseg.utils import conn, dilate, rasterize_line, relabel_compact
from voxroom_online.isaac_runtime.mapping.online_roomseg.wall_lines import (
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
from voxroom_online.isaac_runtime.mapping.room_segmentation import RoomMask, RoomProposalState, RoomSegmentationConfig
from voxroom_online.isaac_runtime.mapping.room_segmentation import _proposal_masks_debug, _room_from_mask
from voxroom_online.isaac_runtime.mapping.vertical_profile import VerticalProfileMap


HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND = "height_profile_door_wall_v8"
HEIGHT_PROFILE_DOOR_WALL_V8_ALGORITHM = "height_profile_door_wall_v8"
HEIGHT_PROFILE_DOOR_WALL_V8_CONTEXT = "height_profile_door_wall_v8_vlm"


@dataclass
class HeightProfileDoorWallRoomSegConfig:
    enabled: bool = True
    resolution_m: float = 0.05
    map_info: MapInfo | None = None
    evidence: HeightProfileRoomsegEvidenceConfig = field(default_factory=HeightProfileRoomsegEvidenceConfig)
    door: HeightProfileDoorConfig = field(default_factory=HeightProfileDoorConfig)
    line_walls: LineWallsConfig = field(default_factory=LineWallsConfig)
    line_filtering: LineFilteringConfig = field(default_factory=LineFilteringConfig)
    wall_run_snap: WallRunSnapConfig = field(default_factory=WallRunSnapConfig)
    wall_run_merge: WallRunMergeConfig = field(default_factory=WallRunMergeConfig)
    step1_gap_fill_enabled: bool = True
    step1_gap_fill_max_gap_m: float = 0.30
    step1_gap_fill_max_lateral_offset_m: float = 0.15
    step1_gap_fill_min_endpoint_support: float = 0.25
    line_extension: LineExtensionConfig = field(default_factory=LineExtensionConfig)
    door_neck: DoorNeckConfig = field(default_factory=DoorNeckConfig)
    topology_test: TopologyTestConfig = field(default_factory=TopologyTestConfig)
    reject_step2_if_intersects_door: bool = True
    door_intersection_dilation_cells: int = 1
    reject_step2_if_tiny_side_width_cells_leq: int = 3
    final_connectivity: int = 4
    merge_small_components_enabled: bool = False
    min_observed_free_cells: int = 1
    min_room_area_m2: float = 0.05
    debug_dump: bool = False
    debug_dir: str = "debug/height_profile_door_wall_roomseg"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "HeightProfileDoorWallRoomSegConfig":
        raw_root = dict(data or {})
        raw = dict(raw_root)
        online = dict(raw_root.get("online_roomseg", {}) or {})
        step1 = dict(raw_root.get("height_profile_step1", {}) or {})
        step2 = dict(raw_root.get("height_profile_step2", {}) or {})
        debug_layers = dict(raw_root.get("debug_layers", {}) or {})

        if "enabled" in debug_layers:
            raw["debug_dump"] = bool(debug_layers.get("enabled"))
        if "output_dir" in debug_layers:
            raw["debug_dir"] = str(debug_layers.get("output_dir"))

        evidence_raw = dict(raw_root.get("height_profile_evidence", {}) or {})
        evidence_raw["height_profile"] = dict(raw_root.get("height_profile", evidence_raw.get("height_profile", {})) or {})
        door_raw = dict(raw_root.get("height_profile_door", {}) or {})
        topology_raw = dict(raw_root.get("topology_test", online.get("topology_test", online.get("topology", {}))) or {})
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
            "evidence": HeightProfileRoomsegEvidenceConfig.from_mapping(evidence_raw, resolution_m=raw.get("resolution_m")),
            "door": HeightProfileDoorConfig.from_mapping(door_raw),
            "line_walls": LineWallsConfig.from_mapping(online.get("line_walls", raw_root.get("line_walls"))),
            "line_filtering": LineFilteringConfig.from_mapping(online.get("line_filtering", raw_root.get("line_filtering"))),
            "wall_run_snap": WallRunSnapConfig.from_mapping(online.get("wall_run_snap", raw_root.get("wall_run_snap"))),
            "wall_run_merge": WallRunMergeConfig.from_mapping(online.get("wall_run_merge", raw_root.get("wall_run_merge"))),
            "line_extension": LineExtensionConfig.from_mapping(line_extension_raw),
            "door_neck": DoorNeckConfig.from_mapping(online.get("door_neck", raw_root.get("door_neck"))),
            "topology_test": TopologyTestConfig.from_mapping(topology_raw),
        }
        step1_key_map = {
            "gap_fill_enabled": "step1_gap_fill_enabled",
            "gap_fill_max_gap_m": "step1_gap_fill_max_gap_m",
            "gap_fill_max_lateral_offset_m": "step1_gap_fill_max_lateral_offset_m",
            "gap_fill_min_endpoint_support": "step1_gap_fill_min_endpoint_support",
        }
        for src, dst in step1_key_map.items():
            if src in step1:
                raw[dst] = step1[src]
        step2_key_map = {
            "reject_if_intersects_door": "reject_step2_if_intersects_door",
            "door_intersection_dilation_cells": "door_intersection_dilation_cells",
            "reject_if_tiny_side_width_cells_leq": "reject_step2_if_tiny_side_width_cells_leq",
        }
        for src, dst in step2_key_map.items():
            if src in step2:
                raw[dst] = step2[src]
        for key, value in overrides.items():
            if value is not None:
                raw[key] = value
        fields = {name for name in cls.__dataclass_fields__}
        base = {key: raw[key] for key in raw if key in fields and key not in nested}
        base.update(nested)
        cfg = cls(**base)
        cfg.evidence.resolution_m = float(cfg.resolution_m)
        return cfg

    def room_config(self) -> RoomSegmentationConfig:
        return RoomSegmentationConfig(
            algorithm=HEIGHT_PROFILE_DOOR_WALL_V8_ALGORITHM,
            source_grid="height_profile_vertical_free",
            proposal_mode=HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND,
            finalization_mode="no_merge",
            min_observed_free_cells=int(self.min_observed_free_cells),
            min_room_area_m2=float(self.min_room_area_m2),
            resolution_m=float(self.resolution_m),
            map_info=self.map_info,
        )


@dataclass
class HeightProfileDoorWallRoomSegResult:
    room_label_map: np.ndarray
    separator_map: np.ndarray
    wall_map: np.ndarray
    door_cut_map: np.ndarray
    step1_wall_gap_fill_map: np.ndarray
    step2_extension_separator_map: np.ndarray
    layers: dict[str, np.ndarray]
    debug: dict[str, object]


class HeightProfileDoorWallRoomSegmenter:
    context_source = HEIGHT_PROFILE_DOOR_WALL_V8_CONTEXT

    def __init__(self, config: HeightProfileDoorWallRoomSegConfig | Mapping[str, object] | None = None, map_info: MapInfo | None = None):
        self.config = config if isinstance(config, HeightProfileDoorWallRoomSegConfig) else HeightProfileDoorWallRoomSegConfig.from_mapping(config or {}, map_info=map_info)
        if map_info is not None:
            self.config.map_info = map_info
            self.config.resolution_m = float(map_info.resolution_m)
            self.config.evidence.resolution_m = float(map_info.resolution_m)
        self.last_debug: dict[str, object] = {}
        self.last_result: HeightProfileDoorWallRoomSegResult | None = None

    def update(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        *,
        step: int,
        object_memory: Iterable[object] | None = None,
        height_profile: HeightColumnProfileMap | None = None,
        vertical_profile: VerticalProfileMap | None = None,
        roomseg_ray_evidence: Mapping[str, np.ndarray] | None = None,
        **kwargs,
    ) -> list[RoomMask]:
        _ = object_memory, vertical_profile, kwargs
        if height_profile is None:
            raise ValueError("height_profile_missing_for_height_profile_door_wall_v8")
        result = run_height_profile_door_wall_roomseg(
            occupancy_map=occupancy_map,
            observed_free_mask=observed_free_mask,
            obstacle_mask=obstacle_mask,
            unknown_mask=unknown_mask,
            height_profile=height_profile,
            navigation_free_mask=observed_free_mask,
            navigation_obstacle_mask=obstacle_mask,
            roomseg_ray_evidence=roomseg_ray_evidence,
            resolution_m=float(self.config.resolution_m),
            config=self.config,
            step=int(step),
        )
        self.last_result = result
        self.last_debug = dict(result.debug)
        return _rooms_from_labels(result.room_label_map, np.asarray(result.layers["height_evidence_unknown_clean"], dtype=bool), self.config.room_config(), int(step), result.debug)


def run_height_profile_door_wall_roomseg(
    *,
    occupancy_map: np.ndarray,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    height_profile: HeightColumnProfileMap,
    navigation_free_mask: np.ndarray,
    navigation_obstacle_mask: np.ndarray,
    roomseg_ray_evidence: Mapping[str, np.ndarray] | None,
    resolution_m: float,
    config: HeightProfileDoorWallRoomSegConfig | Mapping[str, object] | None = None,
    step: int = 0,
) -> HeightProfileDoorWallRoomSegResult:
    cfg = config if isinstance(config, HeightProfileDoorWallRoomSegConfig) else HeightProfileDoorWallRoomSegConfig.from_mapping(config, resolution_m=resolution_m)
    shape = np.asarray(observed_free_mask, dtype=bool).shape
    for name, arr in {
        "occupancy_map": occupancy_map,
        "obstacle_mask": obstacle_mask,
        "unknown_mask": unknown_mask,
        "navigation_free_mask": navigation_free_mask,
        "navigation_obstacle_mask": navigation_obstacle_mask,
    }.items():
        if np.asarray(arr).shape != shape:
            raise ValueError("%s must match observed_free_mask shape" % name)
    if height_profile.shape != shape:
        raise ValueError("height_profile shape must match roomseg masks")

    evidence = build_height_profile_roomseg_evidence(
        height_profile=height_profile,
        navigation_free_mask=np.asarray(navigation_free_mask, dtype=bool),
        navigation_obstacle_mask=np.asarray(navigation_obstacle_mask, dtype=bool),
        unknown_mask_from_navigation=np.asarray(unknown_mask, dtype=bool),
        resolution_m=float(resolution_m),
        config=cfg.evidence,
    )
    segments, wall_debug = extract_line_supported_walls(
        evidence.wall_clean,
        resolution_m=float(resolution_m),
        config=cfg.line_walls,
    )
    raw_line_map = line_supported_wall_mask(segments, shape)
    filtered_lines, filter_debug = filter_and_snap_wall_lines(
        segments,
        wall_candidate_clean=evidence.wall_clean,
        free_clean=evidence.free_clean,
        resolution_m=float(resolution_m),
        config=cfg.line_filtering,
    )
    filtered_line_map = filtered_wall_line_mask(filtered_lines, shape)
    wall_base = evidence.wall_clean | filtered_line_map

    door_result = detect_height_profile_doors(
        classification=evidence.classification,
        free_clean=evidence.free_clean,
        wall_clean=wall_base,
        unknown_clean=evidence.unknown_clean,
        resolution_m=float(resolution_m),
        config=cfg.door,
    )
    door_cut_mask = np.asarray(door_result.door_cut_mask, dtype=bool) & evidence.free_clean

    wall_runs, wall_run_debug = snap_wall_segments_to_runs(
        segments,
        wall_base,
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
    step1_gap_fill_map &= np.asarray(evidence.vertical_observed_xy, dtype=bool)
    step1_gap_fill_map &= ~door_cut_mask
    step1_wall_mask = wall_base | step1_gap_fill_map
    free_after_step1 = evidence.free_clean & ~step1_gap_fill_map
    unknown_after_step1 = evidence.unknown_clean & ~step1_gap_fill_map

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
        filtered_lines=filtered_lines,
        free_after_step1=free_after_step1,
        step1_wall_mask=step1_wall_mask,
        unknown_after_step1=unknown_after_step1,
        door_mask=door_cut_mask | door_result.door_seed_mask,
        resolution_m=float(resolution_m),
        config=line_cfg,
        reject_if_intersects_door=bool(cfg.reject_step2_if_intersects_door),
        door_intersection_dilation_cells=int(cfg.door_intersection_dilation_cells),
    )
    step2_candidates, step2_candidate_debug = build_door_neck_candidates_from_extensions(
        step2_extensions,
        accepted_virtual_targets=None,
        resolution_m=float(resolution_m),
        config=cfg.door_neck,
        start_id=1,
    )
    topology_cfg = cfg.topology_test
    topology_cfg.reject_if_side_width_cells_leq = int(cfg.reject_step2_if_tiny_side_width_cells_leq)
    accepted_step2, rejected_step2, accepted_step2_map, _raw_topology_labels, topology_debug = greedily_select_separators(
        step2_candidates,
        free_clean=free_after_step1,
        unknown_clean=unknown_after_step1,
        wall_candidate_clean=step1_wall_mask,
        corridor_skeleton=np.zeros(shape, dtype=bool),
        resolution_m=float(resolution_m),
        config=topology_cfg,
    )
    step2_extension_separator_map = accepted_step2_map & free_after_step1
    final_virtual_separator_map = door_cut_mask | step2_extension_separator_map
    partition_free = free_after_step1 & ~final_virtual_separator_map
    labels, _count = ndimage.label(partition_free, structure=conn(int(cfg.final_connectivity)))
    labels = relabel_compact(labels.astype(np.int32))
    labels[unknown_after_step1] = 0
    labels[~free_after_step1] = 0
    final_separator_map = step1_wall_mask | door_cut_mask | step2_extension_separator_map

    boundary_source = np.zeros(shape, dtype=np.uint8)
    boundary_source[step1_wall_mask] = 1
    boundary_source[door_cut_mask] = 2
    boundary_source[step2_extension_separator_map] = 3

    step2_layers = _extension_layers(step2_extensions, shape)
    layers = {
        "height_profile_vertical_free_xy": evidence.classification.vertical_free_xy,
        "height_profile_wall_xy": evidence.classification.wall_xy,
        "height_profile_unknown_xy": evidence.classification.unknown_xy,
        "height_profile_vertical_observed_xy": evidence.vertical_observed_xy,
        "height_evidence_free_clean": evidence.free_clean,
        "height_evidence_wall_clean": evidence.wall_clean,
        "height_evidence_unknown_clean": evidence.unknown_clean,
        "height_profile_wall_base_mask": wall_base,
        "height_profile_line_supported_wall_mask": raw_line_map,
        "height_profile_filtered_wall_line_mask": filtered_line_map,
        "height_profile_door_seed_mask": door_result.door_seed_mask,
        "height_profile_door_seed_component_map": door_result.door_seed_component_map,
        "height_profile_door_centerline_candidate_mask": door_result.door_centerline_candidate_mask,
        "height_profile_door_cut_mask": door_cut_mask,
        "height_profile_accepted_door_centerline_mask": door_result.accepted_door_centerline_mask,
        "height_profile_rejected_door_centerline_mask": door_result.rejected_door_centerline_mask,
        "height_profile_step1_wall_gap_fill_map": step1_gap_fill_map,
        "height_profile_step1_wall_mask": step1_wall_mask,
        "height_profile_step2_line_extensions_all": step2_layers["all"],
        "height_profile_step2_line_extensions_accepted": step2_layers["accepted"],
        "height_profile_step2_line_extensions_rejected": step2_layers["rejected"],
        "height_profile_step2_extension_separator_map": step2_extension_separator_map,
        "height_profile_boundary_source_map": boundary_source,
        "height_profile_partition_free": partition_free,
        "height_profile_final_room_label_map": labels,
        "height_profile_room_label_map_visual": labels,
        "height_profile_final_separator_map": final_separator_map,
        "final_room_labels": labels,
        "accepted_separators": final_separator_map,
        "rejected_separators": _rasterize_candidates(rejected_step2, shape),
    }
    report = {
        "step": int(step),
        "backend": HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND,
        "algorithm": HEIGHT_PROFILE_DOOR_WALL_V8_ALGORITHM,
        "stage_order": ["height_profile_evidence", "door_detection", "step1_real_wall_gap_fill", "step2_wall_line_extension", "final_4conn_labels"],
        "wall_segment_count": int(len(segments)),
        "filtered_wall_line_count": int(len(filtered_lines)),
        "height_profile_door_accepted_count": int(door_result.debug.get("height_profile_door_accepted_count", 0)),
        "height_profile_door_rejected_count": int(door_result.debug.get("height_profile_door_rejected_count", 0)),
        "height_profile_step1_gap_fill_cells": int(np.count_nonzero(step1_gap_fill_map)),
        "height_profile_step2_extension_count": int(len(step2_extensions)),
        "height_profile_step2_candidate_count": int(len(step2_candidates)),
        "height_profile_step2_accepted_count": int(len(accepted_step2)),
        "height_profile_step2_rejected_count": int(len(rejected_step2)),
        "height_profile_step2_reject_reason_counts": _extension_and_candidate_reasons(step2_extensions, rejected_step2),
        "height_profile_room_count": _room_count(labels),
        "final_connectivity": int(cfg.final_connectivity),
        "merge_small_components_enabled": bool(cfg.merge_small_components_enabled),
        "frontier_source": "vertical_free",
        "frontier_vertical_free_cells": int(np.count_nonzero(evidence.classification.vertical_free_xy)),
        "frontier_cells_from_vertical_free": int(np.count_nonzero(evidence.classification.vertical_free_xy)),
        "candidates": [candidate.to_dict() for candidate in [*accepted_step2, *rejected_step2]],
    }
    debug = {
        "backend": HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND,
        "actual_backend": HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND,
        "source_backend": HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND,
        "roomseg_backend": HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND,
        "algorithm": HEIGHT_PROFILE_DOOR_WALL_V8_ALGORITHM,
        "source": HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND,
        "context_source": HEIGHT_PROFILE_DOOR_WALL_V8_CONTEXT,
        "room_map_mode": HEIGHT_PROFILE_DOOR_WALL_V8_CONTEXT,
        "strict_fallback_used": False,
        "silent_fallback_used": False,
        "legacy_style_used": False,
        "navigation_obstacle_written": False,
        "step": int(step),
        "resolution_m": float(resolution_m),
        "height_profile_room_count": _room_count(labels),
        "merge_small_components_enabled": bool(cfg.merge_small_components_enabled),
        "height_profile_step2_reject_reason_counts": report["height_profile_step2_reject_reason_counts"],
        "separator_report": report,
        "filtered_wall_lines_report": {
            "filtered_wall_lines": [line.to_dict() for line in filtered_lines],
            **filter_debug,
        },
        "line_extension_report": {
            "pass2": step2_extension_debug,
        },
        "height_profile_step1_gap_fill_report": step1_gap_debug,
        "height_profile_wall_run_report": wall_run_debug,
        "topology_report": {"final": topology_debug},
        "accepted_separators": [candidate.to_dict() for candidate in accepted_step2],
        "rejected_separators": [candidate.to_dict() for candidate in rejected_step2],
        "proposal_room_masks": _proposal_masks_debug(labels),
        "frontier_source": "vertical_free",
        "frontier_vertical_free_cells": int(np.count_nonzero(evidence.classification.vertical_free_xy)),
        "frontier_vertical_observed_cells": int(np.count_nonzero(evidence.vertical_observed_xy)),
        "frontier_vertical_wall_cells": int(np.count_nonzero(evidence.classification.wall_xy)),
        "frontier_cells_from_vertical_free": int(np.count_nonzero(evidence.classification.vertical_free_xy)),
        "frontier_cells_removed_by_navigation_unreachable": 0,
        **evidence.debug,
        **door_result.debug,
        **layers,
        "line_walls": wall_debug,
    }
    if bool(cfg.debug_dump):
        dump = save_online_roomseg_debug(
            out_dir=Path(str(cfg.debug_dir)) / ("height_profile_roomseg_step_%06d" % int(step)),
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
    return HeightProfileDoorWallRoomSegResult(
        room_label_map=labels.astype(np.int32),
        separator_map=final_separator_map.astype(bool),
        wall_map=step1_wall_mask.astype(bool),
        door_cut_map=door_cut_mask.astype(bool),
        step1_wall_gap_fill_map=step1_gap_fill_map.astype(bool),
        step2_extension_separator_map=step2_extension_separator_map.astype(bool),
        layers=layers,
        debug=debug,
    )


def extend_step2_wall_lines(
    *,
    filtered_lines,
    free_after_step1: np.ndarray,
    step1_wall_mask: np.ndarray,
    unknown_after_step1: np.ndarray,
    door_mask: np.ndarray,
    resolution_m: float,
    config: LineExtensionConfig,
    reject_if_intersects_door: bool,
    door_intersection_dilation_cells: int,
) -> tuple[list[LineExtensionHit], dict]:
    from voxroom_online.isaac_runtime.mapping.online_roomseg.separator_candidates import extend_wall_lines_once

    extensions, debug = extend_wall_lines_once(
        filtered_lines,
        free_clean=free_after_step1,
        wall_target_mask=step1_wall_mask,
        virtual_target_mask=None,
        unknown_clean=unknown_after_step1,
        resolution_m=float(resolution_m),
        pass_id=2,
        config=config,
        start_id=1,
    )
    if bool(reject_if_intersects_door):
        door = dilate(np.asarray(door_mask, dtype=bool), int(door_intersection_dilation_cells))
        for hit in extensions:
            if hit.reject_reason is not None:
                continue
            line = rasterize_line(hit.p_start_rc, hit.p_hit_rc, door.shape)
            hit_r, hit_c = np.rint(np.asarray(hit.p_hit_rc, dtype=np.float32)).astype(np.int32).tolist()
            if 0 <= int(hit_r) < door.shape[0] and 0 <= int(hit_c) < door.shape[1] and bool(door[int(hit_r), int(hit_c)]):
                hit.reject_reason = "reject_extension_hits_door"
            elif np.any(line & door):
                hit.reject_reason = "reject_extension_intersects_door"
            if hit.reject_reason is not None:
                hit.debug["height_profile_step2_door_rejection"] = str(hit.reject_reason)
    debug = {
        **dict(debug),
        "accepted_extension_count": int(sum(1 for hit in extensions if hit.reject_reason is None)),
        "rejected_by_reason": _extension_reason_counts(extensions),
        "extensions": [hit.to_dict() for hit in extensions[:1024]],
    }
    return extensions, debug


def _rooms_from_labels(labels: np.ndarray, unknown: np.ndarray, config: RoomSegmentationConfig, step: int, debug: Mapping[str, object]) -> list[RoomMask]:
    out: list[RoomMask] = []
    min_cells = max(1, int(config.min_observed_free_cells))
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = np.asarray(labels == label, dtype=bool)
        if int(np.count_nonzero(mask)) < min_cells:
            continue
        room = _room_from_mask("pending", mask, unknown, [], config, int(step))
        room.source = HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND
        room.metadata["label_id"] = int(label)
        room.metadata["proposal_labels"] = [int(label)]
        room.metadata["source_finalization_mode"] = HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND
        room.metadata["accepted_separator_count"] = _accepted_separator_count(debug)
        out.append(room)
    return out


def _accepted_separator_count(debug: Mapping[str, object]) -> int:
    report = debug.get("separator_report")
    if isinstance(report, Mapping):
        for key in ("height_profile_step2_accepted_count", "accepted_count"):
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
