from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .utils import dilate, rasterize_line
from .wall_lines import FilteredWallLine, SnappedWallRun, WallSegment, snap_wall_segments_to_runs


@dataclass
class PhysicalWallCompletionConfig:
    enabled: bool = True
    max_gap_m: float = 0.45
    max_lateral_offset_m: float = 0.20
    max_angle_deg: float = 10.0
    min_endpoint_support: float = 0.35

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "PhysicalWallCompletionConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class MissedScanGapClosureConfig:
    enabled: bool = True
    max_gap_m: float = 0.45
    max_unknown_ratio: float = 0.70
    max_free_ratio: float = 0.65

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "MissedScanGapClosureConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class ShortUnknownGapClosureConfig:
    enabled: bool = True
    max_gap_m: float = 0.60
    min_unknown_ratio: float = 0.45

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "ShortUnknownGapClosureConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class NoiseWallGapFillConfig:
    enabled: bool = True
    max_gap_m: float = 0.40
    max_lateral_offset_m: float = 0.20
    min_endpoint_support: float = 0.25
    thickness_cells: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "NoiseWallGapFillConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class DoorwayVirtualCutConfig:
    enabled: bool = True
    min_width_m: float = 0.45
    max_width_m: float = 1.60
    max_angle_deg: float = 12.0
    min_wall_endpoint_support: float = 0.30
    require_free_gap: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "DoorwayVirtualCutConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class SingleSidedWallExtensionConfig:
    enabled: bool = True
    max_extension_m: float = 1.20
    max_open_free_ratio: float = 0.70
    require_anchor: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "SingleSidedWallExtensionConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class SeparatorAnchorConfig:
    thickness_cells: int = 1
    doorway_thickness_cells: int = 1
    wall_completion_thickness_cells: int = 1
    corridor_neck_thickness_cells: int = 1
    max_anchor_extension_m: float = 0.35
    require_two_anchors: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "SeparatorAnchorConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class SeparatorCandidate:
    candidate_id: int
    kind: str
    p0_rc: np.ndarray
    p1_rc: np.ndarray
    theta: float
    length_m: float
    confidence: float
    source_segment_ids: list[int]
    wall_support_score: float = 0.0
    free_gap_score: float = 0.0
    doorway_score: float = 0.0
    corridor_preservation_score: float = 0.0
    topology_gain_score: float = 0.0
    fragmentation_penalty: float = 0.0
    corridor_split_penalty: float = 0.0
    accepted: bool = False
    reject_reason: str = ""
    debug: dict = field(default_factory=dict)

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        return rasterize_line(self.p0_rc, self.p1_rc, shape)

    def to_dict(self) -> dict:
        return {
            "candidate_id": int(self.candidate_id),
            "kind": str(self.kind),
            "p0_rc": [int(round(float(v))) for v in self.p0_rc.tolist()],
            "p1_rc": [int(round(float(v))) for v in self.p1_rc.tolist()],
            "theta": float(self.theta),
            "length_m": float(self.length_m),
            "confidence": float(self.confidence),
            "source_segment_ids": [int(v) for v in self.source_segment_ids],
            "wall_support_score": float(self.wall_support_score),
            "free_gap_score": float(self.free_gap_score),
            "doorway_score": float(self.doorway_score),
            "corridor_preservation_score": float(self.corridor_preservation_score),
            "topology_gain_score": float(self.topology_gain_score),
            "fragmentation_penalty": float(self.fragmentation_penalty),
            "corridor_split_penalty": float(self.corridor_split_penalty),
            "accepted": bool(self.accepted),
            "reject_reason": str(self.reject_reason),
            **_jsonable(self.debug),
        }


@dataclass
class LineExtensionConfig:
    enabled: bool = True
    passes: int = 2
    min_extension_m: float = 0.40
    max_extension_m: float = 1.60
    max_probe_m: float = 1.80
    sample_step_m: float = 0.05
    hit_radius_m: float = 0.10
    free_ratio_min: float = 0.65
    unknown_ratio_max: float = 0.50
    wall_mid_ratio_max: float = 0.15
    require_free_between_start_and_hit: bool = True
    min_free_cells_between_start_and_hit: int = 3
    allow_hit_virtual_door_on_pass2: bool = True
    virtual_target_dilation_m: float = 0.08

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "LineExtensionConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class DoorNeckConfig:
    enabled: bool = True
    min_width_m: float = 0.40
    max_width_m: float = 1.60
    min_confidence: float = 0.50
    require_topology_gain: bool = True
    reject_unknown_only_support: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "DoorNeckConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class LineExtensionHit:
    extension_id: int
    source_line_id: int
    source_endpoint: str
    pass_id: int
    p_start_rc: np.ndarray
    p_hit_rc: np.ndarray
    theta: float
    length_m: float
    interior_free_ratio: float
    interior_unknown_ratio: float
    interior_wall_ratio: float
    hit_type: str
    hit_candidate_id: int | None
    confidence: float
    reject_reason: str | None
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "extension_id": int(self.extension_id),
            "source_line_id": int(self.source_line_id),
            "source_endpoint": str(self.source_endpoint),
            "pass_id": int(self.pass_id),
            "p_start_rc": [float(v) for v in np.asarray(self.p_start_rc, dtype=np.float32).tolist()],
            "p_hit_rc": [float(v) for v in np.asarray(self.p_hit_rc, dtype=np.float32).tolist()],
            "theta": float(self.theta),
            "length_m": float(self.length_m),
            "interior_free_ratio": float(self.interior_free_ratio),
            "interior_unknown_ratio": float(self.interior_unknown_ratio),
            "interior_wall_ratio": float(self.interior_wall_ratio),
            "hit_type": str(self.hit_type),
            "hit_candidate_id": None if self.hit_candidate_id is None else int(self.hit_candidate_id),
            "confidence": float(self.confidence),
            "accepted_as_candidate": self.reject_reason is None,
            "reject_reason": None if self.reject_reason is None else str(self.reject_reason),
            "debug": _jsonable(self.debug),
        }


def extend_wall_lines_once(
    filtered_lines: Sequence[FilteredWallLine],
    *,
    free_clean: np.ndarray,
    wall_target_mask: np.ndarray,
    virtual_target_mask: np.ndarray | None,
    unknown_clean: np.ndarray,
    resolution_m: float,
    pass_id: int,
    config: LineExtensionConfig | Mapping[str, object] | None = None,
    start_id: int = 1,
) -> tuple[list[LineExtensionHit], dict]:
    cfg = config if isinstance(config, LineExtensionConfig) else LineExtensionConfig.from_mapping(config)
    free = np.asarray(free_clean, dtype=bool)
    wall = np.asarray(wall_target_mask, dtype=bool)
    virtual = None if virtual_target_mask is None else np.asarray(virtual_target_mask, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    if not bool(cfg.enabled):
        return [], {"enabled": False, "pass_id": int(pass_id), "extension_count": 0}
    hit_radius = max(0, int(round(float(cfg.hit_radius_m) / max(float(resolution_m), 1e-9))))
    virtual_radius = max(0, int(round(float(cfg.virtual_target_dilation_m) / max(float(resolution_m), 1e-9))))
    real_target = dilate(wall, hit_radius)
    virtual_target = dilate(virtual, max(hit_radius, virtual_radius)) if virtual is not None else None
    hits: list[LineExtensionHit] = []
    eid = int(start_id)
    for line in filtered_lines:
        for endpoint in ("p0", "p1"):
            hit = trace_line_extension(
                line,
                endpoint,
                free_clean=free,
                real_wall_target=real_target,
                wall_mid_mask=wall,
                virtual_target=virtual_target,
                virtual_mid_mask=virtual,
                unknown_clean=unknown,
                resolution_m=float(resolution_m),
                pass_id=int(pass_id),
                extension_id=eid,
                config=cfg,
            )
            hits.append(hit)
            eid += 1
    debug = {
        "enabled": True,
        "pass_id": int(pass_id),
        "extension_count": int(len(hits)),
        "accepted_extension_count": int(sum(1 for hit in hits if hit.reject_reason is None)),
        "rejected_by_reason": _extension_reason_counts(hits),
        "extensions": [hit.to_dict() for hit in hits[:1024]],
    }
    return hits, debug


def trace_line_extension(
    line: FilteredWallLine,
    endpoint: str,
    *,
    free_clean: np.ndarray,
    real_wall_target: np.ndarray,
    wall_mid_mask: np.ndarray,
    virtual_target: np.ndarray | None,
    virtual_mid_mask: np.ndarray | None,
    unknown_clean: np.ndarray,
    resolution_m: float,
    pass_id: int,
    extension_id: int,
    config: LineExtensionConfig,
) -> LineExtensionHit:
    free = np.asarray(free_clean, dtype=bool)
    wall = np.asarray(real_wall_target, dtype=bool)
    wall_mid = np.asarray(wall_mid_mask, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    virtual = None if virtual_target is None else np.asarray(virtual_target, dtype=bool)
    virtual_mid = None if virtual_mid_mask is None else np.asarray(virtual_mid_mask, dtype=bool)
    p_start, direction = _extension_start_and_direction(line, endpoint)
    max_steps = max(1, int(round(float(config.max_probe_m) / max(float(resolution_m), 1e-9))))
    hit_radius = max(0, int(round(float(config.hit_radius_m) / max(float(resolution_m), 1e-9))))
    virtual_radius = max(
        hit_radius,
        int(round(float(config.virtual_target_dilation_m) / max(float(resolution_m), 1e-9))),
    )
    sampled: list[tuple[int, int]] = []
    hit_type = "none"
    hit_candidate_id = None
    hit_point = np.asarray(p_start, dtype=np.float32)
    hit_debug: dict[str, object] = {}
    last_point = np.asarray(p_start, dtype=np.float32)
    for step in range(1, max_steps + 1):
        rc = np.rint(np.asarray(p_start, dtype=np.float32) + np.asarray(direction, dtype=np.float32) * float(step)).astype(np.int32)
        if not _inside(rc, free.shape):
            return _line_extension_result(
                line,
                endpoint,
                p_start,
                last_point,
                sampled,
                hit_type="none",
                hit_candidate_id=None,
                reject_reason="reject_extension_no_wall_or_virtual_door_hit",
                pass_id=pass_id,
                extension_id=extension_id,
                resolution_m=float(resolution_m),
                config=config,
                debug={"stop": "out_of_bounds", "_free_clean": free, "_wall_target": wall_mid, "_unknown_clean": unknown},
            )
        sampled.append((int(rc[0]), int(rc[1])))
        last_point = rc.astype(np.float32)
        free_count = int(sum(1 for r, c in sampled if bool(free[r, c])))
        if bool(wall_mid[int(rc[0]), int(rc[1])]):
            if free_count < int(config.min_free_cells_between_start_and_hit):
                return _line_extension_result(
                    line,
                    endpoint,
                    p_start,
                    last_point,
                    sampled,
                    hit_type="blocked_wall",
                    hit_candidate_id=None,
                    reject_reason="reject_extension_blocked_by_near_wall",
                    pass_id=pass_id,
                    extension_id=extension_id,
                    resolution_m=float(resolution_m),
                    config=config,
                    debug={
                        "stop": "blocked_by_near_wall_before_min_free",
                        "blocked_cell": [int(rc[0]), int(rc[1])],
                        "_free_clean": free,
                        "_wall_target": wall_mid,
                        "_unknown_clean": unknown,
                    },
                )
            hit_type = "real_wall"
            hit_point = _snap_hit_point_to_target(rc, wall_mid, direction, radius_cells=hit_radius)
            hit_debug = {
                "hit_cell_before_snap": [int(rc[0]), int(rc[1])],
                "hit_cell_snap_target": "real_wall",
            }
            break
        if free_count < int(config.min_free_cells_between_start_and_hit):
            continue
        if bool(wall[int(rc[0]), int(rc[1])]):
            hit_type = "real_wall"
            hit_point = _snap_hit_point_to_target(rc, wall_mid, direction, radius_cells=hit_radius)
            hit_debug = {
                "hit_cell_before_snap": [int(rc[0]), int(rc[1])],
                "hit_cell_snap_target": "real_wall",
            }
            break
        if (
            virtual is not None
            and bool(config.allow_hit_virtual_door_on_pass2)
            and int(pass_id) >= 2
            and bool(virtual[int(rc[0]), int(rc[1])])
        ):
            hit_type = "virtual_door"
            hit_point = _snap_hit_point_to_target(
                rc,
                virtual_mid if virtual_mid is not None else virtual,
                direction,
                radius_cells=virtual_radius,
            )
            hit_debug = {
                "hit_cell_before_snap": [int(rc[0]), int(rc[1])],
                "hit_cell_snap_target": "virtual_door",
            }
            break
    if hit_type == "none":
        reason = "reject_extension_hit_unknown" if _sample_ratio(sampled, unknown) > float(config.unknown_ratio_max) else "reject_extension_no_wall_or_virtual_door_hit"
        return _line_extension_result(
            line,
            endpoint,
            p_start,
            last_point,
            sampled,
            hit_type=hit_type,
            hit_candidate_id=hit_candidate_id,
            reject_reason=reason,
            pass_id=pass_id,
            extension_id=extension_id,
            resolution_m=float(resolution_m),
            config=config,
            debug={"stop": "max_probe", "_free_clean": free, "_wall_target": wall_mid, "_unknown_clean": unknown},
        )
    return _line_extension_result(
        line,
        endpoint,
        p_start,
        hit_point,
        sampled,
        hit_type=hit_type,
        hit_candidate_id=hit_candidate_id,
        reject_reason=None,
        pass_id=pass_id,
        extension_id=extension_id,
        resolution_m=float(resolution_m),
        config=config,
        debug={**hit_debug, "stop": "target_hit", "_free_clean": free, "_wall_target": wall_mid, "_unknown_clean": unknown},
    )


def build_door_neck_candidates_from_extensions(
    extensions: Sequence[LineExtensionHit],
    *,
    accepted_virtual_targets: Sequence[SeparatorCandidate] | None,
    resolution_m: float,
    config: DoorNeckConfig | Mapping[str, object] | None = None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], dict]:
    cfg = config if isinstance(config, DoorNeckConfig) else DoorNeckConfig.from_mapping(config)
    if not bool(cfg.enabled):
        return [], {"enabled": False, "candidate_count": 0}
    _ = accepted_virtual_targets, resolution_m
    candidates: list[SeparatorCandidate] = []
    rejected: list[dict] = []
    cid = int(start_id)
    for hit in extensions:
        if hit.reject_reason is not None:
            rejected.append(hit.to_dict())
            continue
        if not (float(cfg.min_width_m) <= float(hit.length_m) <= float(cfg.max_width_m)):
            hit.reject_reason = "reject_extension_length_out_of_door_range"
            rejected.append(hit.to_dict())
            continue
        if float(hit.confidence) < float(cfg.min_confidence):
            hit.reject_reason = "reject_low_line_extension_confidence"
            rejected.append(hit.to_dict())
            continue
        pass_bonus = 1.0 if int(hit.pass_id) == 1 else (0.85 if str(hit.hit_type) == "virtual_door" else 0.95)
        confidence = float(
            np.clip(
                0.30 * float(hit.interior_free_ratio)
                + 0.25 * float(hit.confidence)
                + 0.20 * (1.0 if str(hit.hit_type) in {"real_wall", "virtual_door", "virtual_neck"} else 0.0)
                + 0.15 * 1.0
                + 0.10 * pass_bonus,
                0.0,
                1.0,
            )
        )
        candidate = SeparatorCandidate(
            candidate_id=cid,
            kind="line_extension_door_neck",
            p0_rc=np.asarray(hit.p_start_rc, dtype=np.float32),
            p1_rc=np.asarray(hit.p_hit_rc, dtype=np.float32),
            theta=float(hit.theta),
            length_m=float(hit.length_m),
            confidence=confidence,
            source_segment_ids=[int(hit.source_line_id)],
            wall_support_score=float(hit.confidence),
            free_gap_score=float(hit.interior_free_ratio),
            doorway_score=float(hit.interior_free_ratio if str(hit.hit_type) in {"real_wall", "virtual_door"} else 0.0),
            debug={
                "candidate_source": "line_extension",
                "kind_detail": "door_neck",
                "pass_id": int(hit.pass_id),
                "source_extension_ids": [int(hit.extension_id)],
                "source_line_ids": [int(hit.source_line_id)],
                "hit_types": [str(hit.hit_type)],
                "width_m": float(hit.length_m),
                "neck_score": float(hit.interior_free_ratio),
                "doorway_score": float(hit.interior_free_ratio),
                "corridor_false_pair_score": 0.0,
                "extension": hit.to_dict(),
            },
        )
        candidates.append(candidate)
        cid += 1
    debug = {
        "enabled": True,
        "candidate_count": int(len(candidates)),
        "rejected_extension_count": int(len(rejected)),
        "candidates": [candidate.to_dict() for candidate in candidates[:1024]],
        "rejected_extensions": rejected[:1024],
    }
    return candidates, debug


def build_step2_separator_candidates_from_extensions(
    extensions: Sequence[LineExtensionHit],
    *,
    accepted_virtual_targets: Sequence[SeparatorCandidate] | None,
    resolution_m: float,
    config: DoorNeckConfig | Mapping[str, object] | None = None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], dict]:
    """Build Step2 corridor separators from accepted wall-extension hits.

    The older helper names these candidates as door necks.  Voxel room
    segmentation uses the same geometric evidence for corridor partitioning,
    but keeping a distinct kind makes topology reports and per-kind thresholds
    auditable.
    """
    candidates, debug = build_door_neck_candidates_from_extensions(
        extensions,
        accepted_virtual_targets=accepted_virtual_targets,
        resolution_m=float(resolution_m),
        config=config,
        start_id=int(start_id),
    )
    for candidate in candidates:
        candidate.kind = "line_extension_corridor_separator"
        candidate.debug["kind_detail"] = "corridor_separator"
        candidate.debug["candidate_source"] = "step2_line_extension"
    debug = dict(debug)
    debug["candidate_kind"] = "line_extension_corridor_separator"
    debug["candidates"] = [candidate.to_dict() for candidate in candidates[:1024]]
    return candidates, debug


def build_door_neck_candidates_from_extension_intersections(
    extensions: Sequence[LineExtensionHit],
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    line_config: LineExtensionConfig | Mapping[str, object] | None = None,
    door_config: DoorNeckConfig | Mapping[str, object] | None = None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], np.ndarray, dict]:
    """Create door/neck candidates where two wall-extension probes cross.

    A red extension probe crossing another red extension probe is meaningful
    structural evidence, even when neither probe hits an already materialized
    wall cell.  This helper turns that crossing into virtual-neck endpoints so
    the normal topology test can decide whether the resulting separator really
    splits navigable free space.
    """
    line_cfg = line_config if isinstance(line_config, LineExtensionConfig) else LineExtensionConfig.from_mapping(line_config)
    door_cfg = door_config if isinstance(door_config, DoorNeckConfig) else DoorNeckConfig.from_mapping(door_config)
    free = np.asarray(free_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    target = np.zeros_like(free, dtype=bool)
    if not bool(line_cfg.enabled) or not bool(door_cfg.enabled):
        return [], target, {"enabled": False, "candidate_count": 0, "reason": "disabled"}

    usable: list[tuple[LineExtensionHit, str, int, int, int]] = []
    for hit in extensions:
        if not _extension_can_seed_intersection(hit):
            continue
        axis_info = _extension_axis_info(hit, free.shape)
        if axis_info is None:
            continue
        axis, fixed, lo, hi = axis_info
        if hi <= lo:
            continue
        usable.append((hit, axis, fixed, lo, hi))

    candidates: list[SeparatorCandidate] = []
    events: list[dict] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    cid = int(start_id)
    for idx, (a, axis_a, fixed_a, lo_a, hi_a) in enumerate(usable):
        for b, axis_b, fixed_b, lo_b, hi_b in usable[idx + 1 :]:
            if int(a.source_line_id) == int(b.source_line_id):
                continue
            if axis_a == axis_b:
                continue
            if axis_a == "horizontal":
                rc = np.asarray([fixed_a, fixed_b], dtype=np.int32)
                inside = lo_a <= fixed_b <= hi_a and lo_b <= fixed_a <= hi_b
            else:
                rc = np.asarray([fixed_b, fixed_a], dtype=np.int32)
                inside = lo_b <= fixed_a <= hi_b and lo_a <= fixed_b <= hi_a
            if not inside or not _inside(rc, free.shape) or not bool(free[int(rc[0]), int(rc[1])]):
                continue
            pair_events = []
            for hit in (a, b):
                candidate = _candidate_from_extension_intersection(
                    hit,
                    other_hit=b if hit is a else a,
                    intersection_rc=rc,
                    free_clean=free,
                    unknown_clean=unknown,
                    resolution_m=float(resolution_m),
                    line_config=line_cfg,
                    door_config=door_cfg,
                    candidate_id=cid,
                )
                if candidate is None:
                    continue
                key = (
                    int(candidate.source_segment_ids[0]),
                    int(candidate.source_segment_ids[1]),
                    int(round(float(candidate.p0_rc[0]))),
                    int(round(float(candidate.p0_rc[1]))),
                    int(rc[0]) * int(free.shape[1]) + int(rc[1]),
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                target[int(rc[0]), int(rc[1])] = True
                pair_events.append(
                    {
                        "candidate_id": int(candidate.candidate_id),
                        "source_extension_ids": list(candidate.debug.get("source_extension_ids", [])),
                        "length_m": float(candidate.length_m),
                    }
                )
                cid += 1
            if pair_events:
                events.append(
                    {
                        "intersection_rc": [int(rc[0]), int(rc[1])],
                        "source_extension_ids": [int(a.extension_id), int(b.extension_id)],
                        "source_line_ids": [int(a.source_line_id), int(b.source_line_id)],
                        "candidates": pair_events,
                    }
                )
    debug = {
        "enabled": True,
        "usable_extension_count": int(len(usable)),
        "intersection_count": int(len(events)),
        "candidate_count": int(len(candidates)),
        "virtual_target_cell_count": int(np.count_nonzero(target)),
        "events": events[:512],
    }
    return candidates, target.astype(bool), debug


def rasterize_candidates(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int], thickness_cells: int = 0) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        out |= separator_mask_for_candidate(candidate, shape, thickness_cells)
    return out.astype(bool)


def fill_noise_wall_gaps_from_runs(
    runs: Sequence[SnappedWallRun],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    config: NoiseWallGapFillConfig | Mapping[str, object] | None = None,
) -> tuple[np.ndarray, dict]:
    cfg = config if isinstance(config, NoiseWallGapFillConfig) else NoiseWallGapFillConfig.from_mapping(config)
    out = np.zeros(shape, dtype=bool)
    if not bool(cfg.enabled):
        return out, {"enabled": False, "filled_gap_count": 0, "filled_cell_count": 0, "strict_less_than_max_gap": True}
    max_gap_m = float(cfg.max_gap_m)
    max_lateral_cells = max(0, int(round(float(cfg.max_lateral_offset_m) / max(float(resolution_m), 1e-9))))
    thickness = max(0, int(cfg.thickness_cells))
    events: list[dict] = []
    seen: set[tuple[str, int, int, int]] = set()
    for idx, a in enumerate(runs):
        for b in runs[idx + 1 :]:
            if str(a.axis) != str(b.axis):
                continue
            if abs(int(a.line) - int(b.line)) > max_lateral_cells:
                continue
            gap = _run_gap(a, b)
            if gap is None:
                continue
            gap_start, gap_end = gap
            gap_cells = int(gap_end - gap_start + 1)
            if gap_cells <= 0:
                continue
            length_m = float(gap_cells * float(resolution_m))
            if not (length_m < max_gap_m - 1e-9):
                continue
            endpoint_support = float(min(float(a.confidence), float(b.confidence)))
            if endpoint_support < float(cfg.min_endpoint_support):
                continue
            line = int(round(float((int(a.line) + int(b.line)) / 2.0)))
            key = (str(a.axis), int(line), int(gap_start), int(gap_end))
            if key in seen:
                continue
            seen.add(key)
            p0, p1, _theta = _candidate_points(str(a.axis), line, gap_start, gap_end)
            mask = rasterize_line(p0, p1, shape)
            if thickness > 0:
                mask = dilate(mask, thickness)
            out |= mask
            events.append(
                {
                    "axis": str(a.axis),
                    "line": int(line),
                    "gap_start": int(gap_start),
                    "gap_end": int(gap_end),
                    "gap_cells": int(gap_cells),
                    "gap_m": float(length_m),
                    "endpoint_support": float(endpoint_support),
                    "source_run_ids": [int(a.run_id), int(b.run_id)],
                    "strict_less_than_max_gap_m": float(max_gap_m),
                }
            )
    return out.astype(bool), {
        "enabled": True,
        "max_gap_m": float(max_gap_m),
        "strict_less_than_max_gap": True,
        "max_lateral_offset_cells": int(max_lateral_cells),
        "input_run_count": int(len(runs)),
        "filled_gap_count": int(len(events)),
        "filled_cell_count": int(np.count_nonzero(out)),
        "events": events[:512],
    }


def generate_wall_gap_candidates(
    segments: Sequence[WallSegment],
    *,
    free_clean: np.ndarray,
    wall_candidate_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    physical_config: PhysicalWallCompletionConfig | Mapping[str, object] | None = None,
    doorway_config: DoorwayVirtualCutConfig | Mapping[str, object] | None = None,
    missed_scan_config: MissedScanGapClosureConfig | Mapping[str, object] | None = None,
    short_unknown_config: ShortUnknownGapClosureConfig | Mapping[str, object] | None = None,
    single_sided_config: SingleSidedWallExtensionConfig | Mapping[str, object] | None = None,
    runs: Sequence[SnappedWallRun] | None = None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], dict]:
    physical = physical_config if isinstance(physical_config, PhysicalWallCompletionConfig) else PhysicalWallCompletionConfig.from_mapping(physical_config)
    doorway = doorway_config if isinstance(doorway_config, DoorwayVirtualCutConfig) else DoorwayVirtualCutConfig.from_mapping(doorway_config)
    missed = missed_scan_config if isinstance(missed_scan_config, MissedScanGapClosureConfig) else MissedScanGapClosureConfig.from_mapping(missed_scan_config)
    short_unknown = (
        short_unknown_config
        if isinstance(short_unknown_config, ShortUnknownGapClosureConfig)
        else ShortUnknownGapClosureConfig.from_mapping(short_unknown_config)
    )
    single_sided = (
        single_sided_config
        if isinstance(single_sided_config, SingleSidedWallExtensionConfig)
        else SingleSidedWallExtensionConfig.from_mapping(single_sided_config)
    )
    free = np.asarray(free_clean, dtype=bool)
    wall = np.asarray(wall_candidate_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    if runs is None:
        runs, snap_debug = snap_wall_segments_to_runs(
            segments,
            wall,
            resolution_m=float(resolution_m),
            max_angle_to_axis_deg=25.0,
            support_band_cells=2,
            min_run_length_m=0.25,
            min_support_ratio=0.15,
        )
    else:
        snap_debug = {"used_precomputed_runs": True, "run_count": int(len(runs))}
    candidates: list[SeparatorCandidate] = []
    sid = int(start_id)
    seen: set[tuple[str, str, int, int, int]] = set()
    max_lateral_cells = max(0, int(round(max(float(physical.max_lateral_offset_m), 0.20) / max(float(resolution_m), 1e-9))))
    for idx, a in enumerate(runs):
        for b in runs[idx + 1 :]:
            if str(a.axis) != str(b.axis):
                continue
            if abs(int(a.line) - int(b.line)) > max_lateral_cells:
                continue
            gap = _run_gap(a, b)
            if gap is None:
                continue
            gap_start, gap_end = gap
            gap_cells = int(gap_end - gap_start + 1)
            if gap_cells <= 0:
                continue
            length_m = float(gap_cells * float(resolution_m))
            line = int(round(float((int(a.line) + int(b.line)) / 2.0)))
            p0, p1, theta = _candidate_points(str(a.axis), line, gap_start, gap_end)
            stats = _line_stats(p0, p1, free, wall, unknown)
            source_ids = sorted({*map(int, a.source_segment_ids), *map(int, b.source_segment_ids)})
            wall_support = float(min(float(a.confidence), float(b.confidence)))
            base_debug = {
                "axis": str(a.axis),
                "line": int(line),
                "gap_m": float(length_m),
                "gap_cells": int(gap_cells),
                "free_ratio": float(stats["free_ratio"]),
                "unknown_ratio": float(stats["unknown_ratio"]),
                "wall_ratio": float(stats["wall_ratio"]),
                "source_run_ids": [int(a.run_id), int(b.run_id)],
                "p0_before_anchor_extension": [int(v) for v in np.rint(p0).astype(int).tolist()],
                "p1_before_anchor_extension": [int(v) for v in np.rint(p1).astype(int).tolist()],
            }
            for kind, confidence, score in _candidate_kinds_for_gap(
                length_m=length_m,
                free_ratio=float(stats["free_ratio"]),
                unknown_ratio=float(stats["unknown_ratio"]),
                wall_support=wall_support,
                physical=physical,
                missed=missed,
                short_unknown=short_unknown,
                doorway=doorway,
            ):
                key = (kind, str(a.axis), int(line), int(gap_start), int(gap_end))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    SeparatorCandidate(
                        candidate_id=sid,
                        kind=kind,
                        p0_rc=p0.copy(),
                        p1_rc=p1.copy(),
                        theta=theta,
                        length_m=length_m,
                        confidence=float(confidence),
                        source_segment_ids=source_ids,
                        wall_support_score=wall_support,
                        free_gap_score=float(stats["free_ratio"]),
                        doorway_score=float(score if kind == "doorway_virtual_cut" else 0.0),
                        debug={**base_debug, "candidate_source": "wall_run_gap"},
                    )
                )
                sid += 1
    if bool(single_sided.enabled):
        extra, sid = _single_sided_candidates(
            runs,
            free,
            wall,
            unknown,
            resolution_m=float(resolution_m),
            config=single_sided,
            start_id=sid,
            seen=seen,
        )
        candidates.extend(extra)
    return candidates, {
        "physical_wall_completion_enabled": bool(physical.enabled),
        "missed_scan_gap_closure_enabled": bool(missed.enabled),
        "short_unknown_gap_closure_enabled": bool(short_unknown.enabled),
        "doorway_virtual_cut_enabled": bool(doorway.enabled),
        "single_sided_wall_extension_enabled": bool(single_sided.enabled),
        "wall_gap_candidate_count": int(len(candidates)),
        "candidate_count_by_kind": _kind_counts(candidates),
        "snap_debug": snap_debug,
    }


def extend_candidate_to_anchors(
    candidate: SeparatorCandidate,
    *,
    free_clean: np.ndarray,
    wall_candidate_clean: np.ndarray,
    unknown_clean: np.ndarray,
    existing_separator_map: np.ndarray,
    resolution_m: float,
    max_anchor_extension_m: float,
) -> tuple[SeparatorCandidate, dict]:
    free = np.asarray(free_clean, dtype=bool)
    wall = np.asarray(wall_candidate_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    existing = np.asarray(existing_separator_map, dtype=bool)
    max_steps = max(0, int(round(float(max_anchor_extension_m) / max(float(resolution_m), 1e-9))))
    p0_before = np.rint(candidate.p0_rc).astype(np.int32)
    p1_before = np.rint(candidate.p1_rc).astype(np.int32)
    direction = _unit_grid_direction(p0_before, p1_before)
    p0_after, a0 = _extend_endpoint_to_anchor(p0_before, -direction, free, wall, unknown, existing, max_steps)
    p1_after, a1 = _extend_endpoint_to_anchor(p1_before, direction, free, wall, unknown, existing, max_steps)
    candidate.p0_rc = p0_after.astype(np.float32)
    candidate.p1_rc = p1_after.astype(np.float32)
    candidate.length_m = float(max(1, int(np.max(np.abs(p1_after - p0_after)) + 1)) * float(resolution_m))
    score = float((1 if a0["anchored"] else 0) + (1 if a1["anchored"] else 0)) / 2.0
    debug = {
        "p0_before_anchor_extension": [int(v) for v in p0_before.tolist()],
        "p1_before_anchor_extension": [int(v) for v in p1_before.tolist()],
        "p0_after_anchor_extension": [int(v) for v in p0_after.tolist()],
        "p1_after_anchor_extension": [int(v) for v in p1_after.tolist()],
        "anchor_type_p0": str(a0["type"]),
        "anchor_type_p1": str(a1["type"]),
        "anchor_score": float(score),
        "anchor_steps_p0": int(a0["steps"]),
        "anchor_steps_p1": int(a1["steps"]),
    }
    candidate.debug.update(debug)
    return candidate, debug


def _run_gap(a: SnappedWallRun, b: SnappedWallRun) -> tuple[int, int] | None:
    left, right = (a, b) if int(a.start) <= int(b.start) else (b, a)
    if int(left.end) >= int(right.start) - 1:
        return None
    return int(left.end) + 1, int(right.start) - 1


def _line_extension_result(
    line: FilteredWallLine,
    endpoint: str,
    p_start: np.ndarray,
    p_hit: np.ndarray,
    sampled: Sequence[tuple[int, int]],
    *,
    hit_type: str,
    hit_candidate_id: int | None,
    reject_reason: str | None,
    pass_id: int,
    extension_id: int,
    resolution_m: float,
    config: LineExtensionConfig,
    debug: Mapping[str, object] | None = None,
) -> LineExtensionHit:
    # Ratios are computed below from explicit masks by the caller-populated sampled
    # cells. Empty paths stay rejected by length/free-cell checks.
    return _finalize_line_extension_hit(
        line,
        endpoint,
        p_start,
        p_hit,
        sampled,
        hit_type=hit_type,
        hit_candidate_id=hit_candidate_id,
        reject_reason=reject_reason,
        pass_id=pass_id,
        extension_id=extension_id,
        resolution_m=float(resolution_m),
        config=config,
        debug=dict(debug or {}),
    )


def _finalize_line_extension_hit(
    line: FilteredWallLine,
    endpoint: str,
    p_start: np.ndarray,
    p_hit: np.ndarray,
    sampled: Sequence[tuple[int, int]],
    *,
    hit_type: str,
    hit_candidate_id: int | None,
    reject_reason: str | None,
    pass_id: int,
    extension_id: int,
    resolution_m: float,
    config: LineExtensionConfig,
    debug: dict,
) -> LineExtensionHit:
    # The masks are attached by trace_line_extension just before this helper is
    # called. Keeping them out of the public debug prevents huge JSON payloads.
    free = np.asarray(debug.pop("_free_clean"), dtype=bool)
    wall = np.asarray(debug.pop("_wall_target"), dtype=bool)
    unknown = np.asarray(debug.pop("_unknown_clean"), dtype=bool)
    interior = list(sampled[:-1]) if len(sampled) > 1 else list(sampled)
    length_m = float(max(0, len(sampled)) * float(resolution_m))
    free_ratio = _sample_ratio(interior, free)
    unknown_ratio = _sample_ratio(interior, unknown)
    wall_ratio = _sample_ratio(interior, wall)
    reason = reject_reason
    if reason is None:
        if length_m < float(config.min_extension_m) or length_m > float(config.max_extension_m):
            reason = "reject_extension_length_out_of_door_range"
        elif free_ratio < float(config.free_ratio_min):
            reason = "reject_extension_not_enough_free"
        elif unknown_ratio > float(config.unknown_ratio_max):
            reason = "reject_extension_hit_unknown"
        elif wall_ratio > float(config.wall_mid_ratio_max):
            reason = "reject_extension_crosses_mid_wall"
        elif str(hit_type) not in {"real_wall", "virtual_door", "virtual_neck"}:
            reason = "reject_extension_no_wall_or_virtual_door_hit"
    confidence = float(
        np.clip(
            0.45 * float(line.confidence)
            + 0.35 * float(free_ratio)
            + 0.10 * (1.0 - min(1.0, unknown_ratio))
            + 0.10 * (1.0 if str(hit_type) in {"real_wall", "virtual_door", "virtual_neck"} else 0.0),
            0.0,
            1.0,
        )
    )
    return LineExtensionHit(
        extension_id=int(extension_id),
        source_line_id=int(line.line_id),
        source_endpoint=str(endpoint),
        pass_id=int(pass_id),
        p_start_rc=np.asarray(p_start, dtype=np.float32),
        p_hit_rc=np.asarray(p_hit, dtype=np.float32),
        theta=float(line.theta),
        length_m=length_m,
        interior_free_ratio=float(free_ratio),
        interior_unknown_ratio=float(unknown_ratio),
        interior_wall_ratio=float(wall_ratio),
        hit_type=str(hit_type),
        hit_candidate_id=hit_candidate_id,
        confidence=confidence,
        reject_reason=reason,
        debug={
            **dict(debug),
            "source_line_confidence": float(line.confidence),
            "source_line_length_m": float(line.length_m),
            "sampled_cell_count": int(len(sampled)),
        },
    )


def _extension_start_and_direction(line: FilteredWallLine, endpoint: str) -> tuple[np.ndarray, np.ndarray]:
    p0 = np.asarray(line.p0_rc, dtype=np.float32)
    p1 = np.asarray(line.p1_rc, dtype=np.float32)
    if str(endpoint) == "p0":
        delta = p0 - p1
        start = p0
    else:
        delta = p1 - p0
        start = p1
    if abs(float(delta[0])) >= abs(float(delta[1])):
        direction = np.asarray([1.0 if float(delta[0]) >= 0.0 else -1.0, 0.0], dtype=np.float32)
    else:
        direction = np.asarray([0.0, 1.0 if float(delta[1]) >= 0.0 else -1.0], dtype=np.float32)
    return start.astype(np.float32), direction


def _sample_ratio(cells: Sequence[tuple[int, int]], mask: np.ndarray) -> float:
    if not cells:
        return 0.0
    arr = np.asarray(mask, dtype=bool)
    count = 0
    valid = 0
    for r, c in cells:
        if 0 <= int(r) < arr.shape[0] and 0 <= int(c) < arr.shape[1]:
            valid += 1
            if bool(arr[int(r), int(c)]):
                count += 1
    return float(count) / float(max(1, valid))


def _snap_hit_point_to_target(
    rc: np.ndarray,
    target_mask: np.ndarray | None,
    direction: np.ndarray,
    *,
    radius_cells: int,
) -> np.ndarray:
    if target_mask is None:
        return np.asarray(rc, dtype=np.float32)
    target = np.asarray(target_mask, dtype=bool)
    point = np.asarray(rc, dtype=np.int32)
    if not _inside(point, target.shape):
        return point.astype(np.float32)
    if bool(target[int(point[0]), int(point[1])]):
        return point.astype(np.float32)
    radius = max(0, int(radius_cells))
    r0, r1 = max(0, int(point[0]) - radius), min(target.shape[0], int(point[0]) + radius + 1)
    c0, c1 = max(0, int(point[1]) - radius), min(target.shape[1], int(point[1]) + radius + 1)
    rows, cols = np.nonzero(target[r0:r1, c0:c1])
    if rows.size == 0:
        return point.astype(np.float32)
    coords = np.stack([rows + r0, cols + c0], axis=1).astype(np.int32)
    delta = coords.astype(np.float32) - point.astype(np.float32)
    unit = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-6:
        unit = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        unit = unit / norm
    projection = delta @ unit
    forward = projection >= -1e-6
    if np.any(forward):
        coords = coords[forward]
        delta = delta[forward]
        projection = projection[forward]
    if abs(float(unit[0])) >= abs(float(unit[1])):
        lateral = np.abs(delta[:, 1])
    else:
        lateral = np.abs(delta[:, 0])
    dist2 = np.sum(delta * delta, axis=1)
    order = np.lexsort((dist2, np.abs(projection), lateral))
    return coords[int(order[0])].astype(np.float32)


def _extension_reason_counts(hits: Sequence[LineExtensionHit]) -> dict:
    out: dict[str, int] = {}
    for hit in hits:
        reason = str(hit.reject_reason or "accepted")
        out[reason] = out.get(reason, 0) + 1
    return out


def _extension_can_seed_intersection(hit: LineExtensionHit) -> bool:
    reason = "" if hit.reject_reason is None else str(hit.reject_reason)
    if reason in {
        "reject_extension_blocked_by_near_wall",
        "reject_extension_crosses_mid_wall",
        "reject_extension_not_enough_free",
        "reject_extension_length_out_of_door_range",
        "reject_low_line_extension_confidence",
    }:
        return False
    if str(hit.hit_type) == "blocked_wall":
        return False
    return bool(float(hit.length_m) > 0.0)


def _extension_axis_info(hit: LineExtensionHit, shape: tuple[int, int]) -> tuple[str, int, int, int] | None:
    p0 = np.rint(np.asarray(hit.p_start_rc, dtype=np.float32)).astype(np.int32)
    p1 = np.rint(np.asarray(hit.p_hit_rc, dtype=np.float32)).astype(np.int32)
    if not _inside(p0, shape) or not _inside(p1, shape):
        return None
    dr = int(p1[0] - p0[0])
    dc = int(p1[1] - p0[1])
    if abs(dc) >= abs(dr):
        fixed = int(round(float((int(p0[0]) + int(p1[0])) / 2.0)))
        return "horizontal", fixed, int(min(p0[1], p1[1])), int(max(p0[1], p1[1]))
    fixed = int(round(float((int(p0[1]) + int(p1[1])) / 2.0)))
    return "vertical", fixed, int(min(p0[0], p1[0])), int(max(p0[0], p1[0]))


def _ordered_line_cells(p0_rc: np.ndarray, p1_rc: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
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
    coords = np.stack([rows[inside], cols[inside]], axis=1)
    if len(coords) <= 1:
        return coords.astype(np.int32)
    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
    return coords[keep].astype(np.int32)


def _candidate_from_extension_intersection(
    hit: LineExtensionHit,
    *,
    other_hit: LineExtensionHit,
    intersection_rc: np.ndarray,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    line_config: LineExtensionConfig,
    door_config: DoorNeckConfig,
    candidate_id: int,
) -> SeparatorCandidate | None:
    free = np.asarray(free_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    start = np.rint(np.asarray(hit.p_start_rc, dtype=np.float32)).astype(np.int32)
    end = np.rint(np.asarray(intersection_rc, dtype=np.float32)).astype(np.int32)
    if not _inside(start, free.shape) or not _inside(end, free.shape):
        return None
    cells = _ordered_line_cells(start.astype(np.float32), end.astype(np.float32), free.shape)
    if len(cells) <= 1:
        return None
    interior = [(int(r), int(c)) for r, c in cells[1:]]
    length_m = float((len(cells) - 1) * float(resolution_m))
    if length_m < float(door_config.min_width_m) or length_m > float(door_config.max_width_m):
        return None
    free_ratio = _sample_ratio(interior, free)
    unknown_ratio = _sample_ratio(interior, unknown)
    if free_ratio < float(line_config.free_ratio_min) or unknown_ratio > float(line_config.unknown_ratio_max):
        return None
    theta = 0.0 if int(start[0]) == int(end[0]) else float(np.pi / 2.0)
    confidence = float(
        np.clip(
            0.35 * float(hit.confidence)
            + 0.25 * float(other_hit.confidence)
            + 0.30 * float(free_ratio)
            + 0.10 * (1.0 - min(1.0, unknown_ratio)),
            0.0,
            1.0,
        )
    )
    if confidence < float(door_config.min_confidence):
        return None
    return SeparatorCandidate(
        candidate_id=int(candidate_id),
        kind="line_extension_door_neck",
        p0_rc=start.astype(np.float32),
        p1_rc=end.astype(np.float32),
        theta=float(theta),
        length_m=float(length_m),
        confidence=float(confidence),
        source_segment_ids=[int(hit.source_line_id), int(other_hit.source_line_id)],
        wall_support_score=float(max(float(hit.confidence), float(other_hit.confidence))),
        free_gap_score=float(free_ratio),
        doorway_score=float(free_ratio),
        debug={
            "candidate_source": "extension_intersection",
            "kind_detail": "door_neck",
            "pass_id": int(max(int(hit.pass_id), int(other_hit.pass_id))),
            "source_extension_ids": [int(hit.extension_id), int(other_hit.extension_id)],
            "source_line_ids": [int(hit.source_line_id), int(other_hit.source_line_id)],
            "hit_types": ["virtual_neck"],
            "intersection_rc": [int(end[0]), int(end[1])],
            "width_m": float(length_m),
            "neck_score": float(free_ratio),
            "doorway_score": float(free_ratio),
            "intersection_free_ratio": float(free_ratio),
            "intersection_unknown_ratio": float(unknown_ratio),
            "primary_extension": hit.to_dict(),
            "cross_extension": other_hit.to_dict(),
        },
    )


def _candidate_points(axis: str, line: int, gap_start: int, gap_end: int) -> tuple[np.ndarray, np.ndarray, float]:
    if axis == "horizontal":
        return (
            np.asarray([line, gap_start], dtype=np.float32),
            np.asarray([line, gap_end], dtype=np.float32),
            0.0,
        )
    return (
        np.asarray([gap_start, line], dtype=np.float32),
        np.asarray([gap_end, line], dtype=np.float32),
        float(np.pi / 2.0),
    )


def _line_stats(p0: np.ndarray, p1: np.ndarray, free: np.ndarray, wall: np.ndarray, unknown: np.ndarray) -> dict:
    line = rasterize_line(p0, p1, free.shape)
    cells = max(1, int(np.count_nonzero(line)))
    return {
        "free_ratio": float(np.count_nonzero(line & free)) / float(cells),
        "unknown_ratio": float(np.count_nonzero(line & unknown)) / float(cells),
        "wall_ratio": float(np.count_nonzero(line & wall)) / float(cells),
        "cell_count": int(cells),
    }


def _candidate_kinds_for_gap(
    *,
    length_m: float,
    free_ratio: float,
    unknown_ratio: float,
    wall_support: float,
    physical: PhysicalWallCompletionConfig,
    missed: MissedScanGapClosureConfig,
    short_unknown: ShortUnknownGapClosureConfig,
    doorway: DoorwayVirtualCutConfig,
) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    shorter_than_door = bool(length_m < float(doorway.min_width_m))
    gap_tolerance_m = 0.051
    if bool(physical.enabled) and length_m <= float(physical.max_gap_m) + gap_tolerance_m and wall_support >= float(physical.min_endpoint_support):
        if shorter_than_door or free_ratio <= 0.80 or unknown_ratio >= 0.15:
            out.append(("physical_wall_completion", float(np.clip(0.75 + 0.2 * wall_support + 0.05 * unknown_ratio, 0.0, 1.0)), 0.0))
    if bool(missed.enabled) and length_m <= float(missed.max_gap_m) + gap_tolerance_m and (shorter_than_door or free_ratio <= float(missed.max_free_ratio)):
        if unknown_ratio <= float(missed.max_unknown_ratio):
            out.append(("missed_scan_gap_closure", float(np.clip(0.65 + 0.25 * wall_support + 0.1 * (1.0 - free_ratio), 0.0, 1.0)), 0.0))
    if bool(short_unknown.enabled) and length_m <= float(short_unknown.max_gap_m) and unknown_ratio >= float(short_unknown.min_unknown_ratio):
        out.append(("short_unknown_gap_closure", float(np.clip(0.55 + 0.35 * unknown_ratio + 0.1 * wall_support, 0.0, 1.0)), 0.0))
    if (
        bool(doorway.enabled)
        and float(doorway.min_width_m) <= length_m <= float(doorway.max_width_m)
        and wall_support >= float(doorway.min_wall_endpoint_support)
        and (not bool(doorway.require_free_gap) or free_ratio >= 0.55)
    ):
        mid = 0.5 * (float(doorway.min_width_m) + float(doorway.max_width_m))
        width_score = 1.0 - min(1.0, abs(float(length_m) - mid) / max(mid, 1e-6))
        out.append(("doorway_virtual_cut", float(np.clip(0.60 + 0.30 * free_ratio + 0.10 * width_score, 0.0, 1.0)), free_ratio))
    return out


def _single_sided_candidates(
    runs: Sequence[SnappedWallRun],
    free: np.ndarray,
    wall: np.ndarray,
    unknown: np.ndarray,
    *,
    resolution_m: float,
    config: SingleSidedWallExtensionConfig,
    start_id: int,
    seen: set[tuple[str, str, int, int, int]],
) -> tuple[list[SeparatorCandidate], int]:
    max_steps = max(1, int(round(float(config.max_extension_m) / max(float(resolution_m), 1e-9))))
    candidates: list[SeparatorCandidate] = []
    cid = int(start_id)
    for run in runs:
        for side, sign in (("start", -1), ("end", 1)):
            if run.axis == "horizontal":
                base = np.asarray([run.line, run.start if side == "start" else run.end], dtype=np.int32)
                direction = np.asarray([0, sign], dtype=np.int32)
            else:
                base = np.asarray([run.start if side == "start" else run.end, run.line], dtype=np.int32)
                direction = np.asarray([sign, 0], dtype=np.int32)
            endpoint, anchor = _extend_endpoint_to_anchor(base, direction, free, wall, unknown, np.zeros_like(free, dtype=bool), max_steps)
            if not bool(anchor["anchored"]) or int(anchor["steps"]) <= 1:
                continue
            p0 = base.astype(np.float32)
            p1 = endpoint.astype(np.float32)
            stats = _line_stats(p0, p1, free, wall, unknown)
            if stats["free_ratio"] > float(config.max_open_free_ratio):
                continue
            key = ("single_sided_wall_extension", str(run.axis), int(run.line), int(base[0] * free.shape[1] + base[1]), int(endpoint[0] * free.shape[1] + endpoint[1]))
            if key in seen:
                continue
            seen.add(key)
            theta = 0.0 if run.axis == "horizontal" else float(np.pi / 2.0)
            candidates.append(
                SeparatorCandidate(
                    candidate_id=cid,
                    kind="single_sided_wall_extension",
                    p0_rc=p0,
                    p1_rc=p1,
                    theta=theta,
                    length_m=float(max(1, int(anchor["steps"])) * float(resolution_m)),
                    confidence=float(np.clip(0.45 + 0.45 * float(run.confidence), 0.0, 1.0)),
                    source_segment_ids=[int(v) for v in run.source_segment_ids],
                    wall_support_score=float(run.confidence),
                    free_gap_score=float(stats["free_ratio"]),
                    debug={
                        "axis": str(run.axis),
                        "line": int(run.line),
                        "gap_m": float(max(1, int(anchor["steps"])) * float(resolution_m)),
                        "free_ratio": float(stats["free_ratio"]),
                        "unknown_ratio": float(stats["unknown_ratio"]),
                        "source_run_ids": [int(run.run_id)],
                        "candidate_source": "single_sided_wall_extension",
                        "extension_anchor_type": str(anchor["type"]),
                    },
                )
            )
            cid += 1
    return candidates, cid


def _unit_grid_direction(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    delta = np.asarray(p1, dtype=np.int32) - np.asarray(p0, dtype=np.int32)
    if abs(int(delta[0])) >= abs(int(delta[1])):
        return np.asarray([1 if int(delta[0]) >= 0 else -1, 0], dtype=np.int32)
    return np.asarray([0, 1 if int(delta[1]) >= 0 else -1], dtype=np.int32)


def _extend_endpoint_to_anchor(
    start: np.ndarray,
    direction: np.ndarray,
    free: np.ndarray,
    wall: np.ndarray,
    unknown: np.ndarray,
    existing: np.ndarray,
    max_steps: int,
) -> tuple[np.ndarray, dict]:
    point = np.asarray(start, dtype=np.int32).copy()
    anchor = _anchor_type(point, free, wall, unknown, existing)
    if anchor != "":
        return point, {"anchored": True, "type": anchor, "steps": 0}
    last = point.copy()
    for step in range(1, int(max_steps) + 1):
        candidate = point + np.asarray(direction, dtype=np.int32) * int(step)
        if not _inside(candidate, free.shape):
            return last, {"anchored": True, "type": "touch_map_boundary", "steps": int(step - 1)}
        last = candidate
        anchor = _anchor_type(candidate, free, wall, unknown, existing)
        if anchor != "":
            return candidate, {"anchored": True, "type": anchor, "steps": int(step)}
    return last, {"anchored": False, "type": "none", "steps": int(max_steps)}


def _anchor_type(point: np.ndarray, free: np.ndarray, wall: np.ndarray, unknown: np.ndarray, existing: np.ndarray) -> str:
    r, c = int(point[0]), int(point[1])
    if not _inside(point, free.shape):
        return "touch_map_boundary"
    if r == 0 or c == 0 or r == free.shape[0] - 1 or c == free.shape[1] - 1:
        return "touch_map_boundary"
    r0, r1 = max(0, r - 1), min(free.shape[0], r + 2)
    c0, c1 = max(0, c - 1), min(free.shape[1], c + 2)
    if np.any(existing[r0:r1, c0:c1]):
        return "touch_existing_separator"
    if np.any(wall[r0:r1, c0:c1]):
        return "touch_wall"
    if np.any(unknown[r0:r1, c0:c1]):
        return "touch_unknown_boundary"
    if not bool(free[r, c]) or np.any(~free[r0:r1, c0:c1]):
        return "touch_room_boundary"
    return ""


def _inside(point: np.ndarray, shape: tuple[int, int]) -> bool:
    return 0 <= int(point[0]) < int(shape[0]) and 0 <= int(point[1]) < int(shape[1])


def separator_mask_for_candidate(candidate: SeparatorCandidate, shape: tuple[int, int], thickness_cells: int = 0) -> np.ndarray:
    mask = candidate.mask(shape)
    if int(thickness_cells) > 0:
        mask = dilate(mask, int(thickness_cells))
    return mask.astype(bool)


def _kind_counts(candidates: Sequence[SeparatorCandidate]) -> dict:
    out: dict[str, int] = {}
    for candidate in candidates:
        out[str(candidate.kind)] = out.get(str(candidate.kind), 0) + 1
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
