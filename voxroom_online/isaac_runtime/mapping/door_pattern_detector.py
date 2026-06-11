from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage


@dataclass
class DoorPatternConfig:
    enabled: bool = True
    strict_rule_enabled: bool = True
    min_free_run_cells: int = 3
    min_occupied_run_cells: int = 3
    min_unknown_run_cells: int = 1
    scan_dr: int = -1
    scan_dc: int = 0
    turn_dirs: tuple[tuple[int, int], ...] = ((0, -1), (0, 1))
    allow_diagonal_turns: bool = False
    rule_a_turn_probe_max_cells: int | None = None
    unknown_tail_padding_cells: int = 8
    reject_unknown_below: bool = True
    allow_start_lift: bool = False
    max_start_lift_cells: int = 0
    partial_seed_enabled: bool = False
    partial_min_free_run_cells: int = 2
    partial_min_occupied_seed_cells: int = 1
    partial_max_unknown_below_cells: int = 0
    partial_allow_unknown_tail_missing: bool = True
    partial_allow_occupied_run_shorter_than_strict: bool = True
    partial_seed_connectivity: int = 8
    partial_seed_min_component_cells: int = 2
    partial_seed_max_component_cells: int = 80
    partial_turn_probe_max_cells: int | None = None
    line_fit_method: str = "pca_then_ransac"
    line_fit_min_inliers: int = 2
    line_fit_max_residual_cells: float = 1.25
    line_fit_min_length_cells: int = 2
    door_line_extension_enabled: bool = True
    door_line_extend_max_cells: int = 40
    door_line_max_width_m: float = 1.60
    door_line_min_width_m: float = 0.20
    door_line_inner_unknown_ratio_max: float = 0.25
    door_line_inner_wall_ratio_max: float = 0.15
    door_line_min_free_or_seed_ratio: float = 0.60
    door_line_wall_anchor_radius_cells: int = 1
    door_line_wall_anchor_min_cells: int = 1
    door_line_use_terminal_wall_as_anchor: bool = True
    reject_if_extension_hits_other_door: bool = True
    other_door_intersection_radius_cells: int = 1
    door_cut_thickness_cells: int = 1
    door_cut_lateral_radius_cells: int = 1
    merge_nearby_door_cells: int = 2
    max_candidates: int = 4096

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "DoorPatternConfig":
        if isinstance(data, cls):
            return data
        raw = dict(data or {})
        if "turn_dirs" in raw:
            raw["turn_dirs"] = tuple((int(item[0]), int(item[1])) for item in list(raw["turn_dirs"] or []))
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class DoorPatternCandidate:
    candidate_id: int
    pattern_type: str
    start_rc: tuple[int, int]
    free_cells: list[tuple[int, int]] = field(default_factory=list)
    occupied_cells: list[tuple[int, int]] = field(default_factory=list)
    unknown_cells: list[tuple[int, int]] = field(default_factory=list)
    turn_rc: tuple[int, int] | None = None
    door_cut_cells: list[tuple[int, int]] = field(default_factory=list)
    accepted: bool = False
    reject_reason: str | None = None

    def serialize(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.candidate_id),
            "pattern_type": str(self.pattern_type),
            "start_rc": [int(self.start_rc[0]), int(self.start_rc[1])],
            "turn_rc": None if self.turn_rc is None else [int(self.turn_rc[0]), int(self.turn_rc[1])],
            "free_len": int(len(self.free_cells)),
            "occupied_len": int(len(self.occupied_cells)),
            "unknown_len": int(len(self.unknown_cells)),
            "door_cut_len": int(len(self.door_cut_cells)),
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
        }


@dataclass
class DoorLineExtensionCandidate:
    candidate_id: int
    seed_component_id: int
    seed_cells: list[tuple[int, int]]
    fitted_direction_rc: tuple[float, float]
    fitted_center_rc: tuple[float, float]
    line_cells_before_extension: list[tuple[int, int]] = field(default_factory=list)
    extension_cells: list[tuple[int, int]] = field(default_factory=list)
    door_cut_cells: list[tuple[int, int]] = field(default_factory=list)
    wall_anchor_a: tuple[int, int] | None = None
    wall_anchor_b: tuple[int, int] | None = None
    endpoint_a_kind: str = "unknown"
    endpoint_b_kind: str = "unknown"
    accepted: bool = False
    reject_reason: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.candidate_id),
            "seed_component_id": int(self.seed_component_id),
            "seed_len": int(len(self.seed_cells)),
            "fitted_direction_rc": [float(self.fitted_direction_rc[0]), float(self.fitted_direction_rc[1])],
            "fitted_center_rc": [float(self.fitted_center_rc[0]), float(self.fitted_center_rc[1])],
            "line_cells_before_extension_len": int(len(self.line_cells_before_extension)),
            "extension_cells_len": int(len(self.extension_cells)),
            "door_cut_len": int(len(self.door_cut_cells)),
            "wall_anchor_a": None if self.wall_anchor_a is None else [int(self.wall_anchor_a[0]), int(self.wall_anchor_a[1])],
            "wall_anchor_b": None if self.wall_anchor_b is None else [int(self.wall_anchor_b[0]), int(self.wall_anchor_b[1])],
            "endpoint_a_kind": str(self.endpoint_a_kind),
            "endpoint_b_kind": str(self.endpoint_b_kind),
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
            "debug": jsonable(self.debug),
        }


@dataclass
class DoorPatternDetectionResult:
    detected_door_mask: np.ndarray
    door_cut_mask: np.ndarray
    pattern_type_map: np.ndarray
    partial_door_seed_mask: np.ndarray
    partial_door_line_mask: np.ndarray
    partial_door_extension_cut_mask: np.ndarray
    strict_door_cut_mask: np.ndarray
    rejected_door_extension_mask: np.ndarray
    candidates: list[DoorPatternCandidate]
    line_candidates: list[DoorLineExtensionCandidate]
    debug: dict[str, Any]


def detect_pre_extension_doors(
    *,
    free_mask: np.ndarray,
    occupied_mask: np.ndarray,
    unknown_mask: np.ndarray,
    config: DoorPatternConfig | Mapping[str, object] | None = None,
    observed_mask: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    resolution_m: float = 1.0,
) -> DoorPatternDetectionResult:
    """Detect strict occupancy-pattern doors before roomseg wall extension."""

    cfg = config if isinstance(config, DoorPatternConfig) else DoorPatternConfig.from_mapping(config)
    free = np.asarray(free_mask, dtype=bool).copy()
    occ = np.asarray(occupied_mask, dtype=bool).copy()
    unk = np.asarray(unknown_mask, dtype=bool).copy()

    if free.shape != occ.shape or free.shape != unk.shape:
        raise ValueError("door pattern detector masks must have same HxW shape")

    detected_door_mask = np.zeros_like(free, dtype=bool)
    door_cut_mask = np.zeros_like(free, dtype=bool)
    strict_door_cut_mask = np.zeros_like(free, dtype=bool)
    partial_door_seed_mask = np.zeros_like(free, dtype=bool)
    partial_door_line_mask = np.zeros_like(free, dtype=bool)
    partial_door_extension_cut_mask = np.zeros_like(free, dtype=bool)
    rejected_door_extension_mask = np.zeros_like(free, dtype=bool)
    pattern_type_map = np.zeros(free.shape, dtype=np.uint8)
    candidates: list[DoorPatternCandidate] = []
    line_candidates: list[DoorLineExtensionCandidate] = []

    if not bool(cfg.enabled):
        return DoorPatternDetectionResult(
            detected_door_mask=detected_door_mask,
            door_cut_mask=door_cut_mask,
            pattern_type_map=pattern_type_map,
            partial_door_seed_mask=partial_door_seed_mask,
            partial_door_line_mask=partial_door_line_mask,
            partial_door_extension_cut_mask=partial_door_extension_cut_mask,
            strict_door_cut_mask=strict_door_cut_mask,
            rejected_door_extension_mask=rejected_door_extension_mask,
            candidates=[],
            line_candidates=[],
            debug={
                "pre_extension_door_detection_enabled": False,
                "pre_extension_door_num_candidates": 0,
                "pre_extension_door_num_accepted": 0,
            },
        )

    rois = resolve_detection_rois(free=free, occ=occ, unk=unk, observed_mask=observed_mask, roi=roi, cfg=cfg)
    next_id = 1

    for current_roi in rois:
        min_r, min_c, max_r, max_c = current_roi
        _ = min_r
        bottom_r = max_r
        for c in range(min_c, max_c + 1):
            if not bool(cfg.strict_rule_enabled):
                continue
            start = resolve_bottom_start(
                free=free,
                occ=occ,
                unk=unk,
                bottom_r=bottom_r,
                col=c,
                roi=current_roi,
                cfg=cfg,
            )
            if start is None:
                continue

            cand_b = try_match_rule_b_free_occupied_unknown(
                candidate_id=next_id,
                start=start,
                free=free,
                occ=occ,
                unk=unk,
                roi=current_roi,
                cfg=cfg,
            )
            if cand_b.accepted:
                candidates.append(cand_b)
                mark_candidate(cand_b, detected_door_mask, strict_door_cut_mask, pattern_type_map, 2, free)
                next_id += 1
                if next_id > int(cfg.max_candidates):
                    break
                continue

            cand_a = try_match_rule_a_free_turn_occupied(
                candidate_id=next_id,
                start=start,
                free=free,
                occ=occ,
                unk=unk,
                roi=current_roi,
                cfg=cfg,
            )
            if cand_a.accepted:
                candidates.append(cand_a)
                mark_candidate(cand_a, detected_door_mask, strict_door_cut_mask, pattern_type_map, 1, free)
                next_id += 1
                if next_id > int(cfg.max_candidates):
                    break

    strict_door_cut_mask = merge_nearby_door_cuts(strict_door_cut_mask, cfg) & free
    if bool(cfg.partial_seed_enabled):
        partial_door_seed_mask, partial_seed_type_map = detect_partial_door_seed_mask(
            free=free,
            occ=occ,
            unk=unk,
            rois=rois,
            cfg=cfg,
        )
        _ = partial_seed_type_map
        if bool(cfg.door_line_extension_enabled):
            line_candidates = fit_and_extend_partial_door_lines(
                seed_mask=partial_door_seed_mask,
                free=free,
                occ=occ,
                unk=unk,
                strict_door_detected=detected_door_mask,
                strict_door_cut=strict_door_cut_mask,
                cfg=cfg,
                resolution_m=float(resolution_m),
            )
            for line_candidate in line_candidates:
                cells = line_candidate.extension_cells or line_candidate.line_cells_before_extension
                for r, c in cells:
                    if 0 <= int(r) < free.shape[0] and 0 <= int(c) < free.shape[1]:
                        if line_candidate.accepted:
                            partial_door_line_mask[int(r), int(c)] = True
                        else:
                            rejected_door_extension_mask[int(r), int(c)] = True
                if line_candidate.accepted:
                    for r, c in line_candidate.door_cut_cells:
                        if bool(free[int(r), int(c)]):
                            partial_door_extension_cut_mask[int(r), int(c)] = True
                            detected_door_mask[int(r), int(c)] = True

    partial_door_extension_cut_mask = partial_door_extension_cut_mask & free
    door_cut_mask = merge_nearby_door_cuts(strict_door_cut_mask | partial_door_extension_cut_mask, cfg) & free
    rejected_counts = Counter(str(c.reject_reason or "accepted") for c in line_candidates if not bool(c.accepted))
    debug = {
        "pre_extension_door_detection_enabled": True,
        "pre_extension_door_num_candidates": int(len(candidates)),
        "pre_extension_door_num_accepted": int(sum(c.accepted for c in candidates)),
        "pre_extension_door_rule_a_count": int(
            sum(c.accepted and c.pattern_type == "free_turn_occupied" for c in candidates)
        ),
        "pre_extension_door_rule_b_count": int(
            sum(c.accepted and c.pattern_type == "free_occupied_unknown" for c in candidates)
        ),
        "pre_extension_door_cut_cells": int(np.count_nonzero(door_cut_mask)),
        "pre_extension_door_detected_cells": int(np.count_nonzero(detected_door_mask)),
        "pre_extension_door_rois": [list(map(int, r)) for r in rois],
        "pre_extension_door_candidates": [c.serialize() for c in candidates],
        "strict_pre_extension_door_cut_mask": strict_door_cut_mask,
        "strict_pre_extension_door_count": int(sum(c.accepted for c in candidates)),
        "partial_door_seed_mask": partial_door_seed_mask,
        "partial_door_seed_component_count": int(_component_count(partial_door_seed_mask, int(cfg.partial_seed_connectivity))),
        "partial_door_line_mask": partial_door_line_mask,
        "partial_door_extension_cut_mask": partial_door_extension_cut_mask,
        "partial_door_line_candidate_count": int(len(line_candidates)),
        "partial_door_line_accepted_count": int(sum(c.accepted for c in line_candidates)),
        "partial_door_line_rejected_count": int(sum(not c.accepted for c in line_candidates)),
        "partial_door_line_reject_reason_counts": dict(rejected_counts),
        "rejected_door_extension_mask": rejected_door_extension_mask,
        "partial_door_line_candidates": [c.serialize() for c in line_candidates],
    }
    return DoorPatternDetectionResult(
        detected_door_mask=detected_door_mask,
        door_cut_mask=door_cut_mask,
        pattern_type_map=pattern_type_map,
        partial_door_seed_mask=partial_door_seed_mask,
        partial_door_line_mask=partial_door_line_mask,
        partial_door_extension_cut_mask=partial_door_extension_cut_mask,
        strict_door_cut_mask=strict_door_cut_mask,
        rejected_door_extension_mask=rejected_door_extension_mask,
        candidates=candidates,
        line_candidates=line_candidates,
        debug=debug,
    )


def resolve_detection_rois(
    *,
    free: np.ndarray,
    occ: np.ndarray,
    unk: np.ndarray,
    observed_mask: np.ndarray | None,
    roi: tuple[int, int, int, int] | None,
    cfg: DoorPatternConfig,
) -> list[tuple[int, int, int, int]]:
    _ = unk
    h, w = free.shape
    if roi is not None:
        clipped = clip_roi(roi, h, w)
        return [] if clipped is None else [clipped]
    if observed_mask is not None:
        obs = np.asarray(observed_mask, dtype=bool)
        if obs.shape != free.shape:
            raise ValueError("observed_mask must have same HxW shape")
        bbox = bbox_from_mask(obs)
        return [] if bbox is None else [bbox]
    bbox = bbox_from_mask(free | occ)
    if bbox is None:
        return []
    min_r, min_c, max_r, max_c = bbox
    min_r = max(0, int(min_r) - max(0, int(cfg.unknown_tail_padding_cells)))
    return [(min_r, int(min_c), int(max_r), int(max_c))]


def clip_roi(roi: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int] | None:
    min_r, min_c, max_r, max_c = (int(v) for v in roi)
    min_r = max(0, min(min_r, h - 1))
    max_r = max(0, min(max_r, h - 1))
    min_c = max(0, min(min_c, w - 1))
    max_c = max(0, min(max_c, w - 1))
    if min_r > max_r or min_c > max_c:
        return None
    return min_r, min_c, max_r, max_c


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size <= 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def resolve_bottom_start(
    *,
    free: np.ndarray,
    occ: np.ndarray,
    unk: np.ndarray,
    bottom_r: int,
    col: int,
    roi: tuple[int, int, int, int],
    cfg: DoorPatternConfig,
) -> tuple[int, int] | None:
    min_r, _min_c, _max_r, _max_c = roi
    start = (int(bottom_r), int(col))
    if cell_state(start, free, occ, unk) == "F":
        return start
    if not bool(cfg.allow_start_lift):
        return None
    for lift in range(1, max(0, int(cfg.max_start_lift_cells)) + 1):
        row = int(bottom_r) + int(cfg.scan_dr) * lift
        if row < min_r:
            break
        lifted = (row, int(col))
        if cell_state(lifted, free, occ, unk) == "F":
            return lifted
    return None


def try_match_rule_b_free_occupied_unknown(
    *,
    candidate_id: int,
    start: tuple[int, int],
    free: np.ndarray,
    occ: np.ndarray,
    unk: np.ndarray,
    roi: tuple[int, int, int, int],
    cfg: DoorPatternConfig,
) -> DoorPatternCandidate:
    cells = trace_line(start, int(cfg.scan_dr), int(cfg.scan_dc), roi)
    i = 0
    free_cells: list[tuple[int, int]] = []
    occupied_cells: list[tuple[int, int]] = []
    unknown_cells: list[tuple[int, int]] = []

    while i < len(cells) and is_strict_free(cells[i], free, occ, unk):
        free_cells.append(cells[i])
        i += 1
    if len(free_cells) < int(cfg.min_free_run_cells):
        return reject(candidate_id, "free_occupied_unknown", start, "free_run_too_short")
    if bool(cfg.reject_unknown_below) and i < len(cells) and is_unknown(cells[i], free, occ, unk):
        return reject(candidate_id, "free_occupied_unknown", start, "unknown_below_occupied")

    while i < len(cells) and is_strict_occupied(cells[i], free, occ, unk):
        occupied_cells.append(cells[i])
        i += 1
    if len(occupied_cells) < int(cfg.min_occupied_run_cells):
        return reject(candidate_id, "free_occupied_unknown", start, "occupied_run_too_short")

    while i < len(cells) and is_unknown(cells[i], free, occ, unk):
        unknown_cells.append(cells[i])
        i += 1
    if len(unknown_cells) < int(cfg.min_unknown_run_cells):
        return reject(candidate_id, "free_occupied_unknown", start, "unknown_tail_missing")
    if i != len(cells):
        return reject(candidate_id, "free_occupied_unknown", start, "non_unknown_after_unknown_tail")

    return DoorPatternCandidate(
        candidate_id=int(candidate_id),
        pattern_type="free_occupied_unknown",
        start_rc=start,
        free_cells=free_cells,
        occupied_cells=occupied_cells,
        unknown_cells=unknown_cells,
        turn_rc=None,
        door_cut_cells=select_cut_cells(free_cells, cfg, free.shape),
        accepted=True,
    )


def try_match_rule_a_free_turn_occupied(
    *,
    candidate_id: int,
    start: tuple[int, int],
    free: np.ndarray,
    occ: np.ndarray,
    unk: np.ndarray,
    roi: tuple[int, int, int, int],
    cfg: DoorPatternConfig,
) -> DoorPatternCandidate:
    main_cells = trace_line(start, int(cfg.scan_dr), int(cfg.scan_dc), roi)
    free_cells: list[tuple[int, int]] = []
    for rc in main_cells:
        if is_unknown(rc, free, occ, unk):
            return reject(candidate_id, "free_turn_occupied", start, "unknown_in_lower_free_run")
        if is_strict_free(rc, free, occ, unk):
            free_cells.append(rc)
            continue
        break
    if len(free_cells) < int(cfg.min_free_run_cells):
        return reject(candidate_id, "free_turn_occupied", start, "free_run_too_short")

    turn_base = free_cells[-1]
    turn_dirs = list(cfg.turn_dirs)
    if bool(cfg.allow_diagonal_turns):
        turn_dirs.extend([(-1, -1), (-1, 1)])
    for dr, dc in turn_dirs:
        turn_path = trace_line(
            turn_base,
            int(dr),
            int(dc),
            roi,
            skip_start=True,
            max_cells=cfg.rule_a_turn_probe_max_cells,
        )
        occupied_cells: list[tuple[int, int]] = []
        failed = False
        for rc in turn_path:
            if is_strict_occupied(rc, free, occ, unk):
                occupied_cells.append(rc)
                continue
            failed = True
            break
        if failed or len(occupied_cells) < int(cfg.min_occupied_run_cells):
            continue
        return DoorPatternCandidate(
            candidate_id=int(candidate_id),
            pattern_type="free_turn_occupied",
            start_rc=start,
            free_cells=free_cells,
            occupied_cells=occupied_cells,
            unknown_cells=[],
            turn_rc=turn_base,
            door_cut_cells=select_cut_cells(free_cells, cfg, free.shape),
            accepted=True,
        )
    return reject(candidate_id, "free_turn_occupied", start, "no_valid_turn_with_all_occupied_after")


def trace_line(
    start: tuple[int, int],
    dr: int,
    dc: int,
    roi: tuple[int, int, int, int],
    *,
    skip_start: bool = False,
    max_cells: int | None = None,
) -> list[tuple[int, int]]:
    min_r, min_c, max_r, max_c = roi
    r, c = int(start[0]), int(start[1])
    if skip_start:
        r += int(dr)
        c += int(dc)
    out: list[tuple[int, int]] = []
    while int(min_r) <= r <= int(max_r) and int(min_c) <= c <= int(max_c):
        out.append((int(r), int(c)))
        if max_cells is not None and len(out) >= int(max_cells):
            break
        r += int(dr)
        c += int(dc)
    return out


def cell_state(rc: tuple[int, int], free: np.ndarray, occ: np.ndarray, unk: np.ndarray) -> str:
    r, c = int(rc[0]), int(rc[1])
    if bool(unk[r, c]):
        return "U"
    if bool(occ[r, c]):
        return "O"
    if bool(free[r, c]):
        return "F"
    return "U"


def is_unknown(rc: tuple[int, int], free: np.ndarray, occ: np.ndarray, unk: np.ndarray) -> bool:
    return cell_state(rc, free, occ, unk) == "U"


def is_strict_free(rc: tuple[int, int], free: np.ndarray, occ: np.ndarray, unk: np.ndarray) -> bool:
    r, c = int(rc[0]), int(rc[1])
    return bool(free[r, c]) and not bool(occ[r, c]) and not bool(unk[r, c])


def is_strict_occupied(rc: tuple[int, int], free: np.ndarray, occ: np.ndarray, unk: np.ndarray) -> bool:
    r, c = int(rc[0]), int(rc[1])
    return bool(occ[r, c]) and not bool(free[r, c]) and not bool(unk[r, c])


def select_cut_cells(
    free_cells: Sequence[tuple[int, int]],
    cfg: DoorPatternConfig,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    if not free_cells:
        return []
    thickness = max(1, int(cfg.door_cut_thickness_cells))
    lateral = max(0, int(cfg.door_cut_lateral_radius_cells))
    base_cells = list(free_cells)[-thickness:]
    lat_dirs = _lateral_dirs(int(cfg.scan_dr), int(cfg.scan_dc))
    out: list[tuple[int, int]] = []
    for r, c in base_cells:
        out.append((int(r), int(c)))
        for ldr, ldc in lat_dirs:
            for dist in range(1, lateral + 1):
                out.append((int(r) + int(ldr) * dist, int(c) + int(ldc) * dist))
    return unique_in_bounds(out, shape)


def _lateral_dirs(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if int(dr) == 0 and int(dc) == 0:
        return (0, -1), (0, 1)
    return (-int(dc), int(dr)), (int(dc), -int(dr))


def unique_in_bounds(cells: Sequence[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = int(shape[0]), int(shape[1])
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for r, c in cells:
        rc = (int(r), int(c))
        if rc in seen:
            continue
        if 0 <= rc[0] < h and 0 <= rc[1] < w:
            seen.add(rc)
            out.append(rc)
    return out


def mark_candidate(
    cand: DoorPatternCandidate,
    detected_door_mask: np.ndarray,
    door_cut_mask: np.ndarray,
    pattern_type_map: np.ndarray,
    pattern_value: int,
    free: np.ndarray,
) -> None:
    for rc in [*cand.free_cells, *cand.occupied_cells, *cand.unknown_cells]:
        detected_door_mask[rc] = True
        pattern_type_map[rc] = np.uint8(pattern_value)
    for rc in cand.door_cut_cells:
        if bool(free[rc]):
            door_cut_mask[rc] = True


def merge_nearby_door_cuts(door_cut_mask: np.ndarray, cfg: DoorPatternConfig) -> np.ndarray:
    cut = np.asarray(door_cut_mask, dtype=bool)
    radius = int(cfg.merge_nearby_door_cells)
    if radius <= 0 or not np.any(cut):
        return cut
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndimage.binary_dilation(cut, structure=structure).astype(bool)


def reject(candidate_id: int, pattern_type: str, start: tuple[int, int], reason: str) -> DoorPatternCandidate:
    return DoorPatternCandidate(
        candidate_id=int(candidate_id),
        pattern_type=str(pattern_type),
        start_rc=(int(start[0]), int(start[1])),
        accepted=False,
        reject_reason=str(reason),
    )


def detect_partial_door_seed_mask(
    *,
    free: np.ndarray,
    occ: np.ndarray,
    unk: np.ndarray,
    rois: Sequence[tuple[int, int, int, int]],
    cfg: DoorPatternConfig,
) -> tuple[np.ndarray, np.ndarray]:
    seed = np.zeros_like(free, dtype=bool)
    seed_type = np.zeros(free.shape, dtype=np.uint8)
    for roi in rois:
        min_r, min_c, max_r, max_c = roi
        _ = min_r
        for col in range(int(min_c), int(max_c) + 1):
            start = resolve_bottom_start(
                free=free,
                occ=occ,
                unk=unk,
                bottom_r=int(max_r),
                col=col,
                roi=roi,
                cfg=cfg,
            )
            if start is None:
                continue
            cells = trace_line(start, int(cfg.scan_dr), int(cfg.scan_dc), roi)
            free_cells: list[tuple[int, int]] = []
            unknown_below = 0
            idx = 0
            while idx < len(cells):
                rc = cells[idx]
                if is_strict_free(rc, free, occ, unk):
                    free_cells.append(rc)
                    idx += 1
                    continue
                if is_unknown(rc, free, occ, unk):
                    unknown_below += 1
                break
            if len(free_cells) < int(cfg.partial_min_free_run_cells):
                continue
            if unknown_below > int(cfg.partial_max_unknown_below_cells):
                continue
            occupied_cells: list[tuple[int, int]] = []
            while idx < len(cells) and is_strict_occupied(cells[idx], free, occ, unk):
                occupied_cells.append(cells[idx])
                idx += 1
            if len(occupied_cells) >= int(cfg.partial_min_occupied_seed_cells):
                for rc in occupied_cells:
                    seed[rc] = True
                    seed_type[rc] = np.uint8(1)

            turn_base = free_cells[-1]
            turn_dirs = list(cfg.turn_dirs)
            if bool(cfg.allow_diagonal_turns):
                turn_dirs.extend([(-1, -1), (-1, 1)])
            for dr, dc in turn_dirs:
                turn_cells = trace_line(
                    turn_base,
                    int(dr),
                    int(dc),
                    roi,
                    skip_start=True,
                    max_cells=cfg.partial_turn_probe_max_cells,
                )
                turn_occupied: list[tuple[int, int]] = []
                for rc in turn_cells:
                    if is_strict_occupied(rc, free, occ, unk):
                        turn_occupied.append(rc)
                        continue
                    break
                if len(turn_occupied) >= int(cfg.partial_min_occupied_seed_cells):
                    for rc in turn_occupied:
                        seed[rc] = True
                        seed_type[rc] = np.uint8(2)
    return seed.astype(bool), seed_type.astype(np.uint8)


def fit_and_extend_partial_door_lines(
    *,
    seed_mask: np.ndarray,
    free: np.ndarray,
    occ: np.ndarray,
    unk: np.ndarray,
    strict_door_detected: np.ndarray,
    strict_door_cut: np.ndarray,
    cfg: DoorPatternConfig,
    resolution_m: float,
) -> list[DoorLineExtensionCandidate]:
    seed = np.asarray(seed_mask, dtype=bool)
    if not np.any(seed):
        return []
    labels, count = ndimage.label(seed, structure=_conn(int(cfg.partial_seed_connectivity)))
    accepted_line_mask = np.zeros_like(seed, dtype=bool)
    out: list[DoorLineExtensionCandidate] = []
    next_id = 1
    for comp_id in range(1, int(count) + 1):
        comp_mask = labels == comp_id
        cells_arr = np.argwhere(comp_mask)
        size = int(cells_arr.shape[0])
        if size < int(cfg.partial_seed_min_component_cells):
            continue
        if size > int(cfg.partial_seed_max_component_cells):
            continue
        fit = _fit_line(cells_arr, cfg)
        if fit is None:
            cells = [(int(r), int(c)) for r, c in cells_arr.tolist()]
            out.append(
                DoorLineExtensionCandidate(
                    candidate_id=next_id,
                    seed_component_id=comp_id,
                    seed_cells=cells,
                    fitted_direction_rc=(0.0, 1.0),
                    fitted_center_rc=(float(np.mean(cells_arr[:, 0])), float(np.mean(cells_arr[:, 1]))),
                    accepted=False,
                    reject_reason="line_fit_failed",
                )
            )
            next_id += 1
            continue
        center, direction, inlier_cells = fit
        all_other = (np.asarray(strict_door_detected, dtype=bool) | seed | np.asarray(strict_door_cut, dtype=bool) | accepted_line_mask) & ~comp_mask
        candidate = _extend_line_candidate(
            candidate_id=next_id,
            seed_component_id=comp_id,
            seed_cells=[(int(r), int(c)) for r, c in inlier_cells.tolist()],
            center=center,
            direction=direction,
            comp_mask=comp_mask,
            free=free,
            occ=occ,
            unk=unk,
            other_door_mask=all_other,
            cfg=cfg,
            resolution_m=float(resolution_m),
        )
        if candidate.accepted:
            for r, c in candidate.extension_cells:
                if 0 <= int(r) < seed.shape[0] and 0 <= int(c) < seed.shape[1]:
                    accepted_line_mask[int(r), int(c)] = True
        out.append(candidate)
        next_id += 1
    return out


def _fit_line(cells: np.ndarray, cfg: DoorPatternConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pts = np.asarray(cells, dtype=np.float32)
    if pts.shape[0] < max(2, int(cfg.line_fit_min_inliers)):
        return None
    center = np.mean(pts, axis=0)
    pts0 = pts - center
    try:
        _u, _s, vh = np.linalg.svd(pts0, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = np.asarray(vh[0], dtype=np.float32)
    if float(np.linalg.norm(direction)) <= 1e-6:
        return None
    direction = direction / float(np.linalg.norm(direction))
    residual = _perpendicular_distances(pts, center, direction)
    inliers = residual <= float(cfg.line_fit_max_residual_cells)
    if int(np.count_nonzero(inliers)) < int(cfg.line_fit_min_inliers):
        return None
    pts_in = pts[inliers]
    center = np.mean(pts_in, axis=0)
    pts0 = pts_in - center
    try:
        _u, _s, vh = np.linalg.svd(pts0, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = np.asarray(vh[0], dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    proj = np.dot(pts_in - center, direction)
    if float(np.max(proj) - np.min(proj)) + 1.0 < float(cfg.line_fit_min_length_cells):
        return None
    return center.astype(np.float32), direction.astype(np.float32), pts_in.astype(np.int32)


def _extend_line_candidate(
    *,
    candidate_id: int,
    seed_component_id: int,
    seed_cells: list[tuple[int, int]],
    center: np.ndarray,
    direction: np.ndarray,
    comp_mask: np.ndarray,
    free: np.ndarray,
    occ: np.ndarray,
    unk: np.ndarray,
    other_door_mask: np.ndarray,
    cfg: DoorPatternConfig,
    resolution_m: float,
) -> DoorLineExtensionCandidate:
    wall_anchor = np.asarray(occ, dtype=bool) & ~np.asarray(comp_mask, dtype=bool)
    seed_pts = np.asarray(seed_cells, dtype=np.float32)
    projections = np.dot(seed_pts - center[None, :], direction)
    t_min = int(np.floor(float(np.min(projections)))) - 1
    t_max = int(np.ceil(float(np.max(projections)))) + 1
    neg = _walk_line_side(center, direction, t_min, -1, free.shape, cfg)
    pos = _walk_line_side(center, direction, t_max, 1, free.shape, cfg)
    endpoint_a, anchor_a, cells_a = _first_endpoint(
        neg,
        wall_anchor=wall_anchor,
        other_door_mask=other_door_mask,
        free=free,
        unk=unk,
        cfg=cfg,
    )
    endpoint_b, anchor_b, cells_b = _first_endpoint(
        pos,
        wall_anchor=wall_anchor,
        other_door_mask=other_door_mask,
        free=free,
        unk=unk,
        cfg=cfg,
    )
    line_before = _sample_line_cells(center, direction, t_min, t_max, free.shape)
    extension_cells = unique_in_bounds([*cells_a, *line_before, *cells_b], free.shape)
    candidate = DoorLineExtensionCandidate(
        candidate_id=int(candidate_id),
        seed_component_id=int(seed_component_id),
        seed_cells=list(seed_cells),
        fitted_direction_rc=(float(direction[0]), float(direction[1])),
        fitted_center_rc=(float(center[0]), float(center[1])),
        line_cells_before_extension=line_before,
        extension_cells=extension_cells,
        wall_anchor_a=anchor_a,
        wall_anchor_b=anchor_b,
        endpoint_a_kind=endpoint_a,
        endpoint_b_kind=endpoint_b,
        accepted=False,
        reject_reason=None,
    )
    if endpoint_a == "other_door" or endpoint_b == "other_door":
        return _reject_line(candidate, "extension_endpoint_is_other_door")
    if endpoint_a != "wall" or endpoint_b != "wall" or anchor_a is None or anchor_b is None:
        return _reject_line(candidate, "door_line_does_not_hit_two_walls")
    full_line = unique_in_bounds(_bresenham(anchor_a, anchor_b), free.shape)
    inner_line = _trim_anchor_margin(full_line, max(1, int(cfg.door_line_wall_anchor_radius_cells)))
    if bool(cfg.reject_if_extension_hits_other_door) and _intersects_dilated(inner_line, other_door_mask, int(cfg.other_door_intersection_radius_cells)):
        candidate.extension_cells = full_line
        return _reject_line(candidate, "extension_intersects_other_door")
    width_m = float(len(full_line)) * float(resolution_m)
    candidate.debug["width_m"] = float(width_m)
    if width_m < float(cfg.door_line_min_width_m) or width_m > float(cfg.door_line_max_width_m):
        candidate.extension_cells = full_line
        return _reject_line(candidate, "door_line_width_out_of_range")
    if not inner_line:
        candidate.extension_cells = full_line
        return _reject_line(candidate, "door_line_too_short_after_anchor_margin")
    rr = np.asarray([r for r, _c in inner_line], dtype=np.int32)
    cc = np.asarray([c for _r, c in inner_line], dtype=np.int32)
    unknown_ratio = float(np.mean(np.asarray(unk, dtype=bool)[rr, cc])) if rr.size else 1.0
    wall_ratio = float(np.mean(np.asarray(occ, dtype=bool)[rr, cc] & ~np.asarray(comp_mask, dtype=bool)[rr, cc])) if rr.size else 1.0
    free_or_seed_ratio = float(np.mean(np.asarray(free, dtype=bool)[rr, cc] | np.asarray(comp_mask, dtype=bool)[rr, cc])) if rr.size else 0.0
    candidate.debug.update(
        {
            "inner_unknown_ratio": float(unknown_ratio),
            "inner_wall_ratio": float(wall_ratio),
            "inner_free_or_seed_ratio": float(free_or_seed_ratio),
        }
    )
    if unknown_ratio > float(cfg.door_line_inner_unknown_ratio_max):
        candidate.extension_cells = full_line
        return _reject_line(candidate, "door_line_crosses_too_much_unknown")
    if wall_ratio > float(cfg.door_line_inner_wall_ratio_max):
        candidate.extension_cells = full_line
        return _reject_line(candidate, "door_line_crosses_wall_midspan")
    if free_or_seed_ratio < float(cfg.door_line_min_free_or_seed_ratio):
        candidate.extension_cells = full_line
        return _reject_line(candidate, "door_line_not_supported_by_free_or_seed")
    door_cut = [rc for rc in inner_line if bool(np.asarray(free, dtype=bool)[rc])]
    if not door_cut:
        candidate.extension_cells = full_line
        return _reject_line(candidate, "door_cut_has_no_free_cells")
    candidate.extension_cells = full_line
    candidate.door_cut_cells = door_cut
    candidate.accepted = True
    candidate.reject_reason = None
    return candidate


def _walk_line_side(
    center: np.ndarray,
    direction: np.ndarray,
    start_t: int,
    step_sign: int,
    shape: tuple[int, int],
    cfg: DoorPatternConfig,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    t = int(start_t)
    for _ in range(max(1, int(cfg.door_line_extend_max_cells))):
        rc_float = center + direction * float(t)
        rc = (int(round(float(rc_float[0]))), int(round(float(rc_float[1]))))
        if not (0 <= rc[0] < shape[0] and 0 <= rc[1] < shape[1]):
            break
        if rc not in seen:
            out.append(rc)
            seen.add(rc)
        t += int(step_sign)
    return out


def _first_endpoint(
    cells: Sequence[tuple[int, int]],
    *,
    wall_anchor: np.ndarray,
    other_door_mask: np.ndarray,
    free: np.ndarray,
    unk: np.ndarray,
    cfg: DoorPatternConfig,
) -> tuple[str, tuple[int, int] | None, list[tuple[int, int]]]:
    traversed: list[tuple[int, int]] = []
    for rc in cells:
        traversed.append(rc)
        if _near_mask(rc, other_door_mask, int(cfg.other_door_intersection_radius_cells)):
            return "other_door", None, traversed
        if _near_mask(rc, wall_anchor, int(cfg.door_line_wall_anchor_radius_cells)):
            anchor = _nearest_mask_cell(rc, wall_anchor, int(cfg.door_line_wall_anchor_radius_cells))
            return "wall", anchor, traversed
        if bool(np.asarray(unk, dtype=bool)[rc]):
            return "unknown", None, traversed
    if traversed:
        last = traversed[-1]
        if bool(np.asarray(free, dtype=bool)[last]):
            return "free_end", None, traversed
    return "out_of_roi", None, traversed


def _sample_line_cells(
    center: np.ndarray,
    direction: np.ndarray,
    t_min: int,
    t_max: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for t in range(int(t_min), int(t_max) + 1):
        rc_float = center + direction * float(t)
        rc = (int(round(float(rc_float[0]))), int(round(float(rc_float[1]))))
        if 0 <= rc[0] < shape[0] and 0 <= rc[1] < shape[1] and (not out or out[-1] != rc):
            out.append(rc)
    return out


def _trim_anchor_margin(cells: Sequence[tuple[int, int]], margin: int) -> list[tuple[int, int]]:
    if len(cells) <= 2 * int(margin):
        return []
    return list(cells)[int(margin) : len(cells) - int(margin)]


def _reject_line(candidate: DoorLineExtensionCandidate, reason: str) -> DoorLineExtensionCandidate:
    candidate.accepted = False
    candidate.reject_reason = str(reason)
    return candidate


def _perpendicular_distances(points: np.ndarray, center: np.ndarray, direction: np.ndarray) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float32) - np.asarray(center, dtype=np.float32)[None, :]
    proj = np.dot(delta, direction)
    closest = np.asarray(center, dtype=np.float32)[None, :] + proj[:, None] * np.asarray(direction, dtype=np.float32)[None, :]
    return np.linalg.norm(np.asarray(points, dtype=np.float32) - closest, axis=1)


def _near_mask(rc: tuple[int, int], mask: np.ndarray, radius: int) -> bool:
    r, c = int(rc[0]), int(rc[1])
    arr = np.asarray(mask, dtype=bool)
    rr0, rr1 = max(0, r - int(radius)), min(arr.shape[0], r + int(radius) + 1)
    cc0, cc1 = max(0, c - int(radius)), min(arr.shape[1], c + int(radius) + 1)
    return bool(np.any(arr[rr0:rr1, cc0:cc1]))


def _nearest_mask_cell(rc: tuple[int, int], mask: np.ndarray, radius: int) -> tuple[int, int] | None:
    r, c = int(rc[0]), int(rc[1])
    arr = np.asarray(mask, dtype=bool)
    best: tuple[float, tuple[int, int]] | None = None
    for rr in range(max(0, r - int(radius)), min(arr.shape[0], r + int(radius) + 1)):
        for cc in range(max(0, c - int(radius)), min(arr.shape[1], c + int(radius) + 1)):
            if not bool(arr[rr, cc]):
                continue
            dist = float((rr - r) ** 2 + (cc - c) ** 2)
            if best is None or dist < best[0]:
                best = (dist, (int(rr), int(cc)))
    return None if best is None else best[1]


def _intersects_dilated(cells: Sequence[tuple[int, int]], mask: np.ndarray, radius: int) -> bool:
    if not cells:
        return False
    arr = np.asarray(mask, dtype=bool)
    if int(radius) > 0 and np.any(arr):
        arr = ndimage.binary_dilation(arr, structure=np.ones((2 * int(radius) + 1, 2 * int(radius) + 1), dtype=bool))
    for rc in cells:
        if 0 <= int(rc[0]) < arr.shape[0] and 0 <= int(rc[1]) < arr.shape[1] and bool(arr[int(rc[0]), int(rc[1])]):
            return True
    return False


def _bresenham(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = int(a[0]), int(a[1])
    r1, c1 = int(b[0]), int(b[1])
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    out: list[tuple[int, int]] = []
    r, c = r0, c0
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


def _conn(connectivity: int) -> np.ndarray:
    return np.ones((3, 3), dtype=bool) if int(connectivity) == 8 else np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _component_count(mask: np.ndarray, connectivity: int) -> int:
    _labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=_conn(connectivity))
    return int(count)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
