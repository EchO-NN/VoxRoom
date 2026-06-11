from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from voxroom_online.isaac_runtime.mapping.height_column_profile import (
    HEIGHT_STATE_CONFLICT,
    HEIGHT_STATE_FREE,
    HEIGHT_STATE_OCCUPIED,
    HEIGHT_STATE_UNKNOWN,
    HeightColumnClassification,
)
from voxroom_online.isaac_runtime.mapping.online_roomseg.utils import conn, dilate, rasterize_line


@dataclass
class HeightProfileDoorConfig:
    enabled: bool = True
    z_start_m: float = 0.10
    top_occupied_min_z_m: float = 1.80
    min_lower_free_bins: int = 3
    min_top_occupied_bins: int = 3
    lower_unknown_max_bins: int = 0
    lower_occupied_max_bins: int = 0
    reject_unknown_before_top_occupied: bool = True
    reject_conflict_before_top_occupied: bool = True
    reject_non_unknown_after_top_occupied_tail: bool = True
    allow_profile_end_as_unknown_tail: bool = True
    allow_end_of_profile_as_unknown_tail: bool = True
    min_unknown_tail_bins: int = 0
    reject_free_after_top_occupied: bool = True
    reject_occupied_after_unknown_tail: bool = True
    seed_connectivity: int = 8
    min_seed_component_cells: int = 1
    max_seed_component_cells: int = 120
    min_component_line_length_m: float = 0.10
    max_component_thickness_m: float = 0.30
    line_fit_max_residual_cells: float = 1.25
    single_seed_infer_from_wall_enabled: bool = True
    single_seed_wall_search_radius_m: float = 1.20
    door_width_min_m: float = 0.35
    door_width_max_m: float = 1.60
    extend_max_m: float = 1.80
    wall_anchor_radius_cells: int = 1
    wall_anchor_min_cells: int = 1
    inner_unknown_ratio_max: float = 0.10
    inner_wall_ratio_max: float = 0.10
    inner_free_or_seed_ratio_min: float = 0.50
    cut_thickness_cells: int = 1
    conflict_dilation_cells: int = 1
    reject_if_intersects_other_door: bool = True
    reject_if_endpoint_is_other_door: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "HeightProfileDoorConfig":
        if isinstance(data, cls):
            return data
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class DoorSeedCellEvidence:
    row: int
    col: int
    first_occupied_z_m: float | None
    lower_free_bins: int
    lower_unknown_bins: int
    lower_occupied_bins: int
    top_occupied_bins: int
    unknown_tail_bins: int
    conflict_bins: int
    accepted: bool
    reject_reason: str | None

    def to_dict(self) -> dict:
        return {
            "row": int(self.row),
            "col": int(self.col),
            "first_occupied_z_m": None if self.first_occupied_z_m is None else float(self.first_occupied_z_m),
            "lower_free_bins": int(self.lower_free_bins),
            "lower_unknown_bins": int(self.lower_unknown_bins),
            "lower_occupied_bins": int(self.lower_occupied_bins),
            "top_occupied_bins": int(self.top_occupied_bins),
            "unknown_tail_bins": int(self.unknown_tail_bins),
            "conflict_bins": int(self.conflict_bins),
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
        }


@dataclass
class HeightProfileDoorLineCandidate:
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
        return {
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


@dataclass
class HeightProfileDoorDetectionResult:
    door_seed_mask: np.ndarray
    door_seed_component_map: np.ndarray
    door_centerline_candidate_mask: np.ndarray
    accepted_door_centerline_mask: np.ndarray
    rejected_door_centerline_mask: np.ndarray
    door_cut_mask: np.ndarray
    door_seed_evidence_map: np.ndarray
    candidates: list[HeightProfileDoorLineCandidate]
    debug: dict[str, object]


def detect_height_profile_doors(
    *,
    classification: HeightColumnClassification,
    free_clean: np.ndarray,
    wall_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    config: HeightProfileDoorConfig | Mapping[str, object] | None = None,
) -> HeightProfileDoorDetectionResult:
    cfg = config if isinstance(config, HeightProfileDoorConfig) else HeightProfileDoorConfig.from_mapping(config)
    free = np.asarray(free_clean, dtype=bool)
    wall = np.asarray(wall_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    shape = free.shape
    seed = np.zeros(shape, dtype=bool)
    evidence_reason = np.zeros(shape, dtype=np.uint8)
    seed_evidence: list[DoorSeedCellEvidence] = []
    rejected_seed_reasons: Counter[str] = Counter()
    if not bool(cfg.enabled):
        return _empty_result(shape, enabled=False)

    z_centers = _z_centers_from_classification(classification)
    z_state = np.asarray(classification.z_state, dtype=np.uint8)
    active_z = np.asarray(classification.active_z_bin_mask, dtype=bool)
    if z_state.ndim != 3 or z_state.shape[1:] != shape:
        raise ValueError("classification z_state must have shape (Z,H,W) matching clean masks")
    if active_z.shape != z_state.shape:
        raise ValueError("classification active_z_bin_mask must match z_state")

    for r in range(shape[0]):
        for c in range(shape[1]):
            ev = classify_door_seed_at_xy(z_state[:, r, c], z_centers, cfg, row=r, col=c, active_z_bin_mask=active_z[:, r, c])
            if ev.accepted:
                seed[r, c] = True
                evidence_reason[r, c] = 1
                seed_evidence.append(ev)
            else:
                rejected_seed_reasons[str(ev.reject_reason)] += 1
                evidence_reason[r, c] = _seed_reject_code(str(ev.reject_reason))

    labels, count = ndimage.label(seed, structure=conn(int(cfg.seed_connectivity)))
    candidates: list[HeightProfileDoorLineCandidate] = []
    cid = 1
    for comp_id in range(1, int(count) + 1):
        comp = labels == int(comp_id)
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(comp))]
        if len(cells) < int(cfg.min_seed_component_cells):
            candidates.append(_rejected_candidate(cid, comp_id, cells, "seed_component_too_small"))
            cid += 1
            continue
        if len(cells) > int(cfg.max_seed_component_cells):
            candidates.append(_rejected_candidate(cid, comp_id, cells, "seed_component_too_large"))
            cid += 1
            continue
        candidate = _candidate_from_seed_component(
            candidate_id=cid,
            component_id=int(comp_id),
            seed_cells=cells,
            seed_component_mask=comp,
            all_seed_mask=seed,
            free_clean=free,
            wall_clean=wall,
            unknown_clean=unknown,
            resolution_m=float(resolution_m),
            cfg=cfg,
        )
        candidates.append(candidate)
        cid += 1

    _batch_reject_conflicting_doors(candidates, seed, shape, cfg)
    candidate_mask = np.zeros(shape, dtype=bool)
    accepted_mask = np.zeros(shape, dtype=bool)
    rejected_mask = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        cells = candidate.door_cut_cells or candidate.extended_centerline_cells or candidate.seed_projected_centerline_cells
        mask = _cells_to_mask(cells, shape)
        if int(cfg.cut_thickness_cells) > 1:
            mask = dilate(mask, int(cfg.cut_thickness_cells) - 1)
        candidate_mask |= mask
        if candidate.accepted:
            accepted_mask |= mask
        else:
            rejected_mask |= mask

    reason_counts = Counter(str(candidate.reject_reason) for candidate in candidates if not candidate.accepted)
    debug = {
        "height_profile_door_seed_mask": seed.astype(bool),
        "height_profile_door_seed_component_map": labels.astype(np.int32),
        "height_profile_door_centerline_candidate_mask": candidate_mask.astype(bool),
        "height_profile_accepted_door_centerline_mask": accepted_mask.astype(bool),
        "height_profile_rejected_door_centerline_mask": rejected_mask.astype(bool),
        "height_profile_door_cut_mask": accepted_mask.astype(bool),
        "height_profile_door_candidates": [candidate.to_dict() for candidate in candidates],
        "height_profile_door_seed_evidence": [item.to_dict() for item in seed_evidence[:2048]],
        "height_profile_door_seed_reject_reason_counts": dict(rejected_seed_reasons),
        **_seed_reject_reason_maps(evidence_reason),
        "height_profile_door_reject_reason_counts": dict(reason_counts),
        "height_profile_door_accepted_count": int(sum(1 for candidate in candidates if candidate.accepted)),
        "height_profile_door_rejected_count": int(sum(1 for candidate in candidates if not candidate.accepted)),
        "height_profile_door_seed_cells": int(np.count_nonzero(seed)),
    }
    return HeightProfileDoorDetectionResult(
        door_seed_mask=seed.astype(bool),
        door_seed_component_map=labels.astype(np.int32),
        door_centerline_candidate_mask=candidate_mask.astype(bool),
        accepted_door_centerline_mask=accepted_mask.astype(bool),
        rejected_door_centerline_mask=rejected_mask.astype(bool),
        door_cut_mask=accepted_mask.astype(bool),
        door_seed_evidence_map=evidence_reason.astype(np.uint8),
        candidates=candidates,
        debug=debug,
    )


def classify_door_seed_at_xy(
    z_state_col: np.ndarray,
    z_centers_m: np.ndarray,
    cfg: HeightProfileDoorConfig,
    *,
    row: int = 0,
    col: int = 0,
    active_z_bin_mask: np.ndarray | None = None,
) -> DoorSeedCellEvidence:
    states = np.asarray(z_state_col, dtype=np.uint8).reshape(-1)
    centers = np.asarray(z_centers_m, dtype=np.float32).reshape(-1)
    active = np.ones(states.shape, dtype=bool) if active_z_bin_mask is None else np.asarray(active_z_bin_mask, dtype=bool).reshape(-1)
    idxs = np.flatnonzero(active & (centers >= float(cfg.z_start_m)))
    if idxs.size == 0:
        return _seed_ev(row, col, None, 0, 0, 0, 0, 0, False, "no_active_z_bins")
    pos = 0
    lower_free = 0
    lower_unknown = 0
    lower_occupied = 0
    conflict_bins = 0
    first_occ_z = None
    while pos < len(idxs):
        idx = int(idxs[pos])
        state = int(states[idx])
        if state == int(HEIGHT_STATE_FREE):
            lower_free += 1
            pos += 1
            continue
        if state == int(HEIGHT_STATE_UNKNOWN):
            lower_unknown += 1
            reason = "unknown_before_top_occupied" if bool(cfg.reject_unknown_before_top_occupied) else "no_top_occupied"
            return _seed_ev(row, col, None, lower_free, lower_unknown, lower_occupied, 0, 0, False, reason, conflict_bins)
        if state == int(HEIGHT_STATE_CONFLICT):
            conflict_bins += 1
            reason = "conflict_before_top_occupied" if bool(cfg.reject_conflict_before_top_occupied) else "no_top_occupied"
            return _seed_ev(row, col, None, lower_free, lower_unknown, lower_occupied, 0, 0, False, reason, conflict_bins)
        if state == int(HEIGHT_STATE_OCCUPIED):
            first_occ_z = float(centers[idx])
            break
        pos += 1
    if first_occ_z is None:
        return _seed_ev(row, col, None, lower_free, lower_unknown, lower_occupied, 0, 0, False, "no_top_occupied", conflict_bins)
    if lower_free < int(cfg.min_lower_free_bins):
        return _seed_ev(row, col, first_occ_z, lower_free, lower_unknown, lower_occupied, 0, 0, False, "lower_free_bins_too_few", conflict_bins)
    if lower_occupied > int(cfg.lower_occupied_max_bins):
        return _seed_ev(row, col, first_occ_z, lower_free, lower_unknown, lower_occupied, 0, 0, False, "occupied_below_top", conflict_bins)
    if first_occ_z < float(cfg.top_occupied_min_z_m):
        return _seed_ev(row, col, first_occ_z, lower_free, lower_unknown, lower_occupied, 0, 0, False, "top_occupied_too_low", conflict_bins)

    top_occ = 0
    while pos < len(idxs) and int(states[int(idxs[pos])]) == int(HEIGHT_STATE_OCCUPIED):
        top_occ += 1
        pos += 1
    if top_occ < int(cfg.min_top_occupied_bins):
        return _seed_ev(row, col, first_occ_z, lower_free, lower_unknown, lower_occupied, top_occ, 0, False, "top_occupied_run_too_short", conflict_bins)

    unknown_tail = 0
    while pos < len(idxs) and int(states[int(idxs[pos])]) == int(HEIGHT_STATE_UNKNOWN):
        unknown_tail += 1
        pos += 1
    allow_end = bool(cfg.allow_profile_end_as_unknown_tail) or bool(cfg.allow_end_of_profile_as_unknown_tail)
    if pos >= len(idxs) and allow_end and unknown_tail >= int(cfg.min_unknown_tail_bins):
        return _seed_ev(row, col, first_occ_z, lower_free, lower_unknown, lower_occupied, top_occ, unknown_tail, True, None, conflict_bins)
    if unknown_tail < int(cfg.min_unknown_tail_bins):
        return _seed_ev(row, col, first_occ_z, lower_free, lower_unknown, lower_occupied, top_occ, unknown_tail, False, "unknown_tail_missing", conflict_bins)
    while pos < len(idxs):
        state = int(states[int(idxs[pos])])
        if state == int(HEIGHT_STATE_CONFLICT):
            return _seed_ev(row, col, first_occ_z, lower_free, lower_unknown, lower_occupied, top_occ, unknown_tail, False, "conflict_after_top_occupied_tail", conflict_bins + 1)
        if state != int(HEIGHT_STATE_UNKNOWN) and bool(cfg.reject_non_unknown_after_top_occupied_tail):
            return _seed_ev(row, col, first_occ_z, lower_free, lower_unknown, lower_occupied, top_occ, unknown_tail, False, "non_unknown_after_unknown_tail", conflict_bins)
        if state == int(HEIGHT_STATE_FREE) and bool(cfg.reject_free_after_top_occupied):
            return _seed_ev(row, col, first_occ_z, lower_free, lower_unknown, lower_occupied, top_occ, unknown_tail, False, "free_after_top_occupied", conflict_bins)
        if state == int(HEIGHT_STATE_OCCUPIED) and bool(cfg.reject_occupied_after_unknown_tail):
            return _seed_ev(row, col, first_occ_z, lower_free, lower_unknown, lower_occupied, top_occ, unknown_tail, False, "occupied_after_unknown_tail", conflict_bins)
        pos += 1
    return _seed_ev(row, col, first_occ_z, lower_free, lower_unknown, lower_occupied, top_occ, unknown_tail, True, None, conflict_bins)


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
    cfg: HeightProfileDoorConfig,
) -> HeightProfileDoorLineCandidate:
    pts = np.asarray(seed_cells, dtype=np.float32)
    if len(seed_cells) == 1:
        inferred = _infer_single_seed_direction(seed_cells[0], wall_clean, resolution_m, cfg)
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
        minor = np.asarray([-major[1], major[0]], dtype=np.float32)
        residual = np.abs(np.dot(centered, minor))
        residual_max = float(np.max(residual)) if residual.size else 0.0
        if residual_max > float(cfg.line_fit_max_residual_cells):
            return _rejected_candidate(candidate_id, component_id, seed_cells, "door_seed_not_line_like", center, major)
    minor = np.asarray([-major[1], major[0]], dtype=np.float32)
    seed_centerline = _project_seed_component_to_centerline(pts, center, major, minor, free_clean.shape)
    if not seed_centerline:
        return _rejected_candidate(candidate_id, component_id, seed_cells, "door_centerline_projection_empty", center, major)
    length_m = _cell_line_span_m(seed_centerline, resolution_m)
    thickness_m = _component_thickness_m(pts, center, minor, resolution_m)
    if thickness_m > float(cfg.max_component_thickness_m):
        return _rejected_candidate(candidate_id, component_id, seed_cells, "door_seed_component_too_thick", center, major)

    ordered = _sort_cells_along_direction(seed_centerline, center, major)
    start = np.asarray(ordered[0], dtype=np.float32)
    end = np.asarray(ordered[-1], dtype=np.float32)
    anchor_a, extension_a, reason_a = _walk_to_wall(
        start_rc=start,
        direction=-major,
        wall_clean=wall_clean,
        unknown_clean=unknown_clean,
        other_door_mask=all_seed_mask & ~seed_component_mask,
        resolution_m=resolution_m,
        cfg=cfg,
    )
    anchor_b, extension_b, reason_b = _walk_to_wall(
        start_rc=end,
        direction=major,
        wall_clean=wall_clean,
        unknown_clean=unknown_clean,
        other_door_mask=all_seed_mask & ~seed_component_mask,
        resolution_m=resolution_m,
        cfg=cfg,
    )
    if anchor_a is None or anchor_b is None:
        candidate = _rejected_candidate(
            candidate_id,
            component_id,
            seed_cells,
            "door_line_does_not_hit_two_walls",
            center,
            major,
        )
        candidate.seed_projected_centerline_cells = seed_centerline
        candidate.extended_centerline_cells = [*list(reversed(extension_a)), *ordered, *extension_b]
        candidate.debug.update({"anchor_a_reject_reason": reason_a, "anchor_b_reject_reason": reason_b})
        return candidate

    full_cells = _line_cells(anchor_a, anchor_b, free_clean.shape)
    width_m = float(len(full_cells) * resolution_m)
    if width_m < float(cfg.door_width_min_m) or width_m > float(cfg.door_width_max_m):
        return HeightProfileDoorLineCandidate(
            candidate_id=candidate_id,
            seed_component_id=component_id,
            seed_cells=seed_cells,
            center_rc=(float(center[0]), float(center[1])),
            major_dir_rc=(float(major[0]), float(major[1])),
            minor_dir_rc=(float(minor[0]), float(minor[1])),
            seed_projected_centerline_cells=seed_centerline,
            extended_centerline_cells=full_cells,
            door_cut_cells=full_cells,
            wall_anchor_a=anchor_a,
            wall_anchor_b=anchor_b,
            width_m=width_m,
            accepted=False,
            reject_reason="door_width_out_of_range",
            debug={"width_m": width_m, "component_length_m": float(length_m), "residual_max_cells": float(residual_max)},
        )

    inner = full_cells[1:-1] if len(full_cells) > 2 else full_cells
    own_seed = np.zeros_like(free_clean, dtype=bool)
    for r, c in seed_cells:
        own_seed[int(r), int(c)] = True
    inner_unknown_ratio = _ratio(inner, unknown_clean)
    inner_wall_ratio = _ratio(inner, wall_clean & ~own_seed)
    inner_free_or_seed_ratio = _ratio(inner, free_clean | own_seed)
    reject_reason = None
    if inner_unknown_ratio > float(cfg.inner_unknown_ratio_max):
        reject_reason = "door_inner_unknown_ratio_too_high"
    elif inner_wall_ratio > float(cfg.inner_wall_ratio_max):
        reject_reason = "door_inner_wall_ratio_too_high"
    elif inner_free_or_seed_ratio < float(cfg.inner_free_or_seed_ratio_min):
        reject_reason = "door_inner_free_or_seed_ratio_too_low"

    score = float(
        0.35 * min(1.0, len(seed_cells) / 6.0)
        + 0.25 * min(1.0, width_m / 0.6)
        + 0.25 * (1.0 - min(1.0, inner_unknown_ratio))
        + 0.15 * min(1.0, inner_free_or_seed_ratio)
    )
    return HeightProfileDoorLineCandidate(
        candidate_id=candidate_id,
        seed_component_id=component_id,
        seed_cells=seed_cells,
        center_rc=(float(center[0]), float(center[1])),
        major_dir_rc=(float(major[0]), float(major[1])),
        minor_dir_rc=(float(minor[0]), float(minor[1])),
        seed_projected_centerline_cells=seed_centerline,
        extended_centerline_cells=full_cells,
        door_cut_cells=full_cells,
        wall_anchor_a=anchor_a,
        wall_anchor_b=anchor_b,
        width_m=width_m,
        accepted=reject_reason is None,
        reject_reason=reject_reason,
        debug={
            "score": float(score),
            "component_length_m": float(length_m),
            "component_thickness_m": float(thickness_m),
            "residual_max_cells": float(residual_max),
            "inner_unknown_ratio": float(inner_unknown_ratio),
            "inner_wall_ratio": float(inner_wall_ratio),
            "inner_free_or_seed_ratio": float(inner_free_or_seed_ratio),
        },
    )


def _batch_reject_conflicting_doors(
    candidates: list[HeightProfileDoorLineCandidate],
    all_seed_mask: np.ndarray,
    shape: tuple[int, int],
    cfg: HeightProfileDoorConfig,
) -> None:
    provisional = [item for item in candidates if item.accepted]
    if not provisional:
        return
    line_masks = {item.candidate_id: _cells_to_mask(item.door_cut_cells, shape) for item in provisional}
    endpoint_masks = {}
    for item in provisional:
        mask = np.zeros(shape, dtype=bool)
        for endpoint in (item.wall_anchor_a, item.wall_anchor_b):
            if endpoint is not None:
                r, c = int(endpoint[0]), int(endpoint[1])
                if 0 <= r < shape[0] and 0 <= c < shape[1]:
                    mask[r, c] = True
        endpoint_masks[item.candidate_id] = dilate(mask, int(cfg.conflict_dilation_cells))

    for item in provisional:
        own_seed = _cells_to_mask(item.seed_cells, shape)
        other_seed = all_seed_mask & ~own_seed
        if bool(cfg.reject_if_intersects_other_door) and np.any(line_masks[item.candidate_id] & other_seed):
            item.accepted = False
            item.reject_reason = "extension_intersects_other_door_candidate"
        if bool(cfg.reject_if_endpoint_is_other_door) and np.any(endpoint_masks[item.candidate_id] & other_seed):
            item.accepted = False
            item.reject_reason = "extension_endpoint_is_other_door"

    provisional = [item for item in candidates if item.accepted]
    for idx, a in enumerate(provisional):
        for b in provisional[idx + 1 :]:
            if not (a.accepted and b.accepted):
                continue
            if not np.any(line_masks[a.candidate_id] & line_masks[b.candidate_id]):
                continue
            score_a = float(a.debug.get("score", 0.0))
            score_b = float(b.debug.get("score", 0.0))
            if abs(score_a - score_b) < 0.05:
                a.accepted = False
                b.accepted = False
                a.reject_reason = "extension_intersects_other_door_candidate"
                b.reject_reason = "extension_intersects_other_door_candidate"
            elif score_a < score_b:
                a.accepted = False
                a.reject_reason = "extension_intersects_other_door_candidate"
            else:
                b.accepted = False
                b.reject_reason = "extension_intersects_other_door_candidate"


def _walk_to_wall(
    *,
    start_rc: np.ndarray,
    direction: np.ndarray,
    wall_clean: np.ndarray,
    unknown_clean: np.ndarray,
    other_door_mask: np.ndarray,
    resolution_m: float,
    cfg: HeightProfileDoorConfig,
) -> tuple[tuple[int, int] | None, list[tuple[int, int]], str | None]:
    max_steps = max(1, int(round(float(cfg.extend_max_m) / max(float(resolution_m), 1e-9))))
    direction = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return None, [], "zero_direction"
    direction = direction / norm
    cells: list[tuple[int, int]] = []
    for step in range(1, max_steps + 1):
        rc = np.rint(np.asarray(start_rc, dtype=np.float32) + direction * float(step)).astype(np.int32)
        r, c = int(rc[0]), int(rc[1])
        if not (0 <= r < wall_clean.shape[0] and 0 <= c < wall_clean.shape[1]):
            return None, cells, "out_of_bounds"
        cells.append((r, c))
        if bool(other_door_mask[r, c]):
            return None, cells, "extension_hits_door_instead_of_wall"
        if bool(unknown_clean[r, c]):
            return None, cells, "extension_hit_unknown"
        if _wall_anchor_at((r, c), wall_clean, int(cfg.wall_anchor_radius_cells), int(cfg.wall_anchor_min_cells)):
            return (r, c), cells, None
    return None, cells, "extend_max_without_wall"


def _infer_single_seed_direction(
    seed_rc: tuple[int, int],
    wall_clean: np.ndarray,
    resolution_m: float,
    cfg: HeightProfileDoorConfig,
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
            if float(cfg.door_width_min_m) <= width_m <= float(cfg.door_width_max_m):
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
    mask = rasterize_line(p0, p1, shape)
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(mask))]


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


def _sort_cells_along_direction(cells: list[tuple[int, int]], center: np.ndarray, major: np.ndarray) -> list[tuple[int, int]]:
    return sorted(cells, key=lambda cell: float(np.dot(np.asarray(cell, dtype=np.float32) - center, major)))


def _cell_line_span_m(cells: list[tuple[int, int]], resolution_m: float) -> float:
    if not cells:
        return 0.0
    rows = [int(r) for r, _c in cells]
    cols = [int(c) for _r, c in cells]
    return float((max(max(rows) - min(rows), max(cols) - min(cols)) + 1) * resolution_m)


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
    lower_unknown: int,
    lower_occupied: int,
    top_occ: int,
    unknown_tail: int,
    accepted: bool,
    reason: str | None,
    conflict_bins: int = 0,
) -> DoorSeedCellEvidence:
    return DoorSeedCellEvidence(
        row=int(row),
        col=int(col),
        first_occupied_z_m=first_occ_z,
        lower_free_bins=int(lower_free),
        lower_unknown_bins=int(lower_unknown),
        lower_occupied_bins=int(lower_occupied),
        top_occupied_bins=int(top_occ),
        unknown_tail_bins=int(unknown_tail),
        conflict_bins=int(conflict_bins),
        accepted=bool(accepted),
        reject_reason=reason,
    )


_SEED_REJECT_CODES = {
    "lower_free_bins_too_few": 2,
    "top_occupied_too_low": 3,
    "unknown_before_top_occupied": 4,
    "no_top_occupied": 5,
    "top_occupied_run_too_short": 6,
    "conflict_before_top_occupied": 7,
}


def _seed_reject_code(reason: str) -> int:
    return int(_SEED_REJECT_CODES.get(str(reason), 255))


def _seed_reject_reason_maps(reason_map: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "height_profile_door_reject_reason_map": np.asarray(reason_map, dtype=np.uint8),
        "height_profile_door_lower_free_bins_too_few_cells": np.asarray(reason_map == _SEED_REJECT_CODES["lower_free_bins_too_few"], dtype=bool),
        "height_profile_door_top_occupied_too_low_cells": np.asarray(reason_map == _SEED_REJECT_CODES["top_occupied_too_low"], dtype=bool),
        "height_profile_door_unknown_before_top_occupied_cells": np.asarray(reason_map == _SEED_REJECT_CODES["unknown_before_top_occupied"], dtype=bool),
        "height_profile_door_no_top_occupied_cells": np.asarray(reason_map == _SEED_REJECT_CODES["no_top_occupied"], dtype=bool),
        "height_profile_door_top_occupied_run_too_short_cells": np.asarray(reason_map == _SEED_REJECT_CODES["top_occupied_run_too_short"], dtype=bool),
        "height_profile_door_conflict_before_top_occupied_cells": np.asarray(reason_map == _SEED_REJECT_CODES["conflict_before_top_occupied"], dtype=bool),
    }


def _rejected_candidate(
    candidate_id: int,
    component_id: int,
    seed_cells: list[tuple[int, int]],
    reason: str,
    center: np.ndarray | None = None,
    major: np.ndarray | None = None,
) -> HeightProfileDoorLineCandidate:
    if center is None:
        center_arr = np.mean(np.asarray(seed_cells or [(0, 0)], dtype=np.float32), axis=0)
    else:
        center_arr = np.asarray(center, dtype=np.float32)
    if major is None:
        major_arr = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        major_arr = np.asarray(major, dtype=np.float32)
    norm = float(np.linalg.norm(major_arr))
    if norm <= 1e-6:
        major_arr = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        major_arr = major_arr / norm
    minor = np.asarray([-major_arr[1], major_arr[0]], dtype=np.float32)
    return HeightProfileDoorLineCandidate(
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


def _z_centers_from_classification(classification: HeightColumnClassification) -> np.ndarray:
    debug = dict(classification.debug or {})
    z_min = float(debug.get("height_profile_z_min_m", 0.10))
    z_size = float(debug.get("height_profile_z_bin_size_m", 0.05))
    count = int(classification.z_state.shape[0])
    return (z_min + (np.arange(count, dtype=np.float32) + 0.5) * z_size).astype(np.float32)


def _empty_result(shape: tuple[int, int], *, enabled: bool) -> HeightProfileDoorDetectionResult:
    zero = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    debug = {
        "height_profile_door_enabled": bool(enabled),
        "height_profile_door_accepted_count": 0,
        "height_profile_door_rejected_count": 0,
        "height_profile_door_candidates": [],
    }
    return HeightProfileDoorDetectionResult(
        door_seed_mask=zero.copy(),
        door_seed_component_map=labels,
        door_centerline_candidate_mask=zero.copy(),
        accepted_door_centerline_mask=zero.copy(),
        rejected_door_centerline_mask=zero.copy(),
        door_cut_mask=zero.copy(),
        door_seed_evidence_map=np.zeros(shape, dtype=np.uint8),
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
