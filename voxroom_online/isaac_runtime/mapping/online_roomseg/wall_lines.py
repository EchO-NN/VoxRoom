from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from .utils import component_metrics, dilate, label_components, rasterize_line


@dataclass
class LineWallsConfig:
    enabled: bool = True
    hough_enabled: bool = True
    pca_enabled: bool = True
    min_line_length_m: float = 0.45
    min_support_ratio: float = 0.35
    max_angle_to_dominant_deg: float = 15.0
    axis_snap_lateral_tolerance_cells: int = 2
    axis_snap_min_projected_support_ratio: float = 0.35

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "LineWallsConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class WallSegment:
    segment_id: int
    p0_rc: np.ndarray
    p1_rc: np.ndarray
    theta: float
    length_m: float
    support_ratio: float
    mean_wall_score: float
    source: str

    def to_dict(self) -> dict:
        return {
            "segment_id": int(self.segment_id),
            "p0_rc": [int(round(float(v))) for v in self.p0_rc.tolist()],
            "p1_rc": [int(round(float(v))) for v in self.p1_rc.tolist()],
            "theta": float(self.theta),
            "length_m": float(self.length_m),
            "support_ratio": float(self.support_ratio),
            "mean_wall_score": float(self.mean_wall_score),
            "source": str(self.source),
        }


@dataclass
class LineFilteringConfig:
    enabled: bool = True
    dominant_angle_bin_deg: float = 5.0
    max_snap_angle_deg: float = 12.0
    merge_collinear_gap_m: float = 0.35
    merge_lateral_offset_m: float = 0.12
    min_filtered_line_length_m: float = 0.60
    min_filtered_support_ratio: float = 0.45
    endpoint_refine_window_m: float = 0.25
    endpoint_min_wall_support_m: float = 0.20
    max_wall_thickness_m: float = 0.30
    min_confidence: float = 0.45

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "LineFilteringConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class FilteredWallLine:
    line_id: int
    p0_rc: np.ndarray
    p1_rc: np.ndarray
    theta: float
    normal_theta: float
    length_m: float
    support_ratio: float
    mean_wall_score: float
    thickness_m: float
    source_segment_ids: list[int]
    endpoint_quality: dict = field(default_factory=dict)
    confidence: float = 0.0
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "line_id": int(self.line_id),
            "p0_rc": [float(v) for v in np.asarray(self.p0_rc, dtype=np.float32).tolist()],
            "p1_rc": [float(v) for v in np.asarray(self.p1_rc, dtype=np.float32).tolist()],
            "theta": float(self.theta),
            "normal_theta": float(self.normal_theta),
            "length_m": float(self.length_m),
            "support_ratio": float(self.support_ratio),
            "mean_wall_score": float(self.mean_wall_score),
            "thickness_m": float(self.thickness_m),
            "source_segment_ids": [int(v) for v in self.source_segment_ids],
            "endpoint_quality": _jsonable(self.endpoint_quality),
            "confidence": float(self.confidence),
            "debug": _jsonable(self.debug),
        }


@dataclass
class WallRunSnapConfig:
    enabled: bool = True
    max_angle_to_axis_deg: float = 25.0
    support_band_cells: int = 2
    min_run_length_m: float = 0.35
    min_support_ratio: float = 0.25
    close_holes_m: float = 0.20

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "WallRunSnapConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class WallRunMergeConfig:
    enabled: bool = True
    max_lateral_offset_cells: int = 3
    max_merge_gap_m: float = 0.30
    min_overlap_m: float = 0.05

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "WallRunMergeConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class SnappedWallRun:
    run_id: int
    axis: str
    line: int
    start: int
    end: int
    support_cells: np.ndarray
    source_segment_ids: list[int]
    support_ratio: float
    length_m: float
    lateral_std_cells: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "run_id": int(self.run_id),
            "axis": str(self.axis),
            "line": int(self.line),
            "start": int(self.start),
            "end": int(self.end),
            "support_cell_count": int(len(self.support_cells)),
            "source_segment_ids": [int(v) for v in self.source_segment_ids],
            "support_ratio": float(self.support_ratio),
            "length_m": float(self.length_m),
            "lateral_std_cells": float(self.lateral_std_cells),
            "confidence": float(self.confidence),
        }


def extract_line_supported_walls(
    wall_candidate_clean: np.ndarray,
    *,
    resolution_m: float,
    config: LineWallsConfig | Mapping[str, object] | None = None,
) -> tuple[list[WallSegment], dict]:
    cfg = config if isinstance(config, LineWallsConfig) else LineWallsConfig.from_mapping(config)
    wall = np.asarray(wall_candidate_clean, dtype=bool)
    if not bool(cfg.enabled) or not np.any(wall):
        return [], {"enabled": bool(cfg.enabled), "segments": [], "dominant_directions": []}
    segments: list[WallSegment] = []
    next_id = 1
    if bool(cfg.hough_enabled):
        hough = _axis_run_segments(wall, float(resolution_m), cfg, next_id)
        segments.extend(hough)
        next_id += len(hough)
    if bool(cfg.pca_enabled):
        pca = _pca_component_segments(wall, float(resolution_m), cfg, next_id)
        segments.extend(pca)
    segments = _dedupe_segments(segments)
    dominant = _dominant_directions(segments)
    debug = {
        "enabled": True,
        "hough_enabled": bool(cfg.hough_enabled),
        "pca_enabled": bool(cfg.pca_enabled),
        "segment_count": int(len(segments)),
        "segments": [segment.to_dict() for segment in segments[:512]],
        "dominant_directions": dominant,
    }
    return segments, debug


def filter_and_snap_wall_lines(
    raw_segments: Sequence[WallSegment],
    *,
    wall_candidate_clean: np.ndarray,
    free_clean: np.ndarray,
    resolution_m: float,
    config: LineFilteringConfig | Mapping[str, object] | None = None,
) -> tuple[list[FilteredWallLine], dict]:
    cfg = config if isinstance(config, LineFilteringConfig) else LineFilteringConfig.from_mapping(config)
    wall = np.asarray(wall_candidate_clean, dtype=bool)
    free = np.asarray(free_clean, dtype=bool)
    if wall.shape != free.shape:
        raise ValueError("wall_candidate_clean and free_clean must have the same HxW shape")
    if not bool(cfg.enabled):
        lines = [_filtered_line_from_segment(idx, segment, float(resolution_m), "raw_disabled") for idx, segment in enumerate(raw_segments, start=1)]
        return lines, {"enabled": False, "filtered_wall_line_count": int(len(lines))}
    if not raw_segments or not np.any(wall):
        return [], {"enabled": True, "filtered_wall_line_count": 0, "dominant_directions": [], "rejected": {"empty": int(len(raw_segments))}}

    directions = estimate_dominant_wall_directions(
        raw_segments,
        dominant_angle_bin_deg=float(cfg.dominant_angle_bin_deg),
    )
    snap_cfg = WallRunSnapConfig(
        enabled=True,
        max_angle_to_axis_deg=float(cfg.max_snap_angle_deg),
        support_band_cells=max(1, int(round(float(cfg.max_wall_thickness_m) / max(float(resolution_m), 1e-9)))),
        min_run_length_m=float(cfg.min_filtered_line_length_m),
        min_support_ratio=float(cfg.min_filtered_support_ratio),
        close_holes_m=float(cfg.merge_collinear_gap_m),
    )
    runs, snap_debug = snap_wall_segments_to_runs(
        raw_segments,
        wall,
        resolution_m=float(resolution_m),
        max_angle_to_axis_deg=float(snap_cfg.max_angle_to_axis_deg),
        support_band_cells=int(snap_cfg.support_band_cells),
        min_run_length_m=float(snap_cfg.min_run_length_m),
        min_support_ratio=float(snap_cfg.min_support_ratio),
        close_holes_m=float(snap_cfg.close_holes_m),
    )
    merge_cfg = WallRunMergeConfig(
        enabled=True,
        max_lateral_offset_cells=max(0, int(round(float(cfg.merge_lateral_offset_m) / max(float(resolution_m), 1e-9)))),
        max_merge_gap_m=float(cfg.merge_collinear_gap_m),
        min_overlap_m=0.0,
    )
    merged_runs, merge_debug = _merge_snapped_wall_runs_without_free_gap(
        runs,
        free,
        resolution_m=float(resolution_m),
        max_lateral_offset_cells=int(merge_cfg.max_lateral_offset_cells),
        max_merge_gap_m=float(merge_cfg.max_merge_gap_m),
        min_overlap_m=float(merge_cfg.min_overlap_m),
        max_free_gap_ratio=0.30,
    )
    rejected: dict[str, int] = {}
    lines: list[FilteredWallLine] = []
    for run in merged_runs:
        line = _filtered_line_from_run(
            len(lines) + 1,
            run,
            wall,
            free,
            resolution_m=float(resolution_m),
            config=cfg,
            dominant_directions=directions,
        )
        if line.length_m < float(cfg.min_filtered_line_length_m):
            rejected["short"] = rejected.get("short", 0) + 1
            continue
        if line.support_ratio < float(cfg.min_filtered_support_ratio):
            rejected["low_support_ratio"] = rejected.get("low_support_ratio", 0) + 1
            continue
        if line.thickness_m > float(cfg.max_wall_thickness_m) + 1e-6:
            rejected["thick_component"] = rejected.get("thick_component", 0) + 1
            continue
        if min(float(line.endpoint_quality.get("p0_support_m", 0.0)), float(line.endpoint_quality.get("p1_support_m", 0.0))) < float(cfg.endpoint_min_wall_support_m):
            rejected["weak_endpoint_support"] = rejected.get("weak_endpoint_support", 0) + 1
            continue
        if line.confidence < float(cfg.min_confidence):
            rejected["low_confidence"] = rejected.get("low_confidence", 0) + 1
            continue
        line.line_id = len(lines) + 1
        lines.append(line)
    debug = {
        "enabled": True,
        "raw_segment_count": int(len(raw_segments)),
        "dominant_directions": directions,
        "snapped_wall_run_count": int(len(runs)),
        "merged_wall_run_count": int(len(merged_runs)),
        "filtered_wall_line_count": int(len(lines)),
        "rejected": rejected,
        "snap_debug": snap_debug,
        "merge_debug": merge_debug,
        "filtered_wall_lines": [line.to_dict() for line in lines[:512]],
    }
    return lines, debug


def estimate_dominant_wall_directions(
    segments: Sequence[WallSegment],
    *,
    dominant_angle_bin_deg: float = 5.0,
) -> list[dict]:
    if not segments:
        return []
    bin_rad = max(np.deg2rad(float(dominant_angle_bin_deg)), 1e-3)
    buckets: dict[int, float] = {}
    for segment in segments:
        theta = float(segment.theta) % float(np.pi)
        key = int(round(theta / bin_rad))
        buckets[key] = buckets.get(key, 0.0) + float(segment.length_m) * max(0.05, float(segment.support_ratio))
    ranked = sorted(buckets.items(), key=lambda item: -item[1])
    total = max(1e-9, sum(buckets.values()))
    out: list[dict] = []
    for key, support in ranked[:4]:
        theta = (float(key) * bin_rad) % float(np.pi)
        out.append({"theta": float(theta), "support_m": float(support), "confidence": float(support / total)})
    return out


def snap_wall_segments_to_runs(
    segments: Sequence[WallSegment],
    wall_candidate_clean: np.ndarray,
    *,
    resolution_m: float,
    max_angle_to_axis_deg: float = 25.0,
    support_band_cells: int = 2,
    min_run_length_m: float = 0.35,
    min_support_ratio: float = 0.25,
    close_holes_m: float = 0.20,
) -> tuple[list[SnappedWallRun], dict]:
    wall = np.asarray(wall_candidate_clean, dtype=bool)
    if wall.size == 0 or not np.any(wall):
        return [], {"enabled": True, "snapped_wall_run_count": 0, "rejected": {"empty_wall_map": int(len(segments))}}
    max_angle = np.deg2rad(float(max_angle_to_axis_deg))
    min_cells = max(2, int(round(float(min_run_length_m) / max(float(resolution_m), 1e-9))))
    close_cells = max(0, int(round(float(close_holes_m) / max(float(resolution_m), 1e-9))))
    band = max(0, int(support_band_cells))
    runs: list[SnappedWallRun] = []
    rejected: dict[str, int] = {}
    for segment in segments:
        axis, angle_delta = _nearest_axis(float(segment.theta))
        if float(angle_delta) > float(max_angle):
            rejected["angle"] = rejected.get("angle", 0) + 1
            continue
        raw = _support_profile_for_segment(segment, wall, axis=axis, support_band_cells=band)
        if raw is None:
            rejected["no_support"] = rejected.get("no_support", 0) + 1
            continue
        major_values, support_cells = raw
        if major_values.size == 0:
            rejected["no_support"] = rejected.get("no_support", 0) + 1
            continue
        lo, hi = int(np.min(major_values)), int(np.max(major_values))
        if hi < lo:
            rejected["empty_span"] = rejected.get("empty_span", 0) + 1
            continue
        profile = np.zeros(hi - lo + 1, dtype=bool)
        profile[np.asarray(major_values, dtype=np.int32) - lo] = True
        if close_cells > 0 and profile.size > 1:
            profile = ndimage.binary_closing(profile, structure=np.ones(close_cells + 1, dtype=bool), border_value=1).astype(bool)
        run_indices = np.flatnonzero(profile)
        if run_indices.size == 0:
            rejected["closed_empty"] = rejected.get("closed_empty", 0) + 1
            continue
        start = int(lo + int(run_indices.min()))
        end = int(lo + int(run_indices.max()))
        span_cells = int(end - start + 1)
        if span_cells < min_cells:
            rejected["short"] = rejected.get("short", 0) + 1
            continue
        in_span = (major_values >= start) & (major_values <= end)
        cells = np.asarray(support_cells[in_span], dtype=np.int32)
        if cells.size == 0:
            rejected["no_support_in_span"] = rejected.get("no_support_in_span", 0) + 1
            continue
        support_positions = np.unique(major_values[in_span])
        support_ratio = float(len(support_positions)) / float(max(1, span_cells))
        if support_ratio < float(min_support_ratio):
            rejected["low_support_ratio"] = rejected.get("low_support_ratio", 0) + 1
            continue
        if axis == "horizontal":
            lateral = cells[:, 0]
        else:
            lateral = cells[:, 1]
        line = int(round(float(np.median(lateral))))
        lateral_std = float(np.std(lateral.astype(np.float32))) if lateral.size else 0.0
        confidence = float(np.clip(0.65 * support_ratio + 0.35 * float(segment.support_ratio), 0.0, 1.0))
        runs.append(
            SnappedWallRun(
                run_id=len(runs) + 1,
                axis=axis,
                line=line,
                start=start,
                end=end,
                support_cells=cells,
                source_segment_ids=[int(segment.segment_id)],
                support_ratio=support_ratio,
                length_m=float(span_cells * float(resolution_m)),
                lateral_std_cells=lateral_std,
                confidence=confidence,
            )
        )
    debug = {
        "enabled": True,
        "input_segment_count": int(len(segments)),
        "snapped_wall_run_count": int(len(runs)),
        "rejected": rejected,
        "runs": [run.to_dict() for run in runs[:512]],
    }
    return runs, debug


def merge_snapped_wall_runs(
    runs: Sequence[SnappedWallRun],
    *,
    resolution_m: float,
    max_lateral_offset_cells: int = 3,
    max_merge_gap_m: float = 0.30,
    min_overlap_m: float = 0.05,
) -> tuple[list[SnappedWallRun], dict]:
    pending = sorted(list(runs), key=lambda item: (str(item.axis), int(item.line), int(item.start), int(item.end)))
    if not pending:
        return [], {"enabled": True, "merged_wall_run_count": 0, "merge_count": 0, "runs": []}
    max_gap_cells = max(0, int(round(float(max_merge_gap_m) / max(float(resolution_m), 1e-9))))
    min_overlap_cells = max(0, int(round(float(min_overlap_m) / max(float(resolution_m), 1e-9))))
    max_offset = max(0, int(max_lateral_offset_cells))
    groups: list[list[SnappedWallRun]] = []
    for run in pending:
        placed = False
        for group in groups:
            if any(_runs_compatible(run, other, max_offset=max_offset, max_gap=max_gap_cells, min_overlap=min_overlap_cells) for other in group):
                group.append(run)
                placed = True
                break
        if not placed:
            groups.append([run])
    merged = [_merge_group(idx, group, float(resolution_m)) for idx, group in enumerate(groups, start=1)]
    debug = {
        "enabled": True,
        "input_run_count": int(len(runs)),
        "merged_wall_run_count": int(len(merged)),
        "merge_count": int(sum(max(0, len(group) - 1) for group in groups)),
        "runs": [run.to_dict() for run in merged[:512]],
    }
    return merged, debug


def _merge_snapped_wall_runs_without_free_gap(
    runs: Sequence[SnappedWallRun],
    free_clean: np.ndarray,
    *,
    resolution_m: float,
    max_lateral_offset_cells: int,
    max_merge_gap_m: float,
    min_overlap_m: float,
    max_free_gap_ratio: float,
) -> tuple[list[SnappedWallRun], dict]:
    pending = sorted(list(runs), key=lambda item: (str(item.axis), int(item.line), int(item.start), int(item.end)))
    if not pending:
        return [], {"enabled": True, "merged_wall_run_count": 0, "merge_count": 0, "runs": [], "free_gap_guard": True}
    max_gap_cells = max(0, int(round(float(max_merge_gap_m) / max(float(resolution_m), 1e-9))))
    min_overlap_cells = max(0, int(round(float(min_overlap_m) / max(float(resolution_m), 1e-9))))
    max_offset = max(0, int(max_lateral_offset_cells))
    groups: list[list[SnappedWallRun]] = []
    rejected_free_gap = 0
    for run in pending:
        placed = False
        for group in groups:
            compatible = False
            for other in group:
                if not _runs_compatible(run, other, max_offset=max_offset, max_gap=max_gap_cells, min_overlap=min_overlap_cells):
                    continue
                if _run_gap_free_ratio(run, other, free_clean) > float(max_free_gap_ratio):
                    rejected_free_gap += 1
                    continue
                compatible = True
                break
            if compatible:
                group.append(run)
                placed = True
                break
        if not placed:
            groups.append([run])
    merged = [_merge_group(idx, group, float(resolution_m)) for idx, group in enumerate(groups, start=1)]
    return merged, {
        "enabled": True,
        "input_run_count": int(len(runs)),
        "merged_wall_run_count": int(len(merged)),
        "merge_count": int(sum(max(0, len(group) - 1) for group in groups)),
        "rejected_free_gap_merge_count": int(rejected_free_gap),
        "free_gap_guard": True,
        "runs": [run.to_dict() for run in merged[:512]],
    }


def line_supported_wall_mask(segments: Sequence[WallSegment], shape: tuple[int, int], radius_cells: int = 0) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for segment in segments:
        out |= rasterize_line(segment.p0_rc, segment.p1_rc, shape)
    if int(radius_cells) > 0:
        from .utils import dilate

        out = dilate(out, int(radius_cells))
    return out.astype(bool)


def snapped_wall_run_mask(runs: Sequence[SnappedWallRun], shape: tuple[int, int], radius_cells: int = 0) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for run in runs:
        if run.axis == "horizontal":
            p0 = np.asarray([run.line, run.start], dtype=np.float32)
            p1 = np.asarray([run.line, run.end], dtype=np.float32)
        else:
            p0 = np.asarray([run.start, run.line], dtype=np.float32)
            p1 = np.asarray([run.end, run.line], dtype=np.float32)
        out |= rasterize_line(p0, p1, shape)
    if int(radius_cells) > 0:
        out = dilate(out, int(radius_cells))
    return out.astype(bool)


def filtered_wall_line_mask(lines: Sequence[FilteredWallLine], shape: tuple[int, int], radius_cells: int = 0) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for line in lines:
        out |= rasterize_line(line.p0_rc, line.p1_rc, shape)
    if int(radius_cells) > 0:
        out = dilate(out, int(radius_cells))
    return out.astype(bool)


def _axis_run_segments(wall: np.ndarray, resolution: float, cfg: LineWallsConfig, start_id: int) -> list[WallSegment]:
    min_cells = max(2, int(round(float(cfg.min_line_length_m) / max(float(resolution), 1e-9))))
    lateral_tolerance = max(0, int(getattr(cfg, "axis_snap_lateral_tolerance_cells", 2)))
    min_projected_support = float(getattr(cfg, "axis_snap_min_projected_support_ratio", cfg.min_support_ratio))
    out: list[WallSegment] = []
    sid = int(start_id)
    h, w = wall.shape
    for row in range(h):
        r0, r1 = max(0, row - lateral_tolerance), min(h, row + lateral_tolerance + 1)
        band = wall[r0:r1, :]
        profile = np.any(band, axis=0)
        for c0, c1 in _runs(profile):
            if c1 - c0 < min_cells:
                continue
            rr, cc = np.nonzero(band[:, c0:c1])
            if rr.size == 0:
                continue
            actual_rows = rr + r0
            actual_cols = cc + c0
            snapped_row = int(round(float(np.median(actual_rows))))
            if int(row) != int(snapped_row):
                continue
            support_ratio = float(len(np.unique(actual_cols))) / float(max(1, c1 - c0))
            if support_ratio < min(float(cfg.min_support_ratio), min_projected_support):
                continue
            out.append(
                WallSegment(
                    segment_id=sid,
                    p0_rc=np.asarray([snapped_row, c0], dtype=np.float32),
                    p1_rc=np.asarray([snapped_row, c1 - 1], dtype=np.float32),
                    theta=0.0,
                    length_m=float((c1 - c0) * resolution),
                    support_ratio=support_ratio,
                    mean_wall_score=support_ratio,
                    source="hough_axis_run_lateral_snap" if lateral_tolerance > 0 else "hough_axis_run",
                )
            )
            sid += 1
    for col in range(w):
        c0, c1 = max(0, col - lateral_tolerance), min(w, col + lateral_tolerance + 1)
        band = wall[:, c0:c1]
        profile = np.any(band, axis=1)
        for rr0, rr1 in _runs(profile):
            if rr1 - rr0 < min_cells:
                continue
            rr, cc = np.nonzero(band[rr0:rr1, :])
            if rr.size == 0:
                continue
            actual_rows = rr + rr0
            actual_cols = cc + c0
            snapped_col = int(round(float(np.median(actual_cols))))
            if int(col) != int(snapped_col):
                continue
            support_ratio = float(len(np.unique(actual_rows))) / float(max(1, rr1 - rr0))
            if support_ratio < min(float(cfg.min_support_ratio), min_projected_support):
                continue
            out.append(
                WallSegment(
                    segment_id=sid,
                    p0_rc=np.asarray([rr0, snapped_col], dtype=np.float32),
                    p1_rc=np.asarray([rr1 - 1, snapped_col], dtype=np.float32),
                    theta=float(np.pi / 2.0),
                    length_m=float((rr1 - rr0) * resolution),
                    support_ratio=support_ratio,
                    mean_wall_score=support_ratio,
                    source="hough_axis_run_lateral_snap" if lateral_tolerance > 0 else "hough_axis_run",
                )
            )
            sid += 1
    return out


def _pca_component_segments(wall: np.ndarray, resolution: float, cfg: LineWallsConfig, start_id: int) -> list[WallSegment]:
    labels, count = label_components(wall, 8)
    out: list[WallSegment] = []
    sid = int(start_id)
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        metrics = component_metrics(comp, resolution)
        if metrics["length_m"] < float(cfg.min_line_length_m) or metrics["elongation"] < 1.5:
            continue
        rows, cols = np.nonzero(comp)
        coords = np.stack([rows.astype(np.float32), cols.astype(np.float32)], axis=1)
        center = np.mean(coords, axis=0)
        if coords.shape[0] < 2:
            continue
        cov = np.cov((coords - center).T)
        vals, vecs = np.linalg.eigh(cov)
        direction = vecs[:, int(np.argmax(vals))]
        projection = (coords - center) @ direction
        p0 = center + direction * float(np.min(projection))
        p1 = center + direction * float(np.max(projection))
        length_m = float(np.linalg.norm(p1 - p0) * resolution)
        if length_m < float(cfg.min_line_length_m):
            continue
        line = rasterize_line(p0, p1, wall.shape)
        support = float(np.count_nonzero(line & wall)) / float(max(1, np.count_nonzero(line)))
        if support < float(cfg.min_support_ratio):
            continue
        theta = float(np.arctan2(float(p1[0] - p0[0]), float(p1[1] - p0[1])))
        out.append(
            WallSegment(
                segment_id=sid,
                p0_rc=p0.astype(np.float32),
                p1_rc=p1.astype(np.float32),
                theta=theta,
                length_m=length_m,
                support_ratio=support,
                mean_wall_score=support,
                source="pca_component",
            )
        )
        sid += 1
    return out


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(values, dtype=bool)
    out: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(arr.tolist() + [False]):
        if value and start is None:
            start = int(idx)
        elif not value and start is not None:
            out.append((int(start), int(idx)))
            start = None
    return out


def _dedupe_segments(segments: Sequence[WallSegment]) -> list[WallSegment]:
    out: list[WallSegment] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for segment in segments:
        p0 = tuple(int(round(float(v))) for v in segment.p0_rc)
        p1 = tuple(int(round(float(v))) for v in segment.p1_rc)
        key = (*min(p0, p1), *max(p0, p1), str(segment.source))
        if key in seen:
            continue
        seen.add(key)
        out.append(segment)
    for idx, segment in enumerate(out, start=1):
        segment.segment_id = int(idx)
    return out


def _dominant_directions(segments: Sequence[WallSegment]) -> list[dict]:
    buckets = {"horizontal": 0.0, "vertical": 0.0, "other": 0.0}
    for segment in segments:
        theta = abs(float(segment.theta)) % float(np.pi)
        if theta < np.pi / 6 or theta > 5 * np.pi / 6:
            buckets["horizontal"] += float(segment.length_m)
        elif abs(theta - np.pi / 2) < np.pi / 6:
            buckets["vertical"] += float(segment.length_m)
        else:
            buckets["other"] += float(segment.length_m)
    total = max(1e-9, sum(buckets.values()))
    return [
        {"name": name, "support_m": float(value), "confidence": float(value / total)}
        for name, value in sorted(buckets.items(), key=lambda item: -item[1])
        if value > 0
    ]


def _nearest_axis(theta: float) -> tuple[str, float]:
    wrapped = abs(float(theta)) % float(np.pi)
    horizontal_delta = min(wrapped, abs(float(np.pi) - wrapped))
    vertical_delta = abs(wrapped - float(np.pi / 2.0))
    if horizontal_delta <= vertical_delta:
        return "horizontal", float(horizontal_delta)
    return "vertical", float(vertical_delta)


def _support_profile_for_segment(
    segment: WallSegment,
    wall: np.ndarray,
    *,
    axis: str,
    support_band_cells: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    p0 = np.asarray(segment.p0_rc, dtype=np.float32)
    p1 = np.asarray(segment.p1_rc, dtype=np.float32)
    h, w = wall.shape
    if axis == "horizontal":
        line = int(round(float(np.median([p0[0], p1[0]]))))
        start, end = sorted((int(np.floor(min(p0[1], p1[1]))), int(np.ceil(max(p0[1], p1[1])))))
        r0, r1 = max(0, line - int(support_band_cells)), min(h, line + int(support_band_cells) + 1)
        c0, c1 = max(0, start - 1), min(w, end + 2)
        rows, cols = np.nonzero(wall[r0:r1, c0:c1])
        if rows.size == 0:
            return None
        cells = np.stack([rows + r0, cols + c0], axis=1).astype(np.int32)
        return cells[:, 1].astype(np.int32), cells
    line = int(round(float(np.median([p0[1], p1[1]]))))
    start, end = sorted((int(np.floor(min(p0[0], p1[0]))), int(np.ceil(max(p0[0], p1[0])))))
    r0, r1 = max(0, start - 1), min(h, end + 2)
    c0, c1 = max(0, line - int(support_band_cells)), min(w, line + int(support_band_cells) + 1)
    rows, cols = np.nonzero(wall[r0:r1, c0:c1])
    if rows.size == 0:
        return None
    cells = np.stack([rows + r0, cols + c0], axis=1).astype(np.int32)
    return cells[:, 0].astype(np.int32), cells


def _runs_compatible(
    a: SnappedWallRun,
    b: SnappedWallRun,
    *,
    max_offset: int,
    max_gap: int,
    min_overlap: int,
) -> bool:
    if str(a.axis) != str(b.axis):
        return False
    if abs(int(a.line) - int(b.line)) > int(max_offset):
        return False
    overlap = min(int(a.end), int(b.end)) - max(int(a.start), int(b.start)) + 1
    gap = max(int(a.start), int(b.start)) - min(int(a.end), int(b.end)) - 1
    if overlap >= int(min_overlap):
        return True
    return gap <= int(max_gap)


def _run_gap_free_ratio(a: SnappedWallRun, b: SnappedWallRun, free_clean: np.ndarray) -> float:
    if str(a.axis) != str(b.axis):
        return 1.0
    left, right = (a, b) if int(a.start) <= int(b.start) else (b, a)
    gap_start = int(left.end) + 1
    gap_end = int(right.start) - 1
    if gap_end < gap_start:
        return 0.0
    free = np.asarray(free_clean, dtype=bool)
    if str(a.axis) == "horizontal":
        line = int(round((int(left.line) + int(right.line)) / 2.0))
        if line < 0 or line >= free.shape[0]:
            return 1.0
        values = free[line, max(0, gap_start) : min(free.shape[1], gap_end + 1)]
    else:
        line = int(round((int(left.line) + int(right.line)) / 2.0))
        if line < 0 or line >= free.shape[1]:
            return 1.0
        values = free[max(0, gap_start) : min(free.shape[0], gap_end + 1), line]
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(values)) / float(values.size)


def _merge_group(run_id: int, group: Sequence[SnappedWallRun], resolution_m: float) -> SnappedWallRun:
    cells = np.concatenate([np.asarray(run.support_cells, dtype=np.int32) for run in group], axis=0)
    axis = str(group[0].axis)
    if axis == "horizontal":
        line = int(round(float(np.median(cells[:, 0]))))
        start = int(min(int(run.start) for run in group))
        end = int(max(int(run.end) for run in group))
        lateral = cells[:, 0]
        major = cells[:, 1]
    else:
        line = int(round(float(np.median(cells[:, 1]))))
        start = int(min(int(run.start) for run in group))
        end = int(max(int(run.end) for run in group))
        lateral = cells[:, 1]
        major = cells[:, 0]
    span = max(1, int(end - start + 1))
    support_ratio = float(len(np.unique(major[(major >= start) & (major <= end)]))) / float(span)
    return SnappedWallRun(
        run_id=int(run_id),
        axis=axis,
        line=line,
        start=start,
        end=end,
        support_cells=cells,
        source_segment_ids=sorted({int(v) for run in group for v in run.source_segment_ids}),
        support_ratio=support_ratio,
        length_m=float(span * float(resolution_m)),
        lateral_std_cells=float(np.std(lateral.astype(np.float32))) if lateral.size else 0.0,
        confidence=float(np.clip(np.mean([float(run.confidence) for run in group]), 0.0, 1.0)),
    )


def _filtered_line_from_segment(line_id: int, segment: WallSegment, resolution_m: float, source: str) -> FilteredWallLine:
    p0 = np.asarray(segment.p0_rc, dtype=np.float32)
    p1 = np.asarray(segment.p1_rc, dtype=np.float32)
    theta = float(segment.theta)
    return FilteredWallLine(
        line_id=int(line_id),
        p0_rc=p0,
        p1_rc=p1,
        theta=theta,
        normal_theta=float(theta + np.pi / 2.0),
        length_m=float(np.linalg.norm(p1 - p0) * float(resolution_m)),
        support_ratio=float(segment.support_ratio),
        mean_wall_score=float(segment.mean_wall_score),
        thickness_m=float(resolution_m),
        source_segment_ids=[int(segment.segment_id)],
        endpoint_quality={"p0_support_m": float(resolution_m), "p1_support_m": float(resolution_m)},
        confidence=float(np.clip(segment.support_ratio, 0.0, 1.0)),
        debug={"source": source, "raw_segment": segment.to_dict()},
    )


def _filtered_line_from_run(
    line_id: int,
    run: SnappedWallRun,
    wall: np.ndarray,
    free: np.ndarray,
    *,
    resolution_m: float,
    config: LineFilteringConfig,
    dominant_directions: Sequence[Mapping[str, object]],
) -> FilteredWallLine:
    _ = dominant_directions
    if run.axis == "horizontal":
        p0 = np.asarray([run.line, run.start], dtype=np.float32)
        p1 = np.asarray([run.line, run.end], dtype=np.float32)
        theta = 0.0
        lateral = np.asarray(run.support_cells[:, 0], dtype=np.float32) if len(run.support_cells) else np.asarray([run.line], dtype=np.float32)
    else:
        p0 = np.asarray([run.start, run.line], dtype=np.float32)
        p1 = np.asarray([run.end, run.line], dtype=np.float32)
        theta = float(np.pi / 2.0)
        lateral = np.asarray(run.support_cells[:, 1], dtype=np.float32) if len(run.support_cells) else np.asarray([run.line], dtype=np.float32)
    p0, p1, endpoint_quality = _refine_run_endpoints(
        p0,
        p1,
        run.axis,
        wall,
        resolution_m=float(resolution_m),
        window_m=float(config.endpoint_refine_window_m),
    )
    line_mask = rasterize_line(p0, p1, wall.shape)
    line_cells = max(1, int(np.count_nonzero(line_mask)))
    support_ratio = float(np.count_nonzero(dilate(line_mask, 1) & wall)) / float(line_cells)
    free_overlap_ratio = float(np.count_nonzero(dilate(line_mask, 1) & free)) / float(max(1, int(np.count_nonzero(dilate(line_mask, 1)))))
    thickness_m = float(max(float(resolution_m), (2.0 * float(np.std(lateral)) + 1.0) * float(resolution_m)))
    confidence = float(
        np.clip(
            0.45 * float(run.confidence)
            + 0.35 * support_ratio
            + 0.10 * (1.0 - min(1.0, free_overlap_ratio))
            + 0.10 * min(1.0, float(run.length_m) / max(float(config.min_filtered_line_length_m), 1e-6)),
            0.0,
            1.0,
        )
    )
    return FilteredWallLine(
        line_id=int(line_id),
        p0_rc=p0.astype(np.float32),
        p1_rc=p1.astype(np.float32),
        theta=theta,
        normal_theta=float(theta + np.pi / 2.0),
        length_m=_inclusive_grid_line_length_m(p0, p1, float(resolution_m)),
        support_ratio=support_ratio,
        mean_wall_score=support_ratio,
        thickness_m=thickness_m,
        source_segment_ids=[int(v) for v in run.source_segment_ids],
        endpoint_quality=endpoint_quality,
        confidence=confidence,
        debug={
            "source": "snapped_merged_wall_run",
            "run": run.to_dict(),
            "free_overlap_ratio": float(free_overlap_ratio),
        },
    )


def _inclusive_grid_line_length_m(p0: np.ndarray, p1: np.ndarray, resolution_m: float) -> float:
    a = np.rint(np.asarray(p0, dtype=np.float32)).astype(np.int32)
    b = np.rint(np.asarray(p1, dtype=np.float32)).astype(np.int32)
    span_cells = int(max(abs(int(b[0]) - int(a[0])), abs(int(b[1]) - int(a[1]))) + 1)
    return float(max(1, span_cells) * float(resolution_m))


def _refine_run_endpoints(
    p0: np.ndarray,
    p1: np.ndarray,
    axis: str,
    wall: np.ndarray,
    *,
    resolution_m: float,
    window_m: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    p0_i = np.rint(p0).astype(np.int32)
    p1_i = np.rint(p1).astype(np.int32)
    if str(axis) == "horizontal":
        line = int(p0_i[0])
        start, end = sorted((int(p0_i[1]), int(p1_i[1])))
        values = np.nonzero(wall[max(0, line - 1) : min(wall.shape[0], line + 2), max(0, start) : min(wall.shape[1], end + 1)])[1]
        if values.size:
            start = max(0, start + int(values.min()))
            end = max(start, min(wall.shape[1] - 1, int(p0_i[1]) if end < start else int(start + values.max())))
        p0_new = np.asarray([line, start], dtype=np.float32)
        p1_new = np.asarray([line, end], dtype=np.float32)
    else:
        line = int(p0_i[1])
        start, end = sorted((int(p0_i[0]), int(p1_i[0])))
        values = np.nonzero(wall[max(0, start) : min(wall.shape[0], end + 1), max(0, line - 1) : min(wall.shape[1], line + 2)])[0]
        if values.size:
            start = max(0, start + int(values.min()))
            end = max(start, min(wall.shape[0] - 1, int(start + values.max())))
        p0_new = np.asarray([start, line], dtype=np.float32)
        p1_new = np.asarray([end, line], dtype=np.float32)
    window = max(1, int(round(float(window_m) / max(float(resolution_m), 1e-9))))
    p0_support = _endpoint_support_m(p0_new, wall, axis, window, float(resolution_m))
    p1_support = _endpoint_support_m(p1_new, wall, axis, window, float(resolution_m))
    return p0_new, p1_new, {
        "p0_support_m": float(p0_support),
        "p1_support_m": float(p1_support),
        "endpoint_refine_window_m": float(window_m),
    }


def _endpoint_support_m(point: np.ndarray, wall: np.ndarray, axis: str, window_cells: int, resolution_m: float) -> float:
    r, c = int(round(float(point[0]))), int(round(float(point[1])))
    count = 0
    for offset in range(-max(1, int(window_cells)), max(1, int(window_cells)) + 1):
        if str(axis) == "horizontal":
            cells = [(r + dr, c + offset) for dr in (-1, 0, 1)]
        else:
            cells = [(r + offset, c + dc) for dc in (-1, 0, 1)]
        if any(0 <= rr < wall.shape[0] and 0 <= cc < wall.shape[1] and bool(wall[rr, cc]) for rr, cc in cells):
            count += 1
    return float(count * float(resolution_m))


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
