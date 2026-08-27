from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy import ndimage

from ..dataset.category_normalizer import normalize_category
from ..dataset.occupancy_builder import bbox_grid_bounds


DEFAULT_COVERAGE_MILESTONES = (0.20, 0.40, 0.60, 0.70, 0.80, 0.90)
MANIFEST_SCHEMA_VERSION = "voxroom_roomseg_coverage_eval_v1"
FULL_VOXEL_SNAPSHOT_KEYS = (
    "voxel_occupancy_state_zyx",
    "voxel_occupancy_log_odds_zyx",
    "voxel_sensor_range_count_zyx",
    "voxel_occupancy_z_centers_m",
)


@dataclass(frozen=True)
class CoverageEvent:
    event_id: str
    event_kind: str
    step: int
    threshold: float | None
    coverage_ratio: float
    explored_cells: int
    total_explorable_cells: int
    event_dir: Path

    def metadata(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "step": int(self.step),
            "threshold": None if self.threshold is None else float(self.threshold),
            "coverage_ratio": float(self.coverage_ratio),
            "explored_cells": int(self.explored_cells),
            "total_explorable_cells": int(self.total_explorable_cells),
        }


def parse_coverage_milestones(value: str | Sequence[float] | None) -> tuple[float, ...]:
    if value is None:
        values = list(DEFAULT_COVERAGE_MILESTONES)
    elif isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        if not tokens:
            raise ValueError("coverage milestones must not be empty")
        values = [float(token) for token in tokens]
    else:
        values = [float(item) for item in value]
    normalized = [item / 100.0 if item > 1.0 else item for item in values]
    if not normalized or any(not np.isfinite(item) for item in normalized):
        raise ValueError("coverage milestones must contain finite values")
    if any(item <= 0.0 or item > 1.0 for item in normalized):
        raise ValueError("coverage milestones must be in (0, 1] or percentages in (0, 100]")
    if normalized != sorted(set(normalized)):
        raise ValueError("coverage milestones must be strictly increasing and unique")
    return tuple(float(item) for item in normalized)


def seeded_navigable_component_strict(
    navigable_mask: np.ndarray,
    seed_rc: Sequence[int],
    *,
    allow_diagonal: bool = True,
) -> np.ndarray:
    navigable = np.asarray(navigable_mask, dtype=bool)
    if navigable.ndim != 2 or not np.any(navigable):
        raise ValueError("navigable reference must be a non-empty 2D mask")
    seed = tuple(int(value) for value in seed_rc)
    if len(seed) != 2:
        raise ValueError("navigable reference seed must contain row and column")
    row, column = seed
    if not (0 <= row < navigable.shape[0] and 0 <= column < navigable.shape[1]):
        raise ValueError("navigable reference seed lies outside the map")
    if not navigable[row, column]:
        raise ValueError("navigable reference seed is not navigable")
    connectivity = 2 if allow_diagonal else 1
    labels, _ = ndimage.label(
        navigable,
        structure=ndimage.generate_binary_structure(2, connectivity),
    )
    seed_label = int(labels[row, column])
    if seed_label <= 0:
        raise RuntimeError("navigable reference seed has no connected component")
    component = labels == seed_label
    if not component[row, column] or not np.any(component):
        raise RuntimeError("failed to extract the seeded navigable component")
    return component


def same_floor_navigable_reference_strict(
    navigable_mask: np.ndarray,
    floor_objects: Sequence[Mapping[str, Any]],
    map_info: Any,
    start_grid_rc: Sequence[int],
    start_z_m: float,
    *,
    level_cluster_tolerance_m: float = 0.15,
    max_start_level_delta_m: float = 0.35,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select every navigable cell on the start floor, across disconnected rooms."""
    navigable = np.asarray(navigable_mask, dtype=bool)
    expected_shape = (
        int(getattr(map_info, "height")),
        int(getattr(map_info, "width")),
    )
    if navigable.ndim != 2 or navigable.shape != expected_shape or not np.any(navigable):
        raise ValueError("same-floor reference requires a non-empty map-aligned navigable mask")
    start_rc = tuple(int(value) for value in start_grid_rc)
    if len(start_rc) != 2:
        raise ValueError("same-floor reference start grid must contain row and column")
    start_row, start_col = start_rc
    if not (0 <= start_row < navigable.shape[0] and 0 <= start_col < navigable.shape[1]):
        raise ValueError("same-floor reference start grid lies outside the map")
    if not navigable[start_row, start_col]:
        raise ValueError("same-floor reference start grid is not navigable")

    tolerance = float(level_cluster_tolerance_m)
    max_delta = float(max_start_level_delta_m)
    if tolerance <= 0.0 or max_delta <= 0.0:
        raise ValueError("same-floor level tolerances must be positive")

    surfaces: list[tuple[float, Mapping[str, Any]]] = []
    for obj in floor_objects:
        if normalize_category(obj.get("category", "unknown")) not in {"floor", "ground"}:
            continue
        bbox_min = obj.get("bbox_min_world")
        bbox_max = obj.get("bbox_max_world")
        if bbox_min is None or bbox_max is None or len(bbox_min) < 3 or len(bbox_max) < 3:
            continue
        surfaces.append((float(bbox_max[2]), obj))
    if not surfaces:
        raise RuntimeError("same-floor reference has no floor or ground geometry")
    surfaces.sort(key=lambda item: item[0])

    clusters: list[list[tuple[float, Mapping[str, Any]]]] = []
    for surface in surfaces:
        if not clusters or surface[0] - clusters[-1][0][0] > tolerance:
            clusters.append([])
        clusters[-1].append(surface)

    candidates: list[tuple[int, float, np.ndarray, dict[str, Any]]] = []
    rejected_levels: list[dict[str, Any]] = []
    for cluster in clusters:
        heights = np.asarray([item[0] for item in cluster], dtype=np.float64)
        level_z_m = float(np.median(heights))
        level_delta_m = abs(level_z_m - float(start_z_m))
        if level_delta_m > max_delta:
            continue
        floor_mask = np.zeros_like(navigable, dtype=bool)
        rasterized_objects = 0
        for _surface_z_m, obj in cluster:
            bounds = bbox_grid_bounds(
                obj["bbox_min_world"],
                obj["bbox_max_world"],
                map_info,
            )
            if bounds is None:
                continue
            row0, row1, col0, col1 = bounds
            floor_mask[row0 : row1 + 1, col0 : col1 + 1] = True
            rasterized_objects += 1
        reference = navigable & floor_mask
        metadata = {
            "level_z_m": level_z_m,
            "level_min_z_m": float(np.min(heights)),
            "level_max_z_m": float(np.max(heights)),
            "level_delta_from_start_z_m": level_delta_m,
            "floor_object_count": int(rasterized_objects),
            "navigable_cells": int(np.count_nonzero(reference)),
            "contains_start_grid": bool(reference[start_row, start_col]),
        }
        if rasterized_objects <= 0 or not metadata["contains_start_grid"]:
            rejected_levels.append(metadata)
            continue
        candidates.append(
            (
                int(metadata["navigable_cells"]),
                -float(level_delta_m),
                reference,
                metadata,
            )
        )
    if not candidates:
        raise RuntimeError(
            "same-floor reference found no start-containing floor level: "
            f"start_z_m={float(start_z_m):.3f} rejected={rejected_levels}"
        )

    _cells, _neg_delta, selected, selected_metadata = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    if not np.any(selected) or not selected[start_row, start_col]:
        raise RuntimeError("same-floor reference selection is empty or excludes the start")
    return selected, {
        "reference_source": "same_floor_full_preprocessed_navigable",
        "selection_semantics": "floor_height_cluster_containing_start_max_navigable_area",
        "start_z_m": float(start_z_m),
        "start_grid_rc": [int(start_row), int(start_col)],
        "level_cluster_tolerance_m": tolerance,
        "max_start_level_delta_m": max_delta,
        "candidate_level_count": int(len(candidates)),
        "selected_level": selected_metadata,
        "full_navigable_cells": int(np.count_nonzero(navigable)),
        "same_floor_navigable_cells": int(np.count_nonzero(selected)),
    }


def project_mask_world_aligned_strict(
    source_mask: np.ndarray,
    source_map_info: Any,
    target_map_info: Any,
    target_shape: Sequence[int],
) -> np.ndarray:
    source = np.asarray(source_mask, dtype=bool)
    expected_source_shape = (
        int(getattr(source_map_info, "height")),
        int(getattr(source_map_info, "width")),
    )
    if source.shape != expected_source_shape:
        raise ValueError(
            f"source mask shape {source.shape} differs from map shape {expected_source_shape}"
        )
    target_hw = tuple(int(value) for value in target_shape)
    if len(target_hw) != 2 or min(target_hw) < 1:
        raise ValueError(f"invalid target map shape: {target_hw}")
    source_resolution = float(getattr(source_map_info, "resolution_m"))
    target_resolution = float(getattr(target_map_info, "resolution_m"))
    if not np.isclose(source_resolution, target_resolution, atol=1.0e-9, rtol=0.0):
        raise ValueError(
            "coverage reference projection requires identical map resolution; "
            f"source={source_resolution:.9f}, target={target_resolution:.9f}"
        )
    column_offset_float = (
        float(getattr(source_map_info, "min_x"))
        - float(getattr(target_map_info, "min_x"))
    ) / source_resolution
    row_offset_float = (
        float(getattr(target_map_info, "max_y"))
        - float(getattr(source_map_info, "max_y"))
    ) / source_resolution
    column_offset = int(round(column_offset_float))
    row_offset = int(round(row_offset_float))
    if not np.isclose(
        column_offset_float, column_offset, atol=1.0e-7, rtol=0.0
    ) or not np.isclose(row_offset_float, row_offset, atol=1.0e-7, rtol=0.0):
        raise ValueError(
            "coverage reference and runtime grids are not cell-aligned: "
            f"row_offset={row_offset_float:.9f}, "
            f"column_offset={column_offset_float:.9f}"
        )
    source_rc = np.argwhere(source)
    if source_rc.size == 0:
        raise ValueError("fixed explorable reference is empty")
    target_r = source_rc[:, 0].astype(np.int64) + row_offset
    target_c = source_rc[:, 1].astype(np.int64) + column_offset
    inside = (
        (target_r >= 0)
        & (target_r < target_hw[0])
        & (target_c >= 0)
        & (target_c < target_hw[1])
    )
    if not bool(np.all(inside)):
        raise ValueError(
            "dynamic map does not contain the complete fixed explorable reference: "
            f"outside_cells={int(np.count_nonzero(~inside))}"
        )
    flat = target_r * int(target_hw[1]) + target_c
    if int(np.unique(flat).size) != int(flat.size):
        raise ValueError("fixed explorable reference projection produced cell collisions")
    projected = np.zeros(target_hw, dtype=bool)
    projected[target_r, target_c] = True
    if int(np.count_nonzero(projected)) != int(np.count_nonzero(source)):
        raise RuntimeError("fixed explorable reference projection lost cells")
    return projected


def aligned_centered_map_size_m_strict(
    *,
    center_xy: Sequence[float],
    requested_size_m: float,
    source_map_info: Any,
    resolution_m: float,
    margin_cells: int = 1,
) -> float:
    """Choose a centered square grid that shares the source grid lattice."""

    resolution = float(resolution_m)
    requested_size = float(requested_size_m)
    center = tuple(float(value) for value in center_xy)
    if len(center) != 2 or not np.all(np.isfinite(center)):
        raise ValueError("coverage runtime map center must contain finite x and y")
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("coverage runtime map resolution must be positive")
    if not np.isfinite(requested_size) or requested_size <= 0.0:
        raise ValueError("coverage runtime map size must be positive")
    if int(margin_cells) < 0:
        raise ValueError("coverage runtime map margin must be non-negative")
    source_resolution = float(getattr(source_map_info, "resolution_m"))
    if not np.isclose(source_resolution, resolution, atol=1.0e-9, rtol=0.0):
        raise ValueError(
            "coverage runtime and source maps require identical resolution"
        )
    margin = int(margin_cells) * resolution
    required_half_extent = max(
        center[0] - float(getattr(source_map_info, "min_x")),
        float(getattr(source_map_info, "max_x")) - center[0],
        center[1] - float(getattr(source_map_info, "min_y")),
        float(getattr(source_map_info, "max_y")) - center[1],
    ) + margin
    minimum_size = max(requested_size, 2.0 * required_half_extent)
    minimum_cells = max(1, int(math.ceil(minimum_size / resolution - 1.0e-9)))
    for cell_count in range(minimum_cells, minimum_cells + 4):
        size_m = float(cell_count * resolution)
        half = 0.5 * size_m
        target_min_x = center[0] - half
        target_max_x = center[0] + half
        target_min_y = center[1] - half
        target_max_y = center[1] + half
        contains_source = (
            target_min_x <= float(getattr(source_map_info, "min_x")) + 1.0e-9
            and target_max_x >= float(getattr(source_map_info, "max_x")) - 1.0e-9
            and target_min_y <= float(getattr(source_map_info, "min_y")) + 1.0e-9
            and target_max_y >= float(getattr(source_map_info, "max_y")) - 1.0e-9
        )
        column_offset = (
            float(getattr(source_map_info, "min_x")) - target_min_x
        ) / resolution
        row_offset = (
            target_max_y - float(getattr(source_map_info, "max_y"))
        ) / resolution
        lattice_aligned = np.isclose(
            column_offset, round(column_offset), atol=1.0e-7, rtol=0.0
        ) and np.isclose(
            row_offset, round(row_offset), atol=1.0e-7, rtol=0.0
        )
        if contains_source and lattice_aligned:
            return size_m
    raise ValueError(
        "cannot create a centered runtime map aligned with the fixed scene grid; "
        "the episode start is not on the source map lattice"
    )


class CoverageMilestoneTracker:
    def __init__(
        self,
        *,
        output_dir: Path,
        simulator: str,
        reference_mask: np.ndarray,
        resolution_m: float,
        milestones: str | Sequence[float] | None = None,
        reference_arrays: Mapping[str, object] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.events_dir = self.output_dir / "events"
        self.events_dir.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.output_dir / "manifest.json"
        self.reference_npz_path = self.output_dir / "scene_reference.npz"
        self.reference_json_path = self.output_dir / "scene_reference.json"
        self.simulator = str(simulator)
        self.reference_mask = np.asarray(reference_mask, dtype=bool).copy()
        if self.reference_mask.ndim != 2 or not np.any(self.reference_mask):
            raise ValueError("fixed explorable reference must be a non-empty 2D mask")
        self.resolution_m = float(resolution_m)
        if not np.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("coverage reference resolution must be positive")
        self.milestones = parse_coverage_milestones(milestones)
        self.total_explorable_cells = int(np.count_nonzero(self.reference_mask))
        self._next_milestone_index = 0
        self._last_explored_cells = -1
        self._final_emitted = False
        self._events: list[dict[str, Any]] = []
        self._metadata = dict(metadata or {})
        arrays = {
            "reference_explorable_mask": self.reference_mask.astype(np.uint8),
            "resolution_m": np.asarray(self.resolution_m, dtype=np.float64),
            "total_explorable_cells": np.asarray(
                self.total_explorable_cells, dtype=np.int64
            ),
            **{
                str(key): np.asarray(value)
                for key, value in dict(reference_arrays or {}).items()
            },
        }
        np.savez_compressed(self.reference_npz_path, **arrays)
        _write_json_atomic(
            self.reference_json_path,
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "simulator": self.simulator,
                "resolution_m": self.resolution_m,
                "shape_hw": list(self.reference_mask.shape),
                "total_explorable_cells": self.total_explorable_cells,
                "total_explorable_area_m2": self.total_explorable_cells
                * self.resolution_m
                * self.resolution_m,
                "coverage_definition": "current_observed_intersection_fixed_full_explorable",
                "reference_npz": str(self.reference_npz_path),
                **self._metadata,
            },
        )
        self._write_manifest()

    def observe(self, *, step: int, explored_mask: np.ndarray) -> list[CoverageEvent]:
        explored_cells, ratio = self._measure(explored_mask)
        if self._last_explored_cells >= 0 and explored_cells < self._last_explored_cells:
            raise RuntimeError(
                "coverage numerator regressed even though the benchmark requires a cumulative map: "
                f"previous={self._last_explored_cells}, current={explored_cells}"
            )
        self._last_explored_cells = explored_cells
        emitted: list[CoverageEvent] = []
        while self._next_milestone_index < len(self.milestones):
            threshold = self.milestones[self._next_milestone_index]
            if ratio + 1.0e-12 < threshold:
                break
            event_id = f"milestone_{int(round(threshold * 100.0)):03d}"
            emitted.append(
                self._begin_event(
                    event_id=event_id,
                    event_kind="coverage_milestone",
                    step=int(step),
                    threshold=float(threshold),
                    coverage_ratio=ratio,
                    explored_cells=explored_cells,
                )
            )
            self._next_milestone_index += 1
        return emitted

    def finalize(self, *, step: int, explored_mask: np.ndarray) -> CoverageEvent:
        if self._final_emitted:
            raise RuntimeError("terminal coverage event was already emitted")
        explored_cells, ratio = self._measure(explored_mask)
        if self._last_explored_cells >= 0 and explored_cells < self._last_explored_cells:
            raise RuntimeError("terminal coverage numerator regressed")
        self._last_explored_cells = explored_cells
        self._final_emitted = True
        return self._begin_event(
            event_id="final",
            event_kind="terminal_forced",
            step=int(step),
            threshold=None,
            coverage_ratio=ratio,
            explored_cells=explored_cells,
        )

    def complete(self, event: CoverageEvent, *, artifacts: Mapping[str, object]) -> None:
        record = self._event_record(event.event_id)
        if record["status"] != "pending":
            raise RuntimeError(f"coverage event {event.event_id} is not pending")
        record["status"] = "complete"
        record["artifacts"] = _json_ready(dict(artifacts))
        self._write_manifest()

    def fail(self, event: CoverageEvent, exc: BaseException) -> None:
        record = self._event_record(event.event_id)
        record["status"] = "failed"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        self._write_manifest()

    def event_arrays(
        self,
        event: CoverageEvent,
        *,
        explored_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        explored = np.asarray(explored_mask, dtype=bool)
        if explored.shape != self.reference_mask.shape:
            raise ValueError("coverage event explored mask shape changed")
        return {
            "roomseg_eval_reference_explorable_mask": self.reference_mask.astype(
                np.uint8
            ),
            "roomseg_eval_explored_reference_mask": (
                explored & self.reference_mask
            ).astype(np.uint8),
            "roomseg_eval_event_id": np.asarray(event.event_id),
            "roomseg_eval_event_kind": np.asarray(event.event_kind),
            "roomseg_eval_threshold": np.asarray(
                np.nan if event.threshold is None else event.threshold,
                dtype=np.float64,
            ),
            "roomseg_eval_coverage_ratio": np.asarray(
                event.coverage_ratio, dtype=np.float64
            ),
            "roomseg_eval_explored_cells": np.asarray(
                event.explored_cells, dtype=np.int64
            ),
            "roomseg_eval_total_explorable_cells": np.asarray(
                event.total_explorable_cells, dtype=np.int64
            ),
        }

    def _measure(self, explored_mask: np.ndarray) -> tuple[int, float]:
        explored = np.asarray(explored_mask, dtype=bool)
        if explored.shape != self.reference_mask.shape:
            raise ValueError(
                f"explored mask shape {explored.shape} differs from fixed reference "
                f"{self.reference_mask.shape}"
            )
        count = int(np.count_nonzero(explored & self.reference_mask))
        return count, float(count / self.total_explorable_cells)

    def _begin_event(
        self,
        *,
        event_id: str,
        event_kind: str,
        step: int,
        threshold: float | None,
        coverage_ratio: float,
        explored_cells: int,
    ) -> CoverageEvent:
        if any(record["event_id"] == event_id for record in self._events):
            raise RuntimeError(f"duplicate coverage event id: {event_id}")
        event_dir = self.events_dir / event_id
        event_dir.mkdir(parents=False, exist_ok=False)
        event = CoverageEvent(
            event_id=event_id,
            event_kind=event_kind,
            step=int(step),
            threshold=threshold,
            coverage_ratio=float(coverage_ratio),
            explored_cells=int(explored_cells),
            total_explorable_cells=self.total_explorable_cells,
            event_dir=event_dir,
        )
        self._events.append(
            {
                **event.metadata(),
                "status": "pending",
                "event_dir": str(event_dir),
                "coverage_overshoot": None
                if threshold is None
                else float(max(0.0, coverage_ratio - threshold)),
                "artifacts": {},
            }
        )
        self._write_manifest()
        return event

    def _event_record(self, event_id: str) -> dict[str, Any]:
        matches = [record for record in self._events if record["event_id"] == event_id]
        if len(matches) != 1:
            raise RuntimeError(f"coverage event record not found: {event_id}")
        return matches[0]

    def _write_manifest(self) -> None:
        _write_json_atomic(
            self.manifest_path,
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "simulator": self.simulator,
                "coverage_definition": "current_observed_intersection_fixed_full_explorable",
                "milestones": list(self.milestones),
                "terminal_event_required": True,
                "reference_npz": str(self.reference_npz_path),
                "reference_json": str(self.reference_json_path),
                "total_explorable_cells": self.total_explorable_cells,
                "resolution_m": self.resolution_m,
                "events": self._events,
                **self._metadata,
            },
        )


def door_segments_rc_to_line_mask(
    segments_rc: np.ndarray,
    *,
    shape: Sequence[int],
    thickness_cells: int = 3,
) -> np.ndarray:
    target_shape = tuple(int(value) for value in shape)
    if len(target_shape) != 2 or min(target_shape) < 1:
        raise ValueError(f"invalid door-line target shape: {target_shape}")
    raw_segments = np.asarray(segments_rc)
    if raw_segments.size == 0:
        return np.zeros(target_shape, dtype=bool)
    try:
        numeric_segments = np.asarray(raw_segments, dtype=np.float64).reshape(-1, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError("door segments must be an Nx4 numeric array") from exc
    if not np.all(np.isfinite(numeric_segments)):
        raise ValueError("door segment contains non-finite coordinates")
    if not np.allclose(
        numeric_segments,
        np.rint(numeric_segments),
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("door segment coordinates must be integer grid cells")
    segments = np.rint(numeric_segments).astype(np.int64)
    rows = segments[:, (0, 2)]
    columns = segments[:, (1, 3)]
    if (
        np.any(rows < 0)
        or np.any(rows >= target_shape[0])
        or np.any(columns < 0)
        or np.any(columns >= target_shape[1])
    ):
        raise ValueError("door segment lies outside the target map")
    if int(thickness_cells) < 1:
        raise ValueError("door-line thickness must be positive")
    mask = np.zeros(target_shape, dtype=np.uint8)
    for start_r, start_c, end_r, end_c in segments:
        cv2.line(
            mask,
            (int(start_c), int(start_r)),
            (int(end_c), int(end_r)),
            color=1,
            thickness=int(thickness_cells),
            lineType=cv2.LINE_8,
        )
    return mask.astype(bool)


def partition_free_space_by_door_lines(
    free_mask: np.ndarray,
    door_line_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    free = np.asarray(free_mask, dtype=bool)
    door_lines = np.asarray(door_line_mask, dtype=bool)
    if free.ndim != 2 or door_lines.shape != free.shape:
        raise ValueError("free mask and door-line mask must be matching 2D arrays")
    if not np.any(free):
        raise ValueError("door-line partition requires a non-empty observed free domain")
    cut_cells = free & door_lines
    partition_domain = free & ~cut_cells
    structure = np.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    labels, component_count = ndimage.label(partition_domain, structure=structure)
    labels = labels.astype(np.int32, copy=False)
    if int(component_count) < 1:
        raise RuntimeError("door lines removed the entire observed free domain")
    if np.any(cut_cells):
        _distance, nearest = ndimage.distance_transform_edt(
            labels == 0,
            return_distances=True,
            return_indices=True,
        )
        nearest_labels = labels[tuple(nearest)]
        labels[cut_cells] = nearest_labels[cut_cells]
    labels[~free] = 0
    if np.any(free & (labels <= 0)):
        raise RuntimeError("door-line partition left observed free cells unlabeled")
    return labels, {
        "observed_free_cells": int(np.count_nonzero(free)),
        "door_line_cells": int(np.count_nonzero(door_lines)),
        "door_cut_cells": int(np.count_nonzero(cut_cells)),
        "room_count": int(component_count),
    }


def strict_voxel_snapshot_arrays(mapper: Any) -> dict[str, np.ndarray]:
    voxel_grid = getattr(mapper, "voxel_grid", None)
    if voxel_grid is None:
        raise RuntimeError("coverage evaluation requires the full 3D voxel grid")
    state = np.asarray(getattr(voxel_grid, "state", None), dtype=np.uint8)
    log_odds = np.asarray(getattr(voxel_grid, "log_odds", None), dtype=np.int16)
    sensor_range = np.asarray(
        getattr(voxel_grid, "sensor_range_count", None), dtype=np.uint8
    )
    if state.ndim != 3 or min(state.shape) < 1:
        raise RuntimeError(f"invalid 3D voxel state shape: {state.shape}")
    if log_odds.shape != state.shape or sensor_range.shape != state.shape:
        raise RuntimeError(
            "3D voxel state, log odds, and sensor-range arrays must have identical shapes"
        )
    z_centers = np.asarray(getattr(voxel_grid, "z_centers_m", None), dtype=np.float32)
    if z_centers.shape != (state.shape[0],):
        raise RuntimeError("voxel z-center array does not match the 3D voxel state")
    return {
        FULL_VOXEL_SNAPSHOT_KEYS[0]: state.copy(),
        FULL_VOXEL_SNAPSHOT_KEYS[1]: log_odds.copy(),
        FULL_VOXEL_SNAPSHOT_KEYS[2]: sensor_range.copy(),
        FULL_VOXEL_SNAPSHOT_KEYS[3]: z_centers.copy(),
    }


def strict_voxel_navigation_projection_arrays(
    mapper: Any,
    *,
    shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    target_shape = tuple(int(value) for value in shape)
    if len(target_shape) != 2 or min(target_shape) < 1:
        raise ValueError(f"invalid navigation projection shape: {target_shape}")
    projection = getattr(mapper, "last_voxel_navigation_projection", None)
    if projection is None:
        raise RuntimeError(
            "coverage evaluation requires mapper.last_voxel_navigation_projection"
        )
    masks: dict[str, np.ndarray] = {}
    for name in ("free", "occupied", "observed", "unknown"):
        value = getattr(projection, name, None)
        if value is None:
            raise RuntimeError(f"voxel navigation projection is missing {name}")
        mask = np.asarray(value, dtype=bool)
        if mask.shape != target_shape:
            raise RuntimeError(
                f"voxel navigation projection {name} shape {mask.shape} "
                f"differs from {target_shape}"
            )
        masks[name] = mask.copy()
    if np.any(masks["free"] & masks["occupied"]):
        raise RuntimeError("voxel navigation projection marks cells free and occupied")
    if np.any(masks["unknown"] & (masks["free"] | masks["occupied"])):
        raise RuntimeError("voxel navigation unknown overlaps a known state")
    if np.any((masks["free"] | masks["occupied"]) & ~masks["observed"]):
        raise RuntimeError("voxel navigation known state lies outside observed")
    if not np.array_equal(masks["unknown"], ~masks["observed"]):
        raise RuntimeError(
            "voxel navigation unknown mask differs from inverse observed"
        )
    return (
        masks["free"],
        masks["occupied"],
        masks["unknown"],
        "mapper.last_voxel_navigation_projection",
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
