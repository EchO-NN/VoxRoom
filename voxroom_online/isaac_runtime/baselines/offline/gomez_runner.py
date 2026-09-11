from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from voxroom_online.isaac_runtime.evaluation.online_roomseg.hough_raw_seed import (
    pair_raw_door_lines,
    rasterize_lines,
)

from ..data_contract import resolve_map_info
from ..mask_io import (
    build_segmentation_domain_from_source,
    enforce_room_mask_contract,
)
from .base import BaselineResult


GOMEZ_PAPER_DOI = "10.3390/s19204595"
GOMEZ_PAPER_URL = "https://doi.org/10.3390/s19204595"


@dataclass(frozen=True)
class GomezParameters:
    maximum_laser_range_m: float = 5.0
    successive_scan_gap_m: float = 0.8
    minimum_interesting_gap_scans: int = 10
    small_door_min_m: float = 0.8
    small_door_max_m: float = 1.2
    big_door_min_m: float = 1.6
    big_door_max_m: float = 2.4
    secure_approach_distance_m: float = 0.9
    cross_door_distance_m: float = 0.5
    transit_semantic_utility: float = 80.0
    free_semantic_utility: float = 1.0
    minimum_interesting_function_value: float = 30.0
    separator_width_cells: int = 3
    duplicate_endpoint_tolerance_m: float = 0.25
    voxel_hough_z_min_m: float = 0.20
    voxel_hough_z_max_m: float = 2.20
    voxel_hough_min_vertical_line_height_m: float = 0.80
    voxel_hough_max_vertical_angle_deg: float = 12.0
    voxel_hough_max_line_gap_m: float = 0.15
    voxel_hough_endpoint_tolerance_m: float = 0.25
    voxel_hough_cross_section_half_width_m: float = 0.10
    voxel_hough_extension_margin_m: float = 0.30


class GomezIncrementalRunner:
    """Map-replay reproduction of Gomez et al.'s incremental room topology.

    The paper's source code was unavailable to the TVARS authors as well.  This
    runner consumes the raw range-jump points saved by the actual TVARS run;
    it never re-casts a virtual laser at an evaluation checkpoint.  It then
    applies the published door-size rule and vertical door-frame Hough
    confirmation.  The Hough image is a vertical cross-section sampled directly
    from the saved 3-D voxel map instead of the unavailable online RGB frame.
    Confirmed transit lines are persistent across a scene and split the current
    free-space map into rooms.  It intentionally does not claim to be an
    original-source implementation.
    """

    baseline_name = "gomez_incremental"

    def __init__(
        self,
        *,
        map_resolution_m: float | None = None,
        parameters: GomezParameters | None = None,
    ) -> None:
        self.map_resolution_m = map_resolution_m
        self.parameters = parameters or GomezParameters()
        self.scene_id: str | None = None
        self._candidate_lines: list[tuple[int, int, int, int]] = []
        self._accepted_lines: list[tuple[int, int, int, int]] = []
        self._last_step: int | None = None
        self._last_result: BaselineResult | None = None

    def start_scene(self, scene_id: str) -> None:
        self.scene_id = str(scene_id)
        self._candidate_lines = []
        self._accepted_lines = []
        self._last_step = None
        self._last_result = None

    def end_scene(self) -> None:
        self.scene_id = None
        self._candidate_lines = []
        self._accepted_lines = []
        self._last_step = None
        self._last_result = None

    def segment_snapshot(
        self, snapshot_path: Path | str, arrays: Mapping[str, Any]
    ) -> BaselineResult:
        snapshot_path = Path(snapshot_path)
        step = _snapshot_step(snapshot_path, arrays)
        if self._last_step is not None and step < self._last_step:
            raise ValueError(
                "Gomez incremental replay requires non-decreasing checkpoint steps: "
                f"previous={self._last_step}, current={step}, path={snapshot_path}"
            )
        if self._last_step == step:
            if self._last_result is None:
                raise RuntimeError("same-step Gomez replay has no cached result")
            _raw_point_mask, raw_seed_source_snapshot = _saved_tvars_raw_seed_points(
                arrays,
                self._last_result.label_map.shape,
            )
            metadata = dict(self._last_result.metadata)
            metadata.update(
                {
                    "source_snapshot": str(snapshot_path),
                    "raw_seed_source_snapshot": raw_seed_source_snapshot,
                    "same_step_result_reused": True,
                }
            )
            return BaselineResult(
                label_map=np.asarray(self._last_result.label_map, dtype=np.int32).copy(),
                metadata=metadata,
                debug_arrays={
                    key: np.asarray(value).copy()
                    for key, value in self._last_result.debug_arrays.items()
                },
            )
        self._last_step = int(step)

        free, segmentation_source = build_segmentation_domain_from_source(arrays)
        map_info = resolve_map_info(
            snapshot_arrays=arrays,
            default_resolution_m=self.map_resolution_m or 0.05,
        )
        wall = _vertical_wall_mask(arrays, free.shape)
        wall &= ~free
        raw_point_mask, raw_seed_source_snapshot = _saved_tvars_raw_seed_points(
            arrays,
            free.shape,
        )
        raw_rows, raw_cols = np.nonzero(raw_point_mask)
        raw_points = tuple(
            (int(col), int(row)) for row, col in zip(raw_rows, raw_cols)
        )
        detection_range = max(
            20,
            int(np.ceil(self.parameters.big_door_max_m / map_info.resolution_m))
            + 2,
        )
        detected_lines = pair_raw_door_lines(
            obstacle_mask=wall,
            candidate_points_xy=raw_points,
            detection_range=detection_range,
            noisy_component_cells=max(
                5, int(self.parameters.minimum_interesting_gap_scans // 2)
            ),
        )
        size_filtered = [
            line
            for line in detected_lines
            if _is_door_width(
                _line_length_m(line, float(map_info.resolution_m)),
                self.parameters,
            )
        ]
        for line in size_filtered:
            _append_unique_line(
                self._candidate_lines,
                line,
                tolerance_cells=max(
                    1.0,
                    self.parameters.duplicate_endpoint_tolerance_m
                    / float(map_info.resolution_m),
                ),
            )

        hough_confirmed, hough_debug = _voxel_hough_confirm_door_lines(
            arrays,
            self._candidate_lines,
            map_shape=free.shape,
            resolution_m=float(map_info.resolution_m),
            parameters=self.parameters,
        )
        accepted_now: list[tuple[int, int, int, int]] = []
        for line in hough_confirmed:
            if _line_splits_free_space(
                free,
                line,
                width_cells=int(self.parameters.separator_width_cells),
            ):
                accepted_now.append(line)
        for line in accepted_now:
            _append_unique_line(
                self._accepted_lines,
                line,
                tolerance_cells=max(
                    1.0,
                    self.parameters.duplicate_endpoint_tolerance_m
                    / float(map_info.resolution_m),
                ),
            )

        separator = rasterize_lines(
            self._accepted_lines,
            shape=free.shape,
            width_cells=max(1, int(self.parameters.separator_width_cells)),
        )
        labels = _partition_and_restore_door_cells(free, separator)
        labels = enforce_room_mask_contract(labels, arrays, clip_to_eval_domain=True)
        candidate_mask = rasterize_lines(
            self._candidate_lines,
            shape=free.shape,
            width_cells=1,
        )
        hough_confirmed_mask = rasterize_lines(
            hough_confirmed,
            shape=free.shape,
            width_cells=1,
        )
        hough_confirmed_identities = {
            _canonical_line(line) for line in hough_confirmed
        }
        hough_rejected = [
            line
            for line in self._candidate_lines
            if _canonical_line(line) not in hough_confirmed_identities
        ]
        hough_rejected_mask = rasterize_lines(
            hough_rejected,
            shape=free.shape,
            width_cells=1,
        )
        metadata = {
            "method": self.baseline_name,
            "baseline_name": self.baseline_name,
            "runner_type": "paper_reproduction",
            "main_experiment_allowed": True,
            "source_code_available": False,
            "source_unavailable_reported_by_tvars_paper": True,
            "implementation_reference": GOMEZ_PAPER_DOI,
            "implementation_reference_url": GOMEZ_PAPER_URL,
            "reproduction_scope": "segmentation_replay_on_saved_vertical_free_checkpoints",
            "source_snapshot": str(snapshot_path),
            "scene_id": self.scene_id,
            "step": int(step),
            "segmentation_source": segmentation_source,
            "map_info": map_info.to_metadata(),
            "map_resolution_m": float(map_info.resolution_m),
            "incremental_state_reset_per_scene": True,
            "incremental_order_enforced": True,
            "door_lines_persistent": True,
            "raw_frontier_point_count": int(len(raw_points)),
            "raw_door_seed_point_count": int(len(raw_points)),
            "raw_seed_source": "saved_tvars_original_hough_door_seed_map",
            "raw_seed_source_snapshot": raw_seed_source_snapshot,
            "checkpoint_virtual_laser_recomputed": False,
            "door_line_count_detected_this_checkpoint": int(len(detected_lines)),
            "door_line_count_size_filtered_this_checkpoint": int(len(size_filtered)),
            "door_line_candidate_count_persistent": int(len(self._candidate_lines)),
            "door_line_hough_confirmed_candidate_count_current": int(
                len(hough_confirmed)
            ),
            "door_line_hough_rejected_candidate_count_current": int(
                len(hough_rejected)
            ),
            "door_line_accepted_count_persistent": int(len(self._accepted_lines)),
            "parameters": asdict(self.parameters),
            "uses_rgb": False,
            "uses_depth": False,
            "uses_voxel_3d": True,
            "voxel_hough_confirmation_enabled": True,
            "hough_confirmation_source": "voxel_occupancy_state_zyx_vertical_cross_section",
            "voxel_hough_confirmation": hough_debug,
            "uses_oracle_semantics": False,
            "reproduction_limitations": [
                "Original Gomez source code is unavailable.",
                "Raw range-jump points are shared from the paired saved TVARS execution rather than recomputed by Gomez.",
                "A vertical cross-section of the saved 3-D voxel map replaces the paper's online RGB frame as Hough input.",
                "Saved checkpoints cannot reproduce the paper's explicit 0.9 m approach-and-redetect behavior at every candidate.",
            ],
        }
        debug = {
            "gomez_vertical_free_input": free.astype(bool),
            "gomez_vertical_wall_input": wall.astype(bool),
            "gomez_tvars_raw_seed_point_mask": raw_point_mask.astype(bool),
            "gomez_candidate_door_line_mask": candidate_mask.astype(bool),
            "gomez_voxel_hough_confirmed_line_mask": hough_confirmed_mask.astype(bool),
            "gomez_voxel_hough_rejected_line_mask": hough_rejected_mask.astype(bool),
            "gomez_accepted_door_separator_mask": separator.astype(bool),
        }
        result = BaselineResult(
            label_map=np.asarray(labels, dtype=np.int32),
            metadata=metadata,
            debug_arrays=debug,
        )
        self._last_result = result
        return result


def _snapshot_step(path: Path, arrays: Mapping[str, Any]) -> int:
    value = arrays.get("step")
    if value is not None:
        try:
            return int(np.asarray(value).reshape(-1)[0])
        except Exception:
            pass
    name = path.stem
    for token in reversed(name.split("_")):
        if token.isdigit():
            return int(token)
    raise ValueError(f"cannot determine snapshot step: {path}")


def _vertical_wall_mask(
    arrays: Mapping[str, Any], shape: tuple[int, int]
) -> np.ndarray:
    for key in (
        "voxel_wall_xy",
        "structural_wall_clean",
        "roomseg_sanitized_wall",
        "initial_roomseg_occupied_after_fusion",
        "initial_roomseg_occupied",
        "obstacle_mask",
    ):
        value = arrays.get(key)
        if value is None:
            continue
        mask = np.asarray(value, dtype=bool)
        if mask.shape == shape and bool(np.any(mask)):
            return mask.copy()
    raise ValueError("Gomez replay requires a saved Vertical-wall/obstacle layer")


def _saved_tvars_raw_seed_points(
    arrays: Mapping[str, Any], shape: tuple[int, int]
) -> tuple[np.ndarray, str]:
    key = "tvars_original_hough_door_seed_map"
    value = arrays.get(key)
    if value is None:
        raise ValueError(
            "Gomez replay requires the paired saved TVARS raw-seed point map: "
            + key
        )
    mask = np.asarray(value, dtype=bool)
    if mask.shape != tuple(shape):
        raise ValueError("saved TVARS raw-seed point map shape does not match the map")
    source_value = arrays.get("tvars_original_raw_seed_source_snapshot")
    source = ""
    if source_value is not None:
        flat = np.asarray(source_value).reshape(-1)
        source = str(flat[0]) if flat.size else ""
    return mask.copy(), source


def _line_length_m(
    line: tuple[int, int, int, int], resolution_m: float
) -> float:
    x0, y0, x1, y1 = line
    return float(np.hypot(x1 - x0, y1 - y0) * float(resolution_m))


def _is_door_width(width_m: float, params: GomezParameters) -> bool:
    return bool(
        params.small_door_min_m <= width_m <= params.small_door_max_m
        or params.big_door_min_m <= width_m <= params.big_door_max_m
    )


def _canonical_line(
    line: tuple[int, int, int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    endpoints = ((int(line[0]), int(line[1])), (int(line[2]), int(line[3])))
    return tuple(sorted(endpoints))  # type: ignore[return-value]


def _line_distance_cells(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    a0, a1 = _canonical_line(first)
    b0, b1 = _canonical_line(second)
    direct = max(float(np.hypot(a0[0] - b0[0], a0[1] - b0[1])), float(np.hypot(a1[0] - b1[0], a1[1] - b1[1])))
    reverse = max(float(np.hypot(a0[0] - b1[0], a0[1] - b1[1])), float(np.hypot(a1[0] - b0[0], a1[1] - b0[1])))
    return min(direct, reverse)


def _append_unique_line(
    lines: list[tuple[int, int, int, int]],
    line: tuple[int, int, int, int],
    *,
    tolerance_cells: float,
) -> None:
    normalized = tuple(int(v) for v in line)
    if any(
        _line_distance_cells(existing, normalized) <= float(tolerance_cells)
        for existing in lines
    ):
        return
    lines.append(normalized)  # type: ignore[arg-type]


def _voxel_hough_confirm_door_lines(
    arrays: Mapping[str, Any],
    lines: list[tuple[int, int, int, int]],
    *,
    map_shape: tuple[int, int],
    resolution_m: float,
    parameters: GomezParameters,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    state_value = arrays.get("voxel_occupancy_state_zyx")
    if state_value is None:
        raise ValueError(
            "Gomez voxel-Hough confirmation requires "
            "voxel_occupancy_state_zyx"
        )
    state = np.asarray(state_value, dtype=np.uint8)
    if state.ndim != 3 or tuple(state.shape[1:]) != tuple(map_shape):
        raise ValueError(
            "voxel_occupancy_state_zyx must have shape [Z,H,W] matching the map"
        )
    z_centers = _voxel_z_centers(arrays, int(state.shape[0]))
    active_z = np.flatnonzero(
        np.isfinite(z_centers)
        & (z_centers >= float(parameters.voxel_hough_z_min_m))
        & (z_centers <= float(parameters.voxel_hough_z_max_m))
    )
    if active_z.size < 2:
        raise ValueError("Gomez voxel-Hough active height range has fewer than two bins")
    z_resolution_m = float(np.median(np.diff(z_centers[active_z])))
    if not np.isfinite(z_resolution_m) or z_resolution_m <= 0.0:
        raise ValueError("Gomez voxel-Hough requires increasing voxel z centers")

    confirmed: list[tuple[int, int, int, int]] = []
    records: list[dict[str, Any]] = []
    for line in lines:
        record = _voxel_hough_confirm_one_door_line(
            state=state,
            active_z=active_z,
            line=line,
            xy_resolution_m=float(resolution_m),
            z_resolution_m=z_resolution_m,
            parameters=parameters,
        )
        records.append(record)
        if bool(record["confirmed"]):
            confirmed.append(line)

    return confirmed, {
        "algorithm": "probabilistic_hough_vertical_lines_on_voxel_cross_section_v1",
        "input_key": "voxel_occupancy_state_zyx",
        "occupied_state_value": 2,
        "candidate_count": int(len(lines)),
        "confirmed_count": int(len(confirmed)),
        "rejected_count": int(len(lines) - len(confirmed)),
        "active_z_bin_count": int(active_z.size),
        "active_z_min_m": float(z_centers[active_z[0]]),
        "active_z_max_m": float(z_centers[active_z[-1]]),
        "z_resolution_m": float(z_resolution_m),
        "records": records,
    }


def _voxel_z_centers(
    arrays: Mapping[str, Any], voxel_z_count: int
) -> np.ndarray:
    value = arrays.get("voxel_occupancy_z_centers_m")
    if value is not None:
        centers = np.asarray(value, dtype=np.float64).reshape(-1)
        if centers.size != int(voxel_z_count):
            raise ValueError(
                "voxel_occupancy_z_centers_m does not match the voxel grid"
            )
        return centers
    z_min = _array_scalar(arrays, "voxel_occupancy_z_min_m", 0.0)
    z_resolution = _array_scalar(
        arrays,
        "voxel_occupancy_z_resolution_m",
        0.05,
    )
    if z_resolution <= 0.0:
        raise ValueError("voxel_occupancy_z_resolution_m must be positive")
    return z_min + (np.arange(int(voxel_z_count), dtype=np.float64) + 0.5) * z_resolution


def _array_scalar(
    arrays: Mapping[str, Any], key: str, default: float
) -> float:
    value = arrays.get(key)
    if value is None:
        return float(default)
    flat = np.asarray(value).reshape(-1)
    return float(flat[0]) if flat.size else float(default)


def _voxel_hough_confirm_one_door_line(
    *,
    state: np.ndarray,
    active_z: np.ndarray,
    line: tuple[int, int, int, int],
    xy_resolution_m: float,
    z_resolution_m: float,
    parameters: GomezParameters,
) -> dict[str, Any]:
    import cv2

    x0, y0, x1, y1 = (int(value) for value in line)
    delta = np.asarray([float(x1 - x0), float(y1 - y0)], dtype=np.float64)
    length_cells = float(np.linalg.norm(delta))
    if length_cells <= 1e-6:
        return {
            "line_xyxy": [x0, y0, x1, y1],
            "confirmed": False,
            "reason": "degenerate_candidate_line",
            "vertical_line_count": 0,
            "endpoint_0_confirmed": False,
            "endpoint_1_confirmed": False,
        }

    along = delta / length_cells
    perpendicular = np.asarray([-along[1], along[0]], dtype=np.float64)
    margin_cells = max(
        1,
        int(np.ceil(float(parameters.voxel_hough_extension_margin_m) / xy_resolution_m)),
    )
    half_width_cells = max(
        0,
        int(
            np.ceil(
                float(parameters.voxel_hough_cross_section_half_width_m)
                / xy_resolution_m
            )
        ),
    )
    sample_count = max(2, int(np.ceil(length_cells + 2.0 * margin_cells)) + 1)
    positions = np.linspace(
        -float(margin_cells),
        length_cells + float(margin_cells),
        sample_count,
        dtype=np.float64,
    )
    offsets = np.arange(-half_width_cells, half_width_cells + 1, dtype=np.float64)
    centers_x = float(x0) + positions[:, None] * along[0]
    centers_y = float(y0) + positions[:, None] * along[1]
    cols = np.rint(centers_x + offsets[None, :] * perpendicular[0]).astype(np.int32)
    rows = np.rint(centers_y + offsets[None, :] * perpendicular[1]).astype(np.int32)
    valid = (
        (rows >= 0)
        & (rows < int(state.shape[1]))
        & (cols >= 0)
        & (cols < int(state.shape[2]))
    )
    clipped_rows = np.clip(rows, 0, int(state.shape[1]) - 1)
    clipped_cols = np.clip(cols, 0, int(state.shape[2]) - 1)
    sampled = state[
        active_z[:, None, None],
        clipped_rows[None, :, :],
        clipped_cols[None, :, :],
    ]
    occupied_cross_section = np.any(
        (sampled == 2) & valid[None, :, :],
        axis=2,
    )
    image = (occupied_cross_section.astype(np.uint8) * 255)
    max_gap_bins = max(
        0,
        int(
            np.ceil(
                float(parameters.voxel_hough_max_line_gap_m) / z_resolution_m
            )
        ),
    )
    if max_gap_bins > 0:
        image = cv2.morphologyEx(
            image,
            cv2.MORPH_CLOSE,
            np.ones((2 * max_gap_bins + 1, 1), dtype=np.uint8),
        )
    min_height_bins = max(
        2,
        int(
            np.ceil(
                float(parameters.voxel_hough_min_vertical_line_height_m)
                / z_resolution_m
            )
        ),
    )
    detected = cv2.HoughLinesP(
        image,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=max(4, int(np.ceil(0.5 * min_height_bins))),
        minLineLength=int(min_height_bins),
        maxLineGap=int(max_gap_bins),
    )
    vertical_segments: list[tuple[float, int, int, int, int]] = []
    if detected is not None:
        for raw in np.asarray(detected).reshape(-1, 4):
            sx0, sz0, sx1, sz1 = (int(value) for value in raw)
            dz = abs(sz1 - sz0)
            dx = abs(sx1 - sx0)
            if dz < int(min_height_bins):
                continue
            angle_from_vertical_deg = float(np.degrees(np.arctan2(dx, max(1, dz))))
            if angle_from_vertical_deg > float(
                parameters.voxel_hough_max_vertical_angle_deg
            ):
                continue
            vertical_segments.append(
                (0.5 * float(sx0 + sx1), sx0, sz0, sx1, sz1)
            )

    endpoint_tolerance_samples = max(
        1.0,
        float(parameters.voxel_hough_endpoint_tolerance_m)
        / xy_resolution_m,
    )
    sample_step_cells = (
        (length_cells + 2.0 * margin_cells) / float(max(1, sample_count - 1))
    )
    endpoint_tolerance_x = endpoint_tolerance_samples / max(
        sample_step_cells, 1e-6
    )
    endpoint_x = (
        float(margin_cells) / max(sample_step_cells, 1e-6),
        float(margin_cells + length_cells) / max(sample_step_cells, 1e-6),
    )
    endpoint_confirmed = tuple(
        any(abs(segment[0] - expected_x) <= endpoint_tolerance_x for segment in vertical_segments)
        for expected_x in endpoint_x
    )
    confirmed = bool(endpoint_confirmed[0] and endpoint_confirmed[1])
    return {
        "line_xyxy": [x0, y0, x1, y1],
        "confirmed": confirmed,
        "reason": "both_voxel_doorframe_sides_detected" if confirmed else "missing_voxel_doorframe_vertical_line",
        "vertical_line_count": int(len(vertical_segments)),
        "endpoint_0_confirmed": bool(endpoint_confirmed[0]),
        "endpoint_1_confirmed": bool(endpoint_confirmed[1]),
        "cross_section_shape_zs": [int(image.shape[0]), int(image.shape[1])],
        "occupied_cross_section_cells": int(np.count_nonzero(image)),
        "minimum_vertical_line_height_bins": int(min_height_bins),
        "endpoint_tolerance_samples": float(endpoint_tolerance_x),
    }


def _line_splits_free_space(
    free: np.ndarray,
    line: tuple[int, int, int, int],
    *,
    width_cells: int,
) -> bool:
    domain = np.asarray(free, dtype=bool)
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    _before, n_before = ndimage.label(domain, structure=structure)
    cut = rasterize_lines((line,), shape=domain.shape, width_cells=max(1, width_cells))
    if int(np.count_nonzero(cut & domain)) < 2:
        return False
    _after, n_after = ndimage.label(domain & ~cut, structure=structure)
    return int(n_after) > int(n_before)


def _partition_and_restore_door_cells(
    free: np.ndarray, separator: np.ndarray
) -> np.ndarray:
    domain = np.asarray(free, dtype=bool)
    cut = np.asarray(separator, dtype=bool) & domain
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, _count = ndimage.label(domain & ~cut, structure=structure)
    labels = np.asarray(labels, dtype=np.int32)
    missing = cut & (labels == 0)
    if bool(np.any(missing)) and bool(np.any(labels > 0)):
        nearest = ndimage.distance_transform_edt(
            labels == 0, return_distances=False, return_indices=True
        )
        labels[missing] = labels[tuple(axis[missing] for axis in nearest)]
    labels[~domain] = 0
    return labels
