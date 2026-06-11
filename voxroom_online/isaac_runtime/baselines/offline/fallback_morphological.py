from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage


IPA_MORPHOLOGICAL_ALGORITHM_ID = 1
IPA_MORPHOLOGICAL_LOWER_LIMIT_M2 = 0.8
IPA_MORPHOLOGICAL_UPPER_LIMIT_M2 = 47.0
IPA_MORPHOLOGICAL_MAX_EROSIONS = 1000
IPA_UNASSIGNED_FREE_LABEL = 65280


@dataclass(frozen=True)
class MorphologicalFallbackConfig:
    map_resolution_m: float = 0.05
    room_area_lower_limit_m2: float = IPA_MORPHOLOGICAL_LOWER_LIMIT_M2
    room_area_upper_limit_m2: float = IPA_MORPHOLOGICAL_UPPER_LIMIT_M2
    max_iterations: int = IPA_MORPHOLOGICAL_MAX_EROSIONS
    erosion_connectivity: int = 2
    component_connectivity: int = 2
    wavefront_connectivity: int = 2

    def area_limits_cells(self) -> tuple[int, int]:
        cell_area = max(float(self.map_resolution_m) ** 2, 1e-12)
        lower = int(np.ceil(float(self.room_area_lower_limit_m2) / cell_area))
        upper = int(np.floor(float(self.room_area_upper_limit_m2) / cell_area))
        return max(1, lower), max(1, upper)


@dataclass(frozen=True)
class MorphologicalSegmentationResult:
    room_label_map: np.ndarray
    metadata: dict[str, object]
    debug_arrays: dict[str, np.ndarray]


def build_accessible_free_mask(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, object]]:
    """Build the 255-valued IPA free domain; unknown is never treated as free."""

    shape = _infer_shape(arrays)
    candidate, candidate_key = _first_bool_array(
        arrays,
        (
            "navigation_free_room_domain",
            "observed_free_mask",
            "voxel_nav_free_xy",
            "vertical_free_room_domain",
            "voxel_vertical_free_xy",
        ),
        shape,
    )
    if candidate is None:
        occupancy, occupancy_key = _first_bool_array(arrays, ("occupancy_map", "voxel_nav_occupied_xy"), shape)
        if occupancy is None:
            candidate = np.zeros(shape, dtype=bool)
            candidate_key = "missing_free_source"
        else:
            candidate = ~occupancy
            candidate_key = "not_%s" % occupancy_key

    unknown, unknown_key = _first_bool_array(arrays, ("unknown_mask", "voxel_nav_unknown_xy", "voxel_unknown_xy"), shape)
    obstacle, obstacle_key = _first_bool_array(arrays, ("obstacle_mask", "occupancy_map", "voxel_nav_occupied_xy", "voxel_wall_xy"), shape)
    if unknown is None:
        unknown = np.zeros(shape, dtype=bool)
        unknown_key = "absent"
    if obstacle is None:
        obstacle = np.zeros(shape, dtype=bool)
        obstacle_key = "absent"

    candidate = np.asarray(candidate, dtype=bool)
    unknown = np.asarray(unknown, dtype=bool)
    obstacle = np.asarray(obstacle, dtype=bool)
    free = candidate & ~unknown & ~obstacle
    metadata = {
        "free_source": str(candidate_key),
        "unknown_source": str(unknown_key),
        "obstacle_source": str(obstacle_key),
        "candidate_free_cells": int(np.count_nonzero(candidate)),
        "unknown_cells": int(np.count_nonzero(unknown)),
        "obstacle_cells": int(np.count_nonzero(obstacle)),
        "unknown_excluded_from_free_cells": int(np.count_nonzero(candidate & unknown)),
        "obstacle_excluded_from_free_cells": int(np.count_nonzero(candidate & obstacle)),
        "final_free_cells": int(np.count_nonzero(free)),
        "unknown_treated_as_free": False,
    }
    return free.astype(bool), metadata


def build_ipa_input_map(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    free_mask, metadata = build_accessible_free_mask(arrays)
    ipa_map = np.zeros(free_mask.shape, dtype=np.uint8)
    ipa_map[free_mask] = 255
    metadata.update(
        {
            "ipa_input_encoding": "mono8",
            "ipa_input_free_value": 255,
            "ipa_input_inaccessible_value": 0,
        }
    )
    return ipa_map, free_mask, metadata


def morphological_segment(
    free_mask: np.ndarray,
    config: MorphologicalFallbackConfig | None = None,
    *,
    resolution_m: float | None = None,
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    if config is not None and resolution_m is not None:
        raise ValueError("pass either config or resolution_m, not both")
    cfg = config or MorphologicalFallbackConfig(
        map_resolution_m=0.05 if resolution_m is None else float(resolution_m)
    )
    free = np.asarray(free_mask, dtype=bool)
    if free.ndim != 2:
        raise ValueError("free_mask must be 2D")

    erosion_structure = ndimage.generate_binary_structure(2, int(cfg.erosion_connectivity))
    component_structure = ndimage.generate_binary_structure(2, int(cfg.component_connectivity))
    min_area_cells, max_area_cells = cfg.area_limits_cells()
    working = free.copy()
    seed_labels = np.zeros(free.shape, dtype=np.int32)
    next_label = 1
    accepted_components = 0
    last_component_count = 0
    iterations_run = 0

    for iteration in range(max(0, int(cfg.max_iterations))):
        eroded = ndimage.binary_erosion(working, structure=erosion_structure, iterations=1, border_value=0)
        working = np.asarray(eroded, dtype=bool)
        components, component_count = ndimage.label(working, structure=component_structure)
        last_component_count = int(component_count)
        iterations_run = int(iteration + 1)
        if component_count == 0:
            break

        for component_id in range(1, int(component_count) + 1):
            component = components == component_id
            area_cells = int(np.count_nonzero(component))
            if min_area_cells <= area_cells <= max_area_cells:
                seed_labels[component] = int(next_label)
                working[component] = False
                next_label += 1
                accepted_components += 1

        if not np.any(working):
            break

    labels = wavefront_fill_unlabeled(seed_labels, domain=free, connectivity=int(cfg.wavefront_connectivity))
    labels = enforce_room_label_contract(labels, free)
    debug_arrays = {
        "morphological_free_mask": free.astype(np.uint8),
        "morphological_seed_label_map": seed_labels.astype(np.int32),
        "morphological_final_room_label_map": labels.astype(np.int32),
    }
    metadata = {
        "fallback_algorithm": "bormann_morphological_python",
        "fallback_approximates_original": True,
        "fallback_approximation_notes": (
            "Python smoke fallback mirrors IPA erosion, connected-component candidate "
            "selection, and wavefront fill, but uses scipy ndimage components instead of OpenCV contours."
        ),
        "ipa_original_algorithm_id": IPA_MORPHOLOGICAL_ALGORITHM_ID,
        "max_iterations": int(cfg.max_iterations),
        "iterations_run": int(iterations_run),
        "erosion_connectivity": int(cfg.erosion_connectivity),
        "component_connectivity": int(cfg.component_connectivity),
        "wavefront_connectivity": int(cfg.wavefront_connectivity),
        "room_area_lower_limit_m2": float(cfg.room_area_lower_limit_m2),
        "room_area_upper_limit_m2": float(cfg.room_area_upper_limit_m2),
        "room_area_lower_limit_cells": int(min_area_cells),
        "room_area_upper_limit_cells": int(max_area_cells),
        "accepted_seed_components": int(accepted_components),
        "last_eroded_component_count": int(last_component_count),
        "label_count": int(_positive_label_count(labels)),
    }
    return labels.astype(np.int32), metadata, debug_arrays


def segment_snapshot_morphological(
    arrays: Mapping[str, np.ndarray],
    config: MorphologicalFallbackConfig | None = None,
) -> MorphologicalSegmentationResult:
    cfg = config or MorphologicalFallbackConfig()
    _, free_mask, domain_metadata = build_ipa_input_map(arrays)
    labels, algorithm_metadata, debug_arrays = morphological_segment(free_mask, cfg)
    metadata: dict[str, object] = {
        "method": "morphological",
        "runner_type": "python_fallback",
        "main_experiment_allowed": False,
        "fallback_allowed_scope": "smoke_only",
        "uses_rgb": False,
        "uses_depth": False,
        "uses_occupancy": True,
        "uses_oracle_semantics": False,
        "original_repository": "https://github.com/ipa320/ipa_coverage_planning",
        "original_package": "ipa_room_segmentation",
        "original_action": "ipa_building_msgs/MapSegmentation.action",
        "original_algorithm_name": "MorphologicalSegmentation",
        "original_algorithm_id": IPA_MORPHOLOGICAL_ALGORITHM_ID,
        "unknown_treated_as_free": False,
        "final_room_label_map_dtype": "int32",
        "positive_labels_contiguous": True,
        "config": asdict(cfg),
    }
    metadata.update(domain_metadata)
    metadata.update(algorithm_metadata)
    return MorphologicalSegmentationResult(
        room_label_map=np.asarray(labels, dtype=np.int32),
        metadata=metadata,
        debug_arrays=debug_arrays,
    )


def wavefront_fill_unlabeled(labels: np.ndarray, *, domain: np.ndarray, connectivity: int = 2) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    free = np.asarray(domain, dtype=bool)
    if out.shape != free.shape:
        raise ValueError("labels and domain shapes must match")
    out[~free] = 0
    queue: deque[tuple[int, int]] = deque((int(r), int(c)) for r, c in zip(*np.nonzero(out > 0)))
    neighbors = _neighbor_offsets(connectivity)
    while queue:
        row, col = queue.popleft()
        label = int(out[row, col])
        if label <= 0:
            continue
        for dr, dc in neighbors:
            rr = row + dr
            cc = col + dc
            if rr < 0 or rr >= out.shape[0] or cc < 0 or cc >= out.shape[1]:
                continue
            if not free[rr, cc] or out[rr, cc] != 0:
                continue
            out[rr, cc] = label
            queue.append((rr, cc))
    return out


def enforce_room_label_contract(label_map: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int64)
    free = np.asarray(free_mask, dtype=bool)
    if labels.ndim != 2:
        raise ValueError("room label map must be 2D")
    if labels.shape != free.shape:
        raise ValueError("room label map shape %s does not match free mask shape %s" % (labels.shape, free.shape))
    out = np.zeros(labels.shape, dtype=np.int32)
    valid = free & (labels > 0) & (labels < IPA_UNASSIGNED_FREE_LABEL)
    next_label = 1
    for label in np.unique(labels[valid]):
        if int(label) <= 0:
            continue
        out[valid & (labels == int(label))] = int(next_label)
        next_label += 1
    return out


def validate_room_label_contract(label_map: np.ndarray, free_mask: np.ndarray | None = None) -> None:
    labels = np.asarray(label_map)
    if labels.dtype != np.int32:
        raise ValueError("final_room_label_map must be np.int32, got %s" % labels.dtype)
    if labels.ndim != 2:
        raise ValueError("final_room_label_map must be 2D")
    if np.any(labels < 0):
        raise ValueError("final_room_label_map contains negative labels")
    positives = sorted(int(v) for v in np.unique(labels) if int(v) > 0)
    expected = list(range(1, len(positives) + 1))
    if positives != expected:
        raise ValueError("positive labels must be contiguous 1..K, got %s" % positives)
    if free_mask is not None:
        free = np.asarray(free_mask, dtype=bool)
        if free.shape != labels.shape:
            raise ValueError("free_mask shape must match final_room_label_map")
        if np.any((labels > 0) & ~free):
            raise ValueError("final_room_label_map labels cells outside the free domain")


def _infer_shape(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    for key in (
        "final_room_label_map",
        "navigation_free_room_domain",
        "observed_free_mask",
        "occupancy_map",
        "unknown_mask",
        "voxel_nav_free_xy",
        "voxel_vertical_free_xy",
    ):
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim >= 2:
            return int(arr.shape[0]), int(arr.shape[1])
    raise KeyError("cannot infer snapshot map shape")


def _first_bool_array(
    arrays: Mapping[str, np.ndarray],
    keys: tuple[str, ...],
    shape: tuple[int, int],
) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.shape != shape:
            continue
        return arr.astype(bool), key
    return None, None


def _neighbor_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    if int(connectivity) <= 1:
        return ((-1, 0), (0, -1), (0, 1), (1, 0))
    return (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )


def _positive_label_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0]))
