from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from voxroom_online.isaac_runtime.baselines.data_contract import resolve_map_info
from voxroom_online.isaac_runtime.baselines.mask_io import build_metric_domain_from_source, relabel_consecutive


ORIGINAL_REPO = "https://github.com/ipa320/ipa_coverage_planning"
ORIGINAL_REPO_COMMIT = "986c18384ed884dadd3bc857cd0c47c13b7d4716"
ORIGINAL_PACKAGE = "ipa_room_segmentation"
ORIGINAL_ACTION = "ipa_building_msgs/MapSegmentation.action"
DISTANCE_ALGORITHM_ID = 2


@dataclass(frozen=True)
class DistanceFallbackParameters:
    map_resolution_m: float = 0.05
    room_area_factor_lower_limit_distance: float = 0.35
    room_area_factor_upper_limit_distance: float = 163.0
    erode_iterations: int = 1
    connectivity: int = 8


@dataclass(frozen=True)
class DistanceFallbackStats:
    threshold: int
    seed_count: int
    orphan_seed_count: int
    assigned_cells: int
    domain_cells: int


def distance_segment(
    free_mask: np.ndarray,
    *,
    resolution_m: float = 0.05,
    room_area_factor_lower_limit_distance: float = 0.35,
    room_area_factor_upper_limit_distance: float = 163.0,
    erode_iterations: int = 1,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray], DistanceFallbackStats]:
    """Segment a binary free-space domain with a Bormann-style distance fallback.

    This is a smoke-test fallback for the original IPA implementation. The input
    must already exclude unknown space; all seed selection and wavefront filling
    are clipped to this domain.
    """
    domain = np.asarray(free_mask, dtype=bool)
    if domain.ndim != 2:
        raise ValueError(f"free_mask must be 2D, got shape={domain.shape}")
    params = DistanceFallbackParameters(
        map_resolution_m=float(resolution_m),
        room_area_factor_lower_limit_distance=float(room_area_factor_lower_limit_distance),
        room_area_factor_upper_limit_distance=float(room_area_factor_upper_limit_distance),
        erode_iterations=int(erode_iterations),
    )
    if params.map_resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    if params.room_area_factor_lower_limit_distance < 0:
        raise ValueError("lower area limit must be non-negative")
    if params.room_area_factor_upper_limit_distance < params.room_area_factor_lower_limit_distance:
        raise ValueError("upper area limit must be >= lower area limit")

    eroded_domain = _erode_free_domain(domain, iterations=params.erode_iterations)
    distance_cells = ndimage.distance_transform_edt(eroded_domain).astype(np.float32)
    distance_u8 = np.clip(np.rint(distance_cells), 0, 255).astype(np.uint8)
    seed_labels, selected_threshold = _select_seed_labels(distance_u8, eroded_domain, params)
    seed_labels, orphan_seed_count = _ensure_seed_per_domain_component(seed_labels, domain, distance_cells)
    filled = _wavefront_fill_unlabeled(seed_labels, domain=domain, distance_cells=distance_cells)
    filled = relabel_consecutive(filled)
    filled[~domain] = 0
    stats = DistanceFallbackStats(
        threshold=int(selected_threshold),
        seed_count=int(seed_labels.max()),
        orphan_seed_count=int(orphan_seed_count),
        assigned_cells=int(np.count_nonzero(filled)),
        domain_cells=int(np.count_nonzero(domain)),
    )
    debug = {
        "distance_transform_cells": distance_cells.astype(np.float32),
        "distance_transform_u8": distance_u8.astype(np.uint8),
        "distance_eroded_free_mask": eroded_domain.astype(bool),
        "distance_initial_seed_labels": seed_labels.astype(np.int32),
        "distance_metric_domain": domain.astype(bool),
    }
    if return_debug:
        return filled.astype(np.int32, copy=False), debug, stats
    return filled.astype(np.int32, copy=False)


def segment_snapshot_arrays(
    snapshot_path: Path | str,
    arrays: Mapping[str, Any],
    *,
    default_resolution_m: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Run the smoke-test Python fallback on one replay snapshot."""
    snapshot_path = Path(snapshot_path)
    domain = build_metric_domain_from_source(arrays)
    map_info = resolve_map_info(snapshot_arrays=arrays, default_resolution_m=float(default_resolution_m))
    labels, debug, stats = distance_segment(domain, resolution_m=map_info.resolution_m, return_debug=True)
    metadata = build_fallback_metadata(
        source_snapshot=snapshot_path,
        input_free_definition=_input_free_definition(arrays),
        map_resolution_m=map_info.resolution_m,
        map_origin_xy_m=(map_info.min_x, map_info.min_y),
        parameters=DistanceFallbackParameters(map_resolution_m=map_info.resolution_m),
        stats=stats,
    )
    return labels, metadata, debug


def build_fallback_metadata(
    *,
    source_snapshot: Path | str | None,
    input_free_definition: str,
    map_resolution_m: float,
    map_origin_xy_m: tuple[float, float],
    parameters: DistanceFallbackParameters,
    stats: DistanceFallbackStats,
) -> dict[str, Any]:
    return {
        "method": "distance_transform",
        "source_snapshot": "" if source_snapshot is None else str(source_snapshot),
        "input_free_definition": str(input_free_definition),
        "unknown_treated_as": "occupied/inaccessible",
        "map_resolution_m": float(map_resolution_m),
        "map_origin_xy_m": [float(map_origin_xy_m[0]), float(map_origin_xy_m[1])],
        "uses_rgb": False,
        "uses_depth": False,
        "uses_oracle_semantics": False,
        "runner_type": "python_fallback",
        "fallback_scope": "smoke_only_not_main_experiment",
        "main_experiment_allowed": False,
        "original_repo": ORIGINAL_REPO,
        "original_repo_commit": ORIGINAL_REPO_COMMIT,
        "original_package": ORIGINAL_PACKAGE,
        "original_action": ORIGINAL_ACTION,
        "original_algorithm_id": DISTANCE_ALGORITHM_ID,
        "parameters": asdict(parameters),
        "stats": asdict(stats),
    }


def _erode_free_domain(domain: np.ndarray, *, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return np.asarray(domain, dtype=bool).copy()
    structure = np.ones((3, 3), dtype=bool)
    return ndimage.binary_erosion(np.asarray(domain, dtype=bool), structure=structure, iterations=int(iterations), border_value=0)


def _select_seed_labels(
    distance_u8: np.ndarray,
    eroded_domain: np.ndarray,
    params: DistanceFallbackParameters,
) -> tuple[np.ndarray, int]:
    thresholds = [int(v) for v in np.unique(distance_u8[eroded_domain]) if int(v) > 0]
    labels = np.zeros(distance_u8.shape, dtype=np.int32)
    best_labels = labels.copy()
    best_threshold = 0
    best_score = -1
    structure = np.ones((3, 3), dtype=bool)
    for threshold in sorted(thresholds, reverse=True):
        candidate = eroded_domain & (distance_u8 >= threshold)
        cc, count = ndimage.label(candidate, structure=structure)
        current = np.zeros(distance_u8.shape, dtype=np.int32)
        next_id = 1
        for component_id in range(1, int(count) + 1):
            component = cc == component_id
            area_m2 = float(np.count_nonzero(component)) * params.map_resolution_m * params.map_resolution_m
            if (
                area_m2 >= params.room_area_factor_lower_limit_distance
                and area_m2 <= params.room_area_factor_upper_limit_distance
            ):
                current[component] = next_id
                next_id += 1
        score = next_id - 1
        if score >= best_score:
            best_score = score
            best_threshold = int(threshold)
            best_labels = current
    best_labels[~eroded_domain] = 0
    return best_labels, best_threshold


def _ensure_seed_per_domain_component(
    seed_labels: np.ndarray,
    domain: np.ndarray,
    distance_cells: np.ndarray,
) -> tuple[np.ndarray, int]:
    labels = np.asarray(seed_labels, dtype=np.int32).copy()
    labels[~domain] = 0
    structure = np.ones((3, 3), dtype=bool)
    cc, count = ndimage.label(domain, structure=structure)
    next_id = int(labels.max()) + 1
    orphan_count = 0
    for component_id in range(1, int(count) + 1):
        component = cc == component_id
        if np.any(labels[component] > 0):
            continue
        rows, cols = np.nonzero(component)
        if rows.size == 0:
            continue
        distances = distance_cells[rows, cols]
        index = int(np.argmax(distances))
        labels[int(rows[index]), int(cols[index])] = next_id
        next_id += 1
        orphan_count += 1
    return labels, orphan_count


def _wavefront_fill_unlabeled(
    labels: np.ndarray,
    *,
    domain: np.ndarray,
    distance_cells: np.ndarray,
) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    metric_domain = np.asarray(domain, dtype=bool)
    out[~metric_domain] = 0
    heap: list[tuple[float, int, int, int, int]] = []
    order = 0
    seed_rows, seed_cols = np.nonzero((out > 0) & metric_domain)
    for r, c in zip(seed_rows.tolist(), seed_cols.tolist(), strict=False):
        heapq.heappush(heap, (-float(distance_cells[r, c]), order, int(r), int(c), int(out[r, c])))
        order += 1
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    rows, cols = out.shape
    while heap:
        _, _, r, c, label = heapq.heappop(heap)
        if out[r, c] != label:
            continue
        for dr, dc in neighbors:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                continue
            if not metric_domain[nr, nc] or out[nr, nc] != 0:
                continue
            out[nr, nc] = label
            heapq.heappush(heap, (-float(distance_cells[nr, nc]), order, nr, nc, label))
            order += 1
    out[~metric_domain] = 0
    return out


def _input_free_definition(arrays: Mapping[str, Any]) -> str:
    nav = arrays.get("navigation_free_room_domain")
    if nav is not None:
        arr = np.asarray(nav, dtype=bool)
        if arr.ndim == 2 and bool(arr.any()):
            return "navigation_free_room_domain"
    return "observed_free_minus_obstacle_unknown"


def _smoke_arrays() -> dict[str, np.ndarray]:
    shape = (28, 48)
    observed = np.zeros(shape, dtype=bool)
    observed[5:23, 5:21] = True
    observed[5:23, 27:43] = True
    observed[10:18, 21:27] = True
    unknown = np.zeros(shape, dtype=bool)
    unknown[10:18, 21:27] = True
    occupancy = np.zeros(shape, dtype=bool)
    return {
        "occupancy_map": occupancy,
        "observed_free_mask": observed,
        "obstacle_mask": occupancy,
        "unknown_mask": unknown,
        "final_room_label_map": np.zeros(shape, dtype=np.int32),
        "map_resolution_m": np.asarray(0.05, dtype=np.float32),
        "map_origin_xy_m": np.asarray([0.0, 0.0], dtype=np.float32),
        "map_width_cells": np.asarray(shape[1], dtype=np.int32),
        "map_height_cells": np.asarray(shape[0], dtype=np.int32),
    }


def _run_smoke() -> int:
    arrays = _smoke_arrays()
    labels, metadata, debug = segment_snapshot_arrays("smoke/roomseg_step_000001.npz", arrays)
    unknown = np.asarray(arrays["unknown_mask"], dtype=bool)
    domain = build_metric_domain_from_source(arrays)
    assert labels.shape == domain.shape
    assert not np.any(labels[unknown] > 0)
    assert not np.any(labels[~domain] > 0)
    assert np.count_nonzero(labels[domain]) == int(np.count_nonzero(domain))
    assert np.all(debug["distance_initial_seed_labels"][~domain] == 0)
    print(
        json.dumps(
            {
                "labels": sorted(int(x) for x in np.unique(labels) if int(x) > 0),
                "metadata_runner_type": metadata["runner_type"],
                "metadata_fallback_scope": metadata["fallback_scope"],
                "unknown_labeled_cells": int(np.count_nonzero(labels[unknown])),
                "assigned_domain_cells": int(np.count_nonzero(labels[domain])),
                "domain_cells": int(np.count_nonzero(domain)),
                "seed_count": int(metadata["stats"]["seed_count"]),
                "orphan_seed_count": int(metadata["stats"]["orphan_seed_count"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-only Python fallback for Bormann distance transform segmentation.")
    parser.add_argument("--smoke", action="store_true", help="Run an asymmetric toy-map smoke test.")
    args = parser.parse_args(argv)
    if args.smoke:
        return _run_smoke()
    parser.error("fallback_distance.py is smoke-only; pass --smoke or call distance_segment() from tests.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
