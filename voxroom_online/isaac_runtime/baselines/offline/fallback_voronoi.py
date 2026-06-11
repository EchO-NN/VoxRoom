from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from voxroom_online.isaac_runtime.baselines.data_contract import resolve_map_info
from voxroom_online.isaac_runtime.baselines.mask_io import relabel_consecutive

from .fallback_utils import draw_grid_line, label_components, nearest_seed_fill


ORIGINAL_REPO = "https://github.com/ipa320/ipa_coverage_planning"
ORIGINAL_REPO_COMMIT = "986c18384ed884dadd3bc857cd0c47c13b7d4716"
ORIGINAL_PACKAGE = "ipa_room_segmentation"
ORIGINAL_ACTION = "ipa_building_msgs/MapSegmentation.action"
VORONOI_ALGORITHM_ID = 3


@dataclass(frozen=True)
class VoronoiFallbackParameters:
    map_resolution_m: float = 0.05
    room_area_factor_lower_limit_voronoi: float = 0.1
    room_area_factor_upper_limit_voronoi: float = 1000000.0
    voronoi_neighborhood_index: int = 280
    max_iterations: int = 150
    min_critical_point_distance_factor: float = 0.5
    merge_area_threshold_m2: float = 12.5
    force_merge_area_m2: float = 2.0
    min_critical_line_angle_deg: float = 95.0
    skeleton_min_distance_cells: float = 1.0
    prune_iterations: int = 30


@dataclass(frozen=True)
class VoronoiFallbackStats:
    free_cells: int
    ridge_cells: int
    pruned_ridge_cells: int
    critical_candidates: int
    critical_lines_drawn: int
    initial_region_count: int
    merged_region_count: int
    final_room_count: int


@dataclass(frozen=True)
class CriticalLine:
    point_rc: tuple[int, int]
    basis_a_rc: tuple[int, int]
    basis_b_rc: tuple[int, int]
    angle_deg: float
    length_cells: float


def build_voronoi_free_mask(arrays: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the IPA/Voronoi accessible domain; unknown is never marked free."""

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
        require_any=True,
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


def build_voronoi_ipa_input_image(arrays: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    free_mask, metadata = build_voronoi_free_mask(arrays)
    ipa_image = np.zeros(free_mask.shape, dtype=np.uint8)
    ipa_image[free_mask] = np.uint8(255)
    metadata.update(
        {
            "ipa_input_encoding": "mono8",
            "ipa_input_free_value": 255,
            "ipa_input_inaccessible_value": 0,
        }
    )
    return ipa_image, free_mask, metadata


def segment_snapshot_arrays(
    snapshot_path: Path | str,
    arrays: Mapping[str, Any],
    *,
    default_resolution_m: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Run the smoke-only Python Voronoi fallback on one replay snapshot."""

    snapshot_path = Path(snapshot_path)
    map_info = resolve_map_info(snapshot_arrays=arrays, default_resolution_m=float(default_resolution_m))
    free_mask, domain_metadata = build_voronoi_free_mask(arrays)
    params = VoronoiFallbackParameters(map_resolution_m=float(map_info.resolution_m))
    labels, algorithm_metadata, debug_arrays = voronoi_segment(
        free_mask,
        resolution_m=float(map_info.resolution_m),
        merge_area_threshold_m2=float(params.merge_area_threshold_m2),
        force_merge_area_m2=float(params.force_merge_area_m2),
        room_area_factor_lower_limit_voronoi=float(params.room_area_factor_lower_limit_voronoi),
        room_area_factor_upper_limit_voronoi=float(params.room_area_factor_upper_limit_voronoi),
        voronoi_neighborhood_index=int(params.voronoi_neighborhood_index),
        max_iterations=int(params.max_iterations),
        min_critical_point_distance_factor=float(params.min_critical_point_distance_factor),
        min_critical_line_angle_deg=float(params.min_critical_line_angle_deg),
        skeleton_min_distance_cells=float(params.skeleton_min_distance_cells),
        prune_iterations=int(params.prune_iterations),
    )
    metadata = build_fallback_metadata(
        source_snapshot=snapshot_path,
        input_free_definition=str(domain_metadata["free_source"]),
        map_resolution_m=float(map_info.resolution_m),
        map_origin_xy_m=(float(map_info.min_x), float(map_info.min_y)),
        parameters=params,
        algorithm_metadata=algorithm_metadata,
        domain_metadata=domain_metadata,
    )
    return np.asarray(labels, dtype=np.int32), metadata, debug_arrays


def voronoi_segment(
    free_mask: np.ndarray,
    *,
    resolution_m: float = 0.05,
    merge_area_threshold_m2: float = 12.5,
    force_merge_area_m2: float = 2.0,
    room_area_factor_lower_limit_voronoi: float = 0.1,
    room_area_factor_upper_limit_voronoi: float = 1000000.0,
    voronoi_neighborhood_index: int = 280,
    max_iterations: int = 150,
    min_critical_point_distance_factor: float = 0.5,
    min_critical_line_angle_deg: float = 95.0,
    skeleton_min_distance_cells: float = 1.0,
    prune_iterations: int = 30,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Approximate Bormann Voronoi segmentation for smoke tests only.

    The main experiment path must call ipa_room_segmentation. This fallback keeps
    the same high-level stages: generalized Voronoi skeleton, critical points,
    critical lines, connected regions, and small-region merge heuristics.
    """

    free = np.asarray(free_mask, dtype=bool)
    if free.ndim != 2:
        raise ValueError(f"free_mask must be 2D, got shape={free.shape}")
    if float(resolution_m) <= 0:
        raise ValueError("resolution_m must be positive")

    params = VoronoiFallbackParameters(
        map_resolution_m=float(resolution_m),
        room_area_factor_lower_limit_voronoi=float(room_area_factor_lower_limit_voronoi),
        room_area_factor_upper_limit_voronoi=float(room_area_factor_upper_limit_voronoi),
        voronoi_neighborhood_index=int(voronoi_neighborhood_index),
        max_iterations=int(max_iterations),
        min_critical_point_distance_factor=float(min_critical_point_distance_factor),
        merge_area_threshold_m2=float(merge_area_threshold_m2),
        force_merge_area_m2=float(force_merge_area_m2),
        min_critical_line_angle_deg=float(min_critical_line_angle_deg),
        skeleton_min_distance_cells=float(skeleton_min_distance_cells),
        prune_iterations=int(prune_iterations),
    )
    if not bool(np.any(free)):
        empty = np.zeros(free.shape, dtype=np.int32)
        stats = VoronoiFallbackStats(0, 0, 0, 0, 0, 0, 0, 0)
        return empty, _algorithm_metadata(params, stats, "empty_free_domain"), {}

    distance_cells = ndimage.distance_transform_edt(free).astype(np.float32)
    ridge = _approximate_voronoi_ridge(
        distance_cells,
        free,
        min_distance_cells=float(params.skeleton_min_distance_cells),
    )
    pruned_ridge = _prune_skeleton_endpoints(ridge, iterations=min(int(params.prune_iterations), int(params.max_iterations)))
    critical_candidates = _critical_narrow_points(
        distance_cells,
        pruned_ridge,
        neighborhood_index=int(params.voronoi_neighborhood_index),
        max_iterations=int(params.max_iterations),
    )
    critical_lines = _select_critical_lines(
        free,
        distance_cells,
        critical_candidates,
        min_angle_deg=float(params.min_critical_line_angle_deg),
        min_distance_factor=float(params.min_critical_point_distance_factor),
    )

    separator = np.zeros(free.shape, dtype=bool)
    for line in critical_lines:
        r, c = line.point_rc
        draw_grid_line(separator, (r, c), line.basis_a_rc)
        draw_grid_line(separator, (r, c), line.basis_b_rc)
    separator &= free

    split_domain = free & ~separator
    component_labels, initial_count = label_components(split_domain, connectivity=4)
    if int(initial_count) == 0:
        initial_labels = np.zeros(free.shape, dtype=np.int32)
    else:
        initial_labels = nearest_seed_fill(component_labels, domain=free)
    initial_labels = _filter_area_limits(
        initial_labels,
        free,
        resolution_m=float(params.map_resolution_m),
        lower_m2=float(params.room_area_factor_lower_limit_voronoi),
        upper_m2=float(params.room_area_factor_upper_limit_voronoi),
    )
    if not bool(np.any(initial_labels > 0)):
        initial_labels, initial_count = label_components(free, connectivity=4)

    merged_labels, merge_count = _merge_small_regions(
        initial_labels,
        free,
        resolution_m=float(params.map_resolution_m),
        merge_area_threshold_m2=float(params.merge_area_threshold_m2),
        force_merge_area_m2=float(params.force_merge_area_m2),
    )
    labels = relabel_consecutive(merged_labels)
    labels[~free] = 0
    stats = VoronoiFallbackStats(
        free_cells=int(np.count_nonzero(free)),
        ridge_cells=int(np.count_nonzero(ridge)),
        pruned_ridge_cells=int(np.count_nonzero(pruned_ridge)),
        critical_candidates=int(np.count_nonzero(critical_candidates)),
        critical_lines_drawn=int(len(critical_lines)),
        initial_region_count=int(initial_count),
        merged_region_count=int(merge_count),
        final_room_count=int(labels.max()),
    )
    metadata = _algorithm_metadata(params, stats, "critical_line_split")
    debug = {
        "voronoi_metric_domain": free.astype(bool),
        "voronoi_distance_transform_cells": distance_cells.astype(np.float32),
        "voronoi_ridge_mask": ridge.astype(bool),
        "voronoi_pruned_ridge_mask": pruned_ridge.astype(bool),
        "voronoi_critical_point_mask": critical_candidates.astype(bool),
        "voronoi_separator_mask": separator.astype(bool),
        "voronoi_initial_label_map": np.asarray(initial_labels, dtype=np.int32),
    }
    return labels.astype(np.int32), metadata, debug


def build_fallback_metadata(
    *,
    source_snapshot: Path | str | None,
    input_free_definition: str,
    map_resolution_m: float,
    map_origin_xy_m: tuple[float, float],
    parameters: VoronoiFallbackParameters,
    algorithm_metadata: Mapping[str, Any],
    domain_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "method": "voronoi",
        "source_snapshot": "" if source_snapshot is None else str(source_snapshot),
        "input_free_definition": str(input_free_definition),
        "unknown_treated_as": "occupied/inaccessible",
        "unknown_treated_as_free": False,
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
        "original_algorithm_id": VORONOI_ALGORITHM_ID,
        "original_algorithm_name": "VoronoiSegmentation",
        "parameters": asdict(parameters),
    }
    metadata.update(dict(domain_metadata))
    metadata.update(dict(algorithm_metadata))
    return metadata


def _algorithm_metadata(
    params: VoronoiFallbackParameters,
    stats: VoronoiFallbackStats,
    reason: str,
) -> dict[str, Any]:
    return {
        "fallback_algorithm": "bormann_voronoi_python",
        "fallback_approximates_original": True,
        "fallback_approximation_notes": (
            "Smoke-test fallback approximates IPA Voronoi with scipy distance ridges, "
            "nearest-obstacle critical lines, and simple small-region merge heuristics. "
            "Main experiments must use ipa_room_segmentation."
        ),
        "reason": str(reason),
        "runner_type": "python_fallback",
        "main_experiment_allowed": False,
        "critical_point_rule": "pruned_distance_ridge_local_minimum_with_two_closest_obstacles",
        "critical_line_rule": "draw_lines_from_critical_point_to_two_nearest_obstacle_pixels",
        "critical_line_min_angle_deg": float(params.min_critical_line_angle_deg),
        "merge_area_threshold_m2": float(params.merge_area_threshold_m2),
        "force_merge_area_m2": float(params.force_merge_area_m2),
        "voronoi_neighborhood_index": int(params.voronoi_neighborhood_index),
        "max_iterations": int(params.max_iterations),
        "min_critical_point_distance_factor": float(params.min_critical_point_distance_factor),
        "room_area_factor_lower_limit_voronoi": float(params.room_area_factor_lower_limit_voronoi),
        "room_area_factor_upper_limit_voronoi": float(params.room_area_factor_upper_limit_voronoi),
        "stats": asdict(stats),
    }


def _approximate_voronoi_ridge(dist: np.ndarray, free: np.ndarray, *, min_distance_cells: float) -> np.ndarray:
    maxed = ndimage.maximum_filter(dist, size=3, mode="nearest")
    ridge = np.asarray(free, dtype=bool) & (dist >= maxed - 1.0e-6) & (dist > float(min_distance_cells))
    if bool(np.any(ridge)):
        return ridge
    return np.asarray(free, dtype=bool) & (dist > float(min_distance_cells))


def _prune_skeleton_endpoints(skeleton: np.ndarray, *, iterations: int) -> np.ndarray:
    out = np.asarray(skeleton, dtype=bool).copy()
    if not bool(np.any(out)):
        return out
    kernel = np.ones((3, 3), dtype=np.int16)
    for _ in range(max(0, int(iterations))):
        neighbor_count = ndimage.convolve(out.astype(np.int16), kernel, mode="constant", cval=0) - out.astype(np.int16)
        endpoints = out & (neighbor_count <= 1)
        if not bool(np.any(endpoints)):
            break
        nodes = out & (neighbor_count >= 3)
        remove = endpoints & ~nodes
        if not bool(np.any(remove)):
            break
        out[remove] = False
    return out


def _critical_narrow_points(
    dist: np.ndarray,
    ridge: np.ndarray,
    *,
    neighborhood_index: int,
    max_iterations: int,
) -> np.ndarray:
    if not bool(np.any(ridge)):
        return np.zeros(ridge.shape, dtype=bool)
    components, _ = label_components(ridge, connectivity=8)
    out = np.zeros(ridge.shape, dtype=bool)
    for component_id in [int(v) for v in np.unique(components) if int(v) > 0]:
        coords = np.argwhere(components == component_id)
        if coords.size == 0:
            continue
        component_dist = dist[components == component_id]
        dynamic_eps = max(3, int(min(max_iterations, max(1, neighborhood_index) / max(float(np.median(component_dist)), 1.0))))
        stride = max(1, dynamic_eps // 2)
        order = np.argsort(component_dist)
        selected: list[tuple[int, int]] = []
        for idx in order.tolist():
            r, c = (int(coords[idx, 0]), int(coords[idx, 1]))
            if any((r - sr) * (r - sr) + (c - sc) * (c - sc) <= stride * stride for sr, sc in selected):
                continue
            selected.append((r, c))
            if len(selected) >= max(1, len(coords) // max(dynamic_eps, 1) + 1):
                break
        for r, c in selected:
            out[r, c] = True
    return out


def _select_critical_lines(
    free: np.ndarray,
    dist: np.ndarray,
    critical_points: np.ndarray,
    *,
    min_angle_deg: float,
    min_distance_factor: float,
) -> list[CriticalLine]:
    candidates: list[CriticalLine] = []
    for r, c in np.argwhere(critical_points):
        basis = _nearest_two_obstacle_pixels(free, int(r), int(c), distance_cells=float(dist[int(r), int(c)]))
        if basis is None:
            continue
        a, b = basis
        angle = _angle_deg((int(r), int(c)), a, b)
        if angle < float(min_angle_deg):
            continue
        length = float(np.hypot(r - a[0], c - a[1]) + np.hypot(r - b[0], c - b[1]))
        candidates.append(CriticalLine((int(r), int(c)), a, b, float(angle), length))

    selected: list[CriticalLine] = []
    for line in sorted(candidates, key=lambda item: (-item.angle_deg, item.length_cells)):
        min_sep = max(1.0, float(dist[line.point_rc]) * float(min_distance_factor))
        if any(_point_distance(line.point_rc, other.point_rc) < min_sep for other in selected):
            continue
        selected.append(line)
    return selected


def _nearest_two_obstacle_pixels(
    free: np.ndarray,
    r: int,
    c: int,
    *,
    distance_cells: float,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    obstacle = ~np.asarray(free, dtype=bool)
    h, w = obstacle.shape
    base_radius = max(3, int(np.ceil(float(distance_cells))) + 3)
    max_radius = max(h, w)
    for radius in (base_radius, base_radius * 2, base_radius * 4, max_radius):
        r0 = max(0, r - radius)
        r1 = min(h, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(w, c + radius + 1)
        coords = np.argwhere(obstacle[r0:r1, c0:c1])
        if coords.shape[0] < 2:
            continue
        coords[:, 0] += r0
        coords[:, 1] += c0
        deltas = coords.astype(np.float32) - np.asarray([[r, c]], dtype=np.float32)
        distances = np.sum(deltas * deltas, axis=1)
        order = np.argsort(distances)
        first = (int(coords[int(order[0]), 0]), int(coords[int(order[0]), 1]))
        min_basis_distance = max(1.0, float(distance_cells))
        for idx in order[1:].tolist():
            second = (int(coords[int(idx), 0]), int(coords[int(idx), 1]))
            if _point_distance(first, second) >= min_basis_distance:
                return first, second
    return None


def _filter_area_limits(
    labels: np.ndarray,
    domain: np.ndarray,
    *,
    resolution_m: float,
    lower_m2: float,
    upper_m2: float,
) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    out[~np.asarray(domain, dtype=bool)] = 0
    cell_area = float(resolution_m) * float(resolution_m)
    for label in [int(v) for v in np.unique(out) if int(v) > 0]:
        area_m2 = float(np.count_nonzero(out == label)) * cell_area
        if area_m2 < float(lower_m2) or area_m2 > float(upper_m2):
            out[out == label] = 0
    if bool(np.any(out > 0)):
        out = nearest_seed_fill(out, domain=domain)
    return relabel_consecutive(out)


def _merge_small_regions(
    labels: np.ndarray,
    domain: np.ndarray,
    *,
    resolution_m: float,
    merge_area_threshold_m2: float,
    force_merge_area_m2: float,
) -> tuple[np.ndarray, int]:
    out = relabel_consecutive(labels)
    out[~np.asarray(domain, dtype=bool)] = 0
    merges = 0
    cell_area = float(resolution_m) * float(resolution_m)
    for _ in range(1000):
        changed = False
        adjacency = _label_adjacency(out)
        perimeters = _label_perimeters(out)
        areas = {int(label): float(np.count_nonzero(out == label)) * cell_area for label in np.unique(out) if int(label) > 0}
        for label, area_m2 in sorted(areas.items(), key=lambda item: item[1]):
            all_neighbors = adjacency.get(int(label), {})
            neighbors = {int(k): int(v) for k, v in all_neighbors.items() if int(k) > 0}
            if not neighbors:
                continue
            perimeter = max(1, int(perimeters.get(int(label), 1)))
            target = max(neighbors, key=lambda neighbor: int(neighbors[neighbor]))
            shared_ratio = float(neighbors[target]) / float(perimeter)
            wall_ratio = float(all_neighbors.get(0, 0)) / float(perimeter)
            force_merge = area_m2 < float(force_merge_area_m2)
            heuristic_merge = (
                area_m2 < float(merge_area_threshold_m2)
                and (shared_ratio > 0.2 or (len(neighbors) <= 1 and wall_ratio <= 0.75))
            )
            if force_merge or heuristic_merge:
                out[out == int(label)] = int(target)
                out = relabel_consecutive(out)
                merges += 1
                changed = True
                break
        if not changed:
            break
    out[~np.asarray(domain, dtype=bool)] = 0
    return relabel_consecutive(out), int(merges)


def _label_adjacency(labels: np.ndarray) -> dict[int, dict[int, int]]:
    arr = np.asarray(labels, dtype=np.int32)
    adjacency: dict[int, dict[int, int]] = {}
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            label = int(arr[r, c])
            if label <= 0:
                continue
            adjacency.setdefault(label, {})
            for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                nr = r + dr
                nc = c + dc
                neighbor = 0 if nr < 0 or nc < 0 or nr >= h or nc >= w else int(arr[nr, nc])
                if neighbor == label:
                    continue
                adjacency[label][neighbor] = adjacency[label].get(neighbor, 0) + 1
    return adjacency


def _label_perimeters(labels: np.ndarray) -> dict[int, int]:
    return {label: int(sum(counts.values())) for label, counts in _label_adjacency(labels).items()}


def _angle_deg(origin: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> float:
    va = np.asarray([a[0] - origin[0], a[1] - origin[1]], dtype=np.float64)
    vb = np.asarray([b[0] - origin[0], b[1] - origin[1]], dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0:
        return 0.0
    cosine = float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _point_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _infer_shape(arrays: Mapping[str, Any]) -> tuple[int, int]:
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
    arrays: Mapping[str, Any],
    keys: tuple[str, ...],
    shape: tuple[int, int],
    *,
    require_any: bool = False,
) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.shape != shape:
            continue
        out = arr.astype(bool)
        if require_any and not bool(np.any(out)):
            continue
        return out, key
    return None, None


def _smoke_arrays() -> dict[str, np.ndarray]:
    shape = (36, 64)
    observed = np.zeros(shape, dtype=bool)
    observed[6:30, 6:25] = True
    observed[6:30, 39:58] = True
    observed[15:21, 25:39] = True
    unknown = np.zeros(shape, dtype=bool)
    unknown[15:21, 30:34] = True
    obstacle = ~observed
    return {
        "occupancy_map": obstacle,
        "observed_free_mask": observed,
        "obstacle_mask": obstacle,
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
    free, _ = build_voronoi_free_mask(arrays)
    assert labels.shape == free.shape
    assert labels.dtype == np.int32
    assert not np.any(labels[unknown] > 0)
    assert not np.any(labels[~free] > 0)
    assert int(metadata["original_algorithm_id"]) == VORONOI_ALGORITHM_ID
    assert metadata["runner_type"] == "python_fallback"
    assert metadata["main_experiment_allowed"] is False
    assert "voronoi_separator_mask" in debug
    print(
        json.dumps(
            {
                "labels": sorted(int(v) for v in np.unique(labels) if int(v) > 0),
                "metadata_runner_type": metadata["runner_type"],
                "metadata_fallback_scope": metadata["fallback_scope"],
                "main_experiment_allowed": metadata["main_experiment_allowed"],
                "critical_line_min_angle_deg": metadata["critical_line_min_angle_deg"],
                "merge_area_threshold_m2": metadata["merge_area_threshold_m2"],
                "force_merge_area_m2": metadata["force_merge_area_m2"],
                "unknown_labeled_cells": int(np.count_nonzero(labels[unknown] > 0)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-only Python fallback for Bormann Voronoi segmentation.")
    parser.add_argument("--smoke", action="store_true", help="Run a toy-map smoke test.")
    args = parser.parse_args(argv)
    if args.smoke:
        return _run_smoke()
    parser.error("fallback_voronoi.py is smoke-only; pass --smoke or call voronoi_segment() from tests.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
