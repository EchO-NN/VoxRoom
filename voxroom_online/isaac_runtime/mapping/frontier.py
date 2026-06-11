from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo, grid_to_world_xy
from voxroom_online.isaac_runtime.navigation.astar import astar_distance_map

GridCell = Tuple[int, int]


@dataclass
class FrontierCluster:
    center_grid: GridCell
    center_world: Tuple[float, float]
    members: List[GridCell]
    size: int
    path_distance_from_agent: float
    min_path_distance: float = math.nan
    mean_path_distance: float = math.nan
    max_path_distance: float = math.nan
    center_path_distance: float = math.nan
    distance_inverse: float = math.nan
    is_near_fallback: bool = False

    def __post_init__(self) -> None:
        dist = float(self.path_distance_from_agent)
        if not math.isfinite(float(self.min_path_distance)):
            self.min_path_distance = dist
        if not math.isfinite(float(self.mean_path_distance)):
            self.mean_path_distance = dist
        if not math.isfinite(float(self.max_path_distance)):
            self.max_path_distance = dist
        if not math.isfinite(float(self.center_path_distance)):
            self.center_path_distance = dist
        if not math.isfinite(float(self.distance_inverse)):
            self.distance_inverse = _distance_inverse(dist)


def _disk_dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    mask_bool = np.asarray(mask).astype(bool)
    radius = max(0, int(radius_cells))
    if radius <= 0 or not np.any(mask_bool):
        return mask_bool
    try:
        import skimage.morphology

        return skimage.morphology.binary_dilation(mask_bool, skimage.morphology.disk(radius)).astype(bool)
    except Exception:
        out = np.array(mask_bool, copy=True)
        rows, cols = np.nonzero(mask_bool)
        h, w = mask_bool.shape
        offsets = [
            (dr, dc)
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if dr * dr + dc * dc <= radius * radius
        ]
        for row, col in zip(rows, cols):
            for dr, dc in offsets:
                rr, cc = int(row + dr), int(col + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
        return out


def _distance_inverse(dist_m: float, min_distance_m: float = 1.0, span_m: float = 10.0) -> float:
    if not np.isfinite(float(dist_m)):
        return 0.0
    min_d = float(min_distance_m)
    span = max(float(span_m), 1e-6)
    return float(1.0 - (np.clip(float(dist_m), min_d, min_d + span) - min_d) / span)


def frontier_debug_layers(
    free: np.ndarray,
    observed: Optional[np.ndarray] = None,
    occupancy: Optional[np.ndarray] = None,
    obstacle_dilation_radius_cells: int = 4,
    unknown_dilation_radius_cells: int = 1,
    exclude_mask: Optional[np.ndarray] = None,
    unknown_source: str = "observed",
) -> Dict[str, np.ndarray]:
    """Return frontier-based exploration masks used to build frontier cells."""
    free_bool = np.asarray(free).astype(bool)
    occ_bool = np.zeros_like(free_bool, dtype=bool) if occupancy is None else np.asarray(occupancy).astype(bool)
    if occ_bool.shape != free_bool.shape:
        raise ValueError("occupancy and free must have the same shape")
    if unknown_source not in {"observed", "implicit"}:
        raise ValueError(f"unknown_source must be 'observed' or 'implicit', got {unknown_source!r}")

    fbe_map = np.zeros_like(free_bool, dtype=np.int8)
    fbe_map[free_bool] = 1
    dilated_obstacles = _disk_dilate(occ_bool, obstacle_dilation_radius_cells)
    fbe_map[dilated_obstacles] = 3

    if unknown_source == "observed" and observed is not None:
        obs_bool = np.asarray(observed).astype(bool)
        if obs_bool.shape != free_bool.shape:
            raise ValueError("observed and free must have the same shape")
        unknown = (~obs_bool) & (~free_bool) & (~occ_bool)
    else:
        obs_bool = np.asarray(observed).astype(bool) if observed is not None else np.zeros_like(free_bool, dtype=bool)
        if obs_bool.shape != free_bool.shape:
            raise ValueError("observed and free must have the same shape")
        unknown = fbe_map == 0

    unknown_dilated = _disk_dilate(unknown, unknown_dilation_radius_cells)
    frontiers = (fbe_map == 1) & unknown_dilated
    if exclude_mask is not None:
        excluded = np.asarray(exclude_mask).astype(bool)
        if excluded.shape != free_bool.shape:
            raise ValueError("exclude_mask and free must have the same shape")
        frontiers &= ~excluded

    return {
        "free": free_bool,
        "occupied": occ_bool,
        "observed": obs_bool,
        "dilated_obstacles": dilated_obstacles,
        "unknown": unknown,
        "unknown_dilated": unknown_dilated,
        "frontier": frontiers,
        "fbe_map": fbe_map,
    }


def frontier_cells(
    free: np.ndarray,
    observed: Optional[np.ndarray] = None,
    occupancy: Optional[np.ndarray] = None,
    obstacle_dilation_radius_cells: int = 4,
    unknown_dilation_radius_cells: int = 1,
    exclude_mask: Optional[np.ndarray] = None,
    unknown_source: str = "observed",
) -> np.ndarray:
    """Frontier-based exploration map.

    Free cells are 1, obstacle-dilated cells are 3, unknown cells are 0, and
    frontier cells are free cells intersecting the dilation of unknown.
    unknown_source="observed" uses the mapper's explicit observed mask so
    observed but non-free clearance cells do not become fake unknown.
    unknown_source="implicit" keeps the original tensor-map behavior where
    cells not in free or occupancy are unknown.
    """
    return frontier_debug_layers(
        free=free,
        observed=observed,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=obstacle_dilation_radius_cells,
        unknown_dilation_radius_cells=unknown_dilation_radius_cells,
        exclude_mask=exclude_mask,
        unknown_source=unknown_source,
    )["frontier"]


def _connected_components(mask: np.ndarray) -> List[List[GridCell]]:
    src = np.asarray(mask).astype(bool)
    if not np.any(src):
        return []
    visited = np.zeros_like(src, dtype=bool)
    h, w = src.shape
    components: List[List[GridCell]] = []
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    for start_row, start_col in zip(*np.nonzero(src)):
        start = (int(start_row), int(start_col))
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[GridCell] = deque([start])
        members: List[GridCell] = []
        while queue:
            row, col = queue.popleft()
            members.append((row, col))
            for dr, dc in offsets:
                rr, cc = row + dr, col + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                if visited[rr, cc] or not src[rr, cc]:
                    continue
                visited[rr, cc] = True
                queue.append((rr, cc))
        components.append(members)
    return components


def extract_frontiers(
    free: np.ndarray,
    observed: np.ndarray,
    traversible: np.ndarray,
    map_info: MapInfo,
    agent_grid: GridCell,
    min_cluster_size: int = 3,
    min_distance_m: float = 1.0,
    max_count: int = 64,
    occupancy: Optional[np.ndarray] = None,
    obstacle_dilation_radius_cells: int = 4,
    unknown_dilation_radius_cells: int = 1,
    exclude_mask: Optional[np.ndarray] = None,
    unknown_source: str = "observed",
    cluster_distance_mode: str = "mean",
    allow_near_frontier_fallback: bool = False,
    require_reachable: bool = True,
) -> List[FrontierCluster]:
    if cluster_distance_mode not in {"mean", "min", "center"}:
        raise ValueError(
            f"cluster_distance_mode must be 'mean', 'min', or 'center', got {cluster_distance_mode!r}"
        )
    cells = frontier_cells(
        free=free,
        observed=observed,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=obstacle_dilation_radius_cells,
        unknown_dilation_radius_cells=unknown_dilation_radius_cells,
        exclude_mask=exclude_mask,
        unknown_source=unknown_source,
    )
    dist_map = astar_distance_map(traversible, agent_grid, map_info.resolution_m, allow_diagonal=True)
    clusters: List[FrontierCluster] = []
    near_clusters: List[FrontierCluster] = []
    for members in _connected_components(cells):
        finite_members = (
            [(row, col) for row, col in members if np.isfinite(float(dist_map[row, col]))]
            if bool(require_reachable)
            else [(int(row), int(col)) for row, col in members]
        )
        if len(finite_members) < max(1, int(min_cluster_size)):
            continue
        member_arr = np.asarray(finite_members, dtype=np.float32)
        raw_member_dists = np.asarray([float(dist_map[row, col]) for row, col in finite_members], dtype=np.float32)
        if bool(require_reachable):
            member_dists = raw_member_dists
        else:
            agent_arr = np.asarray(agent_grid, dtype=np.float32)
            geometric_dists = np.linalg.norm(member_arr - agent_arr[None, :], axis=1).astype(np.float32) * float(map_info.resolution_m)
            member_dists = np.where(np.isfinite(raw_member_dists), raw_member_dists, geometric_dists).astype(np.float32)
        centroid = np.mean(member_arr, axis=0)
        center_idx = int(np.argmin(np.sum((member_arr - centroid) ** 2, axis=1)))
        center = tuple(int(v) for v in member_arr[center_idx])
        center_dist = float(dist_map[center])
        if not np.isfinite(center_dist) and bool(require_reachable):
            center_idx = int(np.argmin(member_dists))
            center = tuple(int(v) for v in member_arr[center_idx])
            center_dist = float(dist_map[center])
        if not np.isfinite(center_dist):
            center_dist = float(
                np.linalg.norm(np.asarray(center, dtype=np.float32) - np.asarray(agent_grid, dtype=np.float32))
                * float(map_info.resolution_m)
            )
        min_dist = float(np.min(member_dists))
        mean_dist = float(np.mean(member_dists))
        max_dist = float(np.max(member_dists))
        if cluster_distance_mode == "min":
            cluster_dist = min_dist
        elif cluster_distance_mode == "center":
            cluster_dist = center_dist
        else:
            cluster_dist = mean_dist
        wx, wy = grid_to_world_xy(center[0], center[1], map_info)
        cluster = FrontierCluster(
            center_grid=center,
            center_world=(wx, wy),
            members=finite_members,
            size=len(finite_members),
            path_distance_from_agent=cluster_dist,
            min_path_distance=min_dist,
            mean_path_distance=mean_dist,
            max_path_distance=max_dist,
            center_path_distance=center_dist,
            distance_inverse=_distance_inverse(cluster_dist, min_distance_m=float(min_distance_m)),
        )
        if cluster_dist < min_distance_m:
            near_clusters.append(cluster)
            continue
        clusters.append(cluster)
    if not clusters and bool(allow_near_frontier_fallback):
        for cluster in near_clusters:
            cluster.is_near_fallback = True
        clusters = near_clusters
    clusters.sort(key=lambda cluster: (-int(cluster.size), float(cluster.path_distance_from_agent)))
    if int(max_count) > 0:
        return clusters[: int(max_count)]
    return clusters


@dataclass
class FrontierSetPartition:
    real_frontiers: List[FrontierCluster]
    near_fallback_frontiers: List[FrontierCluster]
    filtered_frontiers: List[FrontierCluster]
    suppressed_count: int = 0
    stop_reason: Optional[str] = None


def extract_frontiers_with_partition(
    free: np.ndarray,
    observed: np.ndarray,
    traversible: np.ndarray,
    map_info: MapInfo,
    agent_grid: GridCell,
    min_cluster_size: int = 3,
    min_distance_m: float = 1.0,
    max_count: int = 64,
    occupancy: Optional[np.ndarray] = None,
    obstacle_dilation_radius_cells: int = 4,
    unknown_dilation_radius_cells: int = 1,
    exclude_mask: Optional[np.ndarray] = None,
    unknown_source: str = "observed",
    cluster_distance_mode: str = "mean",
    allow_near_frontier_fallback: bool = False,
    require_reachable: bool = True,
) -> FrontierSetPartition:
    real_frontiers = extract_frontiers(
        free=free,
        observed=observed,
        traversible=traversible,
        map_info=map_info,
        agent_grid=agent_grid,
        min_cluster_size=min_cluster_size,
        min_distance_m=min_distance_m,
        max_count=max_count,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=obstacle_dilation_radius_cells,
        unknown_dilation_radius_cells=unknown_dilation_radius_cells,
        exclude_mask=exclude_mask,
        unknown_source=unknown_source,
        cluster_distance_mode=cluster_distance_mode,
        allow_near_frontier_fallback=False,
        require_reachable=require_reachable,
    )
    near_fallback_frontiers: List[FrontierCluster] = []
    filtered_frontiers = list(real_frontiers)
    stop_reason = None
    if not real_frontiers and bool(allow_near_frontier_fallback):
        near_fallback_frontiers = extract_frontiers(
            free=free,
            observed=observed,
            traversible=traversible,
            map_info=map_info,
            agent_grid=agent_grid,
            min_cluster_size=min_cluster_size,
            min_distance_m=min_distance_m,
            max_count=max_count,
            occupancy=occupancy,
            obstacle_dilation_radius_cells=obstacle_dilation_radius_cells,
            unknown_dilation_radius_cells=unknown_dilation_radius_cells,
            exclude_mask=exclude_mask,
            unknown_source=unknown_source,
            cluster_distance_mode=cluster_distance_mode,
            allow_near_frontier_fallback=True,
            require_reachable=require_reachable,
        )
        filtered_frontiers = list(near_fallback_frontiers)
        if near_fallback_frontiers:
            stop_reason = "no_real_frontiers_only_near_fallback"
    elif not real_frontiers:
        stop_reason = "no_frontiers"
    return FrontierSetPartition(
        real_frontiers=list(real_frontiers),
        near_fallback_frontiers=list(near_fallback_frontiers),
        filtered_frontiers=list(filtered_frontiers),
        suppressed_count=0,
        stop_reason=stop_reason,
    )
