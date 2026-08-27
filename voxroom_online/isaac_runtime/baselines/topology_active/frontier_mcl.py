from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage

from .geometry import draw_line, rc_distance
from .schema import DoorPair, FrontierCluster, MCLResult


def compute_current_room_mcl(
    *,
    free_mask: np.ndarray,
    unknown_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    door_pairs: list[DoorPair],
    closed_door_ids: set[int],
    start_rc: tuple[int, int],
    previous_mcl: np.ndarray | None,
) -> MCLResult:
    free = np.asarray(free_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    obstacle = np.asarray(obstacle_mask, dtype=bool)
    separators = np.zeros(free.shape, dtype=bool)
    for pair in door_pairs:
        if int(pair.door_id) in closed_door_ids or not closed_door_ids:
            draw_line(separators, pair.p0_rc, pair.p1_rc)
    mcl = _bfs_room(free=free, obstacle=obstacle, separators=separators, start_rc=start_rc)
    if previous_mcl is not None and np.asarray(previous_mcl).shape == free.shape:
        mcl = (mcl | np.asarray(previous_mcl, dtype=bool)) & free & ~obstacle
    frontier = _frontier_mask(mcl=mcl, unknown=unknown, obstacle=obstacle)
    clusters = _frontier_clusters(frontier, start_rc=start_rc)
    return MCLResult(mcl_mask=mcl, frontier_mask=frontier, frontier_clusters=clusters)


def _bfs_room(*, free: np.ndarray, obstacle: np.ndarray, separators: np.ndarray, start_rc: tuple[int, int]) -> np.ndarray:
    h, w = free.shape
    out = np.zeros(free.shape, dtype=bool)
    sr, sc = int(start_rc[0]), int(start_rc[1])
    if sr < 0 or sr >= h or sc < 0 or sc >= w or not bool(free[sr, sc]):
        return out
    queue: deque[tuple[int, int]] = deque([(sr, sc)])
    out[sr, sc] = True
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if out[nr, nc] or not free[nr, nc] or obstacle[nr, nc] or separators[nr, nc]:
                continue
            out[nr, nc] = True
            queue.append((nr, nc))
    return out


def _frontier_mask(*, mcl: np.ndarray, unknown: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    unknown_adjacent = ndimage.binary_dilation(unknown & ~obstacle, structure=ndimage.generate_binary_structure(2, 1))
    return np.asarray(mcl, dtype=bool) & unknown_adjacent


def _frontier_clusters(frontier: np.ndarray, *, start_rc: tuple[int, int]) -> list[FrontierCluster]:
    labels, count = ndimage.label(np.asarray(frontier, dtype=bool), structure=ndimage.generate_binary_structure(2, 1))
    clusters: list[FrontierCluster] = []
    for idx in range(1, int(count) + 1):
        mask = labels == idx
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        centroid = np.mean(coords.astype(np.float32), axis=0)
        d2 = np.sum((coords.astype(np.float32) - centroid[None, :]) ** 2, axis=1)
        rep = coords[int(np.argmin(d2))]
        clusters.append(
            FrontierCluster(
                cluster_id=int(idx),
                mask=mask,
                size=int(coords.shape[0]),
                centroid_rc=(float(centroid[0]), float(centroid[1])),
                representative_rc=(int(rep[0]), int(rep[1])),
                distance_to_robot_cells=rc_distance((float(start_rc[0]), float(start_rc[1])), (float(centroid[0]), float(centroid[1]))),
            )
        )
    clusters.sort(key=lambda item: (-int(item.size), float(item.distance_to_robot_cells)))
    return clusters

