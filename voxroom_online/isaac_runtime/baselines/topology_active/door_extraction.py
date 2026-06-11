from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage

from .geometry import nearest_free_along_normal, rc_distance, unit_normal
from .schema import DoorPair, DoorPointCandidate


@dataclass(frozen=True)
class DoorPairConfig:
    checking_size_cells: int = 2
    min_door_width_m: float = 0.45
    max_door_width_m: float = 2.0
    nms_midpoint_radius_cells: float = 4.0
    waypoint_search_cells: int = 12


@dataclass(frozen=True)
class RayCastingDoorExtractor:
    num_rays: int = 360
    gap_threshold_cells: float = 1.0
    checking_size_cells: int = 2
    door_detection_range_cells: int = 20
    bot_near_range_cells: int = 30

    def extract(
        self,
        *,
        occupancy_map: np.ndarray,
        free_mask: np.ndarray,
        unknown_mask: np.ndarray,
        robot_rc: tuple[int, int],
        map_info: object | None = None,
    ) -> list[DoorPointCandidate]:
        _ = map_info
        occupied = np.asarray(occupancy_map, dtype=bool)
        free = np.asarray(free_mask, dtype=bool)
        unknown = np.asarray(unknown_mask, dtype=bool)
        h, w = occupied.shape
        rr, cc = int(robot_rc[0]), int(robot_rc[1])
        if rr < 0 or rr >= h or cc < 0 or cc >= w:
            return []
        distances = np.zeros(int(self.num_rays), dtype=np.float32)
        terminals: list[tuple[int, int]] = []
        for idx in range(int(self.num_rays)):
            yaw = (2.0 * math.pi * float(idx)) / float(self.num_rays)
            dist, cell = _trace_ray(occupied, unknown, (rr, cc), yaw, int(self.door_detection_range_cells))
            distances[idx] = float(dist)
            terminals.append(cell)
        candidates: list[DoorPointCandidate] = []
        for idx in range(int(self.num_rays)):
            j = (idx + 1) % int(self.num_rays)
            if abs(float(distances[idx]) - float(distances[j])) < float(self.gap_threshold_cells):
                continue
            cell = terminals[idx] if distances[idx] <= distances[j] else terminals[j]
            if rc_distance(cell, (rr, cc)) < float(self.bot_near_range_cells):
                continue
            r, c = cell
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if not _near_free_to_occupied_transition(free, occupied, r, c, int(self.checking_size_cells)):
                continue
            candidates.append(
                DoorPointCandidate(
                    rc=(int(r), int(c)),
                    source="raycast",
                    score=float(abs(float(distances[idx]) - float(distances[j]))),
                    yaw_rad=float(2.0 * math.pi * idx / max(1, int(self.num_rays))),
                    metadata={"dist_a": float(distances[idx]), "dist_b": float(distances[j])},
                )
            )
        return _nms_candidates(candidates, radius_cells=max(1, int(self.checking_size_cells)))


def fuse_door_pairs(
    candidates: Iterable[DoorPointCandidate],
    *,
    free_mask: np.ndarray,
    occupied_mask: np.ndarray,
    resolution_m: float,
    min_door_width_m: float = 0.45,
    max_door_width_m: float = 2.0,
) -> list[DoorPair]:
    items = _nms_candidates(list(candidates), radius_cells=max(1, int(round(0.10 / max(float(resolution_m), 1.0e-6)))))
    free = np.asarray(free_mask, dtype=bool)
    occupied = np.asarray(occupied_mask, dtype=bool)
    min_cells = max(1, int(round(float(min_door_width_m) / max(float(resolution_m), 1.0e-6))))
    max_cells = max(min_cells, int(round(float(max_door_width_m) / max(float(resolution_m), 1.0e-6))))
    pairs: list[DoorPair] = []
    next_id = 1
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            width = rc_distance(a.rc, b.rc)
            if width < min_cells or width > max_cells:
                continue
            line_free_ratio = _line_free_ratio(a.rc, b.rc, free, occupied)
            if line_free_ratio < 0.45:
                continue
            midpoint = ((float(a.rc[0]) + float(b.rc[0])) * 0.5, (float(a.rc[1]) + float(b.rc[1])) * 0.5)
            normal = unit_normal(a.rc, b.rc)
            wp_a = nearest_free_along_normal(midpoint, normal, free, direction=1.0, max_cells=max_cells)
            wp_b = nearest_free_along_normal(midpoint, normal, free, direction=-1.0, max_cells=max_cells)
            if wp_a is None or wp_b is None:
                continue
            pairs.append(
                DoorPair(
                    door_id=next_id,
                    p0_rc=a.rc,
                    p1_rc=b.rc,
                    midpoint_rc=midpoint,
                    width_cells=float(width),
                    normal_rc=normal,
                    waypoint_a_rc=wp_a,
                    waypoint_b_rc=wp_b,
                    score=float(a.score + b.score + line_free_ratio),
                    sources=frozenset({a.source, b.source}),
                )
            )
            next_id += 1
    pairs.sort(key=lambda item: item.score, reverse=True)
    return _nms_pairs(pairs)


def fuse_door_candidates_into_pairs(
    candidates: Iterable[DoorPointCandidate],
    *,
    free_mask: np.ndarray,
    occupancy_map: np.ndarray,
    map_info: object | None,
    config: DoorPairConfig | None = None,
) -> list[DoorPair]:
    cfg = config or DoorPairConfig()
    return fuse_door_pairs(
        candidates,
        free_mask=free_mask,
        occupied_mask=occupancy_map,
        resolution_m=float(getattr(map_info, "resolution_m", 0.05) or 0.05),
        min_door_width_m=float(cfg.min_door_width_m),
        max_door_width_m=float(cfg.max_door_width_m),
    )


def _trace_ray(occupied: np.ndarray, unknown: np.ndarray, origin: tuple[int, int], yaw: float, max_range: int) -> tuple[float, tuple[int, int]]:
    h, w = occupied.shape
    r0, c0 = int(origin[0]), int(origin[1])
    last = (r0, c0)
    for step in range(1, int(max_range) + 1):
        r = int(round(float(r0) + math.sin(float(yaw)) * float(step)))
        c = int(round(float(c0) + math.cos(float(yaw)) * float(step)))
        if r < 0 or r >= h or c < 0 or c >= w:
            return float(step), last
        last = (r, c)
        if bool(occupied[r, c]) or bool(unknown[r, c]):
            return float(step), last
    return float(max_range), last


def _near_free_to_occupied_transition(free: np.ndarray, occupied: np.ndarray, r: int, c: int, radius: int) -> bool:
    r0 = max(0, int(r) - int(radius))
    r1 = min(free.shape[0], int(r) + int(radius) + 1)
    c0 = max(0, int(c) - int(radius))
    c1 = min(free.shape[1], int(c) + int(radius) + 1)
    return bool(np.any(free[r0:r1, c0:c1])) and bool(np.any(occupied[r0:r1, c0:c1]))


def _nms_candidates(candidates: list[DoorPointCandidate], *, radius_cells: int) -> list[DoorPointCandidate]:
    kept: list[DoorPointCandidate] = []
    for cand in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(rc_distance(cand.rc, other.rc) <= float(radius_cells) for other in kept):
            continue
        kept.append(cand)
    return kept


def _line_free_ratio(a: tuple[int, int], b: tuple[int, int], free: np.ndarray, occupied: np.ndarray) -> float:
    from .geometry import bresenham_cells

    cells = bresenham_cells(a, b)
    if not cells:
        return 0.0
    ok = 0
    for r, c in cells:
        if 0 <= r < free.shape[0] and 0 <= c < free.shape[1] and bool(free[r, c]) and not bool(occupied[r, c]):
            ok += 1
    return float(ok) / float(len(cells))


def _nms_pairs(pairs: list[DoorPair], *, radius_cells: float = 4.0) -> list[DoorPair]:
    kept: list[DoorPair] = []
    for pair in pairs:
        if any(rc_distance(pair.midpoint_rc, other.midpoint_rc) <= float(radius_cells) for other in kept):
            continue
        kept.append(pair)
    return kept
