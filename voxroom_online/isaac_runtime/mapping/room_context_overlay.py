from __future__ import annotations

from collections import Counter, deque
import math
from typing import Optional

import numpy as np


def build_navigation_free_room_context_overlay(
    final_labels: np.ndarray,
    navigation_free: np.ndarray,
    unknown: np.ndarray,
    obstacle: Optional[np.ndarray],
    structural_boundary: Optional[np.ndarray],
    resolution_m: float,
    max_absorb_distance_m: float = 1.25,
    min_seed_room_area_cells: int = 20,
    protect_unknown: bool = True,
    do_not_cross_structural_boundary: bool = True,
    do_not_cross_obstacle: bool = True,
    absorbed_reliability: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a room-context-only overlay over known navigation-free cells.

    The returned labels never feed structural room segmentation. They are a safe
    context map for visualization and frontier-room attribution.
    """

    labels = np.asarray(final_labels, dtype=np.int32)
    nav = np.asarray(navigation_free, dtype=bool)
    unk = np.asarray(unknown, dtype=bool)
    if labels.shape != nav.shape or labels.shape != unk.shape:
        raise ValueError("context overlay arrays must have matching HxW shape")
    obs = np.zeros_like(nav, dtype=bool) if obstacle is None else np.asarray(obstacle, dtype=bool)
    boundary = np.zeros_like(nav, dtype=bool) if structural_boundary is None else np.asarray(structural_boundary, dtype=bool)
    if obs.shape != nav.shape or boundary.shape != nav.shape:
        raise ValueError("context overlay obstacle/boundary arrays must match final_labels")

    context_labels = labels.copy()
    reliability = np.zeros(labels.shape, dtype=np.float32)
    reliability[labels > 0] = 1.0

    allowed = nav.copy()
    if bool(protect_unknown):
        allowed &= ~unk
    if bool(do_not_cross_obstacle):
        allowed &= ~obs
    if bool(do_not_cross_structural_boundary):
        allowed &= ~boundary

    label_counts = Counter(int(v) for v in labels[labels > 0].tolist())
    valid_labels = {label for label, count in label_counts.items() if int(count) >= int(min_seed_room_area_cells)}
    seed_mask = allowed & np.isin(labels, list(valid_labels))
    candidate = allowed & (labels <= 0)
    max_steps = max(0, int(math.ceil(float(max_absorb_distance_m) / max(float(resolution_m), 1e-9))))

    distance = np.full(labels.shape, np.iinfo(np.int32).max, dtype=np.int32)
    visited_label = np.zeros(labels.shape, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for row, col in zip(*np.nonzero(seed_mask)):
        rr, cc = int(row), int(col)
        distance[rr, cc] = 0
        visited_label[rr, cc] = int(labels[rr, cc])
        queue.append((rr, cc))

    while queue:
        row, col = queue.popleft()
        next_dist = int(distance[row, col]) + 1
        if next_dist > max_steps:
            continue
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if nr < 0 or nc < 0 or nr >= labels.shape[0] or nc >= labels.shape[1]:
                continue
            if not bool(allowed[nr, nc]):
                continue
            if next_dist >= int(distance[nr, nc]):
                continue
            distance[nr, nc] = next_dist
            visited_label[nr, nc] = int(visited_label[row, col])
            queue.append((nr, nc))

    absorbed = candidate & (visited_label > 0) & (distance <= max_steps)
    context_labels[absorbed] = visited_label[absorbed]
    if np.any(absorbed):
        decay = 1.0 - (distance[absorbed].astype(np.float32) / float(max(max_steps, 1)))
        reliability[absorbed] = np.maximum(0.05, float(absorbed_reliability) * np.clip(decay, 0.25, 1.0))

    remaining = nav & (context_labels <= 0) & (~unk if bool(protect_unknown) else True)
    debug = {
        "nav_free_overlay_enabled": True,
        "candidate_cells": int(np.count_nonzero(candidate)),
        "absorbed_cells": int(np.count_nonzero(absorbed)),
        "remaining_unlabeled_nav_free_cells": int(np.count_nonzero(remaining)),
        "max_absorb_distance_m": float(max_absorb_distance_m),
        "max_absorb_steps": int(max_steps),
        "min_seed_room_area_cells": int(min_seed_room_area_cells),
        "used_structural_boundary_guard": bool(do_not_cross_structural_boundary),
        "used_unknown_guard": bool(protect_unknown),
        "used_obstacle_guard": bool(do_not_cross_obstacle),
        "seed_labels": sorted(int(v) for v in valid_labels),
    }
    return context_labels.astype(np.int32), reliability.astype(np.float32), debug
