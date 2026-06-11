from __future__ import annotations

from collections import Counter, deque
import math
from typing import Optional, Sequence

import numpy as np


def assign_frontier_room_context(
    frontier_member_rcs: Sequence[Sequence[int]] | np.ndarray,
    room_label_map: np.ndarray,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    agent_rc: Optional[Sequence[int]],
    resolution_m: float,
    local_radius_m: float = 0.35,
    nearest_fallback_radius_m: float = 1.25,
    min_label_ratio: float = 0.20,
) -> dict:
    """Assign room context from the known-free side of a frontier cluster."""

    labels = np.asarray(room_label_map, dtype=np.int32)
    free = np.asarray(observed_free, dtype=bool)
    unk = np.asarray(unknown, dtype=bool)
    if labels.shape != free.shape or labels.shape != unk.shape:
        raise ValueError("frontier room context arrays must have matching HxW shape")
    members = np.asarray(frontier_member_rcs, dtype=np.int32)
    if members.size == 0:
        return _none_result()
    members = members.reshape((-1, 2))
    h, w = labels.shape
    frontier_mask = np.zeros(labels.shape, dtype=bool)
    for row, col in members:
        rr, cc = int(row), int(col)
        if 0 <= rr < h and 0 <= cc < w:
            frontier_mask[rr, cc] = True
    radius_cells = max(0, int(math.ceil(float(local_radius_m) / max(float(resolution_m), 1e-9))))
    near = _dilate_disk(frontier_mask, radius_cells)
    known_free_labels = near & free & ~unk & (labels > 0)
    local = _distribution(labels[known_free_labels])
    if local["total"] > 0:
        best_label, best_ratio = _best_distribution(local["counts"])
        if best_label is not None and best_ratio >= float(min_label_ratio):
            return {
                "room_id": int(best_label),
                "room_distribution": {str(k): float(v) for k, v in local["ratios"].items()},
                "confidence": float(best_ratio),
                "method": "known_free_side",
                "num_support_cells": int(local["total"]),
            }

    fallback = _nearest_labeled_free(
        members,
        labels,
        allowed=free & ~unk,
        max_steps=max(0, int(math.ceil(float(nearest_fallback_radius_m) / max(float(resolution_m), 1e-9)))),
    )
    if fallback is not None:
        label, steps = fallback
        max_steps = max(1, int(math.ceil(float(nearest_fallback_radius_m) / max(float(resolution_m), 1e-9))))
        return {
            "room_id": int(label),
            "room_distribution": {str(int(label)): 1.0},
            "confidence": float(max(0.05, 0.5 * (1.0 - float(steps) / float(max_steps + 1)))),
            "method": "nearest_labeled_free",
            "num_support_cells": 1,
        }
    return _none_result()


def _none_result() -> dict:
    return {
        "room_id": None,
        "room_distribution": {},
        "confidence": 0.0,
        "method": "none",
        "num_support_cells": 0,
    }


def _distribution(values: np.ndarray) -> dict:
    raw = [int(v) for v in np.asarray(values, dtype=np.int32).ravel().tolist() if int(v) > 0]
    counts = Counter(raw)
    total = sum(counts.values())
    ratios = {label: float(count) / float(max(total, 1)) for label, count in counts.items()}
    return {"counts": counts, "ratios": ratios, "total": int(total)}


def _best_distribution(counts: Counter) -> tuple[int | None, float]:
    if not counts:
        return None, 0.0
    total = sum(counts.values())
    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return int(label), float(count) / float(max(total, 1))


def _nearest_labeled_free(
    starts: np.ndarray,
    labels: np.ndarray,
    allowed: np.ndarray,
    max_steps: int,
) -> tuple[int, int] | None:
    h, w = labels.shape
    visited = np.zeros(labels.shape, dtype=bool)
    queue: deque[tuple[int, int, int]] = deque()
    for row, col in starts:
        rr, cc = int(row), int(col)
        if 0 <= rr < h and 0 <= cc < w and not visited[rr, cc]:
            visited[rr, cc] = True
            queue.append((rr, cc, 0))
    while queue:
        row, col, dist = queue.popleft()
        if bool(allowed[row, col]) and int(labels[row, col]) > 0:
            return int(labels[row, col]), int(dist)
        if dist >= int(max_steps):
            continue
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if nr < 0 or nc < 0 or nr >= h or nc >= w or visited[nr, nc]:
                continue
            if not bool(allowed[nr, nc]) and int(labels[nr, nc]) <= 0:
                continue
            visited[nr, nc] = True
            queue.append((nr, nc, dist + 1))
    return None


def _dilate_disk(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_cells))
    if radius <= 0 or not np.any(src):
        return src.copy()
    out = np.zeros_like(src, dtype=bool)
    h, w = src.shape
    offsets = [
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if dr * dr + dc * dc <= radius * radius
    ]
    for row, col in zip(*np.nonzero(src)):
        for dr, dc in offsets:
            rr, cc = int(row + dr), int(col + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out

