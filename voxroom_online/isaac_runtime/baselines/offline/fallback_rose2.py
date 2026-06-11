from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np


def smoke_segment_free_components(
    arrays: Mapping[str, np.ndarray],
    *,
    min_area_cells: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Contract smoke only: label connected free-space components.

    This is deliberately not a ROSE2 reimplementation.  It exists so the
    offline baseline wrapper can verify shape, dtype, clipping, and NPZ output
    plumbing on machines without ROS.
    """

    shape = _infer_shape(arrays)
    free = _output_domain(arrays, shape)
    labels = np.zeros(shape, dtype=np.int32)
    next_label = 1
    min_area = max(1, int(min_area_cells))
    visited = np.zeros(shape, dtype=bool)
    for row, col in zip(*np.nonzero(free), strict=False):
        if visited[int(row), int(col)]:
            continue
        cells = _flood_fill(free, visited, int(row), int(col))
        if len(cells) < min_area:
            continue
        for r, c in cells:
            labels[r, c] = next_label
        next_label += 1

    metadata = {
        "runner_type": "python_smoke_fallback",
        "method": "rose2",
        "not_original_rose2": True,
        "allowed_usage": "smoke_test_only",
        "main_experiment_allowed": False,
        "fallback_algorithm": "connected_components_of_free_domain",
        "min_area_cells": int(min_area),
        "rooms": int(next_label - 1),
    }
    return labels, metadata


def _infer_shape(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    for key in (
        "final_room_label_map",
        "navigation_free_room_domain",
        "observed_free_mask",
        "occupancy_map",
        "obstacle_mask",
        "unknown_mask",
    ):
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 2:
            return (int(arr.shape[0]), int(arr.shape[1]))
    raise ValueError("cannot infer 2D snapshot shape for ROSE2 smoke fallback")


def _output_domain(arrays: Mapping[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    for key in ("navigation_free_room_domain", "observed_free_mask", "voxel_nav_free_xy", "vertical_free_room_domain"):
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=bool)
        if arr.shape == shape and np.any(arr):
            domain = arr.copy()
            break
    else:
        domain = np.ones(shape, dtype=bool)

    obstacle = _optional_bool(arrays.get("obstacle_mask", arrays.get("occupancy_map")), shape)
    unknown = _optional_bool(arrays.get("unknown_mask"), shape)
    if obstacle is not None:
        domain &= ~obstacle
    if unknown is not None:
        domain &= ~unknown
    return domain


def _optional_bool(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        return None
    return arr


def _flood_fill(free: np.ndarray, visited: np.ndarray, row: int, col: int) -> list[tuple[int, int]]:
    height, width = free.shape
    cells: list[tuple[int, int]] = []
    queue: deque[tuple[int, int]] = deque([(row, col)])
    visited[row, col] = True
    while queue:
        r, c = queue.popleft()
        cells.append((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if visited[nr, nc] or not bool(free[nr, nc]):
                continue
            visited[nr, nc] = True
            queue.append((nr, nc))
    return cells
