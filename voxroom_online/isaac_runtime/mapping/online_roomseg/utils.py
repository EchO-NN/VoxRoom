from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage


GridCell = tuple[int, int]


def conn(connectivity: int = 4) -> np.ndarray:
    if int(connectivity) == 8:
        return np.ones((3, 3), dtype=np.uint8)
    return np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def label_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    return ndimage.label(np.asarray(mask, dtype=bool), structure=conn(connectivity))


def component_slices(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    labels, count = label_components(mask, connectivity)
    return [labels == idx for idx in range(1, int(count) + 1)]


def relabel_compact(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    out = np.zeros_like(arr, dtype=np.int32)
    next_label = 1
    for label in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        out[arr == label] = int(next_label)
        next_label += 1
    return out


def rasterize_line(p0: Sequence[float] | Sequence[int], p1: Sequence[float] | Sequence[int], shape: tuple[int, int]) -> np.ndarray:
    r0, c0 = int(round(float(p0[0]))), int(round(float(p0[1])))
    r1, c1 = int(round(float(p1[0]))), int(round(float(p1[1])))
    steps = max(abs(r1 - r0), abs(c1 - c0)) + 1
    out = np.zeros(shape, dtype=bool)
    if steps <= 1:
        if 0 <= r0 < shape[0] and 0 <= c0 < shape[1]:
            out[r0, c0] = True
        return out
    rows = np.rint(np.linspace(r0, r1, steps)).astype(np.int32)
    cols = np.rint(np.linspace(c0, c1, steps)).astype(np.int32)
    valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    out[rows[valid], cols[valid]] = True
    return out


def line_cells(p0: Sequence[float] | Sequence[int], p1: Sequence[float] | Sequence[int], shape: tuple[int, int]) -> list[GridCell]:
    mask = rasterize_line(p0, p1, shape)
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(mask))]


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return ndimage.binary_dilation(np.asarray(mask, dtype=bool), structure=disk(radius)).astype(bool)


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return ndimage.binary_erosion(np.asarray(mask, dtype=bool), structure=disk(radius)).astype(bool)


def disk(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((yy * yy + xx * xx) <= radius * radius).astype(bool)


def remove_small_components(mask: np.ndarray, min_cells: int, connectivity: int = 4) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if int(min_cells) <= 1:
        return src.copy()
    labels, count = label_components(src, connectivity)
    out = np.zeros_like(src, dtype=bool)
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        if int(np.count_nonzero(comp)) >= int(min_cells):
            out |= comp
    return out


def fill_small_holes(
    free: np.ndarray,
    *,
    occupied: np.ndarray,
    max_area_cells: int,
    max_radius_cells: int,
    occupied_ratio_max: float,
) -> tuple[np.ndarray, list[dict]]:
    free_arr = np.asarray(free, dtype=bool)
    occ = np.asarray(occupied, dtype=bool)
    labels, count = label_components(~free_arr, 4)
    out = free_arr.copy()
    debug: list[dict] = []
    h, w = free_arr.shape
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        rows, cols = np.nonzero(comp)
        if rows.size == 0:
            continue
        touches_border = bool(np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)))
        if touches_border:
            continue
        area = int(rows.size)
        span = max(int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1))
        occ_ratio = float(np.count_nonzero(comp & occ)) / float(max(1, area))
        accepted = bool(
            area <= int(max_area_cells)
            and span <= max(1, 2 * int(max_radius_cells) + 1)
            and occ_ratio <= float(occupied_ratio_max)
            and np.any(dilate(comp, 1) & free_arr)
        )
        if accepted:
            out[comp] = True
        debug.append(
            {
                "component": int(idx),
                "area_cells": int(area),
                "span_cells": int(span),
                "occupied_ratio": float(occ_ratio),
                "accepted": bool(accepted),
            }
        )
    return out, debug


def component_metrics(mask: np.ndarray, resolution_m: float, distance_m: np.ndarray | None = None) -> dict:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return {
            "area_cells": 0,
            "area_m2": 0.0,
            "aspect_ratio": 0.0,
            "length_m": 0.0,
            "thickness_m": 0.0,
            "elongation": 0.0,
            "median_width_m": 0.0,
            "corridor_like": False,
            "room_like": False,
        }
    coords = np.stack([rows.astype(np.float32), cols.astype(np.float32)], axis=1)
    spans = np.ptp(coords, axis=0) + 1.0
    bbox_aspect = float(max(spans) / max(1.0, min(spans)))
    if coords.shape[0] >= 2:
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        cov = np.cov(centered.T)
        vals = np.linalg.eigvalsh(cov)
        vals = np.maximum(np.sort(vals), 1e-6)
        elongation = float(np.sqrt(vals[-1] / vals[0]))
    else:
        elongation = 1.0
    length_m = float(max(spans) * float(resolution_m))
    thickness_m = float(min(spans) * float(resolution_m))
    area_m2 = float(rows.size) * float(resolution_m) ** 2
    if distance_m is not None and np.any(mask):
        widths = 2.0 * np.asarray(distance_m, dtype=np.float32)[mask] * float(resolution_m)
        median_width = float(np.median(widths)) if widths.size else thickness_m
    else:
        median_width = thickness_m
    corridor_like = bool(bbox_aspect >= 2.5 and length_m >= 1.5 and median_width <= 1.8)
    room_like = bool(area_m2 >= 1.5 and (median_width >= 1.0 or bbox_aspect < 2.5))
    return {
        "area_cells": int(rows.size),
        "area_m2": float(area_m2),
        "aspect_ratio": float(bbox_aspect),
        "length_m": float(length_m),
        "thickness_m": float(thickness_m),
        "elongation": float(max(elongation, bbox_aspect)),
        "median_width_m": float(median_width),
        "corridor_like": bool(corridor_like),
        "room_like": bool(room_like),
        "bbox": [int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1],
    }


def sample_cells(mask: np.ndarray, max_cells: int = 128) -> list[list[int]]:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    cells = [[int(r), int(c)] for r, c in zip(rows.tolist(), cols.tolist())]
    if len(cells) <= int(max_cells):
        return cells
    stride = max(1, len(cells) // int(max_cells))
    return cells[::stride][: int(max_cells)]
