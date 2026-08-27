from __future__ import annotations

import math
from typing import Iterable

import numpy as np

GridCell = tuple[int, int]


def bresenham_cells(p0: Iterable[float], p1: Iterable[float]) -> list[tuple[int, int]]:
    r0, c0 = [int(round(float(v))) for v in p0]
    r1, c1 = [int(round(float(v))) for v in p1]
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((int(r), int(c)))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def bresenham_line(p0: Iterable[float], p1: Iterable[float]) -> list[tuple[int, int]]:
    return bresenham_cells(p0, p1)


def in_bounds(shape: tuple[int, int], rc: Iterable[float]) -> bool:
    row, col = [int(round(float(v))) for v in rc]
    return 0 <= row < int(shape[0]) and 0 <= col < int(shape[1])


def draw_line(mask: np.ndarray, p0: Iterable[float], p1: Iterable[float]) -> None:
    h, w = mask.shape
    for r, c in bresenham_cells(p0, p1):
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = True


def unit_normal(p0: tuple[int, int], p1: tuple[int, int]) -> tuple[float, float]:
    dr = float(p1[0] - p0[0])
    dc = float(p1[1] - p0[1])
    norm = math.hypot(dr, dc)
    if norm <= 1.0e-6:
        return 0.0, 1.0
    return -dc / norm, dr / norm


def nearest_free_along_normal(
    midpoint: tuple[float, float],
    normal: tuple[float, float],
    free_mask: np.ndarray,
    *,
    direction: float,
    max_cells: int = 20,
) -> tuple[int, int] | None:
    h, w = free_mask.shape
    for step in range(1, int(max_cells) + 1):
        r = int(round(float(midpoint[0]) + float(direction) * float(normal[0]) * step))
        c = int(round(float(midpoint[1]) + float(direction) * float(normal[1]) * step))
        if r < 0 or r >= h or c < 0 or c >= w:
            break
        if bool(free_mask[r, c]):
            return int(r), int(c)
    return None


def rc_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))
