from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
from scipy import ndimage


FOUR_CONNECTED = ndimage.generate_binary_structure(2, 1)
EIGHT_CONNECTED = ndimage.generate_binary_structure(2, 2)


def label_components(mask: np.ndarray, *, connectivity: int = 4) -> tuple[np.ndarray, int]:
    structure = FOUR_CONNECTED if int(connectivity) == 4 else EIGHT_CONNECTED
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    return labels.astype(np.int32), int(count)


def sorted_component_ids(labels: np.ndarray) -> list[int]:
    ids = [int(v) for v in np.unique(labels) if int(v) > 0]
    ids.sort(key=lambda v: int(np.count_nonzero(labels == v)), reverse=True)
    return ids


def relabel_components(mask: np.ndarray, *, min_area_cells: int = 1, connectivity: int = 4) -> np.ndarray:
    comps, _ = label_components(mask, connectivity=connectivity)
    out = np.zeros(comps.shape, dtype=np.int32)
    next_id = 1
    for comp_id in sorted_component_ids(comps):
        comp = comps == comp_id
        if int(np.count_nonzero(comp)) < int(min_area_cells):
            continue
        out[comp] = next_id
        next_id += 1
    return out


def wavefront_fill_unlabeled(labels: np.ndarray, *, domain: np.ndarray) -> np.ndarray:
    domain = np.asarray(domain, dtype=bool)
    out = np.asarray(labels, dtype=np.int32).copy()
    out[~domain] = 0
    queue: deque[tuple[int, int]] = deque()
    rows, cols = np.where((out > 0) & domain)
    for r, c in zip(rows.tolist(), cols.tolist()):
        queue.append((int(r), int(c)))
    h, w = out.shape
    while queue:
        r, c = queue.popleft()
        label = int(out[r, c])
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if not domain[nr, nc] or out[nr, nc] > 0:
                continue
            out[nr, nc] = label
            queue.append((nr, nc))
    return out


def nearest_seed_fill(seed_labels: np.ndarray, *, domain: np.ndarray) -> np.ndarray:
    domain = np.asarray(domain, dtype=bool)
    seeds = np.asarray(seed_labels, dtype=np.int32)
    out = np.zeros(seeds.shape, dtype=np.int32)
    valid_seed = (seeds > 0) & domain
    if not bool(np.any(valid_seed)):
        return out
    _, indices = ndimage.distance_transform_edt(~valid_seed, return_indices=True)
    rr = indices[0]
    cc = indices[1]
    out[domain] = seeds[rr[domain], cc[domain]]
    out[~domain] = 0
    return out


def local_maxima_mask(values: np.ndarray, *, domain: np.ndarray, footprint_size: int = 9, min_value: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    domain = np.asarray(domain, dtype=bool)
    size = max(3, int(footprint_size))
    if size % 2 == 0:
        size += 1
    maxed = ndimage.maximum_filter(values, size=size, mode="nearest")
    return domain & (values >= float(min_value)) & (values >= maxed - 1.0e-6)


def split_large_seed_components(seed_mask: np.ndarray, *, max_seed_area_cells: int) -> np.ndarray:
    comps, _ = label_components(seed_mask, connectivity=8)
    out = np.zeros(comps.shape, dtype=bool)
    for comp_id in sorted_component_ids(comps):
        comp = comps == comp_id
        if int(np.count_nonzero(comp)) <= int(max_seed_area_cells):
            out |= comp
            continue
        coords = np.argwhere(comp)
        if coords.size == 0:
            continue
        centroid = np.mean(coords, axis=0)
        d2 = np.sum((coords.astype(np.float32) - centroid[None, :]) ** 2, axis=1)
        r, c = coords[int(np.argmin(d2))]
        out[int(r), int(c)] = True
    return out


def draw_grid_line(mask: np.ndarray, p0: Iterable[int], p1: Iterable[int]) -> None:
    r0, c0 = [int(round(float(v))) for v in p0]
    r1, c1 = [int(round(float(v))) for v in p1]
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    h, w = mask.shape
    while True:
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = True
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

