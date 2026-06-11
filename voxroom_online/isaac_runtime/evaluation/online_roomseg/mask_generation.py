from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage

from .annotation_schema import LineAnnotation, MergeGroup
from .common import relabel_positive_labels_sequentially


@dataclass(frozen=True)
class GtGenerationConfig:
    line_width_cells: int = 3
    preclose_radius_cells: int = 1
    connectivity: int = 4
    min_room_area_cells: int = 1
    assign_cut_pixels: bool = True


@dataclass(frozen=True)
class GtGenerationResult:
    labels: np.ndarray
    component_labels: np.ndarray
    line_mask: np.ndarray
    metadata: dict


def bresenham_rc(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = int(p0[0]), int(p0[1])
    r1, c1 = int(p1[0]), int(p1[1])
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    cells: list[tuple[int, int]] = []
    if dc > dr:
        err = dc // 2
        r = r0
        for c in range(c0, c1 + sc, sc):
            cells.append((int(r), int(c)))
            err -= dr
            if err < 0:
                r += sr
                err += dc
    else:
        err = dr // 2
        c = c0
        for r in range(r0, r1 + sr, sr):
            cells.append((int(r), int(c)))
            err -= dc
            if err < 0:
                c += sc
                err += dr
    return cells


def generate_gt_from_annotation(
    *,
    eval_domain: np.ndarray,
    split_lines: Iterable[LineAnnotation],
    merge_groups: Iterable[MergeGroup],
    obstacle_mask: np.ndarray | None = None,
    segmentation_domain: np.ndarray | None = None,
    config: GtGenerationConfig | None = None,
) -> GtGenerationResult:
    cfg = config or GtGenerationConfig()
    split_lines_list = list(split_lines)
    merge_groups_list = list(merge_groups)
    output_domain = np.asarray(eval_domain, dtype=bool)
    if output_domain.ndim != 2:
        raise ValueError("eval_domain must be 2D")
    split_domain = output_domain if segmentation_domain is None else np.asarray(segmentation_domain, dtype=bool)
    if split_domain.shape != output_domain.shape:
        raise ValueError("segmentation_domain shape mismatch")
    obstacle = np.zeros_like(output_domain, dtype=bool) if obstacle_mask is None else np.asarray(obstacle_mask, dtype=bool)
    if obstacle.shape != output_domain.shape:
        raise ValueError("obstacle_mask shape mismatch")
    closed_domain = split_domain.copy()
    if int(cfg.preclose_radius_cells) > 0:
        closed_domain = ndimage.binary_closing(split_domain, structure=disk(int(cfg.preclose_radius_cells))).astype(bool)
        closed_domain |= split_domain
    line_mask = rasterize_split_lines(
        shape=output_domain.shape,
        split_lines=split_lines_list,
        default_width=max(1, int(cfg.line_width_cells)),
        domain=closed_domain,
    )
    cut_domain = closed_domain & ~line_mask
    component_labels, count = ndimage.label(cut_domain, structure=connectivity_structure(int(cfg.connectivity)))
    labels = np.asarray(component_labels, dtype=np.int32)
    labels, small_debug = merge_small_components(labels, min_area_cells=max(1, int(cfg.min_room_area_cells)))
    labels = apply_merge_groups(labels, merge_groups_list)
    if bool(cfg.assign_cut_pixels):
        labels, fill_debug = assign_cut_pixels_to_nearest_room(labels, domain=split_domain, line_mask=line_mask)
    else:
        fill_debug = {"cut_pixels_assigned": 0, "cut_pixels_unassigned": int(np.count_nonzero(split_domain & line_mask & (labels == 0)))}
    final_gt, projection_debug = project_labels_to_output_domain(labels, output_domain)
    final_gt = relabel_positive_labels_sequentially(final_gt)
    room_areas = {str(int(v)): int(np.count_nonzero(final_gt == int(v))) for v in np.unique(final_gt) if int(v) > 0}
    metadata = {
        "shape": [int(output_domain.shape[0]), int(output_domain.shape[1])],
        "domain_pixels": int(np.count_nonzero(output_domain)),
        "output_domain_pixels": int(np.count_nonzero(output_domain)),
        "segmentation_domain_pixels": int(np.count_nonzero(split_domain)),
        "room_count": int(len(room_areas)),
        "line_count": int(len(split_lines_list)),
        "separator_line_count": int(sum(1 for line in split_lines_list if getattr(line, "kind", "separator") != "wall_completion")),
        "wall_completion_line_count": int(sum(1 for line in split_lines_list if getattr(line, "kind", "separator") == "wall_completion")),
        "line_width_cells_effective": int(cfg.line_width_cells),
        "preclose_radius_cells": int(cfg.preclose_radius_cells),
        "connectivity": int(cfg.connectivity),
        "unlabeled_domain_pixels": int(np.count_nonzero(output_domain & (final_gt == 0))),
        "room_areas_cells": room_areas,
        **small_debug,
        **fill_debug,
        **projection_debug,
    }
    return GtGenerationResult(labels=final_gt, component_labels=component_labels.astype(np.int32), line_mask=line_mask.astype(bool), metadata=metadata)


def rasterize_split_lines(
    *,
    shape: tuple[int, int],
    split_lines: Iterable[LineAnnotation],
    default_width: int,
    domain: np.ndarray | None = None,
) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    core = np.zeros((h, w), dtype=bool)
    for line in split_lines:
        for rr, cc in bresenham_rc(line.p0_rc, line.p1_rc):
            if 0 <= rr < h and 0 <= cc < w:
                core[rr, cc] = True
    max_width = max([max(1, int(getattr(line, "width_cells", default_width))) for line in split_lines] + [max(1, int(default_width))])
    # Per-line width is retained in schema. T0 applies the maximum line width as
    # one mask because overlapping widths are semantically equivalent separators.
    radius = max(0, int(max_width) // 2)
    line_mask = ndimage.binary_dilation(core, structure=disk(radius)).astype(bool) if radius > 0 else core
    if domain is not None:
        line_mask &= np.asarray(domain, dtype=bool)
    return line_mask


def merge_small_components(labels: np.ndarray, *, min_area_cells: int) -> tuple[np.ndarray, dict]:
    arr = np.asarray(labels, dtype=np.int32).copy()
    if int(min_area_cells) <= 1:
        return arr, {"small_components_merged": 0, "small_component_unmerged": []}
    merged = 0
    unmerged: list[int] = []
    for label in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        comp = arr == label
        if int(np.count_nonzero(comp)) >= int(min_area_cells):
            continue
        nearest = _nearest_neighbor_label(arr, comp)
        if nearest > 0:
            arr[comp] = int(nearest)
            merged += 1
        else:
            unmerged.append(int(label))
    return relabel_positive_labels_sequentially(arr), {"small_components_merged": int(merged), "small_component_unmerged": unmerged}


def apply_merge_groups(labels: np.ndarray, merge_groups: Iterable[MergeGroup]) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    for group in merge_groups:
        ids = [int(v) for v in group.component_ids if int(v) > 0]
        if not ids:
            continue
        canonical = min(ids)
        out[np.isin(out, ids)] = int(canonical)
    return relabel_positive_labels_sequentially(out)


def assign_cut_pixels_to_nearest_room(labels: np.ndarray, *, domain: np.ndarray, line_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    out = np.asarray(labels, dtype=np.int32).copy()
    zero_cut = np.asarray(domain, dtype=bool) & np.asarray(line_mask, dtype=bool) & (out == 0)
    if not np.any(zero_cut):
        return out, {"cut_pixels_assigned": 0, "cut_pixels_unassigned": 0}
    if not np.any(out > 0):
        return out, {"cut_pixels_assigned": 0, "cut_pixels_unassigned": int(np.count_nonzero(zero_cut))}
    indices = ndimage.distance_transform_edt(out == 0, return_distances=False, return_indices=True)
    nearest_labels = out[indices[0], indices[1]]
    assignable = zero_cut & (nearest_labels > 0)
    out[assignable] = nearest_labels[assignable]
    unassigned = int(np.count_nonzero(zero_cut & (out == 0)))
    return out, {"cut_pixels_assigned": int(np.count_nonzero(assignable)), "cut_pixels_unassigned": unassigned}


def project_labels_to_output_domain(labels: np.ndarray, output_domain: np.ndarray) -> tuple[np.ndarray, dict]:
    label_map = np.asarray(labels, dtype=np.int32)
    domain = np.asarray(output_domain, dtype=bool)
    if label_map.shape != domain.shape:
        raise ValueError("labels and output_domain shape mismatch")
    out = np.zeros_like(label_map, dtype=np.int32)
    direct = domain & (label_map > 0)
    out[direct] = label_map[direct]
    missing = domain & (out == 0)
    if np.any(missing) and np.any(label_map > 0):
        indices = ndimage.distance_transform_edt(label_map <= 0, return_distances=False, return_indices=True)
        nearest = label_map[indices[0], indices[1]]
        fillable = missing & (nearest > 0)
        out[fillable] = nearest[fillable]
    unassigned = int(np.count_nonzero(domain & (out == 0)))
    return out, {
        "projected_direct_pixels": int(np.count_nonzero(direct)),
        "projected_nearest_fill_pixels": int(np.count_nonzero(missing & (out > 0))),
        "projected_unassigned_domain_pixels": unassigned,
    }


def connectivity_structure(connectivity: int) -> np.ndarray:
    if int(connectivity) == 8:
        return np.ones((3, 3), dtype=bool)
    return np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def disk(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((yy * yy + xx * xx) <= radius * radius).astype(bool)


def _nearest_neighbor_label(labels: np.ndarray, component: np.ndarray) -> int:
    others = np.asarray(labels, dtype=np.int32).copy()
    others[component] = 0
    if not np.any(others > 0):
        return 0
    indices = ndimage.distance_transform_edt(others == 0, return_distances=False, return_indices=True)
    nearest = others[indices[0], indices[1]]
    values, counts = np.unique(nearest[component], return_counts=True)
    candidates = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) > 0]
    if not candidates:
        return 0
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return int(candidates[0][0])
