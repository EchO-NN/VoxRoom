from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import positive_labels, relabel_positive_labels_sequentially


@dataclass(frozen=True)
class RoomsegSnapshotMetric:
    n_gt: int
    n_pred: int
    usr: float
    osr: float
    miou_room: float
    matched_pairs: list[dict[str, Any]]
    unmatched_pred_labels: list[int]


@dataclass(frozen=True)
class PreparedMetricLabelMaps:
    gt: np.ndarray
    pred: np.ndarray
    metric_domain: np.ndarray
    stats: dict[str, Any]


def compute_room_count_rates(n_gt: int, n_pred: int) -> tuple[float, float]:
    ng = int(n_gt)
    npred = int(n_pred)
    denom = max(ng, 1)
    usr = max(ng - npred, 0) / float(denom)
    osr = max(npred - ng, 0) / float(denom)
    return float(usr), float(osr)


def compute_iou_matrix(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    gt_arr = np.asarray(gt, dtype=np.int32)
    pred_arr = np.asarray(pred, dtype=np.int32)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError("gt and pred shapes must match")
    gt_labels = positive_labels(gt_arr)
    pred_labels = positive_labels(pred_arr)
    if not gt_labels or not pred_labels:
        return np.zeros((len(gt_labels), len(pred_labels)), dtype=np.float64), gt_labels, pred_labels
    gt_remap = np.zeros(int(max(gt_labels)) + 1, dtype=np.int32)
    pred_remap = np.zeros(int(max(pred_labels)) + 1, dtype=np.int32)
    for i, lab in enumerate(gt_labels, start=1):
        gt_remap[int(lab)] = int(i)
    for j, lab in enumerate(pred_labels, start=1):
        pred_remap[int(lab)] = int(j)
    g = gt_remap[gt_arr]
    p = pred_remap[pred_arr]
    valid = (g > 0) | (p > 0)
    pair_index = g[valid] * (len(pred_labels) + 1) + p[valid]
    counts = np.bincount(pair_index, minlength=(len(gt_labels) + 1) * (len(pred_labels) + 1))
    counts = counts.reshape((len(gt_labels) + 1, len(pred_labels) + 1))
    inter = counts[1:, 1:].astype(np.float64)
    gt_area = np.bincount(g[g > 0], minlength=len(gt_labels) + 1)[1:].astype(np.float64)
    pred_area = np.bincount(p[p > 0], minlength=len(pred_labels) + 1)[1:].astype(np.float64)
    union = gt_area[:, None] + pred_area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0), gt_labels, pred_labels


def compute_matched_room_miou(gt: np.ndarray, pred: np.ndarray) -> tuple[float, list[dict[str, Any]], list[int], np.ndarray]:
    iou, gt_labels, pred_labels = compute_iou_matrix(gt, pred)
    n_gt = len(gt_labels)
    n_pred = len(pred_labels)
    if n_gt == 0 or n_pred == 0:
        pairs = [{"gt_label": int(label), "pred_label": None, "iou": 0.0} for label in gt_labels]
        return 0.0, pairs, [int(v) for v in pred_labels], iou
    row_ind, col_ind = linear_sum_assignment(-iou)
    matched_by_row = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    matched_pred = set(matched_by_row.values())
    total = 0.0
    pairs: list[dict[str, Any]] = []
    for i, gt_label in enumerate(gt_labels):
        if i in matched_by_row:
            j = matched_by_row[i]
            value = float(iou[i, j])
            total += value
            pairs.append({"gt_label": int(gt_label), "pred_label": int(pred_labels[j]), "iou": value})
        else:
            pairs.append({"gt_label": int(gt_label), "pred_label": None, "iou": 0.0})
    unmatched_pred = [int(label) for j, label in enumerate(pred_labels) if j not in matched_pred]
    return float(total / float(max(n_gt, 1))), pairs, unmatched_pred, iou


def compute_snapshot_metrics(gt: np.ndarray, pred: np.ndarray, csr: int | None = None, *, min_room_area_cells: int = 0) -> dict[str, Any]:
    if int(min_room_area_cells) > 1:
        prepared = prepare_metric_label_maps(gt, pred, min_room_area_cells=int(min_room_area_cells))
        gt_arr = prepared.gt
        pred_arr = prepared.pred
        area_filter_stats = prepared.stats
    else:
        gt_arr = relabel_positive_labels_sequentially(np.asarray(gt, dtype=np.int32))
        pred_arr = relabel_positive_labels_sequentially(np.asarray(pred, dtype=np.int32))
        area_filter_stats = {
            "min_room_area_cells": int(max(0, int(min_room_area_cells))),
            "gt_label_masks_before_filter": len(positive_labels(gt_arr)),
            "gt_label_masks_after_filter": len(positive_labels(gt_arr)),
            "gt_label_masks_filtered_small": 0,
            "pred_label_masks_before_filter": len(positive_labels(pred_arr)),
            "pred_label_masks_after_filter": len(positive_labels(pred_arr)),
            "pred_label_masks_filtered_small": 0,
            "gt_cells_before_filter": int(np.count_nonzero(gt_arr > 0)),
            "gt_cells_after_filter": int(np.count_nonzero(gt_arr > 0)),
            "pred_cells_before_filter": int(np.count_nonzero(pred_arr > 0)),
            "pred_cells_after_filter": int(np.count_nonzero(pred_arr > 0)),
        }
    n_gt = len(positive_labels(gt_arr))
    n_pred = len(positive_labels(pred_arr))
    usr, osr = compute_room_count_rates(n_gt, n_pred)
    miou, pairs, unmatched_pred, iou = compute_matched_room_miou(gt_arr, pred_arr)
    return {
        "n_gt": int(n_gt),
        "n_pred": int(n_pred),
        "usr": float(usr),
        "osr": float(osr),
        "miou_room": float(miou),
        "csr": None if csr is None else int(csr),
        "matched_pairs": pairs,
        "unmatched_pred_labels": unmatched_pred,
        "iou_matrix": iou,
        **area_filter_stats,
    }


def clip_prediction_to_metric_domain(pred_raw: np.ndarray, metric_domain: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(np.asarray(pred_raw, dtype=np.int32), dtype=np.int32)
    domain = np.asarray(metric_domain, dtype=bool)
    pred[domain] = np.asarray(pred_raw, dtype=np.int32)[domain]
    return relabel_positive_labels_sequentially(pred)


def min_area_cells_from_m2(min_area_m2: float, cell_size_m: float) -> int:
    if float(min_area_m2) <= 0.0:
        return 0
    if float(cell_size_m) <= 0.0:
        raise ValueError("cell_size_m must be positive")
    cells = float(min_area_m2) / max(float(cell_size_m) ** 2, 1e-12)
    return int(math.ceil(cells - 1e-9))


def filter_small_label_masks(label_map: np.ndarray, *, min_area_cells: int) -> np.ndarray:
    filtered, _ = _filter_small_label_masks_with_stats(label_map, min_area_cells=int(min_area_cells))
    return filtered


def prepare_metric_label_maps(gt: np.ndarray, pred_raw: np.ndarray, *, min_room_area_cells: int) -> PreparedMetricLabelMaps:
    gt_raw = relabel_positive_labels_sequentially(np.asarray(gt, dtype=np.int32))
    pred_raw_arr = np.asarray(pred_raw, dtype=np.int32)
    if gt_raw.shape != pred_raw_arr.shape:
        raise ValueError("gt and pred shapes must match")

    pred_in_raw_domain = clip_prediction_to_metric_domain(pred_raw_arr, gt_raw > 0)
    gt_filtered, gt_stats = _filter_small_label_masks_with_stats(gt_raw, min_area_cells=int(min_room_area_cells))
    pred_small_filtered, pred_stats_initial = _filter_small_label_masks_with_stats(
        pred_in_raw_domain,
        min_area_cells=int(min_room_area_cells),
    )
    metric_domain = gt_filtered > 0
    pred_in_filtered_domain = clip_prediction_to_metric_domain(pred_small_filtered, metric_domain)
    pred_filtered, pred_stats_final = _filter_small_label_masks_with_stats(
        pred_in_filtered_domain,
        min_area_cells=int(min_room_area_cells),
    )
    stats = {
        "min_room_area_cells": int(max(0, int(min_room_area_cells))),
        "gt_label_masks_before_filter": int(gt_stats["label_masks_before_filter"]),
        "gt_label_masks_after_filter": int(gt_stats["label_masks_after_filter"]),
        "gt_label_masks_filtered_small": int(gt_stats["label_masks_filtered_small"]),
        "gt_cells_before_filter": int(gt_stats["cells_before_filter"]),
        "gt_cells_after_filter": int(gt_stats["cells_after_filter"]),
        "pred_label_masks_before_filter": int(pred_stats_initial["label_masks_before_filter"]),
        "pred_label_masks_after_filter": int(pred_stats_final["label_masks_after_filter"]),
        "pred_label_masks_filtered_small": int(
            pred_stats_initial["label_masks_filtered_small"]
            + max(0, int(pred_stats_initial["label_masks_after_filter"]) - int(pred_stats_final["label_masks_after_filter"]))
        ),
        "pred_cells_before_filter": int(pred_stats_initial["cells_before_filter"]),
        "pred_cells_after_filter": int(pred_stats_final["cells_after_filter"]),
    }
    return PreparedMetricLabelMaps(
        gt=gt_filtered,
        pred=pred_filtered,
        metric_domain=metric_domain.astype(bool),
        stats=stats,
    )


def _filter_small_label_masks_with_stats(label_map: np.ndarray, *, min_area_cells: int) -> tuple[np.ndarray, dict[str, int]]:
    arr = relabel_positive_labels_sequentially(np.asarray(label_map, dtype=np.int32))
    labels = positive_labels(arr)
    out = arr.copy()
    removed = 0
    removed_cells = 0
    threshold = int(max(0, int(min_area_cells)))
    for label in labels:
        mask = arr == int(label)
        cells = int(np.count_nonzero(mask))
        if threshold > 1 and cells < threshold:
            out[mask] = 0
            removed += 1
            removed_cells += cells
    out = relabel_positive_labels_sequentially(out)
    return out, {
        "label_masks_before_filter": int(len(labels)),
        "label_masks_after_filter": int(len(positive_labels(out))),
        "label_masks_filtered_small": int(removed),
        "cells_before_filter": int(np.count_nonzero(arr > 0)),
        "cells_after_filter": int(np.count_nonzero(out > 0)),
        "cells_filtered_small": int(removed_cells),
    }
