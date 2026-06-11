from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.evaluation.online_roomseg.metrics import (
    compute_iou_matrix,
    compute_matched_room_miou,
    compute_room_count_rates,
    compute_snapshot_metrics,
    filter_small_label_masks,
    min_area_cells_from_m2,
    prepare_metric_label_maps,
)


def test_usr_osr_basic_cases() -> None:
    assert compute_room_count_rates(2, 1) == (0.5, 0.0)
    assert compute_room_count_rates(2, 3) == (0.0, 0.5)
    assert compute_room_count_rates(2, 2) == (0.0, 0.0)


def test_min_area_cells_from_m2_uses_ceiling() -> None:
    assert min_area_cells_from_m2(0.3, 0.05) == 120


def test_hungarian_miou_perfect_match() -> None:
    gt = np.array([[1, 1, 2, 2]], dtype=np.int32)
    pred = np.array([[1, 1, 2, 2]], dtype=np.int32)

    miou, pairs, unmatched, _ = compute_matched_room_miou(gt, pred)

    assert np.isclose(miou, 1.0)
    assert len(pairs) == 2
    assert unmatched == []


def test_hungarian_miou_undersegmentation_counts_unmatched_gt_as_zero() -> None:
    gt = np.array([[1, 1, 2, 2]], dtype=np.int32)
    pred = np.array([[1, 1, 1, 1]], dtype=np.int32)
    iou, gt_labels, pred_labels = compute_iou_matrix(gt, pred)

    miou, pairs, unmatched, _ = compute_matched_room_miou(gt, pred)

    assert gt_labels == [1, 2]
    assert pred_labels == [1]
    assert np.allclose(iou, np.array([[0.5], [0.5]]))
    assert np.isclose(miou, 0.25)
    assert sum(1 for pair in pairs if pair["pred_label"] is None) == 1
    assert unmatched == []


def test_empty_prediction_has_zero_miou_and_full_usr() -> None:
    gt = np.array([[1, 2]], dtype=np.int32)
    pred = np.zeros_like(gt)
    metric = compute_snapshot_metrics(gt, pred, csr=None)

    assert metric["n_gt"] == 2
    assert metric["n_pred"] == 0
    assert np.isclose(metric["usr"], 1.0)
    assert np.isclose(metric["miou_room"], 0.0)


def test_filter_small_label_masks_removes_labels_below_threshold() -> None:
    labels = np.zeros((4, 5), dtype=np.int32)
    labels[:2, :2] = 1
    labels[:, 3:] = 2

    filtered = filter_small_label_masks(labels, min_area_cells=5)

    assert set(np.unique(filtered)) == {0, 1}
    assert np.count_nonzero(filtered == 1) == 8


def test_metric_preparation_filters_gt_and_pred_before_scoring() -> None:
    gt = np.zeros((4, 5), dtype=np.int32)
    gt[:2, :2] = 1
    gt[:, 3:] = 2
    pred = np.zeros_like(gt)
    pred[:2, :2] = 1
    pred[:, 3:] = 2

    prepared = prepare_metric_label_maps(gt, pred, min_room_area_cells=5)
    metric = compute_snapshot_metrics(prepared.gt, prepared.pred, csr=None)

    assert prepared.stats["gt_label_masks_filtered_small"] == 1
    assert prepared.stats["pred_label_masks_filtered_small"] == 1
    assert np.count_nonzero(prepared.metric_domain) == 8
    assert metric["n_gt"] == 1
    assert metric["n_pred"] == 1
    assert np.isclose(metric["miou_room"], 1.0)
