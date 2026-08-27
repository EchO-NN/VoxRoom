from __future__ import annotations

from typing import Iterable

import numpy as np


def binary_classification_metrics(labels: Iterable[int], probabilities: Iterable[float], threshold: float) -> dict[str, float | int]:
    y = np.asarray(list(labels), dtype=np.int32)
    p = np.asarray(list(probabilities), dtype=np.float64)
    _validate_binary_inputs(y, p)
    prediction = p >= float(threshold)
    positive = y == 1
    negative = ~positive
    tp = int(np.count_nonzero(prediction & positive))
    fp = int(np.count_nonzero(prediction & negative))
    fn = int(np.count_nonzero(~prediction & positive))
    tn = int(np.count_nonzero(~prediction & negative))
    precision = float(tp) / float(max(1, tp + fp))
    recall = float(tp) / float(max(1, tp + fn))
    rejected_seed_count = int(tn + fn)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "negative_rejection_rate": float(tn) / float(max(1, tn + fp)),
        "rejected_seed_accuracy": float(tn) / float(max(1, rejected_seed_count)),
        "rejected_seed_count": rejected_seed_count,
        "accuracy": float(tp + tn) / float(max(1, len(y))),
    }


def choose_threshold_for_recall(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    target_recall: float,
) -> dict[str, float | int]:
    y = np.asarray(list(labels), dtype=np.int32)
    p = np.asarray(list(probabilities), dtype=np.float64)
    _validate_binary_inputs(y, p)
    if not np.any(y == 1):
        raise ValueError("threshold selection requires at least one positive sample")
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    feasible: list[dict[str, float | int]] = []
    for threshold in candidates:
        metrics = binary_classification_metrics(y, p, float(threshold))
        if float(metrics["recall"]) + 1e-12 >= float(target_recall):
            feasible.append(metrics)
    if not feasible:
        raise RuntimeError("no threshold satisfies target recall")
    feasible.sort(key=lambda item: (float(item["negative_rejection_rate"]), float(item["precision"]), float(item["threshold"])), reverse=True)
    return feasible[0]


def ranking_metrics(labels: Iterable[int], probabilities: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(labels), dtype=np.int32)
    p = np.asarray(list(probabilities), dtype=np.float64)
    _validate_binary_inputs(y, p)
    return {"pr_auc": _pr_auc(y, p), "roc_auc": _roc_auc(y, p)}


def recall_rejection_table(labels: Iterable[int], probabilities: Iterable[float], targets=(0.95, 0.98, 0.99)) -> dict[str, dict[str, float | int]]:
    y = list(labels)
    p = list(probabilities)
    return {"recall_%.2f" % float(target): choose_threshold_for_recall(y, p, target_recall=float(target)) for target in targets}


def _pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    positives = int(np.count_nonzero(y == 1))
    if positives == 0:
        return 0.0
    order = np.argsort(-p, kind="stable")
    sorted_y = y[order]
    tp = np.cumsum(sorted_y == 1)
    fp = np.cumsum(sorted_y == 0)
    recall = tp / float(positives)
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.trapz(precision, recall))


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    positives = int(np.count_nonzero(y == 1))
    negatives = int(np.count_nonzero(y == 0))
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and p[order[end]] == p[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(np.sum(ranks[y == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)


def _validate_binary_inputs(y: np.ndarray, p: np.ndarray) -> None:
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape:
        raise ValueError("labels and probabilities must be same-length vectors")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("labels must be binary")
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be in [0,1]")
