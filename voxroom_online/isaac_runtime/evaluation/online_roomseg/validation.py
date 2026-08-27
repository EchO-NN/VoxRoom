from __future__ import annotations

from pathlib import Path

import numpy as np

from .annotation_schema import RoomsegAnnotation, approved_or_raise
from .common import positive_labels, relabel_positive_labels_sequentially
from .snapshot_io import SnapshotArrays


def validate_gt_label_map(gt: np.ndarray, snapshot_shape: tuple[int, int], *, allow_unlabeled_domain: bool = False, domain: np.ndarray | None = None) -> np.ndarray:
    arr = relabel_positive_labels_sequentially(np.asarray(gt, dtype=np.int32))
    if arr.shape != tuple(snapshot_shape):
        raise ValueError("gt label map shape mismatch")
    if len(positive_labels(arr)) < 1:
        raise ValueError("gt room_count must be >= 1")
    if domain is not None and not allow_unlabeled_domain:
        missing = np.asarray(domain, dtype=bool) & (arr == 0)
        if np.any(missing):
            raise ValueError("gt has unlabeled domain pixels: %d" % int(np.count_nonzero(missing)))
    return arr


def validate_approved_annotation(annotation: RoomsegAnnotation, snapshot: SnapshotArrays) -> None:
    _ = snapshot
    approved_or_raise(annotation)


def validate_metric_inputs(gt: np.ndarray, pred: np.ndarray, metric_domain: np.ndarray, *, strict_paper: bool, csr_exists: bool, require_csr: bool) -> str | None:
    if np.asarray(gt).shape != np.asarray(pred).shape:
        return "shape_mismatch"
    if not np.any(np.asarray(metric_domain, dtype=bool)):
        return "empty_metric_domain"
    if len(positive_labels(gt)) < 1:
        return "no_gt_rooms"
    if require_csr and not bool(csr_exists):
        return "missing_csr"
    return None


def require_file(path: Path, reason: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError("%s: %s" % (reason, path))
