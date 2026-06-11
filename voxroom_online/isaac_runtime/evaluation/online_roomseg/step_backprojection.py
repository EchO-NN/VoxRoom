from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import positive_labels, relabel_subset_sequential, write_json_atomic
from .snapshot_io import SnapshotArrays, load_snapshot_arrays
from .visualization import save_step_gt_overlay


@dataclass(frozen=True)
class StepGt:
    label_map: np.ndarray
    metric_domain: np.ndarray
    metadata: dict


def backproject_final_gt_to_snapshot(
    final_gt: np.ndarray,
    snapshot: SnapshotArrays,
    *,
    episode_uid: str,
    source_final_step: int,
    min_visible_gt_cells: int = 1,
) -> StepGt:
    final = np.asarray(final_gt, dtype=np.int32)
    if final.shape != snapshot.shape:
        raise ValueError("final_gt shape does not match snapshot shape")
    raw_domain = np.asarray(snapshot.segmentation_domain, dtype=bool)
    annotation_domain = final > 0
    metric_domain = raw_domain & annotation_domain
    gt_t = np.zeros(final.shape, dtype=np.int32)
    gt_t[metric_domain] = final[metric_domain]
    visible: list[int] = []
    areas: dict[str, int] = {}
    for label in positive_labels(gt_t):
        area = int(np.count_nonzero(gt_t == int(label)))
        if area >= int(min_visible_gt_cells):
            visible.append(int(label))
            areas[str(int(label))] = int(area)
    gt_t = relabel_subset_sequential(gt_t, visible)
    relabeled_areas = {str(int(label)): int(np.count_nonzero(gt_t == int(label))) for label in positive_labels(gt_t)}
    metadata = {
        "episode_uid": str(episode_uid),
        "step": int(snapshot.step),
        "source_final_step": int(source_final_step),
        "snapshot_path": str(snapshot.path),
        "domain_key": str(snapshot.domain_key),
        "metric_domain_key": str(snapshot.segmentation_domain_key),
        "navigation_domain_pixels": int(np.count_nonzero(snapshot.eval_domain)),
        "raw_domain_pixels": int(np.count_nonzero(raw_domain)),
        "annotation_domain_pixels": int(np.count_nonzero(annotation_domain)),
        "metric_domain_pixels": int(np.count_nonzero(metric_domain)),
        "dropped_unannotated_domain_pixels": int(np.count_nonzero(raw_domain & ~annotation_domain)),
        "visible_gt_room_count": int(len(positive_labels(gt_t))),
        "visible_gt_room_areas_cells": relabeled_areas,
        "min_visible_gt_cells": int(min_visible_gt_cells),
    }
    return StepGt(label_map=gt_t.astype(np.int32), metric_domain=metric_domain.astype(bool), metadata=metadata)


def write_step_gt(
    *,
    step_gt: StepGt,
    snapshot: SnapshotArrays,
    out_dir: Path,
    stem: str,
    extra_metadata: dict | None = None,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / f"{stem}.gt_labels.npy"
    overlay_path = out_dir / f"{stem}.gt_overlay.png"
    metadata_path = out_dir / f"{stem}.gt_metadata.json"
    np.save(labels_path, np.asarray(step_gt.label_map, dtype=np.int32))
    save_step_gt_overlay(
        overlay_path,
        gt=step_gt.label_map,
        metric_domain=step_gt.metric_domain,
        raw_domain=snapshot.segmentation_domain,
        title=f"step {snapshot.step} GT",
    )
    metadata = dict(step_gt.metadata)
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    metadata.update({"gt_label_npy": str(labels_path), "gt_overlay_png": str(overlay_path), "gt_metadata_json": str(metadata_path)})
    write_json_atomic(metadata_path, metadata)
    return {"gt_label_npy": str(labels_path), "gt_overlay_png": str(overlay_path), "gt_metadata_json": str(metadata_path)}


def backproject_episode(
    *,
    episode: dict,
    final_gt_path: Path,
    step_gt_dir: Path,
    min_visible_gt_cells: int = 1,
    snapshot_policy: str = "all",
    extra_metadata: dict | None = None,
) -> list[dict]:
    final_gt = np.asarray(np.load(final_gt_path), dtype=np.int32)
    snapshots = list(episode.get("snapshots", []))
    if snapshot_policy == "last":
        snapshots = [snap for snap in snapshots if bool(snap.get("is_last"))]
    outputs: list[dict] = []
    ep_dir = Path(step_gt_dir) / str(episode["episode_uid"])
    for snap_record in snapshots:
        snapshot = load_snapshot_arrays(Path(snap_record["snapshot_path"]))
        step_gt = backproject_final_gt_to_snapshot(
            final_gt,
            snapshot,
            episode_uid=str(episode["episode_uid"]),
            source_final_step=int(episode["last_snapshot_step"]),
            min_visible_gt_cells=int(min_visible_gt_cells),
        )
        stem = Path(snap_record["snapshot_path"]).stem
        paths = write_step_gt(step_gt=step_gt, snapshot=snapshot, out_dir=ep_dir, stem=stem, extra_metadata=extra_metadata)
        outputs.append({**step_gt.metadata, **(extra_metadata or {}), **paths})
    return outputs
