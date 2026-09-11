#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxroom_online.isaac_runtime.evaluation.online_roomseg.annotation_schema import (
    load_annotation,
    validate_annotation,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.metrics import (
    compute_snapshot_metrics,
    min_area_cells_from_m2,
    prepare_metric_label_maps,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import (
    load_snapshot_arrays,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.step_backprojection import (
    backproject_final_gt_to_snapshot,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.validation import (
    validate_gt_label_map,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_map(index: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["episode_uid"]): dict(row) for row in index.get("episodes", [])}


def _snapshot_map(episode: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["coverage_event_id"]): dict(row)
        for row in episode.get("snapshots", [])
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "process_count": len({str(row["episode_uid"]) for row in rows}),
        "checkpoint_evaluation_count": len(rows),
    }
    for key, label in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("miou_room", "mIoU_room"),
    ):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        mean = float(np.mean(values)) if values.size else 0.0
        std = float(np.std(values)) if values.size else 0.0
        result[label] = mean
        result[label + "_percent"] = 100.0 * mean
        result[label + "_std"] = std
        result[label + "_std_percent"] = 100.0 * std
    return result


def _event_sort(value: str) -> tuple[int, str]:
    text = str(value)
    if text.startswith("milestone_"):
        try:
            return int(text.split("_")[-1]), text
        except ValueError:
            pass
    return (10_000 if text == "final" else 9_000), text


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "variant",
        "scene_id",
        "episode_uid",
        "coverage_event_id",
        "step",
        "coverage_ratio",
        "precision",
        "recall",
        "f1",
        "miou_room",
        "precision_percent",
        "recall_percent",
        "f1_percent",
        "miou_room_percent",
        "n_gt",
        "n_pred",
        "metric_domain_pixels",
        "source_snapshot_path",
        "prediction_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# VoxRoom ablation — %s / %s" % (summary["dataset"], summary["variant"]),
        "",
        "Every approved recorded checkpoint is evaluated once. Rooms smaller than 0.5 m² are ignored. P/R use the paper's region-overlap definitions; F1 is their per-checkpoint harmonic mean; room mIoU uses one-to-one Hungarian matching.",
        "",
        "| Scope | Processes | Checkpoints | P (%) | R (%) | F1 (%) | room mIoU (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ("all_approved", "excluding_training_and_validation_scenes"):
        value = summary["scopes"][scope]
        lines.append(
            "| %s | %d | %d | %.3f | %.3f | %.3f | %.3f |"
            % (
                scope,
                value["process_count"],
                value["checkpoint_evaluation_count"],
                value["Precision_percent"],
                value["Recall_percent"],
                value["F1_percent"],
                value["mIoU_room_percent"],
            )
        )
    lines.extend(
        [
            "",
            "## All-approved by checkpoint",
            "",
            "| Checkpoint | Processes | P (%) | R (%) | F1 (%) | room mIoU (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for event_id in sorted(summary["by_checkpoint_all_approved"], key=_event_sort):
        value = summary["by_checkpoint_all_approved"][event_id]
        lines.append(
            "| %s | %d | %.3f | %.3f | %.3f | %.3f |"
            % (
                event_id,
                value["process_count"],
                value["Precision_percent"],
                value["Recall_percent"],
                value["F1_percent"],
                value["mIoU_room_percent"],
            )
        )
    replay = summary.get("replay_contract", {})
    if replay.get("source_history_limitation"):
        lines.extend(
            [
                "",
                "## Replay limitation",
                "",
                str(replay["source_history_limitation"]),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one replayed VoxRoom ablation with P/R/F1/room-mIoU.")
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--prediction-index", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--exclusion-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", choices=["interioragent", "grscene"], required=True)
    parser.add_argument("--min-room-area-m2", type=float, default=0.5)
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    args = parser.parse_args()

    source_index_path = Path(args.source_index).expanduser().resolve()
    prediction_index_path = Path(args.prediction_index).expanduser().resolve()
    annotation_dir = Path(args.annotation_dir).expanduser().resolve()
    gt_dir = Path(args.gt_dir).expanduser().resolve()
    exclusion_path = Path(args.exclusion_manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    prediction_index = json.loads(prediction_index_path.read_text(encoding="utf-8"))
    if not bool(prediction_index.get("complete")):
        raise ValueError("prediction replay is not complete")
    variant = str(prediction_index["variant"])
    source_episodes = _episode_map(source_index)
    prediction_episodes = _episode_map(prediction_index)
    excluded = set(
        str(value)
        for value in json.loads(exclusion_path.read_text(encoding="utf-8")).get(
            "excluded_scene_ids", []
        )
    )
    min_area_cells = min_area_cells_from_m2(
        float(args.min_room_area_m2), float(args.cell_size_m)
    )
    rows: list[dict[str, Any]] = []
    for uid, prediction_episode in prediction_episodes.items():
        if uid not in source_episodes:
            raise ValueError("prediction episode absent from source index: %s" % uid)
        source_episode = source_episodes[uid]
        annotation_path = annotation_dir / uid / "last_step.annotation.json"
        gt_path = gt_dir / uid / "last_step.gt_labels.npy"
        if not annotation_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError("approved GT is incomplete: %s" % uid)
        final_snapshot = load_snapshot_arrays(Path(source_episode["last_snapshot_path"]))
        annotation = load_annotation(annotation_path)
        if annotation.review.status != "approved":
            raise ValueError("prediction index contains unapproved episode: %s" % uid)
        validate_annotation(annotation, final_snapshot)
        final_gt = np.asarray(np.load(gt_path), dtype=np.int32)
        validate_gt_label_map(final_gt, final_snapshot.shape, domain=final_snapshot.eval_domain)
        source_snapshots = _snapshot_map(source_episode)
        for prediction_record in prediction_episode.get("snapshots", []):
            event_id = str(prediction_record["coverage_event_id"])
            source_record = source_snapshots[event_id]
            source_snapshot = load_snapshot_arrays(Path(source_record["snapshot_path"]))
            step_gt = backproject_final_gt_to_snapshot(
                final_gt,
                source_snapshot,
                episode_uid=uid,
                source_final_step=int(source_episode["last_snapshot_step"]),
            ).label_map
            prediction_path = Path(prediction_record["snapshot_path"])
            with np.load(prediction_path, allow_pickle=False) as arrays:
                pred = np.asarray(arrays["final_room_label_map"], dtype=np.int32)
            prepared = prepare_metric_label_maps(
                step_gt,
                pred,
                min_room_area_cells=min_area_cells,
            )
            metric = compute_snapshot_metrics(prepared.gt, prepared.pred)
            precision = float(metric["precision"])
            recall = float(metric["recall"])
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "dataset": str(args.dataset),
                    "variant": variant,
                    "scene_id": str(source_episode["scene_id"]),
                    "episode_uid": uid,
                    "coverage_event_id": event_id,
                    "step": int(source_record["step"]),
                    "coverage_ratio": source_record.get("coverage_ratio"),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "miou_room": float(metric["miou_room"]),
                    "precision_percent": 100.0 * precision,
                    "recall_percent": 100.0 * recall,
                    "f1_percent": 100.0 * f1,
                    "miou_room_percent": 100.0 * float(metric["miou_room"]),
                    "n_gt": int(metric["n_gt"]),
                    "n_pred": int(metric["n_pred"]),
                    "metric_domain_pixels": int(np.count_nonzero(prepared.metric_domain)),
                    "source_snapshot_path": str(source_record["snapshot_path"]),
                    "prediction_path": str(prediction_path),
                }
            )
    rows.sort(key=lambda row: (row["scene_id"], _event_sort(str(row["coverage_event_id"]))))
    excluded_rows = [row for row in rows if str(row["scene_id"]) not in excluded]
    by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_checkpoint[str(row["coverage_event_id"])].append(row)
    summary = {
        "schema_version": "voxroom_ablation_region_metrics_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "variant": variant,
        "metric_protocol": "paper_region_overlap_precision_recall_all_recorded_checkpoints",
        "precision_definition": "mean_pred_room(max_gt(intersection/pred_room_area))",
        "recall_definition": "mean_gt_room(max_pred(intersection/gt_room_area))",
        "f1_definition": "per_checkpoint_harmonic_mean_of_precision_and_recall_then_macro_average",
        "miou_room_definition": "hungarian_one_to_one_matched_room_iou_sum_divided_by_gt_room_count_unmatched_gt_zero",
        "min_room_area_m2": float(args.min_room_area_m2),
        "cell_size_m": float(args.cell_size_m),
        "min_room_area_cells": int(min_area_cells),
        "scopes": {
            "all_approved": _aggregate(rows),
            "excluding_training_and_validation_scenes": _aggregate(excluded_rows),
        },
        "by_checkpoint_all_approved": {
            event_id: _aggregate(selected)
            for event_id, selected in sorted(by_checkpoint.items(), key=lambda item: _event_sort(item[0]))
        },
        "excluded_scene_ids": sorted(excluded),
        "replay_contract": {
            key: prediction_index.get(key)
            for key in (
                "candidate_policy",
                "source_history_limitation",
                "checkpoint",
                "checkpoint_sha256",
                "learning_config",
            )
        },
        "inputs": {
            "source_index": str(source_index_path),
            "source_index_sha256": _sha256(source_index_path),
            "prediction_index": str(prediction_index_path),
            "prediction_index_sha256": _sha256(prediction_index_path),
            "annotation_dir": str(annotation_dir),
            "gt_dir": str(gt_dir),
            "exclusion_manifest": str(exclusion_path),
            "exclusion_manifest_sha256": _sha256(exclusion_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "per_checkpoint_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "per_checkpoint_metrics.csv", rows)
    _write_report(out_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
