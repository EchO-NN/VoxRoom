#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from voxroom_online.isaac_runtime.evaluation.online_roomseg.annotation_schema import (
    load_annotation,
    validate_annotation,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.common import (
    write_json_atomic,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.metrics import (
    compute_snapshot_metrics,
    min_area_cells_from_m2,
    prepare_metric_label_maps,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_index import (
    load_index,
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
from voxroom_online.isaac_runtime.evaluation.online_roomseg.visualization import (
    save_match_visualization,
)


METHODS = ("voxroom", "tvars_original")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_map(index: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {
        str(episode["episode_uid"]): dict(episode)
        for episode in index.get("episodes", [])
    }
    if len(rows) != len(index.get("episodes", [])):
        raise ValueError("index contains duplicate episode_uid values")
    return rows


def _dataset_name(episode: Mapping[str, Any]) -> str:
    snapshot = str(episode.get("last_snapshot_path", ""))
    if "/interioragent/" in snapshot:
        return "interioragent"
    if "/habitat/" in snapshot:
        return "habitat"
    if "/grscene/" in snapshot:
        return "grscene"
    return "unknown"


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0)) if values else 0.0


def _aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    precision = [float(row["precision"]) for row in selected]
    recall = [float(row["recall"]) for row in selected]
    f1 = [float(row["f1"]) for row in selected]
    miou_room = [float(row["miou_room"]) for row in selected]
    return {
        "evaluation_count": len(selected),
        "process_count": len({str(row["episode_uid"]) for row in selected}),
        "Precision": _mean(precision),
        "Recall": _mean(recall),
        "F1": _mean(f1),
        "mIoU_room": _mean(miou_room),
        "Precision_percent": 100.0 * _mean(precision),
        "Recall_percent": 100.0 * _mean(recall),
        "F1_percent": 100.0 * _mean(f1),
        "mIoU_room_percent": 100.0 * _mean(miou_room),
        "scene_std_Precision": _std(precision),
        "scene_std_Recall": _std(recall),
        "scene_std_F1": _std(f1),
        "scene_std_mIoU_room": _std(miou_room),
        "scene_std_Precision_percent": 100.0 * _std(precision),
        "scene_std_Recall_percent": 100.0 * _std(recall),
        "scene_std_F1_percent": 100.0 * _std(f1),
        "scene_std_mIoU_room_percent": 100.0 * _std(miou_room),
    }


def _paired_delta_summary(
    paired_rows: list[dict[str, Any]],
    method_summary: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "Precision_percentage_points": float(
            method_summary["voxroom"]["Precision_percent"]
            - method_summary["tvars_original"]["Precision_percent"]
        ),
        "Recall_percentage_points": float(
            method_summary["voxroom"]["Recall_percent"]
            - method_summary["tvars_original"]["Recall_percent"]
        ),
        "F1_percentage_points": float(
            method_summary["voxroom"]["F1_percent"]
            - method_summary["tvars_original"]["F1_percent"]
        ),
        "mIoU_room_percentage_points": float(
            method_summary["voxroom"]["mIoU_room_percent"]
            - method_summary["tvars_original"]["mIoU_room_percent"]
        ),
        "voxroom_precision_wins": int(
            sum(row["delta_precision_points"] > 0.0 for row in paired_rows)
        ),
        "voxroom_recall_wins": int(
            sum(row["delta_recall_points"] > 0.0 for row in paired_rows)
        ),
        "voxroom_f1_wins": int(
            sum(row["delta_f1_points"] > 0.0 for row in paired_rows)
        ),
        "voxroom_miou_room_wins": int(
            sum(row["delta_miou_room_points"] > 0.0 for row in paired_rows)
        ),
        "precision_ties": int(
            sum(
                bool(np.isclose(row["delta_precision_points"], 0.0))
                for row in paired_rows
            )
        ),
        "recall_ties": int(
            sum(
                bool(np.isclose(row["delta_recall_points"], 0.0))
                for row in paired_rows
            )
        ),
        "f1_ties": int(
            sum(
                bool(np.isclose(row["delta_f1_points"], 0.0))
                for row in paired_rows
            )
        ),
        "miou_room_ties": int(
            sum(
                bool(np.isclose(row["delta_miou_room_points"], 0.0))
                for row in paired_rows
            )
        ),
    }


def _snapshot_map(episode: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {
        str(snapshot["coverage_event_id"]): dict(snapshot)
        for snapshot in episode.get("snapshots", [])
    }
    if len(rows) != len(episode.get("snapshots", [])):
        raise ValueError(
            "episode contains duplicate coverage_event_id values: %s"
            % episode.get("episode_uid")
        )
    return rows


def _checkpoint_sort_key(event_id: str) -> tuple[int, str]:
    text = str(event_id)
    if text.startswith("milestone_"):
        try:
            return int(text.removeprefix("milestone_")), text
        except ValueError:
            pass
    if text == "final":
        return 10_000, text
    return 9_000, text


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "scene_id",
        "episode_uid",
        "method",
        "coverage_event_id",
        "coverage_event_kind",
        "coverage_ratio",
        "coverage_threshold",
        "is_last",
        "step",
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
        "gt_filtered_small",
        "pred_filtered_small",
        "snapshot_path",
        "match_visualization",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _write_report(
    path: Path,
    *,
    summary: Mapping[str, Any],
    paired_rows: list[dict[str, Any]],
) -> None:
    vox = summary["methods"]["voxroom"]
    tvars = summary["methods"]["tvars_original"]
    report_scope = str(summary.get("dataset") or "All approved datasets")
    lines = [
        "# %s — P / R / F1 / room mIoU on Approved GT" % report_scope,
        "",
        "Protocol: every recorded segmentation checkpoint (20%, 40%, 60%, 70%, 80%, 90%, and final) is evaluated once; the same checkpoint GT domain is used for both methods; rooms smaller than {area:g} m^2 are ignored.".format(
            area=float(summary["min_room_area_m2"])
        ),
        "",
        "P = mean over predicted rooms of their largest overlap fraction with any GT room.",
        "",
        "R = mean over GT rooms of their largest overlap fraction with any predicted room.",
        "",
        "F1 = the per-checkpoint harmonic mean of P and R, averaged over checkpoints.",
        "",
        "room mIoU = mean GT-room IoU after one-to-one Hungarian matching; unmatched GT rooms contribute zero.",
        "",
        "| Method | Processes | Checkpoint evaluations | P (%) | R (%) | F1 (%) | room mIoU (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| VoxRoom | {processes} | {n} | {p:.3f} | {r:.3f} | {f1:.3f} | {miou:.3f} |".format(
            processes=vox["process_count"],
            n=vox["evaluation_count"],
            p=vox["Precision_percent"],
            r=vox["Recall_percent"],
            f1=vox["F1_percent"],
            miou=vox["mIoU_room_percent"],
        ),
        "| TVARS | {processes} | {n} | {p:.3f} | {r:.3f} | {f1:.3f} | {miou:.3f} |".format(
            processes=tvars["process_count"],
            n=tvars["evaluation_count"],
            p=tvars["Precision_percent"],
            r=tvars["Recall_percent"],
            f1=tvars["F1_percent"],
            miou=tvars["mIoU_room_percent"],
        ),
        "",
        "Paired delta (VoxRoom - TVARS): P {p:+.3f}, R {r:+.3f}, F1 {f1:+.3f}, room mIoU {miou:+.3f} points.".format(
            p=summary["paired_delta"]["Precision_percentage_points"],
            r=summary["paired_delta"]["Recall_percentage_points"],
            f1=summary["paired_delta"]["F1_percentage_points"],
            miou=summary["paired_delta"]["mIoU_room_percentage_points"],
        ),
        "",
        "## By checkpoint",
        "",
        "| Checkpoint | Processes | VoxRoom P | VoxRoom R | VoxRoom F1 | VoxRoom mIoU | TVARS P | TVARS R | TVARS F1 | TVARS mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for event_id in sorted(summary["by_checkpoint"], key=_checkpoint_sort_key):
        checkpoint = summary["by_checkpoint"][event_id]
        checkpoint_vox = checkpoint["voxroom"]
        checkpoint_tvars = checkpoint["tvars_original"]
        lines.append(
            "| {event} | {n} | {vp:.3f} | {vr:.3f} | {vf1:.3f} | {vm:.3f} | {tp:.3f} | {tr:.3f} | {tf1:.3f} | {tm:.3f} |".format(
                event=event_id,
                n=checkpoint_vox["process_count"],
                vp=checkpoint_vox["Precision_percent"],
                vr=checkpoint_vox["Recall_percent"],
                vf1=checkpoint_vox["F1_percent"],
                vm=checkpoint_vox["mIoU_room_percent"],
                tp=checkpoint_tvars["Precision_percent"],
                tr=checkpoint_tvars["Recall_percent"],
                tf1=checkpoint_tvars["F1_percent"],
                tm=checkpoint_tvars["mIoU_room_percent"],
            )
        )
    lines.extend([
        "",
        "## Per exploration process and checkpoint",
        "",
        "| Dataset | Scene | Checkpoint | Step | VoxRoom P | VoxRoom R | VoxRoom F1 | VoxRoom mIoU | TVARS P | TVARS R | TVARS F1 | TVARS mIoU |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in paired_rows:
        lines.append(
            "| {dataset} | {scene_id} | {coverage_event_id} | {step} | {voxroom_precision_percent:.3f} | {voxroom_recall_percent:.3f} | "
            "{voxroom_f1_percent:.3f} | {voxroom_miou_room_percent:.3f} | {tvars_precision_percent:.3f} | "
            "{tvars_recall_percent:.3f} | {tvars_f1_percent:.3f} | {tvars_miou_room_percent:.3f} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate every recorded VoxRoom and TVARS segmentation checkpoint with P/R/F1/room-mIoU on approved GT."
    )
    parser.add_argument("--voxroom-index", required=True)
    parser.add_argument("--tvars-index", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-room-area-m2", type=float, default=0.5)
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    args = parser.parse_args()

    voxroom_index_path = Path(args.voxroom_index).resolve()
    tvars_index_path = Path(args.tvars_index).resolve()
    annotation_dir = Path(args.annotation_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    paper_path = Path(args.paper).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    voxroom_by_uid = _episode_map(load_index(voxroom_index_path))
    tvars_by_uid = _episode_map(load_index(tvars_index_path))
    if set(voxroom_by_uid) != set(tvars_by_uid):
        raise ValueError("VoxRoom and TVARS indexes do not contain the same episodes")

    approved: list[str] = []
    annotation_hashes: dict[str, str] = {}
    gt_hashes: dict[str, str] = {}
    for uid in sorted(voxroom_by_uid):
        annotation_path = annotation_dir / uid / "last_step.annotation.json"
        if not annotation_path.is_file():
            continue
        annotation = load_annotation(annotation_path)
        if annotation.review.status != "approved":
            continue
        gt_path = gt_dir / uid / "last_step.gt_labels.npy"
        gt_meta_path = gt_dir / uid / "last_step.gt_metadata.json"
        if not gt_path.is_file() or not gt_meta_path.is_file():
            raise FileNotFoundError("approved annotation has no generated GT: %s" % uid)
        gt_meta = json.loads(gt_meta_path.read_text(encoding="utf-8"))
        if str(gt_meta.get("annotation_review_status")) != "approved":
            raise ValueError("generated GT is not approved: %s" % uid)
        approved.append(uid)
        annotation_hashes[uid] = _sha256(annotation_path)
        gt_hashes[uid] = _sha256(gt_path)
    if not approved:
        raise ValueError("no approved annotations found")

    min_area_cells = min_area_cells_from_m2(
        float(args.min_room_area_m2), float(args.cell_size_m)
    )
    rows: list[dict[str, Any]] = []
    by_checkpoint_method: dict[tuple[str, str, str], dict[str, Any]] = {}
    for uid in approved:
        vox_episode = voxroom_by_uid[uid]
        tvars_episode = tvars_by_uid[uid]
        vox_final_snapshot = load_snapshot_arrays(
            Path(vox_episode["last_snapshot_path"])
        )
        annotation = load_annotation(
            annotation_dir / uid / "last_step.annotation.json"
        )
        validate_annotation(annotation, vox_final_snapshot)
        gt = np.asarray(
            np.load(gt_dir / uid / "last_step.gt_labels.npy"), dtype=np.int32
        )
        validate_gt_label_map(
            gt, vox_final_snapshot.shape, domain=vox_final_snapshot.eval_domain
        )

        vox_snapshots = _snapshot_map(vox_episode)
        tvars_snapshots = _snapshot_map(tvars_episode)
        if set(vox_snapshots) != set(tvars_snapshots):
            raise ValueError("checkpoint sets differ for %s" % uid)

        ordered_event_ids = [
            str(snapshot["coverage_event_id"])
            for snapshot in vox_episode.get("snapshots", [])
        ]
        for event_id in ordered_event_ids:
            vox_record = vox_snapshots[event_id]
            tvars_record = tvars_snapshots[event_id]
            if int(vox_record["step"]) != int(tvars_record["step"]):
                raise ValueError("checkpoint steps differ for %s %s" % (uid, event_id))
            vox_snapshot = load_snapshot_arrays(Path(vox_record["snapshot_path"]))
            tvars_snapshot = load_snapshot_arrays(Path(tvars_record["snapshot_path"]))
            if vox_snapshot.shape != gt.shape or tvars_snapshot.shape != gt.shape:
                raise ValueError(
                    "checkpoint shape differs from approved GT: %s %s"
                    % (uid, event_id)
                )
            vox_explored = np.asarray(
                vox_snapshot.coverage_explored_domain
                if vox_snapshot.coverage_explored_domain is not None
                else vox_snapshot.eval_domain,
                dtype=bool,
            )
            tvars_explored = np.asarray(
                tvars_snapshot.coverage_explored_domain
                if tvars_snapshot.coverage_explored_domain is not None
                else tvars_snapshot.eval_domain,
                dtype=bool,
            )
            if not np.array_equal(vox_explored, tvars_explored):
                raise ValueError(
                    "VoxRoom and TVARS explored domains differ: %s %s"
                    % (uid, event_id)
                )
            step_gt = backproject_final_gt_to_snapshot(
                gt,
                vox_snapshot,
                episode_uid=uid,
                source_final_step=int(vox_episode["last_snapshot_step"]),
            ).label_map

            for method, episode, snapshot, record in (
                ("voxroom", vox_episode, vox_snapshot, vox_record),
                ("tvars_original", tvars_episode, tvars_snapshot, tvars_record),
            ):
                prepared = prepare_metric_label_maps(
                    step_gt,
                    snapshot.final_room_label_map,
                    min_room_area_cells=min_area_cells,
                )
                metric = compute_snapshot_metrics(prepared.gt, prepared.pred)
                precision = float(metric["precision"])
                recall = float(metric["recall"])
                f1 = (
                    2.0 * precision * recall / (precision + recall)
                    if precision + recall > 0.0
                    else 0.0
                )
                vis_path = (
                    out_dir
                    / "match_visualizations"
                    / method
                    / uid
                    / (event_id + ".pred_gt_match.png")
                )
                save_match_visualization(
                    vis_path,
                    pred=prepared.pred,
                    gt=prepared.gt,
                    iou_matrix=np.asarray(metric["iou_matrix"]),
                    metric={
                        **metric,
                        "episode_uid": uid,
                        "scene_id": episode.get("scene_id"),
                        "step": int(record["step"]),
                    },
                )
                row = {
                    "dataset": _dataset_name(episode),
                    "scene_id": str(episode.get("scene_id")),
                    "episode_uid": uid,
                    "method": method,
                    "coverage_event_id": event_id,
                    "coverage_event_kind": record.get("coverage_event_kind"),
                    "coverage_ratio": record.get("coverage_ratio"),
                    "coverage_threshold": record.get("coverage_threshold"),
                    "is_last": bool(record.get("is_last")),
                    "step": int(record["step"]),
                    "precision": precision,
                    "recall": recall,
                    "f1": float(f1),
                    "miou_room": float(metric["miou_room"]),
                    "precision_percent": 100.0 * precision,
                    "recall_percent": 100.0 * recall,
                    "f1_percent": 100.0 * float(f1),
                    "miou_room_percent": 100.0 * float(metric["miou_room"]),
                    "n_gt": int(metric["n_gt"]),
                    "n_pred": int(metric["n_pred"]),
                    "metric_domain_pixels": int(
                        np.count_nonzero(prepared.metric_domain)
                    ),
                    "gt_filtered_small": int(
                        prepared.stats["gt_label_masks_filtered_small"]
                    ),
                    "pred_filtered_small": int(
                        prepared.stats["pred_label_masks_filtered_small"]
                    ),
                    "snapshot_path": str(snapshot.path),
                    "match_visualization": str(vis_path),
                }
                rows.append(row)
                by_checkpoint_method[(uid, event_id, method)] = row

    paired_rows: list[dict[str, Any]] = []
    for uid in approved:
        for snapshot in voxroom_by_uid[uid].get("snapshots", []):
            event_id = str(snapshot["coverage_event_id"])
            vox = by_checkpoint_method[(uid, event_id, "voxroom")]
            tvars = by_checkpoint_method[(uid, event_id, "tvars_original")]
            paired_rows.append(
                {
                    "dataset": vox["dataset"],
                    "scene_id": vox["scene_id"],
                    "episode_uid": uid,
                    "coverage_event_id": event_id,
                    "coverage_ratio": vox["coverage_ratio"],
                    "coverage_threshold": vox["coverage_threshold"],
                    "step": vox["step"],
                    "voxroom_precision_percent": vox["precision_percent"],
                    "voxroom_recall_percent": vox["recall_percent"],
                    "voxroom_f1_percent": vox["f1_percent"],
                    "voxroom_miou_room_percent": vox["miou_room_percent"],
                    "tvars_precision_percent": tvars["precision_percent"],
                    "tvars_recall_percent": tvars["recall_percent"],
                    "tvars_f1_percent": tvars["f1_percent"],
                    "tvars_miou_room_percent": tvars["miou_room_percent"],
                    "delta_precision_points": vox["precision_percent"]
                    - tvars["precision_percent"],
                    "delta_recall_points": vox["recall_percent"]
                    - tvars["recall_percent"],
                    "delta_f1_points": vox["f1_percent"]
                    - tvars["f1_percent"],
                    "delta_miou_room_points": vox["miou_room_percent"]
                    - tvars["miou_room_percent"],
                }
            )
    paired_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["scene_id"],
            _checkpoint_sort_key(row["coverage_event_id"]),
        )
    )

    method_summary = {method: _aggregate(rows, method) for method in METHODS}
    dataset_summary: dict[str, dict[str, Any]] = {}
    dataset_checkpoint_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        dataset_summary[dataset] = {
            method: _aggregate(dataset_rows, method) for method in METHODS
        }
        dataset_checkpoint_summary[dataset] = {}
        for event_id in sorted(
            {str(row["coverage_event_id"]) for row in dataset_rows},
            key=_checkpoint_sort_key,
        ):
            selected = [
                row
                for row in dataset_rows
                if str(row["coverage_event_id"]) == event_id
            ]
            dataset_checkpoint_summary[dataset][event_id] = {
                method: _aggregate(selected, method) for method in METHODS
            }
    checkpoint_summary: dict[str, dict[str, Any]] = {}
    for event_id in sorted(
        {str(row["coverage_event_id"]) for row in rows},
        key=_checkpoint_sort_key,
    ):
        checkpoint_rows = [
            row for row in rows if str(row["coverage_event_id"]) == event_id
        ]
        checkpoint_summary[event_id] = {
            method: _aggregate(checkpoint_rows, method) for method in METHODS
        }
    summary = {
        "schema_version": "voxroom_region_overlap_pr_f1_room_miou_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_protocol": "paper_region_overlap_precision_recall_all_recorded_checkpoints_paired_gt",
        "paper_usage": "metric_definition_only_not_paper_experiment_protocol",
        "paper": str(paper_path),
        "paper_sha256": _sha256(paper_path),
        "approved_scene_count": len(approved),
        "approved_process_count": len(approved),
        "checkpoint_evaluation_count_per_method": len(paired_rows),
        "algorithm_evaluation_count": len(rows),
        "approved_episode_uids": approved,
        "dataset_scene_counts": {
            dataset: sum(
                1
                for uid in approved
                if _dataset_name(voxroom_by_uid[uid]) == dataset
            )
            for dataset in sorted(
                {_dataset_name(voxroom_by_uid[uid]) for uid in approved}
            )
        },
        "evaluation_snapshot_policy": "all_recorded_segmentation_checkpoints",
        "comparison_domain": "same_approved_final_gt_clipped_to_shared_checkpoint_explored_domain_for_both_methods",
        "segmentation_source": "voxel_vertical_free_xy",
        "min_room_area_m2": float(args.min_room_area_m2),
        "cell_size_m": float(args.cell_size_m),
        "min_room_area_cells": int(min_area_cells),
        "precision_definition": "mean_pred_room(max_gt(intersection/pred_room_area))",
        "recall_definition": "mean_gt_room(max_pred(intersection/gt_room_area))",
        "f1_definition": "per_checkpoint_harmonic_mean_of_precision_and_recall_then_macro_average",
        "miou_room_definition": "hungarian_one_to_one_matched_room_iou_sum_divided_by_gt_room_count_unmatched_gt_zero",
        "methods": method_summary,
        "by_dataset": dataset_summary,
        "by_dataset_and_checkpoint": dataset_checkpoint_summary,
        "by_checkpoint": checkpoint_summary,
        "paired_delta": _paired_delta_summary(paired_rows, method_summary),
        "inputs": {
            "voxroom_index": str(voxroom_index_path),
            "voxroom_index_sha256": _sha256(voxroom_index_path),
            "tvars_index": str(tvars_index_path),
            "tvars_index_sha256": _sha256(tvars_index_path),
            "annotation_dir": str(annotation_dir),
            "gt_dir": str(gt_dir),
            "annotation_sha256_by_episode": annotation_hashes,
            "gt_sha256_by_episode": gt_hashes,
        },
    }

    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["scene_id"],
            _checkpoint_sort_key(row["coverage_event_id"]),
            row["method"],
        )
    )
    write_json_atomic(out_dir / "summary.json", summary)
    write_json_atomic(out_dir / "per_checkpoint_metrics.json", rows)
    write_json_atomic(out_dir / "paired_comparison.json", paired_rows)
    _write_csv(out_dir / "per_checkpoint_metrics.csv", rows)
    _write_report(out_dir / "report.md", summary=summary, paired_rows=paired_rows)
    for dataset in sorted(dataset_summary):
        dataset_dir = out_dir / dataset
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        dataset_paired = [row for row in paired_rows if row["dataset"] == dataset]
        dataset_methods = dataset_summary[dataset]
        dataset_report_summary = {
            "schema_version": summary["schema_version"],
            "created_at": summary["created_at"],
            "dataset": dataset,
            "metric_protocol": summary["metric_protocol"],
            "paper_usage": summary["paper_usage"],
            "paper": summary["paper"],
            "approved_process_count": dataset_methods["voxroom"][
                "process_count"
            ],
            "checkpoint_evaluation_count_per_method": dataset_methods["voxroom"][
                "evaluation_count"
            ],
            "algorithm_evaluation_count": len(dataset_rows),
            "evaluation_snapshot_policy": summary["evaluation_snapshot_policy"],
            "comparison_domain": summary["comparison_domain"],
            "min_room_area_m2": summary["min_room_area_m2"],
            "cell_size_m": summary["cell_size_m"],
            "min_room_area_cells": summary["min_room_area_cells"],
            "precision_definition": summary["precision_definition"],
            "recall_definition": summary["recall_definition"],
            "f1_definition": summary["f1_definition"],
            "miou_room_definition": summary["miou_room_definition"],
            "methods": dataset_methods,
            "by_checkpoint": dataset_checkpoint_summary[dataset],
            "paired_delta": _paired_delta_summary(
                dataset_paired, dataset_methods
            ),
        }
        write_json_atomic(dataset_dir / "summary.json", dataset_report_summary)
        write_json_atomic(
            dataset_dir / "per_checkpoint_metrics.json", dataset_rows
        )
        write_json_atomic(
            dataset_dir / "paired_comparison.json", dataset_paired
        )
        _write_csv(dataset_dir / "per_checkpoint_metrics.csv", dataset_rows)
        _write_report(
            dataset_dir / "report.md",
            summary=dataset_report_summary,
            paired_rows=dataset_paired,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
