#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from voxroom_online.isaac_runtime.comparison.metadata_gate import (
    assert_main_experiment_metadata,
)
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


BASELINE_METHODS = (
    "dude_incremental",
    "gomez_incremental",
    "dude_offline",
    "rose2",
    "morphological",
    "distance_transform",
    "voronoi",
)


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


def _snapshot_map(episode: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {
        str(snapshot["coverage_event_id"]): dict(snapshot)
        for snapshot in episode.get("snapshots", [])
    }
    if len(rows) != len(episode.get("snapshots", [])):
        raise ValueError("episode contains duplicate coverage_event_id values")
    return rows


def _checkpoint_sort_key(event_id: str) -> tuple[int, str]:
    text = str(event_id)
    if text.startswith("milestone_"):
        try:
            return int(text.removeprefix("milestone_")), text
        except ValueError:
            pass
    return (10_000 if text == "final" else 9_000), text


def _load_replay_rows(
    replay_root: Path,
    method: str,
    *,
    expected_segmentation_input_mode: str | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    path = replay_root / "manifests" / (method + ".json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("method")) != method:
        raise ValueError(f"replay manifest method mismatch: {path}")
    if expected_segmentation_input_mode is not None and str(
        payload.get("segmentation_input_mode")
    ) != str(expected_segmentation_input_mode):
        raise ValueError(
            "replay manifest segmentation input mismatch: %s expected=%s actual=%s"
            % (
                path,
                expected_segmentation_input_mode,
                payload.get("segmentation_input_mode"),
            )
        )
    rows = {
        (str(row["episode_uid"]), str(row["coverage_event_id"])): dict(row)
        for row in payload.get("rows", [])
    }
    if len(rows) != len(payload.get("rows", [])):
        raise ValueError(f"duplicate replay rows: {path}")
    return rows


def _load_prediction(
    path: Path,
    method: str,
    *,
    expected_segmentation_input_mode: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        labels = np.asarray(data["final_room_label_map"], dtype=np.int32)
        metadata = json.loads(str(data["baseline_metadata_json"]))
    assert_main_experiment_metadata(metadata, method)
    if expected_segmentation_input_mode is not None and str(
        metadata.get("segmentation_input_mode")
    ) != str(expected_segmentation_input_mode):
        raise ValueError(
            "prediction segmentation input mismatch: %s expected=%s actual=%s"
            % (
                path,
                expected_segmentation_input_mode,
                metadata.get("segmentation_input_mode"),
            )
        )
    return labels, dict(metadata)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0


def _aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    result: dict[str, Any] = {
        "evaluation_count": len(selected),
        "process_count": len({str(row["episode_uid"]) for row in selected}),
    }
    for key, output in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("miou_room", "mIoU_room"),
    ):
        values = [float(row[key]) for row in selected]
        result[output] = _mean(values)
        result[output + "_percent"] = 100.0 * _mean(values)
        result["scene_std_" + output] = _std(values)
        result["scene_std_" + output + "_percent"] = 100.0 * _std(values)
    return result


def _summarize(
    rows: list[dict[str, Any]],
    *,
    methods: list[str],
    scope: str,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    method_summary = {method: _aggregate(rows, method) for method in methods}
    counts = {int(value["evaluation_count"]) for value in method_summary.values()}
    if len(counts) != 1:
        raise ValueError(f"unpaired method evaluation counts in {scope}: {method_summary}")
    by_checkpoint: dict[str, dict[str, Any]] = {}
    for event_id in sorted(
        {str(row["coverage_event_id"]) for row in rows},
        key=_checkpoint_sort_key,
    ):
        selected = [row for row in rows if row["coverage_event_id"] == event_id]
        by_checkpoint[event_id] = {
            method: _aggregate(selected, method) for method in methods
        }
    return {
        **dict(common),
        "scope": scope,
        "approved_scene_count": len({str(row["episode_uid"]) for row in rows}),
        "checkpoint_evaluation_count_per_method": next(iter(counts), 0),
        "algorithm_evaluation_count": len(rows),
        "methods": method_summary,
        "by_checkpoint": by_checkpoint,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_csv_rows(
    summary: Mapping[str, Any],
    *,
    scope: str,
    methods: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, Mapping[str, Any]]] = [
        ("all_checkpoints", summary["methods"]),
    ]
    groups.extend(
        (str(checkpoint), values)
        for checkpoint, values in summary.get("by_checkpoint", {}).items()
    )
    for checkpoint, values in groups:
        for method in methods:
            metric = values[method]
            rows.append(
                {
                    "scope": scope,
                    "checkpoint": checkpoint,
                    "method": method,
                    "process_count": int(metric["process_count"]),
                    "evaluation_count": int(metric["evaluation_count"]),
                    "precision_percent": float(metric["Precision_percent"]),
                    "recall_percent": float(metric["Recall_percent"]),
                    "f1_percent": float(metric["F1_percent"]),
                    "miou_room_percent": float(metric["mIoU_room_percent"]),
                }
            )
    return rows


def _write_report(
    path: Path,
    *,
    all_summary: Mapping[str, Any],
    excluded_summary: Mapping[str, Any] | None,
    methods: list[str],
    dataset_name: str,
) -> None:
    display_name = {
        "grscene": "GRScene",
        "interioragent": "InteriorAgent",
    }.get(str(dataset_name), str(dataset_name).replace("_", " ").title())
    input_mode = str(
        all_summary.get("paper_baseline_segmentation_input_mode")
        or "legacy_unspecified"
    )
    input_description = {
        "raw_vertical_free": "raw online `voxel_vertical_free_xy` (Vertical-Free)",
        "raw_nav_free_no_clearance": (
            "raw no-clearance `voxel_nav_free_xy` (Nav-Free)"
        ),
    }.get(input_mode, input_mode)
    lines = [
        "# %s paper segmentation baselines" % display_name,
        "",
        (
            "Every replayed paper baseline uses the same saved checkpoint and %s. "
            "Fixed coverage/reference masks are excluded from algorithm input. "
            "All metrics use the same backprojected approved GT, 0.5 m² small-room "
            "filter, and P/R/F1/room-mIoU implementation."
        )
        % input_description,
        "",
        (
            "The `voxroom` and `tvars_original` rows are their saved original "
            "predictions and remain comparison anchors; the input variant named "
            "above applies to the seven replayed paper baselines."
        ),
        "",
        "Incremental Gomez is a checkpoint replay reproduction: the original source was unavailable to the TVARS authors. To avoid recasting a virtual laser from each checkpoint, it imports the actual unfiltered range-jump endpoint map (tvars_original_hough_door_seed_map) from the paired saved TVARS run, then applies Gomez door-size rules and persistent door lines. Its door-frame confirmation runs a vertical-line Hough transform on a vertical cross-section sampled from the saved 3-D voxel map, replacing the unavailable per-frame RGB Hough input. Saved checkpoints still cannot reproduce the paper's explicit 0.9 m approach-and-redetect behavior at every candidate.",
        "",
        "Incremental DUDE, offline DUDE, ROSE2, Morphological, Distance, and Voronoi use the original repositories recorded in replay_summary.json. Offline DUDE starts a fresh upstream node for every checkpoint; incremental state is reset only between scenes.",
        "",
    ]
    for title, summary in (
        ("All approved %s scenes" % display_name, all_summary),
        ("Excluding VoxRoom training/validation scenes", excluded_summary),
    ):
        if summary is None:
            continue
        lines.extend(
            [
                "## " + title,
                "",
                "| Method | Scenes | Evaluations | P (%) | R (%) | F1 (%) | room mIoU (%) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            row = summary["methods"][method]
            lines.append(
                "| {m} | {s} | {n} | {p:.3f} | {r:.3f} | {f:.3f} | {i:.3f} |".format(
                    m=method,
                    s=int(row["process_count"]),
                    n=int(row["evaluation_count"]),
                    p=float(row["Precision_percent"]),
                    r=float(row["Recall_percent"]),
                    f=float(row["F1_percent"]),
                    i=float(row["mIoU_room_percent"]),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    dataset_name = str(args.dataset_name).strip().lower()
    if not dataset_name:
        raise ValueError("dataset name must not be empty")
    voxroom_index_path = Path(args.voxroom_index).resolve()
    tvars_index_path = Path(args.tvars_index).resolve()
    replay_root = Path(args.replay_root).resolve()
    replay_summary_path = replay_root / "replay_summary.json"
    annotation_dir = Path(args.annotation_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    paper_path = Path(args.paper).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    voxroom = _episode_map(load_index(voxroom_index_path))
    tvars = _episode_map(load_index(tvars_index_path))
    if set(voxroom) != set(tvars):
        raise ValueError("VoxRoom and TVARS indexes contain different episodes")
    replay_summary = json.loads(replay_summary_path.read_text(encoding="utf-8"))
    segmentation_input_mode = replay_summary.get("segmentation_input_mode")
    if segmentation_input_mode is None:
        raise ValueError(
            "replay summary lacks explicit segmentation_input_mode: %s"
            % replay_summary_path
        )
    segmentation_input_mode = str(segmentation_input_mode)
    selected_baselines = tuple(str(method) for method in args.methods)
    replay = {
        method: _load_replay_rows(
            replay_root,
            method,
            expected_segmentation_input_mode=segmentation_input_mode,
        )
        for method in selected_baselines
    }
    methods = ["voxroom", "tvars_original", *selected_baselines]
    min_area_cells = min_area_cells_from_m2(
        float(args.min_room_area_m2), float(args.cell_size_m)
    )

    rows: list[dict[str, Any]] = []
    for uid in sorted(voxroom):
        annotation_path = annotation_dir / uid / "last_step.annotation.json"
        if not annotation_path.is_file():
            continue
        annotation = load_annotation(annotation_path)
        if annotation.review.status != "approved":
            continue
        gt_path = gt_dir / uid / "last_step.gt_labels.npy"
        gt_meta_path = gt_dir / uid / "last_step.gt_metadata.json"
        if not gt_path.is_file() or not gt_meta_path.is_file():
            raise FileNotFoundError(f"approved GT is incomplete: {uid}")
        gt_meta = json.loads(gt_meta_path.read_text(encoding="utf-8"))
        if str(gt_meta.get("annotation_review_status")) != "approved":
            raise ValueError(f"GT is not approved: {uid}")

        vox_episode = voxroom[uid]
        tvars_episode = tvars[uid]
        vox_final = load_snapshot_arrays(Path(vox_episode["last_snapshot_path"]))
        validate_annotation(annotation, vox_final)
        gt = np.asarray(np.load(gt_path), dtype=np.int32)
        validate_gt_label_map(gt, vox_final.shape, domain=vox_final.eval_domain)
        tvars_events = _snapshot_map(tvars_episode)

        for source_record in vox_episode.get("snapshots", []):
            event_id = str(source_record["coverage_event_id"])
            source_path = Path(str(source_record["snapshot_path"]))
            source = load_snapshot_arrays(source_path)
            step_gt = backproject_final_gt_to_snapshot(
                gt,
                source,
                episode_uid=uid,
                source_final_step=int(vox_episode["last_snapshot_step"]),
            ).label_map
            predictions: list[tuple[str, np.ndarray, str, dict[str, Any]]] = [
                ("voxroom", source.final_room_label_map, str(source_path), {}),
            ]
            tvars_record = tvars_events[event_id]
            tvars_snapshot = load_snapshot_arrays(Path(tvars_record["snapshot_path"]))
            if int(tvars_record["step"]) != int(source_record["step"]):
                raise ValueError(f"TVARS step mismatch: {uid} {event_id}")
            predictions.append(
                (
                    "tvars_original",
                    tvars_snapshot.final_room_label_map,
                    str(tvars_snapshot.path),
                    {},
                )
            )
            for method in selected_baselines:
                replay_row = replay[method].get((uid, event_id))
                if replay_row is None:
                    raise KeyError(f"missing {method} replay: {uid} {event_id}")
                pred_path = Path(str(replay_row["prediction_npz"]))
                labels, metadata = _load_prediction(
                    pred_path,
                    method,
                    expected_segmentation_input_mode=segmentation_input_mode,
                )
                predictions.append((method, labels, str(pred_path), metadata))

            for method, pred, prediction_path, metadata in predictions:
                if np.asarray(pred).shape != source.shape:
                    raise ValueError(
                        f"prediction shape mismatch: {method} {uid} {event_id}"
                    )
                prepared = prepare_metric_label_maps(
                    step_gt,
                    pred,
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
                rows.append(
                    {
                        "dataset": dataset_name,
                        "scene_id": str(vox_episode.get("scene_id")),
                        "episode_uid": uid,
                        "method": method,
                        "coverage_event_id": event_id,
                        "coverage_event_kind": source_record.get("coverage_event_kind"),
                        "coverage_ratio": source_record.get("coverage_ratio"),
                        "coverage_threshold": source_record.get("coverage_threshold"),
                        "is_last": bool(source_record.get("is_last")),
                        "step": int(source_record["step"]),
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
                        "metric_domain_pixels": int(
                            np.count_nonzero(prepared.metric_domain)
                        ),
                        "gt_filtered_small": int(
                            prepared.stats["gt_label_masks_filtered_small"]
                        ),
                        "pred_filtered_small": int(
                            prepared.stats["pred_label_masks_filtered_small"]
                        ),
                        "source_snapshot_path": str(source_path),
                        "prediction_path": prediction_path,
                        "runner_type": metadata.get("runner_type"),
                        "implementation_commit": metadata.get(
                            "original_repo_commit"
                        ),
                    }
                )

    rows.sort(
        key=lambda row: (
            row["scene_id"],
            _checkpoint_sort_key(str(row["coverage_event_id"])),
            methods.index(str(row["method"])),
        )
    )
    common = {
        "schema_version": "voxroom_all_paper_segmentation_baselines_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "metric_protocol": "paper_region_overlap_precision_recall_all_recorded_checkpoints_paired_gt",
        "evaluation_snapshot_policy": "all_recorded_segmentation_checkpoints",
        "segmentation_source": {
            "raw_vertical_free": "voxel_vertical_free_xy",
            "raw_nav_free_no_clearance": "voxel_nav_free_xy",
        }.get(segmentation_input_mode, segmentation_input_mode),
        "paper_baseline_segmentation_input_mode": segmentation_input_mode,
        "coverage_reference_used_as_segmentation_input": False,
        "min_room_area_m2": float(args.min_room_area_m2),
        "cell_size_m": float(args.cell_size_m),
        "min_room_area_cells": int(min_area_cells),
        "precision_definition": "mean_pred_room(max_gt(intersection/pred_room_area))",
        "recall_definition": "mean_gt_room(max_pred(intersection/gt_room_area))",
        "f1_definition": "per_checkpoint harmonic mean of P and R",
        "miou_room_definition": "Hungarian one-to-one matched room IoU divided by GT room count",
        "paper": str(paper_path),
        "paper_sha256": _sha256(paper_path),
        "methods_order": methods,
        "inputs": {
            "voxroom_index": str(voxroom_index_path),
            "voxroom_index_sha256": _sha256(voxroom_index_path),
            "tvars_index": str(tvars_index_path),
            "tvars_index_sha256": _sha256(tvars_index_path),
            "replay_root": str(replay_root),
            "replay_summary": str(replay_summary_path),
            "replay_summary_sha256": _sha256(replay_summary_path),
            "annotation_dir": str(annotation_dir),
            "gt_dir": str(gt_dir),
        },
        "implementation_provenance": {
            "voxroom_repo_commit": replay_summary.get("voxroom_repo_commit"),
            "voxroom_python_source_tree_sha256": replay_summary.get(
                "voxroom_python_source_tree_sha256"
            ),
            "external_repositories": replay_summary.get("external_repositories"),
        },
    }
    all_summary = _summarize(
        rows, methods=methods, scope="all_approved", common=common
    )
    excluded_summary: dict[str, Any] | None = None
    excluded_rows: list[dict[str, Any]] = []
    if args.training_exclusion_manifest:
        exclusion_path = Path(args.training_exclusion_manifest).resolve()
        exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
        excluded_scene_ids = {
            str(value) for value in exclusion.get("excluded_scene_ids", [])
        }
        excluded_rows = [
            row for row in rows if str(row["scene_id"]) not in excluded_scene_ids
        ]
        excluded_summary = _summarize(
            excluded_rows,
            methods=methods,
            scope="excluding_voxroom_training_validation_scenes",
            common={
                **common,
                "training_exclusion_manifest": str(exclusion_path),
                "training_exclusion_manifest_sha256": _sha256(exclusion_path),
                "excluded_scene_ids": sorted(excluded_scene_ids),
            },
        )

    _write_csv(out_dir / "per_checkpoint_metrics.csv", rows)
    write_json_atomic(out_dir / "summary.json", all_summary)
    if excluded_summary is not None:
        _write_csv(
            out_dir / "per_checkpoint_metrics_excluding_training.csv",
            excluded_rows,
        )
        write_json_atomic(
            out_dir / "summary_excluding_training.json", excluded_summary
        )
    aggregate_rows = _aggregate_csv_rows(
        all_summary,
        scope="all",
        methods=methods,
    )
    if excluded_summary is not None:
        aggregate_rows.extend(
            _aggregate_csv_rows(
                excluded_summary,
                scope="excluding_training",
                methods=methods,
            )
        )
    _write_csv(out_dir / "aggregate.csv", aggregate_rows)
    _write_report(
        out_dir / "report.md",
        all_summary=all_summary,
        excluded_summary=excluded_summary,
        methods=methods,
        dataset_name=dataset_name,
    )
    return all_summary, excluded_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate VoxRoom, TVARS, and all segmentation baselines listed in the TVARS paper."
    )
    parser.add_argument("--dataset-name", default="grscene")
    parser.add_argument("--voxroom-index", required=True)
    parser.add_argument("--tvars-index", required=True)
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--training-exclusion-manifest")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=BASELINE_METHODS,
        default=list(BASELINE_METHODS),
        help="Replayed baselines to evaluate in addition to VoxRoom and TVARS.",
    )
    parser.add_argument("--min-room-area-m2", type=float, default=0.5)
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    args = parser.parse_args(argv)
    all_summary, excluded_summary = run(args)
    print(
        json.dumps(
            {"all": all_summary, "excluding_training": excluded_summary},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
