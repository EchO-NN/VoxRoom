#!/usr/bin/env python3
"""Run CMU SysNav's official room segmenter on a saved checkpoint index."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


METHOD = "cmu_sysnav_official_room_segmentation_full_snapshot_input"
SYSNAV_COMMIT = "0fa15cc8bc18be7409272fb65fa514a9a5bca6b0"
EVENT_ORDER = {
    "milestone_020": 20,
    "milestone_040": 40,
    "milestone_060": 60,
    "milestone_070": 70,
    "milestone_080": 80,
    "milestone_090": 90,
    "final": 1000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, action="append", required=True)
    parser.add_argument("--gt-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sysnav-root", type=Path, default=Path("/home/joey/SysNav"))
    parser.add_argument(
        "--checkpoint-runner",
        type=Path,
        default=Path(__file__).with_name("run_sysnav_roomseg_checkpoint.py"),
    )
    parser.add_argument("--training-exclusion-manifest", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-room-area-m2", type=float, default=0.5)
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    parser.add_argument("--only-approved", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def dataset_from_path(path: str) -> str:
    parts = {part.lower() for part in Path(path).parts}
    for dataset in ("interioragent", "grscene", "habitat"):
        if dataset in parts:
            return dataset
    return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_path(root: Path, episode_uid: str, event_id: str) -> Path:
    return root / "predictions" / episode_uid / f"{event_id}.npz"


def prediction_valid(path: Path, source: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as arrays:
            labels = np.asarray(arrays["sysnav_room_label_map"])
            metadata = json.loads(str(np.asarray(arrays["metadata_json"]).reshape(())))
        return (
            labels.ndim == 2
            and metadata.get("method") == METHOD
            and metadata.get("snapshot") == str(source.resolve())
            and metadata.get("sysnav_commit") == SYSNAV_COMMIT
        )
    except Exception:
        return False


def run_checkpoint(task: dict) -> dict:
    source = Path(task["source"])
    output = Path(task["output"])
    if task["resume"] and prediction_valid(output, source):
        with np.load(output, allow_pickle=False) as arrays:
            metadata = json.loads(str(np.asarray(arrays["metadata_json"]).reshape(())))
        return {**task, "status": "resumed", "room_count": metadata["projected_room_count"]}

    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    old_ld = env.get("LD_LIBRARY_PATH", "")
    build_lib = str(Path(task["sysnav_root"]) / "build/tare_planner")
    env["LD_LIBRARY_PATH"] = build_lib + (":" + old_ld if old_ld else "")
    command = [
        "/usr/bin/python3",
        str(task["checkpoint_runner"]),
        str(source),
        str(output),
        "--sysnav-root",
        str(task["sysnav_root"]),
        "--timeout",
        "180",
    ]
    started = time.perf_counter()
    attempts: list[str] = []
    result = None
    # Fast consecutive launches can occasionally leave a DDS discovery entry
    # behind on one ROS domain.  Retry the same immutable snapshot on another
    # domain; this changes transport discovery only, never algorithm inputs.
    for retry_index in range(3):
        domain_id = 70 + ((int(task["domain_id"]) - 70 + 47 * retry_index) % 150)
        env["ROS_DOMAIN_ID"] = str(domain_id)
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=210)
        attempts.append(
            f"attempt={retry_index + 1} ROS_DOMAIN_ID={domain_id} returncode={result.returncode}\n"
            + result.stdout
            + result.stderr
        )
        if result.returncode == 0:
            break
    output.with_suffix(".driver.log").write_text(
        "\n\n".join(attempts), encoding="utf-8"
    )
    assert result is not None
    if result.returncode != 0:
        raise RuntimeError(
            f"SysNav checkpoint failed after 3 DDS-domain attempts: {source}\n"
            + (result.stderr or result.stdout)[-2000:]
        )
    with np.load(output, allow_pickle=False) as arrays:
        metadata = json.loads(str(np.asarray(arrays["metadata_json"]).reshape(())))
    return {
        **task,
        "status": "written",
        "room_count": metadata["projected_room_count"],
        "runtime_seconds": time.perf_counter() - started,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_exclusions(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(str(value) for value in payload.get("excluded_scene_ids", []))
    return excluded


def approved_gt_records(
    annotation_dirs: list[Path], gt_dirs: list[Path]
) -> tuple[dict[str, dict], dict[str, dict]]:
    if len(annotation_dirs) != len(gt_dirs):
        raise ValueError("--annotation-dir and --gt-dir must be supplied in pairs")
    by_uid: dict[str, dict] = {}
    by_scene: dict[str, dict] = {}
    for annotation_dir, gt_dir in zip(annotation_dirs, gt_dirs, strict=True):
        for annotation_path in annotation_dir.glob("*/last_step.annotation.json"):
            try:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            uid = str(annotation.get("episode_uid") or annotation_path.parent.name)
            scene_id = str(annotation.get("scene_id") or annotation.get("run_name") or uid)
            gt_path = gt_dir / annotation_path.parent.name / "last_step.gt_labels.npy"
            gt_meta_path = gt_dir / annotation_path.parent.name / "last_step.gt_metadata.json"
            if annotation.get("review", {}).get("status") != "approved":
                continue
            if not gt_path.is_file() or not gt_meta_path.is_file():
                continue
            gt_meta = json.loads(gt_meta_path.read_text(encoding="utf-8"))
            if gt_meta.get("annotation_review_status") != "approved":
                continue
            record = {
                "episode_uid": uid,
                "scene_id": scene_id,
                "gt_path": gt_path,
                "annotation_path": annotation_path,
            }
            by_uid[uid] = record
            by_scene.setdefault(scene_id, record)
    return by_uid, by_scene


def aggregate(rows: list[dict], scope: str) -> dict:
    result = {
        "scope": scope,
        "scene_count": len({row["episode_uid"] for row in rows}),
        "evaluation_count": len(rows),
        "dataset_scene_counts": {
            dataset: len({row["episode_uid"] for row in rows if row["dataset"] == dataset})
            for dataset in sorted({row["dataset"] for row in rows})
        },
    }
    for key in ("precision", "recall", "f1", "miou_room"):
        value = float(np.mean([row[key] for row in rows])) if rows else float("nan")
        result[key] = value
        result[f"{key}_percent"] = 100.0 * value
    result["mean_predicted_room_count"] = (
        float(np.mean([row["n_pred"] for row in rows])) if rows else float("nan")
    )
    result["mean_gt_room_count"] = (
        float(np.mean([row["n_gt"] for row in rows])) if rows else float("nan")
    )
    return result


def evaluate(
    episodes: list[dict],
    output_root: Path,
    annotation_dirs: list[Path],
    gt_dirs: list[Path],
    excluded: set[str],
    min_room_area_m2: float,
    cell_size_m: float,
) -> tuple[list[dict], dict]:
    approved_by_uid, approved_by_scene = approved_gt_records(annotation_dirs, gt_dirs)
    min_area_cells = min_area_cells_from_m2(min_room_area_m2, cell_size_m)
    rows: list[dict] = []
    for episode in episodes:
        uid = str(episode["episode_uid"])
        scene_id = str(episode.get("scene_id") or uid)
        gt_record = approved_by_uid.get(uid)
        gt_match = "episode_uid"
        if gt_record is None:
            gt_record = approved_by_scene.get(scene_id)
            gt_match = "scene_id_same_world_grid"
        if gt_record is None:
            continue
        final_gt = np.asarray(np.load(gt_record["gt_path"]), dtype=np.int32)
        dataset = dataset_from_path(str(episode["last_snapshot_path"]))
        for snapshot in episode.get("snapshots", []):
            event_id = str(snapshot["coverage_event_id"])
            prediction = prediction_path(output_root, uid, event_id)
            if not prediction.is_file():
                continue
            source = load_snapshot_arrays(Path(snapshot["snapshot_path"]))
            step_gt = backproject_final_gt_to_snapshot(
                final_gt,
                source,
                episode_uid=uid,
                source_final_step=int(episode["last_snapshot_step"]),
            ).label_map
            with np.load(prediction, allow_pickle=False) as arrays:
                pred = np.asarray(arrays["sysnav_room_label_map"], dtype=np.int32)
            prepared = prepare_metric_label_maps(
                step_gt, pred, min_room_area_cells=min_area_cells
            )
            metric = compute_snapshot_metrics(prepared.gt, prepared.pred)
            precision = float(metric["precision"])
            recall = float(metric["recall"])
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "scene_id": scene_id,
                    "episode_uid": uid,
                    "gt_episode_uid": gt_record["episode_uid"],
                    "gt_match": gt_match,
                    "excluded_as_training_or_validation": scene_id in excluded,
                    "coverage_event_id": event_id,
                    "coverage_ratio": snapshot.get("coverage_ratio"),
                    "step": int(snapshot["step"]),
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
                    "source_snapshot_path": str(source.path),
                    "prediction_path": str(prediction),
                }
            )
    rows.sort(key=lambda row: (row["dataset"], row["scene_id"], EVENT_ORDER.get(row["coverage_event_id"], 999)))
    test = [row for row in rows if not row["excluded_as_training_or_validation"]]
    paper = [row for row in test if row["dataset"] in {"interioragent", "grscene"}]
    summary = {
        "approved_episode_count_available": len(approved_by_uid),
        "all_approved_all_checkpoints": aggregate(rows, "all_approved_all_checkpoints"),
        "all_approved_final": aggregate([row for row in rows if row["coverage_event_id"] == "final"], "all_approved_final"),
        "paper_datasets_excluding_training_all_checkpoints": aggregate(paper, "paper_datasets_excluding_training_all_checkpoints"),
        "paper_datasets_excluding_training_final": aggregate([row for row in paper if row["coverage_event_id"] == "final"], "paper_datasets_excluding_training_final"),
        "by_checkpoint_paper_datasets_excluding_training": {
            event: aggregate([row for row in paper if row["coverage_event_id"] == event], event)
            for event in sorted({row["coverage_event_id"] for row in paper}, key=lambda value: EVENT_ORDER.get(value, 999))
        },
    }
    return rows, summary


def write_report(path: Path, summary: dict) -> None:
    evaluation = summary["evaluation"]
    lines = [
        "# CMU SysNav official-source room-segmentation replay",
        "",
        f"Official repository commit: `{SYSNAV_COMMIT}`.",
        "",
        "Input: saved 3-D occupied voxels and raw `voxel_nav_free_xy` converted into SysNav's registered-scan, freespace, occupied-state and odometry streams. The plane-fit radius covers the complete snapshot map. No VoxRoom labels, seeds, separators or GT are supplied to SysNav.",
        "",
        "Metrics: paper region P/R, their per-checkpoint F1, and one-to-one room mIoU; rooms below 0.5 m² are removed before scoring.",
        "",
        "## Paper test scope",
        "",
        "| Scope | Scenes | Checks | P (%) | R (%) | F1 (%) | room mIoU (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("paper_datasets_excluding_training_all_checkpoints", "all checkpoints"),
        ("paper_datasets_excluding_training_final", "final only"),
    ):
        row = evaluation[key]
        lines.append(
            f"| {label} | {row['scene_count']} | {row['evaluation_count']} | "
            f"{row['precision_percent']:.3f} | {row['recall_percent']:.3f} | "
            f"{row['f1_percent']:.3f} | {row['miou_room_percent']:.3f} |"
        )
    lines += [
        "",
        "## Per checkpoint",
        "",
        "| Checkpoint | Scenes | P (%) | R (%) | F1 (%) | room mIoU (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for event, row in evaluation["by_checkpoint_paper_datasets_excluding_training"].items():
        lines.append(
            f"| {event} | {row['scene_count']} | {row['precision_percent']:.3f} | "
            f"{row['recall_percent']:.3f} | {row['f1_percent']:.3f} | {row['miou_room_percent']:.3f} |"
        )
    if summary["missing_paper_dataset_gt_locally"]:
        lines += [
            "",
            "> This report is partial because no approved GRScene ground truth was supplied.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate_metrics(path: Path, evaluation: dict) -> None:
    rows: list[dict] = []
    scopes = [
        ("all_checkpoints", evaluation["paper_datasets_excluding_training_all_checkpoints"]),
        ("final_only", evaluation["paper_datasets_excluding_training_final"]),
    ]
    scopes.extend(evaluation["by_checkpoint_paper_datasets_excluding_training"].items())
    for name, metric in scopes:
        dataset_counts = metric.get("dataset_scene_counts", {})
        rows.append(
            {
                "scope": name,
                "scene_count": metric["scene_count"],
                "evaluation_count": metric["evaluation_count"],
                "grscene_scene_count": dataset_counts.get("grscene", 0),
                "interioragent_scene_count": dataset_counts.get("interioragent", 0),
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "miou_room": metric["miou_room"],
                "precision_percent": metric["precision_percent"],
                "recall_percent": metric["recall_percent"],
                "f1_percent": metric["f1_percent"],
                "miou_room_percent": metric["miou_room_percent"],
                "mean_predicted_room_count": metric["mean_predicted_room_count"],
                "mean_gt_room_count": metric["mean_gt_room_count"],
            }
        )
    write_csv(path, rows)


def main() -> int:
    args = parse_args()
    index_path = args.index.resolve()
    annotation_dirs = [path.resolve() for path in args.annotation_dir]
    gt_dirs = [path.resolve() for path in args.gt_dir]
    if len(annotation_dirs) != len(gt_dirs):
        raise ValueError("--annotation-dir and --gt-dir must be supplied in pairs")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    episodes = sorted(payload.get("episodes", []), key=lambda item: str(item["episode_uid"]))
    approved_by_uid, approved_by_scene = approved_gt_records(annotation_dirs, gt_dirs)
    if args.only_approved:
        episodes = [
            episode
            for episode in episodes
            if str(episode["episode_uid"]) in approved_by_uid
            or str(episode.get("scene_id") or episode["episode_uid"]) in approved_by_scene
        ]

    tasks: list[dict] = []
    for task_index, episode in enumerate(episodes):
        snapshots = list(episode.get("snapshots", []))
        if args.final_only:
            snapshots = [item for item in snapshots if str(item["coverage_event_id"]) == "final"]
        for snapshot in snapshots:
            event_id = str(snapshot["coverage_event_id"])
            uid = str(episode["episode_uid"])
            tasks.append(
                {
                    "episode_uid": uid,
                    "scene_id": str(episode.get("scene_id") or uid),
                    "dataset": dataset_from_path(str(snapshot["snapshot_path"])),
                    "event_id": event_id,
                    "step": int(snapshot["step"]),
                    "source": str(Path(snapshot["snapshot_path"]).resolve()),
                    "output": str(prediction_path(output_root, uid, event_id)),
                    "checkpoint_runner": str(args.checkpoint_runner.resolve()),
                    "sysnav_root": str(args.sysnav_root.resolve()),
                    "domain_id": 70 + (len(tasks) % 120),
                    "resume": bool(args.resume),
                }
            )

    started = time.perf_counter()
    manifest: list[dict] = []
    if args.evaluate_only:
        for task in tasks:
            output = Path(task["output"])
            if not prediction_valid(output, Path(task["source"])):
                continue
            with np.load(output, allow_pickle=False) as arrays:
                metadata = json.loads(str(np.asarray(arrays["metadata_json"]).reshape(())))
            manifest.append(
                {
                    **{key: task[key] for key in ("dataset", "scene_id", "episode_uid", "event_id", "step", "source", "output")},
                    "status": "existing",
                    "room_count": int(metadata["projected_room_count"]),
                }
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_checkpoint, task): task for task in tasks}
            for count, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                manifest.append({key: row[key] for key in ("dataset", "scene_id", "episode_uid", "event_id", "step", "source", "output", "status", "room_count")})
                print(f"[{count}/{len(tasks)}] {row['dataset']}/{row['scene_id']} {row['event_id']} rooms={row['room_count']} {row['status']}", flush=True)
    manifest.sort(key=lambda row: (row["dataset"], row["scene_id"], EVENT_ORDER.get(row["event_id"], 999)))
    write_csv(output_root / "prediction_manifest.csv", manifest)

    excluded = load_exclusions([path.resolve() for path in args.training_exclusion_manifest])
    metric_rows, evaluation = evaluate(
        episodes,
        output_root,
        annotation_dirs,
        gt_dirs,
        excluded,
        args.min_room_area_m2,
        args.cell_size_m,
    )
    write_csv(output_root / "per_checkpoint_metrics.csv", metric_rows)
    write_aggregate_metrics(output_root / "aggregate_metrics.csv", evaluation)
    missing_grscene_gt = not any(row["dataset"] == "grscene" for row in metric_rows)
    summary = {
        "schema_version": "cmu_sysnav_official_full_snapshot_replay_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": METHOD,
        "sysnav_repository": "https://github.com/zwandering/SysNav",
        "sysnav_commit": SYSNAV_COMMIT,
        "source_index": str(index_path),
        "source_index_sha256": sha256(index_path),
        "prediction_scene_count": len({row["episode_uid"] for row in manifest}),
        "prediction_checkpoint_count": len(manifest),
        "elapsed_seconds": time.perf_counter() - started,
        "workers": args.workers,
        "algorithm_input_contract": {
            "registered_scan": "occupied voxel surface plus voxel_nav_free_xy floor",
            "freespace_cloud": "voxel_nav_free_xy at robot height",
            "occupied_cloud": "occupied voxels within robot_z +/- 0.5 m, intensity 0",
            "state_estimation": "saved checkpoint robot pose",
            "region_growing_scope": "complete snapshot map",
            "voxroom_room_labels_used": False,
            "door_seeds_or_separators_used": False,
            "ground_truth_used_for_inference": False,
        },
        "evaluation": evaluation,
        "missing_paper_dataset_gt_locally": missing_grscene_gt,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(output_root / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
