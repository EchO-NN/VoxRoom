from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from voxroom_online.isaac_runtime.evaluation.online_roomseg.annotation_schema import (
    approved_review,
    make_initial_annotation,
    save_annotation_atomic,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.cli import main
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import load_snapshot_arrays


def test_eval_cli_end_to_end_geometric_only(tmp_path: Path) -> None:
    result_root = _make_result_root(tmp_path)
    eval_dir = tmp_path / "eval"
    index_path = eval_dir / "index.json"
    annotation_dir = eval_dir / "annotations"
    gt_dir = eval_dir / "gt_final"
    step_gt_dir = eval_dir / "gt_by_step"
    metrics_dir = eval_dir / "metrics"

    assert main(["index", "--result-root", str(result_root), "--out", str(index_path), "--snapshot-policy", "all"]) == 0
    index = json.load(index_path.open("r", encoding="utf-8"))
    for episode in index["episodes"]:
        snapshot = load_snapshot_arrays(Path(episode["last_snapshot_path"]))
        annotation = make_initial_annotation(episode=episode, snapshot_arrays=snapshot, line_width_cells=1, preclose_radius_cells=0)
        annotation = replace(annotation, review=approved_review(approved_by="test"))
        save_annotation_atomic(annotation, annotation_dir / episode["episode_uid"] / "last_step.annotation.json")

    assert main(["build-gt", "--index", str(index_path), "--annotation-dir", str(annotation_dir), "--gt-dir", str(gt_dir)]) == 0
    assert main(["backproject", "--index", str(index_path), "--gt-dir", str(gt_dir), "--step-gt-dir", str(step_gt_dir)]) == 0
    assert main(
        [
            "compute",
            "--index",
            str(index_path),
            "--step-gt-dir",
            str(step_gt_dir),
            "--out-dir",
            str(metrics_dir),
            "--no-require-csr",
            "--min-room-area-m2",
            "0.0",
        ]
    ) == 0

    summary = json.load((metrics_dir / "summary_metrics.json").open("r", encoding="utf-8"))
    assert summary["episode_count"] == 2
    assert summary["snapshot_count"] == 4
    assert summary["CSR"] is None
    assert summary["csr_status"] == "not_required_geometric_only"
    assert np.isclose(summary["USR"], 0.25)
    assert np.isclose(summary["OSR"], 0.0)
    assert np.isclose(summary["mIoU_room"], 0.625)
    assert (metrics_dir / "per_snapshot_metrics.jsonl").exists()
    assert (metrics_dir / "per_snapshot_metrics.csv").exists()
    assert (metrics_dir / "metrics_report.md").exists()
    assert (metrics_dir / "review_gallery.html").exists()
    assert (
        main(
            [
                "compute",
                "--index",
                str(index_path),
                "--step-gt-dir",
                str(step_gt_dir),
                "--out-dir",
                str(eval_dir / "metrics_missing_csr"),
            ]
        )
        == 1
    )


def _make_result_root(tmp_path: Path) -> Path:
    root = tmp_path / "result" / "run_a"
    _make_scene_a(root / "scene_perfect")
    _make_scene_b(root / "scene_underseg")
    return root


def _make_scene_a(scene: Path) -> None:
    domain = np.ones((4, 4), dtype=bool)
    pred = np.ones((4, 4), dtype=np.int32)
    for step in (10, 20):
        _write_snapshot(scene / "roomseg_snapshots" / f"roomseg_step_{step:06d}.npz", domain=domain, pred=pred)
    _write_results(scene, steps=20)


def _make_scene_b(scene: Path) -> None:
    domain = np.zeros((4, 5), dtype=bool)
    domain[:, :2] = True
    domain[:, 3:] = True
    pred = np.zeros((4, 5), dtype=np.int32)
    pred[domain] = 1
    for step in (10, 20):
        _write_snapshot(scene / "roomseg_snapshots" / f"roomseg_step_{step:06d}.npz", domain=domain, pred=pred)
    _write_results(scene, steps=20)


def _write_snapshot(path: Path, *, domain: np.ndarray, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        final_room_label_map=np.asarray(pred, dtype=np.int32),
        navigation_free_room_domain=np.asarray(domain, dtype=bool),
        observed_free_mask=np.asarray(domain, dtype=bool),
        obstacle_mask=np.zeros(domain.shape, dtype=bool),
        unknown_mask=np.zeros(domain.shape, dtype=bool),
    )
    path.with_suffix(".navigation_room_masks.png").write_bytes(b"png")


def _write_results(scene: Path, *, steps: int) -> None:
    with (scene / "results.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"steps": int(steps)}) + "\n")
