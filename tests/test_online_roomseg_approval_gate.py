from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voxroom_online.isaac_runtime.evaluation.online_roomseg.annotation_schema import (
    make_initial_annotation,
    save_annotation_atomic,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.cli import main
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import load_snapshot_arrays


def test_backproject_rejects_draft_final_gt(tmp_path: Path) -> None:
    result_root = _make_result_root(tmp_path)
    index_path = tmp_path / "eval" / "index.json"
    annotation_dir = tmp_path / "eval" / "annotations"
    gt_dir = tmp_path / "eval" / "gt_final"
    step_gt_dir = tmp_path / "eval" / "gt_by_step"

    assert main(["index", "--result-root", str(result_root), "--out", str(index_path), "--snapshot-policy", "all"]) == 0
    snapshot = load_snapshot_arrays(result_root / "scene_001" / "roomseg_snapshots" / "roomseg_step_000020.npz")
    episode = json.load(index_path.open("r", encoding="utf-8"))["episodes"][0]
    annotation = make_initial_annotation(episode=episode, snapshot_arrays=snapshot, line_width_cells=1, preclose_radius_cells=0)
    annotation_path = annotation_dir / episode["episode_uid"] / "last_step.annotation.json"
    save_annotation_atomic(annotation, annotation_path)

    assert main(["build-gt", "--index", str(index_path), "--annotation-dir", str(annotation_dir), "--gt-dir", str(gt_dir), "--include-draft"]) == 0
    assert main(["backproject", "--index", str(index_path), "--gt-dir", str(gt_dir), "--step-gt-dir", str(step_gt_dir)]) == 1
    assert main(["backproject", "--index", str(index_path), "--gt-dir", str(gt_dir), "--step-gt-dir", str(step_gt_dir), "--allow-draft"]) == 0
    assert (
        main(
            [
                "review-csr",
                "--index",
                str(index_path),
                "--step-gt-dir",
                str(step_gt_dir),
                "--csr-dir",
                str(tmp_path / "eval" / "csr"),
                "--default-csr",
                "1",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "compute",
                "--index",
                str(index_path),
                "--step-gt-dir",
                str(step_gt_dir),
                "--csr-dir",
                str(tmp_path / "eval" / "csr"),
                "--out-dir",
                str(tmp_path / "eval" / "metrics_non_strict"),
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "compute",
                "--index",
                str(index_path),
                "--step-gt-dir",
                str(step_gt_dir),
                "--csr-dir",
                str(tmp_path / "eval" / "csr"),
                "--out-dir",
                str(tmp_path / "eval" / "metrics"),
                "--strict-paper",
            ]
        )
        == 1
    )


def _make_result_root(tmp_path: Path) -> Path:
    root = tmp_path / "result" / "run_a"
    scene = root / "scene_001"
    for step in (10, 20):
        _write_snapshot(scene / "roomseg_snapshots" / f"roomseg_step_{step:06d}.npz")
    return root


def _write_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    domain = np.ones((4, 4), dtype=bool)
    np.savez_compressed(
        path,
        final_room_label_map=np.ones((4, 4), dtype=np.int32),
        navigation_free_room_domain=domain,
        observed_free_mask=domain,
        obstacle_mask=np.zeros_like(domain),
        unknown_mask=np.zeros_like(domain),
    )
    path.with_suffix(".navigation_room_masks.png").write_bytes(b"png")
