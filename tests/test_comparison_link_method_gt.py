from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from voxroom_online.isaac_runtime.baselines.mask_io import save_baseline_snapshot_npz
from voxroom_online.isaac_runtime.baselines.evaluation_bridge import link_baseline_run_root
from voxroom_online.isaac_runtime.comparison.link_method_gt import link_method_gt
from voxroom_online.isaac_runtime.evaluation.online_roomseg.annotation_schema import (
    approved_review,
    make_initial_annotation,
    save_annotation_atomic,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.cli import main as eval_main
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import load_snapshot_arrays


def test_link_method_gt_maps_source_gt_to_baseline_uid_and_compute_runs(tmp_path: Path) -> None:
    source_root = tmp_path / "result" / "comparison_random_frontier_v1"
    scene = source_root / "kujiale_0001"
    _write_source_snapshot(scene / "roomseg_snapshots" / "roomseg_step_000001.npz")
    _write_source_snapshot(scene / "roomseg_snapshots" / "roomseg_step_000002.npz")
    (scene / "results.jsonl").write_text(json.dumps({"episode_id": "episode_a", "steps": 2}) + "\n", encoding="utf-8")
    _write_baseline_snapshots(source_root)

    eval_root = tmp_path / "eval"
    source_index = eval_root / "indexes" / "index_voxroom_all.json"
    source_last = eval_root / "indexes" / "index_voxroom_last.json"
    method_root = eval_root / "method_roots" / "dude_incremental"
    method_index = eval_root / "indexes" / "index_dude_incremental.json"
    assert eval_main(["index", "--result-root", str(source_root), "--out", str(source_index), "--snapshot-policy", "all"]) == 0
    assert eval_main(["index", "--result-root", str(source_root), "--out", str(source_last), "--snapshot-policy", "last"]) == 0
    _approve_last_annotations(source_last, eval_root / "annotations")
    assert eval_main(["build-gt", "--index", str(source_last), "--annotation-dir", str(eval_root / "annotations"), "--gt-dir", str(eval_root / "gt_final")]) == 0
    assert eval_main(["backproject", "--index", str(source_index), "--gt-dir", str(eval_root / "gt_final"), "--step-gt-dir", str(eval_root / "gt_steps_by_method" / "voxroom")]) == 0

    link_baseline_run_root(
        source_run_root=source_root,
        baseline="dude_incremental",
        linked_run_root=method_root,
        overwrite=True,
    )
    assert eval_main(["index", "--result-root", str(method_root), "--out", str(method_index), "--snapshot-policy", "all"]) == 0
    src_uid = json.load(source_index.open("r", encoding="utf-8"))["episodes"][0]["episode_uid"]
    dst_uid = json.load(method_index.open("r", encoding="utf-8"))["episodes"][0]["episode_uid"]
    assert src_uid != dst_uid

    manifest = link_method_gt(
        source_index_path=source_index,
        method_index_path=method_index,
        source_step_gt_dir=eval_root / "gt_steps_by_method" / "voxroom",
        out_step_gt_dir=eval_root / "gt_steps_by_method" / "dude_incremental",
        overwrite=True,
    )
    assert len(manifest["rows"]) == 2
    assert (eval_root / "gt_steps_by_method" / "dude_incremental" / dst_uid / "roomseg_step_000001.gt_labels.npy").exists()
    metadata = json.load((eval_root / "gt_steps_by_method" / "dude_incremental" / dst_uid / "roomseg_step_000001.gt_metadata.json").open("r", encoding="utf-8"))
    assert metadata["source_episode_uid"] == src_uid
    assert metadata["episode_uid"] == dst_uid

    assert (
        eval_main(
            [
                "compute",
                "--index",
                str(method_index),
                "--step-gt-dir",
                str(eval_root / "gt_steps_by_method" / "dude_incremental"),
                "--out-dir",
                str(eval_root / "metrics" / "dude_incremental"),
                "--no-require-csr",
            ]
        )
        == 0
    )


def _write_source_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    domain = np.ones((5, 6), dtype=bool)
    unknown = np.zeros_like(domain)
    unknown[0, 0] = True
    domain[0, 0] = False
    labels = np.zeros(domain.shape, dtype=np.int32)
    labels[domain] = 1
    np.savez_compressed(
        path,
        occupancy_map=domain.astype(np.uint8),
        final_room_label_map=labels,
        navigation_free_room_domain=domain,
        observed_free_mask=domain,
        obstacle_mask=np.zeros_like(domain),
        unknown_mask=unknown,
        frontier_map=np.zeros_like(domain),
        selected_frontier_center_rc=np.asarray([1, 1], dtype=np.int32),
        agent_rc=np.asarray([2, 2], dtype=np.int32),
        map_resolution_m=np.asarray(0.05, dtype=np.float32),
        map_origin_xy_m=np.asarray([0.0, 0.0], dtype=np.float32),
        map_width_cells=np.asarray(domain.shape[1], dtype=np.int32),
        map_height_cells=np.asarray(domain.shape[0], dtype=np.int32),
    )
    path.with_suffix(".navigation_room_masks.png").write_bytes(b"png")


def _write_baseline_snapshots(source_root: Path) -> None:
    for source_npz in sorted(source_root.glob("*/roomseg_snapshots/roomseg_step_*.npz")):
        with np.load(source_npz, allow_pickle=False) as data:
            arrays = {k: np.asarray(data[k]).copy() for k in data.files}
        metadata = {
            "method": "dude_incremental",
            "runner_type": "original_ros",
            "main_experiment_allowed": True,
            "original_repo": "lfermin77/Incremental_DuDe_ROS",
            "original_repo_commit": "abc123",
            "input_topic": "/map",
            "output_topic": "/tagged_image",
            "incremental_state_reset_per_scene": True,
            "map_resolution_m": 0.05,
        }
        save_baseline_snapshot_npz(
            source_npz_path=source_npz,
            output_npz_path=source_npz.parent.parent / "baselines" / "dude_incremental" / "roomseg_snapshots" / source_npz.name,
            baseline_label_map=arrays["final_room_label_map"],
            baseline_name="dude_incremental",
            metadata=metadata,
        )
        out_png = source_npz.parent.parent / "baselines" / "dude_incremental" / "roomseg_snapshots" / source_npz.with_suffix(".navigation_room_masks.png").name
        out_png.write_bytes(b"png")


def _approve_last_annotations(index_path: Path, annotation_dir: Path) -> None:
    index = json.load(index_path.open("r", encoding="utf-8"))
    for episode in index["episodes"]:
        snapshot = load_snapshot_arrays(Path(episode["last_snapshot_path"]))
        annotation = make_initial_annotation(episode=episode, snapshot_arrays=snapshot, line_width_cells=1, preclose_radius_cells=0)
        annotation = replace(annotation, review=approved_review(approved_by="test"))
        save_annotation_atomic(annotation, annotation_dir / episode["episode_uid"] / "last_step.annotation.json")

