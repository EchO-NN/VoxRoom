from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from voxroom_online.isaac_runtime.baselines.evaluation_bridge import link_baseline_run_root
from voxroom_online.isaac_runtime.baselines.mask_io import save_baseline_snapshot_npz
from voxroom_online.isaac_runtime.comparison.link_method_gt import link_method_gt
from voxroom_online.isaac_runtime.comparison.make_final_report import make_final_report
from voxroom_online.isaac_runtime.comparison.validate_paper_ready import validate_paper_ready
from voxroom_online.isaac_runtime.evaluation.online_roomseg.annotation_schema import (
    approved_review,
    make_initial_annotation,
    save_annotation_atomic,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.cli import main as eval_main
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import load_snapshot_arrays


METHODS = ("topology_visual_active", "dude_incremental", "rose2", "morphological", "distance_transform", "voronoi")


def test_closed_loop_eval_smoke_with_method_gt_mapping(tmp_path: Path) -> None:
    run_root = tmp_path / "result" / "comparison_random_frontier_v1"
    _make_source_run(run_root)
    for method in METHODS:
        _write_method_outputs(run_root, method)
    _write_t0_locks(run_root)

    eval_root = tmp_path / "result" / "comparison_random_frontier_v1_eval"
    assert eval_main(["index", "--result-root", str(run_root), "--out", str(eval_root / "indexes" / "index_voxroom_all.json"), "--snapshot-policy", "all"]) == 0
    assert eval_main(["index", "--result-root", str(run_root), "--out", str(eval_root / "indexes" / "index_voxroom_last.json"), "--snapshot-policy", "last"]) == 0
    _approve_last_annotations(eval_root / "indexes" / "index_voxroom_last.json", eval_root / "annotations")
    assert eval_main(["build-gt", "--index", str(eval_root / "indexes" / "index_voxroom_last.json"), "--annotation-dir", str(eval_root / "annotations"), "--gt-dir", str(eval_root / "gt_final")]) == 0
    assert eval_main(["backproject", "--index", str(eval_root / "indexes" / "index_voxroom_all.json"), "--gt-dir", str(eval_root / "gt_final"), "--step-gt-dir", str(eval_root / "gt_steps_by_method" / "voxroom")]) == 0
    assert eval_main(["compute", "--index", str(eval_root / "indexes" / "index_voxroom_all.json"), "--step-gt-dir", str(eval_root / "gt_steps_by_method" / "voxroom"), "--out-dir", str(eval_root / "metrics" / "voxroom"), "--no-require-csr"]) == 0

    for method in METHODS:
        method_root = eval_root / "method_roots" / method
        link_baseline_run_root(source_run_root=run_root, baseline=method, linked_run_root=method_root, overwrite=True)
        assert eval_main(["index", "--result-root", str(method_root), "--out", str(eval_root / "indexes" / f"index_{method}.json"), "--snapshot-policy", "all"]) == 0
        link_method_gt(
            source_index_path=eval_root / "indexes" / "index_voxroom_all.json",
            method_index_path=eval_root / "indexes" / f"index_{method}.json",
            source_step_gt_dir=eval_root / "gt_steps_by_method" / "voxroom",
            out_step_gt_dir=eval_root / "gt_steps_by_method" / method,
            overwrite=True,
        )
        assert (
            eval_main(
                [
                    "compute",
                    "--index",
                    str(eval_root / "indexes" / f"index_{method}.json"),
                    "--step-gt-dir",
                    str(eval_root / "gt_steps_by_method" / method),
                    "--out-dir",
                    str(eval_root / "metrics" / method),
                    "--no-require-csr",
                ]
            )
            == 0
        )
    report = make_final_report(eval_root=eval_root, methods=("voxroom", *METHODS), out_dir=eval_root / "reports")
    assert len(report["rows"]) == 7
    by_method = {row["method"]: row for row in report["rows"]}
    assert by_method["dude_incremental"]["main_experiment_gate_passed"] is True
    assert "original_ros" in by_method["dude_incremental"]["runner_types"]
    assert "abc123" in by_method["dude_incremental"]["original_repo_commits"]
    assert (eval_root / "reports" / "final_comparison_table.csv").exists()
    assert (eval_root / "reports" / "provenance_manifest.json").exists()
    ready = validate_paper_ready(
        run_root=run_root,
        eval_root=eval_root,
        methods=("voxroom", *METHODS),
        require_csr=False,
        require_native=True,
        require_original_commits=True,
        require_no_placeholder=True,
    )
    assert ready["paper_ready"] is True


def _make_source_run(root: Path) -> None:
    scene = root / "kujiale_0001"
    for step in (1, 2):
        _write_source_snapshot(scene / "roomseg_snapshots" / f"roomseg_step_{step:06d}.npz")
    (scene / "results.jsonl").write_text(json.dumps({"episode_id": "ep_a", "steps": 2}) + "\n", encoding="utf-8")


def _write_source_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    domain = np.ones((4, 5), dtype=bool)
    labels = np.zeros(domain.shape, dtype=np.int32)
    labels[:, :2] = 1
    labels[:, 3:] = 2
    np.savez_compressed(
        path,
        occupancy_map=domain.astype(np.uint8),
        final_room_label_map=labels,
        navigation_free_room_domain=domain,
        observed_free_mask=domain,
        obstacle_mask=np.zeros_like(domain),
        unknown_mask=np.zeros_like(domain),
        frontier_map=np.zeros_like(domain),
        selected_frontier_center_rc=np.asarray([1, 1], dtype=np.int32),
        agent_rc=np.asarray([2, 2], dtype=np.int32),
        map_resolution_m=np.asarray(0.05, dtype=np.float32),
        map_origin_xy_m=np.asarray([0.0, 0.0], dtype=np.float32),
        map_width_cells=np.asarray(domain.shape[1], dtype=np.int32),
        map_height_cells=np.asarray(domain.shape[0], dtype=np.int32),
    )
    path.with_suffix(".navigation_room_masks.png").write_bytes(b"png")


def _write_method_outputs(root: Path, method: str) -> None:
    for source_npz in sorted(root.glob("*/roomseg_snapshots/roomseg_step_*.npz")):
        with np.load(source_npz, allow_pickle=False) as data:
            arrays = {k: np.asarray(data[k]).copy() for k in data.files}
        metadata = _metadata(method)
        save_baseline_snapshot_npz(
            source_npz_path=source_npz,
            output_npz_path=source_npz.parent.parent / "baselines" / method / "roomseg_snapshots" / source_npz.name,
            baseline_label_map=arrays["final_room_label_map"],
            baseline_name=method,
            metadata=metadata,
        )
        out_png = source_npz.parent.parent / "baselines" / method / "roomseg_snapshots" / source_npz.with_suffix(".navigation_room_masks.png").name
        out_png.write_bytes(b"png")


def _write_t0_locks(root: Path) -> None:
    external_repos = {
        "ACTIVE_ROOM_SEG_ROOT": {"path": "external/active", "exists": True, "commit": "abc123"},
        "INCREMENTAL_DUDE_ROS_ROOT": {"path": "external/dude", "exists": True, "commit": "abc123"},
        "ROSE2_ROOT": {"path": "external/rose2", "exists": True, "commit": "abc123"},
        "IPA_COVERAGE_ROOT": {"path": "external/ipa", "exists": True, "commit": "abc123"},
    }
    (root / "environment.lock.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "strict_verify": True,
                "ros_python_check": {"ok": True},
                "topology_checkpoint": {"exists": True, "path": "checkpoint.pth"},
                "external_repos": external_repos,
                "build_artifacts": {"dude_executable": {"exists": True}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "repos.lock.json").write_text(json.dumps({"schema_version": 2, "repos": external_repos}) + "\n", encoding="utf-8")
    (root / "baseline_replay_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "strict_native": True,
                "fallback_python": False,
                "allow_baseline_failure": False,
                "baselines": ["dude_incremental", "rose2", "morphological", "distance_transform", "voronoi"],
                "rows": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _metadata(method: str) -> dict:
    common = {"method": method, "main_experiment_allowed": True, "original_repo_commit": "abc123", "map_resolution_m": 0.05}
    if method == "topology_visual_active":
        return {
            **common,
            "runner_type": "original_repo_adapter",
            "original_repo": "FreeformRobotics/Active_room_segmentation",
            "door_detector": "original_detr",
            "door_detector_available": True,
            "detector_adapter_verified": True,
            "projection_verified": True,
            "topology_state_machine_verified": True,
            "checkpoint_sha256": "sha",
            "policy_control": "never",
            "panorama_views_saved": 12,
        }
    if method == "dude_incremental":
        return {
            **common,
            "runner_type": "original_ros",
            "original_repo": "lfermin77/Incremental_DuDe_ROS",
            "input_topic": "/map",
            "output_topic": "/tagged_image",
            "incremental_state_reset_per_scene": True,
            "incremental_order_enforced": True,
        }
    if method == "rose2":
        return {
            **common,
            "runner_type": "original_ros",
            "original_repo": "aislabunimi/ROSE2",
            "input_topic": "/map",
            "output_source": "/rooms",
            "output_topic_confirmed_from_source": True,
        }
    return {
        **common,
        "runner_type": "original_ros_action",
        "original_repo": "ipa320/ipa_coverage_planning",
        "original_package": "ipa_room_segmentation",
        "room_segmentation_algorithm": {"morphological": 1, "distance_transform": 2, "voronoi": 3}[method],
        "algorithm_id_verified": True,
        "action_type": "ipa_building_msgs/MapSegmentationAction",
        "action_server": "room_segmentation_server",
        "input_image_encoding": "mono8",
        "input_free_value": 255,
        "input_occupied_value": 0,
    }


def _approve_last_annotations(index_path: Path, annotation_dir: Path) -> None:
    index = json.load(index_path.open("r", encoding="utf-8"))
    for episode in index["episodes"]:
        snapshot = load_snapshot_arrays(Path(episode["last_snapshot_path"]))
        annotation = make_initial_annotation(episode=episode, snapshot_arrays=snapshot, line_width_cells=1, preclose_radius_cells=0)
        annotation = replace(annotation, review=approved_review(approved_by="test"))
        save_annotation_atomic(annotation, annotation_dir / episode["episode_uid"] / "last_step.annotation.json")
