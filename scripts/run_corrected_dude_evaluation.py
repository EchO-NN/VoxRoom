#!/usr/bin/env python3
"""Frozen, held-out native DUDE replay, with explicit input/provenance audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from voxroom_online.isaac_runtime.baselines.offline.dude_image_io import DUDE_OUTPUT_COORDINATE_CONTRACT

DATA = Path("/media/echo/data/voxroom_roomseg_evaluation")
DATASETS = {"interioragent": DATA / "interioragent_all_available_gt_20260828",
            "grscene": DATA / "grscene_all_available_gt_20260817"}
EXT = Path("/home/echo/VoxRoom-Online-exp/external_baselines")
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(dataset, out):
    data = DATASETS[dataset]
    exclusions = set().union(*(set(json.loads((p / "training_scene_exclusion_manifest.json").read_text())["excluded_scene_ids"])
                              for p in DATASETS.values()))
    sources = {m: data / ("index_" + m + ".json") for m in ("voxroom", "tvars_original")}
    indexes = {m: json.loads(p.read_text()) for m, p in sources.items()}
    tvars = {e["episode_uid"]: e for e in indexes["tvars_original"]["episodes"]}
    selected, skipped, audited = [], [], []
    for episode in indexes["voxroom"]["episodes"]:
        uid, scene = episode["episode_uid"], episode["scene_id"]
        annotation_path = data / "annotations" / uid / "last_step.annotation.json"
        if not annotation_path.is_file():
            continue
        annotation = json.loads(annotation_path.read_text())
        if annotation["review"]["status"] != "approved":
            continue
        if scene in exclusions:
            skipped.append({"scene_id": scene, "episode_uid": uid})
            continue
        for suffix in ("last_step.gt_labels.npy", "last_step.gt_metadata.json"):
            if not (data / "final_gt" / uid / suffix).is_file():
                raise FileNotFoundError(str(data / "final_gt" / uid / suffix))
        tv_events = {r["coverage_event_id"]: r for r in tvars[uid]["snapshots"]}
        seen = set()
        for record in episode["snapshots"]:
            event = record["coverage_event_id"]
            assert event not in seen
            seen.add(event)
            tv = tv_events[event]
            assert int(record["step"]) == int(tv["step"])
            for rec in (record, tv):
                if not Path(rec["snapshot_path"]).is_file():
                    raise FileNotFoundError(rec["snapshot_path"])
            audited.append({"dataset": dataset, "scene_id": scene, "episode_uid": uid,
                            "event": event, "step": record["step"],
                            "snapshot_path": record["snapshot_path"]})
        selected.append(episode)
    assert selected, "no held-out approved scenes"
    assert len({e["scene_id"] for e in selected}) == len(selected)
    for method, index in indexes.items():
        chosen = selected if method == "voxroom" else [tvars[e["episode_uid"]] for e in selected]
        write_json(out / "inputs" / ("index_" + method + ".json"), {**index, "episodes": chosen})
    write_json(out / "inputs/exclusion_manifest.json", {
        "policy": "exclude every training and validation scene across both datasets",
        "excluded_scene_ids": sorted(exclusions)})
    upstream = EXT / "dude_ws/src/Incremental_DuDe_ROS"
    patch = subprocess.check_output(["git", "-C", str(upstream), "diff"], text=True)
    (out / "inputs/upstream_build_compatibility.patch").write_text(patch)
    audit = {"dataset": dataset, "scene_count": len(selected), "checkpoint_count": len(audited),
             "excluded_approved_scenes": skipped, "excluded_scene_ids": sorted(exclusions),
             "source_index_sha256": {m: digest(p) for m, p in sources.items()},
             "upstream_commit": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
             "upstream_worktree_patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
             "native_binary_sha256": digest(EXT / "dude_ws/devel/lib/inc_dude/inc_dude"),
             "code_sha256": {str(p.relative_to(ROOT)): digest(p) for folder in ("voxroom_online", "scripts")
                             for p in sorted((ROOT / folder).rglob("*.py"))},
             "output_coordinate_contract": DUDE_OUTPUT_COORDINATE_CONTRACT,
             "concavity_threshold_m": 2.5, "input": "raw_vertical_free, no GT/reference clipping before algorithm",
             "online_scope": "incremental replay of saved coverage snapshots; not full sensor stream",
             "records": audited}
    write_json(out / "input_audit.json", audit)
    print(json.dumps({k: audit[k] for k in ("dataset", "scene_count", "checkpoint_count")}), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = prepare(args.dataset, out)
    if args.prepare_only:
        return
    if (out / "replay").exists() or (out / "metrics").exists():
        raise FileExistsError("refusing to mix or overwrite a previous replay; choose a fresh output directory")
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT), VOXROOM_REPO_ROOT=str(ROOT),
               ROS_MASTER_URI="http://127.0.0.1:" + ("11681" if args.dataset == "interioragent" else "11682"),
               ROS_IP="127.0.0.1", ROS_HOSTNAME="localhost", ROS_LOG_DIR=str(out / "ros_logs"),
               PATH=str(EXT / "ros_noetic_env/bin") + ":" + os.environ["PATH"],
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    (out / "ros_logs").mkdir(exist_ok=True)
    commands = [
        [sys.executable, "-u", str(ROOT / "scripts/replay_grscene_paper_segmentation_baselines.py"),
         "--voxroom-index", str(out / "inputs/index_voxroom.json"), "--output-root", str(out / "replay"),
         "--methods", "dude_incremental", "dude_offline", "--segmentation-input-map", "raw_vertical_free",
         "--strict-main", "--no-resume", "--dude-concavity-threshold-m", "2.5", "--map-resolution-m", "0.05",
         "--dude-repo-root", str(EXT / "dude_ws/src/Incremental_DuDe_ROS"), "--dude-ws", str(EXT / "dude_ws"),
         "--ipa-ros-workspace", str(EXT / "ipa_ws"), "--rose2-ros-workspace", str(EXT / "rose2_ws"),
         "--ros-baseline-setup", str(EXT / "ros_noetic_env/setup.bash"),
         "--ros-baseline-python", str(EXT / "ros_noetic_env/bin/python")],
        [sys.executable, "-u", str(ROOT / "scripts/evaluate_grscene_all_paper_segmentation_baselines.py"),
         "--dataset-name", args.dataset, "--voxroom-index", str(out / "inputs/index_voxroom.json"),
         "--tvars-index", str(out / "inputs/index_tvars_original.json"), "--replay-root", str(out / "replay"),
         "--annotation-dir", str(DATASETS[args.dataset] / "annotations"),
         "--gt-dir", str(DATASETS[args.dataset] / "final_gt"),
         "--paper", str(DATASETS[args.dataset] / "inputs/Topology-Based_Visual_Active_Room_Segmentation.pdf"),
         "--out-dir", str(out / "metrics"), "--methods", "dude_incremental", "dude_offline",
         "--training-exclusion-manifest", str(out / "inputs/exclusion_manifest.json"),
         "--min-room-area-m2", "0.5", "--cell-size-m", "0.05"],
    ]
    write_json(out / "commands.json", commands)
    try:
        for stage, command in zip(("replay", "metrics"), commands):
            write_json(out / "status.json", {"stage": stage, "status": "running", "time": time.time()})
            with (out / (stage + ".log")).open("w") as log:
                subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((out / "metrics/summary_excluding_training.json").read_text())
        assert summary["checkpoint_evaluation_count_per_method"] == audit["checkpoint_count"]
        write_json(out / "status.json", {"stage": "complete", "status": "complete", "time": time.time()})
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "error": repr(exc), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
