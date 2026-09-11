#!/usr/bin/env python3
"""Single-variable DUDE experiment: tau 2.5 -> 1.5, identical saved input grid.

Uses the frozen corrected ROS 1 DUDE bridge and the identical native binary.
Does not import OccuSG's map projection, tracker or room materialization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from collections import defaultdict

import numpy as np

OLD = Path("/media/echo/data/voxroom_dude_corrected_20260910")
EXT = Path("/home/echo/VoxRoom-Online-exp/external_baselines")
OCCUSG = Path("/media/echo/data/voxroom_occusg_paper_20260910/results")
DATA = Path("/media/echo/data/voxroom_roomseg_evaluation")
DATASETS = {"interioragent": DATA / "interioragent_all_available_gt_20260828",
            "grscene": DATA / "grscene_all_available_gt_20260817"}
METRICS = ("precision", "recall", "f1", "miou_room")
EVENTS = ("milestone_020", "milestone_040", "milestone_060", "milestone_070", "milestone_080", "milestone_090", "final")
METHOD = "dude_incremental_tau1p5_raw_vertical_free"
TAU = 1.5
INPUT_KEYS = {"step", "occupancy_map", "observed_free_mask", "obstacle_mask", "unknown_mask",
              "voxel_vertical_free_xy", "height_profile_vertical_free_xy", "vertical_free_room_domain",
              "voxel_wall_xy", "structural_wall_clean", "roomseg_sanitized_wall", "navigation_free_room_domain",
              "roomseg_eval_reference_explorable_mask", "roomseg_eval_explored_reference_mask"}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return row["dataset"], row["episode_uid"], row["coverage_event_id"]


def load_input(path):
    # Omit 3D volumes, neural features, seeds and predicted room labels: the ROS
    # 1 bridge needs only map geometry/masks. Verify its output against the
    # *actual saved ROS input grid* from the tau=2.5 experiment below.
    from voxroom_online.isaac_runtime.baselines.mask_io import SEGMENTATION_INPUT_MODE_KEY
    with np.load(path, allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files if k in INPUT_KEYS or k.startswith("map_")}
    arrays[SEGMENTATION_INPUT_MODE_KEY] = np.asarray("raw_vertical_free")
    return arrays


def assert_identical_input(arrays, old_prediction):
    from voxroom_online.isaac_runtime.baselines.ros_grid_io import snapshot_to_ros_occupancy_grid
    from voxroom_online.isaac_runtime.baselines.data_contract import resolve_map_info
    grid = snapshot_to_ros_occupancy_grid(arrays)
    with np.load(old_prediction, allow_pickle=False) as old:
        old_grid = old["dude_ros_occupancy_grid"]
        metadata = json.loads(str(old["baseline_metadata_json"]))
    np.testing.assert_array_equal(grid, old_grid, err_msg="Input changed from corrected tau=2.5 run")
    assert metadata["segmentation_input_mode"] == "raw_vertical_free"
    assert float(metadata["parameters"]["concavity_threshold_m"]) == 2.5
    current_info = resolve_map_info(snapshot_arrays=arrays, default_resolution_m=.05).to_metadata()
    for field in ("resolution_m", "min_x", "min_y", "max_x", "max_y", "width", "height"):
        assert current_info[field] == metadata["map_info"][field], (field, current_info, metadata["map_info"])
    return grid, hashlib.sha256(grid.tobytes()).hexdigest()


def score_prediction(dataset, episode, record, labels, gt):
    from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import load_snapshot_arrays
    from voxroom_online.isaac_runtime.evaluation.online_roomseg.step_backprojection import backproject_final_gt_to_snapshot
    from voxroom_online.isaac_runtime.evaluation.online_roomseg.metrics import prepare_metric_label_maps, compute_snapshot_metrics
    source = load_snapshot_arrays(Path(record["snapshot_path"]))
    step_gt = backproject_final_gt_to_snapshot(gt, source, episode_uid=episode["episode_uid"],
                                              source_final_step=int(episode["last_snapshot_step"])).label_map
    prepared = prepare_metric_label_maps(step_gt, labels, min_room_area_cells=200)
    metric = compute_snapshot_metrics(prepared.gt, prepared.pred)
    p, r = float(metric["precision"]), float(metric["recall"])
    return {"dataset": dataset, "method": METHOD, "scene_id": episode["scene_id"], "episode_uid": episode["episode_uid"],
            "coverage_event_id": record["coverage_event_id"], "step": int(record["step"]),
            "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0., "miou_room": float(metric["miou_room"]),
            "n_gt": int(metric["n_gt"]), "n_pred": int(metric["n_pred"]), "metric_domain_pixels": int(np.count_nonzero(prepared.metric_domain)),
            "source_snapshot_path": record["snapshot_path"]}


def run(dataset, root):
    from voxroom_online.isaac_runtime.baselines.offline.dude_runner import DudeIncrementalRunner
    from voxroom_online.isaac_runtime.baselines.offline.dude_image_io import DUDE_OUTPUT_COORDINATE_CONTRACT
    from voxroom_online.isaac_runtime.baselines.mask_io import enforce_room_mask_contract
    from voxroom_online.isaac_runtime.comparison.metadata_gate import assert_main_experiment_metadata

    output = root / dataset
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists() and json.loads((output / "status.json").read_text()).get("status") == "complete":
        print(f"{dataset}: already complete; not restarting", flush=True)
        return
    code = root / "code"
    os.environ.update(PYTHONPATH=str(code), VOXROOM_REPO_ROOT=str(code),
                      ROS_MASTER_URI="http://127.0.0.1:" + ("11691" if dataset == "interioragent" else "11692"),
                      ROS_IP="127.0.0.1", ROS_HOSTNAME="localhost", ROS_LOG_DIR=str(output / "ros_logs"),
                      PATH=str(EXT / "ros_noetic_env/bin") + ":" + os.environ["PATH"],
                      TMPDIR=str(root / "tmp"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    (root / "tmp").mkdir(exist_ok=True)
    (output / "ros_logs").mkdir(exist_ok=True)
    old_audit = json.loads((OLD / dataset / "input_audit.json").read_text())
    index_path = OLD / dataset / "inputs/index_voxroom.json"
    index = json.loads(index_path.read_text())
    episodes = index["episodes"]
    excluded = set(old_audit["excluded_scene_ids"])
    assert len(episodes) == old_audit["scene_count"]
    assert sum(len(e["snapshots"]) for e in episodes) == old_audit["checkpoint_count"]
    assert all(e["scene_id"] not in excluded for e in episodes)
    binary = EXT / "dude_ws/devel/lib/inc_dude/inc_dude"
    assert digest(binary) == old_audit["native_binary_sha256"], "native DUDE binary changed"
    for relative, expected in old_audit["code_sha256"].items():
        if relative.startswith("voxroom_online/"):
            assert digest(code / relative) == expected, f"frozen bridge changed: {relative}"
    provenance = {"method": METHOD, "scene_count": len(episodes), "checkpoint_count": old_audit["checkpoint_count"],
                  "excluded_scene_ids": sorted(excluded), "input_index_sha256": digest(index_path),
                  "source_corrected_run": str(OLD / dataset), "driver_sha256": digest(__file__),
                  "native_binary_sha256": digest(binary), "original_upstream_commit": old_audit["upstream_commit"],
                  "output_coordinate_contract": DUDE_OUTPUT_COORDINATE_CONTRACT,
                  "algorithm_parameter_change": {"concavity_threshold_m": {"before": 2.5, "after": TAU}},
                  "unchanged": ["raw Vertical Free + original wall/unknown grid, including map origin and resolution",
                                "corrected ROS 1 bridge and original native DUDE binary", "native morphology (2/10 filters, obstacle dilation 4)",
                                "per-scene incremental replay, no OccuSG tracker or room-materialization transplant",
                                "same held-out scenes/checkpoints/GT projection, 0.5 m2 metric filtering"],
                  "input_equality_policy": "assert exact grid equality before prediction AND on saved bridge output against each old corrected run grid",
                  "transport_only_change": "900 s timeout instead of 180 s; no timeout counted as an empty prediction",
                  "excluded_occusg_parameters": "0.3 m projection would change the input; region tracker/rejection/room graph are additions absent from original DUDE"}
    save_json(output / "inputs/index.json", index)
    save_json(output / "inputs/provenance.json", provenance)
    manifest = json.loads((OLD / dataset / "replay/manifests/dude_incremental.json").read_text())
    old_predictions = {(r["episode_uid"], r["coverage_event_id"]): Path(r["prediction_npz"]) for r in manifest["rows"]}
    rows = []
    try:
        for position, episode in enumerate(episodes):
            uid = episode["episode_uid"]
            scene_output = output / "replay" / uid
            scene_output.mkdir(parents=True, exist_ok=True)
            if (scene_output / "complete.json").exists():
                cached = json.loads((scene_output / "metrics.json").read_text())
                assert len(cached) == len(episode["snapshots"])
                rows.extend(cached)
                continue
            # Completed scenes remain untouched. An interrupted scene must replay
            # from its first checkpoint to rebuild the native incremental state.
            save_json(output / "status.json", {"status": "running", "scene_index": position, "scene": episode["scene_id"],
                                                "completed_checkpoints": len(rows), "total_checkpoints": old_audit["checkpoint_count"], "time": time.time()})
            runner = DudeIncrementalRunner(repo_root=EXT / "dude_ws/src/Incremental_DuDe_ROS", dude_ws=EXT / "dude_ws",
                                           concavity_threshold_m=TAU, use_incremental=True, fallback_python=False, map_resolution_m=.05,
                                           ros_setup=str(EXT / "ros_noetic_env/setup.bash"), ros_python=str(EXT / "ros_noetic_env/bin/python"), timeout_s=900.)
            scene_rows = []
            runner.start_scene(uid)
            gt = np.load(DATASETS[dataset] / "final_gt" / uid / "last_step.gt_labels.npy")
            try:
                for record in episode["snapshots"]:
                    event, path = record["coverage_event_id"], Path(record["snapshot_path"])
                    start = time.monotonic()
                    arrays = load_input(path)
                    original_prediction = old_predictions[uid, event]
                    grid, grid_hash = assert_identical_input(arrays, original_prediction)
                    save_json(scene_output / "status.json", {"status": "running", "event": event, "time": time.time()})
                    result = runner.segment_snapshot(path, arrays)
                    metadata = {**result.metadata, "segmentation_input_mode": "raw_vertical_free", "coverage_reference_used_as_segmentation_input": False,
                                "experiment": METHOD, "input_identical_to_corrected_tau2p5": True, "input_grid_sha256": grid_hash,
                                "reference_prediction": str(original_prediction)}
                    assert_main_experiment_metadata(metadata, "dude_incremental")
                    assert metadata["parameters"]["concavity_threshold_m"] == TAU
                    assert metadata["output_vertical_flip_applied"] is True
                    np.testing.assert_array_equal(result.debug_arrays["dude_ros_occupancy_grid"], grid)
                    np.testing.assert_array_equal(np.flipud(result.debug_arrays["dude_tagged_image_native"]).astype(np.int32),
                                                  result.debug_arrays["dude_labels_source_frame"])
                    np.testing.assert_array_equal(result.label_map, enforce_room_mask_contract(result.debug_arrays["dude_labels_source_frame"], arrays, clip_to_eval_domain=True))
                    prediction_path = scene_output / (event + ".npz")
                    tmp = prediction_path.with_suffix(".npz.tmp")
                    with tmp.open("wb") as stream:
                        np.savez_compressed(stream, final_room_label_map=result.label_map,
                                            baseline_metadata_json=np.asarray(json.dumps(metadata)), **result.debug_arrays)
                    tmp.replace(prediction_path)
                    row = score_prediction(dataset, episode, record, result.label_map, gt)
                    row.update(prediction_path=str(prediction_path), input_grid_sha256=grid_hash, seconds=time.monotonic()-start)
                    scene_rows.append(row)
                    save_json(scene_output / "metrics.json", scene_rows)
                    save_csv(output / "per_checkpoint_metrics.csv", rows + scene_rows)
                    print(json.dumps({"dataset": dataset, "scene": episode["scene_id"], "event": event, "done": len(rows)+len(scene_rows),
                                      "input_identical": True, "f1": row["f1"], "miou": row["miou_room"], "seconds": row["seconds"]}), flush=True)
            finally:
                # Preserve diagnostic logs before closing the existing tempfile.
                if runner._node_output is not None:
                    runner._node_output.flush()
                    runner._node_output.seek(0)
                    with (scene_output / "native.log").open("wb") as native_log:
                        import shutil
                        shutil.copyfileobj(runner._node_output, native_log)
                runner.end_scene()
            rows.extend(scene_rows)
            save_json(scene_output / "complete.json", {"status": "complete", "checkpoints": len(scene_rows), "time": time.time()})
        assert len(rows) == old_audit["checkpoint_count"]
        save_csv(output / "per_checkpoint_metrics.csv", rows)
        save_json(output / "status.json", {"status": "complete", "checkpoint_count": len(rows), "scene_count": len(episodes), "time": time.time()})
    except Exception as exc:
        save_json(output / "status.json", {"status": "failed", "error": repr(exc), "completed_checkpoints": len(rows), "time": time.time()})
        raise


def aggregate(records, method, scope):
    return {"method": method, "scope": scope, "scene_count": len({(r["dataset"], r["scene_id"]) for r in records}), "checkpoint_count": len(records),
            **{m+"_percent": statistics.fmean(100*float(r[m]) for r in records) for m in METRICS}}


def summarize(root, wait):
    while True:
        states = {d: json.loads((root/d/"status.json").read_text()) if (root/d/"status.json").exists() else {} for d in DATASETS}
        if any(s.get("status") == "failed" for s in states.values()):
            raise RuntimeError(states)
        if all(s.get("status") == "complete" for s in states.values()):
            break
        if not wait:
            raise RuntimeError(f"incomplete: {states}")
        time.sleep(30)
    rows, old_rows, main_rows, occu_rows = [], [], [], []
    for dataset in DATASETS:
        current = read_csv(root/dataset/"per_checkpoint_metrics.csv")
        audit = json.loads((root/dataset/"inputs/provenance.json").read_text())
        assert len(current) == audit["checkpoint_count"]
        assert not ({r["scene_id"] for r in current} & set(audit["excluded_scene_ids"]))
        lookup = {key(r): r for r in current}
        assert len(lookup) == len(current)
        episodes = json.loads((root/dataset/"inputs/index.json").read_text())["episodes"]
        by_uid = {e["episode_uid"]: e for e in episodes}
        old_manifest = json.loads((OLD/dataset/"replay/manifests/dude_incremental.json").read_text())
        old_lookup = {(r["episode_uid"], r["coverage_event_id"]): r for r in old_manifest["rows"]}
        gt_cache = {}
        for row in current:
            uid, event = row["episode_uid"], row["coverage_event_id"]
            ep = by_uid[uid]
            record = next(r for r in ep["snapshots"] if r["coverage_event_id"] == event)
            old_path = Path(old_lookup[uid, event]["prediction_npz"])
            with np.load(row["prediction_path"]) as new, np.load(old_path) as old:
                np.testing.assert_array_equal(new["dude_ros_occupancy_grid"], old["dude_ros_occupancy_grid"])
                meta = json.loads(str(new["baseline_metadata_json"]))
                assert meta["parameters"]["concavity_threshold_m"] == TAU
                assert meta["input_identical_to_corrected_tau2p5"] is True
                if uid not in gt_cache:
                    gt_cache[uid] = np.load(DATASETS[dataset]/"final_gt"/uid/"last_step.gt_labels.npy")
                old_score = score_prediction(dataset, ep, record, old["final_room_label_map"], gt_cache[uid])
                old_score["method"] = "dude_incremental_tau2p5_raw_vertical_free"
            for field in ("step", "n_gt", "metric_domain_pixels"):
                assert int(old_score[field]) == int(row[field])
            old_rows.append(old_score)
        pairs = [(Path("/media/echo/data/voxroom_ablation_20260828/evaluation") / (dataset+"_vertical_full_retrain_f1_preview_epoch14/metrics/per_checkpoint_metrics.csv"), main_rows),
                 (OCCUSG/dataset/"per_checkpoint_metrics.csv", occu_rows)]
        for path, target in pairs:
            matched = []
            for row in read_csv(path):
                if key(row) not in lookup:
                    continue
                ref = lookup[key(row)]
                for field in ("step", "n_gt", "metric_domain_pixels"):
                    assert int(row[field]) == int(ref[field]), (field, row, ref)
                assert Path(row["source_snapshot_path"]).resolve() == Path(ref["source_snapshot_path"]).resolve()
                matched.append(row)
            assert len(matched) == len(current)
            target.extend(matched)
        rows.extend(current)
    assert len(rows) == 462 and len({(r["dataset"], r["scene_id"]) for r in rows}) == 68
    tables = root/"tables"
    tables.mkdir(exist_ok=True)
    save_csv(tables/"per_checkpoint_metrics.csv", rows)
    comparison, progress, scene_stats = [], [], []
    for method, data in (("DUDE_corrected_tau2p5_vertical", old_rows), ("DUDE_corrected_tau1p5_vertical", rows),
                         ("OccuSG_official_saved_voxels", occu_rows), ("VoxRoom_main_epoch14", main_rows)):
        comparison.append(aggregate(data, method, "all_checkpoints_weighted"))
        comparison.append(aggregate([r for r in data if r["coverage_event_id"] == "final"], method, "final_only"))
        for event in EVENTS:
            progress.append(aggregate([r for r in data if r["coverage_event_id"] == event], method, event))
        groups = defaultdict(list)
        for row in data:
            groups[row["dataset"], row["scene_id"]].append(row)
        for (dataset, scene), selected in groups.items():
            item = {"method": method, "dataset": dataset, "scene_id": scene, "checkpoint_count": len(selected)}
            for metric in METRICS:
                values = [100*float(r[metric]) for r in selected]
                mean, sd = statistics.fmean(values), statistics.pstdev(values)
                item.update({metric+"_mean": mean, metric+"_sd": sd, metric+"_cv": sd/mean if mean else None, metric+"_worst": min(values)})
            scene_stats.append(item)
    save_csv(tables/"aggregate.csv", comparison)
    save_csv(tables/"progress_metrics.csv", progress)
    save_csv(tables/"per_scene_stability.csv", scene_stats)
    summary = []
    for method in dict.fromkeys(r["method"] for r in scene_stats):
        selected = [r for r in scene_stats if r["method"] == method]
        item = {"method": method, "scene_count": len(selected)}
        for metric in METRICS:
            for suffix in ("_mean", "_sd", "_cv", "_worst"):
                vals = [r[metric+suffix] for r in selected if r[metric+suffix] is not None]
                item[metric+suffix] = statistics.fmean(vals) if vals else None
        summary.append(item)
    save_csv(tables/"stability_summary.csv", summary)
    report = ["# 修复后 DUDE：相同 Vertical Free 输入，阈值对齐 OccuSG", "",
              "68 个独立测试场景、462 个可用进度点；排除训练和验证场景。输入 ROS 栅格、地图原点和分辨率与旧修复版逐格核验相同。",
              "唯一算法参数变化为凹度阈值 2.5 m → 1.5 m。原版 DUDE 的形态学参数本来就与 OccuSG 一致；没有换投影、移植跟踪器或房间生成模块。这是参数对照，不是完整 OccuSG。",
              "统一房间重叠 P/R、逐点 F1 后平均、Hungarian 匹配 IoU 总和除以 GT 房间数；评测小区域过滤为 0.5 m²。两数据集按进度点数量合并。", "",
              "| 方法 | 范围 | P (%) | R (%) | F1 (%) | mIoU (%) |", "|---|---|---:|---:|---:|---:|"]
    for row in comparison:
        report.append(f"| {row['method']} | {row['scope']} | " + " | ".join(f"{row[m+'_percent']:.3f}" for m in METRICS) + " |")
    report += ["", "逐进度与稳定性明细见同目录 CSV。稳定性先逐场景计算（SD 使用 ddof=0），再跨场景等权平均。", "",
               "OccuSG 行为此前官方源码的已存体素快照回放，不是原论文完整 RGB-D 系统成绩。旧阈值结果重新从已有预测计算，没有重跑旧分割。"]
    (tables/"report.md").write_text("\n".join(report)+"\n")
    save_json(root/"completed.json", {"status": "complete", "scenes": 68, "checkpoints": 462, "input_grids_identical": True, "time": time.time()})
    print(json.dumps(comparison, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        if args.dataset is None:
            parser.error("--dataset required")
        run(args.dataset, args.root)
    else:
        summarize(args.root, args.wait)


if __name__ == "__main__":
    main()
