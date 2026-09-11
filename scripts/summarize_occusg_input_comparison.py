#!/usr/bin/env python3
"""Paired evaluation of OccuSG with its projection vs raw Vertical Free."""
import argparse
from collections import Counter, defaultdict
import json
import math
import re
from pathlib import Path
import statistics
import time

import numpy as np

from summarize_occusg_replay import METRICS, EVENTS, read, write, key, aggregate


def matched_rows(path, lookup):
    rows = [r for r in read(path) if key(r) in lookup]
    assert len(rows) == len(lookup) and {key(r) for r in rows} == set(lookup)
    for row in rows:
        ref = lookup[key(row)]
        assert Path(row["source_snapshot_path"]).resolve() == Path(ref["source_snapshot_path"]).resolve()
        for field in ("step", "n_gt", "metric_domain_pixels"):
            assert int(row[field]) == int(ref[field]), (field, row, ref)
    return rows


def audit_input_grid(prediction_path, reference_path, cropped=False):
    with np.load(prediction_path) as prediction, np.load(reference_path) as ref:
        full = ref["dude_ros_occupancy_grid"]
        info = json.loads(str(ref["baseline_metadata_json"]))["map_info"]
        actual = prediction["map_uav"]
        origin = [info["min_x"], info["min_y"]]
        if cropped:
            from run_occusg_saved_voxels import crop_unknown_border
            expected, expected_info = crop_unknown_border(full, {"shape": list(full.shape), "origin": origin, "resolution": info["resolution_m"]})
            metadata = json.loads(prediction_path.with_name("metadata.json").read_text())["geometry"]
            for field in ("crop_bounds_yxyx", "origin", "source_origin", "shape", "source_shape", "removed_known_cells", "removed_unknown_cells", "full_input_grid_sha256", "input_grid_sha256"):
                assert metadata[field] == expected_info[field], (field, metadata, expected_info)
            np.testing.assert_array_equal(actual, expected)
            origin = expected_info["origin"]
        else:
            np.testing.assert_array_equal(actual, full)
        assert prediction["room_label_map"].shape == full.shape
        np.testing.assert_array_equal(prediction["map_uav_origin_xy"], origin)
        assert float(prediction["map_uav_resolution"]) == float(np.float32(info["resolution_m"]))
        return {"source_height": full.shape[0], "source_width": full.shape[1],
                "input_height": actual.shape[0], "input_width": actual.shape[1],
                "unknown_ratio_before": float(np.mean(full < 0)), "unknown_ratio_after": float(np.mean(actual < 0)),
                "removed_known_cells": int(np.sum(full >= 0) - np.sum(actual >= 0)),
                "removed_unknown_cells": int(full.size - actual.size)}


def rejection_summary(root, method):
    reasons, scenes = Counter(), set()
    for path in root.glob("*/replay/*/native_*.log"):
        matches = re.findall(r"Rejected suspicious decomposition update reason=(\S+)", path.read_text())
        reasons.update(matches)
        if matches:
            scenes.add(path.parent.name)
    return {"method": method, "rejected_updates": sum(reasons.values()), "affected_scenes": len(scenes),
            "unknown_ratio_rejections": reasons["unknown_ratio_too_high"], "reasons_json": json.dumps(dict(reasons))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--dude-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--uncropped-root", type=Path)
    parser.add_argument("--cropped", action="store_true")
    args = parser.parse_args()
    if args.cropped and args.uncropped_root is None:
        parser.error("--cropped requires --uncropped-root for paired padding control")
    input_mode = "raw_vertical_free_cropped" if args.cropped else "raw_vertical_free"
    current_method = "OccuSG_VerticalFree_cropped" if args.cropped else "OccuSG_VerticalFree"
    datasets = ("interioragent", "grscene")
    while True:
        states = [json.loads(p.read_text()) if p.exists() else {} for root in (args.root, args.dude_root)
                  for d in datasets for p in [root / d / "status.json"]]
        if any(s.get("status") == "failed" for s in states):
            raise RuntimeError(states)
        if all(s.get("status") == "complete" for s in states):
            break
        if not args.wait:
            raise RuntimeError(f"Not complete: {states}")
        time.sleep(30)
    assert json.loads((args.original_root / "completed.json").read_text())["status"] == "complete"
    if args.cropped:
        assert json.loads((args.uncropped_root / "completed.json").read_text())["status"] == "complete"
    groups = {m: [] for m in (current_method, "OccuSG_official_projection", "DUDE_tau1p5_VerticalFree", "VoxRoom_main_epoch14")}
    if args.cropped:
        groups["OccuSG_VerticalFree_uncropped"] = []
    input_audit = []
    for dataset, expected_count in zip(datasets, (64, 398)):
        current = read(args.root / dataset / "per_checkpoint_metrics.csv")
        provenance = json.loads((args.root / dataset / "inputs/provenance.json").read_text())
        assert provenance["input_mode"] == input_mode
        if args.cropped:
            before = json.loads((args.uncropped_root / dataset / "inputs/provenance.json").read_text())
            for field in ("backend_sha256", "official_parameter_files", "upstream_commit", "source_index_sha256"):
                assert provenance[field] == before[field], field
        assert len(current) == expected_count == provenance["checkpoints"]
        assert not {r["scene_id"] for r in current} & set(provenance["excluded_training_and_validation_scene_ids"])
        lookup = {key(r): r for r in current}
        assert len(lookup) == len(current)
        for row in current:
            assert row["method"] == "occusg_official_" + input_mode
            assert all(math.isfinite(float(row[m])) and 0 <= float(row[m]) <= 1 for m in METRICS)
            p, r = float(row["precision"]), float(row["recall"])
            assert math.isclose(float(row["f1"]), 2*p*r/(p+r) if p+r else 0, abs_tol=1e-12)
            uid, event = row["episode_uid"], row["coverage_event_id"]
            scene_path = args.root / dataset / "replay" / uid
            audit = audit_input_grid(scene_path / event / "prediction.npz", args.reference_root / dataset / uid / (event + ".npz"), args.cropped)
            input_audit.append({"dataset": dataset, "episode_uid": uid, "coverage_event_id": event, **audit})
        for uid in {r["episode_uid"] for r in current}:
            actual = json.loads((args.root / dataset / "replay" / uid / "effective_parameters.json").read_text())
            original = json.loads((args.original_root / dataset / "replay" / uid / "effective_parameters.json").read_text())
            assert "map_conversion_node" not in actual
            for component in ("incremental_decomposer", "scene_graph_region"):
                assert actual[component] == original[component], (dataset, uid, component)
        groups[current_method].extend(current)
        pairs = [
            ("OccuSG_official_projection", args.original_root / dataset / "per_checkpoint_metrics.csv"),
            ("DUDE_tau1p5_VerticalFree", args.dude_root / dataset / "per_checkpoint_metrics.csv"),
            ("VoxRoom_main_epoch14", Path("/media/echo/data/voxroom_ablation_20260828/evaluation") / f"{dataset}_vertical_full_retrain_f1_preview_epoch14/metrics/per_checkpoint_metrics.csv"),
        ]
        if args.cropped:
            pairs.append(("OccuSG_VerticalFree_uncropped", args.uncropped_root / dataset / "per_checkpoint_metrics.csv"))
        for method, path in pairs:
            groups[method].extend(matched_rows(path, lookup))
    tables = args.root / "tables"
    tables.mkdir(exist_ok=True)
    write(tables / "input_boundary_audit.csv", input_audit)
    guards = [rejection_summary(args.root, current_method), rejection_summary(args.original_root, "OccuSG_official_projection")]
    if args.cropped:
        guards.append(rejection_summary(args.uncropped_root, "OccuSG_VerticalFree_uncropped"))
    write(tables / "update_guard_audit.csv", guards)
    combined, progress, scene_rows, stability = [], [], [], []
    for method, rows in groups.items():
        assert len(rows) == 462 and len({(r["dataset"], r["scene_id"]) for r in rows}) == 68
        write(tables / (method + "_per_checkpoint.csv"), rows)
        for scope, selected in [("all_checkpoints_weighted", rows), ("final_only", [r for r in rows if r["coverage_event_id"] == "final"])]:
            combined.append(aggregate(selected, method, scope))
        for dataset in datasets:
            combined.append(aggregate([r for r in rows if r["dataset"] == dataset], method, dataset + "_all_checkpoints"))
        for event in EVENTS:
            progress.append(aggregate([r for r in rows if r["coverage_event_id"] == event], method, event))
        by_scene = defaultdict(list)
        for row in rows:
            by_scene[row["dataset"], row["scene_id"]].append(row)
        method_scenes = []
        for (dataset, scene), selected in by_scene.items():
            item = {"method": method, "dataset": dataset, "scene_id": scene, "checkpoint_count": len(selected)}
            for metric in METRICS:
                values = [100*float(r[metric]) for r in selected]
                mean, sd = statistics.fmean(values), statistics.pstdev(values)
                item.update({metric+"_mean_percent": mean, metric+"_sd_pp": sd,
                             metric+"_cv": sd/mean if mean else None, metric+"_worst_percent": min(values)})
            method_scenes.append(item)
        scene_rows.extend(method_scenes)
        summary = {"method": method, "scene_count": len(method_scenes)}
        for metric in METRICS:
            for suffix in ("_mean_percent", "_sd_pp", "_cv", "_worst_percent"):
                values = [r[metric+suffix] for r in method_scenes if r[metric+suffix] is not None]
                summary[metric+suffix] = statistics.fmean(values) if values else None
            summary[metric+"_cv_valid_scenes"] = sum(r[metric+"_cv"] is not None for r in method_scenes)
        stability.append(summary)
    for name, rows in (("aggregate", combined), ("progress_metrics", progress), ("per_scene_stability", scene_rows), ("stability_summary", stability)):
        write(tables / (name + ".csv"), rows)
    lines = ["# OccuSG 输入地图对照", "", "68 个独立测试场景、462 个进度点；排除训练和验证场景。", "",
             "OccuSG_VerticalFree 仅将官方 OctoMap→mapUAV 投影替换成原保存的 Vertical Free + wall/unknown 栅格。DUDE 1.5 m、区域跟踪、异常更新保护、房间生成参数与原 OccuSG 快照实验逐场景核对相同。输入逐格与修复版 DUDE 核对相同。", "",
             "相同源快照、step、GT 房间数、评测域面积均逐点核对。P/R 为房间重叠指标，F1 逐点计算后平均；mIoU 为 Hungarian IoU 总和/nGT，0.5 m² 小房间过滤。不是原论文完整 RGB-D/语义流或其原始指标。", "",
             "| 方法 | 范围 | 点数 | P (%) | R (%) | F1 (%) | mIoU (%) |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in combined:
        lines.append(f"| {row['method']} | {row['scope']} | {row['checkpoint_count']} | " + " | ".join(f"{row[m+'_percent']:.3f}" for m in METRICS) + " |")
    lines.extend(["", "合并结果按有效进度点数加权，不将两数据集均值直接平均。稳定性先逐场景计算 mean、总体 SD、CV、worst，再对场景等权平均；缺失进度不补齐。", "",
                  "只有 OccuSG 两种输入之间是输入单变量对照。与原版 DUDE 的比较还包含跟踪、更新保护和房间生成流程的差别。"])
    if args.cropped:
        lines[0] = "# OccuSG Vertical Free：排除外围 unknown 填充影响"
        lines[4] = "本轮仅裁掉原始 Vertical Free 栅格外围全 unknown 行列，保留每个已观测格、内部 unknown、分辨率和世界坐标。没有改 98% unknown 保护、1.5 m 分解阈值、跟踪或房间生成参数。预测仍回栅格到原始完整地图，评测域不变。"
        lines.extend(["", "输入边界审计：input_boundary_audit.csv；每张裁剪图补回外围 unknown 后必须与历史输入逐格相同；所有 removed_known_cells 必须为 0。", "",
                      "| 方法 | 拒绝更新次数 | 涉及场景 | unknown 比例触发次数 |", "|---|---:|---:|---:|"])
        for guard in guards:
            lines.append(f"| {guard['method']} | {guard['rejected_updates']} | {guard['affected_scenes']} | {guard['unknown_ratio_rejections']} |")
        lines.extend(["", "max_region_area_drop_ratio=0.85 和 max_region_count_drop_ratio=0.85 在源码中代表允许下降 85%（保留比例低于 15% 才拒绝），不是下降 15% 就拒绝。当前所有阈值与官方 YAML 一致。",
                      "裁剪会改变数组尺寸和原点（不改变已知格的世界位置）；这轮衡量边界适配修正的整体效果，不把所有分数变化单独归因于某一次拒绝更新。"])
    (tables / "report.md").write_text("\n".join(lines) + "\n")
    (args.root / "completed.json").write_text(json.dumps({"status": "complete", "scenes": 68, "checkpoints": 462, "time": time.time()}, indent=2))
    print("All 462 predictions, input grids, backend parameters and paired metrics verified.", flush=True)


if __name__ == "__main__":
    main()
