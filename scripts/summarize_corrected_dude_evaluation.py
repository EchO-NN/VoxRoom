#!/usr/bin/env python3
"""Validate and aggregate the corrected DUDE replay once both datasets finish."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time
from collections import defaultdict

import numpy as np

from voxroom_online.isaac_runtime.baselines.offline.dude_image_io import DUDE_OUTPUT_COORDINATE_CONTRACT
from voxroom_online.isaac_runtime.baselines.mask_io import enforce_room_mask_contract

EVENTS = ["milestone_020", "milestone_040", "milestone_060", "milestone_070", "milestone_080", "milestone_090", "final"]
METRICS = ("precision", "recall", "f1", "miou_room")
METHODS = ("dude_incremental", "dude_offline")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, method, scope):
    return {"method": method, "scope": scope,
            "scene_count": len({(r["dataset"], r["scene_id"]) for r in rows}), "checkpoint_count": len(rows),
            **{k + "_percent": statistics.fmean(100 * float(r[k]) for r in rows) for k in METRICS}}


def key(row):
    return row["dataset"], row["episode_uid"], row["coverage_event_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    while True:
        statuses = {d: json.loads((root / d / "status.json").read_text())
                    if (root / d / "status.json").is_file() else {} for d in ("interioragent", "grscene")}
        if any(v.get("status") == "failed" for v in statuses.values()):
            raise RuntimeError("dataset evaluation failed: " + json.dumps(statuses))
        if all(v.get("status") == "complete" for v in statuses.values()):
            break
        if not args.wait:
            raise RuntimeError("evaluations still running: " + json.dumps(statuses))
        time.sleep(30)
    out = root / "tables"
    out.mkdir(exist_ok=False)
    rows, old_rows, main_rows, audits = [], [], [], []
    for dataset in ("interioragent", "grscene"):
        audit = json.loads((root / dataset / "input_audit.json").read_text())
        audits.append(audit)
        excluded = set(audit["excluded_scene_ids"])
        records = read_csv(root / dataset / "metrics/per_checkpoint_metrics_excluding_training.csv")
        selected = [r for r in records if r["method"] in METHODS]
        assert len(selected) == 2 * audit["checkpoint_count"]
        assert all(r["scene_id"] not in excluded for r in selected)
        reference = {key(r): r for r in selected if r["method"] == METHODS[0]}
        for row in selected:
            p, r, f = (float(row[k]) for k in ("precision", "recall", "f1"))
            assert math.isclose(f, 2*p*r/(p+r) if p+r else 0, abs_tol=1e-12)
            assert all(math.isfinite(float(row[k])) and 0 <= float(row[k]) <= 1 for k in METRICS)
            with np.load(row["prediction_path"], allow_pickle=False) as pred:
                metadata = json.loads(str(pred["baseline_metadata_json"]))
                assert metadata["output_coordinate_contract"] == DUDE_OUTPUT_COORDINATE_CONTRACT
                assert metadata["parameters"]["concavity_threshold_m"] == 2.5
                assert metadata["runner_type"] == "original_ros"
                assert metadata["coverage_reference_used_as_segmentation_input"] is False
                assert metadata["segmentation_input_mode"] == "raw_vertical_free"
                native = pred["dude_tagged_image_native"]
                source_frame = pred["dude_labels_source_frame"]
                np.testing.assert_array_equal(np.flipud(native).astype(np.int32), source_frame)
                with np.load(row["source_snapshot_path"], allow_pickle=False) as source:
                    minimal = {k: source[k] for k in ("occupancy_map", "observed_free_mask", "obstacle_mask", "unknown_mask",
                               "roomseg_eval_reference_explorable_mask", "roomseg_eval_explored_reference_mask", "navigation_free_room_domain")
                               if k in source.files}
                np.testing.assert_array_equal(pred["final_room_label_map"],
                    enforce_room_mask_contract(source_frame, minimal, clip_to_eval_domain=True))
        rows.extend(selected)
        old_path = Path("/media/echo/data/voxroom_roomseg_evaluation") / (dataset + "_paper_segmentation_baselines_raw_vertical_20260828") / "metrics_all_methods/per_checkpoint_metrics.csv"
        main_path = Path("/media/echo/data/voxroom_ablation_20260828/evaluation") / (dataset + "_vertical_full_retrain_f1_preview_epoch14/metrics/per_checkpoint_metrics.csv")
        for path, target, allowed in ((old_path, old_rows, METHODS), (main_path, main_rows, None)):
            for row in read_csv(path):
                if key(row) not in reference or (allowed is not None and row.get("method") not in allowed):
                    continue
                ref = reference[key(row)]
                assert Path(row["source_snapshot_path"]).resolve() == Path(ref["source_snapshot_path"]).resolve()
                assert int(row["step"]) == int(ref["step"])
                assert int(row["n_gt"]) == int(ref["n_gt"])
                assert int(row["metric_domain_pixels"]) == int(ref["metric_domain_pixels"])
                target.append(row)
    assert len(rows) == 924
    assert len(main_rows) == 462 and len(old_rows) == 924
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        assert len({key(r) for r in selected}) == 462
        assert len({(r["dataset"], r["scene_id"]) for r in selected}) == 68
    all_table = [aggregate([r for r in rows if r["method"] == m], m, "all_checkpoints_weighted") for m in METHODS]
    all_table += [aggregate([r for r in rows if r["method"] == m and r["coverage_event_id"] == "final"], m, "final_only") for m in METHODS]
    progress = [aggregate([r for r in rows if r["method"] == m and r["coverage_event_id"] == e], m, e) for m in METHODS for e in EVENTS]
    comparison = [aggregate(main_rows, "VoxRoom_paper_main_epoch14", "same_462_checkpoints")]
    for method in METHODS:
        comparison.append(aggregate([r for r in old_rows if r["method"] == method], method + "_OLD_INVALID_COORDINATES", "same_462_checkpoints"))
        comparison.append(aggregate([r for r in rows if r["method"] == method], method + "_CORRECTED", "same_462_checkpoints"))
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["dataset"], row["scene_id"]].append(row)
    per_scene = []
    for (method, dataset, scene), selected in sorted(groups.items()):
        item = {"method": method, "dataset": dataset, "scene_id": scene, "checkpoint_count": len(selected),
                "events": ";".join(e for e in EVENTS if e in {r["coverage_event_id"] for r in selected})}
        for metric in METRICS:
            values = [100 * float(r[metric]) for r in selected]
            mean, sd = statistics.fmean(values), statistics.pstdev(values)
            item.update({metric + "_mean_percent": mean, metric + "_sd_pp": sd,
                         metric + "_cv_percent": 100 * sd / mean if mean else None,
                         metric + "_worst_percent": min(values)})
        per_scene.append(item)
    stability = []
    for method in METHODS:
        selected = [r for r in per_scene if r["method"] == method]
        for metric in METRICS:
            cvs = [r[metric + "_cv_percent"] for r in selected if r[metric + "_cv_percent"] is not None]
            stability.append({"method": method, "metric": metric, "scene_count": len(selected),
                "mean_percent": statistics.fmean(r[metric + "_mean_percent"] for r in selected),
                "mean_within_scene_sd_pp": statistics.fmean(r[metric + "_sd_pp"] for r in selected),
                "mean_within_scene_cv_percent": statistics.fmean(cvs) if cvs else None,
                "cv_defined_scene_count": len(cvs),
                "mean_scene_worst_percent": statistics.fmean(r[metric + "_worst_percent"] for r in selected)})
    for name, table in (("per_checkpoint_metrics", rows), ("overall_metrics", all_table),
                        ("progress_metrics", progress), ("before_after_and_main", comparison),
                        ("per_scene_stability", per_scene), ("stability_summary", stability)):
        write_csv(out / (name + ".csv"), table)
    lines = ["# DUDE 修正接入后的重新测评", "",
        "原版 C++ DUDE；不使用 Python 替代算法。输入为同一批已保存的原始 Vertical-Free 地图，算法输入不裁剪到 GT/reference mask。",
        "", "68 个独立测试场景（InteriorAgent 10，GRScene 58），每版 462 个进度点。排除训练与验证场景；不补缺失进度。",
        "", "修复 tagged image 上下翻转：先翻回源地图坐标，再裁剪评测域。凹度阈值 2.5 m 原样传入原版程序，不取整。保留上游形态学预处理。",
        "", "在线版：每个场景新建状态，按保存进度顺序增量更新；离线版：每个进度点重置原版程序。在线版是保存快照的增量回放，不是全传感器帧的原论文在线实验。",
        "", "## 同一组 462 个进度点对照", "", "| 方法 | P (%) | R (%) | F1 (%) | room-mIoU (%) |", "|---|---:|---:|---:|---:|"]
    for row in comparison:
        lines.append("| " + row["method"] + " | " + " | ".join(f"{row[k + '_percent']:.3f}" for k in METRICS) + " |")
    lines.extend(["", "OLD_INVALID_COORDINATES 仅用于诊断前后差异，不能作为论文 DUDE 的正式成绩。旧结果未覆盖。VoxRoom 对照为现有论文表使用的 epoch 14，非本次重新推理。",
        "", "## 逐进度结果", "", "| 方法 | 进度 | 点数 | P (%) | R (%) | F1 (%) | room-mIoU (%) |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in progress:
        lines.append(f"| {row['method']} | {row['scope']} | {row['checkpoint_count']} | " + " | ".join(f"{row[k + '_percent']:.3f}" for k in METRICS) + " |")
    lines.extend(["", "## 统计口径与复核", "",
        "P/R 是房间区域匹配指标；每个进度先算 F1，再对全部有效进度点加权合并。room-mIoU 为 Hungarian 一对一匹配的 IoU 总和除以 GT 房间数。小于 0.5 m² 的房间按原评测协议过滤。",
        "", "稳定性在同一场景的实际进度点内计算 mean、总体 SD、CV=SD/mean 和 worst=min，再对场景等权平均。零均值的 CV 标为未定义，并报告有效场景数。",
        "", "每一个预测均检查输出翻转、2.5 m 参数和评测裁剪一致性；主方法/旧 DUDE 对比逐项核对 source snapshot、step、GT 房间数及评测面积一致。原版源码 commit、编译兼容补丁、二进制 SHA256 及运行代码哈希记录在各数据集 input_audit.json。"])
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (root / "completed.json").write_text(json.dumps({"status": "complete", "tables": str(out), "prediction_count": len(rows), "time": time.time()}, indent=2) + "\n")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
