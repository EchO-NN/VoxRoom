#!/usr/bin/env python3
"""Aggregate only complete, held-out official OccuSG saved-voxel replays."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import time

METRICS = ("precision", "recall", "f1", "miou_room")
EVENTS = ("milestone_020", "milestone_040", "milestone_060", "milestone_070", "milestone_080", "milestone_090", "final")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return row["dataset"], row["episode_uid"], row["coverage_event_id"]


def aggregate(rows, method, scope):
    return {"method": method, "scope": scope, "scene_count": len({(r["dataset"], r["scene_id"]) for r in rows}),
            "checkpoint_count": len(rows), **{m+"_percent": statistics.fmean(100*float(r[m]) for r in rows) for m in METRICS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    while True:
        states = {d: json.loads((args.root / d / "status.json").read_text()) if (args.root / d / "status.json").exists() else {} for d in ("interioragent", "grscene")}
        if any(s.get("status") == "failed" for s in states.values()):
            raise RuntimeError(states)
        if all(s.get("status") == "complete" for s in states.values()):
            break
        if not args.wait:
            raise RuntimeError(f"Not complete: {states}")
        time.sleep(30)
    native, main_rows = [], []
    for dataset in states:
        current = read(args.root / dataset / "per_checkpoint_metrics.csv")
        provenance = json.loads((args.root / dataset / "inputs/provenance.json").read_text())
        assert len(current) == provenance["checkpoints"]
        assert len({r["scene_id"] for r in current}) == provenance["scenes"]
        assert not ({r["scene_id"] for r in current} & set(provenance["excluded_training_and_validation_scene_ids"]))
        lookup = {key(r): r for r in current}
        assert len(lookup) == len(current)
        for row in current:
            assert all(math.isfinite(float(row[m])) and 0 <= float(row[m]) <= 1 for m in METRICS)
            p, r = float(row["precision"]), float(row["recall"])
            assert math.isclose(float(row["f1"]), 2*p*r/(p+r) if p+r else 0, abs_tol=1e-12)
        comparison = Path("/media/echo/data/voxroom_ablation_20260828/evaluation") / f"{dataset}_vertical_full_retrain_f1_preview_epoch14/metrics/per_checkpoint_metrics.csv"
        matched = []
        for row in read(comparison):
            if key(row) not in lookup:
                continue
            ref = lookup[key(row)]
            assert Path(row["source_snapshot_path"]).resolve() == Path(ref["source_snapshot_path"]).resolve()
            for field in ("step", "n_gt", "metric_domain_pixels"):
                assert int(row[field]) == int(ref[field]), (field, row, ref)
            matched.append(row)
        assert len(matched) == len(current)
        native.extend(current)
        main_rows.extend(matched)
    assert len(native) == 462
    assert len({(r["dataset"], r["scene_id"]) for r in native}) == 68
    tables = args.root / "tables"
    tables.mkdir(exist_ok=True)
    write(tables / "occusg_per_checkpoint.csv", native)
    all_rows, progress, per_scene, stability = [], [], [], []
    for method, records in (("OccuSG_official_saved_voxel_replay", native), ("VoxRoom_paper_main_epoch14", main_rows)):
        all_rows.append(aggregate(records, method, "all_available_checkpoints_weighted"))
        all_rows.append(aggregate([r for r in records if r["coverage_event_id"] == "final"], method, "final_only"))
        for event in EVENTS:
            selected = [r for r in records if r["coverage_event_id"] == event]
            progress.append(aggregate(selected, method, event))
        groups = defaultdict(list)
        for row in records:
            groups[row["dataset"], row["scene_id"]].append(row)
        this_scene = []
        for (dataset, scene), group in groups.items():
            item = {"method": method, "dataset": dataset, "scene_id": scene, "checkpoint_count": len(group)}
            for metric in METRICS:
                values = [100*float(r[metric]) for r in group]
                mean, sd = statistics.fmean(values), statistics.pstdev(values)
                item.update({metric+"_mean_percent": mean, metric+"_sd_pp": sd,
                             metric+"_cv": sd/mean if mean else None, metric+"_worst_percent": min(values)})
            this_scene.append(item)
        per_scene.extend(this_scene)
        summary = {"method": method, "scene_count": len(this_scene)}
        for metric in METRICS:
            for suffix in ("_mean_percent", "_sd_pp", "_cv", "_worst_percent"):
                values = [r[metric+suffix] for r in this_scene if r[metric+suffix] is not None]
                summary[metric+suffix] = statistics.fmean(values) if values else None
            summary[metric+"_cv_valid_scenes"] = sum(r[metric+"_cv"] is not None for r in this_scene)
        stability.append(summary)
    write(tables / "aggregate.csv", all_rows)
    write(tables / "progress_metrics.csv", progress)
    write(tables / "per_scene_stability.csv", per_scene)
    write(tables / "stability_summary.csv", stability)
    lines = ["# OccuSG 官方参数：已存体素快照评测", "", "排除训练和验证场景：68 个场景，462 个可用进度点。InteriorAgent 与 GRScene 按实际进度点数量合并；缺失进度不补齐。", "",
             "这不是论文原数据集、原始 RGB-D 重建或完整连续流复现。使用固定官方提交的投影、增量 DUDE、区域跟踪及房间实体化；0.05 m 体素、0.3 m 连续净空、1.5 分解阈值。输入为我们保存的完整三维占用状态和稀疏机器人位姿，无目标检测流。具体适配边界见 inputs/provenance.json。", "",
             "P/R 为已有房间重叠口径；F1 先逐进度计算再平均；mIoU 为 Hungarian 匹配 IoU 总和除以 GT 房间数，并非原论文仅匹配成功房间的 mIoU。0.5 m² 过滤只属于统一评测协议。", "",
             "| 方法 | 范围 | 点数 | P (%) | R (%) | F1 (%) | mIoU (%) |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in all_rows:
        lines.append(f"| {row['method']} | {row['scope']} | {row['checkpoint_count']} | " + " | ".join(f"{row[m+'_percent']:.3f}" for m in METRICS) + " |")
    lines += ["", "进度明细：progress_metrics.csv；逐场景逐进度：occusg_per_checkpoint.csv。", "稳定性：先在同一场景的可用进度上计算 mean、总体 SD（ddof=0）、CV、worst，再对场景等权平均。平均为 0 的 CV 留空，不当成 0。", "",
              "官方源码：https://github.com/crcz25/OccuSG ，论文：https://arxiv.org/abs/2606.13727 。对照主方法固定为已有论文表的 epoch14 权重，不声称是后来训练的权重。"]
    (tables / "report.md").write_text("\n".join(lines) + "\n")
    (args.root / "completed.json").write_text(json.dumps({"status": "complete", "scenes": 68, "checkpoints": 462, "time": time.time()}, indent=2))
    print("All held-out tables complete", flush=True)


if __name__ == "__main__":
    main()
