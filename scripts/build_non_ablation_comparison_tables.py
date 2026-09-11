#!/usr/bin/env python3
"""Build paper tables for VoxRoom and non-ablation comparison methods only."""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import build_paper_ready_result_tables as source


ROOT = Path("/home/joey/Active_room_segmentation")
OUT = ROOT / "results" / "non_ablation_comparisons_20260831"
EVENTS = source.CHECKPOINTS[1:]
MIN_POINTS = 6
METHODS = [
    source.MAIN_METHOD,
    *(source.METHOD_LABELS[key] for key in source.BASELINE_METHODS),
]
METRICS = source.METRICS
METRIC_SHORT = {
    "precision_percent": "p",
    "recall_percent": "r",
    "f1_percent": "f1",
    "miou_room_percent": "miou",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    observations = (
        source.load_latest_main_observations()
        + source.load_baseline_observations()
    )

    # Exactly one result per method / dataset / scene / progress event.
    indexed: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in observations:
        key = (
            str(row["method"]),
            str(row["dataset"]),
            str(row["scene_id"]),
            str(row["coverage_event_id"]),
        )
        if key in indexed:
            raise RuntimeError(f"duplicate observation: {key}")
        indexed[key] = row

    by_method_scene: dict[
        str, dict[tuple[str, str], list[dict[str, object]]]
    ] = {method: defaultdict(list) for method in METHODS}
    for row in observations:
        by_method_scene[str(row["method"])][
            (str(row["dataset"]), str(row["scene_id"]))
        ].append(row)

    # A scene is eligible when every compared method has at least six matching
    # progress events. Event sets must be identical across methods so missing
    # checkpoints never favor one method.
    all_scenes = set.intersection(
        *(set(by_method_scene[method]) for method in METHODS)
    )
    eligible: list[tuple[str, str]] = []
    for identity in sorted(all_scenes):
        event_sets = [
            {str(row["coverage_event_id"]) for row in by_method_scene[method][identity]}
            for method in METHODS
        ]
        if len(event_sets[0]) >= MIN_POINTS and all(
            events == event_sets[0] for events in event_sets[1:]
        ):
            eligible.append(identity)
    if not eligible:
        raise RuntimeError("no common eligible held-out scenes")

    progress_rows: list[dict[str, object]] = []
    per_scene_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for method in METHODS:
        method_rows = [
            row
            for identity in eligible
            for row in by_method_scene[method][identity]
        ]
        for event in EVENTS:
            selected = [
                row for row in method_rows if row["coverage_event_id"] == event
            ]
            progress_rows.append(
                {
                    "scope": "excluding_training_and_validation_scenes",
                    "method": method,
                    "method_key": selected[0]["method_key"],
                    "checkpoint": event,
                    "checkpoint_label": source.CHECKPOINT_LABEL[event],
                    "scene_count": len(selected),
                    **{
                        metric: statistics.fmean(float(row[metric]) for row in selected)
                        for metric in METRICS
                    },
                }
            )

        method_scene_stats: list[dict[str, object]] = []
        for dataset, scene_id in eligible:
            rows = sorted(
                by_method_scene[method][(dataset, scene_id)],
                key=lambda row: EVENTS.index(str(row["coverage_event_id"])),
            )
            item: dict[str, object] = {
                "scope": "excluding_training_and_validation_scenes",
                "method": method,
                "method_key": rows[0]["method_key"],
                "dataset": dataset,
                "scene_id": scene_id,
                "episode_uid": rows[0].get("episode_uid", ""),
                "progress_point_count": len(rows),
                "progress_points": ";".join(str(row["coverage_event_id"]) for row in rows),
            }
            for metric in METRICS:
                short = METRIC_SHORT[metric]
                values = [float(row[metric]) for row in rows]
                mean = statistics.fmean(values)
                sd = statistics.pstdev(values)
                item[f"{short}_mean_percent"] = mean
                item[f"{short}_sd_percent"] = sd
                item[f"{short}_cv_percent"] = 100.0 * sd / mean if mean else 0.0
                item[f"{short}_worst_percent"] = min(values)
            per_scene_rows.append(item)
            method_scene_stats.append(item)

        summary: dict[str, object] = {
            "scope": "excluding_training_and_validation_scenes",
            "method": method,
            "method_key": method_scene_stats[0]["method_key"],
            "scene_count": len(method_scene_stats),
            "six_point_scene_count": sum(
                int(row["progress_point_count"] == 6) for row in method_scene_stats
            ),
            "seven_point_scene_count": sum(
                int(row["progress_point_count"] == 7) for row in method_scene_stats
            ),
            "evaluation_count": sum(
                int(row["progress_point_count"]) for row in method_scene_stats
            ),
        }
        for metric in METRICS:
            short = METRIC_SHORT[metric]
            for statistic_name in ("mean", "sd", "cv", "worst"):
                column = f"{short}_{statistic_name}_percent"
                summary[column] = statistics.fmean(
                    float(row[column]) for row in method_scene_stats
                )
        summary_rows.append(summary)

    distribution = Counter(
        int(row["progress_point_count"])
        for row in per_scene_rows
        if row["method"] == METHODS[0]
    )
    lines = [
        "# 非消融对比实验汇总（独立测试场景）",
        "",
        "本文件只包含完整 VoxRoom 主方法和八个对照方法，不包含任何消融实验。训练场景以及用于 checkpoint 选择的验证场景均已排除。",
        "",
        f"稳定性统计先在每个场景内部、按实际存在的探索进度计算 Mean、总体 SD（ddof=0）、CV=SD/Mean×100% 和 Worst，再对场景级统计量取平均。允许缺少一个进度点：共 {len(eligible)} 个场景，其中 {distribution[7]} 个有 7 点、{distribution[6]} 个有 6 点；每种方法共 {sum(k*v for k,v in distribution.items())} 个评测点。所有方法在同一场景使用完全相同的进度点。",
        "",
        "## 总体性能与场景内稳定性",
        "",
        "| 方法 | 场景 | 点数 | F1 Mean | F1 SD | F1 CV | F1 Avg.Worst | mIoU Mean | mIoU SD | mIoU CV | mIoU Avg.Worst |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['scene_count']} | {row['evaluation_count']} | "
            f"{f(float(row['f1_mean_percent']))} | {f(float(row['f1_sd_percent']))} | "
            f"{f(float(row['f1_cv_percent']))}% | {f(float(row['f1_worst_percent']))} | "
            f"{f(float(row['miou_mean_percent']))} | {f(float(row['miou_sd_percent']))} | "
            f"{f(float(row['miou_cv_percent']))}% | {f(float(row['miou_worst_percent']))} |"
        )
    lines.extend(
        [
            "",
            "## P 与 R",
            "",
            "| 方法 | P Mean | P SD | P CV | P Avg.Worst | R Mean | R SD | R CV | R Avg.Worst |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {f(float(row['p_mean_percent']))} | "
            f"{f(float(row['p_sd_percent']))} | {f(float(row['p_cv_percent']))}% | "
            f"{f(float(row['p_worst_percent']))} | {f(float(row['r_mean_percent']))} | "
            f"{f(float(row['r_sd_percent']))} | {f(float(row['r_cv_percent']))}% | "
            f"{f(float(row['r_worst_percent']))} |"
        )
    lines.extend(
        [
            "",
            "## 文件说明",
            "",
            "- `comparison_stability_summary.csv`：每种方法一行的论文汇总表。",
            "- `comparison_progress_metrics.csv`：20% 至 Final 的逐进度 P/R/F1/room-mIoU。",
            "- `comparison_per_scene_stability.csv`：每个场景先计算的 Mean/SD/CV/Worst，便于逐项复核。",
        ]
    )

    write_csv(OUT / "comparison_stability_summary.csv", summary_rows)
    write_csv(OUT / "comparison_progress_metrics.csv", progress_rows)
    write_csv(OUT / "comparison_per_scene_stability.csv", per_scene_rows)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output={OUT}")
    print(f"eligible_scenes={len(eligible)} distribution={dict(sorted(distribution.items()))}")


if __name__ == "__main__":
    main()
