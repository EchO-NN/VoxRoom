#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/joey/Active_room_segmentation")
OUT = ROOT / "results" / "paper_ready_tables_20260831"
REMOTE = "echo@10.42.0.1"
REMOTE_ROOT = "/media/echo/data/voxroom_ablation_20260828"

DATASETS = ("interioragent", "grscene")
SCOPES = ("all_approved", "excluding_training_and_validation_scenes")
CHECKPOINTS = (
    "all_checkpoints",
    "milestone_020",
    "milestone_040",
    "milestone_060",
    "milestone_070",
    "milestone_080",
    "milestone_090",
    "final",
)
CHECKPOINT_LABEL = {
    "all_checkpoints": "全过程平均",
    "milestone_020": "20%",
    "milestone_040": "40%",
    "milestone_060": "60%",
    "milestone_070": "70%",
    "milestone_080": "80%",
    "milestone_090": "90%",
    "final": "Final",
}
METRICS = (
    "precision_percent",
    "recall_percent",
    "f1_percent",
    "miou_room_percent",
)

MAIN_METHOD = "VoxRoom（最新 Vertical-Free，F1 选模 epoch 14）"
BASELINE_METHODS = (
    "tvars_original",
    "dude_incremental",
    "gomez_incremental",
    "dude_offline",
    "rose2",
    "morphological",
    "distance_transform",
    "voronoi",
)
METHOD_LABELS = {
    "tvars_original": "TVARS",
    "dude_incremental": "DUDE（在线）",
    "gomez_incremental": "Gomez（在线）",
    "dude_offline": "DUDE（离线）",
    "rose2": "ROSE2",
    "morphological": "Morphological",
    "distance_transform": "Distance Transform",
    "voronoi": "Voronoi",
}

ABLATION_ORDER = (
    "no_tvars_raw_seed",
    "no_voxel_raw_seed",
    "no_neural_filter",
    "nav_no_clearance_full",
    "vertical_2d_only",
    "vertical_3d_only",
    "all_vertical_free_cells",
)
ABLATION_LABELS = {
    "no_tvars_raw_seed": "1 不使用 TVARS raw seed",
    "no_voxel_raw_seed": "2 不使用 voxel raw seed",
    "no_neural_filter": "3 不使用神经网络（raw seed 直接分割）",
    "nav_no_clearance_full": "4 神经网络改用无-clearance Nav-Free",
    "vertical_2d_only": "5 仅 41x41 二维分支",
    "vertical_3d_only": "6 仅 19x19 三维分支",
    "all_vertical_free_cells": "7 无 raw-seed 预筛选（全图 free cell）",
}


def remote_text(path: str) -> str:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", REMOTE, "cat", path],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return proc.stdout


def mean(rows: list[dict[str, str]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def load_latest_main_rows() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for dataset in DATASETS:
        base = (
            f"{REMOTE_ROOT}/evaluation/"
            f"{dataset}_vertical_full_retrain_f1_preview_epoch14/metrics"
        )
        summary = json.loads(remote_text(f"{base}/summary.json"))
        raw_rows = list(csv.DictReader(io.StringIO(remote_text(f"{base}/per_checkpoint_metrics.csv"))))
        excluded = set(summary["excluded_scene_ids"])

        for scope in SCOPES:
            scoped = raw_rows
            if scope == "excluding_training_and_validation_scenes":
                scoped = [row for row in raw_rows if row["scene_id"] not in excluded]
            for checkpoint in CHECKPOINTS:
                selected = scoped
                if checkpoint != "all_checkpoints":
                    selected = [row for row in scoped if row["coverage_event_id"] == checkpoint]
                if not selected:
                    continue
                result.append(
                    {
                        "dataset": dataset,
                        "scope": scope,
                        "checkpoint": checkpoint,
                        "checkpoint_label": CHECKPOINT_LABEL[checkpoint],
                        "method": MAIN_METHOD,
                        "method_key": "voxroom_latest_vertical_f1_epoch14",
                        "input_map": "raw_vertical_free",
                        "process_count": len({row["scene_id"] for row in selected}),
                        "evaluation_count": len(selected),
                        **{metric: mean(selected, metric) for metric in METRICS},
                        "result_status": "completed_preview_evaluation_training_paused",
                        "source": f"remote:{base}",
                    }
                )
    return result


def load_latest_main_observations() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for dataset in DATASETS:
        base = (
            f"{REMOTE_ROOT}/evaluation/"
            f"{dataset}_vertical_full_retrain_f1_preview_epoch14/metrics"
        )
        summary = json.loads(remote_text(f"{base}/summary.json"))
        raw_rows = list(
            csv.DictReader(
                io.StringIO(remote_text(f"{base}/per_checkpoint_metrics.csv"))
            )
        )
        excluded = set(summary["excluded_scene_ids"])
        for row in raw_rows:
            if row["scene_id"] in excluded:
                continue
            result.append(
                {
                    "dataset": dataset,
                    "scene_id": row["scene_id"],
                    "episode_uid": row.get("episode_uid", ""),
                    "coverage_event_id": row["coverage_event_id"],
                    "method": MAIN_METHOD,
                    "method_key": "voxroom_latest_vertical_f1_epoch14",
                    **{metric: float(row[metric]) for metric in METRICS},
                }
            )
    return result


def load_baseline_rows() -> list[dict[str, object]]:
    sources = {
        "interioragent": ROOT / "results/interioragent_paper_baselines_raw_vertical_20260828/aggregate.csv",
        "grscene": ROOT / "results/grscene_paper_segmentation_baselines_raw_vertical_20260828/aggregate.csv",
    }
    gomez_sources = {
        "interioragent": ROOT / "results/interioragent_gomez_saved_tvars_raw_voxel_hough_20260831/aggregate.csv",
        "grscene": ROOT / "results/grscene_gomez_saved_tvars_raw_voxel_hough_20260831/aggregate.csv",
    }
    scope_map = {
        "all": "all_approved",
        "excluding_training": "excluding_training_and_validation_scenes",
    }
    result: list[dict[str, object]] = []
    for dataset, path in sources.items():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                method_key = row["method"]
                if method_key not in BASELINE_METHODS:
                    continue
                if method_key == "gomez_incremental":
                    continue
                checkpoint = row["checkpoint"]
                result.append(
                    {
                        "dataset": dataset,
                        "scope": scope_map[row["scope"]],
                        "checkpoint": checkpoint,
                        "checkpoint_label": CHECKPOINT_LABEL[checkpoint],
                        "method": METHOD_LABELS[method_key],
                        "method_key": method_key,
                        "input_map": "raw_vertical_free",
                        "process_count": int(row["process_count"]),
                        "evaluation_count": int(row["evaluation_count"]),
                        **{metric: float(row[metric]) for metric in METRICS},
                        "result_status": "completed_final",
                        "source": str(path.relative_to(ROOT)),
                    }
                )
    for dataset, path in gomez_sources.items():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["method"] != "gomez_incremental":
                    continue
                checkpoint = row["checkpoint"]
                result.append(
                    {
                        "dataset": dataset,
                        "scope": scope_map[row["scope"]],
                        "checkpoint": checkpoint,
                        "checkpoint_label": CHECKPOINT_LABEL[checkpoint],
                        "method": METHOD_LABELS["gomez_incremental"],
                        "method_key": "gomez_incremental",
                        "input_map": "raw_vertical_free_with_saved_tvars_raw_seed_and_voxel_hough_confirmation",
                        "process_count": int(row["process_count"]),
                        "evaluation_count": int(row["evaluation_count"]),
                        **{metric: float(row[metric]) for metric in METRICS},
                        "result_status": "completed_final_saved_tvars_raw_seed_voxel_hough",
                        "source": str(path.relative_to(ROOT)),
                    }
                )
    return result


def load_baseline_observations() -> list[dict[str, object]]:
    sources = {
        "interioragent": ROOT
        / "results/interioragent_paper_baselines_raw_vertical_20260828/per_checkpoint_metrics_excluding_training.csv",
        "grscene": ROOT
        / "results/grscene_paper_segmentation_baselines_raw_vertical_20260828/per_checkpoint_metrics_excluding_training.csv",
    }
    gomez_sources = {
        "interioragent": ROOT
        / "results/interioragent_gomez_saved_tvars_raw_voxel_hough_20260831/per_checkpoint_metrics_excluding_training.csv",
        "grscene": ROOT
        / "results/grscene_gomez_saved_tvars_raw_voxel_hough_20260831/per_checkpoint_metrics_excluding_training.csv",
    }
    result: list[dict[str, object]] = []
    for dataset, path in sources.items():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                method_key = row["method"]
                if method_key not in BASELINE_METHODS or method_key == "gomez_incremental":
                    continue
                result.append(
                    {
                        "dataset": dataset,
                        "scene_id": row["scene_id"],
                        "episode_uid": row["episode_uid"],
                        "coverage_event_id": row["coverage_event_id"],
                        "method": METHOD_LABELS[method_key],
                        "method_key": method_key,
                        **{metric: float(row[metric]) for metric in METRICS},
                    }
                )
    for dataset, path in gomez_sources.items():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["method"] != "gomez_incremental":
                    continue
                result.append(
                    {
                        "dataset": dataset,
                        "scene_id": row["scene_id"],
                        "episode_uid": row["episode_uid"],
                        "coverage_event_id": row["coverage_event_id"],
                        "method": METHOD_LABELS["gomez_incremental"],
                        "method_key": "gomez_incremental",
                        **{metric: float(row[metric]) for metric in METRICS},
                    }
                )
    return result


def calculate_stability_statistics(
    observations: list[dict[str, object]],
) -> list[dict[str, object]]:
    methods = [MAIN_METHOD, *(METHOD_LABELS[key] for key in BASELINE_METHODS)]
    required_events = set(CHECKPOINTS[1:])
    result: list[dict[str, object]] = []
    for method in methods:
        selected = [row for row in observations if row["method"] == method]
        if not selected:
            raise RuntimeError(f"no held-out checkpoint observations for {method}")
        by_scene: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in selected:
            by_scene[(str(row["dataset"]), str(row["scene_id"]))].append(row)
        complete_scenes = {
            identity: rows
            for identity, rows in by_scene.items()
            if {str(row["coverage_event_id"]) for row in rows} == required_events
            and len(rows) == len(required_events)
        }
        if not complete_scenes:
            raise RuntimeError(f"no complete seven-checkpoint scenes for {method}")
        for metric in METRICS:
            scene_statistics: list[dict[str, object]] = []
            for identity, scene_rows in complete_scenes.items():
                ordered = sorted(
                    scene_rows,
                    key=lambda row: CHECKPOINTS.index(
                        str(row["coverage_event_id"])
                    ),
                )
                values = [float(row[metric]) for row in ordered]
                scene_mean = statistics.fmean(values)
                scene_sd = statistics.pstdev(values)
                worst_index = min(range(len(values)), key=values.__getitem__)
                scene_statistics.append(
                    {
                        "identity": identity,
                        "mean": scene_mean,
                        "sd": scene_sd,
                        "cv_percent": (
                            100.0 * scene_sd / scene_mean
                            if scene_mean != 0.0
                            else 0.0
                        ),
                        "worst": values[worst_index],
                        "worst_row": ordered[worst_index],
                    }
                )
            absolute_worst = min(
                scene_statistics,
                key=lambda item: float(item["worst"]),
            )
            absolute_worst_row = dict(absolute_worst["worst_row"])
            result.append(
                {
                    "scope": "excluding_training_and_validation_scenes",
                    "method": method,
                    "method_key": selected[0]["method_key"],
                    "metric": metric,
                    "scene_count": len(scene_statistics),
                    "checkpoints_per_scene": len(required_events),
                    "mean_percent": statistics.fmean(
                        float(item["mean"]) for item in scene_statistics
                    ),
                    "sd_percent": statistics.fmean(
                        float(item["sd"]) for item in scene_statistics
                    ),
                    "cv_percent": statistics.fmean(
                        float(item["cv_percent"]) for item in scene_statistics
                    ),
                    "worst_percent": statistics.fmean(
                        float(item["worst"]) for item in scene_statistics
                    ),
                    "absolute_worst_percent": float(absolute_worst["worst"]),
                    "absolute_worst_dataset": absolute_worst_row["dataset"],
                    "absolute_worst_scene_id": absolute_worst_row["scene_id"],
                    "absolute_worst_episode_uid": absolute_worst_row["episode_uid"],
                    "absolute_worst_coverage_event_id": absolute_worst_row[
                        "coverage_event_id"
                    ],
                    "mean_definition": "mean_of_per_scene_means_across_7_progress_points",
                    "sd_definition": "mean_of_per_scene_population_sd_ddof_0_across_7_progress_points",
                    "cv_definition": "mean_of_per_scene_100_x_population_sd_divided_by_scene_mean",
                    "worst_definition": "mean_of_per_scene_minimum_across_7_progress_points",
                }
            )
    return result


def load_ablation_rows() -> list[dict[str, object]]:
    path = ROOT / "results/experiment_archive_20260829/catalog/voxroom_ablations_canonical.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    result: list[dict[str, object]] = []
    for scope in SCOPES:
        for variant in ABLATION_ORDER:
            matches = [
                row
                for row in source_rows
                if row["method_or_variant"] == variant
                and row["scope"] == scope
                and row["dataset"] in DATASETS
                and row["validity"] == "valid"
            ]
            if matches:
                for row in matches:
                    result.append(
                        {
                            "scope": scope,
                            "variant": variant,
                            "ablation": ABLATION_LABELS[variant],
                            "dataset": row["dataset"],
                            "process_count": int(row["process_count"]),
                            "evaluation_count": int(row["evaluation_count"]),
                            "precision_percent": float(row["precision_percent"]),
                            "recall_percent": float(row["recall_percent"]),
                            "f1_percent": float(row["f1_percent"]),
                            "miou_room_percent": float(row["miou_room_percent"]),
                            "result_status": row["result_status"],
                            "source": row["source_file"],
                        }
                    )
                result.append(
                    {
                        "scope": scope,
                        "variant": variant,
                        "ablation": ABLATION_LABELS[variant],
                        "dataset": "combined_weighted",
                        "process_count": sum(int(row["process_count"]) for row in matches),
                        "evaluation_count": sum(int(row["evaluation_count"]) for row in matches),
                        **{
                            metric: sum(
                                float(row[metric]) * int(row["evaluation_count"])
                                for row in matches
                            )
                            / sum(int(row["evaluation_count"]) for row in matches)
                            for metric in METRICS
                        },
                        "result_status": matches[0]["result_status"],
                        "source": "weighted by checkpoint evaluation count",
                    }
                )
            else:
                for dataset in (*DATASETS, "combined_weighted"):
                    result.append(
                        {
                            "scope": scope,
                            "variant": variant,
                            "ablation": ABLATION_LABELS[variant],
                            "dataset": dataset,
                            "process_count": "",
                            "evaluation_count": "",
                            **{metric: "" for metric in METRICS},
                            "result_status": "pending_training_not_evaluated",
                            "source": "",
                        }
                    )
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if value == "" or value is None:
        return "-"
    return f"{float(value):.3f}"


def combine_weighted(full_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    primary_scope = "excluding_training_and_validation_scenes"
    result: list[dict[str, object]] = []
    methods = [MAIN_METHOD, *(METHOD_LABELS[key] for key in BASELINE_METHODS)]
    for method in methods:
        for checkpoint in CHECKPOINTS:
            matches = [
                row
                for row in full_rows
                if row["scope"] == primary_scope
                and row["method"] == method
                and row["checkpoint"] == checkpoint
            ]
            if len(matches) != 2:
                raise RuntimeError(f"expected two dataset rows: {method} {checkpoint}")
            total = sum(int(row["evaluation_count"]) for row in matches)
            result.append(
                {
                    "scope": primary_scope,
                    "checkpoint": checkpoint,
                    "checkpoint_label": CHECKPOINT_LABEL[checkpoint],
                    "method": method,
                    "method_key": matches[0]["method_key"],
                    "input_map": matches[0]["input_map"],
                    "process_count": sum(int(row["process_count"]) for row in matches),
                    "evaluation_count": total,
                    **{
                        metric: sum(
                            float(row[metric]) * int(row["evaluation_count"])
                            for row in matches
                        )
                        / total
                        for metric in METRICS
                    },
                    "result_status": matches[0]["result_status"],
                    "source": "InteriorAgent and GRScene weighted by checkpoint evaluation count",
                }
            )
    return result


def render_report(combined_rows: list[dict[str, object]], ablation_rows: list[dict[str, object]]) -> str:
    primary_scope = "excluding_training_and_validation_scenes"
    methods = [MAIN_METHOD, *(METHOD_LABELS[key] for key in BASELINE_METHODS)]
    lines = [
        "# 论文用结果初步汇总",
        "",
        "生成日期：2026-08-31。数值均为百分数。",
        "",
        "## 口径与当前状态",
        "",
        "- 完全排除 VoxRoom 训练场景与用于 checkpoint 选择的验证场景，只统计未参与训练和选模的测试场景。",
        "- InteriorAgent 与 GRScene 不分开报告，按有效 checkpoint 数加权合并；全过程平均共 462 个 checkpoint（InteriorAgent 64、GRScene 398）。",
        "- 主方法：Vertical-Free 41x41 二维上下文 + 19x19 三维分支，VoxRoom/TVARS raw-seed 并集，阈值 0.5。使用当前 F1 最佳 epoch 14；全流程回放已完成，但训练本身暂停在 epoch 17，因此属于最新可用预览而非最终收敛模型。",
        "- 对照方法：均使用未经预拆分的原始 Vertical-Free map；TVARS 与论文中的七种传统分割基线均已完成全流程评测。",
        "- 全流程节点：20%、40%、60%、70%、80%、90% 和 Final。`全过程平均` 是所有已记录节点的逐 checkpoint 宏平均，不是 Final。",
        "- P/R 是论文的区域最佳重叠指标；F1 是每个 checkpoint 的 P/R 调和平均后再宏平均；mIoU 是匈牙利一对一房间匹配指标。小于 0.5 m² 的房间被忽略。",
        "",
        "## 主方法与对照方法：合并后的全过程",
    ]
    for method in methods:
        lines.extend(
            [
                "",
                f"### {method}",
                "",
                "| 探索进度 | checkpoint 数 | P | R | F1 | room-mIoU |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for checkpoint in CHECKPOINTS:
            row = next(
                row for row in combined_rows
                if row["method"] == method and row["checkpoint"] == checkpoint
            )
            lines.append(
                f"| {row['checkpoint_label']} | {row['evaluation_count']} | "
                f"{fmt(row['precision_percent'])} | "
                f"{fmt(row['recall_percent'])} | {fmt(row['f1_percent'])} | "
                f"{fmt(row['miou_room_percent'])} |"
            )

    weighted_ablation_rows = [
        row
        for row in ablation_rows
        if row["scope"] == primary_scope and row["dataset"] == "combined_weighted"
    ]
    main_average = next(
        row for row in combined_rows
        if row["method"] == MAIN_METHOD and row["checkpoint"] == "all_checkpoints"
    )
    lines.extend(
        [
            "",
            "## 消融实验：合并后的全过程加权平均",
            "",
            "| 方法/消融 | checkpoint 数 | P | R | F1 | room-mIoU | 状态 |",
            "|---|---:|---:|---:|---:|---:|---|",
            f"| 完整主方法（基准） | {main_average['evaluation_count']} | "
            f"{fmt(main_average['precision_percent'])} | {fmt(main_average['recall_percent'])} | "
            f"{fmt(main_average['f1_percent'])} | {fmt(main_average['miou_room_percent'])} | "
            "当前 F1 最佳 epoch 14 预览 |",
        ]
    )
    for variant in ABLATION_ORDER:
        row = next(row for row in weighted_ablation_rows if row["variant"] == variant)
        lines.append(
            f"| {row['ablation']} | {row['evaluation_count'] or '-'} | "
            f"{fmt(row['precision_percent'])} | "
            f"{fmt(row['recall_percent'])} | {fmt(row['f1_percent'])} | "
            f"{fmt(row['miou_room_percent'])} | {row['result_status']} |"
        )
    lines.extend(
        [
            "",
            "注意：第 4 项正在按验证 F1 重新训练，表中暂时是此前 Accuracy 选模的当前最佳预览；第 6 项也是未完成训练的当前最佳预览；第 7 项训练暂停且尚无可用评测，不能写入论文数值表。",
            "",
            "## 文件",
            "",
            "- `main_and_baselines_full_process.csv`：两个数据集、两个 scope、全部进度点、P/R/F1/mIoU。",
            "- `main_and_baselines_combined_weighted.csv`：排除训练/验证场景后，两数据集按 checkpoint 数加权合并的论文主表。",
            "- `ablations_average.csv`：消融的单数据集结果与合并加权平均。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_stability_section(
    stability_rows: list[dict[str, object]],
) -> str:
    methods = [MAIN_METHOD, *(METHOD_LABELS[key] for key in BASELINE_METHODS)]
    metric_labels = {
        "precision_percent": "P",
        "recall_percent": "R",
        "f1_percent": "F1",
        "miou_room_percent": "room-mIoU",
    }
    lines = [
        "## 全过程稳定性统计",
        "",
        "统计范围排除训练和 checkpoint 选模场景，并只保留同时具有 20%、40%、60%、70%、80%、90% 和 Final 七个进度点的场景。先在每个场景内部计算七点 Mean、总体 SD（ddof=0）、CV=SD/Mean×100% 和 Worst，再分别对场景级统计量取平均。数值单位均为百分数。",
    ]
    for first_metric, second_metric in (
        ("precision_percent", "recall_percent"),
        ("f1_percent", "miou_room_percent"),
    ):
        first_label = metric_labels[first_metric]
        second_label = metric_labels[second_metric]
        lines.extend(
            [
                "",
                f"### {first_label} 与 {second_label}",
                "",
                f"| 方法 | {first_label} Mean | {first_label} SD | {first_label} CV | {first_label} Avg.Worst | {second_label} Mean | {second_label} SD | {second_label} CV | {second_label} Avg.Worst |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            first = next(
                row
                for row in stability_rows
                if row["method"] == method and row["metric"] == first_metric
            )
            second = next(
                row
                for row in stability_rows
                if row["method"] == method and row["metric"] == second_metric
            )
            lines.append(
                f"| {method} | {fmt(first['mean_percent'])} | "
                f"{fmt(first['sd_percent'])} | {fmt(first['cv_percent'])}% | "
                f"{fmt(first['worst_percent'])} | {fmt(second['mean_percent'])} | "
                f"{fmt(second['sd_percent'])} | {fmt(second['cv_percent'])}% | "
                f"{fmt(second['worst_percent'])} |"
            )
    lines.extend(
        [
            "",
            "各列均为场景级统计量的跨场景平均；全体场景中的绝对最差 checkpoint 及其数据集、场景、episode 和探索节点另见 `main_and_baselines_stability_statistics.csv`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full_rows = load_latest_main_rows() + load_baseline_rows()
    method_order = {
        method: index
        for index, method in enumerate([MAIN_METHOD, *(METHOD_LABELS[key] for key in BASELINE_METHODS)])
    }
    full_rows.sort(
        key=lambda row: (
            DATASETS.index(str(row["dataset"])),
            SCOPES.index(str(row["scope"])),
            method_order[str(row["method"])],
            CHECKPOINTS.index(str(row["checkpoint"])),
        )
    )
    ablation_rows = load_ablation_rows()
    combined_rows = combine_weighted(full_rows)
    observations = load_latest_main_observations() + load_baseline_observations()
    stability_rows = calculate_stability_statistics(observations)
    write_csv(OUT / "main_and_baselines_full_process.csv", full_rows)
    write_csv(OUT / "main_and_baselines_combined_weighted.csv", combined_rows)
    write_csv(OUT / "ablations_average.csv", ablation_rows)
    write_csv(OUT / "main_and_baselines_stability_statistics.csv", stability_rows)
    generated_report = render_report(combined_rows, ablation_rows)
    generated_report += "\n" + render_stability_section(stability_rows)
    # report.md contains the curated paper-writing notes and must not be
    # overwritten when only the underlying result tables are regenerated.
    (OUT / "report.generated.md").write_text(generated_report, encoding="utf-8")
    curated_report = OUT / "report.md"
    if not curated_report.exists():
        curated_report.write_text(generated_report, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
