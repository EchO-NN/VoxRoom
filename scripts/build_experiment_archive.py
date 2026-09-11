#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


METRIC_FIELDS = (
    "precision_percent",
    "recall_percent",
    "f1_percent",
    "miou_room_percent",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_scope(value: str) -> str:
    return {
        "": "all_approved",
        "all": "all_approved",
        "all_approved": "all_approved",
        "excluding_training": "excluding_training_and_validation_scenes",
        "excluding_training_and_validation_scenes": "excluding_training_and_validation_scenes",
    }.get(value, value)


def infer_dataset(path: Path, row: dict[str, str]) -> str:
    if row.get("dataset"):
        return row["dataset"]
    text = path.as_posix().lower()
    if "interioragent" in text:
        return "interioragent"
    if "grscene" in text:
        return "grscene"
    return "unknown"


def infer_input_mode(path: Path, row: dict[str, str]) -> str:
    if row.get("input_mode"):
        return row["input_mode"]
    text = path.as_posix().lower()
    if "raw_nav_no_clearance" in text:
        return "raw_nav_no_clearance"
    if "raw_vertical" in text:
        return "raw_vertical_free"
    if "interioragent_paper_baselines_20260828" in text:
        return "invalid_reference_mask_clipped_vertical"
    if "paper_baselines_20260827" in text:
        return "legacy_vertical_superseded"
    return "saved_algorithm_output"


def source_quality(path: Path) -> tuple[str, str]:
    text = path.as_posix().lower()
    if path.name == "grscene_interioragent_20260828_all_paper_baselines_aggregate.csv":
        return "invalid_mixed", "convenience copy mixes invalid leaked InteriorAgent rows and superseded GRScene rows"
    if path.name == "interioragent_20260828_all_paper_baselines_aggregate.csv":
        return "invalid", "convenience copy of the reference-mask-leaked InteriorAgent replay"
    if path.name == "grscene_20260827_all_paper_baselines_aggregate.csv":
        return "superseded", "convenience copy superseded by explicit raw Vertical-Free replay"
    if "interioragent_paper_baselines_20260828" in text and "raw_" not in text:
        return "invalid", "reference-mask leakage; retained only for audit"
    if "grscene_paper_baselines_20260827" in text:
        return "superseded", "superseded by explicit raw Vertical-Free replay"
    if path.name.endswith("_aggregate.csv"):
        return "duplicate_summary", "top-level convenience copy"
    return "valid", ""


def aggregate_rows(local_root: Path, archive_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = sorted(set(local_root.rglob("aggregate.csv")) | set(local_root.glob("*_aggregate.csv")))
    for path in candidates:
        quality, note = source_quality(path)
        with path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                if not all(raw.get(field) not in (None, "") for field in METRIC_FIELDS):
                    continue
                rows.append(
                    {
                        "experiment_id": path.parent.name if path.name == "aggregate.csv" else path.stem,
                        "source_file": str(path.relative_to(archive_root)),
                        "source_kind": "aggregate_csv",
                        "result_status": "completed",
                        "validity": quality,
                        "validity_note": note,
                        "dataset": infer_dataset(path, raw),
                        "input_mode": infer_input_mode(path, raw),
                        "scope": normalize_scope(raw.get("scope", "")),
                        "checkpoint": raw.get("checkpoint", "all_checkpoints"),
                        "method_or_variant": raw.get("method", raw.get("variant", "")),
                        "process_count": raw.get("process_count", raw.get("scene_count", "")),
                        "evaluation_count": raw.get("evaluation_count", raw.get("checkpoint_evaluation_count", "")),
                        **{field: raw[field] for field in METRIC_FIELDS},
                    }
                )
    return rows


def ablation_summary_rows(
    evaluation_root: Path,
    archive_root: Path,
    *,
    experiment_id: str,
    default_status: str,
    include_smoke: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(evaluation_root.glob("*/metrics/summary.json")):
        is_smoke = path.parent.parent.name.startswith("smoke_")
        if is_smoke and not include_smoke:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        variant = str(data.get("variant", ""))
        status = default_status
        if variant == "vertical_2d_only" and default_status == "preliminary_current_best":
            status = "completed_final_model_replay"
        if is_smoke:
            status = "smoke_test_only"
        for scope, metric in data.get("scopes", {}).items():
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "source_file": str(path.relative_to(archive_root)),
                    "source_kind": "ablation_summary_json",
                    "result_status": status,
                    "validity": "audit_only" if is_smoke else "valid",
                    "validity_note": "partial smoke test; not a full experiment" if is_smoke else "",
                    "dataset": data.get("dataset", ""),
                    "input_mode": "nav_no_clearance" if variant == "nav_no_clearance_full" else "vertical_free",
                    "scope": normalize_scope(scope),
                    "checkpoint": "all_checkpoints",
                    "method_or_variant": variant,
                    "process_count": metric.get("process_count", ""),
                    "evaluation_count": metric.get("checkpoint_evaluation_count", ""),
                    "precision_percent": metric.get("Precision_percent", ""),
                    "recall_percent": metric.get("Recall_percent", ""),
                    "f1_percent": metric.get("F1_percent", ""),
                    "miou_room_percent": metric.get("mIoU_room_percent", ""),
                }
            )
    return rows


def read_aggregate(
    path: Path,
    archive_root: Path,
    experiment_id: str,
    input_mode: str,
) -> list[dict[str, Any]]:
    dataset = "interioragent" if "interioragent" in path.as_posix() else "grscene"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("checkpoint") != "all_checkpoints":
                continue
            method = raw["method"]
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "source_file": str(path.relative_to(archive_root)),
                    "source_kind": "canonical_aggregate_csv",
                    "result_status": "completed_final",
                    "validity": "valid",
                    "validity_note": "",
                    "dataset": dataset,
                    "input_mode": input_mode,
                    "input_applies_to_method": "false" if method in {"voxroom", "tvars_original"} else "true",
                    "scope": normalize_scope(raw.get("scope", "")),
                    "checkpoint": "all_checkpoints",
                    "method_or_variant": method,
                    "process_count": raw.get("process_count", ""),
                    "evaluation_count": raw.get("evaluation_count", ""),
                    **{field: raw[field] for field in METRIC_FIELDS},
                }
            )
    return rows


def canonical_rows(archive_root: Path) -> list[dict[str, Any]]:
    local = archive_root / "local_results"
    remote = archive_root / "remote_voxroom_ablation"
    rows: list[dict[str, Any]] = []
    baseline_sources = (
        (
            local / "interioragent_paper_baselines_raw_vertical_20260828/aggregate.csv",
            "interioragent_paper_baselines_raw_vertical_20260828",
            "raw_vertical_free",
        ),
        (
            local / "interioragent_paper_baselines_raw_nav_no_clearance_20260828/aggregate.csv",
            "interioragent_paper_baselines_raw_nav_no_clearance_20260828",
            "raw_nav_no_clearance",
        ),
        (
            local / "grscene_paper_segmentation_baselines_raw_vertical_20260828/aggregate.csv",
            "grscene_paper_baselines_raw_vertical_20260828",
            "raw_vertical_free",
        ),
        (
            local / "grscene_paper_segmentation_baselines_raw_nav_no_clearance_20260828/aggregate.csv",
            "grscene_paper_baselines_raw_nav_no_clearance_20260828",
            "raw_nav_no_clearance",
        ),
    )
    for path, experiment_id, input_mode in baseline_sources:
        rows.extend(read_aggregate(path, archive_root, experiment_id, input_mode))
    rows.extend(
        ablation_summary_rows(
            remote / "evaluation",
            archive_root,
            experiment_id="voxroom_ablation_formal_completed",
            default_status="completed_final",
            include_smoke=False,
        )
    )
    rows.extend(
        ablation_summary_rows(
            remote / "previews/current_best_20260829_1610/evaluation",
            archive_root,
            experiment_id="voxroom_ablation_current_best_preview_20260829_1610",
            default_status="preliminary_current_best",
        )
    )
    return rows


def training_rows(archive_root: Path) -> list[dict[str, Any]]:
    root = archive_root / "remote_voxroom_ablation/training"
    rows: list[dict[str, Any]] = []
    for variant in (
        "nav_no_clearance_full",
        "vertical_2d_only",
        "vertical_3d_only",
        "all_vertical_free_cells",
    ):
        directory = root / variant
        summary_path = directory / "training_summary.json"
        history_path = directory / "training_history.json"
        checkpoint_path = directory / "best.pt"
        history: list[dict[str, Any]] = []
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        best = next((item for item in reversed(history) if item.get("is_best_checkpoint")), {})
        if summary_path.is_file():
            status = "completed"
        elif history:
            status = "in_progress_snapshot"
        else:
            status = "dataset_ready_training_not_started"
        row = {
            "variant": variant,
            "status_at_archive_time": status,
            "epochs_completed": len(history),
            "selected_or_current_best_epoch": best.get("epoch", ""),
            "validation_accuracy": best.get("accuracy", ""),
            "validation_precision": best.get("precision", ""),
            "validation_recall": best.get("recall", ""),
            "validation_f1": best.get("f1", ""),
            "validation_pr_auc": best.get("pr_auc", ""),
            "early_stop_no_improve": history[-1].get("early_stopping_epochs_without_improvement", "") if history else "",
            "checkpoint_path_in_archive": str(checkpoint_path.relative_to(archive_root)) if checkpoint_path.is_file() else "",
            "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path.is_file() else "",
        }
        rows.append(row)
    return rows


def experiment_catalog(archive_root: Path) -> list[dict[str, str]]:
    return [
        {"experiment_id": "grscene_voxroom_tvars_20260818", "category": "algorithm_comparison", "datasets": "grscene", "status": "completed", "validity": "valid", "canonical": "yes", "source": "local_results/grscene_metrics_20260818", "note": "VoxRoom/TVARS checkpoint metrics; includes exclusion-scope companion directory"},
        {"experiment_id": "grscene_paper_baselines_legacy_20260827", "category": "paper_baselines", "datasets": "grscene", "status": "completed", "validity": "superseded", "canonical": "no", "source": "local_results/grscene_paper_baselines_20260827", "note": "retained for audit; replaced by explicit raw Vertical-Free replay"},
        {"experiment_id": "interioragent_paper_baselines_leaked_20260828", "category": "paper_baselines", "datasets": "interioragent", "status": "completed", "validity": "invalid", "canonical": "no", "source": "local_results/interioragent_paper_baselines_20260828", "note": "reference-mask leakage; never cite these baseline rows"},
        {"experiment_id": "interioragent_paper_baselines_raw_vertical_20260828", "category": "paper_baselines", "datasets": "interioragent", "status": "completed", "validity": "valid", "canonical": "yes", "source": "local_results/interioragent_paper_baselines_raw_vertical_20260828", "note": "raw voxel_vertical_free_xy"},
        {"experiment_id": "interioragent_paper_baselines_raw_nav_no_clearance_20260828", "category": "paper_baselines", "datasets": "interioragent", "status": "completed", "validity": "valid", "canonical": "yes", "source": "local_results/interioragent_paper_baselines_raw_nav_no_clearance_20260828", "note": "raw voxel_nav_free_xy without clearance"},
        {"experiment_id": "grscene_paper_baselines_raw_vertical_20260828", "category": "paper_baselines", "datasets": "grscene", "status": "completed", "validity": "valid", "canonical": "yes", "source": "local_results/grscene_paper_segmentation_baselines_raw_vertical_20260828", "note": "raw voxel_vertical_free_xy"},
        {"experiment_id": "grscene_paper_baselines_raw_nav_no_clearance_20260828", "category": "paper_baselines", "datasets": "grscene", "status": "completed", "validity": "valid", "canonical": "yes", "source": "local_results/grscene_paper_segmentation_baselines_raw_nav_no_clearance_20260828", "note": "raw voxel_nav_free_xy without clearance"},
        {"experiment_id": "paper_baseline_input_map_comparison_20260828", "category": "input_map_comparison", "datasets": "interioragent;grscene", "status": "completed", "validity": "valid", "canonical": "yes", "source": "local_results/paper_baseline_input_map_comparison_20260828", "note": "comparison summary of four explicit raw-input runs"},
        {"experiment_id": "voxroom_ablation_1_3_20260829", "category": "ablation", "datasets": "interioragent;grscene", "status": "completed", "validity": "valid", "canonical": "yes", "source": "remote_voxroom_ablation/evaluation", "note": "no TVARS raw seed; no voxel raw seed; no neural filter, plus two controls"},
        {"experiment_id": "voxroom_ablation_4_6_preview_20260829", "category": "ablation", "datasets": "interioragent;grscene", "status": "completed_preview", "validity": "valid", "canonical": "yes_with_preliminary_flag", "source": "remote_voxroom_ablation/previews/current_best_20260829_1610", "note": "Nav and 3D checkpoints are preliminary; 2D model had completed training"},
        {"experiment_id": "voxroom_ablation_4_6_training", "category": "training", "datasets": "combined_training_set", "status": "in_progress", "validity": "valid", "canonical": "not_yet", "source": "remote_voxroom_ablation/training", "note": "Nav and 3D training active at archive time; 2D complete"},
        {"experiment_id": "voxroom_ablation_7_all_free", "category": "training", "datasets": "combined_training_set", "status": "dataset_ready_training_not_started", "validity": "valid", "canonical": "not_yet", "source": "remote_voxroom_ablation/training/all_free_dataset", "note": "766955 unique free-cell samples; large JSONL remains on remote data disk"},
    ]


def artifact_manifest(archive_root: Path, catalog_root: Path) -> list[dict[str, Any]]:
    ignored = {
        catalog_root / "artifact_manifest.csv",
        catalog_root / "archive_manifest.json",
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in archive_root.rglob("*") if item.is_file() and item not in ignored):
        stat = path.stat()
        rows.append(
            {
                "relative_path": str(path.relative_to(archive_root)),
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256(path),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a complete, audited experiment-results archive index.")
    parser.add_argument("--archive-root", required=True)
    args = parser.parse_args()
    archive_root = Path(args.archive_root).expanduser().resolve()
    catalog_root = archive_root / "catalog"
    catalog_root.mkdir(parents=True, exist_ok=True)

    metric_fields = [
        "experiment_id", "source_file", "source_kind", "result_status", "validity", "validity_note",
        "dataset", "input_mode", "scope", "checkpoint", "method_or_variant", "process_count",
        "evaluation_count", *METRIC_FIELDS,
    ]
    canonical_fields = metric_fields[:]
    canonical_fields.insert(canonical_fields.index("scope"), "input_applies_to_method")
    all_metrics = aggregate_rows(archive_root / "local_results", archive_root)
    all_metrics.extend(
        ablation_summary_rows(
            archive_root / "remote_voxroom_ablation/evaluation",
            archive_root,
            experiment_id="voxroom_ablation_formal_completed",
            default_status="completed_final",
        )
    )
    all_metrics.extend(
        ablation_summary_rows(
            archive_root / "remote_voxroom_ablation/previews/current_best_20260829_1610/evaluation",
            archive_root,
            experiment_id="voxroom_ablation_current_best_preview_20260829_1610",
            default_status="preliminary_current_best",
        )
    )
    canonical = canonical_rows(archive_root)
    training = training_rows(archive_root)
    experiments = experiment_catalog(archive_root)

    write_csv(catalog_root / "experiments.csv", experiments, list(experiments[0]))
    write_csv(catalog_root / "metrics_all_sources.csv", all_metrics, metric_fields)
    write_csv(catalog_root / "metrics_canonical.csv", canonical, canonical_fields)
    write_csv(
        catalog_root / "metrics_canonical_all_approved.csv",
        [row for row in canonical if row["scope"] == "all_approved"],
        canonical_fields,
    )
    write_csv(
        catalog_root / "metrics_canonical_excluding_training_validation.csv",
        [row for row in canonical if row["scope"] == "excluding_training_and_validation_scenes"],
        canonical_fields,
    )
    write_csv(
        catalog_root / "paper_baselines_canonical.csv",
        [row for row in canonical if "paper_baselines" in str(row["experiment_id"])],
        canonical_fields,
    )
    write_csv(
        catalog_root / "voxroom_ablations_canonical.csv",
        [row for row in canonical if "ablation" in str(row["experiment_id"])],
        canonical_fields,
    )
    write_csv(
        catalog_root / "invalid_superseded_metric_rows.csv",
        [row for row in all_metrics if row["validity"] not in {"valid", "duplicate_summary"}],
        metric_fields,
    )
    write_csv(catalog_root / "training_runs.csv", training, list(training[0]))

    all_free_report = archive_root / "remote_voxroom_ablation/training/all_free_dataset/dataset_vertical_all_free_unique.jsonl.build_report.json"
    if not all_free_report.is_file():
        all_free_report = archive_root / "local_results/voxroom_ablation_20260829/training/dataset_vertical_all_free_unique.jsonl.build_report.json"
    all_free = json.loads(all_free_report.read_text(encoding="utf-8")) if all_free_report.is_file() else {}

    report_lines = [
        "# 全实验结果归档（2026-08-29）",
        "",
        "本目录保存截至归档时已经完成或已经产生可用中间结果的实验。所有指标均保留 P、R、F1 和 room mIoU；默认房间面积过滤阈值为 0.5 m²。",
        "",
        "## 首选入口",
        "",
        "- `catalog/metrics_canonical.csv`：可直接用于论文表格的正式结果，以及显式标记为 preliminary 的当前最佳消融预览。",
        "- `catalog/metrics_canonical_all_approved.csv`：全部标注场景的精简首选表。",
        "- `catalog/metrics_canonical_excluding_training_validation.csv`：排除训练/验证场景后的精简首选表。",
        "- `catalog/metrics_all_sources.csv`：所有历史汇总行，包含重复、废弃和无效实验，便于审计。",
        "- `catalog/paper_baselines_canonical.csv`：四组正确输入的论文基线。",
        "- `catalog/voxroom_ablations_canonical.csv`：消融与对照组，包含结果状态字段。",
        "- `catalog/invalid_superseded_metric_rows.csv`：禁止引用或已被替代的历史指标。",
        "- `catalog/experiments.csv`：实验状态、有效性和替代关系。",
        "- `catalog/training_runs.csv`：④–⑦训练快照及 checkpoint 哈希。",
        "- `catalog/artifact_manifest.csv`：归档文件大小、时间和 SHA-256。",
        "",
        "## 必须遵守的结果状态",
        "",
        "- `interioragent_paper_baselines_20260828` 使用了带 GT 房间结构的 reference mask，属于输入泄漏，已保留但标为 invalid，不能引用。",
        "- raw Vertical-Free 和 no-clearance Nav-Free 四组论文基线是正式输入对照。",
        "- 消融①–③及两个控制组是正式完成结果。",
        "- 消融⑤仅二维模型已训练结束；消融④ Nav 和⑥仅三维的场景指标是训练中 checkpoint 的预览，最终训练结束后必须更新。",
        "- 消融⑦尚未训练；仅完成全 free-cell 数据构建。",
        "",
        "## 归档统计",
        "",
        f"- 实验条目：{len(experiments)}",
        f"- 全来源汇总行：{len(all_metrics)}",
        f"- 规范化首选指标行：{len(canonical)}",
        f"- 消融⑦去重样本：{all_free.get('row_count', 'unknown')}（正样本 {all_free.get('positive_count', 'unknown')}，负样本 {all_free.get('negative_count', 'unknown')}）",
        "",
        "## 大文件位置",
        "",
        "为避免再次占满本机磁盘，3.3 GB 原始 all-free JSONL 和 678 MB 去重 JSONL 未重复拷贝。本归档保存了构建报告、源/输出 SHA-256、样本统计和远端绝对路径。原文件仍位于：",
        "",
        "`/media/echo/data/voxroom_ablation_20260828/training/all_free_dataset/`",
        "",
        "远端完整实验根目录：",
        "",
        "`/media/echo/data/voxroom_ablation_20260828/`",
    ]
    (archive_root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest_rows = artifact_manifest(archive_root, catalog_root)
    write_csv(catalog_root / "artifact_manifest.csv", manifest_rows, list(manifest_rows[0]))
    manifest_payload = {
        "schema_version": "voxroom_complete_experiment_archive_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive_root),
        "artifact_count": len(manifest_rows),
        "artifact_bytes": sum(int(row["size_bytes"]) for row in manifest_rows),
        "artifact_manifest_sha256": sha256(catalog_root / "artifact_manifest.csv"),
        "experiments": len(experiments),
        "all_metric_rows": len(all_metrics),
        "canonical_metric_rows": len(canonical),
    }
    (catalog_root / "archive_manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
