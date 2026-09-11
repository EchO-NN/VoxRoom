#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VARIANT_ORDER = (
    "production_full_model_original",
    "saved_full_model_control",
    "no_tvars_raw_seed",
    "no_voxel_raw_seed",
    "no_neural_filter",
    "nav_no_clearance_full",
    "vertical_2d_only",
    "vertical_3d_only",
    "all_vertical_free_cells",
)
DATASETS = ("interioragent", "grscene")
SCOPES = ("all_approved", "excluding_training_and_validation_scenes")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate completed InteriorAgent/GRScene VoxRoom ablations.")
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    evaluation_root = Path(args.evaluation_root).expanduser().resolve()
    training_root = Path(args.training_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    for variant in VARIANT_ORDER:
        for dataset in DATASETS:
            path = evaluation_root / (dataset + "_" + variant) / "metrics" / "summary.json"
            if not path.is_file():
                raise FileNotFoundError("missing metrics: %s" % path)
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("variant") != variant or summary.get("dataset") != dataset:
                raise ValueError("metrics identity mismatch: %s" % path)
            summaries[(dataset, variant)] = summary
            input_hashes[str(path)] = _sha256(path)

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in VARIANT_ORDER:
            summary = summaries[(dataset, variant)]
            for scope in SCOPES:
                metric = summary["scopes"][scope]
                control = summaries[(dataset, "saved_full_model_control")]["scopes"][scope]
                production = summaries[(dataset, "production_full_model_original")]["scopes"][scope]
                row = {
                    "dataset": dataset,
                    "variant": variant,
                    "scope": scope,
                    "process_count": int(metric["process_count"]),
                    "checkpoint_evaluation_count": int(metric["checkpoint_evaluation_count"]),
                    "precision_percent": float(metric["Precision_percent"]),
                    "recall_percent": float(metric["Recall_percent"]),
                    "f1_percent": float(metric["F1_percent"]),
                    "miou_room_percent": float(metric["mIoU_room_percent"]),
                    "delta_precision_points_vs_saved_control": float(metric["Precision_percent"] - control["Precision_percent"]),
                    "delta_recall_points_vs_saved_control": float(metric["Recall_percent"] - control["Recall_percent"]),
                    "delta_f1_points_vs_saved_control": float(metric["F1_percent"] - control["F1_percent"]),
                    "delta_miou_room_points_vs_saved_control": float(metric["mIoU_room_percent"] - control["mIoU_room_percent"]),
                    "delta_f1_points_vs_production": float(metric["F1_percent"] - production["F1_percent"]),
                    "delta_miou_room_points_vs_production": float(metric["mIoU_room_percent"] - production["mIoU_room_percent"]),
                    "candidate_policy": summary.get("replay_contract", {}).get("candidate_policy"),
                }
                rows.append(row)

    training: dict[str, Any] = {}
    for variant in (
        "nav_no_clearance_full",
        "vertical_2d_only",
        "vertical_3d_only",
        "all_vertical_free_cells",
    ):
        directory = training_root / variant
        summary_path = directory / "training_summary.json"
        history_path = directory / "training_history.json"
        checkpoint_path = directory / "best.pt"
        curve_path = directory / "training_loss.png"
        if not all(path.is_file() for path in (summary_path, history_path, checkpoint_path, curve_path)):
            raise FileNotFoundError("incomplete training artifacts: %s" % directory)
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        training[variant] = {
            "summary": value,
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "history_path": str(history_path),
            "history_sha256": _sha256(history_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "loss_curve_path": str(curve_path),
            "loss_curve_sha256": _sha256(curve_path),
        }

    fields = list(rows[0])
    csv_path = out_dir / "aggregate.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "voxroom_seven_ablation_aggregate_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variants": list(VARIANT_ORDER),
        "requested_ablation_count": 7,
        "references": ["production_full_model_original", "saved_full_model_control"],
        "datasets": list(DATASETS),
        "scopes": list(SCOPES),
        "metric_protocol": "all approved recorded checkpoints; 0.5 m2 room cutoff; paper region-overlap P/R plus F1 and Hungarian room mIoU",
        "rows": rows,
        "training": training,
        "input_summary_sha256": input_hashes,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# VoxRoom seven-part ablation",
        "",
        "All entries use every approved recorded segmentation checkpoint, the 0.5 m² room cutoff, paper region-overlap P/R, per-checkpoint F1, and Hungarian room mIoU. `production_full_model_original` is the actual online output; `saved_full_model_control` is the matched sparse offline-replay control. Neither is one of the seven requested ablations.",
    ]
    for scope in SCOPES:
        lines.extend(
            [
                "",
                "## %s" % scope,
                "",
                "| Dataset | Variant | Processes | Checkpoints | P (%) | R (%) | F1 (%) | mIoU (%) | ΔF1 vs matched | ΔmIoU vs matched | ΔF1 vs online | ΔmIoU vs online |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row["scope"] != scope:
                continue
            lines.append(
                "| {dataset} | {variant} | {process_count} | {checkpoint_evaluation_count} | {precision_percent:.3f} | {recall_percent:.3f} | {f1_percent:.3f} | {miou_room_percent:.3f} | {delta_f1_points_vs_saved_control:+.3f} | {delta_miou_room_points_vs_saved_control:+.3f} | {delta_f1_points_vs_production:+.3f} | {delta_miou_room_points_vs_production:+.3f} |".format(**row)
            )
    lines.extend(
        [
            "",
            "## Training artifacts",
            "",
            "| Variant | Epochs | Selected epoch | Validation accuracy | Validation F1 | Loss curve |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for variant, value in training.items():
        summary = value["summary"]
        selected = summary.get("selected_checkpoint_metrics", {})
        lines.append(
            "| %s | %d | %d | %.4f | %.4f | `%s` |"
            % (
                variant,
                int(summary["epochs_completed"]),
                int(summary["selected_checkpoint_epoch"]),
                float(selected.get("accuracy", 0.0)),
                float(selected.get("f1", 0.0)),
                value["loss_curve_path"],
            )
        )
    limitations = sorted(
        {
            str(summary.get("replay_contract", {}).get("source_history_limitation"))
            for summary in summaries.values()
            if summary.get("replay_contract", {}).get("source_history_limitation")
        }
    )
    if limitations:
        lines.extend(["", "## Replay audit note", ""])
        lines.extend("- " + value for value in limitations)
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "row_count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
