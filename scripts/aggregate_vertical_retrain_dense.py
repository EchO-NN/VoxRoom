#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS = ("interioragent", "grscene")
VARIANTS = (
    "production_full_model_original",
    "saved_full_model_control",
    "vertical_full_retrain",
    "all_vertical_free_cells",
)
SCOPES = ("all_approved", "excluding_training_and_validation_scenes")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate the fresh Vertical-Free full model and all-free-cell ablation.")
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    evaluation_root = Path(args.evaluation_root).expanduser().resolve()
    training_root = Path(args.training_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    inputs: dict[str, str] = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            path = evaluation_root / f"{dataset}_{variant}" / "metrics" / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            summaries[(dataset, variant)] = json.loads(path.read_text(encoding="utf-8"))
            inputs[str(path)] = _sha256(path)

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for scope in SCOPES:
            control = summaries[(dataset, "saved_full_model_control")]["scopes"][scope]
            for variant in VARIANTS:
                metric = summaries[(dataset, variant)]["scopes"][scope]
                rows.append(
                    {
                        "dataset": dataset,
                        "scope": scope,
                        "variant": variant,
                        "process_count": int(metric["process_count"]),
                        "checkpoint_evaluation_count": int(metric["checkpoint_evaluation_count"]),
                        "precision_percent": float(metric["Precision_percent"]),
                        "recall_percent": float(metric["Recall_percent"]),
                        "f1_percent": float(metric["F1_percent"]),
                        "miou_room_percent": float(metric["mIoU_room_percent"]),
                        "delta_f1_vs_saved_control": float(metric["F1_percent"] - control["F1_percent"]),
                        "delta_miou_vs_saved_control": float(metric["mIoU_room_percent"] - control["mIoU_room_percent"]),
                    }
                )

    fields = list(rows[0])
    with (out_dir / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    training: dict[str, Any] = {}
    for variant in ("vertical_full_retrain", "all_vertical_free_cells"):
        directory = training_root / variant
        summary_path = directory / "training_summary.json"
        checkpoint_path = directory / "best.pt"
        history_path = directory / "training_history.json"
        curve_path = directory / "training_loss.png"
        for path in (summary_path, checkpoint_path, history_path, curve_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        training[variant] = {
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "history_path": str(history_path),
            "history_sha256": _sha256(history_path),
            "loss_curve_path": str(curve_path),
            "loss_curve_sha256": _sha256(curve_path),
        }

    payload = {
        "schema_version": "voxroom_vertical_retrain_and_all_free_ablation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_protocol": "all approved saved checkpoints; 0.5 m2 room cutoff; paper P/R, F1, Hungarian room mIoU",
        "rows": rows,
        "training": training,
        "input_summary_sha256": inputs,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Fresh Vertical-Free full model and all-free-cell ablation",
        "",
        "The fresh full model uses the original VoxRoom+TVARS raw-seed union, 19x19 local 3D voxels, and 41x41 Vertical-Free context. The all-free-cell ablation removes both raw-seed prefilters and scores every Vertical-Free coordinate.",
    ]
    for scope in SCOPES:
        lines.extend(
            [
                "",
                f"## {scope}",
                "",
                "| Dataset | Variant | P | R | F1 | mIoU | dF1 vs saved control | dmIoU vs saved control |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row["scope"] != scope:
                continue
            lines.append(
                "| {dataset} | {variant} | {precision_percent:.3f} | {recall_percent:.3f} | {f1_percent:.3f} | {miou_room_percent:.3f} | {delta_f1_vs_saved_control:+.3f} | {delta_miou_vs_saved_control:+.3f} |".format(**row)
            )
    lines.extend(["", "## Training", "", "| Variant | Epochs | Selected epoch | Validation F1 | PR-AUC | Checkpoint |", "|---|---:|---:|---:|---:|---|"])
    for variant, value in training.items():
        summary = value["summary"]
        metric = summary["selected_checkpoint_metrics"]
        lines.append(
            f"| {variant} | {int(summary['epochs_completed'])} | {int(summary['selected_checkpoint_epoch'])} | {float(metric['f1']):.4f} | {float(metric['pr_auc']):.4f} | `{value['checkpoint_path']}` |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "row_count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
