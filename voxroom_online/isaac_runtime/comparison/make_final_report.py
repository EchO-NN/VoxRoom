from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np

from voxroom_online.isaac_runtime.comparison.metadata_gate import assert_main_experiment_metadata


DEFAULT_METHODS = ("voxroom", "topology_visual_active", "dude_incremental", "rose2", "morphological", "distance_transform", "voronoi")


def make_final_report(*, eval_root: Path, methods: Sequence[str], out_dir: Path) -> dict[str, Any]:
    eval_root = Path(eval_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {"eval_root": str(eval_root), "methods": {}}
    for method in methods:
        metrics_dir = eval_root / "metrics" / str(method)
        summary = _read_json(metrics_dir / "summary_metrics.json", default={})
        per_snapshot = _read_jsonl(metrics_dir / "per_snapshot_metrics.jsonl")
        method_row = _method_row(str(method), summary, per_snapshot)
        rows.append(method_row)
        provenance["methods"][str(method)] = {
            "metrics_dir": str(metrics_dir),
            "summary_metrics": summary,
            "runner_types": method_row["runner_types"],
            "original_repo_commits": method_row["original_repo_commits"],
        }
    _write_csv(out_dir / "final_comparison_table.csv", rows)
    _write_csv(out_dir / "final_per_scene_table.csv", _per_scene_rows(per_method_rows=rows, eval_root=eval_root))
    (out_dir / "final_comparison_summary.md").write_text(_summary_markdown(rows), encoding="utf-8")
    (out_dir / "provenance_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rows": rows, "out_dir": str(out_dir)}


def _method_row(method: str, summary: dict[str, Any], per_snapshot: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = [_snapshot_metadata(row) for row in per_snapshot]
    runner_types = sorted({str(meta.get("runner_type")) for meta in metadata if meta.get("runner_type")})
    commits = sorted({str(meta.get("original_repo_commit")) for meta in metadata if meta.get("original_repo_commit")})
    invalid_count = sum(1 for row in per_snapshot if row.get("invalid_reason"))
    valid_count = max(0, len(per_snapshot) - invalid_count)
    return {
        "method": method,
        "num_scenes": int(summary.get("episode_count", 0) or 0),
        "num_snapshots": int(summary.get("snapshot_count", len(per_snapshot)) or 0),
        "USR_mean": _float_or_empty(summary.get("USR")),
        "USR_std": _metric_std(per_snapshot, "usr"),
        "OSR_mean": _float_or_empty(summary.get("OSR")),
        "OSR_std": _metric_std(per_snapshot, "osr"),
        "mIoU_room_mean": _float_or_empty(summary.get("mIoU_room")),
        "mIoU_room_std": _metric_std(per_snapshot, "miou_room"),
        "CSR_mean": _float_or_empty(summary.get("CSR")),
        "CSR_std": _metric_std(per_snapshot, "csr"),
        "valid_snapshot_count": int(valid_count),
        "invalid_snapshot_count": int(invalid_count),
        "main_experiment_gate_passed": _main_gate_passed(method, metadata),
        "runner_types": ";".join(runner_types),
        "original_repo_commits": ";".join(commits),
    }


def _snapshot_metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("baseline_metadata")
    if isinstance(value, dict):
        return value
    value = row.get("baseline_metadata_json")
    if isinstance(value, str) and value:
        try:
            return dict(json.loads(value))
        except Exception:
            return {}
    snapshot_path = row.get("snapshot_path")
    if snapshot_path:
        try:
            with np.load(Path(str(snapshot_path)), allow_pickle=False) as data:
                if "baseline_metadata_json" in data.files:
                    return dict(json.loads(str(data["baseline_metadata_json"])))
        except Exception:
            return {}
    return {}


def _main_gate_passed(method: str, metadata: list[dict[str, Any]]) -> bool:
    if method == "voxroom":
        return True
    if not metadata:
        return False
    try:
        for meta in metadata:
            assert_main_experiment_metadata(meta, method)
    except Exception:
        return False
    return True


def _metric_std(rows: list[dict[str, Any]], key: str) -> str:
    vals: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            vals.append(float(value))
        except Exception:
            pass
    if not vals:
        return ""
    if len(vals) == 1:
        return "0.0"
    return repr(float(pstdev(vals)))


def _float_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        return repr(float(value))
    except Exception:
        return ""


def _per_scene_rows(*, per_method_rows: list[dict[str, Any]], eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_row in per_method_rows:
        method = str(method_row["method"])
        path = Path(eval_root) / "metrics" / method / "per_episode_metrics.json"
        for item in _read_json(path, default=[]):
            row = dict(item)
            row["method"] = method
            rows.append(row)
    return rows


def _summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# VoxRoom comparison summary",
        "",
        "| method | scenes | snapshots | USR | OSR | mIoU_room | CSR | main_gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {num_scenes} | {num_snapshots} | {USR_mean} | {OSR_mean} | {mIoU_room_mean} | {CSR_mean} | {main_experiment_gate_passed} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def _read_json(path: Path, *, default: Any) -> Any:
    if not Path(path).exists():
        return default
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "method",
        "num_scenes",
        "num_snapshots",
        "USR_mean",
        "USR_std",
        "OSR_mean",
        "OSR_std",
        "mIoU_room_mean",
        "mIoU_room_std",
        "CSR_mean",
        "CSR_std",
        "valid_snapshot_count",
        "invalid_snapshot_count",
        "main_experiment_gate_passed",
        "runner_types",
        "original_repo_commits",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate per-method VoxRoom comparison metrics into final report files.")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    report = make_final_report(eval_root=Path(args.eval_root), methods=[str(v) for v in args.methods], out_dir=Path(args.out_dir))
    print("wrote comparison report to %s for %d method(s)" % (report["out_dir"], len(report["rows"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
