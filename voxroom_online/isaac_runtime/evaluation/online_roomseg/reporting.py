from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import make_jsonable, write_json_atomic


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(make_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {key: _csv_value(row.get(key)) for key in fieldnames}
            writer.writerow(clean)


def aggregate_episode_first(
    per_snapshot_rows: list[Mapping[str, Any]],
    *,
    strict_paper: bool,
    require_csr: bool,
    precision_recall_only: bool = False,
) -> tuple[list[dict], dict]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in per_snapshot_rows:
        if not bool(row.get("valid", True)):
            continue
        grouped.setdefault(str(row["episode_uid"]), []).append(row)
    per_episode: list[dict] = []
    for episode_uid, rows in sorted(grouped.items()):
        if not rows:
            continue
        csr_values = [row.get("csr") for row in rows if row.get("csr") is not None]
        episode_row = {
            "episode_uid": episode_uid,
            "scene_id": rows[0].get("scene_id"),
            "T_e": int(len(rows)),
            "Precision": _mean([row.get("precision", 0.0) for row in rows]),
            "Recall": _mean([row.get("recall", 0.0) for row in rows]),
        }
        if not precision_recall_only:
            episode_row.update(
                {
                    "CSR": None if not require_csr else _mean(csr_values),
                    "USR": _mean([row.get("usr", 0.0) for row in rows]),
                    "OSR": _mean([row.get("osr", 0.0) for row in rows]),
                    "mIoU_room": _mean(
                        [row.get("miou_room", 0.0) for row in rows]
                    ),
                }
            )
        per_episode.append(episode_row)
    coverage_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in per_snapshot_rows:
        event_id = row.get("coverage_event_id")
        if event_id is None or not bool(row.get("valid", True)):
            continue
        coverage_groups.setdefault(str(event_id), []).append(row)
    coverage_curve = []
    for event_id, rows in coverage_groups.items():
        thresholds = [row.get("coverage_threshold") for row in rows]
        threshold = next((float(value) for value in thresholds if value is not None), None)
        coverage_curve.append(
            {
                "event_id": event_id,
                "event_kind": rows[0].get("coverage_event_kind"),
                "threshold": threshold,
                "mean_measured_coverage": _mean(
                    row.get("coverage_ratio") for row in rows
                ),
                "snapshot_count": int(len(rows)),
                "Precision": _mean(row.get("precision") for row in rows),
                "Recall": _mean(row.get("recall") for row in rows),
            }
        )
    coverage_curve.sort(
        key=lambda row: (
            row["threshold"] is None,
            2.0 if row["threshold"] is None else float(row["threshold"]),
            str(row["event_id"]),
        )
    )
    summary = {
        "schema_version": "voxroom_online_roomseg_metrics_v2",
        "metric_protocol": (
            "bormann_precision_recall_only"
            if precision_recall_only
            else "legacy_diagnostics_plus_bormann_precision_recall"
        ),
        "episode_count": int(len(per_episode)),
        "snapshot_count": int(sum(int(ep["T_e"]) for ep in per_episode)),
        "strict_paper": bool(strict_paper),
        "Precision": _mean([ep["Precision"] for ep in per_episode]),
        "Recall": _mean([ep["Recall"] for ep in per_episode]),
        "coverage_curve": coverage_curve,
    }
    if not precision_recall_only:
        summary.update(
            {
                "CSR": None
                if not require_csr
                else _mean(
                    ep["CSR"] for ep in per_episode if ep["CSR"] is not None
                ),
                "csr_status": (
                    "required" if require_csr else "not_required_geometric_only"
                ),
                "USR": _mean(ep["USR"] for ep in per_episode),
                "OSR": _mean(ep["OSR"] for ep in per_episode),
                "mIoU_room": _mean(ep["mIoU_room"] for ep in per_episode),
                "pooled_diagnostics": {
                    "mean_usr_over_all_snapshots": _mean(
                        row.get("usr", 0.0)
                        for row in per_snapshot_rows
                        if bool(row.get("valid", True))
                    ),
                    "mean_osr_over_all_snapshots": _mean(
                        row.get("osr", 0.0)
                        for row in per_snapshot_rows
                        if bool(row.get("valid", True))
                    ),
                    "mean_miou_over_all_snapshots": _mean(
                        row.get("miou_room", 0.0)
                        for row in per_snapshot_rows
                        if bool(row.get("valid", True))
                    ),
                },
            }
        )
    return per_episode, summary


def write_summary_markdown(path: Path, summary: Mapping[str, Any], per_episode: list[Mapping[str, Any]]) -> None:
    precision_recall_only = summary.get("metric_protocol") == "bormann_precision_recall_only"
    if precision_recall_only:
        lines = [
            "# Room Segmentation Precision / Recall",
            "",
            "| Metric | Value |",
            "|---|---:|",
            "| Episodes | %s |" % summary.get("episode_count"),
            "| Snapshots | %s |" % summary.get("snapshot_count"),
            "| Precision | %s |" % _fmt(summary.get("Precision")),
            "| Recall | %s |" % _fmt(summary.get("Recall")),
            "",
            "## Per Episode",
            "",
            "| Episode | Scene | T | Precision | Recall |",
            "|---|---|---:|---:|---:|",
        ]
        for row in per_episode:
            lines.append(
                "| {episode_uid} | {scene_id} | {T_e} | {Precision} | {Recall} |".format(
                    episode_uid=row.get("episode_uid"),
                    scene_id=row.get("scene_id"),
                    T_e=row.get("T_e"),
                    Precision=_fmt(row.get("Precision")),
                    Recall=_fmt(row.get("Recall")),
                )
            )
        coverage_curve = list(summary.get("coverage_curve", []) or [])
        if coverage_curve:
            lines.extend(
                [
                    "",
                    "## Coverage Curve",
                    "",
                    "| Event | Threshold | Measured Coverage | N | Precision | Recall |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in coverage_curve:
                lines.append(
                    "| {event_id} | {threshold} | {coverage} | {count} | {precision} | {recall} |".format(
                        event_id=row.get("event_id"),
                        threshold=_fmt(row.get("threshold")),
                        coverage=_fmt(row.get("mean_measured_coverage")),
                        count=row.get("snapshot_count"),
                        precision=_fmt(row.get("Precision")),
                        recall=_fmt(row.get("Recall")),
                    )
                )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines = [
        "# VoxRoom-Online RoomSeg Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Episodes | %s |" % summary.get("episode_count"),
        "| Snapshots | %s |" % summary.get("snapshot_count"),
        "| CSR | %s |" % _fmt(summary.get("CSR")),
        "| USR | %s |" % _fmt(summary.get("USR")),
        "| OSR | %s |" % _fmt(summary.get("OSR")),
        "| mIoU_room | %s |" % _fmt(summary.get("mIoU_room")),
        "| Precision | %s |" % _fmt(summary.get("Precision")),
        "| Recall | %s |" % _fmt(summary.get("Recall")),
        "",
        "## Per Episode",
        "",
        "| Episode | Scene | T | CSR | USR | OSR | mIoU |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in per_episode:
        lines.append(
            "| {episode_uid} | {scene_id} | {T_e} | {CSR} | {USR} | {OSR} | {mIoU_room} |".format(
                episode_uid=row.get("episode_uid"),
                scene_id=row.get("scene_id"),
                T_e=row.get("T_e"),
                CSR=_fmt(row.get("CSR")),
                USR=_fmt(row.get("USR")),
                OSR=_fmt(row.get("OSR")),
                mIoU_room=_fmt(row.get("mIoU_room")),
            )
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_gallery(
    path: Path,
    per_snapshot_rows: list[Mapping[str, Any]],
    *,
    precision_recall_only: bool = False,
) -> None:
    rows = ["<html><body><h1>VoxRoom-Online RoomSeg Review Gallery</h1>"]
    current_episode = None
    for row in per_snapshot_rows:
        ep = row.get("episode_uid")
        if ep != current_episode:
            if current_episode is not None:
                rows.append("</section>")
            rows.append(f"<section><h2>{ep}</h2>")
            current_episode = ep
        image = row.get("match_visualization")
        if precision_recall_only:
            caption = (
                "event={event} coverage={coverage:.4f} step={step} "
                "Precision={precision:.4f} Recall={recall:.4f}"
            ).format(
                event=row.get("coverage_event_id"),
                coverage=float(row.get("coverage_ratio") or 0.0),
                step=row.get("step"),
                precision=float(row.get("precision", 0.0)),
                recall=float(row.get("recall", 0.0)),
            )
        else:
            caption = (
                "step={step} Precision={precision:.4f} Recall={recall:.4f} "
                "USR={usr:.4f} OSR={osr:.4f} mIoU={miou:.4f} CSR={csr}"
            ).format(
                step=row.get("step"),
                precision=float(row.get("precision", 0.0)),
                recall=float(row.get("recall", 0.0)),
                usr=float(row.get("usr", 0.0)),
                osr=float(row.get("osr", 0.0)),
                miou=float(row.get("miou_room", 0.0)),
                csr=row.get("csr"),
            )
        rows.append(
            "<div><p>{caption}</p>{img}</div>".format(
                caption=caption,
                img=f'<img src="{image}" style="max-width:1000px">'
                if image
                else "",
            )
        )
    if current_episode is not None:
        rows.append("</section>")
    rows.append("</body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def write_metrics_outputs(out_dir: Path, per_snapshot_rows: list[Mapping[str, Any]], per_episode: list[dict], summary: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "per_snapshot_metrics.jsonl", per_snapshot_rows)
    precision_recall_only = (
        summary.get("metric_protocol") == "bormann_precision_recall_only"
    )
    common_fields = [
        "episode_uid",
        "scene_id",
        "coverage_method",
        "coverage_event_id",
        "coverage_event_kind",
        "coverage_threshold",
        "coverage_ratio",
        "step",
        "metric_domain_pixels",
        "min_room_area_m2",
        "cell_size_m",
        "min_room_area_cells",
        "gt_label_masks_before_filter",
        "gt_label_masks_after_filter",
        "gt_label_masks_filtered_small",
        "pred_label_masks_before_filter",
        "pred_label_masks_after_filter",
        "pred_label_masks_filtered_small",
        "n_gt",
        "n_pred",
    ]
    metric_fields = (
        ["precision", "recall"]
        if precision_recall_only
        else ["usr", "osr", "miou_room", "precision", "recall", "csr"]
    )
    write_csv(
        out_dir / "per_snapshot_metrics.csv",
        per_snapshot_rows,
        common_fields + metric_fields + ["valid", "invalid_reason"],
    )
    write_json_atomic(out_dir / "per_episode_metrics.json", per_episode)
    write_json_atomic(out_dir / "summary_metrics.json", summary)
    write_summary_markdown(out_dir / "metrics_report.md", summary, per_episode)
    write_review_gallery(
        out_dir / "review_gallery.html",
        per_snapshot_rows,
        precision_recall_only=precision_recall_only,
    )


def _mean(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / float(len(vals)))


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    return "%.6f" % float(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(make_jsonable(value), ensure_ascii=False, sort_keys=True)
    return make_jsonable(value)
