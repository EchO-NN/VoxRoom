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


def aggregate_episode_first(per_snapshot_rows: list[Mapping[str, Any]], *, strict_paper: bool, require_csr: bool) -> tuple[list[dict], dict]:
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
        per_episode.append(
            {
                "episode_uid": episode_uid,
                "scene_id": rows[0].get("scene_id"),
                "T_e": int(len(rows)),
                "CSR": None if not require_csr else _mean(csr_values),
                "USR": _mean([row.get("usr", 0.0) for row in rows]),
                "OSR": _mean([row.get("osr", 0.0) for row in rows]),
                "mIoU_room": _mean([row.get("miou_room", 0.0) for row in rows]),
            }
        )
    summary = {
        "schema_version": "voxroom_online_roomseg_metrics_v1",
        "episode_count": int(len(per_episode)),
        "snapshot_count": int(sum(int(ep["T_e"]) for ep in per_episode)),
        "strict_paper": bool(strict_paper),
        "CSR": None if not require_csr else _mean([ep["CSR"] for ep in per_episode if ep["CSR"] is not None]),
        "csr_status": "required" if require_csr else "not_required_geometric_only",
        "USR": _mean([ep["USR"] for ep in per_episode]),
        "OSR": _mean([ep["OSR"] for ep in per_episode]),
        "mIoU_room": _mean([ep["mIoU_room"] for ep in per_episode]),
        "pooled_diagnostics": {
            "mean_usr_over_all_snapshots": _mean([row.get("usr", 0.0) for row in per_snapshot_rows if bool(row.get("valid", True))]),
            "mean_osr_over_all_snapshots": _mean([row.get("osr", 0.0) for row in per_snapshot_rows if bool(row.get("valid", True))]),
            "mean_miou_over_all_snapshots": _mean([row.get("miou_room", 0.0) for row in per_snapshot_rows if bool(row.get("valid", True))]),
        },
    }
    return per_episode, summary


def write_summary_markdown(path: Path, summary: Mapping[str, Any], per_episode: list[Mapping[str, Any]]) -> None:
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


def write_review_gallery(path: Path, per_snapshot_rows: list[Mapping[str, Any]]) -> None:
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
        rows.append(
            "<div><p>step={step} USR={usr:.4f} OSR={osr:.4f} mIoU={miou:.4f} CSR={csr}</p>{img}</div>".format(
                step=row.get("step"),
                usr=float(row.get("usr", 0.0)),
                osr=float(row.get("osr", 0.0)),
                miou=float(row.get("miou_room", 0.0)),
                csr=row.get("csr"),
                img=f'<img src="{image}" style="max-width:1000px">' if image else "",
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
    write_csv(
        out_dir / "per_snapshot_metrics.csv",
        per_snapshot_rows,
        [
            "episode_uid",
            "scene_id",
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
            "usr",
            "osr",
            "miou_room",
            "csr",
            "valid",
            "invalid_reason",
        ],
    )
    write_json_atomic(out_dir / "per_episode_metrics.json", per_episode)
    write_json_atomic(out_dir / "summary_metrics.json", summary)
    write_summary_markdown(out_dir / "metrics_report.md", summary, per_episode)
    write_review_gallery(out_dir / "review_gallery.html", per_snapshot_rows)


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
