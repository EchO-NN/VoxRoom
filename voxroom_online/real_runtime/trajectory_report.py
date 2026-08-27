from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a VoxRoom ZED 2i trajectory.csv log.")
    parser.add_argument("trajectory_csv", help="Path to outputs/.../trajectory.csv")
    parser.add_argument("--json-out", default=None, help="Optional path for the JSON report.")
    return parser


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, percentile))


def summarize_trajectory(path: str | Path) -> dict[str, Any]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"trajectory log has no data rows: {csv_path}")

    states = Counter(str(row.get("tracking_state", "") or "UNKNOWN").strip().upper() for row in rows)
    skip_reasons = Counter(str(row.get("skip_reason", "") or "").strip() for row in rows)
    mapped_rows = [row for row in rows if str(row.get("mapped", "0")).strip() in {"1", "true", "True"}]

    timestamps_ns = []
    for row in rows:
        value = _float(row.get("image_timestamp_ns"))
        if math.isfinite(value) and value > 0:
            timestamps_ns.append(value)
    duration_s = None
    if len(timestamps_ns) >= 2 and timestamps_ns[-1] >= timestamps_ns[0]:
        duration_s = float((timestamps_ns[-1] - timestamps_ns[0]) * 1.0e-9)

    # Exclude provisional SEARCHING/OFF poses from drift/path statistics.
    ok_rows = [row for row in rows if str(row.get("tracking_state", "") or "").strip().upper() == "OK"]
    xy: list[tuple[float, float]] = []
    xyz: list[tuple[float, float, float]] = []
    for row in ok_rows:
        x = _float(row.get("base_x_m"))
        y = _float(row.get("base_y_m"))
        if math.isfinite(x) and math.isfinite(y):
            xy.append((x, y))
        z = _float(row.get("base_z_m"))
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            xyz.append((x, y, z))
    path_length_m = 0.0
    for a, b in zip(xy, xy[1:]):
        path_length_m += math.hypot(b[0] - a[0], b[1] - a[1])
    path_length_3d_m = 0.0
    for a, b in zip(xyz, xyz[1:]):
        path_length_3d_m += math.sqrt(sum((b[index] - a[index]) ** 2 for index in range(3)))
    endpoint_drift_m = None
    max_displacement_m = None
    if xy:
        endpoint_drift_m = float(math.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1]))
        max_displacement_m = float(max(math.hypot(x - xy[0][0], y - xy[0][1]) for x, y in xy))
    endpoint_displacement_3d_m = None
    vertical_range_m = None
    if xyz:
        endpoint_displacement_3d_m = float(
            math.sqrt(sum((xyz[-1][index] - xyz[0][index]) ** 2 for index in range(3)))
        )
        z_values = [point[2] for point in xyz]
        vertical_range_m = float(max(z_values) - min(z_values))

    tilt_deg: list[float] = []
    for row in ok_rows:
        roll = _float(row.get("camera_roll_deg"))
        pitch = _float(row.get("camera_pitch_deg"))
        if math.isfinite(roll) and math.isfinite(pitch):
            tilt_deg.append(max(abs(roll), abs(pitch)))
    map_ms: list[float] = []
    for row in mapped_rows:
        value = _float(row.get("mapper_total_ms"))
        if math.isfinite(value) and value > 0:
            map_ms.append(value)

    total = len(rows)
    tracking_ok = int(states.get("OK", 0))
    report: dict[str, Any] = {
        "trajectory_csv": str(csv_path),
        "rows": total,
        "duration_s": duration_s,
        "tracking_ok_rows": tracking_ok,
        "tracking_ok_ratio": float(tracking_ok / total),
        "mapped_frames": len(mapped_rows),
        "mapped_hz": (float(len(mapped_rows) / duration_s) if duration_s and duration_s > 0 else None),
        "path_length_m": float(path_length_m),
        "path_length_3d_m": float(path_length_3d_m) if xyz else None,
        "endpoint_displacement_m": endpoint_drift_m,
        "endpoint_displacement_3d_m": endpoint_displacement_3d_m,
        "max_displacement_from_start_m": max_displacement_m,
        "base_vertical_range_m": vertical_range_m,
        "camera_tilt_deg": {
            "p50": _percentile(tilt_deg, 50.0),
            "p95": _percentile(tilt_deg, 95.0),
            "max": max(tilt_deg) if tilt_deg else None,
        },
        "mapper_update_ms": {
            "mean": float(np.mean(map_ms)) if map_ms else None,
            "p95": _percentile(map_ms, 95.0),
            "max": max(map_ms) if map_ms else None,
        },
        "tracking_state_counts": dict(sorted(states.items())),
        "skip_reason_counts": dict(sorted((key or "<none>", value) for key, value in skip_reasons.items())),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = summarize_trajectory(args.trajectory_csv)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
