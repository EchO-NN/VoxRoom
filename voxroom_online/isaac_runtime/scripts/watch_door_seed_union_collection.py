#!/usr/bin/env python3
"""Show the latest persisted hybrid DoorSeed grids without restarting collection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


VOXROOM_BGR = (0, 165, 255)
TVARS_BGR = (255, 70, 180)
OVERLAP_BGR = (55, 215, 55)


def _latest_scene(collection_root: Path, *, dataset: str) -> tuple[dict, Path] | None:
    choices: list[tuple[int, float, dict, Path]] = []
    for manifest_path in collection_root.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        scene_id = str(manifest.get("scene_id", ""))
        is_grscene = "grscene" in scene_id.lower()
        if (dataset == "grscene") != is_grscene:
            continue
        snapshots = list(manifest.get("snapshots", []))
        if not snapshots:
            continue
        last = max(snapshots, key=lambda item: int(item.get("decision_id", -1)))
        is_final = bool(last.get("is_final", False))
        choices.append(
            (
                0 if is_final else 1,
                float(last.get("created_at_unix", manifest_path.stat().st_mtime)),
                manifest,
                manifest_path.parent,
            )
        )
    if not choices:
        return None
    _active, _mtime, manifest, scene_dir = max(choices, key=lambda item: (item[0], item[1]))
    return manifest, scene_dir


def _render_latest(collection_root: Path, *, dataset: str, panel_width: int, panel_height: int) -> np.ndarray:
    selected = _latest_scene(collection_root, dataset=dataset)
    if selected is None:
        return _message_panel(dataset, "waiting for first snapshot", panel_width, panel_height)
    manifest, scene_dir = selected
    records = list(manifest.get("snapshots", []))
    record = max(records, key=lambda item: int(item.get("decision_id", -1)))
    snapshot_path = scene_dir / str(record["path"])
    try:
        with np.load(snapshot_path, allow_pickle=False) as data:
            vertical = np.asarray(data["vertical_class_map_xy"], dtype=np.uint8)
            source = np.asarray(data["raw_seed_source_id_map_xy"], dtype=np.uint8)
    except (OSError, ValueError, KeyError) as exc:
        return _message_panel(dataset, "snapshot read retry: %s" % exc, panel_width, panel_height)

    grid = np.full((*vertical.shape, 3), 150, dtype=np.uint8)
    grid[vertical == 1] = (248, 248, 248)
    grid[vertical == 2] = (25, 25, 25)
    grid[source == 1] = VOXROOM_BGR
    grid[source == 2] = TVARS_BGR
    grid[source == 3] = OVERLAP_BGR

    visible = (vertical != 0) | (source != 0)
    rows, cols = np.nonzero(visible)
    if rows.size:
        margin = 12
        r0 = max(0, int(rows.min()) - margin)
        r1 = min(grid.shape[0], int(rows.max()) + margin + 1)
        c0 = max(0, int(cols.min()) - margin)
        c1 = min(grid.shape[1], int(cols.max()) + margin + 1)
        grid = grid[r0:r1, c0:c1]

    header_height = 96
    usable_w = max(1, panel_width - 20)
    usable_h = max(1, panel_height - header_height - 10)
    scale = min(usable_w / grid.shape[1], usable_h / grid.shape[0])
    resized = cv2.resize(
        grid,
        (max(1, int(round(grid.shape[1] * scale))), max(1, int(round(grid.shape[0] * scale)))),
        interpolation=cv2.INTER_NEAREST,
    )
    panel = np.full((panel_height, panel_width, 3), 238, dtype=np.uint8)
    x0 = (panel_width - resized.shape[1]) // 2
    y0 = header_height + (usable_h - resized.shape[0]) // 2
    panel[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized

    scene_id = str(manifest.get("scene_id", ""))
    step = int(record.get("step", -1))
    state = "FINAL" if bool(record.get("is_final", False)) else "RUNNING"
    title = "%s | %s | step %d | %s" % (dataset.upper(), scene_id, step, state)
    counts = "union %d   VoxRoom %d   TVARS %d" % (
        int(record.get("raw_seed_count", 0)),
        int(record.get("voxroom_raw_seed_count", 0)),
        int(record.get("tvars_vertical_raw_seed_count", 0)),
    )
    cv2.putText(panel, title[:105], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, counts, (12, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    _legend(panel, y=82)
    return panel


def _legend(panel: np.ndarray, *, y: int) -> None:
    entries = (("VoxRoom", VOXROOM_BGR), ("TVARS Vertical", TVARS_BGR), ("overlap", OVERLAP_BGR))
    x = 12
    for label, color in entries:
        cv2.rectangle(panel, (x, y - 12), (x + 18, y + 2), color, thickness=-1)
        cv2.putText(panel, label, (x + 25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
        x += 145


def _message_panel(dataset: str, message: str, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 238, dtype=np.uint8)
    cv2.putText(panel, dataset.upper(), (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(panel, message[:95], (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return panel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--refresh-seconds", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--window-name", default="VoxRoom + TVARS raw DoorSeed live grid")
    args = parser.parse_args()

    width = max(900, int(args.width))
    height = max(500, int(args.height))
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, width, height)
    next_refresh = 0.0
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    while True:
        now = time.monotonic()
        if now >= next_refresh:
            half = width // 2
            left = _render_latest(args.collection_root, dataset="interioragent", panel_width=half, panel_height=height)
            right = _render_latest(args.collection_root, dataset="grscene", panel_width=width - half, panel_height=height)
            canvas = np.concatenate((left, right), axis=1)
            cv2.line(canvas, (half, 0), (half, height - 1), (90, 90, 90), 2)
            next_refresh = now + max(0.2, float(args.refresh_seconds))
        cv2.imshow(args.window_name, canvas)
        if cv2.waitKey(100) & 0xFF in (27, ord("q")):
            break
        try:
            if cv2.getWindowProperty(args.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
