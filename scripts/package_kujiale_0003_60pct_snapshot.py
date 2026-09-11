#!/usr/bin/env python3
"""Validate and package a Kujiale 0003 coverage snapshot for paper figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image


REQUIRED_ARRAYS = (
    "voxel_occupancy_state_zyx",
    "voxel_occupancy_log_odds_zyx",
    "voxel_sensor_range_count_zyx",
    "voxel_occupancy_z_centers_m",
    "voxel_vertical_free_xy",
    "voxel_nav_free_xy",
    "voxel_nav_occupied_xy",
    "voxel_nav_unknown_xy",
    "voxel_voxroom_raw_seed_mask",
    "voxel_tvars_vertical_raw_seed_mask",
    "voxel_voxroom_raw_seed_history_mask",
    "voxel_tvars_vertical_raw_seed_history_mask",
    "voxel_raw_seed_source_id_map",
    "voxel_door_raw_seed_mask",
    "voxel_door_seed_model_probability_xy",
    "voxel_door_seed_model_keep_mask",
    "voxel_door_seed_model_reject_mask",
    "voxel_final_separator_map",
    "voxel_final_room_label_map",
    "roomseg_eval_reference_explorable_mask",
    "roomseg_eval_explored_reference_mask",
    "roomseg_eval_coverage_ratio",
    "demo_rgb",
    "demo_depth_m",
    "demo_pose_world",
    "demo_current_path_rc",
    "demo_full_path_rc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--milestone-percent", type=int, default=60)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=86400.0)
    parser.add_argument("--poll-s", type=float, default=5.0)
    return parser.parse_args()


def read_complete_event(
    manifest_path: Path, *, event_id: str
) -> tuple[dict, dict] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    matches = [
        event
        for event in manifest.get("events", [])
        if event.get("event_id") == event_id
    ]
    if len(matches) != 1 or matches[0].get("status") != "complete":
        return None
    return manifest, matches[0]


def wait_for_event(
    manifest_path: Path,
    *,
    event_id: str,
    milestone_percent: int,
    timeout_s: float,
    poll_s: float,
) -> tuple[dict, dict]:
    started = time.monotonic()
    while True:
        result = read_complete_event(manifest_path, event_id=event_id)
        if result is not None:
            return result
        if time.monotonic() - started >= timeout_s:
            raise TimeoutError(
                f"{milestone_percent}% coverage snapshot was not completed "
                f"within {timeout_s:.0f}s"
            )
        time.sleep(max(0.2, poll_s))


def resolve_artifact(path_value: object, run_root: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    return path if path.is_absolute() else run_root / path


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return "existing_hardlink"
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_depth_preview(depth_m: np.ndarray, path: Path) -> None:
    depth = np.asarray(depth_m, dtype=np.float32)
    finite = np.isfinite(depth) & (depth > 0.0)
    image = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(finite):
        low, high = np.percentile(depth[finite], [2.0, 98.0])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((depth - low) / (high - low), 0.0, 1.0)
        image[finite] = np.rint(255.0 * (1.0 - scaled[finite])).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def save_seed_preview(arrays: np.lib.npyio.NpzFile, path: Path) -> None:
    vertical_free = np.asarray(arrays["voxel_vertical_free_xy"], dtype=bool)
    nav_occupied = np.asarray(arrays["voxel_nav_occupied_xy"], dtype=bool)
    voxroom = np.asarray(arrays["voxel_voxroom_raw_seed_history_mask"], dtype=bool)
    tvars = np.asarray(arrays["voxel_tvars_vertical_raw_seed_history_mask"], dtype=bool)
    accepted = np.asarray(arrays["voxel_door_seed_model_keep_mask"], dtype=bool)
    rejected = np.asarray(arrays["voxel_door_seed_model_reject_mask"], dtype=bool)
    canvas = np.full((*vertical_free.shape, 3), 28, dtype=np.uint8)
    canvas[vertical_free] = (242, 242, 238)
    canvas[nav_occupied] = (70, 70, 70)
    canvas[tvars] = (151, 63, 196)
    canvas[voxroom] = (35, 110, 235)
    canvas[rejected] = (225, 76, 145)
    canvas[accepted] = (35, 190, 85)
    Image.fromarray(canvas, mode="RGB").save(path)


def main() -> int:
    args = parse_args()
    milestone_percent = int(args.milestone_percent)
    if not 1 <= milestone_percent <= 100:
        raise ValueError("--milestone-percent must be in [1, 100]")
    event_id = f"milestone_{milestone_percent:03d}"
    milestone_label = f"{milestone_percent}pct"
    file_prefix = f"kujiale_0003_{milestone_label}"
    run_root = args.run_root.expanduser().resolve()
    manifest_path = run_root / "roomseg_coverage_eval" / "manifest.json"
    result = (
        wait_for_event(
            manifest_path,
            event_id=event_id,
            milestone_percent=milestone_percent,
            timeout_s=args.timeout_s,
            poll_s=args.poll_s,
        )
        if args.wait
        else read_complete_event(manifest_path, event_id=event_id)
    )
    if result is None:
        raise RuntimeError(f"{event_id} is not complete")
    manifest, event = result
    artifacts = dict(event.get("artifacts", {}))
    snapshot = resolve_artifact(artifacts["voxroom_snapshot_npz"], run_root)
    summary = resolve_artifact(artifacts["voxroom_summary_json"], run_root)
    navigation = resolve_artifact(artifacts["voxroom_navigation_png"], run_root)
    reference_npz = resolve_artifact(manifest["reference_npz"], run_root)
    reference_json = resolve_artifact(manifest["reference_json"], run_root)
    for path in (snapshot, summary, navigation, reference_npz, reference_json):
        if not path.is_file():
            raise FileNotFoundError(path)

    package_dir = run_root / f"paper_visualization_{milestone_label}"
    package_dir.mkdir(parents=True, exist_ok=True)
    links = {}
    for source, name in (
        (snapshot, f"{file_prefix}_full_state.npz"),
        (summary, f"{file_prefix}_roomseg_summary.json"),
        (navigation, f"{file_prefix}_navigation_rooms.png"),
        (reference_npz, "kujiale_0003_scene_reference.npz"),
        (reference_json, "kujiale_0003_scene_reference.json"),
        (manifest_path, "coverage_manifest.json"),
    ):
        links[name] = hardlink_or_copy(source, package_dir / name)

    snapshot_info: dict[str, dict[str, object]] = {}
    with np.load(snapshot, allow_pickle=False) as arrays:
        missing = [key for key in REQUIRED_ARRAYS if key not in arrays.files]
        if missing:
            raise RuntimeError(
                f"{milestone_percent}% snapshot is missing arrays: "
                + ", ".join(missing)
            )
        state = np.asarray(arrays["voxel_occupancy_state_zyx"])
        log_odds = np.asarray(arrays["voxel_occupancy_log_odds_zyx"])
        sensor_range = np.asarray(arrays["voxel_sensor_range_count_zyx"])
        z_centers = np.asarray(arrays["voxel_occupancy_z_centers_m"])
        if state.ndim != 3 or log_odds.shape != state.shape or sensor_range.shape != state.shape:
            raise RuntimeError("full 3D voxel arrays have inconsistent shapes")
        if z_centers.shape != (state.shape[0],):
            raise RuntimeError("voxel z centers do not match the voxel grid")
        coverage = float(np.asarray(arrays["roomseg_eval_coverage_ratio"]).reshape(()))
        if coverage < float(milestone_percent) / 100.0:
            raise RuntimeError(f"saved milestone coverage is only {coverage:.6f}")
        for key in REQUIRED_ARRAYS:
            value = np.asarray(arrays[key])
            snapshot_info[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        snapshot_info["all_array_keys"] = {"values": sorted(arrays.files)}
        Image.fromarray(np.asarray(arrays["demo_rgb"], dtype=np.uint8), mode="RGB").save(
            package_dir / f"{file_prefix}_rgb.png"
        )
        save_depth_preview(
            arrays["demo_depth_m"], package_dir / f"{file_prefix}_depth.png"
        )
        save_seed_preview(
            arrays, package_dir / f"{file_prefix}_seed_sources.png"
        )

    step = int(event["step"])
    frames = sorted((run_root / "voxroom_viz_frames").glob("voxroom_step_*.jpg"))
    if frames:
        def frame_step(path: Path) -> int:
            return int(path.stem.rsplit("_", 1)[-1])

        frame = min(frames, key=lambda path: abs(frame_step(path) - step))
        panel_name = f"{file_prefix}_full_panel.jpg"
        links[panel_name] = hardlink_or_copy(
            frame, package_dir / panel_name
        )

    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else run_root / "checkpoint_vertical_epoch14.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    inventory = {
        "schema": "kujiale_0003_paper_visualization_milestone_v1",
        "scene_id": "kujiale_0003",
        "requested_milestone_percent": milestone_percent,
        "event": event,
        "coverage_ratio": float(event["coverage_ratio"]),
        "step": step,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "snapshot": str(snapshot),
        "snapshot_sha256": sha256(snapshot),
        "package_materialization": links,
        "arrays": snapshot_info,
        "seed_source_id_legend": {
            "0": "not_raw_seed",
            "1": "voxroom_only",
            "2": "tvars_vertical_only",
            "3": "voxroom_and_tvars_vertical",
        },
    }
    (package_dir / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (package_dir / "README.md").write_text(
        f"# Kujiale 0003 at {milestone_percent}% exploration\n\n"
        "This directory is the paper-figure package captured at the first step whose "
        f"fixed-reference exploration coverage reached {milestone_percent}%. "
        "The NPZ is the complete "
        "VoxRoom state, including the 3D voxel grid, Vertical-Free/Nav projections, "
        "separate VoxRoom and TVARS raw-seed sources, neural keep/reject outputs, "
        "separators, room labels, RGB-D observation, pose, paths, and coverage masks.\n\n"
        "`inventory.json` records every required array shape and dtype, the exact "
        "coverage/step, source legend, and SHA-256 hashes. The JPEG/PNG files are "
        "convenience previews; use the NPZ arrays for publication rendering.\n",
        encoding="utf-8",
    )
    print(f"[{milestone_label}-package] complete: {package_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
