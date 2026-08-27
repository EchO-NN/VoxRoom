from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from voxroom_online.isaac_runtime.sensors.camera_geometry import pose_world_to_matrix
from voxroom_online.real_runtime.geometry import RigidTransform
from voxroom_online.real_runtime.live_visualization import (
    LiveVisualizationConfig,
    ZedDepthNavVisualizer,
)
from voxroom_online.real_runtime.mapper_runtime import (
    VALID_MODES,
    build_mapper,
    build_room_segmenter,
    ensure_full_voxel_dependencies,
    load_yaml,
    save_runtime_artifacts,
    update_room_segmentation,
)
from voxroom_online.real_runtime.zed_frame_pump import ZedFramePump
from voxroom_online.real_runtime.zed2i_source import Zed2iSource, ZedFrame


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VoxRoom directly on a ZED 2i depth stream and its visual-inertial pose."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "voxroom_online.yaml"))
    parser.add_argument("--real-config", default=str(REPO_ROOT / "configs" / "zed2i_real.yaml"))
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-seconds", type=float, default=None, help="Stop after this many seconds; 0 means unlimited.")
    parser.add_argument("--svo", default=None, help="Replay an SVO/SVO2 file instead of opening a live camera.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs/dependencies without opening the camera.")
    visualization = parser.add_mutually_exclusive_group()
    visualization.add_argument(
        "--visualization",
        dest="visualization",
        action="store_true",
        help="Show the live depth/NavFree popup, overriding the real-sensor config.",
    )
    visualization.add_argument(
        "--no-visualization",
        dest="visualization",
        action="store_false",
        help="Disable the live depth/NavFree popup, overriding the real-sensor config.",
    )
    parser.set_defaults(visualization=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        "\n"
        "WARNING: legacy_local_mapping_experiment\n"
        "This PyZED/nvblox entry has no RTAB-Map global loop closure or pose-graph "
        "optimization.\n"
        "It must not be used as the production global-map path. Use "
        "/home/echo/isaac_ros-dev/scripts/run_zed2i_rtabmap.sh instead.\n",
        file=sys.stderr,
        flush=True,
    )
    base_cfg = load_yaml(args.config)
    real_cfg = load_yaml(args.real_config)
    runtime_cfg = dict(real_cfg.get("runtime", {}) or {})
    zed_cfg = dict(real_cfg.get("zed", {}) or {})
    real_mapping_cfg = dict(real_cfg.get("mapping", {}) or {})
    extrinsics = RigidTransform.from_mapping(dict(real_cfg.get("extrinsics", {}) or {}))

    mode = str(args.mode or runtime_cfg.get("mode", "smoke_2d")).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}; choose from {sorted(VALID_MODES)}")
    pose_mode = str(runtime_cfg.get("pose_mode", "se3")).strip().lower()
    if pose_mode != "se3":
        raise ValueError("ZED2i real mapping requires runtime.pose_mode=se3")
    tracking_cfg = dict(zed_cfg.get("tracking", {}) or {})
    if bool(tracking_cfg.get("enable_2d_ground_mode", False)):
        raise ValueError("6DoF mapping requires zed.tracking.enable_2d_ground_mode=false")
    set_floor_as_origin = bool(tracking_cfg.get("set_floor_as_origin", False))
    validated_floor_alignment = bool(dict(tracking_cfg.get("floor_alignment", {}) or {}).get("enabled", False))
    floor_z_m = float(real_mapping_cfg.get("floor_z_m", 0.0))
    if not np.isfinite(floor_z_m):
        raise ValueError("mapping.floor_z_m must be finite")
    if args.svo:
        zed_cfg["svo_path"] = str(Path(args.svo).expanduser().resolve())
    output_dir = Path(args.output_dir or runtime_cfg.get("output_dir", "outputs/zed2i_real")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roomseg_every_s = max(0.1, float(runtime_cfg.get("roomseg_every_s", 10.0)))
    mapping_hz = max(0.1, float(runtime_cfg.get("mapping_hz", 5.0)))
    capture_cfg = dict(runtime_cfg.get("capture", {}) or {})
    minimum_capture_fps = max(0.0, float(capture_cfg.get("minimum_tracking_fps", 45.0)))
    capture_fps_window_s = max(0.1, float(capture_cfg.get("fps_window_s", 3.0)))
    low_fps_timeout_s = max(0.1, float(capture_cfg.get("low_fps_timeout_s", 2.0)))
    capture_open_timeout_s = max(1.0, float(capture_cfg.get("open_timeout_s", 30.0)))
    visualization_cfg = LiveVisualizationConfig.from_mapping(runtime_cfg.get("visualization", {}))
    if args.visualization is not None:
        visualization_cfg.enabled = bool(args.visualization)
    if visualization_cfg.enabled and mode != "full_voxel" and not args.dry_run:
        raise ValueError("live NavFree visualization requires full_voxel mode; pass --no-visualization for smoke_2d")

    if mode == "full_voxel":
        ensure_full_voxel_dependencies()
    mapper = build_mapper(base_cfg, real_cfg, mode)

    summary = {
        "mode": mode,
        "pose_mode": "se3",
        "mapping_pose_reference": "zed_camera_relative_odometry",
        "diagnostic_pose_reference": "zed_world",
        "floor_z_m": floor_z_m,
        "floor_alignment": (
            "zed_validated_find_floor_plane"
            if validated_floor_alignment
            else ("zed_set_floor_as_origin" if set_floor_as_origin else "configured_floor_z_only")
        ),
        "output_dir": str(output_dir),
        "extrinsics": {
            "base_to_left_camera_xyz_m": list(extrinsics.xyz_m),
            "base_to_left_camera_rpy_deg": list(extrinsics.rpy_deg),
        },
        "map_size_m": float(mapper.size_m),
        "resolution_m": float(mapper.resolution_m),
        "voxel_resolution_xyz_m": (
            [
                float(mapper.resolution_m),
                float(mapper.resolution_m),
                float(mapper.voxel_grid_config.z_resolution_m),
            ]
            if mapper.voxel_grid_config.enabled
            else None
        ),
        "depth_range_m": [float(mapper.depth_min_m), float(mapper.depth_max_m)],
        "voxel_grid_enabled": bool(mapper.voxel_grid_config.enabled),
        "mapping_hz": mapping_hz,
        "roomseg_every_s": roomseg_every_s,
        "capture": {
            "architecture": "dedicated_zed_thread_latest_depth_frame",
            "minimum_tracking_fps": minimum_capture_fps,
            "fps_window_s": capture_fps_window_s,
            "low_fps_timeout_s": low_fps_timeout_s,
        },
        "visualization": visualization_cfg.to_dict(),
    }
    (output_dir / "resolved_run_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    snapshot_every_s = max(0.0, float(runtime_cfg.get("snapshot_every_s", 2.0)))
    include_voxel_state = bool(runtime_cfg.get("save_voxel_state", False))
    max_seconds = float(args.max_seconds if args.max_seconds is not None else runtime_cfg.get("max_seconds", 0.0) or 0.0)

    source = Zed2iSource(zed_cfg, extrinsics)
    frame_pump = ZedFramePump(
        source,
        depth_hz=mapping_hz,
        minimum_capture_fps=minimum_capture_fps,
        fps_window_s=capture_fps_window_s,
        low_fps_timeout_s=low_fps_timeout_s,
    )
    visualizer = ZedDepthNavVisualizer(visualization_cfg, output_dir)
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    trajectory_path = output_dir / "trajectory.csv"
    log_fields = [
        "step",
        "image_timestamp_ns",
        "host_timestamp_ns",
        "tracking_state",
        "odometry_status",
        "spatial_memory_status",
        "tracking_fusion_status",
        "pose_confidence",
        "mapped",
        "skip_reason",
        "base_x_m",
        "base_y_m",
        "base_z_m",
        "base_yaw_rad",
        "base_roll_deg",
        "base_pitch_deg",
        "base_yaw_deg",
        "camera_x_m",
        "camera_y_m",
        "camera_z_m",
        "camera_yaw_rad",
        "camera_roll_deg",
        "camera_pitch_deg",
        "camera_yaw_deg",
        "global_camera_x_m",
        "global_camera_y_m",
        "global_camera_z_m",
        "global_camera_yaw_rad",
        "global_camera_roll_deg",
        "global_camera_pitch_deg",
        "global_camera_yaw_deg",
        "imu_ax",
        "imu_ay",
        "imu_az",
        "imu_gx",
        "imu_gy",
        "imu_gz",
        "mapper_total_ms",
        "room_count",
        "capture_frames",
        "capture_fps_average",
        "capture_fps_recent",
        "depth_frames_dropped",
    ]

    segmenter = None
    room_labels: np.ndarray | None = None
    room_debug: dict[str, Any] = {}
    room_count = 0
    latest_depth: np.ndarray | None = None
    latest_intr = None
    latest_base_pose = None
    latest_camera_pose = None
    latest_base_transform = None
    latest_camera_transform = None
    step = 0
    mapped_frames = 0
    skipped_tracking = 0
    skipped_pose = 0
    roomseg_updates = 0
    start_monotonic = 0.0
    next_snapshot_at = float("inf")
    next_roomseg_at = 0.0
    next_status_at = 0.0

    try:
        frame_pump.start(timeout_s=capture_open_timeout_s)
        # Start experiment timing after camera/model initialization so
        # --max-seconds measures captured data rather than ZED startup time.
        start_monotonic = time.monotonic()
        next_snapshot_at = start_monotonic + snapshot_every_s if snapshot_every_s > 0 else float("inf")
        next_roomseg_at = start_monotonic + roomseg_every_s
        next_status_at = start_monotonic
        (output_dir / "zed_camera_info.json").write_text(
            json.dumps(source.camera_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[zed] {json.dumps(source.camera_info, ensure_ascii=False)}", flush=True)
        with trajectory_path.open("w", newline="", encoding="utf-8") as log_handle:
            writer = csv.DictWriter(log_handle, fieldnames=log_fields)
            writer.writeheader()
            while not stop:
                now = time.monotonic()
                if max_seconds > 0.0 and now - start_monotonic >= max_seconds:
                    break
                frame = frame_pump.get(timeout_s=0.1)
                if frame is None:
                    if frame_pump.finished:
                        frame_pump.raise_if_failed()
                        break
                    continue

                mapped = False
                ok, skip_reason = _frame_is_integrable(frame)
                if ok:
                    assert frame.depth_m is not None and frame.intrinsics is not None
                    assert frame.base_pose_world is not None and frame.camera_pose_world is not None
                    assert frame.base_transform_world is not None and frame.camera_transform_world is not None
                    mapping_base_pose = (
                        float(frame.base_pose_world[0]),
                        float(frame.base_pose_world[1]),
                        floor_z_m,
                        float(frame.base_pose_world[3]),
                    )
                    if mapped_frames == 0:
                        mapper.reset((float(frame.base_pose_world[0]), float(frame.base_pose_world[1])))
                        if mode == "full_voxel":
                            segmenter, warnings = build_room_segmenter(
                                base_cfg,
                                mapper=mapper,
                                repo_root=REPO_ROOT,
                            )
                            for warning in warnings:
                                print(f"[voxroom-warning] {warning}", flush=True)
                    mapper.update(
                        frame.depth_m,
                        frame.intrinsics,
                        mapping_base_pose,
                        frame.camera_transform_world,
                    )
                    step += 1
                    mapped_frames += 1
                    mapped = True
                    skip_reason = ""
                    latest_depth = frame.depth_m.copy()
                    latest_intr = frame.intrinsics
                    latest_base_pose = frame.base_pose_world
                    latest_camera_pose = frame.camera_pose_world
                    latest_base_transform = frame.base_transform_world.copy()
                    latest_camera_transform = frame.camera_transform_world.copy()

                    if segmenter is not None and now >= next_roomseg_at:
                        roomseg_started_at = time.monotonic()
                        while next_roomseg_at <= now:
                            next_roomseg_at += roomseg_every_s
                        room_labels, room_debug, room_count, nav_source = update_room_segmentation(
                            segmenter,
                            mapper,
                            step=step,
                        )
                        roomseg_updates += 1
                        roomseg_total_ms = (time.monotonic() - roomseg_started_at) * 1000.0
                        room_debug["real_runtime_navigation_mask_source"] = nav_source
                        room_debug["real_runtime_roomseg_interval_s"] = roomseg_every_s
                        room_debug["real_runtime_roomseg_update_index"] = roomseg_updates
                        room_debug["real_runtime_roomseg_scheduled_elapsed_s"] = now - start_monotonic
                        room_debug["real_runtime_roomseg_total_ms"] = roomseg_total_ms
                        print(
                            "[voxroom-roomseg] update=%d elapsed_s=%.3f step=%d total_ms=%.1f rooms=%d"
                            % (
                                roomseg_updates,
                                now - start_monotonic,
                                step,
                                roomseg_total_ms,
                                room_count,
                            ),
                            flush=True,
                        )
                    visualizer.update(
                        now=time.monotonic(),
                        step=step,
                        depth_m=frame.depth_m,
                        mapper=mapper,
                        base_pose=frame.base_pose_world,
                        tracking_state=frame.tracking_state,
                        room_count=room_count,
                        mapper_total_ms=float(
                            getattr(mapper, "last_timing_stats", {}).get("update_total_ms", 0.0) or 0.0
                        ),
                    )
                else:
                    if skip_reason.startswith("tracking"):
                        skipped_tracking += 1
                    elif skip_reason.startswith("pose"):
                        skipped_pose += 1

                capture_stats = frame_pump.stats()
                row = _trajectory_row(
                    frame,
                    step=step,
                    mapped=mapped,
                    skip_reason=skip_reason,
                    mapper_total_ms=(
                        float(getattr(mapper, "last_timing_stats", {}).get("update_total_ms", 0.0) or 0.0)
                        if mapped
                        else 0.0
                    ),
                    room_count=room_count,
                )
                row.update(
                    {
                        "capture_frames": capture_stats.capture_frames,
                        "capture_fps_average": capture_stats.capture_fps_average,
                        "capture_fps_recent": capture_stats.capture_fps_recent,
                        "depth_frames_dropped": capture_stats.depth_frames_dropped,
                    }
                )
                writer.writerow(row)
                log_handle.flush()

                if mapped and latest_intr is not None and now >= next_snapshot_at:
                    while next_snapshot_at <= now:
                        next_snapshot_at += snapshot_every_s
                    save_runtime_artifacts(
                        output_dir,
                        mapper=mapper,
                        step=step,
                        base_pose=latest_base_pose,
                        camera_pose=latest_camera_pose,
                        intrinsics=latest_intr,
                        room_labels=room_labels,
                        room_debug=room_debug,
                        latest_depth_m=latest_depth,
                        prefix="latest",
                        include_voxel_state=False,
                        base_transform_world=latest_base_transform,
                        camera_transform_world=latest_camera_transform,
                        floor_z_m=floor_z_m,
                    )

                if now >= next_status_at:
                    next_status_at = now + 1.0
                    pose_text = "unavailable"
                    if frame.base_pose_world is not None:
                        base_rpy = frame.base_rpy_deg or (float("nan"), float("nan"), float("nan"))
                        pose_text = "(%.2f, %.2f, %.2f; rpy %.1f/%.1f/%.1f deg)" % (
                            frame.base_pose_world[0],
                            frame.base_pose_world[1],
                            frame.base_pose_world[2],
                            base_rpy[0],
                            base_rpy[1],
                            base_rpy[2],
                        )
                    print(
                        "[voxroom-real] track=%s odom=%s memory=%s confidence=%s base=%s "
                        "capture_fps=%.1f/%.1f dropped_depth=%d mapped=%d skipped_track=%d skipped_pose=%d "
                        "map_ms=%.1f rooms=%d"
                        % (
                            frame.tracking_state,
                            frame.odometry_status,
                            frame.spatial_memory_status,
                            frame.pose_confidence,
                            pose_text,
                            capture_stats.capture_fps_recent,
                            capture_stats.capture_fps_average,
                            capture_stats.depth_frames_dropped,
                            mapped_frames,
                            skipped_tracking,
                            skipped_pose,
                            float(getattr(mapper, "last_timing_stats", {}).get("update_total_ms", 0.0) or 0.0),
                            room_count,
                        ),
                        flush=True,
                    )
    finally:
        visualizer.close()
        frame_pump.stop()
        frame_pump.join(timeout_s=10.0)
        capture_stats = frame_pump.stats()
        (output_dir / "capture_stats.json").write_text(
            json.dumps(capture_stats.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[zed-capture] {json.dumps(capture_stats.to_dict(), ensure_ascii=False)}", flush=True)

    frame_pump.raise_if_failed()

    if mapped_frames > 0 and latest_intr is not None:
        artifacts = save_runtime_artifacts(
            output_dir,
            mapper=mapper,
            step=step,
            base_pose=latest_base_pose,
            camera_pose=latest_camera_pose,
            intrinsics=latest_intr,
            room_labels=room_labels,
            room_debug=room_debug,
            latest_depth_m=latest_depth,
            prefix="final",
            include_voxel_state=include_voxel_state,
            base_transform_world=latest_base_transform,
            camera_transform_world=latest_camera_transform,
            floor_z_m=floor_z_m,
        )
        print(f"[voxroom-real] final artifacts: {json.dumps(artifacts, ensure_ascii=False)}", flush=True)
    else:
        print("[voxroom-real] no frame was integrated; inspect tracking state, depth, pose, and extrinsics.", flush=True)
        return 2
    return 0


def _frame_is_integrable(frame: ZedFrame) -> tuple[bool, str]:
    state = frame.tracking_state.strip().upper()
    if state != "OK":
        return False, f"tracking_{state.lower()}"
    if (
        frame.base_pose_world is None
        or frame.camera_pose_world is None
        or frame.base_transform_world is None
        or frame.camera_transform_world is None
    ):
        return False, "tracking_pose_unavailable"
    if frame.depth_m is None or frame.intrinsics is None:
        return False, "depth_unavailable"
    try:
        pose_world_to_matrix(frame.base_transform_world)
        pose_world_to_matrix(frame.camera_transform_world)
    except (TypeError, ValueError) as exc:
        return False, f"pose_invalid_{type(exc).__name__.lower()}"
    if not np.all(np.isfinite(np.asarray(frame.base_pose_world, dtype=np.float64))):
        return False, "pose_planar_non_finite"
    if not np.all(np.isfinite(np.asarray(frame.camera_pose_world, dtype=np.float64))):
        return False, "pose_planar_non_finite"
    return True, ""


def _trajectory_row(
    frame: ZedFrame,
    *,
    step: int,
    mapped: bool,
    skip_reason: str,
    mapper_total_ms: float,
    room_count: int,
) -> dict[str, Any]:
    base = frame.base_pose_world or (None, None, None, None)
    camera = frame.camera_pose_world or (None, None, None, None)
    base_rpy = frame.base_rpy_deg or (None, None, None)
    rpy = frame.camera_rpy_deg or (None, None, None)
    global_camera = frame.global_camera_pose_world or (None, None, None, None)
    global_rpy = frame.global_camera_rpy_deg or (None, None, None)
    accel = frame.imu_linear_acceleration or (None, None, None)
    gyro = frame.imu_angular_velocity or (None, None, None)
    return {
        "step": int(step),
        "image_timestamp_ns": int(frame.image_timestamp_ns),
        "host_timestamp_ns": int(frame.host_timestamp_ns),
        "tracking_state": frame.tracking_state,
        "odometry_status": frame.odometry_status,
        "spatial_memory_status": frame.spatial_memory_status,
        "tracking_fusion_status": frame.tracking_fusion_status,
        "pose_confidence": frame.pose_confidence,
        "mapped": int(bool(mapped)),
        "skip_reason": skip_reason,
        "base_x_m": base[0],
        "base_y_m": base[1],
        "base_z_m": base[2],
        "base_yaw_rad": base[3],
        "base_roll_deg": base_rpy[0],
        "base_pitch_deg": base_rpy[1],
        "base_yaw_deg": base_rpy[2],
        "camera_x_m": camera[0],
        "camera_y_m": camera[1],
        "camera_z_m": camera[2],
        "camera_yaw_rad": camera[3],
        "camera_roll_deg": rpy[0],
        "camera_pitch_deg": rpy[1],
        "camera_yaw_deg": rpy[2],
        "global_camera_x_m": global_camera[0],
        "global_camera_y_m": global_camera[1],
        "global_camera_z_m": global_camera[2],
        "global_camera_yaw_rad": global_camera[3],
        "global_camera_roll_deg": global_rpy[0],
        "global_camera_pitch_deg": global_rpy[1],
        "global_camera_yaw_deg": global_rpy[2],
        "imu_ax": accel[0],
        "imu_ay": accel[1],
        "imu_az": accel[2],
        "imu_gx": gyro[0],
        "imu_gy": gyro[1],
        "imu_gz": gyro[2],
        "mapper_total_ms": float(mapper_total_ms),
        "room_count": int(room_count),
    }


if __name__ == "__main__":
    raise SystemExit(main())
