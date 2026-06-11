#!/usr/bin/env python3
"""Open a USD stage in Isaac Sim and keep the GUI alive.

With ``--depth-camera`` this opens the stage through the same IsaacSimServer
used by closed-loop runs, attaches the robot RGB-D camera, saves an optional
RGB/depth frame, and keeps the GUI viewport on that camera.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_episode(path: Path, index: int) -> dict:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"episode file is empty: {path}")
    if index < 0 or index >= len(rows):
        raise IndexError(f"episode index {index} out of range for {path} with {len(rows)} rows")
    return rows[index]


def _resolve_scene_and_spawn(args) -> Tuple[Path, Tuple[float, float, float, float], Optional[dict]]:
    episode = None
    scene_usd = args.scene_usd
    spawn = tuple(args.spawn) if args.spawn is not None else None

    if args.episode_file:
        episode_path = Path(args.episode_file).expanduser().resolve()
        if not episode_path.exists():
            raise FileNotFoundError(str(episode_path))
        episode = _read_episode(episode_path, int(args.episode_index))
        if not scene_usd:
            scene_usd = episode.get("usd_path")
        if spawn is None:
            start_pose = episode.get("start_pose_world")
            if isinstance(start_pose, (list, tuple)) and len(start_pose) >= 4:
                spawn = tuple(float(v) for v in start_pose[:4])

    if not scene_usd:
        raise ValueError("provide --scene-usd or --episode-file")
    if spawn is None:
        spawn = (0.0, 0.0, 0.05, 0.0)

    usd_path = Path(scene_usd).expanduser().resolve()
    if not usd_path.exists():
        raise FileNotFoundError(str(usd_path))
    return usd_path, tuple(float(v) for v in spawn), episode


def _save_depth_outputs(obs: dict, save_frame: str) -> None:
    from PIL import Image

    out = Path(save_frame).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(obs["rgb"])
    Image.fromarray(rgb).save(out)

    depth = np.asarray(obs["depth"], dtype=np.float32)
    np.save(out.with_name(out.stem + "_depth_m.npy"), depth)
    finite = np.isfinite(depth) & (depth > 0.0)
    if np.any(finite):
        finite_depth = depth[finite]
        vis_max = max(float(np.percentile(finite_depth, 99.0)), 1e-6)
        depth_vis = np.clip(np.where(finite, depth, 0.0) / vis_max * 255.0, 0.0, 255.0).astype(np.uint8)
        depth_mm = np.clip(np.where(finite, depth, 0.0) * 1000.0, 0.0, 65535.0).astype(np.uint16)
    else:
        depth_vis = np.zeros(depth.shape, dtype=np.uint8)
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    Image.fromarray(depth_vis).save(out.with_name(out.stem + "_depth_vis.png"))
    Image.fromarray(depth_mm).save(out.with_name(out.stem + "_depth_mm.png"))

    near = np.asarray(obs.get("nearfield_depth", []), dtype=np.float32)
    if obs.get("has_nearfield_depth") and near.ndim == 2 and near.size:
        np.save(out.with_name(out.stem + "_nearfield_depth_m.npy"), near)
        finite_near = np.isfinite(near) & (near > 0.0)
        if np.any(finite_near):
            near_max = max(float(np.percentile(near[finite_near], 99.0)), 1e-6)
            near_vis = np.clip(np.where(finite_near, near, 0.0) / near_max * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            near_vis = np.zeros(near.shape, dtype=np.uint8)
        Image.fromarray(near_vis).save(out.with_name(out.stem + "_nearfield_depth_vis.png"))


class LiveDepthViewer:
    def __init__(self, width: int, height: int, max_depth_m: float = 10.0) -> None:
        import omni.ui as ui

        self.width = int(width)
        self.height = int(height)
        self.max_depth_m = max(0.1, float(max_depth_m))
        self._ui = ui
        self._provider = ui.ByteImageProvider()
        self._window = ui.Window("VoxRoom Depth Camera", width=max(360, self.width), height=max(300, self.height + 42))
        self._label = None
        with self._window.frame:
            with ui.VStack(spacing=4):
                self._label = ui.Label("depth: waiting for frame")
                ui.ImageWithProvider(self._provider)

    def update(self, depth: np.ndarray, source: str = "") -> None:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            return
        finite = np.isfinite(depth) & (depth > 0.0)
        rgba = np.zeros((depth.shape[0], depth.shape[1], 4), dtype=np.uint8)
        if np.any(finite):
            clipped = np.clip(depth, 0.0, self.max_depth_m)
            gray = np.zeros(depth.shape, dtype=np.uint8)
            gray[finite] = np.clip((1.0 - clipped[finite] / self.max_depth_m) * 255.0, 0.0, 255.0).astype(np.uint8)
            rgba[:, :, 0] = gray
            rgba[:, :, 1] = gray
            rgba[:, :, 2] = gray
            rgba[:, :, 3] = 255
            finite_depth = depth[finite]
            if self._label is not None:
                self._label.text = (
                    "depth %s | min=%.2fm max=%.2fm display=0..%.1fm"
                    % (source or "frame", float(np.min(finite_depth)), float(np.max(finite_depth)), self.max_depth_m)
                )
        else:
            rgba[:, :, 3] = 255
            if self._label is not None:
                self._label.text = "depth %s | no finite depth" % (source or "frame")
        self._provider.set_bytes_data(bytearray(rgba.tobytes()), [int(depth.shape[1]), int(depth.shape[0])])


def _open_with_depth_camera(args) -> int:
    from voxroom_online.isaac_runtime.env.isaac_process import IsaacSimServer

    usd_path, spawn, episode = _resolve_scene_and_spawn(args)
    server = IsaacSimServer(
        headless=args.headless,
        width=args.width,
        height=args.height,
        verbose=args.verbose,
        camera_hfov_deg=args.camera_hfov_deg,
        mast_height_m=args.camera_mast_height_m,
        forward_offset_m=args.camera_forward_offset_m,
        camera_pitch_deg=args.camera_pitch_deg,
        camera_near_m=args.camera_near_m,
        camera_far_m=args.camera_far_m,
        enable_depth=True,
        camera_annotator_device=args.camera_annotator_device,
        enable_nearfield_depth=args.nearfield_depth,
        nearfield_width=args.nearfield_width,
        nearfield_height=args.nearfield_height,
        nearfield_hfov_deg=args.nearfield_hfov_deg,
        nearfield_height_m=args.nearfield_height_m,
        nearfield_near_m=args.nearfield_near_m,
        nearfield_far_m=args.nearfield_far_m,
    )
    try:
        if episode is not None:
            print(
                "[open-isaac-stage] episode %s scene %s"
                % (episode.get("episode_id", "<unknown>"), episode.get("scene_id", "<unknown>")),
                flush=True,
            )
        print(f"[open-isaac-stage] opening {usd_path}", flush=True)
        print(f"[open-isaac-stage] spawn {spawn}", flush=True)
        obs = server.reset_episode(str(usd_path), spawn, rgb_device="cpu")
        print("[open-isaac-stage] RGB-D camera ready at %s" % server.camera_prim_path, flush=True)
        print(
            {
                "pose_world": obs["pose_world"],
                "camera_pose_world": obs["camera_pose_world"],
                "depth_source": obs.get("depth_source"),
                "nearfield_depth_source": obs.get("nearfield_depth_source"),
            },
            flush=True,
        )
        if args.save_frame:
            _save_depth_outputs(obs, args.save_frame)
            print(f"[open-isaac-stage] saved RGB/depth outputs under {Path(args.save_frame).parent}", flush=True)

        depth_viewer = None
        if args.depth_viewer and not args.headless:
            depth_viewer = LiveDepthViewer(args.width, args.height, max_depth_m=args.depth_viewer_max_m)
            depth_viewer.update(obs["depth"], obs.get("depth_source", ""))

        if args.keep_open:
            print("[open-isaac-stage] close the Isaac window to exit", flush=True)
            update_idx = 0
            viewer_stride = max(1, int(args.depth_viewer_every_updates))
            while server.app is not None and server.app.is_running():
                server.app.update()
                if depth_viewer is not None:
                    update_idx += 1
                    if update_idx % viewer_stride == 0:
                        obs = server.get_observation(read_rgb=False, read_depth=True, rgb_device="cpu")
                        depth_viewer.update(obs["depth"], obs.get("depth_source", ""))
                time.sleep(0.01)
    finally:
        server.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-usd", default=None)
    parser.add_argument("--episode-file", default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--spawn", nargs=4, type=float, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth-camera", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth-viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth-viewer-max-m", type=float, default=10.0)
    parser.add_argument("--depth-viewer-every-updates", type=int, default=2)
    parser.add_argument("--keep-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-frame", default=None)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-hfov-deg", type=float, default=110.0)
    parser.add_argument("--camera-mast-height-m", type=float, default=1.35)
    parser.add_argument("--camera-forward-offset-m", type=float, default=0.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--camera-near-m", type=float, default=0.02)
    parser.add_argument("--camera-far-m", type=float, default=10.0)
    parser.add_argument("--camera-annotator-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--nearfield-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--nearfield-width", type=int, default=192)
    parser.add_argument("--nearfield-height", type=int, default=192)
    parser.add_argument("--nearfield-hfov-deg", type=float, default=115.0)
    parser.add_argument("--nearfield-height-m", type=float, default=1.15)
    parser.add_argument("--nearfield-near-m", type=float, default=0.02)
    parser.add_argument("--nearfield-far-m", type=float, default=1.8)
    args = parser.parse_args()

    if args.depth_camera:
        return _open_with_depth_camera(args)

    usd_path, _spawn, _episode = _resolve_scene_and_spawn(args)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": bool(args.headless)})

    try:
        from isaacsim.core.utils.stage import is_stage_loading, open_stage

        print(f"[open-isaac-stage] opening {usd_path}", flush=True)
        open_stage(str(usd_path))
        for _ in range(600):
            app.update()
            if not is_stage_loading():
                break
            time.sleep(0.02)
        print("[open-isaac-stage] stage ready", flush=True)
        if args.keep_open:
            print("[open-isaac-stage] close the Isaac window to exit", flush=True)
            while app.is_running():
                app.update()
                time.sleep(0.01)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
