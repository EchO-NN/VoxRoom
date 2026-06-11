from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from voxroom_online.isaac_runtime.mapping.roomseg_ray_valid_wall import build_ray_valid_wall_inference


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay roomseg ray-valid wall inference from a saved debug NPZ.")
    parser.add_argument("--input", required=True, help="roomseg layer dump .npz")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--enable-ray-valid-wall-inference", action="store_true", default=False)
    parser.add_argument("--terminal-wall-splat-radius-cells", type=int, default=1)
    parser.add_argument("--min-terminal-wall-count", type=int, default=1)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=False)
    shape = _infer_shape(data)
    vertical_free = _bool_array(data, "vertical_free_room_domain", shape)
    vertical_occupied = _bool_array(data, "vertical_occupied_0p2_2p0", shape)
    vertical_observed = _bool_array(data, "vertical_observed_0p2_2p0", shape)
    terminal_count = _array(data, "roomseg_terminal_wall_count", shape, np.uint16)
    terminal_splat = _array(data, "roomseg_terminal_wall_splat", shape, np.uint8)
    ray_covered = _array(data, "roomseg_ray_covered_count", shape, np.uint16)
    height_min = _array(data, "roomseg_terminal_wall_height_min", shape, np.float32, fill=np.inf)
    height_max = _array(data, "roomseg_terminal_wall_height_max", shape, np.float32, fill=-np.inf)
    depth_min = _array(data, "roomseg_terminal_wall_depth_min", shape, np.float32, fill=np.inf)

    result = build_ray_valid_wall_inference(
        vertical_free=vertical_free,
        vertical_occupied=vertical_occupied,
        vertical_observed=vertical_observed,
        terminal_wall_count=terminal_count,
        terminal_wall_splat=terminal_splat if np.any(terminal_splat) else None,
        ray_covered_count=ray_covered,
        terminal_wall_height_min=height_min,
        terminal_wall_height_max=height_max,
        terminal_wall_depth_min=depth_min,
        config={
            "enabled": bool(args.enable_ray_valid_wall_inference),
            "min_terminal_wall_count": int(args.min_terminal_wall_count),
            "terminal_wall_splat_radius_cells": int(args.terminal_wall_splat_radius_cells),
            "require_no_vertical_free": True,
            "strict_no_navigation_obstacle_overlay": True,
            "strict_no_navigation_free_overlay": True,
        },
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "vertical_free_room_domain": vertical_free,
        "vertical_occupied_0p2_2p0": vertical_occupied,
        "roomseg_terminal_wall_count": terminal_count,
        "roomseg_terminal_wall_splat": np.asarray(result["roomseg_terminal_wall_splat"], dtype=bool),
        "ray_valid_wall_inference": np.asarray(result["ray_valid_wall_inference"], dtype=bool),
        "initial_roomseg_free_after_ray_wall": np.asarray(result["initial_roomseg_free_after_ray_wall"], dtype=bool),
        "initial_roomseg_occupied_after_ray_wall": np.asarray(result["initial_roomseg_occupied_after_ray_wall"], dtype=bool),
        "initial_roomseg_unknown_after_ray_wall": np.asarray(result["initial_roomseg_unknown_after_ray_wall"], dtype=bool),
        "unknown_before_ray_wall": np.asarray(result["unknown_before_ray_wall"], dtype=bool),
        "unknown_after_ray_wall": np.asarray(result["unknown_after_ray_wall"], dtype=bool),
        "unknown_removed_by_ray_wall": np.asarray(result["unknown_removed_by_ray_wall"], dtype=bool),
    }
    np.savez_compressed(out_dir / "replayed_roomseg_input.npz", **arrays)
    summary = dict(result.get("debug", {}) or {})
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _render_grid(arrays).save(out_dir / "replayed_roomseg_input.layers.png")
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


def _infer_shape(data: np.lib.npyio.NpzFile) -> tuple[int, int]:
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError("input npz does not contain 2D arrays")


def _array(data: np.lib.npyio.NpzFile, key: str, shape: tuple[int, int], dtype, fill=0):
    if key not in data.files:
        return np.full(shape, fill, dtype=dtype)
    arr = np.asarray(data[key], dtype=dtype)
    return arr if arr.shape == shape else np.full(shape, fill, dtype=dtype)


def _bool_array(data: np.lib.npyio.NpzFile, key: str, shape: tuple[int, int]) -> np.ndarray:
    return _array(data, key, shape, bool, fill=False).astype(bool)


def _render_grid(arrays: dict[str, np.ndarray]) -> Image.Image:
    thumbs = [_render_layer(name, arr) for name, arr in arrays.items()]
    tw, th = thumbs[0].size
    cols = 3
    rows = int(np.ceil(len(thumbs) / float(cols)))
    out = Image.new("RGB", (tw * cols, th * rows), (15, 17, 21))
    for idx, img in enumerate(thumbs):
        out.paste(img, ((idx % cols) * tw, (idx // cols) * th))
    return out


def _render_layer(name: str, arr: np.ndarray) -> Image.Image:
    data = np.asarray(arr)
    if data.dtype.kind in {"i", "u"} and np.max(data) > 1:
        norm = data.astype(np.float32)
        norm = norm / max(1.0, float(np.max(norm)))
        rgb = np.repeat((norm * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    else:
        rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
        rgb[np.asarray(data, dtype=bool)] = (225, 225, 225)
    h, w = rgb.shape[:2]
    scale = max(1, min(8, int(220 / max(h, w, 1))))
    image = Image.fromarray(rgb).resize((w * scale, h * scale), Image.Resampling.NEAREST)
    out = Image.new("RGB", (image.width, image.height + 18), (15, 17, 21))
    out.paste(image, (0, 18))
    ImageDraw.Draw(out).text((4, 3), name[:80], fill=(245, 245, 245), font=ImageFont.load_default())
    return out


if __name__ == "__main__":
    raise SystemExit(main())
