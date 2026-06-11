#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


UNKNOWN = np.array([232, 232, 232], dtype=np.uint8)
FREE = np.array([250, 250, 247], dtype=np.uint8)
OCCUPIED = np.array([80, 80, 80], dtype=np.uint8)
WALL = np.array([166, 53, 38], dtype=np.uint8)
DOOR_SEED = np.array([122, 33, 158], dtype=np.uint8)
TEXT = (24, 28, 38)
PANEL_BG = (255, 255, 255)
BORDER = (220, 220, 220)

ROOM_PALETTE = np.array(
    [
        [142, 202, 230],
        [255, 183, 77],
        [149, 213, 178],
        [244, 143, 177],
        [180, 160, 210],
        [233, 216, 166],
        [144, 190, 109],
        [248, 150, 30],
        [115, 171, 132],
        [231, 111, 81],
        [168, 218, 220],
        [203, 153, 201],
    ],
    dtype=np.uint8,
)


def step_from_path(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def bool_array(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], shape: tuple[int, int] | None = None) -> np.ndarray:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key], dtype=bool)
    if shape is None:
        raise KeyError("none of %s found" % (keys,))
    return np.zeros(shape, dtype=bool)


def int_array(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], shape: tuple[int, int]) -> np.ndarray:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key], dtype=np.int32)
    return np.zeros(shape, dtype=np.int32)


def scalar(data: np.lib.npyio.NpzFile, key: str, default: float) -> float:
    if key not in data.files:
        return float(default)
    try:
        return float(np.asarray(data[key]).reshape(-1)[0])
    except Exception:
        return float(default)


def fit_crop(mask: np.ndarray, shape: tuple[int, int], pad: int = 28) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return 0, int(shape[0]), 0, int(shape[1])
    r0 = max(0, int(rows.min()) - pad)
    r1 = min(int(shape[0]), int(rows.max()) + pad + 1)
    c0 = max(0, int(cols.min()) - pad)
    c1 = min(int(shape[1]), int(cols.max()) + pad + 1)
    return r0, r1, c0, c1


def crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = box
    return arr[r0:r1, c0:c1]


def paste_fit(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    target_w = max(1, x1 - x0)
    target_h = max(1, y1 - y0)
    image = image.convert("RGB")
    image.thumbnail((target_w, target_h), Image.Resampling.NEAREST)
    px = x0 + (target_w - image.width) // 2
    py = y0 + (target_h - image.height) // 2
    canvas.paste(image, (px, py))


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill=PANEL_BG, outline=BORDER, width=1)
    draw.text((x0 + 16, y0 + 12), title, fill=TEXT, font=font)
    return x0 + 12, y0 + 48, x1 - 12, y1 - 12


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_depth(depth: np.ndarray) -> Image.Image:
    arr = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    rgb = np.zeros((*arr.shape[:2], 3), dtype=np.uint8)
    rgb[:] = UNKNOWN
    if np.any(valid):
        lo, hi = np.percentile(arr[valid], [2.0, 98.0])
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        gray = (255.0 - norm * 210.0).astype(np.uint8)
        rgb[valid] = np.repeat(gray[..., None], 3, axis=2)[valid]
    return Image.fromarray(rgb)


def render_nav_mask(data: np.lib.npyio.NpzFile, crop_box: tuple[int, int, int, int]) -> Image.Image:
    nav_free = bool_array(data, ("voxel_nav_free_xy", "observed_free_mask", "navigation_free_room_domain"))
    shape = nav_free.shape
    nav_occ = bool_array(data, ("voxel_nav_occupied_xy", "obstacle_mask", "occupancy_map"), shape)
    labels = int_array(data, ("final_room_label_map", "voxel_final_room_label_map"), shape)
    resolution_m = scalar(data, "map_resolution_m", 0.05)
    min_room_cells = max(1, int(math.ceil(0.3 / max(resolution_m * resolution_m, 1e-12))))
    image = np.zeros((*shape, 3), dtype=np.uint8)
    image[:] = UNKNOWN
    image[nav_free] = FREE
    image[nav_occ] = OCCUPIED
    label_ids = [
        int(v)
        for v in np.unique(labels)
        if int(v) > 0 and int(np.count_nonzero(labels == int(v))) >= min_room_cells
    ]
    for idx, label in enumerate(label_ids):
        mask = labels == label
        color = ROOM_PALETTE[idx % len(ROOM_PALETTE)]
        image[mask] = (0.58 * image[mask].astype(np.float32) + 0.42 * color.astype(np.float32)).astype(np.uint8)
    image[nav_occ] = OCCUPIED
    _paint_path(image, _path_array(data, "demo_full_path_rc"), crop_box, (135, 185, 255), radius=1, max_cells=1800)
    _paint_path(image, _path_array(data, "demo_current_path_rc"), crop_box, (0, 102, 255), radius=2, max_cells=900)
    frontier = bool_array(data, ("frontier_map",), shape)
    rr, cc = np.nonzero(frontier)
    if rr.size:
        _paint_path(image, np.stack([rr, cc], axis=1), crop_box, (0, 185, 210), radius=1, max_cells=1400)
    selected = _point_array(data, "selected_frontier_center_rc")
    if selected is not None:
        _paint_path(image, selected.reshape(1, 2), crop_box, (255, 220, 40), radius=5, max_cells=1)
    agent = _point_array(data, "agent_rc")
    if agent is not None:
        _paint_path(image, agent.reshape(1, 2), crop_box, (235, 50, 55), radius=4, max_cells=1)
    return Image.fromarray(crop(image, crop_box))


def _path_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data.files:
        return np.zeros((0, 2), dtype=np.int32)
    arr = np.asarray(data[key], dtype=np.int32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return arr.reshape((-1, 2))


def _point_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key], dtype=np.int32).reshape(-1)
    if arr.size < 2:
        return None
    return arr[:2].astype(np.int32)


def _paint_path(
    image: np.ndarray,
    cells: np.ndarray,
    crop_box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    *,
    radius: int,
    max_cells: int,
) -> None:
    r0, r1, c0, c1 = crop_box
    arr = np.asarray(cells, dtype=np.int32).reshape((-1, 2)) if np.asarray(cells).size else np.zeros((0, 2), dtype=np.int32)
    if arr.size == 0:
        return
    stride = max(1, int(arr.shape[0]) // max(1, int(max_cells)))
    h, w = image.shape[:2]
    for r, c in arr[::stride][:max_cells]:
        r, c = int(r), int(c)
        if not (int(r0) <= r < int(r1) and int(c0) <= c < int(c1)):
            continue
        rr0 = max(0, r - int(radius))
        rr1 = min(h, r + int(radius) + 1)
        cc0 = max(0, c - int(radius))
        cc1 = min(w, c + int(radius) + 1)
        image[rr0:rr1, cc0:cc1] = np.asarray(color, dtype=np.uint8)


def render_vertical_free(data: np.lib.npyio.NpzFile, crop_box: tuple[int, int, int, int]) -> Image.Image:
    vertical_free = bool_array(data, ("voxel_vertical_free_xy", "vertical_free_room_domain", "height_profile_vertical_free_xy"))
    shape = vertical_free.shape
    wall = bool_array(data, ("voxel_wall_xy", "voxel_wall_after_step1_map", "structural_wall_clean", "obstacle_mask"), shape)
    seed = (
        bool_array(data, ("voxel_door_raw_seed_mask",), shape)
        | bool_array(data, ("voxel_door_seed_mask",), shape)
        | bool_array(data, ("voxel_door_seed_line_primitive_mask",), shape)
    )
    image = np.zeros((*shape, 3), dtype=np.uint8)
    image[:] = UNKNOWN
    image[vertical_free] = FREE
    image[wall] = WALL
    image[seed] = DOOR_SEED
    return Image.fromarray(crop(image, crop_box))


def render_frame(path: Path, output_path: Path, *, canvas_size: tuple[int, int]) -> None:
    with np.load(path, allow_pickle=True) as data:
        nav_free = bool_array(data, ("voxel_nav_free_xy", "observed_free_mask", "navigation_free_room_domain"))
        shape = nav_free.shape
        content = nav_free.copy()
        for keys in (
            ("voxel_nav_occupied_xy", "obstacle_mask", "occupancy_map"),
            ("voxel_vertical_free_xy", "vertical_free_room_domain", "height_profile_vertical_free_xy"),
            ("voxel_wall_xy", "voxel_wall_after_step1_map", "structural_wall_clean"),
            ("voxel_door_raw_seed_mask", "voxel_door_seed_mask", "voxel_door_seed_line_primitive_mask"),
        ):
            content |= bool_array(data, keys, shape)
        labels = int_array(data, ("final_room_label_map", "voxel_final_room_label_map"), shape)
        content |= labels > 0
        crop_box = fit_crop(content, shape, pad=34)

        canvas = Image.new("RGB", canvas_size, (248, 248, 246))
        draw = ImageDraw.Draw(canvas)
        title_font = font(22, bold=True)
        step_font = font(26, bold=True)

        margin = 24
        gutter = 20
        header = 52
        width, height = canvas_size
        panel_w = (width - 2 * margin - gutter) // 2
        panel_h = (height - header - 2 * margin - gutter) // 2
        draw.text((margin, 15), "Kujiale 0003 | Step %d" % step_from_path(path), fill=TEXT, font=step_font)

        boxes = [
            (margin, header, margin + panel_w, header + panel_h),
            (margin, header + panel_h + gutter, margin + panel_w, header + 2 * panel_h + gutter),
            (margin + panel_w + gutter, header, margin + 2 * panel_w + gutter, header + panel_h),
            (margin + panel_w + gutter, header + panel_h + gutter, margin + 2 * panel_w + gutter, header + 2 * panel_h + gutter),
        ]
        rgb_box = panel(draw, boxes[0], "RGB", title_font)
        depth_box = panel(draw, boxes[1], "Depth", title_font)
        nav_box = panel(draw, boxes[2], "Navigation Map + Room Mask", title_font)
        vfree_box = panel(draw, boxes[3], "Vertical Free Map", title_font)

        if "demo_rgb" in data.files:
            rgb = Image.fromarray(np.asarray(data["demo_rgb"], dtype=np.uint8))
        else:
            rgb = Image.new("RGB", (640, 360), UNKNOWN.tolist())
        paste_fit(canvas, rgb, rgb_box)

        if "demo_depth_m" in data.files:
            depth_img = render_depth(np.asarray(data["demo_depth_m"], dtype=np.float32))
        else:
            depth_img = Image.new("RGB", (640, 360), UNKNOWN.tolist())
        paste_fit(canvas, depth_img, depth_box)

        paste_fit(canvas, render_nav_mask(data, crop_box), nav_box)
        paste_fit(canvas, render_vertical_free(data, crop_box), vfree_box)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_ffmpeg(pattern: Path, output: Path, *, fps: int, extra: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(pattern),
        *extra,
        str(output),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render README 2x2 demo video from VoxRoom online snapshots.")
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output-mp4", required=True, type=Path)
    parser.add_argument("--output-gif", required=True, type=Path)
    parser.add_argument("--poster", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    paths = sorted(args.snapshot_dir.glob("roomseg_step_*.npz"), key=step_from_path)
    if not paths:
        raise SystemExit("no roomseg_step_*.npz files under %s" % args.snapshot_dir)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to encode README demo assets")

    with tempfile.TemporaryDirectory(prefix="voxroom_readme_demo_") as tmp:
        frame_dir = Path(tmp)
        for idx, path in enumerate(paths):
            render_frame(path, frame_dir / ("frame_%06d.png" % idx), canvas_size=(int(args.width), int(args.height)))
        poster_src = frame_dir / ("frame_%06d.png" % (len(paths) - 1))
        args.poster.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(poster_src, args.poster)
        run_ffmpeg(
            frame_dir / "frame_%06d.png",
            args.output_mp4,
            fps=int(args.fps),
            extra=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        )
        run_ffmpeg(
            frame_dir / "frame_%06d.png",
            args.output_gif,
            fps=max(1, min(int(args.fps), 6)),
            extra=["-vf", "fps=%d,scale=%d:-1:flags=lanczos" % (max(1, min(int(args.fps), 6)), int(args.width))],
        )
    print("[readme-demo] frames=%d mp4=%s gif=%s poster=%s" % (len(paths), args.output_mp4, args.output_gif, args.poster), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
