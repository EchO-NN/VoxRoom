from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


LABEL_ALPHA = 0.64
WALL_COLOR = np.asarray((220, 30, 35), dtype=np.float32)
SPUR_COLOR = np.asarray((0, 210, 220), dtype=np.float32)
PLANNED_SEPARATOR_COLOR = np.asarray((235, 0, 220), dtype=np.float32)
ACTUAL_SEPARATOR_COLOR = np.asarray((110, 55, 185), dtype=np.float32)
CORNER_COLOR = np.asarray((255, 255, 255), dtype=np.uint8)
OCCUPIED_COLOR = np.asarray((72, 72, 72), dtype=np.uint8)
UNEXPLORED_COLOR = np.asarray((214, 214, 214), dtype=np.uint8)
FREE_COLOR = np.asarray((255, 255, 255), dtype=np.uint8)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize orthogonal Voronoi outputs on no-clearance navigation maps.")
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--postprocess-run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--map-resolution-m", type=float, default=0.05)
    parser.add_argument("--small-region-filter-m2", type=float, default=0.2)
    parser.add_argument("--preview-max-images", type=int, default=24)
    parser.add_argument("--tile-max-side", type=int, default=360)
    parser.add_argument("--hide-planned-separator", action="store_true")
    parser.add_argument("--draw-room-numbers", action="store_true")
    args = parser.parse_args(argv)

    source_root = Path(args.source_run_root)
    post_root = Path(args.postprocess_run_root)
    output_root = Path(args.output_root)
    overlay_root = output_root / "overlays"
    scene_sheet_root = output_root / "scene_sheets"
    overlay_root.mkdir(parents=True, exist_ok=True)
    scene_sheet_root.mkdir(parents=True, exist_ok=True)

    post_npzs = sorted(
        post_root.glob(f"*/baselines/{str(args.baseline_name)}/roomseg_snapshots/roomseg_step_*.npz")
    )
    rows: list[dict[str, Any]] = []
    scene_to_images: dict[str, list[Path]] = {}
    min_region_cells = int(math.ceil(float(args.small_region_filter_m2) / (float(args.map_resolution_m) ** 2)))

    for post_npz in post_npzs:
        scene = post_npz.parts[-5]
        source_npz = source_root / scene / "roomseg_snapshots" / post_npz.name
        out_png = overlay_root / scene / f"{post_npz.stem}.{args.baseline_name}_on_nav_no_clearance.png"
        row = visualize_one(
            source_npz=source_npz,
            post_npz=post_npz,
            output_png=out_png,
            min_region_cells=min_region_cells,
            map_resolution_m=float(args.map_resolution_m),
            hide_planned_separator=bool(args.hide_planned_separator),
            draw_room_numbers=bool(args.draw_room_numbers),
        )
        row["scene"] = scene
        row["snapshot"] = post_npz.name
        row["overlay_png"] = str(out_png)
        rows.append(row)
        scene_to_images.setdefault(scene, []).append(out_png)

    scene_sheet_paths: list[Path] = []
    for scene, paths in sorted(scene_to_images.items()):
        sheet_path = scene_sheet_root / f"{scene}.png"
        make_sheet(paths, sheet_path, max_side=int(args.tile_max_side))
        scene_sheet_paths.append(sheet_path)

    preview_candidates = [p for paths in scene_to_images.values() for p in paths]
    preview_sheet = output_root / "preview_sheet.png"
    make_sheet(preview_candidates[: int(args.preview_max_images)], preview_sheet, max_side=int(args.tile_max_side))

    write_csv(output_root / "summary.csv", rows)
    summary = {
        "status": "complete",
        "snapshot_count": len(rows),
        "scene_count": len(scene_to_images),
        "base_map": "original no-clearance voxel_nav_free_xy / voxel_nav_unknown_xy / voxel_nav_occupied_xy",
        "wall_line_visualized": "voxroom_wall_door_no_extension_no_mincc_despurred_mask",
        "spur_removed_visualized": "voxroom_wall_door_no_extension_no_mincc_spur_removed_mask",
        "separator_visualized": "orthogonal_corner_snap_planned_separator_mask full snapped line",
        "actual_separator_visualized": "orthogonal_corner_snap_separator_mask label-domain clipped line",
        "planned_separator_hidden": bool(args.hide_planned_separator),
        "room_numbers_drawn": bool(args.draw_room_numbers),
        "small_region_filter_m2_for_visualization": float(args.small_region_filter_m2),
        "native_room_avg": _avg(rows, "native_room_count"),
        "postprocess_room_avg_before_filter": _avg(rows, "postprocess_room_count_before_filter"),
        "postprocess_room_avg_after_filter": _avg(rows, "postprocess_room_count_after_filter"),
        "planned_separator_cells_total": int(sum(int(r["planned_separator_cells"]) for r in rows)),
        "actual_separator_cells_total": int(sum(int(r["actual_separator_cells"]) for r in rows)),
        "spur_removed_cells_total": int(sum(int(r["spur_removed_cells"]) for r in rows)),
        "preview_sheet": str(preview_sheet),
        "overlays_root": str(overlay_root),
        "scene_sheets_root": str(scene_sheet_root),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def visualize_one(
    *,
    source_npz: Path,
    post_npz: Path,
    output_png: Path,
    min_region_cells: int,
    map_resolution_m: float,
    hide_planned_separator: bool,
    draw_room_numbers: bool,
) -> dict[str, Any]:
    if not source_npz.exists():
        raise FileNotFoundError(source_npz)
    with np.load(source_npz, allow_pickle=False) as source, np.load(post_npz, allow_pickle=False) as post:
        nav_free = _bool(source, "voxel_nav_free_xy")
        nav_unknown = _bool(source, "voxel_nav_unknown_xy", nav_free.shape)
        nav_occupied = _bool(source, "voxel_nav_occupied_xy", nav_free.shape)
        label = np.asarray(post["final_room_label_map"], dtype=np.int32)
        native_label = np.asarray(post["voronoi_original_final_room_label_map"], dtype=np.int32)
        wall = _bool(post, "voxroom_wall_door_no_extension_no_mincc_despurred_mask", label.shape)
        spur = _bool(post, "voxroom_wall_door_no_extension_no_mincc_spur_removed_mask", label.shape)
        planned = _bool(post, "orthogonal_corner_snap_planned_separator_mask", label.shape)
        actual = _bool(post, "orthogonal_corner_snap_separator_mask", label.shape)
        corners = _bool(post, "orthogonal_corner_snap_wall_corner_mask", label.shape)

    filtered_label = filter_small_regions(label, min_region_cells=min_region_cells)
    image = base_nav_image(nav_free=nav_free, nav_unknown=nav_unknown, nav_occupied=nav_occupied)
    image = blend_labels(image, filtered_label)
    paint_mask(image, wall, WALL_COLOR, alpha=0.88)
    paint_mask(image, spur, SPUR_COLOR, alpha=1.0)
    if not bool(hide_planned_separator):
        paint_mask(image, planned, PLANNED_SEPARATOR_COLOR, alpha=1.0)
    paint_mask(image, actual, ACTUAL_SEPARATOR_COLOR, alpha=1.0)
    draw_corner_pixels(image, corners)

    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    if bool(draw_room_numbers):
        draw_room_number_labels(pil, filtered_label)
    draw_legend(pil, hide_planned_separator=hide_planned_separator)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_png)
    return {
        "native_room_count": _room_count(native_label),
        "postprocess_room_count_before_filter": _room_count(label),
        "postprocess_room_count_after_filter": _room_count(filtered_label),
        "planned_separator_cells": int(np.count_nonzero(planned)),
        "actual_separator_cells": int(np.count_nonzero(actual)),
        "wall_cells": int(np.count_nonzero(wall)),
        "spur_removed_cells": int(np.count_nonzero(spur)),
        "corner_cells": int(np.count_nonzero(corners)),
        "room_numbers_drawn": bool(draw_room_numbers),
        "min_region_cells": int(min_region_cells),
        "min_region_m2": float(min_region_cells) * float(map_resolution_m) ** 2,
    }


def base_nav_image(*, nav_free: np.ndarray, nav_unknown: np.ndarray, nav_occupied: np.ndarray) -> np.ndarray:
    image = np.zeros((*nav_free.shape, 3), dtype=np.uint8)
    image[:, :] = FREE_COLOR
    image[nav_unknown.astype(bool)] = UNEXPLORED_COLOR
    image[nav_occupied.astype(bool)] = OCCUPIED_COLOR
    return image.astype(np.float32)


def blend_labels(image: np.ndarray, label: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32)
    for value in (int(v) for v in np.unique(label) if int(v) > 0):
        mask = label == value
        color = np.asarray(label_color(value), dtype=np.float32)
        out[mask] = out[mask] * (1.0 - LABEL_ALPHA) + color * LABEL_ALPHA
    return out


def label_color(value: int) -> tuple[int, int, int]:
    palette = (
        (50, 125, 180),
        (245, 155, 55),
        (70, 170, 105),
        (215, 85, 90),
        (145, 110, 190),
        (150, 105, 75),
        (230, 120, 175),
        (120, 120, 120),
        (185, 185, 45),
        (80, 180, 190),
    )
    return palette[(int(value) - 1) % len(palette)]


def paint_mask(image: np.ndarray, mask: np.ndarray, color: np.ndarray, *, alpha: float) -> None:
    mask = np.asarray(mask, dtype=bool)
    if not bool(np.any(mask)):
        return
    image[mask] = image[mask] * (1.0 - float(alpha)) + color.astype(np.float32) * float(alpha)


def draw_corner_pixels(image: np.ndarray, corner: np.ndarray) -> None:
    if not bool(np.any(corner)):
        return
    dilated = ndimage.binary_dilation(corner.astype(bool), structure=np.ones((3, 3), dtype=bool), iterations=1)
    image[dilated] = np.asarray((20, 20, 20), dtype=np.float32)
    image[corner.astype(bool)] = CORNER_COLOR.astype(np.float32)


def draw_room_number_labels(image: Image.Image, label: np.ndarray) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for value in (int(v) for v in np.unique(label) if int(v) > 0):
        rows, cols = np.nonzero(np.asarray(label) == int(value))
        if rows.size == 0:
            continue
        cy = int(round(float(np.median(rows))))
        cx = int(round(float(np.median(cols))))
        text = str(value)
        bbox = draw.textbbox((cx, cy), text, font=font)
        tw = int(bbox[2] - bbox[0])
        th = int(bbox[3] - bbox[1])
        pad = 2
        x0 = max(0, min(image.width - tw - pad * 2, cx - tw // 2 - pad))
        y0 = max(0, min(image.height - th - pad * 2, cy - th // 2 - pad))
        draw.rectangle((x0, y0, x0 + tw + pad * 2, y0 + th + pad * 2), fill=(255, 255, 255), outline=(20, 20, 20))
        draw.text((x0 + pad, y0 + pad), text, fill=(0, 0, 0), font=font)


def filter_small_regions(label: np.ndarray, *, min_region_cells: int) -> np.ndarray:
    label = np.asarray(label, dtype=np.int32)
    out = label.copy()
    for value in (int(v) for v in np.unique(label) if int(v) > 0):
        mask = label == value
        component, count = ndimage.label(mask, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))
        for component_id in range(1, int(count) + 1):
            component_mask = component == component_id
            if int(np.count_nonzero(component_mask)) < int(min_region_cells):
                out[component_mask] = 0
    return out


def draw_legend(image: Image.Image, *, hide_planned_separator: bool) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    items = [
        ("occupied cell", tuple(int(v) for v in OCCUPIED_COLOR)),
        ("unexplored cell", tuple(int(v) for v in UNEXPLORED_COLOR)),
        ("wall+door", tuple(int(v) for v in WALL_COLOR)),
        ("spur removed", tuple(int(v) for v in SPUR_COLOR)),
        ("actual separator", tuple(int(v) for v in ACTUAL_SEPARATOR_COLOR)),
    ]
    if not bool(hide_planned_separator):
        items.insert(-1, ("planned separator", tuple(int(v) for v in PLANNED_SEPARATOR_COLOR)))
    pad = 6
    swatch = 9
    line_h = 13
    width = max(int(draw.textlength(text, font=font)) for text, _ in items) + swatch + pad * 3
    height = line_h * len(items) + pad * 2
    x0 = max(0, image.width - width - 8)
    y0 = 8
    draw.rectangle((x0, y0, x0 + width, y0 + height), fill=(255, 255, 255), outline=(0, 0, 0))
    y = y0 + pad
    for text, color in items:
        draw.rectangle((x0 + pad, y + 2, x0 + pad + swatch, y + 2 + swatch), fill=color, outline=(0, 0, 0))
        draw.text((x0 + pad * 2 + swatch, y), text, fill=(0, 0, 0), font=font)
        y += line_h


def make_sheet(paths: list[Path], output_path: Path, *, max_side: int) -> None:
    if not paths:
        return
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    cols = min(4, max(1, math.ceil(math.sqrt(len(thumbs)))))
    rows = int(math.ceil(len(thumbs) / cols))
    tile_w = max(img.width for img in thumbs)
    tile_h = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (255, 255, 255))
    for index, image in enumerate(thumbs):
        x = (index % cols) * tile_w + (tile_w - image.width) // 2
        y = (index // cols) * tile_h + (tile_h - image.height) // 2
        sheet.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _bool(z: np.lib.npyio.NpzFile, key: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    if key in z.files:
        return np.asarray(z[key], dtype=bool)
    if shape is None:
        raise KeyError(key)
    return np.zeros(shape, dtype=bool)


def _room_count(label: np.ndarray) -> int:
    return int(sum(1 for value in np.unique(np.asarray(label)) if int(value) > 0))


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
