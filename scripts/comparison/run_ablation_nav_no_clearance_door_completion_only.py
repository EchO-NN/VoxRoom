from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


DOOR_CUT_KEYS = (
    "voxel_door_final_cut_mask",
    "voxel_final_door_cut_mask",
    "voxel_door_partition_cut_accepted_mask",
    "voxel_door_topology_accepted_cut_mask",
    "voxel_stable_door_cut_mask",
    "voxel_door_cut_mask",
)

PALETTE = np.asarray(
    [
        (230, 103, 103),
        (107, 166, 214),
        (125, 196, 139),
        (236, 172, 93),
        (172, 143, 207),
        (101, 186, 196),
        (218, 126, 177),
        (193, 188, 89),
        (141, 187, 232),
        (188, 146, 108),
        (142, 214, 178),
        (218, 151, 119),
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation: build room masks by cutting no-clearance navigation free space "
            "only with door seed completion lines."
        )
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--min-visual-room-area-m2", type=float, default=0.3)
    parser.add_argument("--max-gallery-items", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root
    out_root = args.out_root
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.force:
        raise FileExistsError(f"{out_root} exists; pass --force to overwrite generated files")
    result_root = out_root / "stateful_replay"
    view_root = out_root / "room_mask_views"
    result_root.mkdir(parents=True, exist_ok=True)
    view_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    gallery_items: list[dict[str, str]] = []
    snapshot_paths = sorted(source_root.glob("*/roomseg_snapshots/roomseg_step_*.npz"))
    for snapshot_path in snapshot_paths:
        scene = snapshot_path.parents[1].name
        result_scene_dir = result_root / scene / "roomseg_snapshots"
        view_scene_dir = view_root / scene
        result_scene_dir.mkdir(parents=True, exist_ok=True)
        view_scene_dir.mkdir(parents=True, exist_ok=True)
        out_npz = result_scene_dir / snapshot_path.name
        out_png = view_scene_dir / f"{snapshot_path.stem}.room_mask.png"
        row = process_snapshot(
            snapshot_path,
            out_npz=out_npz,
            out_png=out_png,
            min_visual_room_area_m2=float(args.min_visual_room_area_m2),
        )
        row["scene"] = scene
        rows.append(row)
        if len(gallery_items) < int(args.max_gallery_items):
            gallery_items.append(
                {
                    "scene": scene,
                    "step": str(row.get("step", "")),
                    "path": str(out_png.relative_to(out_root)),
                }
            )

    summary = {
        "ablation": "nav_no_clearance_door_completion_only",
        "source_root": str(source_root),
        "result_root": str(result_root),
        "room_mask_view_root": str(view_root),
        "snapshot_count": len(rows),
        "scene_count": len({str(r["scene"]) for r in rows}),
        "total_nav_free_cells": int(sum(int(r["nav_free_cells"]) for r in rows)),
        "total_door_cut_cells": int(sum(int(r["door_cut_cells"]) for r in rows)),
        "total_labeled_cells": int(sum(int(r["labeled_cells"]) for r in rows)),
        "door_cut_keys": list(DOOR_CUT_KEYS),
        "min_visual_room_area_m2": float(args.min_visual_room_area_m2),
    }
    (out_root / "manifest.json").write_text(json.dumps({"summary": summary, "snapshots": rows}, indent=2), encoding="utf-8")
    write_summary_md(out_root / "summary.md", summary)
    write_gallery(out_root / "room_mask_gallery.html", gallery_items)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def process_snapshot(
    snapshot_path: Path,
    *,
    out_npz: Path,
    out_png: Path,
    min_visual_room_area_m2: float,
) -> dict[str, object]:
    with np.load(snapshot_path, allow_pickle=False) as z:
        nav_free = get_bool(z, "voxel_nav_free_xy", fallback="observed_free_mask")
        shape = nav_free.shape
        nav_occ = get_bool(z, "voxel_nav_occupied_xy", fallback="obstacle_mask", shape=shape)
        nav_unknown = get_bool(z, "voxel_nav_unknown_xy", fallback="unknown_mask", shape=shape)
        door_cut = np.zeros(shape, dtype=bool)
        used_keys: list[str] = []
        for key in DOOR_CUT_KEYS:
            if key in z.files:
                arr = np.asarray(z[key], dtype=bool)
                if arr.shape == shape:
                    door_cut |= arr
                    used_keys.append(key)
        door_cut &= nav_free
        partition_free = nav_free & ~door_cut
        labels, num_labels = ndimage.label(partition_free, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
        labels = relabel_positive(labels.astype(np.int32))
        arrays = {
            "final_room_label_map": labels,
            "voxel_final_room_label_map": labels,
            "observed_free_mask": nav_free.astype(bool),
            "navigation_free_room_domain": nav_free.astype(bool),
            "obstacle_mask": nav_occ.astype(bool),
            "unknown_mask": nav_unknown.astype(bool),
            "voxel_nav_free_xy": nav_free.astype(bool),
            "voxel_nav_occupied_xy": nav_occ.astype(bool),
            "voxel_nav_unknown_xy": nav_unknown.astype(bool),
            "accepted_separators": door_cut.astype(bool),
            "voxel_final_separator_map": door_cut.astype(bool),
            "voxel_door_completion_only_cut_mask": door_cut.astype(bool),
            "map_resolution_m": scalar_from_npz(z, "map_resolution_m", 0.05, np.float32),
            "ablation_source_snapshot_path": np.asarray(str(snapshot_path)),
            "ablation_mode": np.asarray("nav_no_clearance_door_completion_only"),
        }
        if "voxel_vertical_free_xy" in z.files:
            arrays["voxel_vertical_free_xy"] = np.asarray(z["voxel_vertical_free_xy"], dtype=bool)
        if "height_profile_vertical_free_xy" in z.files:
            arrays["height_profile_vertical_free_xy"] = np.asarray(z["height_profile_vertical_free_xy"], dtype=bool)
        np.savez_compressed(out_npz, **arrays)

        resolution_m = float(np.asarray(arrays["map_resolution_m"]).reshape(-1)[0])
        render_room_mask(
            out_png,
            labels=labels,
            nav_free=nav_free,
            nav_occ=nav_occ,
            nav_unknown=nav_unknown,
            min_area_cells=min_area_cells(min_visual_room_area_m2, resolution_m),
        )

    return {
        "snapshot": str(snapshot_path),
        "output_npz": str(out_npz),
        "output_png": str(out_png),
        "step": step_from_name(snapshot_path.stem),
        "nav_free_cells": int(np.count_nonzero(nav_free)),
        "door_cut_cells": int(np.count_nonzero(door_cut)),
        "labeled_cells": int(np.count_nonzero(labels)),
        "room_count": int(np.max(labels)),
        "raw_component_count": int(num_labels),
        "door_cut_keys_present": used_keys,
    }


def get_bool(
    z: np.lib.npyio.NpzFile,
    key: str,
    *,
    fallback: str | None = None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if key in z.files:
        arr = np.asarray(z[key], dtype=bool)
    elif fallback is not None and fallback in z.files:
        arr = np.asarray(z[fallback], dtype=bool)
    elif shape is not None:
        arr = np.zeros(shape, dtype=bool)
    else:
        raise KeyError(key)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{key} shape {arr.shape} != {shape}")
    return arr


def scalar_from_npz(z: np.lib.npyio.NpzFile, key: str, default: float, dtype: type[np.floating]) -> np.ndarray:
    if key in z.files:
        return np.asarray(float(np.asarray(z[key]).reshape(-1)[0]), dtype=dtype)
    return np.asarray(default, dtype=dtype)


def relabel_positive(labels: np.ndarray) -> np.ndarray:
    out = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1
    for label in np.unique(labels):
        label_i = int(label)
        if label_i <= 0:
            continue
        out[labels == label_i] = next_id
        next_id += 1
    return out


def min_area_cells(area_m2: float, resolution_m: float) -> int:
    if area_m2 <= 0:
        return 1
    return max(1, int(math.ceil(area_m2 / max(resolution_m * resolution_m, 1e-12))))


def render_room_mask(
    out_png: Path,
    *,
    labels: np.ndarray,
    nav_free: np.ndarray,
    nav_occ: np.ndarray,
    nav_unknown: np.ndarray,
    min_area_cells: int,
) -> None:
    base = np.full((*labels.shape, 3), (154, 154, 154), dtype=np.uint8)
    base[nav_unknown] = (154, 154, 154)
    base[nav_free] = (246, 246, 242)
    base[nav_occ] = (58, 58, 58)
    image = base.astype(np.float32)
    for label in np.unique(labels):
        label_i = int(label)
        if label_i <= 0:
            continue
        mask = labels == label_i
        if int(np.count_nonzero(mask)) < int(min_area_cells):
            continue
        color = PALETTE[(label_i - 1) % len(PALETTE)].astype(np.float32)
        image[mask] = image[mask] * 0.35 + color * 0.65
    cropped = crop_to_content(image.astype(np.uint8), nav_free | nav_occ | nav_unknown, pad=80)
    Image.fromarray(cropped).save(out_png)


def crop_to_content(image: np.ndarray, content: np.ndarray, *, pad: int) -> np.ndarray:
    ys, xs = np.where(content)
    if ys.size == 0:
        return image
    h, w = content.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    side = max(y1 - y0, x1 - x0)
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    y0 = max(0, min(h - side, cy - side // 2))
    x0 = max(0, min(w - side, cx - side // 2))
    return image[y0 : y0 + side, x0 : x0 + side]


def step_from_name(stem: str) -> int | None:
    marker = "roomseg_step_"
    if marker not in stem:
        return None
    try:
        return int(stem.split(marker, 1)[1].split(".", 1)[0])
    except ValueError:
        return None


def write_summary_md(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Ablation: nav no-clearance + door completion only",
        "",
        f"- Source root: `{summary['source_root']}`",
        f"- Result root: `{summary['result_root']}`",
        f"- Room mask views: `{summary['room_mask_view_root']}`",
        f"- Scene count: {summary['scene_count']}",
        f"- Snapshot count: {summary['snapshot_count']}",
        f"- Total no-clearance nav free cells: {summary['total_nav_free_cells']}",
        f"- Total door completion cut cells: {summary['total_door_cut_cells']}",
        f"- Total labeled cells: {summary['total_labeled_cells']}",
        f"- Small room masks hidden in visualization below: {summary['min_visual_room_area_m2']} m^2",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_gallery(path: Path, items: Iterable[dict[str, str]]) -> None:
    cards = []
    for item in items:
        src = html.escape(item["path"])
        title = html.escape(f"{item['scene']} step {item['step']}")
        cards.append(f"<figure><img src=\"{src}\" loading=\"lazy\"><figcaption>{title}</figcaption></figure>")
    doc = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Nav No-Clearance Door Completion Only Room Masks</title>
  <style>
    body { font-family: sans-serif; margin: 18px; background: #f2f2f2; color: #222; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
    figure { margin: 0; background: white; padding: 8px; border: 1px solid #ddd; }
    img { width: 100%%; image-rendering: pixelated; display: block; }
    figcaption { font-size: 12px; margin-top: 6px; }
  </style>
</head>
<body>
  <h1>Nav No-Clearance + Door Completion Only</h1>
  <div class="grid">
%s
  </div>
</body>
</html>
""" % ("\n".join(cards))
    path.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
