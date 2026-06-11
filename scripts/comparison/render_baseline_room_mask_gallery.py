from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


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
    parser = argparse.ArgumentParser(description="Render room-mask PNGs and an HTML gallery from baseline npz outputs.")
    parser.add_argument("--method-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--title", default="Baseline Room Masks")
    parser.add_argument("--min-visual-room-area-m2", type=float, default=0.3)
    parser.add_argument("--max-gallery-items", type=int, default=240)
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    method_root = args.method_root
    out_root = args.out_root
    if not method_root.exists():
        raise FileNotFoundError(method_root)
    view_root = out_root / "room_mask_views"
    view_root.mkdir(parents=True, exist_ok=True)

    paths = sorted(method_root.glob("kujiale_*/roomseg_snapshots/roomseg_step_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no roomseg snapshots under {method_root}")
    if args.max_snapshots is not None:
        paths = paths[: max(0, int(args.max_snapshots))]

    rows: list[dict[str, Any]] = []
    gallery_items: list[dict[str, str]] = []
    for idx, snapshot_path in enumerate(paths, start=1):
        scene = snapshot_path.parents[1].name
        out_png = view_root / scene / f"{snapshot_path.stem}.room_mask.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        row = render_one(
            snapshot_path,
            out_png=out_png,
            min_visual_room_area_m2=float(args.min_visual_room_area_m2),
            force=bool(args.force),
        )
        row["scene"] = scene
        rows.append(row)
        if len(gallery_items) < int(args.max_gallery_items):
            gallery_items.append(
                {
                    "scene": scene,
                    "step": str(row["step"]),
                    "path": str(out_png.relative_to(out_root)),
                    "rooms": str(row["room_count"]),
                }
            )
        if idx % 200 == 0 or idx == len(paths):
            print(f"rendered {idx}/{len(paths)}", flush=True)

    summary = {
        "method_root": str(method_root),
        "out_root": str(out_root),
        "room_mask_view_root": str(view_root),
        "scene_count": len({str(row["scene"]) for row in rows}),
        "snapshot_count": len(rows),
        "min_visual_room_area_m2": float(args.min_visual_room_area_m2),
        "labeled_cells": int(sum(int(row["labeled_cells"]) for row in rows)),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "room_mask_render_manifest.json").write_text(
        json.dumps({"summary": summary, "snapshots": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_gallery(out_root / "room_mask_gallery.html", gallery_items, title=str(args.title))
    return 0


def render_one(
    snapshot_path: Path,
    *,
    out_png: Path,
    min_visual_room_area_m2: float,
    force: bool,
) -> dict[str, Any]:
    with np.load(snapshot_path, allow_pickle=False) as z:
        labels = first_array(z, ("final_room_label_map", "voxel_final_room_label_map")).astype(np.int32)
        shape = labels.shape
        free = first_bool(z, ("observed_free_mask", "navigation_free_room_domain", "voxel_nav_free_xy", "voxel_vertical_free_xy"), shape)
        occ = first_bool(z, ("obstacle_mask", "occupancy_map", "voxel_nav_occupied_xy"), shape, default=False)
        unknown = first_bool(z, ("unknown_mask", "voxel_nav_unknown_xy"), shape, default=False)
        resolution = scalar(z, "map_resolution_m", default=0.05)
    if force or not out_png.exists():
        render_room_mask(
            out_png,
            labels=labels,
            free=free,
            occupied=occ,
            unknown=unknown,
            min_area_cells=area_to_cells(min_visual_room_area_m2, resolution),
        )
    return {
        "snapshot": str(snapshot_path),
        "output_png": str(out_png),
        "step": step_from_name(snapshot_path.stem),
        "room_count": int(np.max(labels)) if labels.size else 0,
        "free_cells": int(np.count_nonzero(free)),
        "labeled_cells": int(np.count_nonzero(labels > 0)),
    }


def first_array(z: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim == 2:
                return arr.copy()
    raise KeyError(keys)


def first_bool(
    z: np.lib.npyio.NpzFile,
    keys: tuple[str, ...],
    shape: tuple[int, int],
    *,
    default: bool | None = None,
) -> np.ndarray:
    for key in keys:
        if key not in z.files:
            continue
        arr = np.asarray(z[key], dtype=bool)
        if arr.shape == shape:
            return arr.copy()
    if default is None:
        raise KeyError(keys)
    return np.full(shape, bool(default), dtype=bool)


def scalar(z: np.lib.npyio.NpzFile, key: str, *, default: float) -> float:
    if key not in z.files:
        return float(default)
    try:
        value = float(np.asarray(z[key]).reshape(-1)[0])
    except Exception:
        return float(default)
    return value if math.isfinite(value) and value > 0.0 else float(default)


def render_room_mask(
    out_png: Path,
    *,
    labels: np.ndarray,
    free: np.ndarray,
    occupied: np.ndarray,
    unknown: np.ndarray,
    min_area_cells: int,
) -> None:
    base = np.full((*labels.shape, 3), (150, 150, 150), dtype=np.uint8)
    base[unknown] = (150, 150, 150)
    base[free] = (242, 242, 238)
    base[occupied] = (48, 48, 48)
    image = base.astype(np.float32)
    for value in np.unique(labels):
        label = int(value)
        if label <= 0:
            continue
        mask = labels == label
        if int(np.count_nonzero(mask)) < int(min_area_cells):
            continue
        color = PALETTE[(label - 1) % len(PALETTE)].astype(np.float32)
        image[mask] = image[mask] * 0.35 + color * 0.65
    content = free | occupied | unknown | (labels > 0)
    Image.fromarray(crop_to_content(image.astype(np.uint8), content, pad=80)).save(out_png)


def crop_to_content(image: np.ndarray, content: np.ndarray, *, pad: int) -> np.ndarray:
    ys, xs = np.where(content)
    if ys.size == 0:
        return image
    h, w = content.shape
    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(h, int(ys.max()) + int(pad) + 1)
    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(w, int(xs.max()) + int(pad) + 1)
    side = max(y1 - y0, x1 - x0)
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    y0 = max(0, min(h - side, cy - side // 2))
    x0 = max(0, min(w - side, cx - side // 2))
    return image[y0 : y0 + side, x0 : x0 + side]


def area_to_cells(area_m2: float, resolution_m: float) -> int:
    return max(1, int(math.ceil(float(area_m2) / max(float(resolution_m) * float(resolution_m), 1e-12))))


def step_from_name(stem: str) -> int:
    marker = "roomseg_step_"
    if marker not in stem:
        return -1
    try:
        return int(stem.split(marker, 1)[1].split(".", 1)[0])
    except ValueError:
        return -1


def write_gallery(path: Path, items: Iterable[dict[str, str]], *, title: str) -> None:
    cards = []
    for item in items:
        src = html.escape(item["path"])
        caption = html.escape(f"{item['scene']} step {item['step']} rooms={item['rooms']}")
        cards.append(f"<figure><img src=\"{src}\" loading=\"lazy\"><figcaption>{caption}</figcaption></figure>")
    doc = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 18px; background: #f2f2f2; color: #222; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; background: white; padding: 8px; border: 1px solid #ddd; }}
    img {{ width: 100%; image-rendering: pixelated; display: block; }}
    figcaption {{ font-size: 12px; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="grid">
{cards}
  </div>
</body>
</html>
""".format(title=html.escape(title), cards="\n".join(cards))
    path.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
