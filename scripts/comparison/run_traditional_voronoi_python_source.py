from __future__ import annotations

import argparse
import html
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from voxroom_online.isaac_runtime.baselines.mask_io import enforce_room_mask_contract
from voxroom_online.isaac_runtime.baselines.offline.fallback_voronoi import segment_snapshot_arrays


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
        description="Run the traditional Python Voronoi implementation on a prepared 2D source snapshot tree."
    )
    parser.add_argument("--source-run-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--result-subdir", default="stateful_replay")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-visual-room-area-m2", type=float, default=0.3)
    parser.add_argument("--max-gallery-items", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_run_root
    out_root = args.out_root
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    out_root.mkdir(parents=True, exist_ok=True)
    result_root = out_root / str(args.result_subdir)
    view_root = out_root / "room_mask_views"
    result_root.mkdir(parents=True, exist_ok=True)
    view_root.mkdir(parents=True, exist_ok=True)

    snapshot_paths = sorted(source_root.glob("kujiale_*/roomseg_snapshots/roomseg_step_*.npz"))
    if not snapshot_paths:
        raise FileNotFoundError(f"no snapshots under {source_root}")

    jobs: list[tuple[str, str, str, str, float, bool]] = []
    for snapshot_path in snapshot_paths:
        scene = snapshot_path.parents[1].name
        out_npz = result_root / scene / "roomseg_snapshots" / snapshot_path.name
        out_png = view_root / scene / f"{snapshot_path.stem}.room_mask.png"
        jobs.append(
            (
                str(snapshot_path),
                str(out_npz),
                str(out_png),
                scene,
                float(args.min_visual_room_area_m2),
                bool(args.force),
            )
        )

    rows: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for idx, job in enumerate(jobs, start=1):
            rows.append(process_one(job))
            if idx % 50 == 0 or idx == len(jobs):
                print(f"processed {idx}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_one, job): job for job in jobs}
            for idx, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if idx % 50 == 0 or idx == len(jobs):
                    print(f"processed {idx}/{len(jobs)}", flush=True)

    rows.sort(key=lambda r: (str(r["scene"]), int(r["step"])))
    summary = {
        "ablation": "vertical_free_traditional_voronoi_python_full1554",
        "source_run_root": str(source_root),
        "result_root": str(result_root),
        "room_mask_view_root": str(view_root),
        "scene_count": len({str(row["scene"]) for row in rows}),
        "snapshot_count": len(rows),
        "total_free_cells": int(sum(int(row["free_cells"]) for row in rows)),
        "total_labeled_cells": int(sum(int(row["labeled_cells"]) for row in rows)),
        "total_final_rooms": int(sum(int(row["room_count"]) for row in rows)),
        "runner_type": "python_fallback",
        "note": "Traditional Voronoi fallback over vertical_free_lightweight; not width-jump optimized Voronoi.",
        "min_visual_room_area_m2": float(args.min_visual_room_area_m2),
    }
    (out_root / "manifest.json").write_text(
        json.dumps({"summary": summary, "snapshots": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_md(out_root / "summary.md", summary)
    gallery_items = [
        {
            "scene": str(row["scene"]),
            "step": str(row["step"]),
            "path": str(Path(row["output_png"]).relative_to(out_root)),
        }
        for row in rows[: int(args.max_gallery_items)]
    ]
    write_gallery(out_root / "room_mask_gallery.html", gallery_items)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def process_one(job: tuple[str, str, str, str, float, bool]) -> dict[str, Any]:
    snapshot_s, out_npz_s, out_png_s, scene, min_visual_room_area_m2, force = job
    snapshot_path = Path(snapshot_s)
    out_npz = Path(out_npz_s)
    out_png = Path(out_png_s)
    if out_npz.exists() and out_png.exists() and not force:
        return existing_row(snapshot_path, out_npz, out_png, scene)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with np.load(snapshot_path, allow_pickle=False) as z:
        arrays = {str(key): np.asarray(z[key]).copy() for key in z.files}
    labels, metadata, _debug = segment_snapshot_arrays(snapshot_path, arrays, default_resolution_m=0.05)
    labels = enforce_room_mask_contract(labels, arrays, clip_to_eval_domain=True)
    arrays["final_room_label_map"] = labels.astype(np.int32)
    arrays["voxel_final_room_label_map"] = labels.astype(np.int32)
    arrays["baseline_name"] = np.asarray("traditional_voronoi_python")
    arrays["baseline_metadata_json"] = np.asarray(json.dumps(json_ready(metadata), ensure_ascii=False, sort_keys=True))
    arrays["ablation_mode"] = np.asarray("vertical_free_traditional_voronoi_python_full1554")
    np.savez_compressed(out_npz, **arrays)

    resolution = float(np.asarray(arrays.get("map_resolution_m", np.asarray(0.05))).reshape(-1)[0])
    free = first_bool(arrays, ("navigation_free_room_domain", "observed_free_mask", "voxel_vertical_free_xy"), labels.shape)
    occ = first_bool(arrays, ("obstacle_mask", "occupancy_map", "voxel_nav_occupied_xy"), labels.shape, default=False)
    unknown = first_bool(arrays, ("unknown_mask", "voxel_nav_unknown_xy"), labels.shape, default=False)
    render_room_mask(
        out_png,
        labels=labels,
        free=free,
        occupied=occ,
        unknown=unknown,
        min_area_cells=area_to_cells(float(min_visual_room_area_m2), resolution),
    )
    return {
        "scene": scene,
        "step": step_from_name(snapshot_path.stem),
        "input_npz": str(snapshot_path),
        "output_npz": str(out_npz),
        "output_png": str(out_png),
        "free_cells": int(np.count_nonzero(free)),
        "labeled_cells": int(np.count_nonzero(labels)),
        "room_count": int(np.max(labels)) if labels.size else 0,
        "critical_lines_drawn": int(metadata.get("stats", {}).get("critical_lines_drawn", 0)),
        "runner_type": str(metadata.get("runner_type", "")),
    }


def existing_row(snapshot_path: Path, out_npz: Path, out_png: Path, scene: str) -> dict[str, Any]:
    with np.load(out_npz, allow_pickle=False) as z:
        labels = np.asarray(z["final_room_label_map"], dtype=np.int32)
        free = first_bool({str(k): np.asarray(z[k]) for k in z.files}, ("navigation_free_room_domain", "observed_free_mask"), labels.shape)
    return {
        "scene": scene,
        "step": step_from_name(snapshot_path.stem),
        "input_npz": str(snapshot_path),
        "output_npz": str(out_npz),
        "output_png": str(out_png),
        "free_cells": int(np.count_nonzero(free)),
        "labeled_cells": int(np.count_nonzero(labels)),
        "room_count": int(np.max(labels)) if labels.size else 0,
        "critical_lines_drawn": None,
        "runner_type": "existing",
    }


def first_bool(
    arrays: dict[str, Any],
    keys: tuple[str, ...],
    shape: tuple[int, int],
    *,
    default: bool | None = None,
) -> np.ndarray:
    for key in keys:
        if key not in arrays:
            continue
        arr = np.asarray(arrays[key], dtype=bool)
        if arr.shape == shape:
            return arr.copy()
    if default is None:
        raise KeyError(keys)
    return np.full(shape, bool(default), dtype=bool)


def render_room_mask(
    out_png: Path,
    *,
    labels: np.ndarray,
    free: np.ndarray,
    occupied: np.ndarray,
    unknown: np.ndarray,
    min_area_cells: int,
) -> None:
    base = np.full((*labels.shape, 3), (154, 154, 154), dtype=np.uint8)
    base[free] = (246, 246, 242)
    base[unknown] = (154, 154, 154)
    base[occupied] = (58, 58, 58)
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


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Ablation: vertical free + traditional Voronoi",
        "",
        f"- Source root: `{summary['source_run_root']}`",
        f"- Result root: `{summary['result_root']}`",
        f"- Room mask views: `{summary['room_mask_view_root']}`",
        f"- Scene count: {summary['scene_count']}",
        f"- Snapshot count: {summary['snapshot_count']}",
        f"- Runner type: {summary['runner_type']}",
        f"- Note: {summary['note']}",
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
  <title>Vertical Free Traditional Voronoi Room Masks</title>
  <style>
    body { font-family: sans-serif; margin: 18px; background: #f2f2f2; color: #222; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
    figure { margin: 0; background: white; padding: 8px; border: 1px solid #ddd; }
    img { width: 100%%; image-rendering: pixelated; display: block; }
    figcaption { font-size: 12px; margin-top: 6px; }
  </style>
</head>
<body>
  <h1>Vertical Free + Traditional Voronoi</h1>
  <div class="grid">
%s
  </div>
</body>
</html>
""" % ("\n".join(cards))
    path.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
