from __future__ import annotations

import argparse
import html
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxroom_online.isaac_runtime.baselines.mask_io import enforce_room_mask_contract


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


@dataclass(frozen=True)
class WallGapParams:
    max_gap_cells: int = 8
    min_support_cells: int = 10
    free_guard_radius_cells: int = 2
    max_free_band_fraction: float = 0.80
    bridge_perpendicular_radius_cells: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation 5: complete small axis-aligned gaps in occupied wall lines, "
            "then segment the vertical free map with the completed wall constraints."
        )
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-gap-cells", type=int, default=WallGapParams.max_gap_cells)
    parser.add_argument("--min-support-cells", type=int, default=WallGapParams.min_support_cells)
    parser.add_argument("--free-guard-radius-cells", type=int, default=WallGapParams.free_guard_radius_cells)
    parser.add_argument("--max-free-band-fraction", type=float, default=WallGapParams.max_free_band_fraction)
    parser.add_argument(
        "--bridge-perpendicular-radius-cells",
        type=int,
        default=WallGapParams.bridge_perpendicular_radius_cells,
    )
    parser.add_argument("--min-visual-room-area-m2", type=float, default=0.3)
    parser.add_argument("--max-gallery-items", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if out_root.exists() and any(out_root.iterdir()) and not bool(args.force):
        raise FileExistsError(f"{out_root} exists; pass --force to reuse/overwrite generated files")

    params = WallGapParams(
        max_gap_cells=int(args.max_gap_cells),
        min_support_cells=int(args.min_support_cells),
        free_guard_radius_cells=int(args.free_guard_radius_cells),
        max_free_band_fraction=float(args.max_free_band_fraction),
        bridge_perpendicular_radius_cells=int(args.bridge_perpendicular_radius_cells),
    )
    validate_params(params)

    result_root = out_root / "stateful_replay"
    view_root = out_root / "room_mask_views"
    result_root.mkdir(parents=True, exist_ok=True)
    view_root.mkdir(parents=True, exist_ok=True)

    snapshot_paths = sorted(source_root.glob("*/roomseg_snapshots/roomseg_step_*.npz"))
    if not snapshot_paths:
        raise FileNotFoundError(f"no roomseg_step_*.npz snapshots under {source_root}")

    jobs: list[tuple[str, str, str, str, dict[str, Any], float, bool]] = []
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
                asdict(params),
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

    rows.sort(key=lambda row: (str(row["scene"]), int(row["step"])))
    summary = build_summary(
        source_root=source_root,
        out_root=out_root,
        result_root=result_root,
        view_root=view_root,
        rows=rows,
        params=params,
        min_visual_room_area_m2=float(args.min_visual_room_area_m2),
    )
    (out_root / "manifest.json").write_text(
        json.dumps({"summary": summary, "snapshots": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_md(out_root / "summary.md", summary)
    write_gallery(
        out_root / "room_mask_gallery.html",
        [
            {
                "scene": str(row["scene"]),
                "step": str(row["step"]),
                "path": str(Path(row["output_png"]).relative_to(out_root)),
            }
            for row in rows[: int(args.max_gallery_items)]
        ],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def validate_params(params: WallGapParams) -> None:
    if params.max_gap_cells < 1:
        raise ValueError("max_gap_cells must be >= 1")
    if params.min_support_cells < 1:
        raise ValueError("min_support_cells must be >= 1")
    if params.free_guard_radius_cells < 0:
        raise ValueError("free_guard_radius_cells must be >= 0")
    if not 0.0 <= params.max_free_band_fraction <= 1.0:
        raise ValueError("max_free_band_fraction must be in [0, 1]")
    if params.bridge_perpendicular_radius_cells < 0:
        raise ValueError("bridge_perpendicular_radius_cells must be >= 0")


def process_one(job: tuple[str, str, str, str, dict[str, Any], float, bool]) -> dict[str, Any]:
    snapshot_s, out_npz_s, out_png_s, scene, params_d, min_visual_room_area_m2, force = job
    snapshot_path = Path(snapshot_s)
    out_npz = Path(out_npz_s)
    out_png = Path(out_png_s)
    params = WallGapParams(**params_d)
    if out_npz.exists() and out_png.exists() and not force:
        return existing_row(snapshot_path, out_npz, out_png, scene)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with np.load(snapshot_path, allow_pickle=False) as z:
        arrays: dict[str, Any] = {str(key): np.asarray(z[key]).copy() for key in z.files}

    free = first_bool(
        arrays,
        (
            "voxel_vertical_free_xy",
            "vertical_free_room_domain",
            "height_profile_vertical_free_xy",
            "voxel_baseline_input_free_xy",
            "navigation_free_room_domain",
            "observed_free_mask",
        ),
    )
    shape = free.shape
    wall_seed = first_bool(arrays, ("occupancy_map", "obstacle_mask", "voxel_nav_occupied_xy"), shape=shape)
    unknown = first_bool(arrays, ("unknown_mask", "voxel_nav_unknown_xy"), shape=shape, default=False)
    nav_free = first_bool(arrays, ("navigation_free_room_domain", "observed_free_mask"), shape=shape, default=False)
    nav_occ = first_bool(arrays, ("obstacle_mask", "occupancy_map", "voxel_nav_occupied_xy"), shape=shape, default=False)
    nav_unknown = first_bool(arrays, ("unknown_mask", "voxel_nav_unknown_xy"), shape=shape, default=False)

    completed_wall, bridge_mask, gap_stats = complete_wall_gaps(wall_seed=wall_seed, free=free, params=params)
    separator = bridge_mask & free & ~wall_seed
    partition_domain = free & ~separator
    labels, raw_components = ndimage.label(
        partition_domain,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
    )
    labels = enforce_room_mask_contract(labels.astype(np.int32), arrays, clip_to_eval_domain=True)

    metadata = {
        "ablation": "vertical_free_wall_gap_completion",
        "algorithm": "axis_aligned_constrained_wall_gap_completion",
        "description": (
            "Scan occupied wall lines horizontally and vertically; bridge only small gaps "
            "between sufficiently long wall supports, rejecting gaps whose local band is "
            "mostly vertical-free/open space. Connected components are then computed on "
            "vertical_free minus accepted bridge cells."
        ),
        "source_snapshot_path": str(snapshot_path),
        "free_source_key": selected_bool_key(
            arrays,
            (
                "voxel_vertical_free_xy",
                "vertical_free_room_domain",
                "height_profile_vertical_free_xy",
                "voxel_baseline_input_free_xy",
                "navigation_free_room_domain",
                "observed_free_mask",
            ),
            shape=shape,
        ),
        "wall_source_key": selected_bool_key(arrays, ("occupancy_map", "obstacle_mask", "voxel_nav_occupied_xy"), shape=shape),
        "params": asdict(params),
        "stats": gap_stats,
        "raw_component_count": int(raw_components),
        "final_room_count": int(np.max(labels)) if labels.size else 0,
        "vertical_free_cells": int(np.count_nonzero(free)),
        "bridge_cells": int(np.count_nonzero(bridge_mask)),
        "separator_cells_in_vertical_free": int(np.count_nonzero(separator)),
        "labeled_cells": int(np.count_nonzero(labels)),
    }

    arrays["final_room_label_map"] = labels.astype(np.int32)
    arrays["voxel_final_room_label_map"] = labels.astype(np.int32)
    arrays["accepted_separators"] = separator.astype(bool)
    arrays["voxel_final_separator_map"] = separator.astype(bool)
    arrays["wall_gap_bridge_mask"] = bridge_mask.astype(bool)
    arrays["wall_gap_separator_mask"] = separator.astype(bool)
    arrays["wall_gap_completed_occupied_mask"] = completed_wall.astype(bool)
    arrays["wall_gap_original_occupied_mask"] = wall_seed.astype(bool)
    arrays["wall_gap_completion_metadata_json"] = np.asarray(json.dumps(json_ready(metadata), ensure_ascii=False, sort_keys=True))
    arrays["baseline_name"] = np.asarray("vertical_free_wall_gap_completion")
    arrays["baseline_metadata_json"] = np.asarray(json.dumps(json_ready(metadata), ensure_ascii=False, sort_keys=True))
    arrays["ablation_mode"] = np.asarray("vertical_free_wall_gap_completion")
    np.savez_compressed(out_npz, **arrays)

    resolution = float(np.asarray(arrays.get("map_resolution_m", np.asarray(0.05))).reshape(-1)[0])
    render_room_mask(
        out_png,
        labels=labels,
        nav_free=nav_free,
        nav_occ=nav_occ,
        nav_unknown=nav_unknown,
        separator=separator,
        min_area_cells=area_to_cells(float(min_visual_room_area_m2), resolution),
    )
    return {
        "scene": scene,
        "step": step_from_name(snapshot_path.stem),
        "input_npz": str(snapshot_path),
        "output_npz": str(out_npz),
        "output_png": str(out_png),
        "vertical_free_cells": int(np.count_nonzero(free)),
        "original_wall_cells": int(np.count_nonzero(wall_seed)),
        "bridge_cells": int(np.count_nonzero(bridge_mask)),
        "separator_cells_in_vertical_free": int(np.count_nonzero(separator)),
        "labeled_cells": int(np.count_nonzero(labels)),
        "room_count": int(np.max(labels)) if labels.size else 0,
        "raw_component_count": int(raw_components),
        **{f"gap_{key}": value for key, value in gap_stats.items()},
    }


def complete_wall_gaps(
    *,
    wall_seed: np.ndarray,
    free: np.ndarray,
    params: WallGapParams,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    wall = np.asarray(wall_seed, dtype=bool)
    free_arr = np.asarray(free, dtype=bool)
    if wall.shape != free_arr.shape:
        raise ValueError(f"wall/free shape mismatch: {wall.shape} vs {free_arr.shape}")
    bridge = np.zeros(wall.shape, dtype=bool)
    stats = {
        "candidate_gaps": 0,
        "accepted_gaps": 0,
        "accepted_horizontal_gaps": 0,
        "accepted_vertical_gaps": 0,
        "rejected_short_support": 0,
        "rejected_open_free_band": 0,
    }
    height, width = wall.shape
    for y in range(height):
        scan_line_for_gaps(
            line=wall[y, :],
            free=free_arr,
            bridge=bridge,
            orientation="horizontal",
            fixed_index=y,
            params=params,
            stats=stats,
        )
    for x in range(width):
        scan_line_for_gaps(
            line=wall[:, x],
            free=free_arr,
            bridge=bridge,
            orientation="vertical",
            fixed_index=x,
            params=params,
            stats=stats,
        )
    completed = wall | bridge
    stats["bridge_cells_total"] = int(np.count_nonzero(bridge))
    stats["bridge_cells_in_vertical_free"] = int(np.count_nonzero(bridge & free_arr & ~wall))
    return completed, bridge, stats


def scan_line_for_gaps(
    *,
    line: np.ndarray,
    free: np.ndarray,
    bridge: np.ndarray,
    orientation: str,
    fixed_index: int,
    params: WallGapParams,
    stats: dict[str, int],
) -> None:
    occupied_idx = np.flatnonzero(np.asarray(line, dtype=bool))
    if occupied_idx.size < 2:
        return
    breaks = np.flatnonzero(np.diff(occupied_idx) > 1)
    if breaks.size == 0:
        return
    run_starts = np.concatenate(([0], breaks + 1))
    run_ends = np.concatenate((breaks, [occupied_idx.size - 1]))
    starts = occupied_idx[run_starts]
    ends = occupied_idx[run_ends]
    lengths = ends - starts + 1

    for idx in range(len(starts) - 1):
        gap_start = int(ends[idx] + 1)
        gap_end = int(starts[idx + 1] - 1)
        gap_len = int(gap_end - gap_start + 1)
        if gap_len < 1 or gap_len > int(params.max_gap_cells):
            continue
        stats["candidate_gaps"] += 1
        if int(lengths[idx]) < int(params.min_support_cells) or int(lengths[idx + 1]) < int(params.min_support_cells):
            stats["rejected_short_support"] += 1
            continue
        if local_free_fraction(
            free,
            orientation=orientation,
            fixed_index=fixed_index,
            start=gap_start,
            end=gap_end,
            radius=int(params.free_guard_radius_cells),
        ) > float(params.max_free_band_fraction):
            stats["rejected_open_free_band"] += 1
            continue
        draw_bridge(
            bridge,
            orientation=orientation,
            fixed_index=fixed_index,
            start=gap_start,
            end=gap_end,
            radius=int(params.bridge_perpendicular_radius_cells),
        )
        stats["accepted_gaps"] += 1
        if orientation == "horizontal":
            stats["accepted_horizontal_gaps"] += 1
        else:
            stats["accepted_vertical_gaps"] += 1


def local_free_fraction(
    free: np.ndarray,
    *,
    orientation: str,
    fixed_index: int,
    start: int,
    end: int,
    radius: int,
) -> float:
    height, width = free.shape
    radius = max(0, int(radius))
    if orientation == "horizontal":
        y0 = max(0, int(fixed_index) - radius)
        y1 = min(height, int(fixed_index) + radius + 1)
        x0 = max(0, int(start))
        x1 = min(width, int(end) + 1)
    else:
        y0 = max(0, int(start))
        y1 = min(height, int(end) + 1)
        x0 = max(0, int(fixed_index) - radius)
        x1 = min(width, int(fixed_index) + radius + 1)
    if y1 <= y0 or x1 <= x0:
        return 1.0
    band = free[y0:y1, x0:x1]
    return float(np.count_nonzero(band)) / float(max(1, band.size))


def draw_bridge(
    bridge: np.ndarray,
    *,
    orientation: str,
    fixed_index: int,
    start: int,
    end: int,
    radius: int,
) -> None:
    height, width = bridge.shape
    radius = max(0, int(radius))
    if orientation == "horizontal":
        y0 = max(0, int(fixed_index) - radius)
        y1 = min(height, int(fixed_index) + radius + 1)
        x0 = max(0, int(start))
        x1 = min(width, int(end) + 1)
    else:
        y0 = max(0, int(start))
        y1 = min(height, int(end) + 1)
        x0 = max(0, int(fixed_index) - radius)
        x1 = min(width, int(fixed_index) + radius + 1)
    bridge[y0:y1, x0:x1] = True


def existing_row(snapshot_path: Path, out_npz: Path, out_png: Path, scene: str) -> dict[str, Any]:
    with np.load(out_npz, allow_pickle=False) as z:
        labels = np.asarray(z["final_room_label_map"], dtype=np.int32)
        free = first_bool({str(k): np.asarray(z[k]) for k in z.files}, ("voxel_vertical_free_xy", "vertical_free_room_domain"), shape=labels.shape)
        bridge = first_bool({str(k): np.asarray(z[k]) for k in z.files}, ("wall_gap_bridge_mask",), shape=labels.shape, default=False)
        separator = first_bool({str(k): np.asarray(z[k]) for k in z.files}, ("wall_gap_separator_mask",), shape=labels.shape, default=False)
    return {
        "scene": scene,
        "step": step_from_name(snapshot_path.stem),
        "input_npz": str(snapshot_path),
        "output_npz": str(out_npz),
        "output_png": str(out_png),
        "vertical_free_cells": int(np.count_nonzero(free)),
        "original_wall_cells": None,
        "bridge_cells": int(np.count_nonzero(bridge)),
        "separator_cells_in_vertical_free": int(np.count_nonzero(separator)),
        "labeled_cells": int(np.count_nonzero(labels)),
        "room_count": int(np.max(labels)) if labels.size else 0,
        "raw_component_count": None,
    }


def first_bool(
    arrays: dict[str, Any],
    keys: tuple[str, ...],
    *,
    shape: tuple[int, int] | None = None,
    default: bool | None = None,
) -> np.ndarray:
    for key in keys:
        if key not in arrays:
            continue
        arr = np.asarray(arrays[key], dtype=bool)
        if arr.ndim != 2:
            continue
        if shape is not None and arr.shape != shape:
            continue
        return arr.copy()
    if default is None:
        raise KeyError(keys)
    if shape is None:
        raise ValueError("shape is required when default is used")
    return np.full(shape, bool(default), dtype=bool)


def selected_bool_key(arrays: dict[str, Any], keys: tuple[str, ...], *, shape: tuple[int, int]) -> str | None:
    for key in keys:
        if key not in arrays:
            continue
        arr = np.asarray(arrays[key])
        if arr.ndim == 2 and arr.shape == shape:
            return key
    return None


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


def render_room_mask(
    out_png: Path,
    *,
    labels: np.ndarray,
    nav_free: np.ndarray,
    nav_occ: np.ndarray,
    nav_unknown: np.ndarray,
    separator: np.ndarray,
    min_area_cells: int,
) -> None:
    base = np.full((*labels.shape, 3), (154, 154, 154), dtype=np.uint8)
    base[nav_unknown] = (154, 154, 154)
    base[nav_free] = (246, 246, 242)
    base[nav_occ] = (58, 58, 58)
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
    image[separator] = (25, 25, 25)
    content = nav_free | nav_occ | nav_unknown | (labels > 0) | separator
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


def build_summary(
    *,
    source_root: Path,
    out_root: Path,
    result_root: Path,
    view_root: Path,
    rows: list[dict[str, Any]],
    params: WallGapParams,
    min_visual_room_area_m2: float,
) -> dict[str, Any]:
    return {
        "ablation": "vertical_free_wall_gap_completion",
        "algorithm": "axis_aligned_constrained_wall_gap_completion",
        "source_root": str(source_root),
        "out_root": str(out_root),
        "result_root": str(result_root),
        "room_mask_view_root": str(view_root),
        "scene_count": len({str(row["scene"]) for row in rows}),
        "snapshot_count": len(rows),
        "params": asdict(params),
        "total_vertical_free_cells": int(sum(int(row["vertical_free_cells"]) for row in rows)),
        "total_original_wall_cells": int(sum(int(row["original_wall_cells"] or 0) for row in rows)),
        "total_bridge_cells": int(sum(int(row["bridge_cells"]) for row in rows)),
        "total_separator_cells_in_vertical_free": int(sum(int(row["separator_cells_in_vertical_free"]) for row in rows)),
        "total_labeled_cells": int(sum(int(row["labeled_cells"]) for row in rows)),
        "total_final_rooms": int(sum(int(row["room_count"]) for row in rows)),
        "min_visual_room_area_m2": float(min_visual_room_area_m2),
        "note": (
            "This ablation does not use width-jump Voronoi. It bridges only short horizontal/vertical occupied-wall gaps "
            "with support on both sides, then labels 4-connected components of the vertical free map after removing "
            "the accepted bridge cells."
        ),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    params = summary["params"]
    lines = [
        "# Ablation 5: vertical free + wall gap completion",
        "",
        f"- Source root: `{summary['source_root']}`",
        f"- Result root: `{summary['result_root']}`",
        f"- Room mask views: `{summary['room_mask_view_root']}`",
        f"- Scene count: {summary['scene_count']}",
        f"- Snapshot count: {summary['snapshot_count']}",
        f"- Algorithm: `{summary['algorithm']}`",
        f"- Max gap cells: {params['max_gap_cells']}",
        f"- Min support cells: {params['min_support_cells']}",
        f"- Free guard radius cells: {params['free_guard_radius_cells']}",
        f"- Max free band fraction: {params['max_free_band_fraction']}",
        f"- Bridge perpendicular radius cells: {params['bridge_perpendicular_radius_cells']}",
        f"- Total bridge cells: {summary['total_bridge_cells']}",
        f"- Total separator cells in vertical free: {summary['total_separator_cells_in_vertical_free']}",
        f"- Total labeled cells: {summary['total_labeled_cells']}",
        f"- Total final rooms: {summary['total_final_rooms']}",
        "",
        summary["note"],
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
  <title>Vertical Free Wall Gap Completion Ablation</title>
  <style>
    body { font-family: sans-serif; margin: 18px; background: #f2f2f2; color: #222; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
    figure { margin: 0; background: white; padding: 8px; border: 1px solid #ddd; }
    img { width: 100%%; image-rendering: pixelated; display: block; }
    figcaption { font-size: 12px; margin-top: 6px; }
  </style>
</head>
<body>
  <h1>Vertical Free + Wall Gap Completion</h1>
  <div class="grid">
%s
  </div>
</body>
</html>
""" % ("\n".join(cards))
    path.write_text(doc, encoding="utf-8")


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


if __name__ == "__main__":
    raise SystemExit(main())
