from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight vertical-free baseline source snapshots.")
    parser.add_argument("--source-run-root", required=True, type=Path)
    parser.add_argument("--output-run-root", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_run_root
    output_root = args.output_run_root
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for snapshot_path in sorted(source_root.glob("kujiale_*/roomseg_snapshots/roomseg_step_*.npz")):
        scene = snapshot_path.parents[1].name
        out_path = output_root / scene / "roomseg_snapshots" / snapshot_path.name
        if out_path.exists() and not args.force:
            rows.append({"scene_id": scene, "snapshot": snapshot_path.name, "output": str(out_path), "status": "skipped_exists"})
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with np.load(snapshot_path, allow_pickle=False) as z:
            vertical_free = required_bool(z, ("voxel_vertical_free_xy", "vertical_free_room_domain"))
            shape = vertical_free.shape
            nav_unknown = optional_bool(z, ("voxel_nav_unknown_xy", "voxel_unknown_xy", "unknown_mask"), shape)
            nav_occupied = optional_bool(z, ("voxel_nav_occupied_xy", "obstacle_mask", "occupancy_map"), shape)
            unknown = nav_unknown & ~vertical_free
            occupied = nav_occupied & ~vertical_free & ~unknown
            labels = np.zeros(shape, dtype=np.int32)
            arrays: dict[str, np.ndarray] = {
                "occupancy_map": occupied.astype(bool),
                "obstacle_mask": occupied.astype(bool),
                "unknown_mask": unknown.astype(bool),
                "observed_free_mask": vertical_free.astype(bool),
                "navigation_free_room_domain": vertical_free.astype(bool),
                "vertical_free_room_domain": vertical_free.astype(bool),
                "voxel_vertical_free_xy": vertical_free.astype(bool),
                "voxel_baseline_input_free_xy": vertical_free.astype(bool),
                "voxel_nav_occupied_xy": occupied.astype(bool),
                "voxel_nav_unknown_xy": unknown.astype(bool),
                "final_room_label_map": labels,
                "input_variant_name": np.asarray("vertical_free_lightweight"),
                "input_free_source_key": np.asarray("voxel_vertical_free_xy"),
                "input_variant_semantics": np.asarray("pure vertical free map from voxel_vertical_free_xy; lightweight 2D source"),
                "source_snapshot_path": np.asarray(str(snapshot_path)),
            }
            for key in (
                "map_resolution_m",
                "map_origin_x_m",
                "map_origin_y_m",
                "map_origin_xy_m",
                "map_min_x_m",
                "map_max_x_m",
                "map_min_y_m",
                "map_max_y_m",
                "map_width_cells",
                "map_height_cells",
            ):
                if key in z.files:
                    arrays[key] = np.asarray(z[key]).copy()
            if "map_resolution_m" not in arrays:
                arrays["map_resolution_m"] = np.asarray(0.05, dtype=np.float32)
            arrays.setdefault("map_width_cells", np.asarray(int(shape[1]), dtype=np.int32))
            arrays.setdefault("map_height_cells", np.asarray(int(shape[0]), dtype=np.int32))
            arrays.setdefault("map_origin_x_m", np.asarray(0.0, dtype=np.float32))
            arrays.setdefault("map_origin_y_m", np.asarray(0.0, dtype=np.float32))
            arrays.setdefault("map_origin_xy_m", np.asarray([0.0, 0.0], dtype=np.float32))
            resolution = float(np.asarray(arrays["map_resolution_m"]).reshape(-1)[0])
            arrays.setdefault("map_min_x_m", np.asarray(0.0, dtype=np.float32))
            arrays.setdefault("map_min_y_m", np.asarray(0.0, dtype=np.float32))
            arrays.setdefault("map_max_x_m", np.asarray(float(shape[1]) * resolution, dtype=np.float32))
            arrays.setdefault("map_max_y_m", np.asarray(float(shape[0]) * resolution, dtype=np.float32))
            arrays["input_variant_manifest_json"] = np.asarray(
                json.dumps(
                    {
                        "variant": "vertical_free_lightweight",
                        "free_source": "voxel_vertical_free_xy",
                        "free_cells": int(np.count_nonzero(vertical_free)),
                        "occupied_cells": int(np.count_nonzero(occupied)),
                        "unknown_cells": int(np.count_nonzero(unknown)),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            np.savez_compressed(out_path, **arrays)
        rows.append(
            {
                "scene_id": scene,
                "snapshot": snapshot_path.name,
                "input": str(snapshot_path),
                "output": str(out_path),
                "status": "written",
                "free_cells": int(np.count_nonzero(vertical_free)),
                "occupied_cells": int(np.count_nonzero(occupied)),
                "unknown_cells": int(np.count_nonzero(unknown)),
            }
        )
    manifest = {
        "schema_version": 1,
        "variant": "vertical_free_lightweight",
        "source_run_root": str(source_root),
        "output_run_root": str(output_root),
        "snapshot_count": len(rows),
        "scene_count": len({r["scene_id"] for r in rows}),
        "rows": rows,
    }
    (output_root / "lightweight_vertical_free_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"scene_count": manifest["scene_count"], "snapshot_count": manifest["snapshot_count"]}, ensure_ascii=False))
    return 0


def required_bool(z: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key not in z.files:
            continue
        arr = np.asarray(z[key], dtype=bool)
        if arr.ndim == 2 and np.any(arr):
            return arr.copy()
    raise KeyError("missing non-empty bool array from %s" % (keys,))


def optional_bool(z: np.lib.npyio.NpzFile, keys: tuple[str, ...], shape: tuple[int, int]) -> np.ndarray:
    for key in keys:
        if key not in z.files:
            continue
        arr = np.asarray(z[key], dtype=bool)
        if arr.shape == shape:
            return arr.copy()
    return np.zeros(shape, dtype=bool)


if __name__ == "__main__":
    raise SystemExit(main())
