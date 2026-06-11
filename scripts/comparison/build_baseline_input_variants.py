from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


VARIANTS = ("nav_no_clearance", "vertical_free", "vertical_door_extended")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build source roots for baseline input-map variants.")
    parser.add_argument("--stateless-run-root", required=True)
    parser.add_argument("--door-extended-run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args(argv)

    stateless_root = Path(args.stateless_run_root)
    door_root = Path(args.door_extended_run_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stateless_run_root": str(stateless_root),
        "door_extended_run_root": str(door_root),
        "output_root": str(output_root),
        "variants": {},
    }
    for variant in [str(v) for v in args.variants]:
        source_root = door_root if variant == "vertical_door_extended" else stateless_root
        variant_root = output_root / variant
        rows = _build_variant(source_root, variant_root, variant=variant)
        manifest["variants"][variant] = {
            "source_root": str(source_root),
            "output_root": str(variant_root),
            "snapshot_count": len(rows),
            "scene_count": len({row["scene_id"] for row in rows}),
            "free_cells_total": int(sum(row["free_cells"] for row in rows)),
            "free_cells_avg": float(sum(row["free_cells"] for row in rows) / max(len(rows), 1)),
            "rows": rows,
        }
    manifest_path = output_root / "input_variant_manifest.json"
    manifest_path.write_text(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: data["snapshot_count"] for name, data in manifest["variants"].items()}, ensure_ascii=False))
    return 0


def _build_variant(source_root: Path, output_root: Path, *, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot_path in sorted(source_root.glob("kujiale_*/roomseg_snapshots/roomseg_step_*.npz")):
        scene_id = snapshot_path.parents[1].name
        out_path = output_root / scene_id / "roomseg_snapshots" / snapshot_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with np.load(snapshot_path, allow_pickle=False) as data:
            arrays = {str(k): np.asarray(data[k]).copy() for k in data.files}
        free, free_source = _variant_free(arrays, variant=variant)
        shape = free.shape
        unknown = _first_bool(arrays, ("voxel_nav_unknown_xy", "voxel_unknown_xy", "unknown_mask"), shape)
        occupied = _first_bool(arrays, ("voxel_nav_occupied_xy", "obstacle_mask", "occupancy_map"), shape)
        if unknown is None:
            unknown = np.zeros(shape, dtype=bool)
        if occupied is None:
            occupied = np.zeros(shape, dtype=bool)
        unknown = np.asarray(unknown, dtype=bool) & ~free
        occupied = np.asarray(occupied, dtype=bool) & ~free & ~unknown

        arrays["navigation_free_room_domain"] = free.astype(bool)
        arrays["observed_free_mask"] = free.astype(bool)
        arrays["vertical_free_room_domain"] = free.astype(bool)
        arrays["voxel_baseline_input_free_xy"] = free.astype(bool)
        arrays["unknown_mask"] = unknown.astype(bool)
        arrays["voxel_nav_unknown_xy"] = unknown.astype(bool)
        arrays["obstacle_mask"] = occupied.astype(bool)
        arrays["occupancy_map"] = occupied.astype(bool)
        arrays["voxel_nav_occupied_xy"] = occupied.astype(bool)
        arrays["input_variant_name"] = np.asarray(str(variant))
        arrays["input_free_source_key"] = np.asarray(str(free_source))
        arrays["input_variant_semantics"] = np.asarray(_variant_semantics(variant))
        arrays["input_variant_manifest_json"] = np.asarray(
            json.dumps(
                {
                    "variant": variant,
                    "free_source": free_source,
                    "free_cells": int(np.count_nonzero(free)),
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
                "scene_id": scene_id,
                "snapshot": snapshot_path.name,
                "input": str(snapshot_path),
                "output": str(out_path),
                "variant": variant,
                "free_source": free_source,
                "free_cells": int(np.count_nonzero(free)),
                "occupied_cells": int(np.count_nonzero(occupied)),
                "unknown_cells": int(np.count_nonzero(unknown)),
            }
        )
    return rows


def _variant_free(arrays: Mapping[str, np.ndarray], *, variant: str) -> tuple[np.ndarray, str]:
    shape = _shape(arrays)
    if variant == "nav_no_clearance":
        return _required_bool(arrays, ("voxel_nav_free_xy", "observed_free_mask", "navigation_free_room_domain"), shape), "voxel_nav_free_xy"
    if variant == "vertical_free":
        return _required_bool(arrays, ("voxel_vertical_free_xy", "vertical_free_room_domain"), shape), "voxel_vertical_free_xy"
    if variant == "vertical_door_extended":
        return _required_bool(arrays, ("navigation_free_room_domain", "observed_free_mask", "vertical_free_room_domain"), shape), "navigation_free_room_domain"
    raise ValueError("unsupported variant: %s" % variant)


def _variant_semantics(variant: str) -> str:
    if variant == "nav_no_clearance":
        return "no-clearance navigation free map from voxel_nav_free_xy"
    if variant == "vertical_free":
        return "pure vertical free map from voxel_vertical_free_xy"
    if variant == "vertical_door_extended":
        return "vertical free map after runtime wall and door-extension barrier cuts"
    return str(variant)


def _shape(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    for key in ("voxel_nav_free_xy", "observed_free_mask", "navigation_free_room_domain", "voxel_vertical_free_xy", "occupancy_map"):
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError("cannot infer snapshot shape")


def _required_bool(arrays: Mapping[str, np.ndarray], keys: tuple[str, ...], shape: tuple[int, int]) -> np.ndarray:
    for key in keys:
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=bool)
        if arr.shape == shape and bool(arr.any()):
            return arr.copy()
    raise KeyError("missing non-empty free source from %s" % (keys,))


def _first_bool(arrays: Mapping[str, np.ndarray], keys: tuple[str, ...], shape: tuple[int, int]) -> np.ndarray | None:
    for key in keys:
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=bool)
        if arr.shape == shape:
            return arr.copy()
    return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
