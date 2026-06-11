from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage


DOOR_ACCEPTED_CUT_KEYS = (
    "voxel_door_cut_mask",
    "voxel_door_topology_effective_cut_mask",
    "voxel_door_topology_accepted_cut_mask",
    "voxel_current_door_cut_mask",
    "voxel_current_door_topology_effective_mask",
    "voxel_stable_door_cut_mask",
    "voxel_door_stable_cut_mask",
    "voxel_door_partition_cut_mask",
    "voxel_door_partition_cut_accepted_mask",
    "voxel_door_partition_effective_verified_mask",
    "voxel_door_final_cut_mask",
    "partial_door_extension_cut_mask",
    "door_completion_boundary_mask",
    "pre_extension_door_cut_mask",
    "strict_pre_extension_door_cut_mask",
)
DOOR_ACCEPTED_CUT_LEGACY_FALLBACK_KEYS = (
    "voxel_door_centerline_mask",
)
DOOR_SEED_DIAGNOSTIC_KEYS = (
    "voxel_door_raw_seed_mask",
    "voxel_door_seed_mask",
    "voxel_door_seed_line_primitive_mask",
    "voxel_door_extensible_primitive_mask",
    "voxel_door_centerline_visual_mask",
    "voxel_accepted_door_centerline_mask",
)
DOOR_EXCLUDED_ATTEMPT_KEYS = (
    "voxel_door_extension_trials_map",
    "voxel_door_extension_attempt_all_mask",
    "voxel_door_extension_attempt_selected_mask",
    "voxel_door_extension_attempt_rejected_mask",
    "voxel_door_rejected_lines_map",
    "voxel_door_partition_cut_candidate_mask",
    "voxel_door_partition_cut_rejected_mask",
    "voxel_door_visual_only_mask",
    "rejected_door_extension_mask",
)
EXCLUDED_EXTENSION_KEYS = (
    "voxel_line_supported_wall_map",
    "voxel_step2_extension_candidate_map",
    "voxel_step2_extension_separator_map",
    "voxel_step2_rejected_extension_map",
    "voxel_final_separator_map",
)
WALL_KEY = "voxel_wall_after_step1_map"
STRUCTURE_CLOSE = np.ones((3, 3), dtype=bool)
STRUCTURE_4 = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build vertical-free Voronoi source snapshots with step1 wall+door barriers.")
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--output-run-root", required=True)
    parser.add_argument("--map-resolution-m", type=float, default=0.05)
    parser.add_argument("--spur-max-length-m", type=float, default=0.3)
    parser.add_argument("--scene-glob", default="kujiale_*")
    parser.add_argument("--snapshot-glob", default="roomseg_step_*.npz")
    args = parser.parse_args(argv)

    source_root = Path(args.source_run_root)
    output_root = Path(args.output_run_root)
    resolution_m = float(args.map_resolution_m)
    spur_max_cells = max(0, int(round(float(args.spur_max_length_m) / resolution_m)))
    rows: list[dict[str, Any]] = []
    scene_dirs = sorted(p for p in source_root.glob(str(args.scene_glob)) if p.is_dir())
    for scene_dir in scene_dirs:
        scene = scene_dir.name
        out_snap_dir = output_root / scene / "roomseg_snapshots"
        out_snap_dir.mkdir(parents=True, exist_ok=True)
        for src_npz in sorted((scene_dir / "roomseg_snapshots").glob(str(args.snapshot_glob))):
            row = process_snapshot(
                source_npz=src_npz,
                output_npz=out_snap_dir / src_npz.name,
                scene=scene,
                resolution_m=resolution_m,
                spur_max_cells=spur_max_cells,
            )
            rows.append(row)

    manifest = {
        "schema_version": 1,
        "source_run_root": str(source_root),
        "derived_source_root": str(output_root),
        "snapshot_count": len(rows),
        "scene_count": len(scene_dirs),
        "wall_source_key": WALL_KEY,
        "wall_source_semantics": "step1 completed wall map: base wall plus real wall gap fill; excludes step2 line extension",
        "door_source_keys": list(DOOR_ACCEPTED_CUT_KEYS),
        "door_legacy_fallback_keys": list(DOOR_ACCEPTED_CUT_LEGACY_FALLBACK_KEYS),
        "door_source_semantics": (
            "VoxRoom 50b4d59 runtime accepted door cut union: explicit accepted door_cut/topology/partition masks win; "
            "legacy centerline fallback is used only when no explicit accepted cut key exists; diagnostic extension "
            "attempts, visual-only centerlines, and rejected lines are excluded from the barrier"
        ),
        "door_seed_diagnostic_keys": list(DOOR_SEED_DIAGNOSTIC_KEYS),
        "door_excluded_attempt_keys": list(DOOR_EXCLUDED_ATTEMPT_KEYS),
        "excluded_extension_keys": list(EXCLUDED_EXTENSION_KEYS),
        "smoothing": {
            "small_component_removal": False,
            "min_component_cells": None,
            "closing": "3x3 ones iterations=1",
            "spur_pruning": "remove endpoint branches that terminate at a junction within spur_max_length_cells",
            "spur_max_length_m": float(args.spur_max_length_m),
            "spur_max_length_cells": int(spur_max_cells),
            "barrier_dilation": "4-connectivity iterations=1",
        },
        "aggregate": _aggregate(rows),
        "rows": rows,
    }
    manifest_path = output_root.parent / f"{output_root.name}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("generated %d snapshots -> %s" % (len(rows), output_root))
    print("spur_removed_cells_total", manifest["aggregate"].get("spur_removed_cells_total", 0))
    print("avg_barrier_cells", manifest["aggregate"].get("barrier_cells_avg", 0.0))
    return 0


def process_snapshot(
    *,
    source_npz: Path,
    output_npz: Path,
    scene: str,
    resolution_m: float,
    spur_max_cells: int,
) -> dict[str, Any]:
    with np.load(source_npz, allow_pickle=False) as z:
        shape = _first_shape(z)
        vertical_free = _bool(z, "voxel_vertical_free_xy", shape)
        nav_free = _bool(z, "voxel_nav_free_xy", shape)
        nav_unknown = _bool(z, "voxel_nav_unknown_xy", shape)
        nav_occupied = _bool(z, "voxel_nav_occupied_xy", shape)
        wall = _bool(z, WALL_KEY, shape)
        door, door_explicit_keys_present, door_legacy_fallback_used = _door_runtime_cut_union(z, shape)
        door_seed_visual = _mask_union(z, DOOR_SEED_DIAGNOSTIC_KEYS, shape)
        raw = wall | door
        closed = ndimage.binary_closing(raw, structure=STRUCTURE_CLOSE, iterations=1).astype(bool)
        despined, spur_mask, spur_count = prune_endpoint_spurs(closed, max_length_cells=spur_max_cells)
        barrier = ndimage.binary_dilation(despined, structure=STRUCTURE_4, iterations=1).astype(bool)
        free_after = vertical_free & ~barrier
        unknown = nav_unknown & ~free_after
        obstacle = (nav_occupied | barrier) & ~free_after & ~unknown
        excluded_overlap = {key: int(np.count_nonzero(_bool(z, key, shape) & barrier)) for key in EXCLUDED_EXTENSION_KEYS}
        excluded_door_attempt_overlap = {
            key: int(np.count_nonzero(_bool(z, key, shape) & barrier)) for key in DOOR_EXCLUDED_ATTEMPT_KEYS
        }
        accepted_door_source_counts = {key: int(np.count_nonzero(_bool(z, key, shape))) for key in DOOR_ACCEPTED_CUT_KEYS}
        accepted_door_legacy_fallback_counts = {
            key: int(np.count_nonzero(_bool(z, key, shape))) for key in DOOR_ACCEPTED_CUT_LEGACY_FALLBACK_KEYS
        }
        door_seed_diagnostic_counts = {key: int(np.count_nonzero(_bool(z, key, shape))) for key in DOOR_SEED_DIAGNOSTIC_KEYS}

        arrays = {
            "occupancy_map": obstacle.astype(bool),
            "observed_free_mask": free_after.astype(bool),
            "navigation_free_room_domain": free_after.astype(bool),
            "vertical_free_room_domain": free_after.astype(bool),
            "unknown_mask": unknown.astype(bool),
            "obstacle_mask": obstacle.astype(bool),
            "voxel_vertical_free_xy": vertical_free.astype(bool),
            "voxroom_step1_completed_wall_mask": wall.astype(bool),
            "voxroom_door_line_mask": door.astype(bool),
            "voxroom_door_runtime_accepted_cut_mask": door.astype(bool),
            "voxroom_door_seed_diagnostic_mask": door_seed_visual.astype(bool),
            "voxroom_wall_door_no_extension_no_mincc_raw_mask": raw.astype(bool),
            "voxroom_wall_door_no_extension_no_mincc_closed_mask": closed.astype(bool),
            "voxroom_wall_door_no_extension_no_mincc_despurred_mask": despined.astype(bool),
            "voxroom_wall_door_no_extension_no_mincc_spur_removed_mask": spur_mask.astype(bool),
            "voxroom_wall_door_no_extension_no_mincc_barrier_mask": barrier.astype(bool),
            "map_resolution_m": np.asarray(resolution_m, dtype=np.float32),
            "map_width_cells": np.asarray(shape[1], dtype=np.int32),
            "map_height_cells": np.asarray(shape[0], dtype=np.int32),
            "source_snapshot_path": np.asarray(str(source_npz)),
            "input_map_variant": np.asarray("final_vertical_free_with_step1_wall_runtime_door_cut_no_extension_no_mincc_despur"),
            "input_free_source_key": np.asarray("voxel_vertical_free_xy_minus_step1_wall_runtime_door_cut_no_extension_no_mincc_despur"),
            "input_unknown_source_key": np.asarray("voxel_nav_unknown_xy_masked_outside_free_after_barrier"),
            "wall_source_key": np.asarray(WALL_KEY),
            "door_source_keys_json": np.asarray(json.dumps(list(DOOR_ACCEPTED_CUT_KEYS), ensure_ascii=False)),
            "door_explicit_keys_present_json": np.asarray(json.dumps(list(door_explicit_keys_present), ensure_ascii=False)),
            "door_legacy_fallback_keys_json": np.asarray(json.dumps(list(DOOR_ACCEPTED_CUT_LEGACY_FALLBACK_KEYS), ensure_ascii=False)),
            "door_legacy_fallback_used": np.asarray(bool(door_legacy_fallback_used)),
            "door_seed_diagnostic_keys_json": np.asarray(json.dumps(list(DOOR_SEED_DIAGNOSTIC_KEYS), ensure_ascii=False)),
            "door_excluded_attempt_keys_json": np.asarray(json.dumps(list(DOOR_EXCLUDED_ATTEMPT_KEYS), ensure_ascii=False)),
            "door_source_semantics": np.asarray(
                "50b4d59 explicit accepted door cuts only; legacy centerline fallback only if accepted cut keys are absent"
            ),
            "excluded_extension_keys_json": np.asarray(json.dumps(list(EXCLUDED_EXTENSION_KEYS), ensure_ascii=False)),
            "wall_door_smoothing_json": np.asarray(
                json.dumps(
                    {
                        "small_component_removal": False,
                        "min_component_cells": None,
                        "closing": "3x3 ones iterations=1",
                        "spur_pruning": True,
                        "spur_max_length_cells": int(spur_max_cells),
                        "barrier_dilation": "4-connectivity iterations=1",
                        "door_source_semantics": "50b4d59 explicit accepted door cut union",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
            "final_room_label_map": _copy_if_present(z, "final_room_label_map", np.zeros(shape, dtype=np.int32)),
            "agent_rc": _copy_if_present(z, "agent_rc", np.zeros((2,), dtype=np.int32)),
            "selected_frontier_mask": _copy_if_present(z, "selected_frontier_mask", np.zeros(shape, dtype=bool)),
            "frontier_map": _copy_if_present(z, "frontier_map", np.zeros(shape, dtype=bool)),
            "voxel_nav_free_xy": nav_free.astype(bool),
            "voxel_nav_unknown_xy": nav_unknown.astype(bool),
            "voxel_nav_occupied_xy": nav_occupied.astype(bool),
        }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    row = {
        "scene": scene,
        "snapshot": source_npz.name,
        "source_snapshot": str(source_npz),
        "output_snapshot": str(output_npz),
        "vertical_free_cells": int(np.count_nonzero(vertical_free)),
        "raw_wall_door_cells": int(np.count_nonzero(raw)),
        "runtime_accepted_door_cut_cells": int(np.count_nonzero(door)),
        "door_seed_diagnostic_cells": int(np.count_nonzero(door_seed_visual)),
        "closed_wall_door_cells": int(np.count_nonzero(closed)),
        "despurred_wall_door_cells": int(np.count_nonzero(despined)),
        "spur_removed_cells": int(np.count_nonzero(spur_mask)),
        "spur_removed_branch_count": int(spur_count),
        "barrier_cells": int(np.count_nonzero(barrier)),
        "barrier_overlap_vertical_free_cells": int(np.count_nonzero(barrier & vertical_free)),
        "free_after_barrier_cells": int(np.count_nonzero(free_after)),
        "unknown_cells": int(np.count_nonzero(unknown)),
        "occupied_cells": int(np.count_nonzero(obstacle)),
        "small_component_removal": False,
        "min_component_cells": None,
        "spur_max_length_cells": int(spur_max_cells),
        "wall_source_key": WALL_KEY,
        "door_source_keys": list(DOOR_ACCEPTED_CUT_KEYS),
        "door_explicit_keys_present": list(door_explicit_keys_present),
        "door_legacy_fallback_keys": list(DOOR_ACCEPTED_CUT_LEGACY_FALLBACK_KEYS),
        "door_legacy_fallback_used": bool(door_legacy_fallback_used),
        "door_seed_diagnostic_keys": list(DOOR_SEED_DIAGNOSTIC_KEYS),
        "door_excluded_attempt_keys": list(DOOR_EXCLUDED_ATTEMPT_KEYS),
        "accepted_door_source_counts": accepted_door_source_counts,
        "accepted_door_legacy_fallback_counts": accepted_door_legacy_fallback_counts,
        "door_seed_diagnostic_counts": door_seed_diagnostic_counts,
        "excluded_extension_keys": list(EXCLUDED_EXTENSION_KEYS),
        "excluded_key_overlap_with_barrier_cells": excluded_overlap,
        "excluded_door_attempt_overlap_with_barrier_cells": excluded_door_attempt_overlap,
    }
    output_npz.with_suffix(".summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return row


def prune_endpoint_spurs(mask: np.ndarray, *, max_length_cells: int) -> tuple[np.ndarray, np.ndarray, int]:
    out = np.asarray(mask, dtype=bool).copy()
    removed = np.zeros(out.shape, dtype=bool)
    if int(max_length_cells) <= 0:
        return out, removed, 0
    removed_branches = 0
    while True:
        degree = _degree4(out)
        endpoints = np.argwhere(out & (degree == 1))
        to_remove = np.zeros(out.shape, dtype=bool)
        for endpoint in endpoints:
            path, terminates_at_junction = _trace_endpoint_branch(out, degree, tuple(int(v) for v in endpoint), int(max_length_cells))
            if not terminates_at_junction:
                continue
            if len(path) < 2 or not _junction_has_straight_continuation(out, path[-1], path[-2]):
                continue
            if 0 < len(path) - 1 <= int(max_length_cells):
                for r, c in path[:-1]:
                    to_remove[int(r), int(c)] = True
        if not bool(np.any(to_remove)):
            break
        out[to_remove] = False
        removed |= to_remove
        removed_branches += int(ndimage.label(to_remove, structure=STRUCTURE_4)[1])
    return out, removed, int(removed_branches)


def _trace_endpoint_branch(
    mask: np.ndarray,
    degree: np.ndarray,
    start: tuple[int, int],
    max_length_cells: int,
) -> tuple[list[tuple[int, int]], bool]:
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    for _ in range(int(max_length_cells) + 1):
        neighbors = [n for n in _neighbors4(mask, cur) if n != prev]
        if len(neighbors) != 1:
            return path, len(neighbors) > 1
        nxt = neighbors[0]
        path.append(nxt)
        if int(degree[nxt]) >= 3:
            return path, True
        if int(degree[nxt]) <= 1:
            return path, False
        prev = cur
        cur = nxt
    return path, False


def _neighbors4(mask: np.ndarray, rc: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = rc
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(r) + dr
        cc = int(c) + dc
        if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and bool(mask[rr, cc]):
            out.append((rr, cc))
    return out


def _junction_has_straight_continuation(
    mask: np.ndarray,
    junction: tuple[int, int],
    branch_neighbor: tuple[int, int],
) -> bool:
    jr, jc = junction
    remaining = set(_neighbors4(mask, junction))
    remaining.discard(branch_neighbor)
    return bool(
        ((int(jr) - 1, int(jc)) in remaining and (int(jr) + 1, int(jc)) in remaining)
        or ((int(jr), int(jc) - 1) in remaining and (int(jr), int(jc) + 1) in remaining)
    )


def _degree4(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    degree = np.zeros(mask.shape, dtype=np.uint8)
    degree[1:, :] += mask[:-1, :]
    degree[:-1, :] += mask[1:, :]
    degree[:, 1:] += mask[:, :-1]
    degree[:, :-1] += mask[:, 1:]
    return degree


def _bool(z: np.lib.npyio.NpzFile, key: str, shape: tuple[int, int]) -> np.ndarray:
    if key in z.files:
        return np.asarray(z[key], dtype=bool)
    return np.zeros(shape, dtype=bool)


def _mask_union(z: np.lib.npyio.NpzFile, keys: Sequence[str], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for key in keys:
        out |= _bool(z, key, shape)
    return out


def _door_runtime_cut_union(z: np.lib.npyio.NpzFile, shape: tuple[int, int]) -> tuple[np.ndarray, tuple[str, ...], bool]:
    explicit_keys_present = tuple(key for key in DOOR_ACCEPTED_CUT_KEYS if key in z.files)
    if explicit_keys_present:
        return _mask_union(z, DOOR_ACCEPTED_CUT_KEYS, shape), explicit_keys_present, False
    fallback = _mask_union(z, DOOR_ACCEPTED_CUT_LEGACY_FALLBACK_KEYS, shape)
    return fallback, explicit_keys_present, bool(np.any(fallback))


def _first_shape(z: np.lib.npyio.NpzFile) -> tuple[int, int]:
    for key in ("voxel_vertical_free_xy", "voxel_nav_free_xy", "observed_free_mask", "occupancy_map"):
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim == 2:
                return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError("could not infer snapshot map shape")


def _copy_if_present(z: np.lib.npyio.NpzFile, key: str, default: np.ndarray) -> np.ndarray:
    if key in z.files:
        return np.asarray(z[key]).copy()
    return np.asarray(default).copy()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = (
        "raw_wall_door_cells",
        "runtime_accepted_door_cut_cells",
        "door_seed_diagnostic_cells",
        "closed_wall_door_cells",
        "despurred_wall_door_cells",
        "spur_removed_cells",
        "spur_removed_branch_count",
        "barrier_cells",
        "barrier_overlap_vertical_free_cells",
        "free_after_barrier_cells",
    )
    out: dict[str, Any] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows]
        out[f"{key}_total"] = int(sum(vals))
        out[f"{key}_avg"] = float(sum(vals) / len(vals))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
