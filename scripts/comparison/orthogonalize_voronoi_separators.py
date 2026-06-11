from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


STRUCTURE_4 = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-process native Voronoi labels into orthogonal separators snapped to wall convex corners."
    )
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--voronoi-run-root", required=True)
    parser.add_argument("--output-run-root", required=True)
    parser.add_argument("--map-resolution-m", type=float, default=0.05)
    parser.add_argument("--snap-distance-m", type=float, default=0.3)
    parser.add_argument("--corner-min-arm-m", type=float, default=0.3)
    parser.add_argument("--min-retained-length-ratio", type=float, default=0.75)
    parser.add_argument("--max-separator-length-m", type=float, default=0.0)
    parser.add_argument("--min-boundary-cells", type=int, default=3)
    parser.add_argument("--line-thickness-cells", type=int, default=1)
    parser.add_argument("--require-effective-cut", action="store_true")
    parser.add_argument("--min-created-region-area-m2", type=float, default=2.0)
    parser.add_argument("--extend-snap-ray-to-pair-domain-boundary", action="store_true")
    parser.add_argument("--merge-rejected-pairs", action="store_true")
    parser.add_argument("--reject-no-clearance-occupied-crossing", action="store_true")
    parser.add_argument("--skip-first-snapshot-per-scene", action="store_true")
    parser.add_argument("--baseline-name", default="voronoi_orthogonal_corner_snap")
    parser.add_argument("--input-baseline-name", default="voronoi")
    parser.add_argument("--disable-corner-snap", action="store_true")
    args = parser.parse_args(argv)

    source_root = Path(args.source_run_root)
    voronoi_root = Path(args.voronoi_run_root)
    output_root = Path(args.output_run_root)
    rows: list[dict[str, Any]] = []
    input_baseline_name = str(args.input_baseline_name)
    result_paths = sorted(voronoi_root.glob(f"*/baselines/{input_baseline_name}/roomseg_snapshots/roomseg_step_*.npz"))
    if bool(args.skip_first_snapshot_per_scene):
        result_paths = _skip_first_snapshot_per_scene(result_paths)
    for result_npz in result_paths:
        scene = result_npz.parts[-5]
        source_npz = source_root / scene / "roomseg_snapshots" / result_npz.name
        output_npz = output_root / scene / "baselines" / str(args.baseline_name) / "roomseg_snapshots" / result_npz.name
        row = process_snapshot(
            source_npz=source_npz,
            result_npz=result_npz,
            output_npz=output_npz,
            baseline_name=str(args.baseline_name),
            input_baseline_name=input_baseline_name,
            resolution_m=float(args.map_resolution_m),
            snap_distance_m=float(args.snap_distance_m),
            corner_min_arm_m=float(args.corner_min_arm_m),
            min_retained_length_ratio=float(args.min_retained_length_ratio),
            max_separator_length_m=float(args.max_separator_length_m),
            min_boundary_cells=int(args.min_boundary_cells),
            line_thickness_cells=int(args.line_thickness_cells),
            require_effective_cut=bool(args.require_effective_cut),
            min_created_region_area_m2=float(args.min_created_region_area_m2),
            extend_snap_ray_to_pair_domain_boundary=bool(args.extend_snap_ray_to_pair_domain_boundary),
            merge_rejected_pairs=bool(args.merge_rejected_pairs),
            reject_no_clearance_occupied_crossing=bool(args.reject_no_clearance_occupied_crossing),
            corner_snap_enabled=not bool(args.disable_corner_snap),
        )
        rows.append(row)

    manifest = {
        "schema_version": 1,
        "source_run_root": str(source_root),
        "voronoi_run_root": str(voronoi_root),
        "output_run_root": str(output_root),
        "baseline_name": str(args.baseline_name),
        "input_baseline_name": input_baseline_name,
        "map_resolution_m": float(args.map_resolution_m),
        "snap_distance_m": float(args.snap_distance_m),
        "snap_distance_cells": float(args.snap_distance_m) / float(args.map_resolution_m),
        "corner_min_arm_m": float(args.corner_min_arm_m),
        "corner_min_arm_cells": float(args.corner_min_arm_m) / float(args.map_resolution_m),
        "min_retained_length_ratio": float(args.min_retained_length_ratio),
        "max_separator_length_m": float(args.max_separator_length_m),
        "min_boundary_cells": int(args.min_boundary_cells),
        "line_thickness_cells": int(args.line_thickness_cells),
        "require_effective_cut": bool(args.require_effective_cut),
        "min_created_region_area_m2": float(args.min_created_region_area_m2),
        "extend_snap_ray_to_pair_domain_boundary": bool(args.extend_snap_ray_to_pair_domain_boundary),
        "merge_rejected_pairs": bool(args.merge_rejected_pairs),
        "reject_no_clearance_occupied_crossing": bool(args.reject_no_clearance_occupied_crossing),
        "skip_first_snapshot_per_scene": bool(args.skip_first_snapshot_per_scene),
        "corner_snap_enabled": not bool(args.disable_corner_snap),
        "snapshot_count": len(rows),
        "aggregate": _aggregate(rows),
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "orthogonal_corner_snap_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "processed %d snapshot(s), accepted %d/%d separator candidate(s) -> %s"
        % (
            len(rows),
            int(sum(int(r["accepted_separator_count"]) for r in rows)),
            int(sum(int(r["candidate_separator_count"]) for r in rows)),
            output_root,
        )
    )
    return 0


def process_snapshot(
    *,
    source_npz: Path,
    result_npz: Path,
    output_npz: Path,
    baseline_name: str,
    input_baseline_name: str,
    resolution_m: float,
    snap_distance_m: float,
    corner_min_arm_m: float,
    min_retained_length_ratio: float,
    max_separator_length_m: float,
    min_boundary_cells: int,
    line_thickness_cells: int,
    require_effective_cut: bool,
    min_created_region_area_m2: float,
    extend_snap_ray_to_pair_domain_boundary: bool,
    merge_rejected_pairs: bool,
    reject_no_clearance_occupied_crossing: bool,
    corner_snap_enabled: bool,
) -> dict[str, Any]:
    if not source_npz.exists():
        raise FileNotFoundError(source_npz)
    with np.load(source_npz, allow_pickle=False) as source_data, np.load(result_npz, allow_pickle=False) as result_data:
        source_arrays = {str(k): np.asarray(source_data[k]).copy() for k in source_data.files}
        result_arrays = {str(k): np.asarray(result_data[k]).copy() for k in result_data.files}

    label = np.asarray(result_arrays["final_room_label_map"], dtype=np.int32)
    free = _metric_domain(source_arrays)
    no_clearance_occupied = (
        _no_clearance_occupied_mask(source_arrays, free.shape)
        if bool(reject_no_clearance_occupied_crossing)
        else None
    )
    wall, wall_source_key = _wall_line_mask(source_arrays)
    corner_min_arm_cells = max(1, int(math.ceil(float(corner_min_arm_m) / float(resolution_m))))
    corner = detect_wall_convex_corners(wall, min_arm_cells=corner_min_arm_cells)
    candidates = extract_separator_candidates(
        label,
        free,
        corner,
        resolution_m=resolution_m,
        snap_distance_m=snap_distance_m,
        min_retained_length_ratio=min_retained_length_ratio,
        max_separator_length_m=max_separator_length_m,
        min_boundary_cells=min_boundary_cells,
        line_thickness_cells=line_thickness_cells,
        corner_snap_enabled=corner_snap_enabled,
    )
    accepted, rejected, separator, post_label = graph_merge_and_snap_separators(
        candidates,
        label,
        require_effective_cut=require_effective_cut,
        resolution_m=resolution_m,
        min_created_region_area_m2=min_created_region_area_m2,
        extend_snap_ray_to_pair_domain_boundary=extend_snap_ray_to_pair_domain_boundary,
        merge_rejected_pairs=merge_rejected_pairs,
        no_clearance_occupied=no_clearance_occupied,
        reject_no_clearance_occupied_crossing=reject_no_clearance_occupied_crossing,
    )
    post_count = _room_count(post_label)
    _, label_domain_component_count_before = ndimage.label(label > 0, structure=STRUCTURE_4)

    metadata = _metadata(
        result_arrays=result_arrays,
        baseline_name=baseline_name,
        input_baseline_name=input_baseline_name,
        resolution_m=resolution_m,
        snap_distance_m=snap_distance_m,
        corner_min_arm_m=corner_min_arm_m,
        corner_min_arm_cells=corner_min_arm_cells,
        min_retained_length_ratio=min_retained_length_ratio,
        max_separator_length_m=max_separator_length_m,
        line_thickness_cells=line_thickness_cells,
        require_effective_cut=require_effective_cut,
        min_created_region_area_m2=min_created_region_area_m2,
        extend_snap_ray_to_pair_domain_boundary=extend_snap_ray_to_pair_domain_boundary,
        merge_rejected_pairs=merge_rejected_pairs,
        reject_no_clearance_occupied_crossing=reject_no_clearance_occupied_crossing,
        corner_snap_enabled=corner_snap_enabled,
        candidate_count=len(candidates),
        accepted=accepted,
        rejected=rejected,
        original_room_count=_room_count(label),
        post_room_count=int(post_count),
        wall_corner_source_key=wall_source_key,
    )

    out = dict(result_arrays)
    out["voronoi_original_final_room_label_map"] = label
    out["final_room_label_map"] = post_label
    out["baseline_name"] = np.asarray(str(baseline_name))
    out["baseline_metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    out["orthogonal_corner_snap_separator_mask"] = separator.astype(bool)
    out["orthogonal_corner_snap_planned_separator_mask"] = _union_masks([_planned_mask(c) for c in accepted], free.shape)
    out["orthogonal_corner_snap_candidate_mask"] = _union_masks([_planned_mask(c) for c in candidates], free.shape)
    out["orthogonal_corner_snap_rejected_mask"] = _union_masks([_planned_mask(c) for c in rejected], free.shape)
    out["orthogonal_corner_snap_wall_corner_mask"] = corner.astype(bool)
    out["orthogonal_corner_snap_wall_corner_source_mask"] = wall.astype(bool)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)

    row = {
        "scene": result_npz.parts[-5],
        "snapshot": result_npz.name,
        "source_npz": str(source_npz),
        "voronoi_npz": str(result_npz),
        "output_npz": str(output_npz),
        "original_room_count": _room_count(label),
        "orthogonal_room_count": int(post_count),
        "candidate_separator_count": len(candidates),
        "accepted_separator_count": len(accepted),
        "rejected_separator_count": len(rejected),
        "rejected_no_corner_count": int(sum(1 for c in candidates if c["pre_reject_reason"] == "no_corner_within_snap_distance")),
        "rejected_not_cutting_count": int(
            sum(1 for c in rejected if c.get("reject_reason") in {"does_not_increase_components", "does_not_split_pair_domain"})
        ),
        "rejected_small_created_region_count": int(
            sum(1 for c in rejected if c.get("reject_reason") == "creates_small_region_after_global_scan")
        ),
        "rejected_no_clearance_occupied_crossing_count": int(
            sum(1 for c in rejected if c.get("reject_reason") == "crosses_no_clearance_occupied")
        ),
        "wall_corner_count": int(np.count_nonzero(corner)),
        "wall_corner_source_key": wall_source_key,
        "separator_cells": int(np.count_nonzero(separator)),
        "planned_separator_cells": int(np.count_nonzero(_union_masks([_planned_mask(c) for c in accepted], free.shape))),
        "effective_separator_cells": int(np.count_nonzero(separator)),
        "free_cells": int(np.count_nonzero(free)),
        "label_domain_cells": int(np.count_nonzero(label > 0)),
        "label_domain_component_count_before": int(label_domain_component_count_before),
        "label_domain_component_count_after": int(post_count),
        "actual_component_delta": int(post_count) - int(label_domain_component_count_before),
    }
    output_npz.with_suffix(".orthogonal_summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return row


def detect_wall_convex_corners(wall_mask: np.ndarray, *, min_arm_cells: int = 1) -> np.ndarray:
    """Return strict L-shaped right-angle corners with two sufficiently long wall arms.

    Endpoints are excluded because they have only one 4-neighbor. T/cross
    junctions are excluded because they have more than two 4-neighbors. Small
    cap artifacts are filtered by requiring both orthogonal arms to run for at
    least ``min_arm_cells`` cells.
    """
    wall = np.asarray(wall_mask, dtype=bool)
    up = np.zeros_like(wall)
    down = np.zeros_like(wall)
    left = np.zeros_like(wall)
    right = np.zeros_like(wall)
    up[1:, :] = wall[:-1, :]
    down[:-1, :] = wall[1:, :]
    left[:, 1:] = wall[:, :-1]
    right[:, :-1] = wall[:, 1:]
    neighbor_count = up.astype(np.uint8) + down + left + right
    strict_l = wall & (neighbor_count == 2) & (
        (up & left) | (left & down) | (down & right) | (right & up)
    )
    corners = np.zeros(wall.shape, dtype=bool)
    for r, c in np.argwhere(strict_l):
        dirs: list[tuple[int, int]] = []
        if up[r, c]:
            dirs.append((-1, 0))
        if down[r, c]:
            dirs.append((1, 0))
        if left[r, c]:
            dirs.append((0, -1))
        if right[r, c]:
            dirs.append((0, 1))
        if len(dirs) != 2:
            continue
        if all(_ray_length(wall, int(r), int(c), dr, dc) >= int(min_arm_cells) for dr, dc in dirs):
            corners[int(r), int(c)] = True
    return corners


def _ray_length(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    length = 0
    rr = int(r) + int(dr)
    cc = int(c) + int(dc)
    while 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and bool(mask[rr, cc]):
        length += 1
        rr += int(dr)
        cc += int(dc)
    return int(length)


def extract_separator_candidates(
    label: np.ndarray,
    free: np.ndarray,
    corner: np.ndarray,
    *,
    resolution_m: float,
    snap_distance_m: float,
    min_retained_length_ratio: float,
    max_separator_length_m: float,
    min_boundary_cells: int,
    line_thickness_cells: int,
    corner_snap_enabled: bool = True,
) -> list[dict[str, Any]]:
    label = np.asarray(label, dtype=np.int32)
    pair_coords: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for dr, dc in ((1, 0), (0, 1)):
        a = label[:-1, :] if dr else label[:, :-1]
        b = label[1:, :] if dr else label[:, 1:]
        valid = (a > 0) & (b > 0) & (a != b)
        for r, c in np.argwhere(valid):
            la = int(a[r, c])
            lb = int(b[r, c])
            pair = (min(la, lb), max(la, lb))
            pair_coords[pair].append((int(r), int(c)))
            pair_coords[pair].append((int(r + dr), int(c + dc)))

    corner_coords = np.argwhere(corner)
    corner_tree = cKDTree(corner_coords.astype(float)) if len(corner_coords) else None
    snap_distance_cells = float(snap_distance_m) / float(resolution_m)
    candidates: list[dict[str, Any]] = []
    candidate_id = 0
    for pair, coords_list in sorted(pair_coords.items()):
        mask = np.zeros(label.shape, dtype=bool)
        coords = np.asarray(coords_list, dtype=np.int32)
        if coords.size == 0:
            continue
        mask[coords[:, 0], coords[:, 1]] = True
        comp, count = ndimage.label(mask, structure=STRUCTURE_8)
        for comp_id in range(1, int(count) + 1):
            comp_coords = np.argwhere(comp == comp_id)
            if int(len(comp_coords)) < int(min_boundary_cells):
                continue
            candidate_id += 1
            cand = _orthogonal_candidate(
                candidate_id=candidate_id,
                pair=pair,
                coords=comp_coords,
                free=free,
                corner_coords=corner_coords,
                corner_tree=corner_tree,
                snap_distance_cells=snap_distance_cells,
                min_retained_length_ratio=float(min_retained_length_ratio),
                max_separator_length_m=float(max_separator_length_m),
                resolution_m=float(resolution_m),
                line_thickness_cells=line_thickness_cells,
                corner_snap_enabled=corner_snap_enabled,
            )
            if cand is not None:
                candidates.append(cand)
    candidates.sort(key=lambda x: (-int(x["boundary_cells"]), int(x["candidate_id"])))
    return candidates


def _orthogonal_candidate(
    *,
    candidate_id: int,
    pair: tuple[int, int],
    coords: np.ndarray,
    free: np.ndarray,
    corner_coords: np.ndarray,
    corner_tree: cKDTree | None,
    snap_distance_cells: float,
    min_retained_length_ratio: float,
    max_separator_length_m: float,
    resolution_m: float,
    line_thickness_cells: int,
    corner_snap_enabled: bool = True,
) -> dict[str, Any] | None:
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    row_span = int(r_max - r_min)
    col_span = int(c_max - c_min)
    if row_span == 0 and col_span == 0:
        return None
    axis = "vertical" if row_span > col_span else "horizontal"
    if axis == "horizontal":
        fixed = int(round(float(np.median(coords[:, 0]))))
        a0 = int(c_min)
        a1 = int(c_max)
        endpoints = [(fixed, a0), (fixed, a1)]
    else:
        fixed = int(round(float(np.median(coords[:, 1]))))
        a0 = int(r_min)
        a1 = int(r_max)
        endpoints = [(a0, fixed), (a1, fixed)]
    original_length_cells = abs(int(a1) - int(a0)) + 1
    if int(original_length_cells) < 2:
        return None
    pre_reject_reason = ""
    original_length_m = float(original_length_cells) * float(resolution_m)
    if float(max_separator_length_m) > 0.0 and original_length_m > float(max_separator_length_m) + 1.0e-9:
        pre_reject_reason = "too_long_separator"
    if not bool(corner_snap_enabled):
        snap = None
        snapped_endpoints = endpoints
        mask = _draw_axis_line(free.shape, axis, snapped_endpoints, line_thickness_cells)
        snapped_length_cells = int(np.count_nonzero(mask))
        retained_length_ratio = float(snapped_length_cells) / float(original_length_cells)
        corner_rc = None
        corner_distance_cells = None
        if not pre_reject_reason and int(snapped_length_cells) < 2:
            pre_reject_reason = "empty_without_corner_snap"
    else:
        snap = _nearest_endpoint_corner(endpoints, corner_coords, corner_tree, snap_distance_cells)
    if bool(corner_snap_enabled) and snap is None:
        if not pre_reject_reason:
            pre_reject_reason = "no_corner_within_snap_distance"
        mask = np.zeros(free.shape, dtype=bool)
        snapped_endpoints = endpoints
        corner_rc = None
        corner_distance_cells = None
    elif bool(corner_snap_enabled):
        endpoint_index, corner_rc, corner_distance_cells = snap
        snapped_endpoints = _translate_endpoint_to_corner(
            axis=axis,
            endpoints=endpoints,
            endpoint_index=endpoint_index,
            corner_rc=corner_rc,
        )
        mask = _draw_axis_line(free.shape, axis, snapped_endpoints, line_thickness_cells)
        snapped_length_cells = int(np.count_nonzero(mask))
        retained_length_ratio = float(snapped_length_cells) / float(original_length_cells)
        if not pre_reject_reason and int(snapped_length_cells) < 2:
            pre_reject_reason = "empty_after_snap"
        elif not pre_reject_reason and retained_length_ratio < float(min_retained_length_ratio):
            pre_reject_reason = "too_short_after_snap"
    if bool(corner_snap_enabled) and snap is None:
        snapped_length_cells = 0
        retained_length_ratio = 0.0
    return {
        "candidate_id": int(candidate_id),
        "label_pair": [int(pair[0]), int(pair[1])],
        "axis": axis,
        "line_thickness_cells": int(line_thickness_cells),
        "boundary_cells": int(len(coords)),
        "original_length_cells": int(original_length_cells),
        "original_length_m": float(original_length_m),
        "snapped_length_cells": int(snapped_length_cells),
        "retained_length_ratio": float(retained_length_ratio),
        "original_endpoints": [[int(r), int(c)] for r, c in endpoints],
        "snapped_endpoints": [[int(r), int(c)] for r, c in snapped_endpoints],
        "snap_endpoint_index": int(snap[0]) if snap is not None else None,
        "snap_corner_rc": [int(corner_rc[0]), int(corner_rc[1])] if corner_rc is not None else None,
        "snap_distance_cells": float(corner_distance_cells) if corner_distance_cells is not None else None,
        "corner_snap_enabled": bool(corner_snap_enabled),
        "pre_reject_reason": pre_reject_reason,
        "mask": mask.astype(bool),
    }


def _nearest_endpoint_corner(
    endpoints: list[tuple[int, int]],
    corner_coords: np.ndarray,
    corner_tree: cKDTree | None,
    max_distance_cells: float,
) -> tuple[int, tuple[int, int], float] | None:
    if corner_tree is None or len(corner_coords) == 0:
        return None
    best: tuple[int, tuple[int, int], float] | None = None
    for idx, endpoint in enumerate(endpoints):
        distance, corner_idx = corner_tree.query(np.asarray(endpoint, dtype=float), k=1)
        if float(distance) <= float(max_distance_cells):
            corner = tuple(int(v) for v in corner_coords[int(corner_idx)])
            if best is None or float(distance) < best[2]:
                best = (int(idx), corner, float(distance))
    return best


def _translate_endpoint_to_corner(
    *,
    axis: str,
    endpoints: list[tuple[int, int]],
    endpoint_index: int,
    corner_rc: tuple[int, int],
) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = endpoints
    if axis == "horizontal":
        length = abs(int(c1) - int(c0))
        if endpoint_index == 0:
            return [(int(corner_rc[0]), int(corner_rc[1])), (int(corner_rc[0]), int(corner_rc[1]) + length)]
        return [(int(corner_rc[0]), int(corner_rc[1]) - length), (int(corner_rc[0]), int(corner_rc[1]))]
    length = abs(int(r1) - int(r0))
    if endpoint_index == 0:
        return [(int(corner_rc[0]), int(corner_rc[1])), (int(corner_rc[0]) + length, int(corner_rc[1]))]
    return [(int(corner_rc[0]) - length, int(corner_rc[1])), (int(corner_rc[0]), int(corner_rc[1]))]


def _draw_axis_line(
    shape: tuple[int, int],
    axis: str,
    endpoints: list[tuple[int, int]],
    thickness_cells: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    (r0, c0), (r1, c1) = endpoints
    if axis == "horizontal":
        r = int(np.clip(r0, 0, shape[0] - 1))
        c_start, c_end = sorted((int(c0), int(c1)))
        c_start = int(np.clip(c_start, 0, shape[1] - 1))
        c_end = int(np.clip(c_end, 0, shape[1] - 1))
        mask[r, c_start : c_end + 1] = True
    else:
        c = int(np.clip(c0, 0, shape[1] - 1))
        r_start, r_end = sorted((int(r0), int(r1)))
        r_start = int(np.clip(r_start, 0, shape[0] - 1))
        r_end = int(np.clip(r_end, 0, shape[0] - 1))
        mask[r_start : r_end + 1, c] = True
    if int(thickness_cells) > 1:
        iterations = max(0, int(thickness_cells) - 1)
        mask = ndimage.binary_dilation(mask, structure=STRUCTURE_4, iterations=iterations)
    return mask.astype(bool)


def graph_merge_and_snap_separators(
    candidates: list[dict[str, Any]],
    label: np.ndarray,
    *,
    require_effective_cut: bool = False,
    resolution_m: float = 0.05,
    min_created_region_area_m2: float = 0.0,
    extend_snap_ray_to_pair_domain_boundary: bool = False,
    merge_rejected_pairs: bool = False,
    no_clearance_occupied: np.ndarray | None = None,
    reject_no_clearance_occupied_crossing: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    label = np.asarray(label, dtype=np.int32)
    domain = label > 0
    no_clearance_occupied_mask = (
        np.asarray(no_clearance_occupied, dtype=bool)
        if no_clearance_occupied is not None
        else np.zeros(label.shape, dtype=bool)
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        reason = str(candidate.get("pre_reject_reason", ""))
        if reason:
            item = dict(candidate)
            item["reject_reason"] = reason
            rejected.append(item)
            continue
        item = dict(candidate)
        if bool(extend_snap_ray_to_pair_domain_boundary):
            _attach_extended_effective_cut_masks(item, label)
        else:
            pair_domain = _pair_domain(label, item["label_pair"])
            item["planned_mask"] = np.asarray(item["mask"], dtype=bool)
            item["actual_mask"] = np.asarray(item["mask"], dtype=bool) & pair_domain
            _annotate_pair_domain_split(item, pair_domain)
        if bool(reject_no_clearance_occupied_crossing):
            planned_hits = int(np.count_nonzero(_planned_mask(item) & no_clearance_occupied_mask))
            actual_hits = int(np.count_nonzero(_actual_mask(item) & no_clearance_occupied_mask))
            item["no_clearance_occupied_crossing_cells"] = int(planned_hits)
            item["no_clearance_occupied_crossing_actual_cells"] = int(actual_hits)
            if int(planned_hits) > 0:
                item["reject_reason"] = "crosses_no_clearance_occupied"
                rejected.append(item)
                continue
        if bool(require_effective_cut) and not bool(item.get("effective_cut_splits_pair_domain", False)):
            item["reject_reason"] = "does_not_split_pair_domain"
            rejected.append(item)
            continue
        accepted.append(item)

    if float(min_created_region_area_m2) > 0.0 and accepted:
        accepted, small_region_rejected = _reject_candidates_creating_small_final_regions(
            accepted,
            label,
            resolution_m=float(resolution_m),
            min_created_region_area_m2=float(min_created_region_area_m2),
        )
        rejected.extend(small_region_rejected)

    dsu = _DisjointSet(int(label.max()))
    if bool(merge_rejected_pairs):
        merge_pairs = {tuple(int(v) for v in candidate["label_pair"]) for candidate in rejected}
    else:
        pair_source = accepted if bool(require_effective_cut) else candidates
        merge_pairs = {tuple(int(v) for v in candidate["label_pair"]) for candidate in pair_source}
    for pair in merge_pairs:
        dsu.union(pair[0], pair[1])
    split_pairs = {tuple(int(v) for v in candidate["label_pair"]) for candidate in accepted}
    affected_roots = {int(dsu.find(value)) for pair in split_pairs for value in pair}

    mapped = _map_labels_by_dsu_roots(label, dsu)
    separator = _union_masks([_actual_mask(candidate) for candidate in accepted], label.shape) & domain
    mapped[separator] = 0
    mapped[~domain] = 0
    return accepted, rejected, separator, _split_affected_roots_and_relabel(mapped, affected_roots)


def _attach_extended_effective_cut_masks(candidate: dict[str, Any], label: np.ndarray) -> None:
    planned = _extend_snap_ray(candidate, label.shape)
    pair_domain = _pair_domain(label, candidate["label_pair"])
    actual = planned & pair_domain
    candidate["planned_mask"] = planned.astype(bool)
    candidate["actual_mask"] = actual.astype(bool)
    candidate["planned_endpoints"] = _mask_axis_endpoints(planned, str(candidate["axis"]))
    candidate["planned_separator_cells"] = int(np.count_nonzero(planned))
    candidate["actual_separator_cells"] = int(np.count_nonzero(actual))
    _annotate_pair_domain_split(candidate, pair_domain)


def _extend_snap_ray(candidate: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    axis = str(candidate["axis"])
    corner = candidate.get("snap_corner_rc")
    endpoints = candidate.get("snapped_endpoints")
    if corner is None or not isinstance(endpoints, list) or len(endpoints) != 2:
        return np.asarray(candidate["mask"], dtype=bool)
    cr, cc = int(corner[0]), int(corner[1])
    points = [(int(r), int(c)) for r, c in endpoints]
    other = points[1]
    if points[1] == (cr, cc):
        other = points[0]
    elif points[0] != (cr, cc):
        other = max(points, key=lambda p: abs(int(p[0]) - cr) + abs(int(p[1]) - cc))
    if axis == "horizontal":
        step = 1 if int(other[1]) >= cc else -1
        end_c = shape[1] - 1 if step > 0 else 0
        return _draw_axis_line(shape, axis, [(cr, cc), (cr, end_c)], int(candidate.get("line_thickness_cells", 1)))
    step = 1 if int(other[0]) >= cr else -1
    end_r = shape[0] - 1 if step > 0 else 0
    return _draw_axis_line(shape, axis, [(cr, cc), (end_r, cc)], int(candidate.get("line_thickness_cells", 1)))


def _mask_axis_endpoints(mask: np.ndarray, axis: str) -> list[list[int]]:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        return []
    if str(axis) == "horizontal":
        row = int(round(float(np.median(coords[:, 0]))))
        return [[row, int(coords[:, 1].min())], [row, int(coords[:, 1].max())]]
    col = int(round(float(np.median(coords[:, 1]))))
    return [[int(coords[:, 0].min()), col], [int(coords[:, 0].max()), col]]


def _annotate_pair_domain_split(candidate: dict[str, Any], pair_domain: np.ndarray) -> None:
    actual = np.asarray(candidate.get("actual_mask", candidate["mask"]), dtype=bool) & pair_domain
    _, before = ndimage.label(pair_domain, structure=STRUCTURE_4)
    _, after = ndimage.label(pair_domain & ~actual, structure=STRUCTURE_4)
    candidate["effective_component_count_before"] = int(before)
    candidate["effective_component_count_after"] = int(after)
    candidate["effective_cut_splits_pair_domain"] = int(after) > int(before)
    candidate["actual_separator_cells"] = int(np.count_nonzero(actual))


def _reject_candidates_creating_small_final_regions(
    candidates: list[dict[str, Any]],
    label: np.ndarray,
    *,
    resolution_m: float,
    min_created_region_area_m2: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    domain = np.asarray(label, dtype=np.int32) > 0
    if not bool(np.any(domain)):
        return list(candidates), []
    separator = _union_masks([_actual_mask(candidate) for candidate in candidates], label.shape) & domain
    if not bool(np.any(separator)):
        return list(candidates), []

    cell_area = float(resolution_m) * float(resolution_m)
    min_area_cells = max(1.0, float(min_created_region_area_m2) / max(cell_area, 1.0e-12))
    original_components, _ = ndimage.label(domain, structure=STRUCTURE_4)
    original_areas = {
        int(component_id): int(np.count_nonzero(original_components == component_id))
        for component_id in np.unique(original_components)
        if int(component_id) > 0
    }
    split_components, _ = ndimage.label(domain & ~separator, structure=STRUCTURE_4)
    small_component_mask = np.zeros(label.shape, dtype=bool)
    small_area_by_label: dict[int, float] = {}
    for component_id in (int(v) for v in np.unique(split_components) if int(v) > 0):
        component = split_components == component_id
        area_cells = int(np.count_nonzero(component))
        if float(area_cells) >= min_area_cells:
            continue
        original_ids = [int(v) for v in np.unique(original_components[component]) if int(v) > 0]
        original_area = max((int(original_areas.get(v, 0)) for v in original_ids), default=0)
        if float(original_area) <= min_area_cells:
            continue
        small_component_mask |= component
        small_area_by_label[int(component_id)] = float(area_cells) * cell_area

    if not bool(np.any(small_component_mask)):
        return list(candidates), []

    small_touch_mask = ndimage.binary_dilation(small_component_mask, structure=STRUCTURE_4) & separator
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        line = _actual_mask(candidate) & domain
        if not bool(np.any(line & small_touch_mask)):
            accepted.append(candidate)
            continue
        item = dict(candidate)
        neighbor_labels = np.unique(split_components[ndimage.binary_dilation(line, structure=STRUCTURE_4) & small_component_mask])
        areas = [float(small_area_by_label.get(int(v), 0.0)) for v in neighbor_labels if int(v) > 0]
        item["reject_reason"] = "creates_small_region_after_global_scan"
        item["min_created_adjacent_region_area_m2"] = float(min(areas)) if areas else 0.0
        rejected.append(item)
    return accepted, rejected


def _pair_domain(label: np.ndarray, pair: Sequence[int]) -> np.ndarray:
    a, b = int(pair[0]), int(pair[1])
    return (np.asarray(label, dtype=np.int32) == a) | (np.asarray(label, dtype=np.int32) == b)


def _planned_mask(candidate: dict[str, Any]) -> np.ndarray:
    return np.asarray(candidate.get("planned_mask", candidate["mask"]), dtype=bool)


def _actual_mask(candidate: dict[str, Any]) -> np.ndarray:
    return np.asarray(candidate.get("actual_mask", candidate["mask"]), dtype=bool)


class _DisjointSet:
    def __init__(self, max_label: int) -> None:
        self.parent = list(range(int(max_label) + 1))

    def find(self, value: int) -> int:
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _candidate_can_split_domain(
    candidate: dict[str, Any],
    domain: np.ndarray,
    current_separator: np.ndarray,
    component_count: int,
) -> tuple[bool, str, int]:
    line = np.asarray(candidate["mask"], dtype=bool) & domain
    if int(np.count_nonzero(line)) < 2:
        return False, "snapped_line_does_not_touch_label_domain", int(component_count)
    if bool(np.any(line & current_separator)):
        return False, "intersects_existing_separator", int(component_count)
    _, new_count = ndimage.label(domain & ~(current_separator | line), structure=STRUCTURE_4)
    if int(new_count) <= int(component_count):
        return False, "does_not_increase_components", int(new_count)
    return True, "", int(new_count)


def _map_labels_by_dsu(label: np.ndarray, dsu: _DisjointSet) -> np.ndarray:
    label = np.asarray(label, dtype=np.int32)
    out = np.zeros(label.shape, dtype=np.int32)
    root_to_new: dict[int, int] = {}
    next_label = 1
    for value in sorted(int(v) for v in np.unique(label) if int(v) > 0):
        root = int(dsu.find(value))
        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1
        out[label == value] = np.int32(root_to_new[root])
    return out


def _map_labels_by_dsu_roots(label: np.ndarray, dsu: _DisjointSet) -> np.ndarray:
    label = np.asarray(label, dtype=np.int32)
    out = np.zeros(label.shape, dtype=np.int32)
    for value in sorted(int(v) for v in np.unique(label) if int(v) > 0):
        out[label == value] = np.int32(dsu.find(value))
    return out


def _relabel_consecutive(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.int32)
    out = np.zeros(label.shape, dtype=np.int32)
    for new_id, value in enumerate((int(v) for v in np.unique(label) if int(v) > 0), start=1):
        out[label == value] = np.int32(new_id)
    return out


def _split_and_relabel_components(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.int32)
    out = np.zeros(label.shape, dtype=np.int32)
    next_label = 1
    for value in (int(v) for v in np.unique(label) if int(v) > 0):
        component_map, count = ndimage.label(label == value, structure=STRUCTURE_4)
        for component_id in range(1, int(count) + 1):
            out[component_map == component_id] = np.int32(next_label)
            next_label += 1
    return out


def _split_affected_roots_and_relabel(label: np.ndarray, affected_roots: set[int]) -> np.ndarray:
    label = np.asarray(label, dtype=np.int32)
    out = np.zeros(label.shape, dtype=np.int32)
    next_label = 1
    for value in (int(v) for v in np.unique(label) if int(v) > 0):
        mask = label == value
        if value in affected_roots:
            component_map, count = ndimage.label(mask, structure=STRUCTURE_4)
            for component_id in range(1, int(count) + 1):
                out[component_map == component_id] = np.int32(next_label)
                next_label += 1
        else:
            out[mask] = np.int32(next_label)
            next_label += 1
    return out


def _metric_domain(arrays: dict[str, np.ndarray]) -> np.ndarray:
    nav = np.asarray(arrays.get("navigation_free_room_domain", arrays.get("observed_free_mask")), dtype=bool)
    unknown = np.asarray(arrays.get("unknown_mask", np.zeros(nav.shape, dtype=bool)), dtype=bool)
    obstacle = np.asarray(arrays.get("obstacle_mask", arrays.get("occupancy_map", np.zeros(nav.shape, dtype=bool))), dtype=bool)
    return nav & ~unknown & ~obstacle


def _no_clearance_occupied_mask(arrays: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if "voxel_nav_occupied_xy" not in arrays:
        raise KeyError("source snapshot missing voxel_nav_occupied_xy for no-clearance occupied crossing gate")
    occupied = np.asarray(arrays["voxel_nav_occupied_xy"], dtype=bool)
    if tuple(occupied.shape) != tuple(shape):
        raise ValueError(f"voxel_nav_occupied_xy shape {occupied.shape} does not match label shape {shape}")
    return occupied


def _wall_line_mask(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    for key in (
        "voxroom_wall_door_no_extension_no_mincc_despurred_mask",
        "voxroom_wall_door_no_extension_despurred_mask",
        "voxroom_wall_door_no_extension_no_mincc_closed_mask",
        "voxroom_wall_door_no_extension_closed_mask",
        "voxroom_wall_door_no_extension_no_mincc_raw_mask",
        "voxroom_wall_door_no_extension_raw_mask",
        "voxroom_step1_completed_wall_mask",
    ):
        if key in arrays:
            return np.asarray(arrays[key], dtype=bool), str(key)
    raise KeyError("source snapshot missing wall+door raw mask")


def _room_count(label: np.ndarray) -> int:
    return int(sum(1 for value in np.unique(np.asarray(label)) if int(value) > 0))


def _union_masks(masks: Iterable[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for mask in masks:
        out |= np.asarray(mask, dtype=bool)
    return out


def _skip_first_snapshot_per_scene(paths: Sequence[Path]) -> list[Path]:
    scene_to_paths: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(paths):
        scene_to_paths[str(path.parts[-5])].append(path)
    kept: list[Path] = []
    for _, scene_paths in sorted(scene_to_paths.items()):
        kept.extend(sorted(scene_paths)[1:])
    return kept


def _metadata(
    *,
    result_arrays: dict[str, np.ndarray],
    baseline_name: str,
    input_baseline_name: str,
    resolution_m: float,
    snap_distance_m: float,
    corner_min_arm_m: float,
    corner_min_arm_cells: int,
    min_retained_length_ratio: float,
    max_separator_length_m: float,
    line_thickness_cells: int,
    require_effective_cut: bool,
    min_created_region_area_m2: float,
    extend_snap_ray_to_pair_domain_boundary: bool,
    merge_rejected_pairs: bool,
    reject_no_clearance_occupied_crossing: bool,
    corner_snap_enabled: bool,
    candidate_count: int,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    original_room_count: int,
    post_room_count: int,
    wall_corner_source_key: str,
) -> dict[str, Any]:
    parent: dict[str, Any] = {}
    if "baseline_metadata_json" in result_arrays:
        try:
            parent = json.loads(str(np.asarray(result_arrays["baseline_metadata_json"]).item()))
        except Exception:
            parent = {}
    return {
        "method": str(baseline_name),
        "parent_method": parent.get("method", str(input_baseline_name)),
        "parent_runner_type": parent.get("runner_type"),
        "parent_original_algorithm_id": parent.get("original_algorithm_id"),
        "postprocess_type": "orthogonal_separator_corner_snap" if bool(corner_snap_enabled) else "orthogonal_separator_no_corner_snap",
        "main_experiment_allowed": False,
        "uses_original_voronoi_labels": True,
        "corner_snap_enabled": bool(corner_snap_enabled),
        "snap_distance_m": float(snap_distance_m),
        "snap_distance_cells": float(snap_distance_m) / float(resolution_m),
        "corner_min_arm_m": float(corner_min_arm_m),
        "corner_min_arm_cells": int(corner_min_arm_cells),
        "min_retained_length_ratio": float(min_retained_length_ratio),
        "max_separator_length_m": float(max_separator_length_m),
        "line_thickness_cells": int(line_thickness_cells),
        "require_effective_cut": bool(require_effective_cut),
        "min_created_region_area_m2": float(min_created_region_area_m2),
        "extend_snap_ray_to_pair_domain_boundary": bool(extend_snap_ray_to_pair_domain_boundary),
        "merge_rejected_pairs": bool(merge_rejected_pairs),
        "reject_no_clearance_occupied_crossing": bool(reject_no_clearance_occupied_crossing),
        "wall_corner_source_key": str(wall_corner_source_key),
        "axis_policy": "force each extracted Voronoi boundary component to horizontal or vertical by dominant span",
        "corner_policy": (
            "disabled: this baseline is a new algorithm whose separators are not forced to wall convex corners"
            if not bool(corner_snap_enabled)
            else "one endpoint must be within snap distance of a strict L-shaped right-angle corner from the completed and smoothed step1 wall+door line; endpoints, T junctions, cross junctions, and cap artifacts without two long orthogonal arms are not anchors"
        ),
        "length_policy": (
            "without corner snapping, the orthogonalized separator keeps its extracted label-boundary span"
            if not bool(corner_snap_enabled)
            else "the snapped separator keeps the original candidate length; candidates clipped below the retained-length ratio are deleted"
        ),
        "extension_policy": (
            "no corner ray extension is required when corner snapping is disabled; planned mask is the extracted axis-aligned separator unless a caller explicitly supplies another policy"
            if not bool(corner_snap_enabled)
            else "orthogonal_corner_snap_planned_separator_mask stores the full snapped line before label-domain clipping for later extension"
        ),
        "delete_policy": (
            "delete candidate only if it is too short or does not satisfy the requested effective-cut policy"
            if not bool(corner_snap_enabled)
            else "delete candidate only if no endpoint is close to a true wall corner or if the snapped full-length line is too short; extension-ready lines are retained even when they do not yet increase connected components"
        ),
        "effective_cut_policy": (
            "when require_effective_cut is enabled, retain a candidate only if its actual pair-domain-clipped separator "
            "increases that pair domain's connected-component count; only retained pairs are merged and re-cut"
        ),
        "created_region_area_policy": (
            "after all candidate scan/snap masks are collected, reject any separator whose final combined split touches "
            "a newly created region smaller than min_created_region_area_m2"
        ),
        "no_clearance_occupied_crossing_policy": (
            "when enabled, reject any planned separator whose full line crosses at least one occupied cell in "
            "the original no-clearance navigation map voxel_nav_occupied_xy"
        ),
        "rejected_pair_policy": (
            "when merge_rejected_pairs is enabled, native Voronoi adjacencies that fail validation are merged instead of "
            "being preserved as room boundaries"
        ),
        "original_room_count": int(original_room_count),
        "post_room_count": int(post_room_count),
        "candidate_separator_count": int(candidate_count),
        "accepted_separator_count": int(len(accepted)),
        "rejected_separator_count": int(len(rejected)),
        "accepted_separators": [_json_candidate(c) for c in accepted],
        "rejected_reason_counts": _reason_counts(rejected),
    }


def _json_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in candidate.items() if k not in {"mask", "planned_mask", "actual_mask"}}


def _reason_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        reason = str(item.get("reject_reason", "unknown"))
        counts[reason] = int(counts.get(reason, 0) + 1)
    return counts


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    numeric_keys = (
        "original_room_count",
        "orthogonal_room_count",
        "candidate_separator_count",
        "accepted_separator_count",
        "rejected_separator_count",
        "rejected_no_corner_count",
        "rejected_not_cutting_count",
        "rejected_small_created_region_count",
        "rejected_no_clearance_occupied_crossing_count",
        "separator_cells",
        "planned_separator_cells",
        "effective_separator_cells",
        "actual_component_delta",
    )
    out: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(r[key]) for r in rows]
        out[f"{key}_sum"] = int(sum(values))
        out[f"{key}_avg"] = float(sum(values) / len(values))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
