from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify orthogonal corner-snap Voronoi postprocess outputs.")
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--postprocess-run-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--native-run-root", default="")
    parser.add_argument("--native-baseline-name", default="voronoi")
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--visualization-root", default="")
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--skip-first-snapshot-per-scene", action="store_true")
    args = parser.parse_args(argv)

    repo = Path.cwd()
    orthogonal = _load_orthogonal_module(repo / "scripts/comparison/orthogonalize_voronoi_separators.py")
    source_root = Path(args.source_run_root)
    post_root = Path(args.postprocess_run_root)
    native_root = Path(args.native_run_root) if args.native_run_root else None
    native_baseline_name = str(args.native_baseline_name)
    output_json = Path(args.output_json)
    manifest_path = post_root / "orthogonal_corner_snap_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = _read_json(Path(args.source_manifest)) if args.source_manifest else {}
    vis_summary = _read_json(Path(args.visualization_root) / "summary.json") if args.visualization_root else {}

    source_npzs = sorted(source_root.glob("*/roomseg_snapshots/roomseg_step_*.npz"))
    post_npzs = sorted(
        post_root.glob(f"*/baselines/{str(args.baseline_name)}/roomseg_snapshots/roomseg_step_*.npz")
    )
    native_npzs = (
        sorted(native_root.glob(f"*/baselines/{native_baseline_name}/roomseg_snapshots/roomseg_step_*.npz"))
        if native_root is not None
        else []
    )
    if bool(args.skip_first_snapshot_per_scene):
        source_npzs = _skip_first_snapshot_per_scene(source_npzs)
        native_npzs = _skip_first_snapshot_per_scene(native_npzs)
    overlays = sorted((Path(args.visualization_root) / "overlays").glob("*/*.png")) if args.visualization_root else []
    scene_sheets = sorted((Path(args.visualization_root) / "scene_sheets").glob("*.png")) if args.visualization_root else []

    corner_min_arm_cells = int(math.ceil(float(manifest["corner_min_arm_m"]) / float(manifest["map_resolution_m"])))
    snap_distance_cells = float(manifest["snap_distance_cells"])
    min_ratio = float(manifest["min_retained_length_ratio"])
    line_thickness = int(manifest["line_thickness_cells"])
    corner_snap_enabled = bool(manifest.get("corner_snap_enabled", True))
    max_separator_length_m = float(manifest.get("max_separator_length_m", 0.0))

    shape_failures: list[dict[str, Any]] = []
    metadata_failures: list[dict[str, Any]] = []
    corner_failures: list[dict[str, Any]] = []
    separator_failures: list[dict[str, Any]] = []
    accepted_total = 0
    rejected_total = 0
    planned_cells_total = 0
    actual_cells_total = 0
    effective_split_total = 0
    positive_label_outputs = 0
    axis_counts: dict[str, int] = {"horizontal": 0, "vertical": 0}
    max_snap_distance = 0.0
    min_retained = float("inf")

    for post_npz in post_npzs:
        rel = str(post_npz.relative_to(post_root))
        with np.load(post_npz, allow_pickle=False) as z:
            required = (
                "baseline_metadata_json",
                "final_room_label_map",
                "voronoi_original_final_room_label_map",
                "orthogonal_corner_snap_separator_mask",
                "orthogonal_corner_snap_planned_separator_mask",
                "orthogonal_corner_snap_wall_corner_mask",
                "orthogonal_corner_snap_wall_corner_source_mask",
            )
            missing = [key for key in required if key not in z.files]
            if missing:
                metadata_failures.append({"snapshot": rel, "missing_keys": missing})
                continue

            metadata = json.loads(str(np.asarray(z["baseline_metadata_json"]).item()))
            label = np.asarray(z["final_room_label_map"], dtype=np.int32)
            original = np.asarray(z["voronoi_original_final_room_label_map"], dtype=np.int32)
            actual = np.asarray(z["orthogonal_corner_snap_separator_mask"], dtype=bool)
            planned = np.asarray(z["orthogonal_corner_snap_planned_separator_mask"], dtype=bool)
            saved_corner = np.asarray(z["orthogonal_corner_snap_wall_corner_mask"], dtype=bool)
            wall = np.asarray(z["orthogonal_corner_snap_wall_corner_source_mask"], dtype=bool)

            if label.shape != original.shape or label.shape != actual.shape or label.shape != planned.shape:
                shape_failures.append({"snapshot": rel, "reason": "shape mismatch"})
                continue
            if int(np.max(label)) > 0:
                positive_label_outputs += 1

            recomputed_corner = orthogonal.detect_wall_convex_corners(wall, min_arm_cells=corner_min_arm_cells)
            if not np.array_equal(saved_corner, recomputed_corner):
                corner_failures.append({"snapshot": rel, "reason": "saved strict corner mask differs from recomputed mask"})

            expected_postprocess_type = "orthogonal_separator_corner_snap" if corner_snap_enabled else "orthogonal_separator_no_corner_snap"
            if metadata.get("postprocess_type") != expected_postprocess_type:
                metadata_failures.append(
                    {"snapshot": rel, "reason": "unexpected postprocess_type", "value": metadata.get("postprocess_type")}
                )

            accepted = list(metadata.get("accepted_separators", []))
            accepted_total += len(accepted)
            rejected_total += int(metadata.get("rejected_separator_count", 0))
            reconstructed = np.zeros(label.shape, dtype=bool)

            for index, separator in enumerate(accepted):
                axis = str(separator.get("axis"))
                axis_counts[axis] = int(axis_counts.get(axis, 0) + 1)
                if axis not in {"horizontal", "vertical"}:
                    separator_failures.append({"snapshot": rel, "index": index, "reason": "axis is not horizontal/vertical"})
                    continue

                corner = separator.get("snap_corner_rc")
                if corner_snap_enabled:
                    if corner is None:
                        separator_failures.append({"snapshot": rel, "index": index, "reason": "missing snap corner"})
                        continue
                    cr, cc = (int(corner[0]), int(corner[1]))
                    if not (0 <= cr < saved_corner.shape[0] and 0 <= cc < saved_corner.shape[1] and bool(saved_corner[cr, cc])):
                        separator_failures.append(
                            {"snapshot": rel, "index": index, "reason": "snap corner is not a valid saved strict L corner"}
                        )

                    snap_distance = float(separator.get("snap_distance_cells"))
                    max_snap_distance = max(max_snap_distance, snap_distance)
                    if snap_distance > snap_distance_cells + 1e-6:
                        separator_failures.append(
                            {"snapshot": rel, "index": index, "reason": "snap distance exceeds threshold", "value": snap_distance}
                        )
                elif corner is not None:
                    separator_failures.append({"snapshot": rel, "index": index, "reason": "corner present when corner snap disabled"})

                retained = float(separator.get("retained_length_ratio"))
                min_retained = min(min_retained, retained)
                if retained + 1e-9 < min_ratio:
                    separator_failures.append(
                        {"snapshot": rel, "index": index, "reason": "retained length ratio below threshold", "value": retained}
                    )
                original_length_m = float(separator.get("original_length_m", 0.0))
                if max_separator_length_m > 0.0 and original_length_m > max_separator_length_m + 1.0e-9:
                    separator_failures.append(
                        {
                            "snapshot": rel,
                            "index": index,
                            "reason": "separator length exceeds max",
                            "value_m": original_length_m,
                            "max_m": max_separator_length_m,
                        }
                    )

                endpoints = separator.get("planned_endpoints") or separator.get("snapped_endpoints")
                if not isinstance(endpoints, list) or len(endpoints) != 2:
                    separator_failures.append({"snapshot": rel, "index": index, "reason": "invalid planned/snapped endpoints"})
                    continue
                (r0, c0), (r1, c1) = [[int(v) for v in point] for point in endpoints]
                if axis == "horizontal" and r0 != r1:
                    separator_failures.append({"snapshot": rel, "index": index, "reason": "horizontal line is not row-constant"})
                if axis == "vertical" and c0 != c1:
                    separator_failures.append({"snapshot": rel, "index": index, "reason": "vertical line is not column-constant"})

                drawn = orthogonal._draw_axis_line(label.shape, axis, [(r0, c0), (r1, c1)], line_thickness)
                reconstructed |= drawn
                if corner_snap_enabled and not bool(drawn[cr, cc]):
                    separator_failures.append({"snapshot": rel, "index": index, "reason": "drawn line does not include corner"})
                if bool(metadata.get("require_effective_cut", False)):
                    if not bool(separator.get("effective_cut_splits_pair_domain", False)):
                        separator_failures.append(
                            {"snapshot": rel, "index": index, "reason": "accepted separator does not split pair domain"}
                        )
                    else:
                        effective_split_total += 1

            if not np.array_equal(reconstructed, planned):
                separator_failures.append(
                    {
                        "snapshot": rel,
                        "reason": "planned separator mask differs from accepted metadata reconstruction",
                        "expected_cells": int(np.count_nonzero(reconstructed)),
                        "actual_cells": int(np.count_nonzero(planned)),
                    }
                )
            if bool(np.any(actual & ~planned)):
                separator_failures.append({"snapshot": rel, "reason": "actual separator has cells outside planned separator"})
            planned_cells_total += int(np.count_nonzero(planned))
            actual_cells_total += int(np.count_nonzero(actual))

    expected_count = int(args.expected_count) if int(args.expected_count) > 0 else int(manifest["snapshot_count"])
    checks = {
        "source_count_matches_expected": len(source_npzs) == expected_count,
        "post_count_matches_expected": len(post_npzs) == expected_count,
        "native_count_matches_expected": native_root is None or len(native_npzs) == expected_count,
        "manifest_count_matches_expected": int(manifest["snapshot_count"]) == expected_count,
        "all_outputs_have_positive_labels": positive_label_outputs == len(post_npzs),
        "all_shapes_valid": not shape_failures,
        "all_required_metadata_present": not metadata_failures,
        "all_saved_corner_masks_match_recomputed_strict_l_corners": not corner_failures,
        "all_accepted_separators_are_axis_aligned_and_corner_snapped": not separator_failures,
        "planned_separator_cell_sum_matches_manifest": planned_cells_total == int(manifest["aggregate"]["planned_separator_cells_sum"]),
        "actual_separator_cell_sum_matches_manifest": actual_cells_total == int(manifest["aggregate"]["separator_cells_sum"]),
        "accepted_separator_count_matches_manifest": accepted_total == int(manifest["aggregate"]["accepted_separator_count_sum"]),
        "source_small_component_removal_disabled": not source_manifest
        or (
            source_manifest.get("smoothing", {}).get("small_component_removal") is False
            and source_manifest.get("smoothing", {}).get("min_component_cells") is None
        ),
        "visualization_complete": not args.visualization_root
        or (
            vis_summary.get("status") == "complete"
            and int(vis_summary.get("snapshot_count", -1)) == expected_count
            and len(overlays) == expected_count
            and len(scene_sheets) > 0
            and (Path(args.visualization_root) / "preview_sheet.png").exists()
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "repo": str(repo),
        "source_run_root": str(source_root),
        "native_run_root": str(native_root) if native_root is not None else "",
        "native_baseline_name": native_baseline_name,
        "postprocess_run_root": str(post_root),
        "visualization_root": str(args.visualization_root),
        "baseline_name": str(args.baseline_name),
        "corner_snap_enabled": corner_snap_enabled,
        "map_resolution_m": float(manifest["map_resolution_m"]),
        "corner_min_arm_m": float(manifest["corner_min_arm_m"]),
        "snap_distance_m": float(manifest["snap_distance_m"]),
        "min_retained_length_ratio": float(manifest["min_retained_length_ratio"]),
        "max_separator_length_m": max_separator_length_m,
        "counts": {
            "source_npz": len(source_npzs),
            "native_npz": len(native_npzs),
            "post_npz": len(post_npzs),
            "overlay_png": len(overlays),
            "scene_sheets": len(scene_sheets),
            "positive_label_outputs": int(positive_label_outputs),
            "candidate_separators": int(manifest["aggregate"]["candidate_separator_count_sum"]),
            "accepted_separators": int(accepted_total),
            "rejected_separators": int(rejected_total),
            "planned_separator_cells": int(planned_cells_total),
            "actual_separator_cells": int(actual_cells_total),
            "effective_split_separators": int(effective_split_total),
        },
        "separator_quality": {
            "axis_counts": axis_counts,
            "max_snap_distance_cells": float(max_snap_distance),
            "max_snap_distance_m": float(max_snap_distance) * float(manifest["map_resolution_m"]),
            "min_retained_length_ratio_observed": None if min_retained == float("inf") else float(min_retained),
        },
        "checks": checks,
        "failure_samples": {
            "shape_failures": shape_failures[:10],
            "metadata_failures": metadata_failures[:10],
            "corner_failures": corner_failures[:10],
            "separator_failures": separator_failures[:10],
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_json": str(output_json), "counts": report["counts"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _skip_first_snapshot_per_scene(paths: Sequence[Path]) -> list[Path]:
    scene_to_paths: dict[str, list[Path]] = {}
    for path in sorted(paths):
        if "baselines" in path.parts:
            scene = str(path.parts[path.parts.index("baselines") - 1])
        elif "roomseg_snapshots" in path.parts:
            scene = str(path.parts[path.parts.index("roomseg_snapshots") - 1])
        else:
            scene = str(path.parent.name)
        scene_to_paths.setdefault(scene, []).append(path)
    kept: list[Path] = []
    for _, scene_paths in sorted(scene_to_paths.items()):
        kept.extend(sorted(scene_paths)[1:])
    return kept


def _load_orthogonal_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("orthogonalize_voronoi_separators", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
