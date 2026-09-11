#!/usr/bin/env python3
"""Build TVARS, Gomez, and approved-GT room maps for one paper snapshot.

The three outputs share the current snapshot's Vertical-Free partition domain
and no-clearance Nav-Free projection.  TVARS contributes its actually saved
persistent door lines.  Gomez is replayed incrementally from the paired saved
TVARS range-jump points and confirms them from the saved 3-D voxel state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from voxroom_online.isaac_runtime.baselines.data_contract import load_npz_arrays
from voxroom_online.isaac_runtime.baselines.mask_io import (
    SEGMENTATION_INPUT_MODE_KEY,
    relabel_consecutive,
)
from voxroom_online.isaac_runtime.baselines.offline.gomez_runner import (
    GomezIncrementalRunner,
)
from voxroom_online.isaac_runtime.baselines.tvars_original.adapter import (
    _filter_and_project_partition_labels,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.annotation_schema import (
    load_annotation,
)
from voxroom_online.isaac_runtime.evaluation.online_roomseg.mask_generation import (
    GtGenerationConfig,
    generate_gt_from_annotation,
)
from voxroom_online.isaac_runtime.evaluation.roomseg_coverage_milestones import (
    partition_free_space_by_door_lines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--comparison-events-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--events",
        nargs="+",
        default=["milestone_020", "milestone_040", "milestone_060", "milestone_070"],
    )
    parser.add_argument("--map-resolution-m", type=float, default=0.05)
    parser.add_argument("--min-room-area-m2", type=float, default=0.5)
    parser.add_argument(
        "--gt-force-vertical-line-id",
        help=(
            "Presentation-only GT correction: keep the selected line's row "
            "extent and place both endpoints at its rounded mean column."
        ),
    )
    return parser.parse_args()


def _only_npz(directory: Path) -> Path:
    paths = sorted(directory.glob("*.npz"))
    if len(paths) != 1:
        raise ValueError(f"expected exactly one NPZ in {directory}, found {len(paths)}")
    return paths[0].resolve()


def _partition_and_project(
    vertical_free: np.ndarray,
    navigation_free: np.ndarray,
    door_lines: np.ndarray,
    *,
    resolution_m: float,
    min_room_area_m2: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertical_labels, partition_debug = partition_free_space_by_door_lines(
        vertical_free,
        door_lines,
    )
    _filtered_vertical, projected, projection_debug = (
        _filter_and_project_partition_labels(
            vertical_labels,
            navigation_free=navigation_free,
            resolution_m=float(resolution_m),
            min_room_area_m2=float(min_room_area_m2),
        )
    )
    return relabel_consecutive(projected), {
        **dict(partition_debug),
        **dict(projection_debug),
    }


def _align_label_ids_by_overlap(
    labels: np.ndarray,
    reference_labels: np.ndarray,
    *,
    domain: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    source = np.asarray(labels, dtype=np.int32)
    reference = np.asarray(reference_labels, dtype=np.int32)
    valid = np.asarray(domain, dtype=bool)
    if source.shape != reference.shape or valid.shape != source.shape:
        raise ValueError("color-alignment arrays must share one shape")
    source_ids = [int(value) for value in np.unique(source[valid]) if int(value) > 0]
    reference_ids = [
        int(value) for value in np.unique(reference[valid]) if int(value) > 0
    ]
    overlap = np.zeros((len(source_ids), len(reference_ids)), dtype=np.int64)
    for source_index, source_id in enumerate(source_ids):
        source_mask = valid & (source == source_id)
        for reference_index, reference_id in enumerate(reference_ids):
            overlap[source_index, reference_index] = int(
                np.count_nonzero(source_mask & (reference == reference_id))
            )
    mapping: dict[int, int] = {}
    if overlap.size:
        source_indices, reference_indices = linear_sum_assignment(-overlap)
        mapping.update(
            {
                source_ids[int(source_index)]: reference_ids[int(reference_index)]
                for source_index, reference_index in zip(
                    source_indices.tolist(), reference_indices.tolist()
                )
            }
        )
    used_ids = set(mapping.values())
    next_id = 1
    for source_id in source_ids:
        if source_id in mapping:
            continue
        while next_id in used_ids:
            next_id += 1
        mapping[source_id] = next_id
        used_ids.add(next_id)
    aligned = np.zeros(source.shape, dtype=np.int32)
    for source_id, target_id in mapping.items():
        aligned[source == int(source_id)] = int(target_id)
    return aligned, mapping, overlap


def main() -> int:
    args = parse_args()
    base_snapshot = args.base_snapshot.expanduser().resolve()
    events_root = args.comparison_events_root.expanduser().resolve()
    annotation_path = args.annotation.expanduser().resolve()
    output = args.output.expanduser().resolve()

    with np.load(base_snapshot, allow_pickle=False) as arrays:
        vertical_free = np.asarray(arrays["voxel_vertical_free_xy"], dtype=bool)
        navigation_free = np.asarray(arrays["voxel_nav_free_xy"], dtype=bool)
        obstacle = np.asarray(arrays["voxel_nav_occupied_xy"], dtype=bool)
        voxroom_labels = np.asarray(
            arrays["voxel_final_room_label_map"], dtype=np.int32
        )
        shape = navigation_free.shape
        base_origin = (
            float(np.asarray(arrays["map_origin_x_m"]).reshape(())),
            float(np.asarray(arrays["map_origin_y_m"]).reshape(())),
        )

    tvars_70_path = _only_npz(
        events_root
        / "milestone_070"
        / "tvars_original"
        / "roomseg_snapshots"
    )
    with np.load(tvars_70_path, allow_pickle=False) as arrays:
        tvars_door_lines = np.asarray(
            arrays["tvars_original_door_line_map"], dtype=bool
        )
        tvars_origin = (
            float(np.asarray(arrays["map_origin_x_m"]).reshape(())),
            float(np.asarray(arrays["map_origin_y_m"]).reshape(())),
        )
    if tvars_door_lines.shape != shape:
        raise ValueError("TVARS door-line map does not match the paper grid")
    if not np.allclose(base_origin, tvars_origin, atol=1.0e-6, rtol=0.0):
        raise ValueError(
            f"TVARS/base grid origins differ: base={base_origin}, tvars={tvars_origin}"
        )
    tvars_labels, tvars_debug = _partition_and_project(
        vertical_free,
        navigation_free,
        tvars_door_lines,
        resolution_m=float(args.map_resolution_m),
        min_room_area_m2=float(args.min_room_area_m2),
    )

    gomez = GomezIncrementalRunner(map_resolution_m=float(args.map_resolution_m))
    gomez.start_scene("kujiale_0003")
    gomez_result = None
    gomez_replay_rows: list[dict[str, Any]] = []
    try:
        for event_id in args.events:
            voxroom_path = _only_npz(
                events_root / event_id / "voxroom" / "roomseg_snapshots"
            )
            tvars_path = _only_npz(
                events_root / event_id / "tvars_original" / "roomseg_snapshots"
            )
            arrays = load_npz_arrays(voxroom_path)
            with np.load(tvars_path, allow_pickle=False) as tvars_arrays:
                raw_points = np.asarray(
                    tvars_arrays["tvars_original_hough_door_seed_map"],
                    dtype=bool,
                )
            arrays["tvars_original_hough_door_seed_map"] = raw_points
            arrays["tvars_original_raw_seed_source_snapshot"] = np.asarray(
                str(tvars_path)
            )
            arrays[SEGMENTATION_INPUT_MODE_KEY] = np.asarray("raw_vertical_free")
            gomez_result = gomez.segment_snapshot(voxroom_path, arrays)
            gomez_replay_rows.append(
                {
                    "event_id": str(event_id),
                    "voxroom_snapshot": str(voxroom_path),
                    "tvars_snapshot": str(tvars_path),
                    "raw_seed_point_count": int(np.count_nonzero(raw_points)),
                    "persistent_accepted_door_count": int(
                        gomez_result.metadata["door_line_accepted_count_persistent"]
                    ),
                }
            )
    finally:
        gomez.end_scene()
    if gomez_result is None:
        raise RuntimeError("Gomez replay produced no result")
    gomez_door_lines = np.asarray(
        gomez_result.debug_arrays["gomez_accepted_door_separator_mask"],
        dtype=bool,
    )
    gomez_labels, gomez_debug = _partition_and_project(
        vertical_free,
        navigation_free,
        gomez_door_lines,
        resolution_m=float(args.map_resolution_m),
        min_room_area_m2=float(args.min_room_area_m2),
    )

    annotation = load_annotation(annotation_path)
    if annotation.review.status != "approved":
        raise ValueError("Ground Truth annotation is not approved")
    gt_split_lines = list(annotation.split_lines)
    gt_line_override: dict[str, Any] | None = None
    if args.gt_force_vertical_line_id:
        selected_id = str(args.gt_force_vertical_line_id)
        matching = [
            (index, line)
            for index, line in enumerate(gt_split_lines)
            if str(line.id) == selected_id
        ]
        if len(matching) != 1:
            raise ValueError(
                f"expected one GT line named {selected_id}, found {len(matching)}"
            )
        index, line = matching[0]
        center_col = int(round((int(line.p0_rc[1]) + int(line.p1_rc[1])) / 2.0))
        replacement = replace(
            line,
            p0_rc=(int(line.p0_rc[0]), center_col),
            p1_rc=(int(line.p1_rc[0]), center_col),
        )
        gt_split_lines[index] = replacement
        gt_line_override = {
            "line_id": selected_id,
            "original_p0_rc": list(line.p0_rc),
            "original_p1_rc": list(line.p1_rc),
            "rendered_p0_rc": list(replacement.p0_rc),
            "rendered_p1_rc": list(replacement.p1_rc),
            "source_annotation_modified": False,
        }
    gt_result = generate_gt_from_annotation(
        eval_domain=navigation_free,
        segmentation_domain=vertical_free,
        obstacle_mask=obstacle,
        split_lines=gt_split_lines,
        merge_groups=annotation.merge_groups,
        config=GtGenerationConfig(
            line_width_cells=int(annotation.line_width_cells_default),
            preclose_radius_cells=int(annotation.preclose_radius_cells),
            connectivity=4,
            min_room_area_cells=int(annotation.min_room_area_cells),
            assign_cut_pixels=True,
        ),
    )
    ground_truth_labels = relabel_consecutive(gt_result.labels)
    ground_truth_door_lines = np.asarray(gt_result.line_mask, dtype=bool)
    (
        ground_truth_voxroom_color_aligned_labels,
        ground_truth_to_voxroom_label_mapping,
        ground_truth_voxroom_overlap,
    ) = _align_label_ids_by_overlap(
        ground_truth_labels,
        voxroom_labels,
        domain=navigation_free,
    )

    for name, value in (
        ("tvars_label_map", tvars_labels),
        ("tvars_door_line_map", tvars_door_lines),
        ("gomez_label_map", gomez_labels),
        ("gomez_door_line_map", gomez_door_lines),
        ("ground_truth_label_map", ground_truth_labels),
        (
            "ground_truth_voxroom_color_aligned_label_map",
            ground_truth_voxroom_color_aligned_labels,
        ),
        ("ground_truth_door_line_map", ground_truth_door_lines),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} differs from {shape}")

    metadata = {
        "schema": "kujiale_70pct_paper_comparison_maps_v1",
        "base_snapshot": str(base_snapshot),
        "map_shape": list(shape),
        "map_origin_xy_m": list(base_origin),
        "map_resolution_m": float(args.map_resolution_m),
        "partition_domain": "voxel_vertical_free_xy",
        "projection_domain": "voxel_nav_free_xy_no_clearance",
        "min_room_area_m2": float(args.min_room_area_m2),
        "tvars": {
            "persistent_door_source": str(tvars_70_path),
            "door_line_cells": int(np.count_nonzero(tvars_door_lines)),
            "room_count": int(np.max(tvars_labels)),
            "labeled_navigation_cells": int(np.count_nonzero(tvars_labels)),
            "partition_debug": tvars_debug,
        },
        "gomez": {
            "variant": "saved_tvars_raw_seed_plus_voxel_hough",
            "checkpoint_virtual_laser_recomputed": False,
            "replay": gomez_replay_rows,
            "door_line_cells": int(np.count_nonzero(gomez_door_lines)),
            "room_count": int(np.max(gomez_labels)),
            "labeled_navigation_cells": int(np.count_nonzero(gomez_labels)),
            "partition_debug": gomez_debug,
        },
        "ground_truth": {
            "annotation": str(annotation_path),
            "review_status": str(annotation.review.status),
            "door_line_cells": int(np.count_nonzero(ground_truth_door_lines)),
            "room_count": int(np.max(ground_truth_labels)),
            "labeled_navigation_cells": int(np.count_nonzero(ground_truth_labels)),
            "generation_debug": gt_result.metadata,
            "presentation_line_override": gt_line_override,
            "color_alignment": {
                "reference_method": "voxroom",
                "matching": "hungarian_maximum_room_overlap_on_no_clearance_nav_free",
                "ground_truth_label_to_voxroom_label": {
                    str(source): int(target)
                    for source, target in sorted(
                        ground_truth_to_voxroom_label_mapping.items()
                    )
                },
                "overlap_matrix_cells": ground_truth_voxroom_overlap.tolist(),
                "segmentation_geometry_modified": False,
            },
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            tvars_label_map=np.asarray(tvars_labels, dtype=np.int32),
            tvars_door_line_map=np.asarray(tvars_door_lines, dtype=bool),
            gomez_label_map=np.asarray(gomez_labels, dtype=np.int32),
            gomez_door_line_map=np.asarray(gomez_door_lines, dtype=bool),
            ground_truth_label_map=np.asarray(ground_truth_labels, dtype=np.int32),
            ground_truth_voxroom_color_aligned_label_map=np.asarray(
                ground_truth_voxroom_color_aligned_labels,
                dtype=np.int32,
            ),
            ground_truth_door_line_map=np.asarray(
                ground_truth_door_lines, dtype=bool
            ),
            metadata_json=np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            ),
        )
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
