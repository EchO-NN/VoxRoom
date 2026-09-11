#!/usr/bin/env python3
"""Run original IPA Voronoi on the 70% no-clearance Nav-Free map."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _relabel(raw: np.ndarray, domain: np.ndarray) -> np.ndarray:
    out = np.zeros(raw.shape, dtype=np.int32)
    labels = [int(value) for value in np.unique(raw[domain]) if int(value) > 0]
    for new_label, old_label in enumerate(labels, start=1):
        out[domain & (raw == old_label)] = int(new_label)
    return out


def _align(source: np.ndarray, reference: np.ndarray, domain: np.ndarray):
    source_ids = [int(value) for value in np.unique(source[domain]) if int(value) > 0]
    reference_ids = [int(value) for value in np.unique(reference[domain]) if int(value) > 0]
    overlap = np.zeros((len(source_ids), len(reference_ids)), dtype=np.int64)
    for i, source_id in enumerate(source_ids):
        for j, reference_id in enumerate(reference_ids):
            overlap[i, j] = int(
                np.count_nonzero(
                    domain & (source == source_id) & (reference == reference_id)
                )
            )
    mapping: dict[int, int] = {}
    if overlap.size:
        rows, columns = linear_sum_assignment(-overlap)
        mapping.update(
            {
                source_ids[int(row)]: reference_ids[int(column)]
                for row, column in zip(rows.tolist(), columns.tolist())
            }
        )
    used = set(mapping.values())
    next_id = 1
    for source_id in source_ids:
        if source_id in mapping:
            continue
        while next_id in used:
            next_id += 1
        mapping[source_id] = next_id
        used.add(next_id)
    aligned = np.zeros(source.shape, dtype=np.int32)
    for source_id, target_id in mapping.items():
        aligned[source == source_id] = target_id
    return aligned, mapping, overlap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.snapshot.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with np.load(snapshot, allow_pickle=False) as arrays:
        nav_free = np.asarray(arrays["voxel_nav_free_xy"], dtype=bool)
        voxroom = np.asarray(arrays["voxel_final_room_label_map"], dtype=np.int32)
    if nav_free.shape != voxroom.shape:
        raise ValueError("Nav-Free and VoxRoom label maps have different shapes")

    input_path = prefix.with_name(prefix.name + "_ipa_input.png")
    raw_path = prefix.with_name(prefix.name + "_raw_labels.yml.gz")
    input_image = np.zeros(nav_free.shape, dtype=np.uint8)
    input_image[nav_free] = 255
    if not cv2.imwrite(str(input_path), input_image):
        raise RuntimeError("failed to write IPA input image")
    subprocess.run([str(runner), str(input_path), str(raw_path)], check=True)
    storage = cv2.FileStorage(str(raw_path), cv2.FILE_STORAGE_READ)
    raw = storage.getNode("segmented_map").mat()
    storage.release()
    if raw is None:
        raise RuntimeError("IPA output has no segmented_map")
    raw = np.asarray(raw, dtype=np.int32)
    if raw.shape != nav_free.shape:
        raise ValueError("IPA output shape differs from input")

    labels = _relabel(raw, nav_free)
    separator = nav_free & (labels <= 0)
    aligned, mapping, overlap = _align(labels, voxroom, nav_free)
    metadata = {
        "schema": "kujiale_70pct_ipa_voronoi_navfree_map_v1",
        "method": "voronoi",
        "runner_type": "standalone_original_ipa_cpp",
        "source_implementation": "ipa320/ipa_coverage_planning VoronoiSegmentation",
        "source_snapshot": str(snapshot),
        "segmentation_input_mode": "raw_nav_free_no_clearance",
        "segmentation_input_key": "voxel_nav_free_xy",
        "segmentation_input_cells": int(np.count_nonzero(nav_free)),
        "room_count_before_color_alignment": int(labels.max()),
        "room_count_after_color_alignment": int(len([v for v in np.unique(aligned) if int(v) > 0])),
        "unassigned_separator_cells_on_navigation_free": int(np.count_nonzero(separator)),
        "parameters": {
            "map_resolution_m": 0.05,
            "room_area_factor_lower_limit_voronoi": 0.1,
            "room_area_factor_upper_limit_voronoi": 1000000.0,
            "voronoi_neighborhood_index": 280,
            "max_iterations": 150,
            "min_critical_point_distance_factor": 0.5,
            "max_area_for_merging": 12.5,
        },
        "color_alignment": {
            "reference_method": "voxroom",
            "matching": "hungarian_maximum_room_overlap_on_no_clearance_nav_free",
            "voronoi_label_to_voxroom_palette_index": {
                str(key): int(value) for key, value in sorted(mapping.items())
            },
            "overlap_matrix_cells": overlap.tolist(),
            "segmentation_geometry_modified": False,
        },
    }
    npz_path = prefix.with_suffix(".npz")
    with npz_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            voronoi_label_map=labels,
            voronoi_voxroom_color_aligned_label_map=aligned,
            voronoi_separator_map=separator,
            voronoi_raw_segmented_map=raw,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )
    prefix.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
