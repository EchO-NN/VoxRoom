from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


BOOL_LAYERS = [
    "vertical_free_raw",
    "vertical_occupied_raw",
    "vertical_observed_raw",
    "vertical_unknown_raw",
    "free_clean_before_noise_wall_gap_fill",
    "free_clean",
    "wall_candidate_clean",
    "noise_wall_gap_fill",
    "noise_wall_gap_fill_all",
    "structural_wall_free_overlap",
    "wall_target_after_noise_gap_fill",
    "pre_extension_door_detected_map",
    "pre_extension_door_cut_mask",
    "pre_extension_partition_free",
    "step1_step2_accepted_separator_map",
    "pass2_line_extension_completion",
    "wall_target_after_line_extension",
    "completed_wall_after_line_extension",
    "line_supported_walls",
    "line_supported_wall_mask",
    "raw_line_supported_walls",
    "filtered_wall_lines",
    "filtered_wall_endpoints",
    "snapped_wall_runs",
    "merged_wall_runs",
    "physical_wall_completion_candidates",
    "missed_scan_gap_closure_candidates",
    "short_unknown_gap_closure_candidates",
    "doorway_virtual_cut_candidates",
    "single_sided_wall_extension_candidates",
    "corridor_skeleton",
    "corridor_candidate_map",
    "corridor_room_neck_cut_candidates",
    "pass1_line_extensions_all",
    "pass1_line_extensions_accepted",
    "pass1_line_extensions_rejected",
    "pass1_door_neck_candidates",
    "pass1_accepted_separators",
    "pass1_rejected_separators",
    "pass2_virtual_targets",
    "pass2_extension_intersection_targets",
    "pass2_line_extensions_all",
    "pass2_line_extensions_accepted",
    "pass2_line_extensions_rejected",
    "pass2_door_neck_candidates",
    "accepted_separators_before_corridor_merge",
    "accepted_separators_after_corridor_merge",
    "rejected_false_parallel_doors",
    "accepted_separators",
    "rejected_separators",
    "virtual_separator_label_fill",
    "corridor_like_regions",
    "open_living_room_like_regions",
]


def save_online_roomseg_debug(
    *,
    out_dir: str | Path,
    layers: Mapping[str, np.ndarray],
    separator_report: Mapping[str, object],
    extra_reports: Mapping[str, Mapping[str, object]] | None = None,
    save_layers: bool = True,
    save_candidate_json: bool = True,
) -> dict:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if save_layers:
        for name in BOOL_LAYERS:
            if name in layers:
                path = root / ("%s.png" % name)
                _save_bool(path, np.asarray(layers[name], dtype=bool))
                paths[name] = str(path)
        for name in (
            "room_labels_before_separators",
            "pre_extension_room_label_map",
            "room_labels_after_step1_step2_without_pre_extension_doors",
            "room_labels_after_separators",
            "raw_room_labels_before_corridor_merge",
            "room_labels_after_corridor_merge_before_virtual_fill",
            "room_labels_after_corridor_merge",
            "final_room_labels",
        ):
            if name in layers:
                path = root / ("%s.png" % name)
                _save_labels(path, np.asarray(layers[name], dtype=np.int32))
                paths[name] = str(path)
        paths.update(_save_red_wall_composites(root, layers))
    if save_candidate_json:
        path = root / "separator_report.json"
        path.write_text(json.dumps(_jsonable(separator_report), indent=2, ensure_ascii=False), encoding="utf-8")
        paths["separator_report"] = str(path)
        for name, payload in dict(extra_reports or {}).items():
            report_path = root / ("%s.json" % str(name))
            report_path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
            paths[str(name)] = str(report_path)
    return {"paths": paths, "output_dir": str(root)}


def _save_bool(path: Path, mask: np.ndarray) -> None:
    arr = np.zeros((*mask.shape, 3), dtype=np.uint8)
    arr[:, :] = (25, 25, 25)
    arr[np.asarray(mask, dtype=bool)] = (245, 245, 245)
    Image.fromarray(arr, mode="RGB").save(path)


def _save_red_wall_composites(root: Path, layers: Mapping[str, np.ndarray]) -> dict[str, str]:
    paths: dict[str, str] = {}
    free = _optional_mask(layers, "vertical_free_raw")
    vertical_wall = _optional_mask(layers, "vertical_occupied_raw")
    wall_target = _optional_mask(layers, "wall_target_after_noise_gap_fill")
    noise_fill = _optional_mask(layers, "noise_wall_gap_fill_all")
    if free is not None and vertical_wall is not None:
        path = root / "vertical_free_wall_red.png"
        _save_wall_red(path, free=free, wall=vertical_wall)
        paths["vertical_free_wall_red"] = str(path)
    if free is not None and wall_target is not None:
        path = root / "roomseg_wall_target_red.png"
        _save_wall_red(path, free=free, wall=wall_target, noise_fill=noise_fill)
        paths["roomseg_wall_target_red"] = str(path)
    labels = layers.get("final_room_labels")
    if labels is not None and wall_target is not None:
        path = root / "final_room_labels_wall_red.png"
        _save_labels_with_wall_overlay(path, np.asarray(labels, dtype=np.int32), wall_target, noise_fill=noise_fill)
        paths["final_room_labels_wall_red"] = str(path)
    return paths


def _optional_mask(layers: Mapping[str, np.ndarray], name: str) -> np.ndarray | None:
    if name not in layers:
        return None
    return np.asarray(layers[name], dtype=bool)


def _save_wall_red(path: Path, *, free: np.ndarray, wall: np.ndarray, noise_fill: np.ndarray | None = None) -> None:
    arr = np.zeros((*free.shape, 3), dtype=np.uint8)
    arr[np.asarray(free, dtype=bool)] = (245, 245, 245)
    arr[np.asarray(wall, dtype=bool)] = (255, 0, 0)
    if noise_fill is not None:
        arr[np.asarray(noise_fill, dtype=bool)] = (255, 220, 0)
    Image.fromarray(arr, mode="RGB").save(path)


def _save_labels_with_wall_overlay(path: Path, labels: np.ndarray, wall: np.ndarray, noise_fill: np.ndarray | None = None) -> None:
    arr = np.zeros((*labels.shape, 3), dtype=np.uint8)
    positive = sorted(int(v) for v in np.unique(labels) if int(v) > 0)
    for label in positive:
        arr[labels == label] = _label_color(label)
    arr[np.asarray(wall, dtype=bool)] = (255, 0, 0)
    if noise_fill is not None:
        arr[np.asarray(noise_fill, dtype=bool)] = (255, 220, 0)
    Image.fromarray(arr, mode="RGB").save(path)


def _save_labels(path: Path, labels: np.ndarray) -> None:
    arr = np.zeros((*labels.shape, 3), dtype=np.uint8)
    positive = sorted(int(v) for v in np.unique(labels) if int(v) > 0)
    for label in positive:
        color = _label_color(label)
        arr[labels == label] = color
    Image.fromarray(arr, mode="RGB").save(path)


def _label_color(label: int) -> tuple[int, int, int]:
    value = int(label) * 2654435761
    return (
        60 + ((value >> 0) & 0x7F),
        60 + ((value >> 8) & 0x7F),
        60 + ((value >> 16) & 0x7F),
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
