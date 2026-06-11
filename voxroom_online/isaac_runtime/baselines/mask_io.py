from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def relabel_consecutive(label_map: np.ndarray) -> np.ndarray:
    """Return int32 label map with labels 1..K, preserving 0 as background."""
    arr = np.asarray(label_map)
    if arr.ndim != 2:
        raise ValueError(f"label_map must be 2D, got shape={arr.shape}")
    out = np.zeros(arr.shape, dtype=np.int32)
    labels = [int(x) for x in np.unique(arr) if int(x) > 0]
    for new_id, old_id in enumerate(labels, start=1):
        out[arr == old_id] = np.int32(new_id)
    return out


def build_metric_domain_from_source(source_arrays: Mapping[str, Any]) -> np.ndarray:
    """Mirror the evaluator's domain preference."""
    shape_source = source_arrays.get("occupancy_map")
    if shape_source is None:
        shape_source = source_arrays.get("observed_free_mask")
    if shape_source is None:
        raise KeyError("source snapshot missing occupancy_map and observed_free_mask")
    shape = np.asarray(shape_source).shape
    obstacle = np.asarray(source_arrays.get("obstacle_mask", np.zeros(shape, dtype=bool)), dtype=bool)
    unknown = np.asarray(source_arrays.get("unknown_mask", np.zeros(shape, dtype=bool)), dtype=bool)
    if obstacle.shape != shape or unknown.shape != shape:
        raise ValueError("obstacle_mask and unknown_mask shapes must match occupancy/eval shape")
    nav = source_arrays.get("navigation_free_room_domain")
    if nav is not None:
        domain = np.asarray(nav, dtype=bool)
        if domain.ndim == 2 and bool(domain.any()):
            if domain.shape != shape:
                raise ValueError(f"navigation_free_room_domain shape {domain.shape} != source shape {shape}")
            domain = domain & ~obstacle & ~unknown
            if np.count_nonzero(domain & obstacle) or np.count_nonzero(domain & unknown):
                raise ValueError("metric domain overlaps obstacle or unknown cells")
            return domain
    if "observed_free_mask" not in source_arrays:
        raise KeyError("source snapshot missing observed_free_mask and usable navigation_free_room_domain")
    observed_free = np.asarray(source_arrays["observed_free_mask"], dtype=bool)
    if obstacle.shape != observed_free.shape or unknown.shape != observed_free.shape:
        raise ValueError("observed_free_mask, obstacle_mask, and unknown_mask shapes must match")
    domain = observed_free & ~obstacle & ~unknown
    if np.count_nonzero(domain & obstacle) or np.count_nonzero(domain & unknown):
        raise ValueError("metric domain overlaps obstacle or unknown cells")
    return domain


def enforce_room_mask_contract(
    label_map: np.ndarray,
    source_arrays: Mapping[str, Any],
    *,
    clip_to_eval_domain: bool = True,
) -> np.ndarray:
    if "occupancy_map" not in source_arrays:
        raise KeyError("source snapshot missing occupancy_map")
    shape = np.asarray(source_arrays["occupancy_map"]).shape
    label_map = np.asarray(label_map)
    if label_map.shape != shape:
        raise ValueError(f"label shape {label_map.shape} != occupancy shape {shape}")
    out = label_map.astype(np.int32, copy=True)
    out[out < 0] = 0
    if clip_to_eval_domain:
        domain = build_metric_domain_from_source(source_arrays)
        if domain.shape != out.shape:
            raise ValueError(f"domain shape {domain.shape} != label shape {out.shape}")
        out[~domain] = 0
    return relabel_consecutive(out)


def save_baseline_snapshot_npz(
    *,
    source_npz_path: Path,
    output_npz_path: Path,
    baseline_label_map: np.ndarray,
    baseline_name: str,
    metadata: Mapping[str, Any],
    debug_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Copy source snapshot and replace final_room_label_map with baseline output."""
    source_npz_path = Path(source_npz_path)
    output_npz_path = Path(output_npz_path)
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source_npz_path, allow_pickle=False) as data:
        arrays: dict[str, Any] = {str(k): np.asarray(data[k]).copy() for k in data.files}
    if "final_room_label_map" in arrays:
        arrays["voxroom_final_room_label_map"] = np.asarray(arrays["final_room_label_map"], dtype=np.int32)
    final_map = enforce_room_mask_contract(baseline_label_map, arrays, clip_to_eval_domain=True)
    arrays["final_room_label_map"] = final_map
    arrays["baseline_name"] = np.asarray(str(baseline_name))
    arrays["baseline_metadata_json"] = np.asarray(
        json.dumps(_json_ready(dict(metadata)), ensure_ascii=False, sort_keys=True)
    )
    if debug_arrays:
        for key, value in debug_arrays.items():
            arrays[str(key)] = np.asarray(value)
    np.savez_compressed(output_npz_path, **arrays)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
