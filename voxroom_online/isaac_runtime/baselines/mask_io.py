from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SEGMENTATION_INPUT_MODE_KEY = "_baseline_segmentation_input_mode"
SEGMENTATION_INPUT_RAW_VERTICAL_FREE = "raw_vertical_free"
SEGMENTATION_INPUT_RAW_NAV_FREE_NO_CLEARANCE = "raw_nav_free_no_clearance"
SEGMENTATION_INPUT_MODES = (
    SEGMENTATION_INPUT_RAW_VERTICAL_FREE,
    SEGMENTATION_INPUT_RAW_NAV_FREE_NO_CLEARANCE,
)


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
    coverage_domain = source_arrays.get("roomseg_eval_explored_reference_mask")
    if coverage_domain is not None:
        domain = np.asarray(coverage_domain, dtype=bool)
        if domain.shape != shape:
            raise ValueError(
                "roomseg_eval_explored_reference_mask shape does not match source shape"
            )
        reference = source_arrays.get("roomseg_eval_reference_explorable_mask")
        if reference is None:
            raise KeyError(
                "coverage snapshot lacks roomseg_eval_reference_explorable_mask"
            )
        reference_domain = np.asarray(reference, dtype=bool)
        if reference_domain.shape != shape:
            raise ValueError(
                "roomseg_eval_reference_explorable_mask shape does not match source shape"
            )
        if np.any(domain & ~reference_domain):
            raise ValueError("coverage explored domain lies outside fixed reference")
        return domain
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


def build_segmentation_domain_from_source(source_arrays: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    """Return the explicitly selected raw online segmentation input.

    Coverage snapshots also carry fixed-reference masks used to measure coverage
    and to define the metric domain.  In InteriorAgent those masks are cleaned
    independently inside each GT room polygon, so intersecting them with the
    Vertical-Free projection would leak pre-separated room geometry into every
    baseline.  Reference masks must therefore never alter a method's input.
    Reference masks must therefore never alter either Vertical-Free or the
    no-clearance navigation projection.
    """
    shape_source = source_arrays.get("occupancy_map")
    if shape_source is None:
        shape_source = source_arrays.get("final_room_label_map")
    if shape_source is None:
        shape_source = source_arrays.get("observed_free_mask")
    if shape_source is None:
        raise KeyError(
            "source snapshot missing occupancy_map, final_room_label_map, and observed_free_mask"
        )
    shape = np.asarray(shape_source).shape
    if len(shape) != 2:
        raise ValueError("segmentation source shape must be 2D")

    explicit_mode = SEGMENTATION_INPUT_MODE_KEY in source_arrays
    mode = segmentation_input_mode_from_source(source_arrays)
    candidates = (
        (
            "voxel_nav_free_xy",
            "navigation_free_room_domain",
        )
        if mode == SEGMENTATION_INPUT_RAW_NAV_FREE_NO_CLEARANCE
        else (
            "voxel_vertical_free_xy",
            "height_profile_vertical_free_xy",
            "vertical_free_room_domain",
        )
    )
    for key in candidates:
        value = source_arrays.get(key)
        if value is None:
            continue
        free = np.asarray(value, dtype=bool)
        if free.shape != shape or not bool(np.any(free)):
            continue
        return free.copy(), str(key)

    if not explicit_mode and mode == SEGMENTATION_INPUT_RAW_VERTICAL_FREE:
        return (
            build_metric_domain_from_source(source_arrays),
            "metric_domain_fallback_no_vertical_free",
        )
    raise KeyError(
        "snapshot lacks a non-empty %s segmentation input; checked %s"
        % (mode, ", ".join(candidates))
    )


def segmentation_input_mode_from_source(source_arrays: Mapping[str, Any]) -> str:
    raw = source_arrays.get(SEGMENTATION_INPUT_MODE_KEY)
    if raw is None:
        return SEGMENTATION_INPUT_RAW_VERTICAL_FREE
    values = np.asarray(raw).reshape(-1)
    if values.size != 1:
        raise ValueError("baseline segmentation input mode must be scalar")
    mode = str(values[0]).strip()
    if mode not in SEGMENTATION_INPUT_MODES:
        raise ValueError(
            "unsupported baseline segmentation input mode %r; expected one of %s"
            % (mode, ", ".join(SEGMENTATION_INPUT_MODES))
        )
    return mode


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
    excluded_source_keys: Iterable[str] | None = None,
) -> None:
    """Copy source snapshot and replace final_room_label_map with baseline output."""
    source_npz_path = Path(source_npz_path)
    output_npz_path = Path(output_npz_path)
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    excluded = {str(key) for key in (excluded_source_keys or ())}
    with np.load(source_npz_path, allow_pickle=False) as data:
        arrays: dict[str, Any] = {
            str(key): np.asarray(data[key]).copy()
            for key in data.files
            if str(key) not in excluded
        }
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
