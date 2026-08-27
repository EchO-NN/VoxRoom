from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .common import parse_snapshot_step


@dataclass(frozen=True)
class SnapshotArrays:
    path: Path
    step: int
    shape: tuple[int, int]
    final_room_label_map: np.ndarray
    eval_domain: np.ndarray
    domain_key: str
    observed_free_mask: np.ndarray | None
    obstacle_mask: np.ndarray | None
    unknown_mask: np.ndarray | None
    navigation_free_room_domain: np.ndarray | None
    vertical_free_room_domain: np.ndarray | None
    segmentation_domain: np.ndarray
    segmentation_domain_key: str
    coverage_reference_domain: np.ndarray | None
    coverage_explored_domain: np.ndarray | None
    door_seed_raw_mask: np.ndarray | None
    door_seed_keep_mask: np.ndarray | None
    door_seed_reject_mask: np.ndarray | None


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def build_eval_domain(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, str]:
    final = arrays.get("final_room_label_map")
    shape = np.asarray(final).shape if final is not None else None
    observed_free = arrays.get("observed_free_mask")
    if shape is None and observed_free is not None:
        shape = np.asarray(observed_free).shape
    if shape is None:
        raise KeyError("snapshot missing final_room_label_map and observed_free_mask")
    obstacle = np.asarray(arrays.get("obstacle_mask", np.zeros(shape, dtype=bool)), dtype=bool)
    unknown = np.asarray(arrays.get("unknown_mask", np.zeros(shape, dtype=bool)), dtype=bool)
    if obstacle.shape != shape or unknown.shape != shape:
        raise ValueError("obstacle_mask and unknown_mask shapes must match snapshot shape")
    nav = arrays.get("navigation_free_room_domain")
    if nav is not None and np.asarray(nav).ndim == 2 and np.any(np.asarray(nav, dtype=bool)):
        domain = np.asarray(nav, dtype=bool)
        if domain.shape != shape:
            raise ValueError("navigation_free_room_domain shape does not match snapshot shape")
        return domain & ~obstacle & ~unknown, "navigation_free_room_domain"
    if observed_free is None:
        raise KeyError("snapshot missing observed_free_mask and usable navigation_free_room_domain")
    observed = np.asarray(observed_free, dtype=bool)
    if obstacle.shape != observed.shape or unknown.shape != observed.shape:
        raise ValueError("observed_free_mask, obstacle_mask, and unknown_mask shapes must match")
    return observed & ~obstacle & ~unknown, "observed_free_minus_obstacle_unknown"


def build_segmentation_domain(arrays: Mapping[str, np.ndarray], *, fallback_domain: np.ndarray) -> tuple[np.ndarray, str]:
    # GT room boundaries must come from the online vertical-free evidence.  The
    # fixed coverage reference can be built per GT room polygon, so using it as
    # a segmentation source leaks an already separated room layout.
    for key in ("voxel_vertical_free_xy", "height_profile_vertical_free_xy", "vertical_free_room_domain"):
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=bool)
        if arr.ndim == 2 and arr.shape == np.asarray(fallback_domain).shape and np.any(arr):
            return arr, key
    coverage_reference = arrays.get("roomseg_eval_reference_explorable_mask")
    if coverage_reference is not None:
        reference = np.asarray(coverage_reference, dtype=bool)
        if reference.shape != np.asarray(fallback_domain).shape:
            raise ValueError(
                "roomseg_eval_reference_explorable_mask shape does not match snapshot"
            )
        if not np.any(reference):
            raise ValueError("roomseg_eval_reference_explorable_mask is empty")
        return reference, "roomseg_eval_reference_explorable_mask_fallback_no_vertical_free"
    return np.asarray(fallback_domain, dtype=bool), "fallback_eval_domain"


def load_snapshot_arrays(path: Path, *, domain_preference: str = "navigation_free_room_domain") -> SnapshotArrays:
    _ = domain_preference
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    relevant_keys = {
        "final_room_label_map",
        "step",
        "observed_free_mask",
        "obstacle_mask",
        "unknown_mask",
        "navigation_free_room_domain",
        "voxel_vertical_free_xy",
        "height_profile_vertical_free_xy",
        "vertical_free_room_domain",
        "roomseg_eval_reference_explorable_mask",
        "roomseg_eval_explored_reference_mask",
        "voxel_door_raw_seed_mask",
        "voxel_door_seed_mask",
        "voxel_door_seed_model_keep_mask",
        "voxel_door_seed_model_reject_mask",
    }
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name]).copy()
            for name in relevant_keys.intersection(data.files)
        }
    if "final_room_label_map" not in arrays:
        raise KeyError("snapshot missing final_room_label_map")
    pred = np.asarray(arrays["final_room_label_map"], dtype=np.int32)
    if pred.ndim != 2:
        raise ValueError("final_room_label_map must be 2D")
    step, source = parse_snapshot_step(path, path.with_suffix(".summary.json"))
    if step is None:
        meta_step = arrays.get("step")
        if meta_step is not None:
            step = int(np.asarray(meta_step).reshape(-1)[0])
            source = "npz_metadata"
    if step is None:
        raise ValueError("could not parse snapshot step: %s" % path)
    domain, domain_key = build_eval_domain(arrays)
    if domain.shape != pred.shape:
        raise ValueError("eval domain shape %s does not match final_room_label_map shape %s" % (domain.shape, pred.shape))
    segmentation_domain, segmentation_domain_key = build_segmentation_domain(arrays, fallback_domain=domain)
    coverage_reference_domain = _optional_bool(
        arrays.get("roomseg_eval_reference_explorable_mask"), pred.shape
    )
    coverage_explored_domain = _optional_bool(
        arrays.get("roomseg_eval_explored_reference_mask"), pred.shape
    )
    if (coverage_reference_domain is None) != (coverage_explored_domain is None):
        raise ValueError(
            "coverage snapshot must contain both reference and explored domains"
        )
    if coverage_reference_domain is not None and np.any(
        coverage_explored_domain & ~coverage_reference_domain
    ):
        raise ValueError("coverage explored domain lies outside fixed reference")
    door_seed_raw_mask = _optional_bool(
        arrays.get("voxel_door_raw_seed_mask"), pred.shape
    )
    door_seed_keep_mask = _first_available_optional_bool(
        arrays,
        ("voxel_door_seed_model_keep_mask", "voxel_door_seed_mask"),
        pred.shape,
    )
    door_seed_reject_mask = _optional_bool(
        arrays.get("voxel_door_seed_model_reject_mask"), pred.shape
    )
    if door_seed_raw_mask is None and (
        door_seed_keep_mask is not None or door_seed_reject_mask is not None
    ):
        door_seed_raw_mask = np.zeros(pred.shape, dtype=bool)
        if door_seed_keep_mask is not None:
            door_seed_raw_mask |= door_seed_keep_mask
        if door_seed_reject_mask is not None:
            door_seed_raw_mask |= door_seed_reject_mask
    if door_seed_reject_mask is None and door_seed_raw_mask is not None:
        if door_seed_keep_mask is not None:
            door_seed_reject_mask = door_seed_raw_mask & ~door_seed_keep_mask
    if door_seed_raw_mask is not None:
        if door_seed_keep_mask is not None:
            door_seed_keep_mask &= door_seed_raw_mask
        if door_seed_reject_mask is not None:
            door_seed_reject_mask &= door_seed_raw_mask
    return SnapshotArrays(
        path=path,
        step=int(step),
        shape=(int(pred.shape[0]), int(pred.shape[1])),
        final_room_label_map=pred,
        eval_domain=domain.astype(bool),
        domain_key=domain_key,
        observed_free_mask=_optional_bool(arrays.get("observed_free_mask"), pred.shape),
        obstacle_mask=_optional_bool(arrays.get("obstacle_mask"), pred.shape),
        unknown_mask=_optional_bool(arrays.get("unknown_mask"), pred.shape),
        navigation_free_room_domain=_optional_bool(arrays.get("navigation_free_room_domain"), pred.shape),
        vertical_free_room_domain=_first_optional_bool(
            arrays,
            ("voxel_vertical_free_xy", "height_profile_vertical_free_xy", "vertical_free_room_domain"),
            pred.shape,
        ),
        segmentation_domain=segmentation_domain.astype(bool),
        segmentation_domain_key=str(segmentation_domain_key),
        coverage_reference_domain=coverage_reference_domain,
        coverage_explored_domain=coverage_explored_domain,
        door_seed_raw_mask=door_seed_raw_mask,
        door_seed_keep_mask=door_seed_keep_mask,
        door_seed_reject_mask=door_seed_reject_mask,
    )


def validate_snapshot(path: Path, *, allow_empty_domain: bool = False) -> list[str]:
    warnings: list[str] = []
    snap = load_snapshot_arrays(path)
    if not allow_empty_domain and not np.any(snap.eval_domain):
        raise ValueError("eval domain is empty: %s" % path)
    if not np.any(snap.eval_domain):
        warnings.append("eval_domain_empty")
    return warnings


def _optional_bool(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        return None
    return arr


def _first_optional_bool(arrays: Mapping[str, np.ndarray], keys: tuple[str, ...], shape: tuple[int, int]) -> np.ndarray | None:
    for key in keys:
        arr = _optional_bool(arrays.get(key), shape)
        if arr is not None and np.any(arr):
            return arr
    return None


def _first_available_optional_bool(
    arrays: Mapping[str, np.ndarray],
    keys: tuple[str, ...],
    shape: tuple[int, int],
) -> np.ndarray | None:
    for key in keys:
        arr = _optional_bool(arrays.get(key), shape)
        if arr is not None:
            return arr
    return None
