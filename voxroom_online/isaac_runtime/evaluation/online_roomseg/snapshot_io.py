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
    for key in ("voxel_vertical_free_xy", "height_profile_vertical_free_xy", "vertical_free_room_domain"):
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=bool)
        if arr.ndim == 2 and arr.shape == np.asarray(fallback_domain).shape and np.any(arr):
            return arr, key
    return np.asarray(fallback_domain, dtype=bool), "fallback_eval_domain"


def load_snapshot_arrays(path: Path, *, domain_preference: str = "navigation_free_room_domain") -> SnapshotArrays:
    _ = domain_preference
    path = Path(path)
    arrays = load_npz_arrays(path)
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
