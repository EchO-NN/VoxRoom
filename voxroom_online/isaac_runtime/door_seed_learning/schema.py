from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


COLLECTION_SCHEMA_VERSION = "voxroom_door_seed_collection_v4_full_voxel_coverage_milestones"
MANUAL_ANNOTATION_SCHEMA_VERSION = "voxroom_door_seed_manual_annotation_v1"
RESOLVED_LABEL_SCHEMA_VERSION = "voxroom_door_seed_resolved_labels_v1"
DATASET_SCHEMA_VERSION = "voxroom_door_seed_dataset_v2_patch19_context41"
COORDINATE_CONVENTION = "row_col_image_origin_upper_left"
LOCAL_PATCH_SIZE = 19
CONTEXT_PATCH_SIZE = 41

SNAPSHOT_REQUIRED_ARRAYS = (
    "vertical_class_map_xy",
    "nav_class_map_xy",
    "raw_seed_mask_xy",
    "raw_seed_rc",
    "raw_seed_voxel_state_nzyx",
    "raw_seed_patch_valid_nyx",
    "z_centers_m",
)

SNAPSHOT_REQUIRED_METADATA = (
    "map_info_hash",
    "raw_seed_config_hash",
    "input_semantics_hash",
)

SNAPSHOT_RAW_SEED_SOURCE_ARRAYS = (
    "voxroom_raw_seed_mask_xy",
    "tvars_vertical_raw_seed_mask_xy",
    "voxroom_raw_seed_history_mask_xy",
    "tvars_vertical_raw_seed_history_mask_xy",
    "raw_seed_source_id_map_xy",
)

SNAPSHOT_FULL_VOXEL_ARRAYS = (
    "voxel_occupancy_state_zyx",
    "voxel_occupancy_log_odds_zyx",
    "voxel_sensor_range_count_zyx",
    "voxel_occupancy_z_centers_m",
)

# Kept as a compatibility export for callers that used to assert that these
# arrays were absent.  Schema v4 deliberately stores them at coverage events.
SNAPSHOT_FORBIDDEN_ARRAYS: tuple[str, ...] = ()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hash(root: str | Path) -> str:
    base = Path(root)
    digest = hashlib.sha256()
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(base)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hash_arrays(arrays: Mapping[str, np.ndarray], keys: Iterable[str] | None = None) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys if keys is not None else arrays):
        arr = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(str(key).encode("utf-8"))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(arr.shape)))
        digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def map_info_payload(map_info: object, shape: tuple[int, int] | None = None) -> dict[str, object]:
    if hasattr(map_info, "to_dict"):
        raw = dict(map_info.to_dict())
    elif is_dataclass(map_info):
        raw = dict(asdict(map_info))
    elif isinstance(map_info, Mapping):
        raw = dict(map_info)
    else:
        raise TypeError("map_info must provide to_dict, be a dataclass, or be a mapping")
    if shape is not None:
        raw["height"] = int(shape[0])
        raw["width"] = int(shape[1])
    return {
        "height": int(raw["height"]),
        "width": int(raw["width"]),
        "resolution_m": float(raw["resolution_m"]),
        "min_x": float(raw["min_x"]),
        "max_x": float(raw["max_x"]),
        "min_y": float(raw["min_y"]),
        "max_y": float(raw["max_y"]),
        "coordinate_convention": COORDINATE_CONVENTION,
    }


def map_info_hash(map_info: object, shape: tuple[int, int] | None = None) -> str:
    return sha256_bytes(canonical_json_bytes(map_info_payload(map_info, shape)))


def config_hash(config: object) -> str:
    if hasattr(config, "to_dict"):
        value = config.to_dict()
    elif is_dataclass(config):
        value = asdict(config)
    elif isinstance(config, Mapping):
        value = dict(config)
    else:
        value = vars(config)
    return sha256_bytes(canonical_json_bytes(value))


def write_json_atomic(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object in %s" % path)
    return value


def load_snapshot(path: str | Path, *, validate: bool = True) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    if validate:
        validate_snapshot_arrays(arrays)
    return arrays


def validate_snapshot_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    missing = [key for key in SNAPSHOT_REQUIRED_ARRAYS if key not in arrays]
    missing.extend(key for key in SNAPSHOT_REQUIRED_METADATA if key not in arrays)
    if missing:
        raise ValueError("door seed snapshot missing arrays: %s" % ", ".join(missing))
    vertical = np.asarray(arrays["vertical_class_map_xy"])
    nav = np.asarray(arrays["nav_class_map_xy"])
    mask = np.asarray(arrays["raw_seed_mask_xy"])
    rc = np.asarray(arrays["raw_seed_rc"])
    patches = np.asarray(arrays["raw_seed_voxel_state_nzyx"])
    valid = np.asarray(arrays["raw_seed_patch_valid_nyx"])
    z = np.asarray(arrays["z_centers_m"])
    if vertical.ndim != 2 or nav.shape != vertical.shape or mask.shape != vertical.shape:
        raise ValueError("door seed snapshot 2D maps must share one HxW shape")
    if vertical.dtype != np.uint8 or nav.dtype != np.uint8 or mask.dtype != bool:
        raise ValueError("door seed snapshot class maps must be uint8 and raw seed mask must be bool")
    if rc.shape != (int(rc.shape[0]) if rc.ndim == 2 else -1, 2):
        raise ValueError("raw_seed_rc must have shape [N,2]")
    count = int(rc.shape[0])
    if patches.ndim != 4 or patches.shape[0] != count or patches.shape[2:] != (LOCAL_PATCH_SIZE, LOCAL_PATCH_SIZE):
        raise ValueError("raw_seed_voxel_state_nzyx must have shape [N,Z,19,19]")
    if patches.dtype != np.uint8:
        raise ValueError("raw_seed_voxel_state_nzyx must be uint8")
    if valid.shape != (count, LOCAL_PATCH_SIZE, LOCAL_PATCH_SIZE) or valid.dtype != bool:
        raise ValueError("raw_seed_patch_valid_nyx must have shape [N,19,19] and bool dtype")
    if z.shape != (patches.shape[1],) or z.dtype != np.float32:
        raise ValueError("z_centers_m must be float32 [Z]")
    expected_rc = np.argwhere(mask).astype(np.int32)
    if expected_rc.shape != rc.shape or not np.array_equal(expected_rc, rc.astype(np.int32, copy=False)):
        raise ValueError("raw_seed_rc must be row-major sorted and exactly match raw_seed_mask_xy")
    for key in SNAPSHOT_RAW_SEED_SOURCE_ARRAYS:
        if key not in arrays:
            continue
        value = np.asarray(arrays[key])
        if value.shape != vertical.shape:
            raise ValueError("%s must match the snapshot HxW shape" % key)
        expected_dtype = np.uint8 if key == "raw_seed_source_id_map_xy" else bool
        if value.dtype != expected_dtype:
            raise ValueError("%s must have dtype %s" % (key, np.dtype(expected_dtype)))
    full_voxel = bool(scalar_value(arrays, "full_voxel_snapshot", False))
    present_full = [key for key in SNAPSHOT_FULL_VOXEL_ARRAYS if key in arrays]
    if full_voxel and len(present_full) != len(SNAPSHOT_FULL_VOXEL_ARRAYS):
        missing_full = [key for key in SNAPSHOT_FULL_VOXEL_ARRAYS if key not in arrays]
        raise ValueError("full voxel DoorSeed snapshot is missing: %s" % ", ".join(missing_full))
    if present_full and not full_voxel:
        raise ValueError("full voxel arrays require full_voxel_snapshot=true")
    if full_voxel:
        full_state = np.asarray(arrays["voxel_occupancy_state_zyx"])
        full_log_odds = np.asarray(arrays["voxel_occupancy_log_odds_zyx"])
        full_sensor = np.asarray(arrays["voxel_sensor_range_count_zyx"])
        full_z = np.asarray(arrays["voxel_occupancy_z_centers_m"])
        if full_state.dtype != np.uint8 or full_state.ndim != 3:
            raise ValueError("voxel_occupancy_state_zyx must be uint8 [Z,H,W]")
        if full_log_odds.dtype != np.int16 or full_log_odds.shape != full_state.shape:
            raise ValueError("voxel_occupancy_log_odds_zyx must be int16 with the full voxel shape")
        if full_sensor.dtype != np.uint8 or full_sensor.shape != full_state.shape:
            raise ValueError("voxel_sensor_range_count_zyx must be uint8 with the full voxel shape")
        if full_state.shape[1:] != vertical.shape:
            raise ValueError("full voxel XY shape must match the DoorSeed maps")
        if full_z.dtype != np.float32 or full_z.shape != (full_state.shape[0],):
            raise ValueError("voxel_occupancy_z_centers_m must be float32 [Z]")


def scalar_value(arrays: Mapping[str, np.ndarray], key: str, default: object = None) -> object:
    if key not in arrays:
        return default
    value = np.asarray(arrays[key])
    if value.ndim != 0:
        raise ValueError("snapshot metadata %s must be scalar" % key)
    return value.item()


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
