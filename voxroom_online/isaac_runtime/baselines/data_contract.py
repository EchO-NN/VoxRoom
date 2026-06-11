from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MapInfo:
    resolution_m: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: int
    height: int
    source: str = "unknown"

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "map_resolution_m": np.asarray(float(self.resolution_m), dtype=np.float32),
            "map_origin_x_m": np.asarray(float(self.min_x), dtype=np.float32),
            "map_origin_y_m": np.asarray(float(self.min_y), dtype=np.float32),
            "map_origin_xy_m": np.asarray([float(self.min_x), float(self.min_y)], dtype=np.float32),
            "map_min_x_m": np.asarray(float(self.min_x), dtype=np.float32),
            "map_max_x_m": np.asarray(float(self.max_x), dtype=np.float32),
            "map_min_y_m": np.asarray(float(self.min_y), dtype=np.float32),
            "map_max_y_m": np.asarray(float(self.max_y), dtype=np.float32),
            "map_width_cells": np.asarray(int(self.width), dtype=np.int32),
            "map_height_cells": np.asarray(int(self.height), dtype=np.int32),
            "map_info_source": np.asarray(str(self.source)),
        }

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def map_info_extra_arrays(map_info: Any) -> dict[str, np.ndarray]:
    info = resolve_map_info(map_info=map_info)
    return info.to_arrays()


def resolve_map_info(
    *,
    map_info: Any | None = None,
    mapper: Any | None = None,
    map_state: Mapping[str, Any] | None = None,
    snapshot_arrays: Mapping[str, Any] | None = None,
    default_resolution_m: float | None = None,
) -> MapInfo:
    if map_info is not None:
        resolved = _from_map_info_object(map_info)
        if resolved is not None:
            return resolved
    if map_state is not None:
        for key in ("map_info", "info"):
            resolved = _from_map_info_object(map_state.get(key))
            if resolved is not None:
                return resolved
        resolved = _from_mapping(map_state, source="map_state")
        if resolved is not None:
            return resolved
    if mapper is not None:
        for attr in ("grid", "voxel_grid"):
            child = getattr(mapper, attr, None)
            resolved = _from_map_info_object(getattr(child, "map_info", None))
            if resolved is not None:
                return resolved
        resolved = _from_mapping(_object_attrs(mapper), source="mapper")
        if resolved is not None:
            return resolved
    if snapshot_arrays is not None:
        resolved = _from_mapping(snapshot_arrays, source="snapshot")
        if resolved is not None:
            return resolved
        shape = _snapshot_shape(snapshot_arrays)
        if shape is not None and default_resolution_m is not None:
            height, width = shape
            resolution = float(default_resolution_m)
            return MapInfo(
                resolution_m=resolution,
                min_x=0.0,
                max_x=float(width) * resolution,
                min_y=0.0,
                max_y=float(height) * resolution,
                width=int(width),
                height=int(height),
                source="default_resolution_with_zero_origin",
            )
    if default_resolution_m is not None:
        raise ValueError("map shape is required when only default_resolution_m is provided")
    raise ValueError(
        "could not resolve map info; provide map_state/map_info, snapshot map_* arrays, "
        "or an explicit --map-resolution-m fallback"
    )


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return {str(k): np.asarray(data[k]).copy() for k in data.files}


def _from_map_info_object(value: Any) -> MapInfo | None:
    if value is None:
        return None
    if isinstance(value, MapInfo):
        return value
    if hasattr(value, "to_dict"):
        try:
            data = dict(value.to_dict())
        except Exception:
            data = {}
        resolved = _from_mapping(data, source=type(value).__name__)
        if resolved is not None:
            return resolved
    resolved = _from_mapping(_object_attrs(value), source=type(value).__name__)
    return resolved


def _from_mapping(data: Mapping[str, Any], *, source: str) -> MapInfo | None:
    if not data:
        return None
    resolution = _first_float(data, ("resolution_m", "map_resolution_m", "resolution", "cell_size"))
    width = _first_int(data, ("width", "map_width_cells"))
    height = _first_int(data, ("height", "map_height_cells"))
    min_x = _first_float(data, ("min_x", "map_min_x_m", "map_origin_x_m"))
    max_x = _first_float(data, ("max_x", "map_max_x_m"))
    min_y = _first_float(data, ("min_y", "map_min_y_m", "map_origin_y_m"))
    max_y = _first_float(data, ("max_y", "map_max_y_m"))
    origin_xy = _array_value(data.get("map_origin_xy_m"))
    if origin_xy is not None and origin_xy.size >= 2:
        if min_x is None:
            min_x = float(origin_xy.reshape(-1)[0])
        if min_y is None:
            min_y = float(origin_xy.reshape(-1)[1])
    if resolution is None or width is None or height is None:
        return None
    if min_x is None:
        min_x = 0.0
    if min_y is None:
        min_y = 0.0
    if max_x is None:
        max_x = float(min_x) + int(width) * float(resolution)
    if max_y is None:
        max_y = float(min_y) + int(height) * float(resolution)
    return MapInfo(
        resolution_m=float(resolution),
        min_x=float(min_x),
        max_x=float(max_x),
        min_y=float(min_y),
        max_y=float(max_y),
        width=int(width),
        height=int(height),
        source=str(source),
    )


def _object_attrs(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    keys = (
        "resolution_m",
        "map_resolution_m",
        "resolution",
        "cell_size",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "width",
        "height",
    )
    return {key: getattr(obj, key) for key in keys if hasattr(obj, key)}


def _first_float(data: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in data:
            continue
        arr = _array_value(data.get(key))
        if arr is None or arr.size == 0:
            continue
        try:
            return float(arr.reshape(-1)[0])
        except Exception:
            continue
    return None


def _first_int(data: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    value = _first_float(data, keys)
    if value is None:
        return None
    return int(round(float(value)))


def _array_value(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value)
    except Exception:
        return None


def _snapshot_shape(arrays: Mapping[str, Any]) -> tuple[int, int] | None:
    for key in ("occupancy_map", "observed_free_mask", "navigation_free_room_domain", "final_room_label_map"):
        if key in arrays:
            arr = np.asarray(arrays[key])
            if arr.ndim == 2:
                return int(arr.shape[0]), int(arr.shape[1])
    return None

