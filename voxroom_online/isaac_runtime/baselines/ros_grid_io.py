from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .mask_io import build_segmentation_domain_from_source

ROS_UNKNOWN = np.int8(-1)
ROS_FREE = np.int8(0)
ROS_OCCUPIED = np.int8(100)
IPA_OCCUPIED = np.uint8(0)
IPA_FREE = np.uint8(255)


def snapshot_to_ros_occupancy_grid(arrays: Mapping[str, Any]) -> np.ndarray:
    """Return HxW int8 grid with -1 unknown, 0 free, 100 occupied."""
    free, _source = build_segmentation_domain_from_source(arrays)
    obstacle = _first_bool_like(
        arrays,
        (
            "voxel_wall_xy",
            "structural_wall_clean",
            "roomseg_sanitized_wall",
            "obstacle_mask",
            "occupancy_map",
        ),
        free.shape,
    )
    obstacle &= ~free
    grid = np.full(free.shape, ROS_UNKNOWN, dtype=np.int8)
    grid[free] = ROS_FREE
    grid[obstacle] = ROS_OCCUPIED
    return grid


def snapshot_to_ipa_image(arrays: Mapping[str, Any]) -> np.ndarray:
    """Return HxW uint8 image for ipa_room_segmentation: 255 free, 0 inaccessible."""
    free, _source = build_segmentation_domain_from_source(arrays)
    img = np.zeros(free.shape, dtype=np.uint8)
    img[free] = IPA_FREE
    return img


def _first_bool_like(
    arrays: Mapping[str, Any], keys: tuple[str, ...], shape: tuple[int, int]
) -> np.ndarray:
    for key in keys:
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=bool)
        if arr.shape == shape and bool(np.any(arr)):
            return arr.copy()
    return np.zeros(shape, dtype=bool)
