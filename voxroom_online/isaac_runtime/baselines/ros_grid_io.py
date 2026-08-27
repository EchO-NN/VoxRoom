from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .mask_io import build_metric_domain_from_source

ROS_UNKNOWN = np.int8(-1)
ROS_FREE = np.int8(0)
ROS_OCCUPIED = np.int8(100)
IPA_OCCUPIED = np.uint8(0)
IPA_FREE = np.uint8(255)


def snapshot_to_ros_occupancy_grid(arrays: Mapping[str, Any]) -> np.ndarray:
    """Return HxW int8 grid with -1 unknown, 0 free, 100 occupied."""
    free = build_metric_domain_from_source(arrays)
    obstacle = _bool_like(arrays.get("obstacle_mask", arrays.get("occupancy_map")), free.shape)
    unknown = _bool_like(arrays.get("unknown_mask"), free.shape)
    grid = np.full(free.shape, ROS_UNKNOWN, dtype=np.int8)
    grid[free] = ROS_FREE
    grid[obstacle] = ROS_OCCUPIED
    grid[unknown & ~free] = ROS_UNKNOWN
    return grid


def snapshot_to_ipa_image(arrays: Mapping[str, Any]) -> np.ndarray:
    """Return HxW uint8 image for ipa_room_segmentation: 255 free, 0 inaccessible."""
    free = build_metric_domain_from_source(arrays)
    img = np.zeros(free.shape, dtype=np.uint8)
    img[free] = IPA_FREE
    return img


def _bool_like(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        return np.zeros(shape, dtype=bool)
    return arr

