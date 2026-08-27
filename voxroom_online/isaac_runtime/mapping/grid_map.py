from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo, world_xy_to_grid


@dataclass
class OnlineGridMap:
    map_info: MapInfo
    occupied: np.ndarray
    free: np.ndarray
    observed: np.ndarray

    @classmethod
    def centered(cls, center_x: float, center_y: float, size_m: float, resolution_m: float) -> "OnlineGridMap":
        half = float(size_m) * 0.5
        width = int(math.ceil(size_m / resolution_m))
        height = width
        info = MapInfo(
            resolution_m=float(resolution_m),
            min_x=float(center_x - half),
            max_x=float(center_x + half),
            min_y=float(center_y - half),
            max_y=float(center_y + half),
            width=width,
            height=height,
        )
        shape = (height, width)
        return cls(info, np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8))

    def traversible(self, unknown_is_obstacle: bool = True) -> np.ndarray:
        if unknown_is_obstacle:
            return (self.free > 0) & (self.occupied == 0)
        return self.occupied == 0

    def mark_free(self, row: int, col: int) -> None:
        if 0 <= row < self.map_info.height and 0 <= col < self.map_info.width:
            self.free[row, col] = 1
            self.observed[row, col] = 1

    def mark_occupied(self, row: int, col: int) -> None:
        if 0 <= row < self.map_info.height and 0 <= col < self.map_info.width:
            self.occupied[row, col] = 1
            self.observed[row, col] = 1

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return world_xy_to_grid(x, y, self.map_info)
