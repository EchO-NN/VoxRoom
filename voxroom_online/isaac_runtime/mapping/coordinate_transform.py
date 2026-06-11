from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass
class MapInfo:
    resolution_m: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: int
    height: int

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "MapInfo":
        return cls(
            resolution_m=float(data["resolution_m"]),
            min_x=float(data["min_x"]),
            max_x=float(data["max_x"]),
            min_y=float(data["min_y"]),
            max_y=float(data["max_y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def world_xy_to_grid(x: float, y: float, info: MapInfo) -> Tuple[int, int]:
    col = int(math.floor((float(x) - info.min_x) / info.resolution_m))
    row = int(math.floor((info.max_y - float(y)) / info.resolution_m))
    return row, col


def grid_to_world_xy(row: int, col: int, info: MapInfo) -> Tuple[float, float]:
    x = info.min_x + (int(col) + 0.5) * info.resolution_m
    y = info.max_y - (int(row) + 0.5) * info.resolution_m
    return x, y


def is_inside_grid(row: int, col: int, info: MapInfo) -> bool:
    return 0 <= int(row) < info.height and 0 <= int(col) < info.width


def clamp_grid(row: int, col: int, info: MapInfo) -> Tuple[int, int]:
    return max(0, min(info.height - 1, int(row))), max(0, min(info.width - 1, int(col)))

