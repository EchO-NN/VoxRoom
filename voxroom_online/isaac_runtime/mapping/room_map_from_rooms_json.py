from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.dataset.category_normalizer import normalize_category
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo, grid_to_world_xy


@dataclass
class RoomGT:
    room_id: str
    room_type: str
    polygon_xy: List[Tuple[float, float]]
    area_m2: float

    def to_dict(self) -> dict:
        return asdict(self)


def polygon_area(poly: Sequence[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    acc = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def point_in_polygon(x: float, y: float, poly: Sequence[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if yj != yi else 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def load_rooms(path: str) -> List[RoomGT]:
    data = json.load(open(path, "r", encoding="utf-8"))
    rooms: List[RoomGT] = []
    for idx, room in enumerate(data):
        poly_raw = room.get("polygon", room.get("polygon_xy", []))
        poly = [(float(x), float(y)) for x, y in poly_raw]
        rooms.append(
            RoomGT(
                room_id=str(room.get("room_id", idx)),
                room_type=normalize_category(room.get("room_type", "unknown")),
                polygon_xy=poly,
                area_m2=polygon_area(poly),
            )
        )
    return rooms


def assign_object_room(center_world: Sequence[float], rooms: Iterable[RoomGT]) -> Tuple[Optional[str], Optional[str]]:
    x, y = float(center_world[0]), float(center_world[1])
    for room in rooms:
        if point_in_polygon(x, y, room.polygon_xy):
            return room.room_id, room.room_type
    return None, None


def build_room_index_map(rooms: List[RoomGT], map_info: MapInfo, observed_map: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.full((map_info.height, map_info.width), -1, dtype=np.int16)
    for r in range(map_info.height):
        for c in range(map_info.width):
            if observed_map is not None and not bool(observed_map[r, c]):
                continue
            x, y = grid_to_world_xy(r, c, map_info)
            for idx, room in enumerate(rooms):
                if point_in_polygon(x, y, room.polygon_xy):
                    out[r, c] = idx
                    break
    return out
