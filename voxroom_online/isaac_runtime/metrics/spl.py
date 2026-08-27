from __future__ import annotations

import math
from typing import Iterable, Tuple

from voxroom_online.isaac_runtime.navigation.astar import GridAStarPlanner

GridCell = Tuple[int, int]


def distance_to_goal(planner: GridAStarPlanner, current: GridCell, goal_regions: Iterable[GridCell]) -> float:
    return planner.distance(current, goal_regions)


def compute_spl(success: bool, shortest_path_distance_m: float, actual_path_length_m: float) -> float:
    if not success or not math.isfinite(shortest_path_distance_m):
        return 0.0
    denom = max(float(actual_path_length_m), float(shortest_path_distance_m), 1e-6)
    return float(shortest_path_distance_m) / denom


def compute_softspl(initial_distance_m: float, final_distance_m: float, shortest_path_distance_m: float, actual_path_length_m: float) -> float:
    if not math.isfinite(initial_distance_m) or initial_distance_m <= 1e-6:
        progress = 1.0 if final_distance_m <= 1e-6 else 0.0
    else:
        progress = max(0.0, 1.0 - float(final_distance_m) / max(float(initial_distance_m), 1e-6))
    if not math.isfinite(shortest_path_distance_m):
        return 0.0
    denom = max(float(actual_path_length_m), float(shortest_path_distance_m), 1e-6)
    return progress * float(shortest_path_distance_m) / denom

