from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.metrics.path_length import PathLengthAccumulator
from voxroom_online.isaac_runtime.metrics.spl import compute_softspl, compute_spl
from voxroom_online.isaac_runtime.navigation.astar import GridAStarPlanner


@dataclass
class EpisodeEvaluator:
    episode: dict
    full_map_planner: GridAStarPlanner
    path_accum: PathLengthAccumulator = field(default_factory=PathLengthAccumulator)
    start_wall_time: float = field(default_factory=time.time)
    num_steps: int = 0
    num_collisions: int = 0
    num_yolo_calls: int = 0
    num_detector_calls: int = 0
    num_scenegraph_updates: int = 0
    num_frontier_decisions: int = 0
    final_distance_to_goal: float = math.inf
    failure_reason: Optional[str] = None

    def goal_cells(self) -> List[Tuple[int, int]]:
        return [(int(r), int(c)) for r, c in self.episode.get("goal_regions_grid", [])]

    def update_pose(self, pose_world: Sequence[float], current_grid: Tuple[int, int], collided: bool = False) -> None:
        self.num_steps += 1
        self.path_accum.update((pose_world[0], pose_world[1]))
        if collided:
            self.num_collisions += 1
        self.final_distance_to_goal = self.full_map_planner.distance(current_grid, self.goal_cells())

    def finish(self, stop_called: bool, planner: str, detector: str, failure_reason: Optional[str] = None) -> dict:
        shortest = float(self.episode.get("shortest_path_distance_m", math.inf))
        success_distance = float(self.episode.get("success_distance_m", 1.0))
        success = bool(stop_called and self.final_distance_to_goal <= success_distance and failure_reason is None)
        path_length = float(self.path_accum.total_m)
        spl = compute_spl(success, shortest, path_length)
        softspl = compute_softspl(shortest, self.final_distance_to_goal, shortest, path_length)
        return {
            "episode_id": self.episode.get("episode_id"),
            "scene_id": self.episode.get("scene_id"),
            "goal_category": self.episode.get("goal_category"),
            "success": bool(success),
            "spl": float(spl),
            "softspl": float(softspl),
            "distance_to_goal": float(self.final_distance_to_goal),
            "initial_distance_to_goal": float(shortest),
            "shortest_path_distance": float(shortest),
            "path_length": path_length,
            "steps": int(self.num_steps),
            "num_steps": int(self.num_steps),
            "elapsed_wall_time_s": float(time.time() - self.start_wall_time),
            "collisions": int(self.num_collisions),
            "num_collisions": int(self.num_collisions),
            "num_detector_calls": int(self.num_detector_calls or self.num_yolo_calls),
            "num_yolo_calls": int(self.num_yolo_calls),
            "num_scenegraph_updates": int(self.num_scenegraph_updates),
            "num_frontier_decisions": int(self.num_frontier_decisions),
            "planner": planner,
            "detector": detector,
            "failure_reason": failure_reason,
        }
