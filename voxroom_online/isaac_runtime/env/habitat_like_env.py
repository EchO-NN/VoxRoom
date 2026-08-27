from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from voxroom_online.isaac_runtime.dataset.episode_generator import read_jsonl
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.sensors.camera_geometry import camera_pose_from_base


class MapSimHabitatLikeEnv:
    """Lightweight deterministic fallback environment for benchmark plumbing."""

    def __init__(self, episode_file: str, episode_index: int = 0, rgb_shape=(480, 640, 3)):
        self.episodes = read_jsonl(episode_file)
        self.episode = self.episodes[int(episode_index)]
        self.rgb_shape = rgb_shape
        self.pose_world = list(self.episode["start_pose_world"])
        self.step_count = 0

    def reset(self) -> dict:
        self.pose_world = list(self.episode["start_pose_world"])
        self.step_count = 0
        return self.observation()

    def set_pose(self, pose_world) -> dict:
        self.pose_world = list(pose_world)
        self.step_count += 1
        return self.observation()

    def observation(self) -> dict:
        rgb = np.zeros(self.rgb_shape, dtype=np.uint8)
        depth = np.full((self.rgb_shape[0], self.rgb_shape[1]), 3.0, dtype=np.float32)
        return {
            "rgb": rgb,
            "depth": depth,
            "pose_world": list(self.pose_world),
            "camera_pose_world": camera_pose_from_base(tuple(self.pose_world)),
            "sim_time": float(self.step_count) / 10.0,
            "collided": False,
        }

