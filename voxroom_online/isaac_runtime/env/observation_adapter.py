from __future__ import annotations

from typing import Mapping

import numpy as np


def isaac_to_sgnav_obs(obs: Mapping, episode: Mapping) -> dict:
    x, y, _z, yaw = obs["pose_world"]
    return {
        "rgb": obs["rgb"],
        "depth": obs["depth"],
        "gps": np.asarray([x, y], dtype=np.float32),
        "compass": np.asarray([yaw], dtype=np.float32),
        "objectgoal": episode["goal_category"],
        "camera_pose_world": obs.get("camera_pose_world"),
    }

