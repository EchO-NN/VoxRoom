from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "CameraIntrinsics":
        fx = (width * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)
        fy = fx
        return cls(width=int(width), height=int(height), fx=float(fx), fy=float(fy), cx=(width - 1) * 0.5, cy=(height - 1) * 0.5)

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def yaw_to_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def pose_xyzyaw_to_matrix(pose: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, yaw = pose
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = yaw_to_matrix(yaw)
    mat[:3, 3] = [x, y, z]
    return mat


def camera_pose_from_base(base_pose: Tuple[float, float, float, float], mast_height_m: float = 1.2, forward_offset_m: float = 0.0) -> Tuple[float, float, float, float]:
    x, y, z, yaw = base_pose
    x += math.cos(yaw) * forward_offset_m
    y += math.sin(yaw) * forward_offset_m
    z += mast_height_m
    return float(x), float(y), float(z), float(yaw)
