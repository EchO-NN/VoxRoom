from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

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


def pose_world_to_matrix(pose: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a validated world transform from planar or full SE(3) input.

    Legacy Isaac callers provide ``(x, y, z, yaw)``. Real sensors provide a
    4x4 transform whose local axes follow VoxRoom's forward-left-up camera
    convention. Keeping both forms here lets the mapper preserve full 6DoF
    geometry without changing the simulation observation contract.
    """
    value = np.asarray(pose, dtype=np.float64)
    if value.shape == (4, 4):
        if not np.all(np.isfinite(value)):
            raise ValueError("pose transform contains non-finite values")
        if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6, rtol=0.0):
            raise ValueError("pose transform must have homogeneous bottom row [0, 0, 0, 1]")
        rotation = value[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4, rtol=0.0):
            raise ValueError("pose transform rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-4, rtol=0.0):
            raise ValueError("pose transform rotation determinant must be +1")
        return value.astype(np.float32)

    flat = value.reshape(-1)
    if flat.size != 4:
        raise ValueError("pose must be (x, y, z, yaw) or a 4x4 SE(3) transform")
    if not np.all(np.isfinite(flat)):
        raise ValueError("pose contains non-finite values")
    return pose_xyzyaw_to_matrix(tuple(float(v) for v in flat))


def camera_pose_from_base(base_pose: Tuple[float, float, float, float], mast_height_m: float = 1.2, forward_offset_m: float = 0.0) -> Tuple[float, float, float, float]:
    x, y, z, yaw = base_pose
    x += math.cos(yaw) * forward_offset_m
    y += math.sin(yaw) * forward_offset_m
    z += mast_height_m
    return float(x), float(y), float(z), float(yaw)
