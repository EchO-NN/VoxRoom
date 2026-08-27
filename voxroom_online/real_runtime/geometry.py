from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform described as translation plus roll/pitch/yaw.

    The transform maps points from the child frame into the parent frame.
    Rotations use the robotics convention ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """

    xyz_m: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, data: dict | None = None) -> "RigidTransform":
        raw = dict(data or {})
        xyz = tuple(float(v) for v in raw.get("base_to_left_camera_xyz_m", (0.0, 0.0, 1.35)))
        rpy = tuple(float(v) for v in raw.get("base_to_left_camera_rpy_deg", (0.0, 0.0, 0.0)))
        if len(xyz) != 3 or len(rpy) != 3:
            raise ValueError("extrinsics xyz and rpy must each contain exactly three values")
        if not np.all(np.isfinite(np.asarray((*xyz, *rpy), dtype=np.float64))):
            raise ValueError("extrinsics must contain only finite values")
        return cls(xyz_m=xyz, rpy_deg=rpy)

    def matrix(self) -> np.ndarray:
        roll, pitch, yaw = (math.radians(float(v)) for v in self.rpy_deg)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = rz @ ry @ rx
        out[:3, 3] = np.asarray(self.xyz_m, dtype=np.float64)
        return out


def quaternion_xyzw_to_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError("quaternion must contain [x, y, z, w]")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("quaternion norm is zero or non-finite")
    x, y, z, w = (q / norm).tolist()
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_from_translation_quaternion(
    translation_xyz: Sequence[float], quaternion_xyzw: Sequence[float]
) -> np.ndarray:
    t = np.asarray(translation_xyz, dtype=np.float64).reshape(-1)
    if t.size != 3 or not np.all(np.isfinite(t)):
        raise ValueError("translation must contain three finite values")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quaternion_xyzw_to_matrix(quaternion_xyzw)
    out[:3, 3] = t
    return out


def rotation_matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch and yaw in radians for Rz(yaw)Ry(pitch)Rx(roll)."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    sin_pitch = float(np.clip(-r[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-r[0, 1]), float(r[1, 1]))
    return float(roll), float(pitch), float(yaw)


def matrix_to_planar_pose(transform: np.ndarray) -> tuple[float, float, float, float]:
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    _, _, yaw = rotation_matrix_to_rpy(t[:3, :3])
    return float(t[0, 3]), float(t[1, 3]), float(t[2, 3]), float(yaw)


def planar_transform_from_matrix(transform: np.ndarray) -> np.ndarray:
    x, y, z, yaw = matrix_to_planar_pose(transform)
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    out[:3, 3] = (x, y, z)
    return out


def camera_and_base_poses(
    t_zed_left_camera: np.ndarray,
    t_base_left_camera: np.ndarray,
    t_world_zed: np.ndarray | None,
    *,
    preserve_zed_floor_z: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve normalized camera/base poses and initialize the local world.

    ``t_zed_left_camera`` is the ZED SDK world pose of the left eye.
    ``t_base_left_camera`` maps left-camera points into ``base_link``.
    The returned local world is initialized at the first planar base pose. If
    ZED floor alignment is active, only XY and yaw are normalized so the SDK's
    detected floor remains at world z=0.
    """
    t_z_c = np.asarray(t_zed_left_camera, dtype=np.float64)
    t_b_c = np.asarray(t_base_left_camera, dtype=np.float64)
    if t_z_c.shape != (4, 4) or t_b_c.shape != (4, 4):
        raise ValueError("camera and extrinsic transforms must be 4x4")
    t_z_b = t_z_c @ np.linalg.inv(t_b_c)
    if t_world_zed is None:
        normalization_anchor = planar_transform_from_matrix(t_z_b)
        if bool(preserve_zed_floor_z):
            normalization_anchor[2, 3] = 0.0
        t_world_zed = np.linalg.inv(normalization_anchor)
    t_w_c = np.asarray(t_world_zed, dtype=np.float64) @ t_z_c
    t_w_b = t_w_c @ np.linalg.inv(t_b_c)
    return t_w_c, t_w_b, np.asarray(t_world_zed, dtype=np.float64)
