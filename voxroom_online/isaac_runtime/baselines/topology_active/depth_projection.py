from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..data_contract import MapInfo
from .detector import CameraIntrinsics, DoorDetection2D, camera_intrinsics_from_mapping


@dataclass(frozen=True)
class ProjectionResult:
    rc: tuple[int, int]
    world_xyz: tuple[float, float, float]
    depth_m: float
    sample_count: int


@dataclass(frozen=True)
class ProjectionAttempt:
    result: ProjectionResult | None
    status: str


def project_door_bbox_to_grid_rc(
    *,
    detection: DoorDetection2D,
    depth: np.ndarray,
    camera_intrinsics: Any,
    camera_pose_world: np.ndarray,
    map_info: MapInfo,
    sample_radius_px: int = 2,
) -> ProjectionResult | None:
    return project_door_bbox_to_grid_rc_with_status(
        detection=detection,
        depth=depth,
        camera_intrinsics=camera_intrinsics,
        camera_pose_world=camera_pose_world,
        map_info=map_info,
        sample_radius_px=sample_radius_px,
    ).result


def project_door_bbox_to_grid_rc_with_status(
    *,
    detection: DoorDetection2D,
    depth: np.ndarray,
    camera_intrinsics: Any,
    camera_pose_world: np.ndarray,
    map_info: MapInfo,
    sample_radius_px: int = 2,
) -> ProjectionAttempt:
    intr = camera_intrinsics_from_mapping(camera_intrinsics)
    if intr is None:
        return ProjectionAttempt(result=None, status="missing_intrinsics")
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 2:
        return ProjectionAttempt(result=None, status="invalid_depth_shape")
    u, v = _door_sample_uv(detection)
    z, count = _median_valid_depth(depth_arr, u=u, v=v, radius=int(sample_radius_px))
    if z is None:
        z, count = _median_valid_depth_in_bbox(depth_arr, detection=detection)
    if z is None:
        return ProjectionAttempt(result=None, status="invalid_depth")
    x_cam = (float(u) - float(intr.cx)) * float(z) / float(intr.fx)
    y_cam = (float(v) - float(intr.cy)) * float(z) / float(intr.fy)
    z_cam = float(z)
    p_world = _camera_point_to_world(
        x_cam=float(x_cam),
        y_cam=float(y_cam),
        z_cam=float(z_cam),
        camera_pose_world=camera_pose_world,
    )
    if p_world is None:
        return ProjectionAttempt(result=None, status="invalid_pose")
    col = int(np.floor((float(p_world[0]) - float(map_info.min_x)) / float(map_info.resolution_m)))
    row = int(np.floor((float(map_info.max_y) - float(p_world[1])) / float(map_info.resolution_m)))
    if row < 0 or row >= int(map_info.height) or col < 0 or col >= int(map_info.width):
        return ProjectionAttempt(result=None, status="out_of_grid")
    return ProjectionAttempt(
        result=ProjectionResult(
            rc=(int(row), int(col)),
            world_xyz=(float(p_world[0]), float(p_world[1]), float(p_world[2])),
            depth_m=float(z),
            sample_count=int(count),
        ),
        status="ok",
    )


def _camera_point_to_world(
    *,
    x_cam: float,
    y_cam: float,
    z_cam: float,
    camera_pose_world: Any,
) -> np.ndarray | None:
    pose = np.asarray(camera_pose_world, dtype=np.float64)
    if pose.shape == (4, 4):
        return pose @ np.asarray([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)

    flat = pose.reshape(-1)
    if flat.size < 4:
        return None

    cam_x, cam_y, cam_z, cam_yaw = [float(v) for v in flat[:4]]
    c = math.cos(cam_yaw)
    s = math.sin(cam_yaw)

    # Runtime Isaac observations store camera_pose_world as (x, y, z, yaw).
    # Match sensors.depth_backproject: camera x is right, y is down, z is
    # forward; world/local robot axes are forward x, left y, up z.
    local_forward_left_up = np.asarray([z_cam, -x_cam, -y_cam], dtype=np.float64)
    world_x = cam_x + c * local_forward_left_up[0] - s * local_forward_left_up[1]
    world_y = cam_y + s * local_forward_left_up[0] + c * local_forward_left_up[1]
    world_z = cam_z + local_forward_left_up[2]
    return np.asarray([world_x, world_y, world_z, 1.0], dtype=np.float64)


def _door_sample_uv(detection: DoorDetection2D) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in detection.bbox_xyxy]
    u = (x0 + x1) * 0.5
    v = y1 - 0.10 * max(0.0, y1 - y0)
    return float(u), float(v)


def _median_valid_depth(depth: np.ndarray, *, u: float, v: float, radius: int) -> tuple[float | None, int]:
    h, w = depth.shape
    cu = int(round(float(u)))
    cv = int(round(float(v)))
    r0 = max(0, cv - max(0, int(radius)))
    r1 = min(h, cv + max(0, int(radius)) + 1)
    c0 = max(0, cu - max(0, int(radius)))
    c1 = min(w, cu + max(0, int(radius)) + 1)
    if r0 >= r1 or c0 >= c1:
        return None, 0
    patch = np.asarray(depth[r0:r1, c0:c1], dtype=np.float32)
    vals = patch[np.isfinite(patch) & (patch > 0.0)]
    if vals.size == 0:
        return None, 0
    return float(np.median(vals)), int(vals.size)


def _median_valid_depth_in_bbox(depth: np.ndarray, *, detection: DoorDetection2D) -> tuple[float | None, int]:
    h, w = depth.shape
    x0, y0, x1, y1 = [int(round(float(v))) for v in detection.bbox_xyxy]
    c0 = max(0, min(w - 1, min(x0, x1)))
    c1 = max(0, min(w, max(x0, x1) + 1))
    r0 = max(0, min(h - 1, min(y0, y1)))
    r1 = max(0, min(h, max(y0, y1) + 1))
    if r0 >= r1 or c0 >= c1:
        return None, 0

    # Prefer the lower half of the door box because it is usually closer to the
    # navigation-map door point. Fall back to the full box if that slice is empty.
    lower_r0 = int(round(0.5 * (r0 + r1)))
    patches = [depth[lower_r0:r1, c0:c1], depth[r0:r1, c0:c1]]
    for patch in patches:
        vals = np.asarray(patch, dtype=np.float32)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size:
            return float(np.median(vals)), int(vals.size)
    return None, 0
