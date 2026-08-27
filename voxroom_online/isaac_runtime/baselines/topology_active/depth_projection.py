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


@dataclass(frozen=True)
class MaskProjectionResult:
    rc: np.ndarray
    world_xyz: np.ndarray
    depth_m: np.ndarray
    sampled_mask_pixels: int
    valid_depth_pixels: int
    height_band_pixels: int


@dataclass(frozen=True)
class MaskProjectionAttempt:
    result: MaskProjectionResult | None
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


def project_door_mask_to_grid_rc_with_status(
    *,
    detection: DoorDetection2D,
    depth: np.ndarray,
    camera_intrinsics: Any,
    camera_pose_world: np.ndarray,
    map_info: MapInfo,
    floor_z: float,
    pixel_stride: int = 2,
    depth_max_m: float = 3.0,
    relative_z_min_m: float = 0.25,
    relative_z_max_m: float = 1.50,
) -> MaskProjectionAttempt:
    """Project the upstream DETR door mask with its native map-builder limits."""

    intr = camera_intrinsics_from_mapping(camera_intrinsics)
    if intr is None:
        return MaskProjectionAttempt(result=None, status="missing_intrinsics")
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 2:
        return MaskProjectionAttempt(result=None, status="invalid_depth_shape")

    raw_mask = getattr(detection, "mask", None)
    if raw_mask is None:
        return MaskProjectionAttempt(result=None, status="missing_mask")
    mask = np.asarray(raw_mask, dtype=bool)
    if mask.shape != depth_arr.shape:
        return MaskProjectionAttempt(result=None, status="invalid_mask_shape")

    stride = int(pixel_stride)
    if stride <= 0:
        raise ValueError("pixel_stride must be positive")
    if not np.isfinite(float(depth_max_m)) or float(depth_max_m) <= 0.0:
        raise ValueError("depth_max_m must be finite and positive")
    if not float(relative_z_min_m) < float(relative_z_max_m):
        raise ValueError("relative_z_min_m must be less than relative_z_max_m")

    sampled_rows, sampled_cols = np.nonzero(mask[::stride, ::stride])
    if sampled_rows.size == 0:
        return MaskProjectionAttempt(result=None, status="empty_mask")
    rows = sampled_rows.astype(np.int64) * stride
    cols = sampled_cols.astype(np.int64) * stride
    z = depth_arr[rows, cols].astype(np.float64)
    valid_depth = np.isfinite(z) & (z > 0.0) & (z <= float(depth_max_m))
    valid_depth_count = int(np.count_nonzero(valid_depth))
    if valid_depth_count == 0:
        return MaskProjectionAttempt(result=None, status="invalid_depth")
    rows = rows[valid_depth]
    cols = cols[valid_depth]
    z = z[valid_depth]

    x_cam = (cols.astype(np.float64) - float(intr.cx)) * z / float(intr.fx)
    y_cam = (rows.astype(np.float64) - float(intr.cy)) * z / float(intr.fy)
    points_world = _camera_points_to_world(
        x_cam=x_cam,
        y_cam=y_cam,
        z_cam=z,
        camera_pose_world=camera_pose_world,
    )
    if points_world is None:
        return MaskProjectionAttempt(result=None, status="invalid_pose")

    relative_z = points_world[:, 2] - float(floor_z)
    in_height_band = (
        (relative_z >= float(relative_z_min_m))
        & (relative_z < float(relative_z_max_m))
    )
    height_band_count = int(np.count_nonzero(in_height_band))
    if height_band_count == 0:
        return MaskProjectionAttempt(result=None, status="outside_height_band")
    points_world = points_world[in_height_band]
    z = z[in_height_band]

    cols_grid = np.floor(
        (points_world[:, 0] - float(map_info.min_x)) / float(map_info.resolution_m)
    ).astype(np.int64)
    rows_grid = np.floor(
        (float(map_info.max_y) - points_world[:, 1]) / float(map_info.resolution_m)
    ).astype(np.int64)
    inside = (
        (rows_grid >= 0)
        & (rows_grid < int(map_info.height))
        & (cols_grid >= 0)
        & (cols_grid < int(map_info.width))
    )
    if not np.any(inside):
        return MaskProjectionAttempt(result=None, status="out_of_grid")
    points_world = points_world[inside, :3]
    z = z[inside]
    rc = np.stack([rows_grid[inside], cols_grid[inside]], axis=1).astype(np.int32)
    rc, unique_indices = np.unique(rc, axis=0, return_index=True)
    points_world = points_world[unique_indices]
    z = z[unique_indices]
    return MaskProjectionAttempt(
        result=MaskProjectionResult(
            rc=rc,
            world_xyz=points_world.astype(np.float64, copy=False),
            depth_m=z.astype(np.float64, copy=False),
            sampled_mask_pixels=int(sampled_rows.size),
            valid_depth_pixels=valid_depth_count,
            height_band_pixels=height_band_count,
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
    points = _camera_points_to_world(
        x_cam=np.asarray([x_cam], dtype=np.float64),
        y_cam=np.asarray([y_cam], dtype=np.float64),
        z_cam=np.asarray([z_cam], dtype=np.float64),
        camera_pose_world=camera_pose_world,
    )
    if points is None:
        return None
    return points[0]


def _camera_points_to_world(
    *,
    x_cam: np.ndarray,
    y_cam: np.ndarray,
    z_cam: np.ndarray,
    camera_pose_world: Any,
) -> np.ndarray | None:
    x_values = np.asarray(x_cam, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y_cam, dtype=np.float64).reshape(-1)
    z_values = np.asarray(z_cam, dtype=np.float64).reshape(-1)
    if not (x_values.size == y_values.size == z_values.size):
        raise ValueError("camera point coordinate arrays must have equal length")

    # Optical coordinates are right/down/forward. VoxRoom poses, including full
    # SE(3) poses, use forward/left/up local axes.
    local_points = np.stack(
        [z_values, -x_values, -y_values, np.ones_like(z_values)],
        axis=1,
    )
    pose = np.asarray(camera_pose_world, dtype=np.float64)
    if pose.shape == (4, 4):
        return (pose @ local_points.T).T

    flat = pose.reshape(-1)
    if flat.size < 4:
        return None

    cam_x, cam_y, cam_z, cam_yaw = [float(v) for v in flat[:4]]
    c = math.cos(cam_yaw)
    s = math.sin(cam_yaw)

    world = np.empty_like(local_points)
    world[:, 0] = cam_x + c * local_points[:, 0] - s * local_points[:, 1]
    world[:, 1] = cam_y + s * local_points[:, 0] + c * local_points[:, 1]
    world[:, 2] = cam_z + local_points[:, 2]
    world[:, 3] = 1.0
    return world


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
