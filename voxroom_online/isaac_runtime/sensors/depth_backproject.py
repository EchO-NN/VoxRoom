from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from voxroom_online.isaac_runtime.perception.detection_types import Detection2D, Detection3D
from voxroom_online.isaac_runtime.sensors.camera_geometry import CameraIntrinsics, pose_xyzyaw_to_matrix


def backproject_pixels(depth: np.ndarray, pixels_uv: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    u = pixels_uv[:, 0].astype(np.float32)
    v = pixels_uv[:, 1].astype(np.float32)
    z = depth[v.astype(np.int64), u.astype(np.int64)].astype(np.float32)
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    # Camera convention used here: x right, y down, z forward. Convert to
    # robot/world convention: forward x, left y, up z before yaw transform.
    return np.stack([z, -x, -y], axis=1)


def backproject_pixels_at_depth_values(pixels_uv: np.ndarray, depth_values_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.float32).reshape(-1, 2)
    depth_values = np.asarray(depth_values_m, dtype=np.float32).reshape(-1)
    if pixels.shape[0] != depth_values.shape[0]:
        raise ValueError("depth_values_m must match pixels_uv length")
    u = pixels[:, 0]
    v = pixels[:, 1]
    z = depth_values
    x = (u - float(intr.cx)) * z / float(intr.fx)
    y = (v - float(intr.cy)) * z / float(intr.fy)
    return np.stack([z, -x, -y], axis=1).astype(np.float32)


def distance_to_camera_to_image_plane_depth(distance_to_camera: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """Convert Euclidean camera distance to image-plane z-depth per pixel."""
    dist = np.asarray(distance_to_camera, dtype=np.float32)
    if dist.ndim == 3:
        dist = dist[:, :, 0]
    if dist.ndim != 2:
        raise ValueError("distance_to_camera must be a 2D image")
    h, w = dist.shape
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys)
    x_over_z = (uu - float(intr.cx)) / float(intr.fx)
    y_over_z = (vv - float(intr.cy)) / float(intr.fy)
    scale = np.sqrt(1.0 + x_over_z * x_over_z + y_over_z * y_over_z).astype(np.float32)
    out = dist / np.maximum(scale, np.float32(1.0e-6))
    invalid = ~np.isfinite(dist)
    if np.any(invalid):
        out[invalid] = dist[invalid]
    return out.astype(np.float32, copy=False)


def transform_points(points: np.ndarray, pose_xyzyaw: Tuple[float, float, float, float]) -> np.ndarray:
    mat = pose_xyzyaw_to_matrix(pose_xyzyaw)
    homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (mat @ homo.T).T[:, :3]


def detection_to_world_points(
    det: Detection2D,
    depth: np.ndarray,
    intr: CameraIntrinsics,
    camera_pose_world: Tuple[float, float, float, float],
    depth_min_m: float = 0.05,
    depth_max_m: float = 5.0,
    min_points: int = 20,
    stride: int = 4,
) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
    x1 = max(0, min(intr.width - 1, x1))
    x2 = max(0, min(intr.width, x2))
    y1 = max(0, min(intr.height - 1, y1))
    y2 = max(0, min(intr.height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    if det.mask is not None:
        mask = np.asarray(det.mask).astype(bool)
        if mask.shape[:2] == depth.shape[:2]:
            ys, xs = np.nonzero(mask[y1:y2, x1:x2])
            if len(xs) > 0:
                sample = max(1, int(stride))
                pixels = np.stack([xs[::sample] + x1, ys[::sample] + y1], axis=1).astype(np.float32)
            else:
                pixels = np.zeros((0, 2), dtype=np.float32)
        else:
            pixels = np.zeros((0, 2), dtype=np.float32)
    else:
        vv, uu = np.mgrid[y1:y2:stride, x1:x2:stride]
        pixels = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1).astype(np.float32)
    if len(pixels) == 0:
        return None
    z = depth[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)].astype(np.float32)
    valid = np.isfinite(z) & (z > depth_min_m) & (z < depth_max_m)
    if int(valid.sum()) < min_points:
        return None
    pixels = pixels[valid]
    z = z[valid]
    if det.mask is None:
        foreground = _foreground_depth_mask(z, min_points=min_points)
        pixels = pixels[foreground]
    points_cam = backproject_pixels(depth, pixels, intr)
    return transform_points(points_cam, camera_pose_world)


def detections_to_3d(
    detections: Iterable[Detection2D],
    depth: np.ndarray,
    intr: CameraIntrinsics,
    camera_pose_world: Tuple[float, float, float, float],
    depth_max_m: float = 5.0,
    min_points: int = 20,
) -> List[Detection3D]:
    out: List[Detection3D] = []
    for det in detections:
        points = detection_to_world_points(det, depth, intr, camera_pose_world, depth_max_m=depth_max_m, min_points=min_points)
        if points is None or len(points) == 0:
            continue
        center = np.median(points, axis=0)
        bbox_world = world_points_bbox(points)
        out.append(
            Detection3D(
                category=det.category,
                raw_label=det.raw_label,
                confidence=det.confidence,
                center_world=tuple(float(v) for v in center),
                bbox_xyxy=det.bbox_xyxy,
                point_cloud_world=points,
                bbox_world=bbox_world,
                mask=None if det.mask is None else np.asarray(det.mask).astype(bool).copy(),
            )
        )
    return out


def world_points_bbox(points_world: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return np.zeros((2, 3), dtype=np.float32)
    return np.stack([np.min(points, axis=0), np.max(points, axis=0)], axis=0).astype(np.float32)


def _foreground_depth_mask(depth_values: np.ndarray, min_points: int = 20) -> np.ndarray:
    """Prefer foreground depth inside a bbox when no SAM mask is available.

    Raw detector boxes often include background wall/floor. Masked pipelines
    use masks before point-cloud projection; for bbox-only fallback we
    keep a near-depth band so object centers are not pulled behind the object.
    """
    z = np.asarray(depth_values, dtype=np.float32).reshape(-1)
    if len(z) == 0:
        return np.zeros((0,), dtype=bool)
    if len(z) <= max(4, int(min_points)):
        return np.ones_like(z, dtype=bool)
    near = float(np.percentile(z, 20.0))
    band = max(0.20, 0.15 * max(near, 0.0))
    keep = z <= near + band
    if int(np.count_nonzero(keep)) >= int(min_points):
        return keep
    return np.ones_like(z, dtype=bool)
