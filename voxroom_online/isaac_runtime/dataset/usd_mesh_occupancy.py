from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo


def _world_xy_array_to_grid(points_xy: np.ndarray, map_info: MapInfo) -> np.ndarray:
    resolution = float(map_info.resolution_m)
    if resolution <= 0.0:
        raise ValueError("map resolution must be positive")
    columns = np.rint(
        (points_xy[..., 0] - float(map_info.min_x)) / resolution - 0.5
    )
    rows = np.rint(
        (float(map_info.max_y) - points_xy[..., 1]) / resolution - 0.5
    )
    return np.stack((columns, rows), axis=-1).astype(np.int32)


def _transform_points_row_vector(
    points_xyz: np.ndarray,
    transform_matrix: np.ndarray,
) -> np.ndarray:
    source_points = np.asarray(points_xyz)
    points = np.asarray(source_points, dtype=np.float64)
    matrix = np.asarray(transform_matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("mesh points must have shape (N, 3)")
    if matrix.shape != (4, 4):
        raise ValueError("USD transform matrix must have shape (4, 4)")
    homogeneous = np.ones((points.shape[0], 4), dtype=np.float64)
    homogeneous[:, :3] = points
    transformed = homogeneous @ matrix
    weights = transformed[:, 3]
    if np.any(np.isclose(weights, 0.0)):
        raise ValueError("USD transform produced a zero homogeneous weight")
    return transformed[:, :3] / weights[:, None]


def _restore_grid_boundary_vertices(
    points_world: np.ndarray,
    points_local: Sequence[Any],
    transform: Any,
    map_info: MapInfo,
) -> tuple[np.ndarray, int]:
    resolution = float(map_info.resolution_m)
    x_cell_boundaries = (
        points_world[:, 0] - float(map_info.min_x)
    ) / resolution
    y_cell_boundaries = (
        float(map_info.max_y) - points_world[:, 1]
    ) / resolution
    tolerance_cells = 5.0e-4
    near_boundary = (
        np.abs(x_cell_boundaries - np.rint(x_cell_boundaries))
        <= tolerance_cells
    ) | (
        np.abs(y_cell_boundaries - np.rint(y_cell_boundaries))
        <= tolerance_cells
    )
    indices = np.flatnonzero(near_boundary)
    if indices.size == 0:
        return points_world, 0
    corrected = np.array(points_world, copy=True)
    corrected[indices] = np.asarray(
        [transform.Transform(points_local[int(index)]) for index in indices],
        dtype=np.float64,
    )
    return corrected, int(indices.size)


def _rasterize_indexed_faces(
    occupancy: np.ndarray,
    points_world: np.ndarray,
    face_vertex_counts: np.ndarray,
    face_vertex_indices: np.ndarray,
    map_info: MapInfo,
    *,
    min_z_m: float | None = None,
    max_z_m: float | None = None,
) -> int:
    if (min_z_m is None) != (max_z_m is None):
        raise ValueError("both face height limits must be provided together")
    if min_z_m is not None and float(min_z_m) > float(max_z_m):
        raise ValueError("minimum face height exceeds maximum face height")

    def intersects_height_band(face_indices: np.ndarray) -> np.ndarray:
        if min_z_m is None:
            return np.ones(face_indices.shape[0], dtype=bool)
        face_z = points_world[face_indices, 2]
        return (np.max(face_z, axis=1) >= float(min_z_m)) & (
            np.min(face_z, axis=1) <= float(max_z_m)
        )

    counts = np.asarray(face_vertex_counts, dtype=np.int64).reshape(-1)
    indices = np.asarray(face_vertex_indices, dtype=np.int64).reshape(-1)
    if counts.size == 0 or indices.size == 0:
        return 0
    if np.any(counts < 0) or int(np.sum(counts)) != int(indices.size):
        raise ValueError("USD mesh face counts do not match face indices")
    if np.any(indices < 0) or np.any(indices >= int(points_world.shape[0])):
        raise ValueError("USD mesh face index is outside the point array")

    valid_face_count = int(np.count_nonzero(counts >= 3))
    if valid_face_count <= 0:
        return 0
    unique_counts = np.unique(counts)
    if unique_counts.size == 1 and int(unique_counts[0]) >= 3:
        vertices_per_face = int(unique_counts[0])
        faces = indices.reshape((-1, vertices_per_face))
        faces = faces[intersects_height_band(faces)]
        if faces.size == 0:
            return 0
        polygons_xy = points_world[faces, :2]
        polygons_grid = _world_xy_array_to_grid(polygons_xy, map_info)
        cv2.fillPoly(
            occupancy,
            polygons_grid.reshape((-1, vertices_per_face, 1, 2)),
            color=1,
        )
        return int(faces.shape[0])

    polygons_grid: list[np.ndarray] = []
    offset = 0
    for count_value in counts:
        count = int(count_value)
        face_indices = indices[offset : offset + count]
        offset += count
        if count < 3:
            continue
        if not intersects_height_band(face_indices.reshape(1, -1))[0]:
            continue
        polygon_xy = points_world[face_indices, :2]
        polygons_grid.append(
            _world_xy_array_to_grid(polygon_xy, map_info).reshape((-1, 1, 2))
        )
    if polygons_grid:
        cv2.fillPoly(occupancy, polygons_grid, color=1)
    return len(polygons_grid)


def world_xy_polygon_to_grid(
    polygon_xy: Sequence[Sequence[float]],
    map_info: MapInfo,
) -> np.ndarray:
    points = np.asarray(polygon_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("world polygon must have shape (N, 2) with N >= 3")
    if not np.all(np.isfinite(points)):
        raise ValueError("world polygon contains non-finite coordinates")
    return _world_xy_array_to_grid(points, map_info)


def rasterize_world_xy_polygons(
    polygons_xy: Sequence[Sequence[Sequence[float]]],
    map_info: MapInfo,
) -> np.ndarray:
    occupancy = np.zeros((int(map_info.height), int(map_info.width)), dtype=np.uint8)
    polygons_grid = [
        world_xy_polygon_to_grid(polygon, map_info).reshape((-1, 1, 2))
        for polygon in polygons_xy
    ]
    if polygons_grid:
        cv2.fillPoly(occupancy, polygons_grid, color=1)
    return occupancy


def rasterize_usd_mesh_occupancy(
    usd_path: str | Path,
    obstacle_objects: Sequence[Mapping[str, Any]],
    map_info: MapInfo,
    *,
    min_z_m: float,
    max_z_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if float(min_z_m) > float(max_z_m):
        raise ValueError("minimum obstacle height exceeds maximum obstacle height")
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception as exc:
        raise RuntimeError("USD mesh occupancy requires pxr.Usd and pxr.UsdGeom") from exc

    stage = Usd.Stage.Open(str(Path(usd_path).expanduser()))
    if stage is None:
        raise RuntimeError("failed to open USD stage for mesh occupancy: %s" % usd_path)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    occupancy = np.zeros((int(map_info.height), int(map_info.width)), dtype=np.uint8)
    unresolved_paths: list[str] = []
    object_count = 0
    objects_intersecting_height_band = 0
    objects_outside_height_band = 0
    mesh_count = 0
    invisible_mesh_count = 0
    face_count = 0
    exact_boundary_vertex_count = 0

    for obj in obstacle_objects:
        prim_path = str(obj.get("prim_path") or obj.get("instance_id") or "")
        if not prim_path:
            raise ValueError("blocking object has no prim_path")
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim or not root_prim.IsValid():
            unresolved_paths.append(prim_path)
            continue
        object_count += 1
        object_has_mesh = False
        object_meshes = 0
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            object_has_mesh = True
            if (
                UsdGeom.Imageable(prim).ComputeVisibility()
                == UsdGeom.Tokens.invisible
            ):
                invisible_mesh_count += 1
                continue
            mesh = UsdGeom.Mesh(prim)
            points_local = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if (
                points_local is None
                or counts is None
                or indices is None
                or len(points_local) == 0
                or len(counts) == 0
                or len(indices) == 0
            ):
                continue
            transform = xform_cache.GetLocalToWorldTransform(prim)
            points_world = _transform_points_row_vector(
                np.asarray(points_local),
                np.asarray(transform, dtype=np.float64),
            )
            points_world, corrected_vertices = _restore_grid_boundary_vertices(
                points_world,
                points_local,
                transform,
                map_info,
            )
            exact_boundary_vertex_count += corrected_vertices
            mesh_min_z = float(np.min(points_world[:, 2]))
            mesh_max_z = float(np.max(points_world[:, 2]))
            if mesh_max_z < float(min_z_m) or mesh_min_z > float(max_z_m):
                continue

            projected_faces = _rasterize_indexed_faces(
                occupancy,
                points_world,
                np.asarray(counts, dtype=np.int64),
                np.asarray(indices, dtype=np.int64),
                map_info,
                min_z_m=min_z_m,
                max_z_m=max_z_m,
            )
            if projected_faces > 0:
                object_meshes += 1
                mesh_count += 1
                face_count += projected_faces
        if not object_has_mesh:
            unresolved_paths.append(prim_path)
        elif object_meshes <= 0:
            objects_outside_height_band += 1
        else:
            objects_intersecting_height_band += 1

    if unresolved_paths:
        sample = unresolved_paths[:20]
        raise RuntimeError(
            "USD mesh occupancy could not resolve blocking geometry: "
            f"count={len(unresolved_paths)} sample={sample}"
        )
    if object_count <= 0 or mesh_count <= 0 or not np.any(occupancy):
        raise RuntimeError("USD mesh occupancy produced no blocking geometry")
    return occupancy, {
        "source": "usd_mesh_face_projection",
        "obstacle_object_count": int(object_count),
        "objects_intersecting_height_band": int(objects_intersecting_height_band),
        "objects_outside_height_band": int(objects_outside_height_band),
        "mesh_count": int(mesh_count),
        "invisible_mesh_count": int(invisible_mesh_count),
        "projected_face_count": int(face_count),
        "exact_boundary_vertex_count": int(exact_boundary_vertex_count),
        "occupied_cells_before_opening_carve": int(np.count_nonzero(occupancy)),
        "unresolved_object_count": 0,
    }
