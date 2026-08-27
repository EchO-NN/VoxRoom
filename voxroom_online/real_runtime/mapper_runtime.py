from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from voxroom_online.isaac_runtime.graph.room_context import resolve_replay_style_navigation_masks
from voxroom_online.isaac_runtime.mapping.online_mapper import OnlineMapper
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_door_wall_roomseg import (
    VoxelOccupancyDoorWallRoomSegConfig,
    VoxelOccupancyDoorWallRoomSegmenter,
)


VALID_MODES = {"smoke_2d", "full_voxel"}
COMPARISON_ROOM_COLORS = np.asarray(
    [
        (68, 199, 183),
        (232, 62, 157),
        (104, 174, 232),
        (201, 178, 124),
        (98, 185, 143),
        (245, 132, 31),
        (145, 104, 207),
        (232, 111, 81),
        (75, 155, 205),
        (170, 194, 89),
        (219, 129, 188),
        (111, 184, 191),
    ],
    dtype=np.uint8,
)


def load_yaml(path: str | Path) -> dict:
    file_path = Path(path).expanduser().resolve()
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {file_path}")
    return data


def build_mapper(base_cfg: dict, real_cfg: dict, mode: str) -> OnlineMapper:
    mode = str(mode).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    mapping = copy.deepcopy(dict(base_cfg.get("mapping", {}) or {}))
    room_cfg = copy.deepcopy(dict(mapping.get("room_segmentation", {}) or {}))
    real_mapping = dict(real_cfg.get("mapping", {}) or {})

    vertical_cfg = dict(room_cfg.get("vertical_or_free", {}) or {})
    height_cfg = dict(room_cfg.get("height_profile", {}) or {})
    voxel_cfg = dict(room_cfg.get("voxel_grid", {}) or {})
    outside_cfg = dict(room_cfg.get("voxel_outside", {}) or {})
    nav_cfg = dict(room_cfg.get("voxel_navigation_projection", {}) or {})
    blind_cfg = dict(room_cfg.get("voxel_navigation_blind_zone", {}) or {})
    voxel_runtime_cfg = dict(mapping.get("voxel_runtime", {}) or {})

    voxel_enabled = mode == "full_voxel"
    voxel_cfg["enabled"] = bool(voxel_enabled)
    if not voxel_enabled:
        voxel_cfg["voxel_grid_drives_navigation"] = False
        # OnlineMapper exposes a voxel-grid object in its debug contract even
        # when integration is disabled.  One z-bin keeps smoke_2d lightweight.
        voxel_cfg["z_min_m"] = 0.0
        voxel_cfg["z_max_m"] = 0.10
        voxel_cfg["z_resolution_m"] = 0.10
        voxel_cfg["active_z_min_m"] = 0.0
        voxel_cfg["active_z_max_fallback_m"] = 0.10
        voxel_cfg["active_z_max_cap_m"] = 0.10

    depth_stride = int(mapping.get("depth_stride_px", 8))
    if voxel_enabled:
        depth_stride = int(dict(room_cfg.get("depth", {}) or {}).get("roomseg_depth_stride_px", depth_stride))

    robot_cfg = dict(base_cfg.get("robot", {}) or {})
    robot_radius_m = float(
        real_mapping.get(
            "robot_radius_m",
            robot_cfg.get("footprint_radius_m", mapping.get("robot_radius_m", 0.14)),
        )
    )
    size_m = float(real_mapping.get("map_size_m", mapping.get("map_size_m", 40.0)))
    resolution_m = float(real_mapping.get("resolution_m", mapping.get("online_resolution_m", 0.05)))
    if voxel_enabled:
        # nvblox and the VoxRoom query grid must address the same cubic voxels.
        voxel_cfg["z_resolution_m"] = resolution_m
    depth_min_m = float(real_mapping.get("depth_min_m", mapping.get("depth_min_m", 0.2)))
    depth_max_m = float(real_mapping.get("depth_max_m", mapping.get("depth_max_m", 3.0)))

    return OnlineMapper(
        size_m=size_m,
        resolution_m=resolution_m,
        depth_max_m=depth_max_m,
        depth_min_m=depth_min_m,
        depth_stride_px=depth_stride,
        obstacle_min_height_m=float(mapping.get("obstacle_min_height_m", 0.20)),
        obstacle_max_height_m=float(mapping.get("obstacle_max_height_m", 0.90)),
        free_min_height_m=float(mapping.get("free_min_height_m", -1.50)),
        free_max_height_m=float(mapping.get("free_max_height_m", 0.10)),
        vertical_profile_free_min_height_m=float(vertical_cfg.get("z_min_m", 0.20)),
        vertical_profile_free_max_height_m=float(vertical_cfg.get("z_max_m", 2.00)),
        splat_point_threshold=int(mapping.get("splat_point_threshold", 6)),
        free_splat_point_threshold=int(mapping.get("free_splat_point_threshold", 1)),
        robot_radius_m=robot_radius_m,
        inflation_radius_m=float(mapping.get("inflation_radius_m", 0.0)),
        height_profile_enabled=bool(height_cfg.get("enabled", True)),
        height_profile_z_min_m=float(height_cfg.get("z_min_m", 0.10)),
        height_profile_z_max_m=float(height_cfg.get("z_max_m", height_cfg.get("storage_z_max_m", 3.20))),
        height_profile_storage_z_max_m=float(height_cfg.get("storage_z_max_m", height_cfg.get("z_max_m", 3.20))),
        height_profile_active_z_max_fallback_m=float(
            height_cfg.get("active_z_max_fallback_m", height_cfg.get("active_z_max_cap_m", 2.80))
        ),
        height_profile_active_z_max_ceiling_ratio=float(height_cfg.get("active_z_max_ceiling_ratio", 0.85)),
        height_profile_active_z_max_cap_m=float(height_cfg.get("active_z_max_cap_m", 2.80)),
        height_profile_z_bin_size_m=float(height_cfg.get("z_bin_size_m", 0.05)),
        ceiling_height_estimator_config=dict(height_cfg.get("ceiling_estimator", {}) or {}),
        voxel_grid_enabled=voxel_enabled,
        voxel_grid_z_min_m=float(voxel_cfg.get("z_min_m", -0.10)),
        voxel_grid_z_max_m=float(voxel_cfg.get("z_max_m", 4.00)),
        voxel_grid_z_resolution_m=float(voxel_cfg.get("z_resolution_m", 0.05)),
        voxel_grid_active_z_min_m=float(voxel_cfg.get("active_z_min_m", 0.10)),
        voxel_grid_active_z_max_fallback_m=float(
            voxel_cfg.get("active_z_max_fallback_m", voxel_cfg.get("active_z_max_cap_m", 2.80))
        ),
        voxel_grid_active_z_max_ceiling_ratio=float(voxel_cfg.get("active_z_max_ceiling_ratio", 0.85)),
        voxel_grid_active_z_max_cap_m=float(voxel_cfg.get("active_z_max_cap_m", 2.80)),
        voxel_grid_config=voxel_cfg,
        voxel_outside_config=outside_cfg,
        voxel_navigation_projection_config=nav_cfg,
        voxel_navigation_blind_zone_config=blind_cfg,
        voxel_runtime_config=voxel_runtime_cfg,
        allow_disabled_voxel_grid_for_sensor_smoke_test=not voxel_enabled,
    )


def prepare_roomseg_config(
    base_cfg: dict,
    *,
    map_info: Any,
    repo_root: Path,
) -> tuple[VoxelOccupancyDoorWallRoomSegConfig, list[str]]:
    mapping = dict(base_cfg.get("mapping", {}) or {})
    room_cfg = copy.deepcopy(dict(mapping.get("room_segmentation", {}) or {}))
    warnings: list[str] = []
    learning = dict(room_cfg.get("door_seed_learning", {}) or {})
    if str(learning.get("mode", "disabled")).strip().lower() == "inference":
        checkpoint_raw = str(learning.get("checkpoint_path", "") or "").strip()
        checkpoint = Path(checkpoint_raw).expanduser()
        if checkpoint_raw and not checkpoint.is_absolute():
            checkpoint = repo_root / checkpoint
        if not checkpoint_raw or not checkpoint.is_file():
            message = (
                "door-seed inference checkpoint is missing; refusing to run a different segmentation policy. "
                f"Configured checkpoint: {checkpoint if checkpoint_raw else '<empty>'}"
            )
            raise FileNotFoundError(message)
        else:
            learning["checkpoint_path"] = str(checkpoint.resolve())
    room_cfg["door_seed_learning"] = learning
    cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
        room_cfg,
        resolution_m=float(map_info.resolution_m),
        map_info=map_info,
    )
    return cfg, warnings


def build_room_segmenter(
    base_cfg: dict,
    *,
    mapper: OnlineMapper,
    repo_root: Path,
) -> tuple[VoxelOccupancyDoorWallRoomSegmenter, list[str]]:
    cfg, warnings = prepare_roomseg_config(
        base_cfg,
        map_info=mapper.grid.map_info,
        repo_root=repo_root,
    )
    return VoxelOccupancyDoorWallRoomSegmenter(cfg, map_info=mapper.grid.map_info), warnings


def update_room_segmentation(
    segmenter: VoxelOccupancyDoorWallRoomSegmenter,
    mapper: OnlineMapper,
    *,
    step: int,
) -> tuple[np.ndarray | None, dict[str, Any], int, str]:
    grid = mapper.grid
    shape = grid.occupied.shape
    unknown_fallback = ~np.asarray(grid.observed, dtype=bool)
    nav_free, nav_obstacle, nav_unknown, nav_source = resolve_replay_style_navigation_masks(
        mapper=mapper,
        shape=shape,
        fallback_free=np.asarray(grid.free, dtype=bool),
        fallback_obstacle=np.asarray(grid.occupied, dtype=bool),
        fallback_unknown=unknown_fallback,
    )
    rooms = segmenter.update(
        nav_obstacle,
        nav_free,
        nav_obstacle,
        nav_unknown,
        step=int(step),
        voxel_grid=getattr(mapper, "voxel_grid", None),
        object_memory=[],
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=nav_obstacle,
        door_seed_no_clearance_free_mask=nav_free,
    )
    result = segmenter.last_result
    labels = None if result is None else np.asarray(result.room_label_map, dtype=np.int32).copy()
    return labels, dict(segmenter.last_debug or {}), int(len(rooms)), nav_source


def ensure_full_voxel_dependencies() -> None:
    errors: list[str] = []
    try:
        import torch

        if not torch.cuda.is_available():
            errors.append("torch is installed but CUDA is not available")
    except Exception as exc:
        errors.append(f"torch import failed: {exc}")
    try:
        import nvblox_torch  # noqa: F401
        from nvblox_torch.mapper import Mapper, QueryType  # noqa: F401
        from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams  # noqa: F401
        from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType  # noqa: F401
        from nvblox_torch.sensor import Sensor  # noqa: F401
    except Exception as exc:
        errors.append(f"nvblox_torch API import failed: {exc}")
    if errors:
        raise RuntimeError(
            "full_voxel mode cannot start:\n  - "
            + "\n  - ".join(errors)
            + "\nRun smoke_2d first, then install a CUDA-compatible PyTorch and the official nvblox_torch package."
        )


def save_runtime_artifacts(
    output_dir: Path,
    *,
    mapper: OnlineMapper,
    step: int,
    base_pose: tuple[float, float, float, float],
    camera_pose: tuple[float, float, float, float],
    intrinsics: Any,
    room_labels: np.ndarray | None,
    room_debug: dict[str, Any] | None,
    latest_depth_m: np.ndarray | None,
    prefix: str,
    include_voxel_state: bool = False,
    base_transform_world: np.ndarray | None = None,
    camera_transform_world: np.ndarray | None = None,
    floor_z_m: float | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = mapper.grid
    map_path = output_dir / f"{prefix}_map.png"
    room_path = output_dir / f"{prefix}_rooms.png"
    depth_path = output_dir / f"{prefix}_depth.png"
    npz_path = output_dir / f"{prefix}_snapshot.npz"
    json_path = output_dir / f"{prefix}_debug.json"

    map_rgb = render_occupancy_map(grid.free, grid.occupied, grid.observed, grid.map_info, base_pose)
    Image.fromarray(map_rgb).save(map_path)
    if room_labels is not None:
        room_rgb = render_room_labels(room_labels, grid.occupied, grid.observed, grid.map_info, base_pose)
        Image.fromarray(room_rgb).save(room_path)
    if latest_depth_m is not None:
        Image.fromarray(render_depth(latest_depth_m)).save(depth_path)

    arrays: dict[str, Any] = {
        "occupancy_map": np.asarray(grid.occupied, dtype=np.uint8),
        "observed_free_mask": np.asarray(grid.free, dtype=np.uint8),
        "observed_mask": np.asarray(grid.observed, dtype=np.uint8),
        "unknown_mask": (~np.asarray(grid.observed, dtype=bool)).astype(np.uint8),
        "room_label_map": (
            np.asarray(room_labels, dtype=np.int32)
            if room_labels is not None
            else np.zeros_like(grid.occupied, dtype=np.int32)
        ),
        "base_pose_world_xyzyaw": np.asarray(base_pose, dtype=np.float64),
        "camera_pose_world_xyzyaw": np.asarray(camera_pose, dtype=np.float64),
        "intrinsics_fx_fy_cx_cy": np.asarray(
            [float(intrinsics.fx), float(intrinsics.fy), float(intrinsics.cx), float(intrinsics.cy)],
            dtype=np.float64,
        ),
        "intrinsics_width_height": np.asarray([int(intrinsics.width), int(intrinsics.height)], dtype=np.int32),
        "map_resolution_m": np.asarray(float(grid.map_info.resolution_m), dtype=np.float64),
        "map_bounds_xyxy_m": np.asarray(
            [grid.map_info.min_x, grid.map_info.min_y, grid.map_info.max_x, grid.map_info.max_y],
            dtype=np.float64,
        ),
        "step": np.asarray(int(step), dtype=np.int64),
    }
    if base_transform_world is not None:
        arrays["base_transform_world"] = np.asarray(base_transform_world, dtype=np.float64)
    if camera_transform_world is not None:
        arrays["camera_transform_world"] = np.asarray(camera_transform_world, dtype=np.float64)
    if floor_z_m is not None:
        arrays["floor_z_m"] = np.asarray(float(floor_z_m), dtype=np.float64)
    if include_voxel_state and getattr(mapper, "voxel_grid", None) is not None:
        arrays["voxel_state"] = np.asarray(mapper.voxel_grid.state, dtype=np.uint8)
        arrays["voxel_z_centers_m"] = np.asarray(mapper.voxel_grid.z_centers_m, dtype=np.float32)
    np.savez_compressed(npz_path, **arrays)

    debug_payload = {
        "step": int(step),
        "pose_mode": "se3" if camera_transform_world is not None else "xyzyaw",
        "floor_z_m": None if floor_z_m is None else float(floor_z_m),
        "base_transform_world": None if base_transform_world is None else _jsonable(np.asarray(base_transform_world).tolist()),
        "camera_transform_world": None
        if camera_transform_world is None
        else _jsonable(np.asarray(camera_transform_world).tolist()),
        "mapper": _jsonable(dict(getattr(mapper, "last_debug_stats", {}) or {})),
        "timing": _jsonable(dict(getattr(mapper, "last_timing_stats", {}) or {})),
        "roomseg": _jsonable(dict(room_debug or {})),
    }
    json_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out = {"map": str(map_path), "snapshot": str(npz_path), "debug": str(json_path)}
    if room_labels is not None:
        out["rooms"] = str(room_path)
    if latest_depth_m is not None:
        out["depth"] = str(depth_path)
    return out


def render_occupancy_map(free: np.ndarray, occupied: np.ndarray, observed: np.ndarray, map_info: Any, base_pose: tuple[float, float, float, float]) -> np.ndarray:
    free_b = np.asarray(free, dtype=bool)
    occ_b = np.asarray(occupied, dtype=bool)
    observed_b = np.asarray(observed, dtype=bool)
    rgb = np.full((*free_b.shape, 3), 105, dtype=np.uint8)
    rgb[observed_b] = (190, 190, 190)
    rgb[free_b & ~occ_b] = (245, 245, 245)
    rgb[occ_b] = (15, 15, 15)
    _draw_robot(rgb, map_info, base_pose)
    return np.flipud(rgb)


def render_room_labels(labels: np.ndarray, occupied: np.ndarray, observed: np.ndarray, map_info: Any, base_pose: tuple[float, float, float, float]) -> np.ndarray:
    labels_i = np.asarray(labels, dtype=np.int32)
    observed_b = np.asarray(observed, dtype=bool)
    rgb = np.full((*labels_i.shape, 3), 90, dtype=np.uint8)
    rgb[observed_b] = (205, 205, 205)
    for label in np.unique(labels_i):
        if label <= 0:
            continue
        rgb[labels_i == label] = _label_color(int(label))
    rgb[np.asarray(occupied, dtype=bool)] = (10, 10, 10)
    _draw_robot(rgb, map_info, base_pose)
    return np.flipud(rgb)


def render_depth(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    image = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return image
    lo, hi = np.percentile(depth[valid], [2.0, 98.0])
    hi = max(float(hi), float(lo) + 1.0e-3)
    scaled = 1.0 - np.clip((depth - float(lo)) / (hi - float(lo)), 0.0, 1.0)
    image[valid] = np.asarray(np.round(scaled[valid] * 255.0), dtype=np.uint8)
    return image


def _draw_robot(rgb: np.ndarray, map_info: Any, pose: tuple[float, float, float, float]) -> None:
    x, y, _, yaw = pose
    col = int(math.floor((float(x) - float(map_info.min_x)) / float(map_info.resolution_m)))
    row = int(math.floor((float(map_info.max_y) - float(y)) / float(map_info.resolution_m)))
    if not (0 <= row < rgb.shape[0] and 0 <= col < rgb.shape[1]):
        return
    tip_x = float(x) + 0.35 * math.cos(float(yaw))
    tip_y = float(y) + 0.35 * math.sin(float(yaw))
    tip_col = int(math.floor((tip_x - float(map_info.min_x)) / float(map_info.resolution_m)))
    tip_row = int(math.floor((float(map_info.max_y) - tip_y) / float(map_info.resolution_m)))
    for alpha in np.linspace(0.0, 1.0, 20):
        rr = int(round(row + alpha * (tip_row - row)))
        cc = int(round(col + alpha * (tip_col - col)))
        if 0 <= rr < rgb.shape[0] and 0 <= cc < rgb.shape[1]:
            rgb[rr, cc] = (0, 77, 64)
    outer_radius = 4
    inner_radius = 2
    rgb[
        max(0, row - outer_radius) : min(rgb.shape[0], row + outer_radius + 1),
        max(0, col - outer_radius) : min(rgb.shape[1], col + outer_radius + 1),
    ] = (0, 77, 64)
    rgb[
        max(0, row - inner_radius) : min(rgb.shape[0], row + inner_radius + 1),
        max(0, col - inner_radius) : min(rgb.shape[1], col + inner_radius + 1),
    ] = (0, 188, 212)


def _label_color(label: int) -> tuple[int, int, int]:
    label = int(label)
    if label < 1:
        raise ValueError("Room labels must be positive")
    color = COMPARISON_ROOM_COLORS[
        (label - 1) % len(COMPARISON_ROOM_COLORS)
    ]
    return tuple(int(channel) for channel in color)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
