from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from voxroom_online.isaac_runtime.baselines.mask_io import (
    save_baseline_snapshot_npz,
)
from voxroom_online.isaac_runtime.debug.roomseg_layer_dump import (
    ROOMSEG_SNAPSHOT_ARRAY_KEYS,
    save_roomseg_layer_dump,
)
from voxroom_online.isaac_runtime.evaluation.roomseg_coverage_milestones import (
    CoverageEvent,
    CoverageMilestoneTracker,
    FULL_VOXEL_SNAPSHOT_KEYS,
    door_segments_rc_to_line_mask,
    partition_free_space_by_door_lines,
    strict_voxel_snapshot_arrays,
)


REQUIRED_FRAME_KEYS = {
    "step",
    "depth_m",
    "intrinsics_fx_fy_cx_cy",
    "intrinsics_width_height",
    "base_pose_world_xyzyaw",
    "camera_transform_world",
}

RAW_DOOR_SEED_COLOR = (220, 50, 45)
ACCEPTED_DOOR_SEED_COLOR = (142, 36, 196)
MODEL_REJECTED_DOOR_SEED_COLOR = (184, 16, 72)
VERTICAL_FREE_COLOR = (250, 250, 247)
STRICT_WALL_COLOR = (239, 48, 45)
VERTICAL_UNKNOWN_COLOR = (232, 232, 232)
ROBOT_POSE_COLOR = (0, 188, 212)
ROBOT_POSE_OUTLINE_COLOR = (0, 77, 64)
TVARS_ACCEPTED_DOOR_LINE_COLOR = (216, 27, 96)
TVARS_DOOR_SEGMENTS_COORDINATE_FRAME = "active_room_native_gt_map_rc_v1"
ACTIVE_ROOM_TO_VOXROOM_GRID_TRANSFORM = (
    "active_room_initial_heading_grid_to_voxroom_world_grid_yaw_affine_v1"
)


def _active_room_to_voxroom_affine_xy(
    shape: Sequence[int],
    initial_yaw_rad: float,
) -> np.ndarray:
    target_shape = tuple(int(value) for value in shape)
    if len(target_shape) != 2 or min(target_shape) < 1:
        raise ValueError(f"invalid Active Room map shape: {target_shape}")
    yaw = float(initial_yaw_rad)
    if not math.isfinite(yaw):
        raise ValueError("Active Room initial yaw must be finite")

    height, width = target_shape
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    # Active Room columns point along the episode's initial heading and rows
    # point left. VoxRoom columns point along world +x and rows along world -y.
    linear = np.asarray(
        [[cosine, -sine], [-sine, -cosine]],
        dtype=np.float64,
    )
    pivot_xy = np.asarray(
        [(float(width) - 1.0) * 0.5, (float(height) - 1.0) * 0.5],
        dtype=np.float64,
    )
    translation = pivot_xy - linear @ pivot_xy
    return np.column_stack((linear, translation))


def _project_active_room_mask_to_voxroom_native(
    mask: np.ndarray,
    *,
    initial_yaw_rad: float,
) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError(f"Active Room mask must be 2D, got {source.shape}")
    height, width = source.shape
    affine_xy = _active_room_to_voxroom_affine_xy(source.shape, initial_yaw_rad)
    projected = cv2.warpAffine(
        source.astype(np.uint8),
        affine_xy,
        (int(width), int(height)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.asarray(projected, dtype=bool)


def _project_active_room_segments_to_voxroom_native(
    segments_rc: np.ndarray,
    *,
    shape: Sequence[int],
    initial_yaw_rad: float,
) -> np.ndarray:
    target_shape = tuple(int(value) for value in shape)
    if len(target_shape) != 2 or min(target_shape) < 1:
        raise ValueError(f"invalid Active Room map shape: {target_shape}")
    segments = np.asarray(segments_rc, dtype=np.float64).reshape(-1, 4)
    if not np.all(np.isfinite(segments)):
        raise ValueError("TVARS door segment contains non-finite coordinates")
    if not np.allclose(segments, np.rint(segments), atol=0.0, rtol=0.0):
        raise ValueError("TVARS door segment coordinates must be integer grid cells")
    if not segments.size:
        return np.empty((0, 4), dtype=np.int32)

    height, width = target_shape
    rows = segments[:, (0, 2)]
    columns = segments[:, (1, 3)]
    if (
        np.any(rows < 0)
        or np.any(rows >= height)
        or np.any(columns < 0)
        or np.any(columns >= width)
    ):
        raise ValueError("TVARS accepted door segment lies outside the Active Room map")

    affine_xy = _active_room_to_voxroom_affine_xy(target_shape, initial_yaw_rad)
    points_xy = np.stack(
        (segments[:, (1, 3)], segments[:, (0, 2)]),
        axis=-1,
    ).reshape(-1, 2)
    projected_xy = (
        points_xy @ affine_xy[:, :2].T + affine_xy[:, 2][None, :]
    )
    projected_xy = np.rint(projected_xy).astype(np.int32).reshape(-1, 2, 2)
    projected = np.column_stack(
        (
            projected_xy[:, 0, 1],
            projected_xy[:, 0, 0],
            projected_xy[:, 1, 1],
            projected_xy[:, 1, 0],
        )
    ).astype(np.int32, copy=False)
    projected_rows = projected[:, (0, 2)]
    projected_columns = projected[:, (1, 3)]
    if (
        np.any(projected_rows < 0)
        or np.any(projected_rows >= height)
        or np.any(projected_columns < 0)
        or np.any(projected_columns >= width)
    ):
        raise RuntimeError("TVARS door projection lies outside the VoxRoom map")
    return projected


def _apply_ceiling_estimator_scene_override(
    base_cfg: Mapping[str, Any],
    scene_id: str,
) -> tuple[dict[str, Any], bool]:
    cfg = copy.deepcopy(dict(base_cfg))
    mapping_cfg = dict(cfg.get("mapping", {}) or {})
    room_cfg = dict(mapping_cfg.get("room_segmentation", {}) or {})
    height_cfg = dict(room_cfg.get("height_profile", {}) or {})
    estimator_cfg = dict(height_cfg.get("ceiling_estimator", {}) or {})
    overrides_raw = estimator_cfg.pop("scene_overrides", {})
    if not isinstance(overrides_raw, Mapping):
        raise TypeError("ceiling_estimator.scene_overrides must be a mapping")
    override = overrides_raw.get(str(scene_id))
    applied = override is not None
    if applied:
        if not isinstance(override, Mapping):
            raise TypeError(f"ceiling estimator override for scene {scene_id!r} must be a mapping")
        estimator_cfg.update(dict(override))
    height_cfg["ceiling_estimator"] = estimator_cfg
    room_cfg["height_profile"] = height_cfg
    mapping_cfg["room_segmentation"] = room_cfg
    cfg["mapping"] = mapping_cfg
    return cfg, applied


def load_habitat_frame(path: str | Path) -> dict[str, Any]:
    frame_path = Path(path).expanduser().resolve()
    if not frame_path.is_file():
        raise FileNotFoundError(f"Habitat bridge frame does not exist: {frame_path}")
    with np.load(frame_path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_FRAME_KEYS.difference(data.files))
        if missing:
            raise KeyError(f"Habitat bridge frame is missing keys: {missing}")
        step = int(np.asarray(data["step"]).reshape(()))
        depth_m = np.asarray(data["depth_m"], dtype=np.float32)
        intrinsics = np.asarray(data["intrinsics_fx_fy_cx_cy"], dtype=np.float64).reshape(-1)
        size = np.asarray(data["intrinsics_width_height"], dtype=np.int64).reshape(-1)
        base_pose = np.asarray(data["base_pose_world_xyzyaw"], dtype=np.float64).reshape(-1)
        camera_transform = np.asarray(data["camera_transform_world"], dtype=np.float64)
        coverage_keys = {
            "habitat_map_shape_hw",
            "habitat_explorable_bits",
            "habitat_explored_bits",
            "habitat_occupied_bits",
            "habitat_map_resolution_m",
            "habitat_floor_reference_source",
            "habitat_floor_height_m",
            "habitat_floor_slice_eps_m",
            "habitat_floor_island_index",
            "habitat_floor_source_cells",
            "habitat_floor_transformed_cells",
            "tvars_door_segments_rc",
            "tvars_door_segments_coordinate_frame",
        }
        present_coverage_keys = coverage_keys.intersection(data.files)
        coverage_payload: dict[str, Any] | None = None
        if present_coverage_keys:
            missing_coverage_keys = sorted(coverage_keys.difference(data.files))
            if missing_coverage_keys:
                raise KeyError(
                    "Habitat coverage frame is incomplete; missing keys: "
                    f"{missing_coverage_keys}"
                )
            habitat_shape = tuple(
                int(value)
                for value in np.asarray(
                    data["habitat_map_shape_hw"], dtype=np.int64
                ).reshape(-1)
            )
            if len(habitat_shape) != 2 or min(habitat_shape) < 1:
                raise ValueError(f"invalid Habitat map shape: {habitat_shape}")
            cell_count = int(habitat_shape[0] * habitat_shape[1])

            def unpack_mask(key: str) -> np.ndarray:
                bits = np.asarray(data[key], dtype=np.uint8).reshape(-1)
                expected_bytes = (cell_count + 7) // 8
                if bits.size != expected_bytes:
                    raise ValueError(
                        f"{key} has {bits.size} bytes, expected {expected_bytes}"
                    )
                return np.unpackbits(
                    bits,
                    bitorder="little",
                    count=cell_count,
                ).astype(bool, copy=False).reshape(habitat_shape)

            coverage_payload = {
                "shape": habitat_shape,
                "explorable": unpack_mask("habitat_explorable_bits"),
                "explored": unpack_mask("habitat_explored_bits"),
                "occupied": unpack_mask("habitat_occupied_bits"),
                "resolution_m": float(
                    np.asarray(data["habitat_map_resolution_m"]).reshape(())
                ),
                "reference_source": str(
                    np.asarray(data["habitat_floor_reference_source"]).reshape(())
                ),
                "floor_height_m": float(
                    np.asarray(data["habitat_floor_height_m"]).reshape(())
                ),
                "floor_slice_eps_m": float(
                    np.asarray(data["habitat_floor_slice_eps_m"]).reshape(())
                ),
                "floor_island_index": int(
                    np.asarray(data["habitat_floor_island_index"]).reshape(())
                ),
                "floor_source_cells": int(
                    np.asarray(data["habitat_floor_source_cells"]).reshape(())
                ),
                "floor_transformed_cells": int(
                    np.asarray(data["habitat_floor_transformed_cells"]).reshape(())
                ),
                "door_segments_rc": np.asarray(
                    data["tvars_door_segments_rc"], dtype=np.int32
                ).reshape(-1, 4),
                "door_segments_coordinate_frame": str(
                    np.asarray(
                        data["tvars_door_segments_coordinate_frame"]
                    ).reshape(())
                ),
            }

    if step < 0:
        raise ValueError(f"Habitat bridge step must be non-negative, got {step}")
    if depth_m.ndim != 2 or depth_m.size == 0:
        raise ValueError(f"Habitat bridge depth must be a non-empty HxW array, got {depth_m.shape}")
    if intrinsics.size != 4 or size.size != 2:
        raise ValueError("Habitat bridge intrinsics must contain fx, fy, cx, cy, width, and height")
    width, height = int(size[0]), int(size[1])
    if (height, width) != depth_m.shape:
        raise ValueError(
            "Habitat bridge depth/intrinsics shape mismatch: "
            f"depth={depth_m.shape}, intrinsics={(height, width)}"
        )
    if base_pose.size != 4:
        raise ValueError("Habitat bridge base pose must use the (x, y, z, yaw) contract")
    if not np.all(np.isfinite(intrinsics)) or intrinsics[0] <= 0.0 or intrinsics[1] <= 0.0:
        raise ValueError("Habitat bridge intrinsics are invalid")
    if camera_transform.shape != (4, 4):
        raise ValueError("Habitat bridge camera transform must be a 4x4 SE(3) matrix")
    if not np.all(np.isfinite(base_pose)) or not np.all(np.isfinite(camera_transform)):
        raise ValueError("Habitat bridge poses contain non-finite values")
    if not np.allclose(camera_transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6, rtol=0.0):
        raise ValueError("Habitat bridge camera transform has an invalid homogeneous bottom row")
    rotation = camera_transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4, rtol=0.0):
        raise ValueError("Habitat bridge camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-4, rtol=0.0):
        raise ValueError("Habitat bridge camera rotation determinant is not +1")
    finite_or_range = np.isfinite(depth_m) | np.isposinf(depth_m)
    if not bool(np.all(finite_or_range)):
        raise ValueError("Habitat bridge depth contains NaN or negative infinity")
    finite_depth = depth_m[np.isfinite(depth_m)]
    if finite_depth.size and float(np.min(finite_depth)) < 0.0:
        raise ValueError("Habitat bridge depth contains negative values")
    if coverage_payload is not None:
        if (
            not np.isfinite(coverage_payload["resolution_m"])
            or coverage_payload["resolution_m"] <= 0.0
        ):
            raise ValueError("Habitat map resolution is invalid")
        if not np.any(coverage_payload["explorable"]):
            raise ValueError("Habitat fixed explorable map is empty")
        if (
            coverage_payload["door_segments_coordinate_frame"]
            != TVARS_DOOR_SEGMENTS_COORDINATE_FRAME
        ):
            raise RuntimeError(
                "Habitat TVARS door segments do not use the native gt_map grid"
            )

    return {
        "step": step,
        "depth_m": depth_m,
        "intrinsics": tuple(float(v) for v in intrinsics),
        "width": width,
        "height": height,
        "base_pose": tuple(float(v) for v in base_pose),
        "camera_transform": camera_transform,
        "frame_path": frame_path,
        "coverage": coverage_payload,
    }


class HabitatVoxRoomRuntime:
    def __init__(
        self,
        *,
        repo_root: Path,
        config_path: Path,
        output_dir: Path,
        map_size_m: float,
        roomseg_every_steps: int,
        visualization_every_steps: int,
        scene_id: str,
        episode_id: str,
        coverage_eval: bool = False,
        coverage_milestones: str = "20,40,60,70,80,90",
    ) -> None:
        from voxroom_online.real_runtime.mapper_runtime import (
            build_mapper,
            ensure_full_voxel_dependencies,
            load_yaml,
        )

        if roomseg_every_steps < 1:
            raise ValueError("roomseg_every_steps must be positive")
        if visualization_every_steps < 1:
            raise ValueError("visualization_every_steps must be positive")
        if map_size_m <= 0.0:
            raise ValueError("map_size_m must be positive")

        self.repo_root = repo_root.resolve()
        self.config_path = config_path.resolve()
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.frames_dir = self.output_dir / "visualization_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_dir / "progress.jsonl"
        self.navigation_path = self.output_dir / "navigation_projection_latest.npz"
        self.scene_id = str(scene_id)
        self.episode_id = str(episode_id)
        self.coverage_eval = bool(coverage_eval)
        self.coverage_milestones = str(coverage_milestones)
        self.roomseg_every_steps = int(roomseg_every_steps)
        self.visualization_every_steps = int(visualization_every_steps)
        self.base_cfg, self.ceiling_scene_override_applied = (
            _apply_ceiling_estimator_scene_override(
                load_yaml(self.config_path),
                self.scene_id,
            )
        )
        effective_ceiling_cfg = dict(
            self.base_cfg["mapping"]["room_segmentation"]["height_profile"][
                "ceiling_estimator"
            ]
        )
        self.ceiling_estimator_source = str(effective_ceiling_cfg.get("source", ""))
        self.config_sha256 = _sha256(self.config_path)
        ensure_full_voxel_dependencies()
        self.mapper = build_mapper(
            self.base_cfg,
            {
                "mapping": {
                    "map_size_m": float(map_size_m),
                    "resolution_m": float(
                        dict(self.base_cfg.get("mapping", {}) or {}).get(
                            "online_resolution_m",
                            0.05,
                        )
                    ),
                }
            },
            "full_voxel",
        )
        self.segmenter: Any | None = None
        self.last_step = -1
        self.last_roomseg_step = -1
        self.last_room_labels: np.ndarray | None = None
        self.last_room_debug: dict[str, Any] = {}
        self.last_room_count = 0
        self.last_raw_door_seed_count = 0
        self.last_accepted_door_seed_count = 0
        self.last_nav_source = ""
        self.last_intrinsics: Any | None = None
        self.last_base_pose: tuple[float, float, float, float] | None = None
        self.last_camera_transform: np.ndarray | None = None
        self.last_depth_m: np.ndarray | None = None
        self.last_visualization_path: Path | None = None
        self.started_at = time.time()
        self.mapper_total_ms = 0.0
        self.roomseg_total_ms = 0.0
        self.roomseg_updates = 0
        self.coverage_tracker: CoverageMilestoneTracker | None = None
        self.coverage_reference_native: np.ndarray | None = None
        self.last_coverage_payload: dict[str, Any] | None = None
        self.active_room_initial_yaw_rad: float | None = None

    def update(self, frame_path: str | Path) -> dict[str, Any]:
        from voxroom_online.isaac_runtime.sensors.camera_geometry import CameraIntrinsics
        from voxroom_online.real_runtime.mapper_runtime import build_room_segmenter

        frame = load_habitat_frame(frame_path)
        step = int(frame["step"])
        expected_step = self.last_step + 1
        if step != expected_step:
            raise RuntimeError(
                f"Habitat bridge step discontinuity: expected {expected_step}, received {step}"
            )
        intr = CameraIntrinsics(
            width=int(frame["width"]),
            height=int(frame["height"]),
            fx=float(frame["intrinsics"][0]),
            fy=float(frame["intrinsics"][1]),
            cx=float(frame["intrinsics"][2]),
            cy=float(frame["intrinsics"][3]),
        )
        base_pose = frame["base_pose"]
        camera_transform = frame["camera_transform"]
        if self.last_step < 0:
            self.active_room_initial_yaw_rad = float(base_pose[3])
            if not math.isfinite(self.active_room_initial_yaw_rad):
                raise RuntimeError("Habitat bridge initial yaw is not finite")
            self.mapper.reset((float(base_pose[0]), float(base_pose[1])))
            self.segmenter, warnings = build_room_segmenter(
                self.base_cfg,
                mapper=self.mapper,
                repo_root=self.repo_root,
            )
            if warnings:
                raise RuntimeError(f"VoxRoom room segmenter emitted warnings: {warnings}")
        if self.coverage_eval:
            coverage_payload = frame.get("coverage")
            if not isinstance(coverage_payload, dict):
                raise RuntimeError(
                    "coverage evaluation requires Habitat explorable, explored, occupied, and door-line data"
                )
            self._accept_coverage_payload(coverage_payload)

        mapper_started = time.perf_counter()
        self.mapper.update(frame["depth_m"], intr, base_pose, camera_transform)
        mapper_ms = (time.perf_counter() - mapper_started) * 1000.0
        self.mapper_total_ms += mapper_ms
        self.last_step = step
        self.last_intrinsics = intr
        self.last_base_pose = base_pose
        self.last_camera_transform = np.asarray(camera_transform, dtype=np.float64).copy()
        self.last_depth_m = np.asarray(frame["depth_m"], dtype=np.float32).copy()
        self._save_navigation_projection(step)

        roomseg_updated = False
        coverage_event_ids: list[str] = []
        if self.coverage_eval:
            if self.coverage_tracker is None or self.last_coverage_payload is None:
                raise RuntimeError("Habitat coverage tracker was not initialized")
            explored_native = np.asarray(
                self.last_coverage_payload["explored_native"], dtype=bool
            ).copy()
            events = self.coverage_tracker.observe(
                step=step,
                explored_mask=explored_native,
            )
            for event in events:
                self._save_coverage_event(event)
            coverage_event_ids = [event.event_id for event in events]
            roomseg_updated = bool(events)
        elif step > 0 and step % self.roomseg_every_steps == 0:
            self._update_roomseg()
            roomseg_updated = True
        if (
            self.last_visualization_path is None
            or step % self.visualization_every_steps == 0
            or roomseg_updated
        ):
            self.last_visualization_path = self._save_visualization(step)

        progress = {
            "step": step,
            "frame_name": frame["frame_path"].name,
            "mapper_ms": mapper_ms,
            "roomseg_updated": roomseg_updated,
            "roomseg_step": self.last_roomseg_step,
            "room_count": self.last_room_count,
            "raw_door_seed_count": self.last_raw_door_seed_count,
            "accepted_door_seed_count": self.last_accepted_door_seed_count,
            "visualization_path": str(self.last_visualization_path),
            "navigation_projection_path": str(self.navigation_path),
            "coverage_event_ids": coverage_event_ids,
            "mapper_timing": _scalar_mapping(
                dict(getattr(self.mapper, "last_timing_stats", {}) or {})
            ),
        }
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(progress, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return {
            "status": "updated",
            "step": step,
            "mapper_ms": mapper_ms,
            "roomseg_updated": roomseg_updated,
            "room_count": self.last_room_count,
            "raw_door_seed_count": self.last_raw_door_seed_count,
            "accepted_door_seed_count": self.last_accepted_door_seed_count,
            "visualization_path": str(self.last_visualization_path),
            "navigation_projection_path": str(self.navigation_path),
            "coverage_event_ids": coverage_event_ids,
        }

    def _accept_coverage_payload(self, payload: Mapping[str, Any]) -> None:
        shape = tuple(int(value) for value in payload["shape"])
        mapper_shape = tuple(int(value) for value in self.mapper.grid.occupied.shape)
        if shape != mapper_shape:
            raise RuntimeError(
                f"Habitat coverage map shape {shape} differs from VoxRoom map {mapper_shape}"
            )
        resolution_m = float(payload["resolution_m"])
        mapper_resolution_m = float(self.mapper.grid.map_info.resolution_m)
        if not np.isclose(resolution_m, mapper_resolution_m, atol=1.0e-9, rtol=0.0):
            raise RuntimeError(
                "Habitat and VoxRoom coverage maps must use identical resolution: "
                f"{resolution_m} != {mapper_resolution_m}"
            )
        explorable_active = np.asarray(payload["explorable"], dtype=bool)
        explored_active = np.asarray(payload["explored"], dtype=bool)
        occupied_active = np.asarray(payload["occupied"], dtype=bool)
        if (
            explorable_active.shape != shape
            or explored_active.shape != shape
            or occupied_active.shape != shape
        ):
            raise ValueError("Habitat coverage masks do not share the declared shape")
        segments_active = np.asarray(
            payload["door_segments_rc"], dtype=np.int32
        ).reshape(-1, 4)
        if segments_active.size:
            rows = segments_active[:, (0, 2)]
            columns = segments_active[:, (1, 3)]
            if (
                np.any(rows < 0)
                or np.any(rows >= shape[0])
                or np.any(columns < 0)
                or np.any(columns >= shape[1])
            ):
                raise ValueError("TVARS accepted door segment lies outside the Habitat map")
        if self.active_room_initial_yaw_rad is None:
            raise RuntimeError("Habitat coverage projection has no episode initial yaw")
        initial_yaw_rad = float(self.active_room_initial_yaw_rad)
        affine_xy = _active_room_to_voxroom_affine_xy(shape, initial_yaw_rad)
        explorable_native = _project_active_room_mask_to_voxroom_native(
            explorable_active,
            initial_yaw_rad=initial_yaw_rad,
        )
        explored_native = _project_active_room_mask_to_voxroom_native(
            explored_active,
            initial_yaw_rad=initial_yaw_rad,
        )
        occupied_native = _project_active_room_mask_to_voxroom_native(
            occupied_active,
            initial_yaw_rad=initial_yaw_rad,
        )
        segments_native = _project_active_room_segments_to_voxroom_native(
            segments_active,
            shape=shape,
            initial_yaw_rad=initial_yaw_rad,
        )
        reference_native = explorable_native.copy()
        if payload["reference_source"] != "habitat_pathfinder_start_floor_island_topdown":
            raise RuntimeError("Habitat coverage did not use the PathFinder floor reference")
        if not np.isclose(
            float(payload["floor_slice_eps_m"]), 0.5, atol=1.0e-9, rtol=0.0
        ):
            raise RuntimeError("Habitat coverage used a different floor slice epsilon")
        active_transformed_cells = int(np.count_nonzero(explorable_active))
        if int(payload["floor_transformed_cells"]) != active_transformed_cells:
            raise RuntimeError("Habitat transformed floor cell count changed in transit")
        source_cells = int(payload["floor_source_cells"])
        if source_cells <= 0:
            raise RuntimeError("Habitat PathFinder floor source is empty")
        active_area_error = abs(active_transformed_cells - source_cells) / float(
            source_cells
        )
        if active_area_error > 0.01:
            raise RuntimeError("Habitat floor reference alignment changed area by over one percent")
        native_transformed_cells = int(np.count_nonzero(explorable_native))
        yaw_projection_area_error = abs(
            native_transformed_cells - active_transformed_cells
        ) / float(active_transformed_cells)
        if yaw_projection_area_error > 0.01:
            raise RuntimeError(
                "Active Room to VoxRoom yaw projection changed floor area by over one percent"
            )
        if self.coverage_tracker is None:
            self.coverage_reference_native = reference_native
            map_info = self.mapper.grid.map_info
            self.coverage_tracker = CoverageMilestoneTracker(
                output_dir=self.output_dir / "roomseg_coverage_eval",
                simulator="habitat",
                reference_mask=reference_native,
                resolution_m=mapper_resolution_m,
                milestones=self.coverage_milestones,
                reference_arrays={
                    "habitat_explorable_mask": explorable_native.astype(np.uint8),
                    "habitat_explorable_mask_active_room_grid": explorable_active.astype(
                        np.uint8
                    ),
                    "active_room_initial_yaw_rad": np.asarray(
                        initial_yaw_rad, dtype=np.float64
                    ),
                    "active_room_to_voxroom_affine_xy": affine_xy.astype(np.float64),
                    "runtime_map_bounds_xyxy_m": np.asarray(
                        [
                            map_info.min_x,
                            map_info.min_y,
                            map_info.max_x,
                            map_info.max_y,
                        ],
                        dtype=np.float64,
                    ),
                },
                metadata={
                    "scene_id": self.scene_id,
                    "episode_id": self.episode_id,
                    "reference_source": payload["reference_source"],
                    "reference_floor_height_m": float(payload["floor_height_m"]),
                    "reference_floor_slice_eps_m": float(
                        payload["floor_slice_eps_m"]
                    ),
                    "reference_floor_island_index": int(
                        payload["floor_island_index"]
                    ),
                    "reference_floor_source_cells": source_cells,
                    "reference_floor_active_room_cells": active_transformed_cells,
                    "reference_floor_voxroom_native_cells": native_transformed_cells,
                    "reference_floor_transform_area_error": active_area_error,
                    "reference_floor_yaw_projection_area_error": yaw_projection_area_error,
                    "reference_floor_grid_transform": "nearest_neighbor",
                    "coordinate_frame": "voxroom_native_grid",
                    "habitat_map_transform": ACTIVE_ROOM_TO_VOXROOM_GRID_TRANSFORM,
                    "active_room_initial_yaw_rad": initial_yaw_rad,
                    "active_room_initial_yaw_deg": math.degrees(initial_yaw_rad),
                    "comparison_method": "tvars_original_accepted_door_line_partition",
                },
            )
        elif not np.array_equal(reference_native, self.coverage_reference_native):
            raise RuntimeError("Habitat fixed explorable reference changed during the episode")
        self.last_coverage_payload = {
            "shape": shape,
            "resolution_m": resolution_m,
            "explorable_active": explorable_active.copy(),
            "explored_active": explored_active.copy(),
            "occupied_active": occupied_active.copy(),
            "door_segments_rc_active": segments_active.copy(),
            "explorable_native": explorable_native,
            "explored_native": explored_native,
            "occupied_native": occupied_native,
            "door_segments_rc_native": segments_native,
            "door_segments_coordinate_frame": str(
                payload["door_segments_coordinate_frame"]
            ),
            "active_room_initial_yaw_rad": initial_yaw_rad,
            "active_room_to_voxroom_affine_xy": affine_xy,
            "reference_source": str(payload["reference_source"]),
            "floor_height_m": float(payload["floor_height_m"]),
            "floor_slice_eps_m": float(payload["floor_slice_eps_m"]),
            "floor_island_index": int(payload["floor_island_index"]),
            "floor_source_cells": source_cells,
            "floor_active_room_cells": active_transformed_cells,
            "floor_voxroom_native_cells": native_transformed_cells,
            "floor_yaw_projection_area_error": yaw_projection_area_error,
        }

    def _save_coverage_event(self, event: CoverageEvent) -> None:
        if self.coverage_tracker is None or self.last_coverage_payload is None:
            raise RuntimeError("Habitat coverage event has no fixed reference or current map")
        try:
            self._update_roomseg()
            masks = self._navigation_masks()
            payload = self.last_coverage_payload
            habitat_explored_native = np.asarray(
                payload["explored_native"], dtype=bool
            )
            habitat_occupied_native = np.asarray(
                payload["occupied_native"], dtype=bool
            )
            habitat_free_native = (
                habitat_explored_native & ~habitat_occupied_native
            )
            door_line_native = door_segments_rc_to_line_mask(
                np.asarray(payload["door_segments_rc_native"], dtype=np.int32),
                shape=payload["shape"],
                thickness_cells=3,
            )
            tvars_labels, partition_debug = partition_free_space_by_door_lines(
                habitat_free_native,
                door_line_native,
            )
            stem = f"roomseg_{event.event_id}_step_{int(event.step):06d}"
            map_info = self.mapper.grid.map_info
            extra_arrays = {
                **strict_voxel_snapshot_arrays(self.mapper),
                **self.coverage_tracker.event_arrays(
                    event,
                    explored_mask=habitat_explored_native,
                ),
                "step": np.asarray(int(event.step), dtype=np.int64),
                "map_resolution_m": np.asarray(
                    float(map_info.resolution_m), dtype=np.float64
                ),
                "map_bounds_xyxy_m": np.asarray(
                    [map_info.min_x, map_info.min_y, map_info.max_x, map_info.max_y],
                    dtype=np.float64,
                ),
                "habitat_explorable_mask_canonical": np.asarray(
                    payload["explorable_native"], dtype=np.uint8
                ),
                "habitat_explored_mask_canonical": habitat_explored_native.astype(
                    np.uint8
                ),
                "habitat_occupied_mask_canonical": habitat_occupied_native.astype(
                    np.uint8
                ),
                "habitat_explorable_mask_active_room_grid": np.asarray(
                    payload["explorable_active"], dtype=np.uint8
                ),
                "habitat_explored_mask_active_room_grid": np.asarray(
                    payload["explored_active"], dtype=np.uint8
                ),
                "habitat_occupied_mask_active_room_grid": np.asarray(
                    payload["occupied_active"], dtype=np.uint8
                ),
                "habitat_observed_free_mask_native": habitat_free_native.astype(np.uint8),
                "tvars_original_door_segments_rc_canonical": np.asarray(
                    payload["door_segments_rc_native"], dtype=np.int32
                ),
                "tvars_original_door_segments_rc_active_room_grid": np.asarray(
                    payload["door_segments_rc_active"], dtype=np.int32
                ),
                "tvars_original_door_segments_coordinate_frame": np.asarray(
                    payload["door_segments_coordinate_frame"]
                ),
                "tvars_original_door_line_map": door_line_native.astype(np.uint8),
                "tvars_original_door_partition_label_map": tvars_labels.astype(
                    np.int32
                ),
                "base_pose_world_xyzyaw": np.asarray(
                    self.last_base_pose, dtype=np.float64
                ),
                "camera_transform_world": np.asarray(
                    self.last_camera_transform, dtype=np.float64
                ),
                "roomseg_eval_simulator": np.asarray("habitat"),
                "roomseg_eval_coordinate_frame": np.asarray("voxroom_native_grid"),
                "roomseg_eval_habitat_transform": np.asarray(
                    ACTIVE_ROOM_TO_VOXROOM_GRID_TRANSFORM
                ),
                "active_room_initial_yaw_rad": np.asarray(
                    payload["active_room_initial_yaw_rad"], dtype=np.float64
                ),
                "active_room_to_voxroom_affine_xy": np.asarray(
                    payload["active_room_to_voxroom_affine_xy"], dtype=np.float64
                ),
            }
            room_debug = {
                **dict(self.last_room_debug),
                "final_room_label_map": np.asarray(
                    self.last_room_labels, dtype=np.int32
                ),
            }
            source_dump = save_roomseg_layer_dump(
                out_dir=event.event_dir / "voxroom" / "roomseg_snapshots",
                step=int(event.step),
                room_debug=room_debug,
                occupancy_map=masks["occupied"],
                observed_free_mask=masks["free"],
                obstacle_mask=masks["occupied"],
                unknown_mask=masks["unknown"],
                frontier_map=np.zeros_like(masks["free"], dtype=bool),
                selected_frontier_members=None,
                selected_frontier_center_rc=None,
                agent_rc=_world_pose_to_grid_rc(
                    self.last_base_pose,
                    map_info,
                    masks["free"].shape,
                ),
                max_saves=8,
                save_npz=True,
                save_png=False,
                save_summary_json=True,
                save_overlay_png=False,
                save_layers_png=False,
                save_navigation_room_masks_png=True,
                npz_keys=ROOMSEG_SNAPSHOT_ARRAY_KEYS,
                extra_npz_arrays=extra_arrays,
                include_selected_frontier_sector=False,
                filename_stem=stem,
            )
            source_paths = dict(source_dump["paths"])
            source_npz = Path(source_paths["npz"])
            baseline_npz = (
                event.event_dir
                / "tvars_original"
                / "roomseg_snapshots"
                / source_npz.name
            )
            baseline_metadata = {
                **event.metadata(),
                "method": "tvars_original_habitat_door_line_partition",
                "label_source": "accepted_door_lines_partition_current_observed_free",
                "coordinate_frame": "voxroom_native_grid",
                "habitat_transform": ACTIVE_ROOM_TO_VOXROOM_GRID_TRANSFORM,
                "active_room_initial_yaw_rad": float(
                    payload["active_room_initial_yaw_rad"]
                ),
                "door_segments_coordinate_frame": str(
                    payload["door_segments_coordinate_frame"]
                ),
                "active_room_initial_yaw_deg": math.degrees(
                    float(payload["active_room_initial_yaw_rad"])
                ),
                "active_room_to_voxroom_affine_xy": np.asarray(
                    payload["active_room_to_voxroom_affine_xy"], dtype=np.float64
                ).tolist(),
                "floor_yaw_projection_area_error": float(
                    payload["floor_yaw_projection_area_error"]
                ),
                "partition": partition_debug,
                "source_snapshot": str(source_npz),
                "shared_voxel_snapshot": str(source_npz),
            }
            save_baseline_snapshot_npz(
                source_npz_path=source_npz,
                output_npz_path=baseline_npz,
                baseline_label_map=tvars_labels,
                baseline_name="tvars_original_habitat",
                metadata=baseline_metadata,
                debug_arrays={
                    "tvars_original_door_line_map": door_line_native,
                    "tvars_original_door_partition_label_map": tvars_labels,
                    "tvars_original_habitat_observed_free_mask": habitat_free_native,
                },
                excluded_source_keys=FULL_VOXEL_SNAPSHOT_KEYS,
            )
            baseline_preview = baseline_npz.with_suffix(".png")
            self._save_baseline_preview(baseline_npz, baseline_preview)
            baseline_summary = baseline_npz.with_suffix(".summary.json")
            _write_json_atomic(
                baseline_summary,
                {
                    **baseline_metadata,
                    "output_npz": str(baseline_npz),
                    "colored_room_preview": str(baseline_preview),
                    "room_count": int(np.max(tvars_labels)),
                },
            )
            self.coverage_tracker.complete(
                event,
                artifacts={
                    "voxroom_snapshot_npz": source_npz,
                    "voxroom_summary_json": source_paths["summary_json"],
                    "voxroom_navigation_png": source_paths[
                        "navigation_room_masks_png"
                    ],
                    "tvars_original_snapshot_npz": baseline_npz,
                    "tvars_original_summary_json": baseline_summary,
                    "tvars_original_preview_png": baseline_preview,
                },
            )
        except BaseException as exc:
            self.coverage_tracker.fail(event, exc)
            raise

    def _save_baseline_preview(self, npz_path: Path, output_path: Path) -> None:
        from voxroom_online.real_runtime.mapper_runtime import render_room_labels

        if self.last_base_pose is None:
            raise RuntimeError("cannot render TVARS coverage result without a robot pose")
        with np.load(npz_path, allow_pickle=False) as data:
            labels = np.asarray(data["final_room_label_map"], dtype=np.int32)
            occupied = np.asarray(data["obstacle_mask"], dtype=bool)
            observed = ~np.asarray(data["unknown_mask"], dtype=bool)
            door_line_native = np.asarray(
                data["tvars_original_door_line_map"], dtype=bool
            )
        if door_line_native.shape != labels.shape:
            raise RuntimeError(
                "TVARS accepted door-line map shape differs from its room labels"
            )
        image = np.asarray(
            render_room_labels(
                labels,
                occupied,
                observed,
                self.mapper.grid.map_info,
                self.last_base_pose,
            ),
            dtype=np.uint8,
        ).copy()
        aligned_door_line = np.flipud(door_line_native)
        robot_overlay = np.all(image == ROBOT_POSE_COLOR, axis=2) | np.all(
            image == ROBOT_POSE_OUTLINE_COLOR, axis=2
        )
        image[aligned_door_line & ~robot_overlay] = TVARS_ACCEPTED_DOOR_LINE_COLOR
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output_path)

    def close(
        self,
        *,
        tvars_door_segments_rc: Sequence[Sequence[int]] | None = None,
        tvars_door_segments_coordinate_frame: str | None = None,
    ) -> dict[str, Any]:
        from voxroom_online.real_runtime.mapper_runtime import save_runtime_artifacts

        if self.last_step < 0:
            raise RuntimeError("Cannot close a VoxRoom Habitat bridge before receiving a frame")
        if self.coverage_eval:
            if self.coverage_tracker is None or self.last_coverage_payload is None:
                raise RuntimeError("Habitat coverage evaluation has no terminal map state")
            if (
                tvars_door_segments_coordinate_frame
                != TVARS_DOOR_SEGMENTS_COORDINATE_FRAME
            ):
                raise RuntimeError(
                    "terminal TVARS door segments do not use the native gt_map grid"
                )
            terminal_segments = np.asarray(
                tvars_door_segments_rc or [], dtype=np.int32
            ).reshape(-1, 4)
            shape = tuple(int(value) for value in self.last_coverage_payload["shape"])
            if terminal_segments.size:
                rows = terminal_segments[:, (0, 2)]
                columns = terminal_segments[:, (1, 3)]
                if (
                    np.any(rows < 0)
                    or np.any(rows >= shape[0])
                    or np.any(columns < 0)
                    or np.any(columns >= shape[1])
                ):
                    raise ValueError("terminal TVARS door segment lies outside the Habitat map")
            if self.active_room_initial_yaw_rad is None:
                raise RuntimeError("terminal TVARS projection has no episode initial yaw")
            self.last_coverage_payload["door_segments_rc_active"] = terminal_segments
            self.last_coverage_payload["door_segments_rc_native"] = (
                _project_active_room_segments_to_voxroom_native(
                    terminal_segments,
                    shape=shape,
                    initial_yaw_rad=self.active_room_initial_yaw_rad,
                )
            )
            terminal_event = self.coverage_tracker.finalize(
                step=self.last_step,
                explored_mask=np.asarray(
                    self.last_coverage_payload["explored_native"], dtype=bool
                ),
            )
            self._save_coverage_event(terminal_event)
        elif self.last_roomseg_step != self.last_step:
            self._update_roomseg()
        self.last_visualization_path = self._save_visualization(self.last_step, final=True)
        if (
            self.last_intrinsics is None
            or self.last_base_pose is None
            or self.last_camera_transform is None
            or self.last_depth_m is None
        ):
            raise RuntimeError("VoxRoom Habitat bridge lost its terminal frame state")
        engine = getattr(self.segmenter, "door_seed_inference_engine", None)
        if engine is None:
            raise RuntimeError("VoxRoom learned door-seed inference engine is not active")
        if getattr(engine, "load_error", None) is not None:
            raise RuntimeError(f"VoxRoom learned door-seed model load failed: {engine.load_error}")
        fallback_count = int(getattr(engine, "fallback_count", 0))
        if fallback_count != 0:
            raise RuntimeError(f"VoxRoom learned door-seed fallback count is {fallback_count}")
        checkpoint_path = Path(engine.config.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"VoxRoom learned door-seed checkpoint is missing: {checkpoint_path}")
        camera_pose_xyzyaw = _camera_pose_xyzyaw(self.last_camera_transform)
        artifacts = save_runtime_artifacts(
            self.output_dir,
            mapper=self.mapper,
            step=self.last_step,
            base_pose=self.last_base_pose,
            camera_pose=camera_pose_xyzyaw,
            intrinsics=self.last_intrinsics,
            room_labels=self.last_room_labels,
            room_debug=self.last_room_debug,
            latest_depth_m=self.last_depth_m,
            prefix=f"final_step_{self.last_step:06d}",
            include_voxel_state=True,
            camera_transform_world=self.last_camera_transform,
            floor_z_m=float(self.last_base_pose[2]),
        )
        result = {
            "status": "completed",
            "source": "habitat_gibson_rgbd_ground_truth_pose",
            "mapper": "voxroom_nvblox_fast_dda",
            "room_segmenter": "voxroom_voxel_door_seed_network",
            "policy_control": "active_room_segmentation",
            "first_step": 0,
            "last_step": self.last_step,
            "processed_frames": self.last_step + 1,
            "roomseg_every_steps": self.roomseg_every_steps,
            "roomseg_updates": self.roomseg_updates,
            "roomseg_coverage_eval": self.coverage_eval,
            "active_room_grid_transform": (
                ACTIVE_ROOM_TO_VOXROOM_GRID_TRANSFORM if self.coverage_eval else None
            ),
            "tvars_door_segments_coordinate_frame": (
                TVARS_DOOR_SEGMENTS_COORDINATE_FRAME if self.coverage_eval else None
            ),
            "active_room_initial_yaw_rad": (
                None
                if self.active_room_initial_yaw_rad is None
                else float(self.active_room_initial_yaw_rad)
            ),
            "roomseg_coverage_manifest": (
                None
                if self.coverage_tracker is None
                else str(self.coverage_tracker.manifest_path)
            ),
            "room_count": self.last_room_count,
            "raw_door_seed_count": self.last_raw_door_seed_count,
            "accepted_door_seed_count": self.last_accepted_door_seed_count,
            "navigation_mask_source": self.last_nav_source,
            "navigation_projection_path": str(self.navigation_path),
            "door_seed_checkpoint_path": str(checkpoint_path),
            "door_seed_checkpoint_sha256": _sha256(checkpoint_path),
            "door_seed_model_inference_count": int(getattr(engine, "inference_count", 0)),
            "door_seed_model_fallback_count": fallback_count,
            "mapper_total_ms": self.mapper_total_ms,
            "mapper_mean_ms": self.mapper_total_ms / float(self.last_step + 1),
            "roomseg_total_ms": self.roomseg_total_ms,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "ceiling_estimator_source": self.ceiling_estimator_source,
            "ceiling_scene_override_applied": self.ceiling_scene_override_applied,
            **_source_identity(self.repo_root),
            "visualization_path": str(self.last_visualization_path),
            "artifacts": artifacts,
            "finished_at_unix": time.time(),
            "elapsed_seconds": time.time() - self.started_at,
        }
        result_path = self.output_dir / "result.json"
        _write_json_atomic(result_path, result)
        return {**result, "result_path": str(result_path)}

    def _update_roomseg(self) -> None:
        if self.segmenter is None:
            raise RuntimeError("VoxRoom room segmenter is not initialized")
        masks = self._navigation_masks()
        started = time.perf_counter()
        rooms = self.segmenter.update(
            masks["occupied"],
            masks["free"],
            masks["occupied"],
            masks["unknown"],
            step=self.last_step,
            voxel_grid=self.mapper.voxel_grid,
            object_memory=[],
            navigation_free_mask=masks["free"],
            navigation_obstacle_mask=masks["occupied"],
            door_seed_no_clearance_free_mask=masks["free"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = self.segmenter.last_result
        labels = None if result is None else np.asarray(result.room_label_map, dtype=np.int32).copy()
        if labels is None:
            raise RuntimeError("VoxRoom room segmenter returned no label map")
        self.last_room_labels = labels
        self.last_room_debug = dict(self.segmenter.last_debug or {})
        raw_seed_mask, accepted_seed_mask = _door_seed_masks(
            self.last_room_debug,
            labels.shape,
        )
        self.last_raw_door_seed_count = int(np.count_nonzero(raw_seed_mask))
        self.last_accepted_door_seed_count = int(
            np.count_nonzero(accepted_seed_mask)
        )
        for key in (
            "strict_fallback_used",
            "silent_fallback_used",
            "voxel_door_seed_model_fallback",
        ):
            if bool(self.last_room_debug.get(key, False)):
                raise RuntimeError(f"VoxRoom room segmentation reported {key}=true")
        self.last_room_count = int(len(rooms))
        self.last_nav_source = "mapper.last_voxel_navigation_projection"
        self.last_roomseg_step = self.last_step
        self.roomseg_updates += 1
        self.roomseg_total_ms += elapsed_ms

    def _navigation_masks(self) -> dict[str, np.ndarray]:
        projection = getattr(self.mapper, "last_voxel_navigation_projection", None)
        if projection is None:
            raise RuntimeError("VoxRoom mapper did not produce a voxel navigation projection")
        shape = self.mapper.grid.occupied.shape
        masks: dict[str, np.ndarray] = {}
        for name in ("free", "occupied", "observed", "unknown"):
            mask = np.asarray(getattr(projection, name, None), dtype=bool)
            if mask.shape != shape:
                raise RuntimeError(
                    f"VoxRoom voxel navigation projection {name} has shape {mask.shape}, expected {shape}"
                )
            masks[name] = mask.copy()
        if np.any(masks["free"] & masks["occupied"]):
            raise RuntimeError("VoxRoom voxel navigation projection marks cells both free and occupied")
        if np.any(masks["unknown"] & (masks["free"] | masks["occupied"])):
            raise RuntimeError("VoxRoom voxel navigation projection unknown mask overlaps known cells")
        if np.any((masks["free"] | masks["occupied"]) & ~masks["observed"]):
            raise RuntimeError("VoxRoom voxel navigation projection has known cells outside observed")
        if not np.array_equal(masks["unknown"], ~masks["observed"]):
            raise RuntimeError("VoxRoom voxel navigation unknown mask differs from inverse observed")
        return masks

    def _save_navigation_projection(self, step: int) -> None:
        masks = self._navigation_masks()
        shape = np.asarray(masks["free"].shape, dtype=np.int32)
        map_info = self.mapper.grid.map_info
        temporary = self.navigation_path.with_name(
            f".{self.navigation_path.name}.{os.getpid()}.tmp"
        )
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                format_version=np.asarray(1, dtype=np.int32),
                step=np.asarray(int(step), dtype=np.int64),
                shape_hw=shape,
                free_bits=np.packbits(masks["free"].reshape(-1), bitorder="little"),
                occupied_bits=np.packbits(
                    masks["occupied"].reshape(-1),
                    bitorder="little",
                ),
                observed_bits=np.packbits(
                    masks["observed"].reshape(-1),
                    bitorder="little",
                ),
                unknown_bits=np.packbits(
                    masks["unknown"].reshape(-1),
                    bitorder="little",
                ),
                resolution_m=np.asarray(
                    float(map_info.resolution_m),
                    dtype=np.float64,
                ),
                bounds_xyxy_m=np.asarray(
                    [
                        float(map_info.min_x),
                        float(map_info.min_y),
                        float(map_info.max_x),
                        float(map_info.max_y),
                    ],
                    dtype=np.float64,
                ),
                source=np.asarray(
                    "mapper.last_voxel_navigation_projection",
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.navigation_path)

    def _save_visualization(self, step: int, *, final: bool = False) -> Path:
        from voxroom_online.real_runtime.mapper_runtime import (
            render_occupancy_map,
            render_room_labels,
        )

        if self.last_base_pose is None:
            raise RuntimeError("Cannot render VoxRoom state without a base pose")
        grid = self.mapper.grid
        occupancy = render_occupancy_map(
            grid.free,
            grid.occupied,
            grid.observed,
            grid.map_info,
            self.last_base_pose,
        )
        if self.last_room_labels is None:
            room_image = np.full_like(occupancy, 238, dtype=np.uint8)
        else:
            room_image = render_room_labels(
                self.last_room_labels,
                grid.occupied,
                grid.observed,
                grid.map_info,
                self.last_base_pose,
            )
        vertical_free_image = _render_vertical_free_map(
            self.last_room_debug,
            grid.occupied.shape,
        )
        visible = _render_visible_mask(grid.observed, self.last_room_labels)
        occupancy = np.asarray(occupancy, dtype=np.uint8).copy()
        room_image = np.asarray(room_image, dtype=np.uint8).copy()
        vertical_free_image = np.asarray(vertical_free_image, dtype=np.uint8).copy()
        occupancy[~visible] = (255, 255, 255)
        room_image[~visible] = (255, 255, 255)
        vertical_free_image[~visible] = (255, 255, 255)
        occupancy, room_image, vertical_free_image, visible = (
            _align_visualization_triplet(
                occupancy,
                room_image,
                vertical_free_image,
                visible,
            )
        )
        raw_seed_mask, accepted_seed_mask = _door_seed_masks(
            self.last_room_debug,
            grid.occupied.shape,
        )
        occupancy = _overlay_door_seed_masks(
            occupancy,
            raw_seed_mask,
            accepted_seed_mask,
            show_accepted=False,
        )
        room_image = _overlay_door_seed_masks(
            room_image,
            raw_seed_mask,
            accepted_seed_mask,
            show_accepted=True,
        )
        occupancy = _overlay_robot_pose(
            occupancy,
            grid.map_info,
            self.last_base_pose,
            grid.occupied.shape,
        )
        room_image = _overlay_robot_pose(
            room_image,
            grid.map_info,
            self.last_base_pose,
            grid.occupied.shape,
        )
        vertical_free_image = _overlay_robot_pose(
            vertical_free_image,
            grid.map_info,
            self.last_base_pose,
            grid.occupied.shape,
        )
        occupancy, room_image, vertical_free_image = _crop_active_map_triplet(
            occupancy,
            room_image,
            vertical_free_image,
            visible,
        )
        header_height = 58
        panel = Image.new(
            "RGB",
            (occupancy.shape[1] * 3, occupancy.shape[0] + header_height),
            "white",
        )
        panel.paste(Image.fromarray(occupancy), (0, header_height))
        panel.paste(
            Image.fromarray(room_image),
            (occupancy.shape[1], header_height),
        )
        panel.paste(
            Image.fromarray(vertical_free_image),
            (occupancy.shape[1] * 2, header_height),
        )
        draw = ImageDraw.Draw(panel)
        draw.text((10, 9), f"VoxRoom nvblox occupancy | step {step}", fill=(20, 20, 20))
        draw.text(
            (occupancy.shape[1] + 10, 9),
            f"VoxRoom room mask | rooms {self.last_room_count}",
            fill=(20, 20, 20),
        )
        draw.rectangle((10, 33, 25, 48), fill=RAW_DOOR_SEED_COLOR)
        draw.text(
            (31, 34),
            f"raw door seed {self.last_raw_door_seed_count}",
            fill=(20, 20, 20),
        )
        robot_legend_x = min(occupancy.shape[1] - 100, 175)
        draw.ellipse(
            (
                robot_legend_x,
                33,
                robot_legend_x + 15,
                48,
            ),
            fill=ROBOT_POSE_COLOR,
            outline=ROBOT_POSE_OUTLINE_COLOR,
            width=2,
        )
        draw.text(
            (robot_legend_x + 21, 34),
            "robot pose",
            fill=(20, 20, 20),
        )
        accepted_legend_x = occupancy.shape[1] + 10
        draw.rectangle(
            (accepted_legend_x, 33, accepted_legend_x + 15, 48),
            fill=ACCEPTED_DOOR_SEED_COLOR,
        )
        draw.text(
            (accepted_legend_x + 21, 34),
            f"accepted door seed {self.last_accepted_door_seed_count}",
            fill=(20, 20, 20),
        )
        vertical_panel_x = occupancy.shape[1] * 2
        draw.text(
            (vertical_panel_x + 10, 9),
            "VoxRoom Vertical Free Map",
            fill=(20, 20, 20),
        )
        vertical_legends = (
            (10, VERTICAL_FREE_COLOR, "free"),
            (85, STRICT_WALL_COLOR, "wall"),
            (160, ACCEPTED_DOOR_SEED_COLOR, "kept"),
            (250, MODEL_REJECTED_DOOR_SEED_COLOR, "rejected"),
        )
        for offset_x, color, label in vertical_legends:
            legend_x = vertical_panel_x + offset_x
            draw.rectangle(
                (legend_x, 33, legend_x + 15, 48),
                fill=color,
                outline=(90, 90, 90),
                width=1,
            )
            draw.text(
                (legend_x + 21, 34),
                label,
                fill=(20, 20, 20),
            )
        frame_path = self.frames_dir / f"frame_{step:06d}.png"
        temporary = frame_path.with_name(f".{frame_path.name}.{os.getpid()}.tmp")
        panel.save(temporary, format="PNG")
        os.replace(temporary, frame_path)
        latest_path = self.output_dir / "latest_visualization.png"
        latest_temporary = latest_path.with_name(f".{latest_path.name}.{os.getpid()}.tmp")
        panel.save(latest_temporary, format="PNG")
        os.replace(latest_temporary, latest_path)
        if final:
            final_path = self.output_dir / "visualization_final.png"
            final_temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
            panel.save(final_temporary, format="PNG")
            os.replace(final_temporary, final_path)
            return final_path
        return latest_path


def _write_protocol_response(stream: Any, payload: Mapping[str, Any]) -> None:
    stream.write(json.dumps(_jsonable(dict(payload)), ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(repo_root: Path) -> dict[str, Any]:
    git_dir = repo_root / ".git"
    tree_sha256, tree_file_count = _source_tree_digest(repo_root)
    if git_dir.exists():
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            timeout=30,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
                text=True,
                timeout=30,
            ).strip()
        )
        return {
            "voxroom_source_commit": commit,
            "voxroom_source_tree_dirty": dirty,
            "voxroom_source_tree_sha256": tree_sha256,
            "voxroom_source_tree_file_count": tree_file_count,
        }
    return {
        "voxroom_source_commit": None,
        "voxroom_source_tree_dirty": None,
        "voxroom_source_tree_sha256": tree_sha256,
        "voxroom_source_tree_file_count": tree_file_count,
    }


def _source_tree_digest(repo_root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for root in (
            repo_root / "voxroom_online",
            repo_root / "configs",
            repo_root / "scripts",
        )
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".yaml", ".yml", ".sh", ".toml"}
        and "__pycache__" not in path.parts
    )
    if not paths:
        raise RuntimeError(f"VoxRoom source tree has no hashable files: {repo_root}")
    for path in paths:
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest(), len(paths)


def _render_visible_mask(
    observed: np.ndarray,
    room_labels: np.ndarray | None,
) -> np.ndarray:
    visible = np.asarray(observed, dtype=bool).copy()
    if room_labels is not None:
        labels = np.asarray(room_labels)
        if labels.shape != visible.shape:
            raise ValueError(
                f"VoxRoom room labels have shape {labels.shape}, expected {visible.shape}"
            )
        visible |= labels > 0
    return np.flipud(visible)


def _align_visualization_pair(
    occupancy: np.ndarray,
    room_image: np.ndarray,
    visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupancy_array = np.asarray(occupancy)
    room_array = np.asarray(room_image)
    visible_array = np.asarray(visible, dtype=bool)
    if (
        occupancy_array.ndim != 3
        or occupancy_array.shape[-1] != 3
        or room_array.shape != occupancy_array.shape
        or visible_array.shape != occupancy_array.shape[:2]
    ):
        raise ValueError("VoxRoom visualization maps must share one HxW geometry")
    return (
        np.transpose(occupancy_array, (1, 0, 2)).copy(),
        np.transpose(room_array, (1, 0, 2)).copy(),
        visible_array.T.copy(),
    )


def _align_visualization_triplet(
    occupancy: np.ndarray,
    room_image: np.ndarray,
    vertical_free_image: np.ndarray,
    visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aligned_occupancy, aligned_room, aligned_visible = _align_visualization_pair(
        occupancy,
        room_image,
        visible,
    )
    vertical_array = np.asarray(vertical_free_image)
    if vertical_array.shape != np.asarray(occupancy).shape:
        raise ValueError("VoxRoom vertical-free map must share the visualization geometry")
    return (
        aligned_occupancy,
        aligned_room,
        np.transpose(vertical_array, (1, 0, 2)).copy(),
        aligned_visible,
    )


def _door_seed_masks(
    room_debug: Mapping[str, Any],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = tuple(int(value) for value in shape)

    def read_mask(name: str) -> np.ndarray:
        value = room_debug.get(name)
        if value is None:
            return np.zeros(expected_shape, dtype=bool)
        mask = np.asarray(value, dtype=bool)
        if mask.shape != expected_shape:
            raise RuntimeError(
                f"VoxRoom {name} has shape {mask.shape}, expected {expected_shape}"
            )
        return mask.copy()

    raw = read_mask("voxel_door_raw_seed_mask")
    accepted = read_mask("voxel_door_seed_model_keep_mask")
    if np.any(accepted & ~raw):
        raise RuntimeError(
            "VoxRoom accepted door seeds are not a subset of raw door seeds"
        )
    return raw, accepted


def _debug_mask(
    room_debug: Mapping[str, Any],
    name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    expected_shape = tuple(int(value) for value in shape)
    value = room_debug.get(name)
    if value is None:
        return np.zeros(expected_shape, dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if mask.shape != expected_shape:
        raise RuntimeError(
            f"VoxRoom {name} has shape {mask.shape}, expected {expected_shape}"
        )
    return mask.copy()


def _render_vertical_free_map(
    room_debug: Mapping[str, Any],
    shape: tuple[int, int],
) -> np.ndarray:
    expected_shape = tuple(int(value) for value in shape)
    vertical_free = (
        _debug_mask(room_debug, "voxel_vertical_free_xy", expected_shape)
        | _debug_mask(room_debug, "vertical_free_room_domain", expected_shape)
    )
    strict_wall = (
        _debug_mask(room_debug, "voxel_wall_xy", expected_shape)
        | _debug_mask(room_debug, "voxel_wall_after_step1_map", expected_shape)
        | _debug_mask(room_debug, "structural_wall_clean", expected_shape)
    )
    raw_seed, accepted_seed = _door_seed_masks(room_debug, expected_shape)
    rejected_seed = _debug_mask(
        room_debug,
        "voxel_door_seed_model_reject_mask",
        expected_shape,
    )
    if np.any(rejected_seed & ~raw_seed):
        raise RuntimeError(
            "VoxRoom rejected door seeds are not a subset of raw door seeds"
        )
    if np.any(rejected_seed & accepted_seed):
        raise RuntimeError(
            "VoxRoom accepted and rejected door-seed masks overlap"
        )

    image = np.full(
        (*expected_shape, 3),
        VERTICAL_UNKNOWN_COLOR,
        dtype=np.uint8,
    )
    image[vertical_free] = VERTICAL_FREE_COLOR
    image[strict_wall] = STRICT_WALL_COLOR
    image[accepted_seed] = ACCEPTED_DOOR_SEED_COLOR
    image[rejected_seed] = MODEL_REJECTED_DOOR_SEED_COLOR
    return np.flipud(image)


def _overlay_door_seed_masks(
    image: np.ndarray,
    raw_seed_mask: np.ndarray,
    accepted_seed_mask: np.ndarray,
    *,
    show_accepted: bool,
) -> np.ndarray:
    output = np.asarray(image, dtype=np.uint8).copy()
    raw = np.asarray(raw_seed_mask, dtype=bool)
    accepted = np.asarray(accepted_seed_mask, dtype=bool)
    expected_shape = output.shape[:2][::-1]
    if output.ndim != 3 or output.shape[2] != 3:
        raise ValueError(f"VoxRoom door-seed overlay image must be HxWx3, got {output.shape}")
    if raw.shape != expected_shape or accepted.shape != expected_shape:
        raise ValueError(
            "VoxRoom door-seed masks must match the pre-alignment mapper grid"
        )
    if np.any(accepted & ~raw):
        raise ValueError("accepted door-seed mask must be a subset of the raw mask")
    aligned_raw = np.flipud(raw).T
    aligned_accepted = np.flipud(accepted).T
    output[aligned_raw] = RAW_DOOR_SEED_COLOR
    if show_accepted:
        output[aligned_accepted] = ACCEPTED_DOOR_SEED_COLOR
    return output


def _world_pose_to_grid_rc(
    base_pose: tuple[float, float, float, float] | None,
    map_info: Any,
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    if base_pose is None:
        raise RuntimeError("VoxRoom base pose is unavailable")
    row = int(
        np.floor(
            (float(map_info.max_y) - float(base_pose[1]))
            / float(map_info.resolution_m)
        )
    )
    column = int(
        np.floor(
            (float(base_pose[0]) - float(map_info.min_x))
            / float(map_info.resolution_m)
        )
    )
    if not (0 <= row < int(grid_shape[0]) and 0 <= column < int(grid_shape[1])):
        raise RuntimeError("VoxRoom robot pose is outside the mapper grid")
    return row, column


def _overlay_robot_pose(
    image: np.ndarray,
    map_info: Any,
    base_pose: tuple[float, float, float, float],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    output = np.asarray(image, dtype=np.uint8).copy()
    if output.ndim != 3 or output.shape[2] != 3:
        raise ValueError(f"VoxRoom robot overlay image must be HxWx3, got {output.shape}")
    grid_height, grid_width = (int(grid_shape[0]), int(grid_shape[1]))
    if output.shape[:2] != (grid_width, grid_height):
        raise ValueError("Aligned VoxRoom image shape differs from mapper grid")
    x, y, _, yaw = (float(value) for value in base_pose)

    def aligned_pixel(world_x: float, world_y: float) -> tuple[int, int]:
        grid_col = int(
            np.floor(
                (world_x - float(map_info.min_x))
                / float(map_info.resolution_m)
            )
        )
        grid_row = int(
            np.floor(
                (float(map_info.max_y) - world_y)
                / float(map_info.resolution_m)
            )
        )
        return grid_col, grid_height - 1 - grid_row

    row, column = aligned_pixel(x, y)
    if not (0 <= row < output.shape[0] and 0 <= column < output.shape[1]):
        raise RuntimeError("VoxRoom robot pose is outside the visualization map")
    tip_row, tip_column = aligned_pixel(
        x + 0.35 * np.cos(yaw),
        y + 0.35 * np.sin(yaw),
    )
    for alpha in np.linspace(0.0, 1.0, 20):
        rr = int(round(row + alpha * (tip_row - row)))
        cc = int(round(column + alpha * (tip_column - column)))
        if 0 <= rr < output.shape[0] and 0 <= cc < output.shape[1]:
            output[rr, cc] = ROBOT_POSE_OUTLINE_COLOR
    outer_radius = 4
    inner_radius = 2
    output[
        max(0, row - outer_radius) : min(output.shape[0], row + outer_radius + 1),
        max(0, column - outer_radius) : min(output.shape[1], column + outer_radius + 1),
    ] = ROBOT_POSE_OUTLINE_COLOR
    output[
        max(0, row - inner_radius) : min(output.shape[0], row + inner_radius + 1),
        max(0, column - inner_radius) : min(output.shape[1], column + inner_radius + 1),
    ] = ROBOT_POSE_COLOR
    return output


def _camera_pose_xyzyaw(camera_transform_world: np.ndarray) -> tuple[float, float, float, float]:
    transform = np.asarray(camera_transform_world, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"camera transform must be 4x4, got {transform.shape}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4, rtol=0.0):
        raise ValueError("camera transform rotation is not orthonormal")
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return (
        float(transform[0, 3]),
        float(transform[1, 3]),
        float(transform[2, 3]),
        yaw,
    )


def _crop_active_map_pair(
    occupancy: np.ndarray,
    room_image: np.ndarray,
    visible: np.ndarray,
    *,
    target_long_side_px: int = 900,
) -> tuple[np.ndarray, np.ndarray]:
    cropped = _crop_active_map_images(
        (occupancy, room_image),
        visible,
        target_long_side_px=target_long_side_px,
    )
    return cropped[0], cropped[1]


def _crop_active_map_triplet(
    occupancy: np.ndarray,
    room_image: np.ndarray,
    vertical_free_image: np.ndarray,
    visible: np.ndarray,
    *,
    target_long_side_px: int = 900,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cropped = _crop_active_map_images(
        (occupancy, room_image, vertical_free_image),
        visible,
        target_long_side_px=target_long_side_px,
    )
    return cropped[0], cropped[1], cropped[2]


def _crop_active_map_images(
    images: tuple[np.ndarray, ...],
    visible: np.ndarray,
    *,
    target_long_side_px: int,
) -> tuple[np.ndarray, ...]:
    if not images:
        raise ValueError("At least one VoxRoom visualization image is required")
    arrays = tuple(np.asarray(image, dtype=np.uint8) for image in images)
    visible = np.asarray(visible, dtype=bool)
    reference = arrays[0]
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError(
            f"VoxRoom visualization must be HxWx3, got {reference.shape}"
        )
    if (
        any(image.shape != reference.shape for image in arrays[1:])
        or visible.shape != reference.shape[:2]
    ):
        raise ValueError("VoxRoom visualization arrays must share one map shape")
    points = np.argwhere(visible)
    if points.size == 0:
        raise RuntimeError("VoxRoom visualization has no observed map cells")
    row_min, column_min = points.min(axis=0)
    row_max, column_max = points.max(axis=0)
    span = max(int(row_max - row_min + 1), int(column_max - column_min + 1))
    margin = max(8, int(round(span * 0.06)))
    height, width = visible.shape
    row_min = max(0, int(row_min) - margin)
    row_max = min(height, int(row_max) + margin + 1)
    column_min = max(0, int(column_min) - margin)
    column_max = min(width, int(column_max) + margin + 1)
    crops = tuple(
        image[row_min:row_max, column_min:column_max]
        for image in arrays
    )
    long_side = max(crops[0].shape[:2])
    scale = max(1, int(target_long_side_px) // max(1, int(long_side)))
    if scale == 1:
        return crops
    size = (crops[0].shape[1] * scale, crops[0].shape[0] * scale)
    return tuple(
        np.asarray(
            Image.fromarray(image).resize(size, Image.Resampling.NEAREST)
        )
        for image in crops
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _scalar_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        if item is None or isinstance(item, (str, bool, int, float, np.generic)):
            output[str(key)] = _jsonable(item)
        elif isinstance(item, Mapping):
            output[str(key)] = _scalar_mapping(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Habitat-to-VoxRoom nvblox bridge worker")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--map-size-m", type=float, default=48.0)
    parser.add_argument("--roomseg-every-steps", type=int, default=10)
    parser.add_argument("--visualization-every-steps", type=int, default=5)
    parser.add_argument("--scene-id", default=os.environ.get("VOXROOM_SCENE_ID", ""))
    parser.add_argument("--episode-id", default=os.environ.get("VOXROOM_EPISODE_ID", ""))
    parser.add_argument("--coverage-eval", action="store_true")
    parser.add_argument("--coverage-milestones", default="20,40,60,70,80,90")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_stdout = sys.stdout
    runtime: HabitatVoxRoomRuntime | None = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime = HabitatVoxRoomRuntime(
                repo_root=Path(args.repo_root),
                config_path=Path(args.config),
                output_dir=Path(args.output_dir),
                map_size_m=float(args.map_size_m),
                roomseg_every_steps=int(args.roomseg_every_steps),
                visualization_every_steps=int(args.visualization_every_steps),
                scene_id=str(args.scene_id),
                episode_id=str(args.episode_id),
                coverage_eval=bool(args.coverage_eval),
                coverage_milestones=str(args.coverage_milestones),
            )
        _write_protocol_response(protocol_stdout, {"status": "ready"})
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                raise RuntimeError("VoxRoom bridge received an empty protocol message")
            request = json.loads(line)
            operation = str(request.get("op", ""))
            with contextlib.redirect_stdout(sys.stderr):
                if operation == "update":
                    response = runtime.update(request["frame_path"])
                elif operation == "close":
                    response = runtime.close(
                        tvars_door_segments_rc=request.get(
                            "tvars_door_segments_rc"
                        ),
                        tvars_door_segments_coordinate_frame=request.get(
                            "tvars_door_segments_coordinate_frame"
                        ),
                    )
                else:
                    raise ValueError(f"Unsupported VoxRoom bridge operation: {operation!r}")
            _write_protocol_response(protocol_stdout, response)
            if operation == "close":
                return 0
        raise RuntimeError("VoxRoom bridge stdin closed before the close operation")
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        _write_protocol_response(
            protocol_stdout,
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
