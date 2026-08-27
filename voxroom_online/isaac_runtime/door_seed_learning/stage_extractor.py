from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Mapping

import numpy as np

from voxroom_online.isaac_runtime.door_seed_learning.schema import config_hash, map_info_hash
from voxroom_online.isaac_runtime.mapping.voxel_door_detector import (
    VoxelDoorDetectorConfig,
    VoxelDoorSeedResult,
    classify_voxel_door_seeds,
)
from voxroom_online.isaac_runtime.mapping.voxel_occupancy_grid import VoxelOccupancyGrid3D
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo
from voxroom_online.isaac_runtime.mapping.voxel_roomseg_evidence import (
    VoxelRoomsegEvidence,
    VoxelRoomsegEvidenceConfig,
    build_voxel_roomseg_evidence,
)


@dataclass
class DoorSeedStageResult:
    evidence: VoxelRoomsegEvidence
    raw_seed_result: VoxelDoorSeedResult
    raw_seed_mask_xy: np.ndarray
    vertical_class_map_xy: np.ndarray
    nav_class_map_xy: np.ndarray
    outside_boundary_mask_xy: np.ndarray
    map_shape: tuple[int, int]
    map_info_hash: str
    raw_seed_config_hash: str
    input_semantics_hash: str
    debug: dict[str, object]
    voxroom_raw_seed_mask_xy: np.ndarray | None = None
    tvars_vertical_raw_seed_mask_xy: np.ndarray | None = None
    voxroom_raw_seed_history_mask_xy: np.ndarray | None = None
    tvars_vertical_raw_seed_history_mask_xy: np.ndarray | None = None


def encode_vertical_class_map(
    vertical_free_xy: np.ndarray,
    strict_wall_xy: np.ndarray,
    *,
    outside_boundary_mask_xy: np.ndarray | None = None,
) -> np.ndarray:
    free = np.asarray(vertical_free_xy, dtype=bool)
    wall = np.asarray(strict_wall_xy, dtype=bool)
    if wall.shape != free.shape:
        raise ValueError("vertical free and strict wall maps must share one shape")
    result = np.zeros(free.shape, dtype=np.uint8)
    result[free] = 1
    result[wall & ~free] = 2
    if outside_boundary_mask_xy is not None:
        outside = np.asarray(outside_boundary_mask_xy, dtype=bool)
        if outside.shape != free.shape:
            raise ValueError("outside boundary map must match vertical class map")
        result[outside] = 0
    return result


def encode_nav_class_map(
    no_clearance_free_xy: np.ndarray,
    occupied_xy: np.ndarray,
    *,
    outside_boundary_mask_xy: np.ndarray | None = None,
) -> np.ndarray:
    free = np.asarray(no_clearance_free_xy, dtype=bool)
    occupied = np.asarray(occupied_xy, dtype=bool)
    if occupied.shape != free.shape:
        raise ValueError("navigation free and occupied maps must share one shape")
    result = np.zeros(free.shape, dtype=np.uint8)
    result[free] = 1
    result[occupied] = 2
    if outside_boundary_mask_xy is not None:
        outside = np.asarray(outside_boundary_mask_xy, dtype=bool)
        if outside.shape != free.shape:
            raise ValueError("outside boundary map must match nav class map")
        result[outside] = 0
    return result


def extract_door_seed_stage(
    *,
    voxel_grid: VoxelOccupancyGrid3D,
    navigation_free_mask: np.ndarray,
    navigation_obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    door_seed_no_clearance_free_mask: np.ndarray | None,
    resolution_m: float,
    voxel_evidence_config: VoxelRoomsegEvidenceConfig | Mapping[str, object] | None,
    door_config: VoxelDoorDetectorConfig | Mapping[str, object] | None,
) -> DoorSeedStageResult:
    shape = tuple(voxel_grid.shape)
    nav_free = _shape_checked(navigation_free_mask, shape, "navigation_free_mask")
    no_clearance_free = _shape_checked(
        door_seed_no_clearance_free_mask if door_seed_no_clearance_free_mask is not None else navigation_free_mask,
        shape,
        "door_seed_no_clearance_free_mask",
    )
    nav_occupied = _shape_checked(navigation_obstacle_mask, shape, "navigation_obstacle_mask")
    nav_unknown = _shape_checked(unknown_mask, shape, "unknown_mask")
    outside = outside_roomseg_boundary_mask(voxel_grid, shape)
    if np.any(outside):
        nav_free[outside] = False
        no_clearance_free[outside] = False
        nav_occupied[outside] = False
        nav_unknown[outside] = False

    evidence_cfg = (
        voxel_evidence_config
        if isinstance(voxel_evidence_config, VoxelRoomsegEvidenceConfig)
        else VoxelRoomsegEvidenceConfig.from_mapping(voxel_evidence_config)
    )
    seed_cfg = door_config if isinstance(door_config, VoxelDoorDetectorConfig) else VoxelDoorDetectorConfig.from_mapping(door_config)
    evidence = build_voxel_roomseg_evidence(
        voxel_grid=voxel_grid,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=nav_occupied,
        unknown_mask_from_navigation=nav_unknown,
        resolution_m=float(resolution_m),
        config=evidence_cfg,
    )
    outside_grid_debug = dict(getattr(voxel_grid, "last_outside_debug", {}) or {})
    for key, value in outside_grid_debug.items():
        if isinstance(value, np.ndarray) and np.asarray(value).shape == shape:
            evidence.debug[key] = np.asarray(value).copy()
        elif not isinstance(value, np.ndarray):
            evidence.debug[key] = value
    evidence.debug["voxel_floor_frustum_seen_count_xy"] = np.asarray(
        getattr(voxel_grid, "floor_frustum_seen_count_xy", np.zeros(shape, dtype=np.uint16)),
        dtype=np.uint16,
    ).copy()
    evidence.debug["voxel_outside_score_xy"] = np.asarray(
        getattr(voxel_grid, "outside_score_xy", np.zeros(shape, dtype=np.uint8)),
        dtype=np.uint8,
    ).copy()
    outside_debug = apply_outside_boundary_to_evidence(evidence, outside)
    raw_seed_result = classify_voxel_door_seeds(
        voxel_grid=voxel_grid,
        config=seed_cfg,
        sensor_range_count=getattr(voxel_grid, "sensor_range_count", None),
    )
    raw_seed_mask = np.asarray(raw_seed_result.door_seed_mask, dtype=bool) & ~outside
    vertical_class = encode_vertical_class_map(
        evidence.vertical_free_xy,
        evidence.wall_xy,
        outside_boundary_mask_xy=outside,
    )
    nav_class = encode_nav_class_map(
        no_clearance_free,
        nav_occupied,
        outside_boundary_mask_xy=outside,
    )
    current_map_hash = map_info_hash(voxel_grid.map_info, shape)
    current_seed_hash = config_hash(seed_cfg)
    grid_cfg = getattr(voxel_grid, "config", None)
    current_input_semantics_hash = config_hash(
        {
            "raw_seed_config": seed_cfg,
            "voxel_evidence_config": evidence_cfg,
            "vertical_class_encoding": "unknown0_free1_wall2_free_priority_outside_unknown_v1",
            "nav_class_encoding": "unknown0_free1_occupied2_occupied_priority_outside_unknown_v1",
            "navigation_mask_contract": "caller_navigation_free_no_clearance_free_occupied_unknown_v1",
            "navigation_projection_config": dict(
                getattr(voxel_grid, "door_seed_navigation_semantics_config", {}) or {}
            ),
            "outside_boundary_enabled": bool(getattr(grid_cfg, "outside_boundary_enabled", True)),
            "outside_use_as_roomseg_domain_boundary": bool(
                getattr(grid_cfg, "outside_use_as_roomseg_domain_boundary", True)
            ),
        }
    )
    debug = {
        **outside_debug,
        "voxel_vertical_free_xy": np.asarray(evidence.vertical_free_xy, dtype=bool),
        "voxel_wall_xy": np.asarray(evidence.wall_xy, dtype=bool),
        "voxel_unknown_xy": np.asarray(evidence.unknown_xy, dtype=bool),
        "voxel_nav_free_xy": no_clearance_free.copy(),
        "voxel_nav_occupied_xy": nav_occupied.copy(),
        "voxel_nav_unknown_xy": nav_unknown.copy(),
        "voxel_door_raw_seed_mask": raw_seed_mask.copy(),
        "voxel_door_seed_mask": raw_seed_mask.copy(),
        "door_seed_stage_map_info_hash": current_map_hash,
        "door_seed_stage_raw_seed_config_hash": current_seed_hash,
        "door_seed_stage_input_semantics_hash": current_input_semantics_hash,
    }
    debug.update(raw_seed_result.debug)
    debug["voxel_door_raw_seed_mask"] = raw_seed_mask.copy()
    debug["voxel_door_seed_mask"] = raw_seed_mask.copy()
    return DoorSeedStageResult(
        evidence=evidence,
        raw_seed_result=raw_seed_result,
        raw_seed_mask_xy=raw_seed_mask,
        vertical_class_map_xy=vertical_class,
        nav_class_map_xy=nav_class,
        outside_boundary_mask_xy=outside,
        map_shape=shape,
        map_info_hash=current_map_hash,
        raw_seed_config_hash=current_seed_hash,
        input_semantics_hash=current_input_semantics_hash,
        debug=debug,
        voxroom_raw_seed_mask_xy=raw_seed_mask.copy(),
        tvars_vertical_raw_seed_mask_xy=np.zeros(shape, dtype=bool),
        voxroom_raw_seed_history_mask_xy=raw_seed_mask.copy(),
        tvars_vertical_raw_seed_history_mask_xy=np.zeros(shape, dtype=bool),
    )


def extract_door_seed_stage_incremental(
    *,
    voxel_grid: VoxelOccupancyGrid3D,
    navigation_free_mask: np.ndarray,
    navigation_obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    door_seed_no_clearance_free_mask: np.ndarray | None,
    resolution_m: float,
    voxel_evidence_config: VoxelRoomsegEvidenceConfig | Mapping[str, object] | None,
    door_config: VoxelDoorDetectorConfig | Mapping[str, object] | None,
    previous_stage: DoorSeedStageResult | None,
    update_mask_xy: np.ndarray | None,
    halo_cells: int = 8,
) -> DoorSeedStageResult:
    """Refresh collection evidence only where the latest camera frusta reached.

    The first call intentionally builds a full-map cache.  Later calls run the
    existing, authoritative classifiers on a cropped voxel grid with a halo and
    commit only cells in ``update_mask_xy``.  Consequently, cells outside the
    accumulated sensor footprint retain their previous classification while
    TVARS can still consume a coherent full-map Vertical Free layer.
    """

    shape = tuple(voxel_grid.shape)
    if previous_stage is None or tuple(previous_stage.map_shape) != shape:
        stage = extract_door_seed_stage(
            voxel_grid=voxel_grid,
            navigation_free_mask=navigation_free_mask,
            navigation_obstacle_mask=navigation_obstacle_mask,
            unknown_mask=unknown_mask,
            door_seed_no_clearance_free_mask=door_seed_no_clearance_free_mask,
            resolution_m=float(resolution_m),
            voxel_evidence_config=voxel_evidence_config,
            door_config=door_config,
        )
        stage.debug.update(
            {
                "door_seed_stage_input_semantics_hash": str(stage.input_semantics_hash),
                "door_seed_incremental_mode": "full_initial",
                "door_seed_incremental_update_cells": int(shape[0] * shape[1]),
                "door_seed_incremental_crop_cells": int(shape[0] * shape[1]),
                "door_seed_incremental_halo_cells": max(0, int(halo_cells)),
            }
        )
        return stage

    update = np.asarray(update_mask_xy, dtype=bool) if update_mask_xy is not None else np.zeros(shape, dtype=bool)
    if update.shape != shape:
        raise ValueError("door seed incremental update mask must match voxel grid shape")
    update_count = int(np.count_nonzero(update))
    if update_count == 0:
        return replace(
            previous_stage,
            debug={
                **dict(previous_stage.debug),
                "door_seed_incremental_mode": "cached_no_sensor_update",
                "door_seed_incremental_update_cells": 0,
                "door_seed_incremental_crop_cells": 0,
                "door_seed_incremental_halo_cells": max(0, int(halo_cells)),
            },
        )

    rows, cols = np.nonzero(update)
    halo = max(0, int(halo_cells))
    r0 = max(0, int(rows.min()) - halo)
    r1 = min(shape[0] - 1, int(rows.max()) + halo)
    c0 = max(0, int(cols.min()) - halo)
    c1 = min(shape[1] - 1, int(cols.max()) + halo)
    row_slice = slice(r0, r1 + 1)
    col_slice = slice(c0, c1 + 1)
    crop_shape = (r1 - r0 + 1, c1 - c0 + 1)
    commit_local = update[row_slice, col_slice]
    cropped_grid = _cropped_voxel_grid(voxel_grid, row_slice=row_slice, col_slice=col_slice)
    no_clearance = (
        None
        if door_seed_no_clearance_free_mask is None
        else np.asarray(door_seed_no_clearance_free_mask, dtype=bool)[row_slice, col_slice]
    )
    local_stage = extract_door_seed_stage(
        voxel_grid=cropped_grid,
        navigation_free_mask=np.asarray(navigation_free_mask, dtype=bool)[row_slice, col_slice],
        navigation_obstacle_mask=np.asarray(navigation_obstacle_mask, dtype=bool)[row_slice, col_slice],
        unknown_mask=np.asarray(unknown_mask, dtype=bool)[row_slice, col_slice],
        door_seed_no_clearance_free_mask=no_clearance,
        resolution_m=float(resolution_m),
        voxel_evidence_config=voxel_evidence_config,
        door_config=door_config,
    )

    evidence = _merge_spatial_dataclass(
        previous_stage.evidence,
        local_stage.evidence,
        full_shape=shape,
        crop_shape=crop_shape,
        row_slice=row_slice,
        col_slice=col_slice,
        commit_local=commit_local,
    )
    raw_seed_result = _merge_spatial_dataclass(
        previous_stage.raw_seed_result,
        local_stage.raw_seed_result,
        full_shape=shape,
        crop_shape=crop_shape,
        row_slice=row_slice,
        col_slice=col_slice,
        commit_local=commit_local,
    )
    raw_seed_result = replace(
        raw_seed_result,
        seed_evidence=_merge_seed_evidence(
            previous_stage.raw_seed_result.seed_evidence,
            local_stage.raw_seed_result.seed_evidence,
            update_mask_xy=update,
            crop_origin_rc=(r0, c0),
            commit_local=commit_local,
        ),
    )
    outside = _merge_spatial_array(
        previous_stage.outside_boundary_mask_xy,
        local_stage.outside_boundary_mask_xy,
        full_shape=shape,
        crop_shape=crop_shape,
        row_slice=row_slice,
        col_slice=col_slice,
        commit_local=commit_local,
    ).astype(bool, copy=False)
    raw_seed_mask = np.asarray(raw_seed_result.door_seed_mask, dtype=bool) & ~outside
    raw_seed_result = replace(raw_seed_result, door_seed_mask=raw_seed_mask)
    raw_seed_result.debug.update(
        {
            "voxel_door_seed_mask": raw_seed_mask.copy(),
            "voxel_door_seed_cells": int(np.count_nonzero(raw_seed_mask)),
        }
    )
    vertical_class = encode_vertical_class_map(
        evidence.vertical_free_xy,
        evidence.wall_xy,
        outside_boundary_mask_xy=outside,
    )
    nav_class = _merge_spatial_array(
        previous_stage.nav_class_map_xy,
        local_stage.nav_class_map_xy,
        full_shape=shape,
        crop_shape=crop_shape,
        row_slice=row_slice,
        col_slice=col_slice,
        commit_local=commit_local,
    ).astype(np.uint8, copy=False)
    debug = _merge_spatial_debug(
        previous_stage.debug,
        local_stage.debug,
        full_shape=shape,
        crop_shape=crop_shape,
        row_slice=row_slice,
        col_slice=col_slice,
        commit_local=commit_local,
    )
    debug.update(
        {
            "voxel_vertical_free_xy": np.asarray(evidence.vertical_free_xy, dtype=bool).copy(),
            "voxel_wall_xy": np.asarray(evidence.wall_xy, dtype=bool).copy(),
            "voxel_unknown_xy": np.asarray(evidence.unknown_xy, dtype=bool).copy(),
            "voxel_door_raw_seed_mask": raw_seed_mask.copy(),
            "voxel_door_seed_mask": raw_seed_mask.copy(),
            "door_seed_stage_map_info_hash": str(previous_stage.map_info_hash),
            "door_seed_stage_raw_seed_config_hash": str(previous_stage.raw_seed_config_hash),
            "door_seed_stage_input_semantics_hash": str(previous_stage.input_semantics_hash),
            "door_seed_incremental_mode": "sensor_frustum_crop",
            "door_seed_incremental_update_cells": update_count,
            "door_seed_incremental_crop_cells": int(crop_shape[0] * crop_shape[1]),
            "door_seed_incremental_crop_bbox_rc": [int(r0), int(r1), int(c0), int(c1)],
            "door_seed_incremental_halo_cells": halo,
        }
    )
    return DoorSeedStageResult(
        evidence=evidence,
        raw_seed_result=raw_seed_result,
        raw_seed_mask_xy=raw_seed_mask,
        vertical_class_map_xy=vertical_class,
        nav_class_map_xy=nav_class,
        outside_boundary_mask_xy=outside,
        map_shape=shape,
        map_info_hash=str(previous_stage.map_info_hash),
        raw_seed_config_hash=str(previous_stage.raw_seed_config_hash),
        input_semantics_hash=str(previous_stage.input_semantics_hash),
        debug=debug,
        voxroom_raw_seed_mask_xy=raw_seed_mask.copy(),
        tvars_vertical_raw_seed_mask_xy=np.zeros(shape, dtype=bool),
        voxroom_raw_seed_history_mask_xy=raw_seed_mask.copy(),
        tvars_vertical_raw_seed_history_mask_xy=np.zeros(shape, dtype=bool),
    )


def _cropped_voxel_grid(
    voxel_grid: VoxelOccupancyGrid3D,
    *,
    row_slice: slice,
    col_slice: slice,
) -> VoxelOccupancyGrid3D:
    r0, r1 = int(row_slice.start), int(row_slice.stop) - 1
    c0, c1 = int(col_slice.start), int(col_slice.stop) - 1
    source_info = voxel_grid.map_info
    resolution = float(source_info.resolution_m)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    map_info = MapInfo(
        resolution_m=resolution,
        min_x=float(source_info.min_x) + float(c0) * resolution,
        max_x=float(source_info.min_x) + float(c1 + 1) * resolution,
        min_y=float(source_info.max_y) - float(r1 + 1) * resolution,
        max_y=float(source_info.max_y) - float(r0) * resolution,
        width=int(width),
        height=int(height),
    )
    cropped = VoxelOccupancyGrid3D(
        log_odds=np.asarray(voxel_grid.log_odds)[:, row_slice, col_slice],
        state=np.asarray(voxel_grid.state)[:, row_slice, col_slice],
        sensor_range_count=np.asarray(voxel_grid.sensor_range_count)[:, row_slice, col_slice],
        floor_frustum_seen_count_xy=np.asarray(voxel_grid.floor_frustum_seen_count_xy)[row_slice, col_slice],
        outside_score_xy=np.asarray(voxel_grid.outside_score_xy)[row_slice, col_slice],
        outside_xy=np.asarray(voxel_grid.outside_xy)[row_slice, col_slice],
        z_min_m=float(voxel_grid.z_min_m),
        z_max_m=float(voxel_grid.z_max_m),
        z_resolution_m=float(voxel_grid.z_resolution_m),
        map_info=map_info,
        config=voxel_grid.config,
        active_z_min_m=float(voxel_grid.active_z_min_m),
        active_z_max_m=voxel_grid.active_z_max_m,
        ceiling_height_m=voxel_grid.ceiling_height_m,
        ceiling_estimate_status=str(voxel_grid.ceiling_estimate_status),
        last_integration_stats=voxel_grid.last_integration_stats,
        last_navigation_debug=_crop_debug_mapping(
            voxel_grid.last_navigation_debug,
            voxel_grid.shape,
            row_slice,
            col_slice,
        ),
        last_projective_frustum_debug=_crop_debug_mapping(
            voxel_grid.last_projective_frustum_debug,
            voxel_grid.shape,
            row_slice,
            col_slice,
        ),
        last_projective_frustum_xy=(
            None
            if voxel_grid.last_projective_frustum_xy is None
            else np.asarray(voxel_grid.last_projective_frustum_xy)[row_slice, col_slice]
        ),
        last_floor_frustum_debug=_crop_debug_mapping(
            voxel_grid.last_floor_frustum_debug,
            voxel_grid.shape,
            row_slice,
            col_slice,
        ),
        last_outside_debug=_crop_debug_mapping(
            voxel_grid.last_outside_debug,
            voxel_grid.shape,
            row_slice,
            col_slice,
        ),
        last_dirty_rc_flags=np.asarray(voxel_grid.last_dirty_rc_flags).reshape(voxel_grid.shape)[
            row_slice, col_slice
        ].reshape(-1)
        if voxel_grid.last_dirty_rc_flags is not None
        else None,
    )
    for name in ("door_seed_navigation_semantics_config",):
        if hasattr(voxel_grid, name):
            setattr(cropped, name, getattr(voxel_grid, name))
    return cropped


def _crop_debug_mapping(
    values: Mapping[str, object] | None,
    full_shape: tuple[int, int],
    row_slice: slice,
    col_slice: slice,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in dict(values or {}).items():
        if isinstance(value, np.ndarray) and value.shape == full_shape:
            result[str(key)] = np.asarray(value)[row_slice, col_slice]
        else:
            result[str(key)] = value
    return result


def _merge_spatial_array(
    previous: np.ndarray,
    current_local: np.ndarray,
    *,
    full_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    row_slice: slice,
    col_slice: slice,
    commit_local: np.ndarray,
) -> np.ndarray:
    old = np.asarray(previous)
    new = np.asarray(current_local)
    if old.shape != full_shape or new.shape != crop_shape:
        return old.copy()
    merged = old.copy()
    target = merged[row_slice, col_slice]
    target[commit_local] = new[commit_local]
    return merged


def _merge_spatial_debug(
    previous: Mapping[str, object] | None,
    current_local: Mapping[str, object] | None,
    *,
    full_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    row_slice: slice,
    col_slice: slice,
    commit_local: np.ndarray,
) -> dict[str, object]:
    merged = dict(previous or {})
    for key, value in dict(current_local or {}).items():
        old = merged.get(key)
        if isinstance(old, np.ndarray) and isinstance(value, np.ndarray):
            if old.shape == full_shape and value.shape == crop_shape:
                merged[key] = _merge_spatial_array(
                    old,
                    value,
                    full_shape=full_shape,
                    crop_shape=crop_shape,
                    row_slice=row_slice,
                    col_slice=col_slice,
                    commit_local=commit_local,
                )
                continue
        if not isinstance(value, np.ndarray):
            merged[key] = value
    return merged


def _merge_spatial_dataclass(
    previous,
    current_local,
    *,
    full_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    row_slice: slice,
    col_slice: slice,
    commit_local: np.ndarray,
):
    if not is_dataclass(previous) or not is_dataclass(current_local):
        return previous
    updates: dict[str, object] = {}
    for item in fields(previous):
        old = getattr(previous, item.name)
        new = getattr(current_local, item.name)
        if isinstance(old, np.ndarray) and isinstance(new, np.ndarray):
            updates[item.name] = _merge_spatial_array(
                old,
                new,
                full_shape=full_shape,
                crop_shape=crop_shape,
                row_slice=row_slice,
                col_slice=col_slice,
                commit_local=commit_local,
            )
        elif isinstance(old, Mapping) and isinstance(new, Mapping):
            updates[item.name] = _merge_spatial_debug(
                old,
                new,
                full_shape=full_shape,
                crop_shape=crop_shape,
                row_slice=row_slice,
                col_slice=col_slice,
                commit_local=commit_local,
            )
        elif is_dataclass(old) and is_dataclass(new):
            updates[item.name] = _merge_spatial_dataclass(
                old,
                new,
                full_shape=full_shape,
                crop_shape=crop_shape,
                row_slice=row_slice,
                col_slice=col_slice,
                commit_local=commit_local,
            )
        else:
            updates[item.name] = old
    return replace(previous, **updates)


def _merge_seed_evidence(
    previous: list,
    current_local: list,
    *,
    update_mask_xy: np.ndarray,
    crop_origin_rc: tuple[int, int],
    commit_local: np.ndarray,
) -> list:
    retained = [
        item
        for item in previous
        if not bool(update_mask_xy[int(item.row), int(item.col)])
    ]
    r0, c0 = int(crop_origin_rc[0]), int(crop_origin_rc[1])
    for item in current_local:
        local_r, local_c = int(item.row), int(item.col)
        if not bool(commit_local[local_r, local_c]):
            continue
        retained.append(replace(item, row=local_r + r0, col=local_c + c0))
    return retained


def outside_roomseg_boundary_mask(voxel_grid: VoxelOccupancyGrid3D, shape: tuple[int, int]) -> np.ndarray:
    cfg = getattr(voxel_grid, "config", None)
    enabled = bool(getattr(cfg, "outside_boundary_enabled", True)) and bool(
        getattr(cfg, "outside_use_as_roomseg_domain_boundary", True)
    )
    if not enabled:
        return np.zeros(shape, dtype=bool)
    outside = np.asarray(getattr(voxel_grid, "outside_xy", np.zeros(shape, dtype=bool)), dtype=bool)
    if outside.shape != tuple(shape):
        return np.zeros(shape, dtype=bool)
    return outside.copy()


def apply_outside_boundary_to_evidence(evidence: VoxelRoomsegEvidence, outside_xy: np.ndarray) -> dict[str, object]:
    outside = np.asarray(outside_xy, dtype=bool)
    if outside.shape != np.asarray(evidence.vertical_free_xy).shape:
        raise ValueError("outside boundary map does not match roomseg evidence")
    removed_free = int(np.count_nonzero(np.asarray(evidence.vertical_free_xy, dtype=bool) & outside))
    removed_wall = int(np.count_nonzero(np.asarray(evidence.wall_xy, dtype=bool) & outside))
    removed_unknown = int(np.count_nonzero(np.asarray(evidence.unknown_xy, dtype=bool) & outside))
    fields = (
        "vertical_free_xy",
        "wall_xy",
        "unknown_xy",
        "occupied_any_xy",
        "raw_occupied_wall_support_xy",
        "strict_raw_wall_xy",
        "wall_suppressed_by_free_xy",
        "unknown_dominant_xy",
        "wall_support_loose_xy",
        "wall_support_unknown_gated_xy",
        "wall_support_rejected_unknown_xy",
        "structural_wall_seed_xy",
        "structural_wall_ratio_xy",
        "wall_rejected_by_free_xy",
        "wall_rejected_by_unknown_xy",
        "nonstructural_occupied_xy",
        "small_unknown_hole_filled_xy",
        "wall_line_support_xy",
        "wall_line_support_raw_xy",
        "wall_line_support_rejected_by_free_xy",
        "wall_line_support_rejected_by_unknown_xy",
        "wall_line_support_rejected_by_observed_xy",
        "wall_line_support_rejected_by_nav_edge_xy",
        "ratio_wall_debug_xy",
        "free_wall_conflict_xy",
        "wall_line_support_strong_xy",
        "wall_line_support_conflict_xy",
        "wall_line_support_near_free_boundary_xy",
        "wall_line_support_rejected_furniture_xy",
        "wall_line_support_weight_xy",
        "wall_support_raw_occupied_xy",
        "wall_support_known_xy",
        "wall_support_unknown_rejected_xy",
        "wall_support_nav_unknown_rejected_xy",
        "wall_support_frontier_band_rejected_xy",
        "wall_support_free_conflict_xy",
        "wall_support_strong_xy",
        "wall_support_for_projection_xy",
        "frontier_unknown_band_xy",
        "strong_structural_support_xy",
        "bridge_only_support_xy",
        "forbidden_frontier_residual_support_xy",
        "forbidden_unknown_boundary_support_xy",
        "free_conflict_support_xy",
        "protected_structural_wall_band_xy",
        "support_seed_for_projection_xy",
        "support_bridge_for_projection_xy",
    )
    for name in fields:
        value = getattr(evidence, name, None)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.shape != outside.shape:
            continue
        updated = arr.copy()
        if updated.dtype == bool:
            updated[outside] = False
        elif np.issubdtype(updated.dtype, np.floating):
            updated[outside] = 0.0
        else:
            continue
        setattr(evidence, name, updated)
    evidence.debug["voxel_outside_xy"] = outside.copy()
    debug = {
        "voxel_outside_roomseg_boundary_enabled": True,
        "voxel_outside_roomseg_boundary_cells": int(np.count_nonzero(outside)),
        "voxel_outside_removed_free_cells": removed_free,
        "voxel_outside_removed_wall_cells": removed_wall,
        "voxel_outside_removed_unknown_cells": removed_unknown,
        "voxel_outside_use_as_wall_evidence": False,
        "voxel_outside_use_as_door_anchor": False,
        "voxel_outside_use_as_separator_anchor": False,
    }
    evidence.debug.update(debug)
    return debug


def _shape_checked(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        raise ValueError("%s must match voxel grid shape" % name)
    return arr.copy()
