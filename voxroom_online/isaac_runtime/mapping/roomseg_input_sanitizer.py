from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage


@dataclass
class RoomsegInputSanitizerConfig:
    enabled: bool = True
    enforce_free_subset_of_navigation: bool = True
    clip_vertical_free_to_navigation: bool = True
    protect_wall_core_from_free_overlap: bool = True
    use_terminal_wall_splat_as_wall: bool = True
    terminal_wall_min_count: int = 1
    terminal_wall_splat_radius_cells: int = 1
    wall_core_min_component_cells: int = 2
    navigation_free_dilation_cells: int = 0
    outside_nav_warn_ratio: float = 0.01
    wall_free_conflict_warn_ratio: float = 0.02

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "RoomsegInputSanitizerConfig":
        if isinstance(data, cls):
            return data
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


def sanitize_roomseg_inputs(
    *,
    vertical_free_raw: np.ndarray,
    wall_raw: np.ndarray,
    unknown_raw: np.ndarray,
    navigation_free_mask: np.ndarray | None,
    navigation_obstacle_mask: np.ndarray | None = None,
    roomseg_ray_evidence: Mapping[str, np.ndarray] | None = None,
    cfg: RoomsegInputSanitizerConfig | Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    config = cfg if isinstance(cfg, RoomsegInputSanitizerConfig) else RoomsegInputSanitizerConfig.from_mapping(cfg)
    free_raw = np.asarray(vertical_free_raw, dtype=bool).copy()
    wall_raw_arr = np.asarray(wall_raw, dtype=bool).copy()
    unknown = np.asarray(unknown_raw, dtype=bool).copy()
    if free_raw.shape != wall_raw_arr.shape or free_raw.shape != unknown.shape:
        raise ValueError("roomseg input sanitizer masks must have same HxW shape")

    shape = free_raw.shape
    nav_available = navigation_free_mask is not None
    if nav_available:
        nav_free = np.asarray(navigation_free_mask, dtype=bool)
        if nav_free.shape != shape:
            raise ValueError("navigation_free_mask must have same HxW shape")
    else:
        nav_free = np.ones(shape, dtype=bool)

    if not bool(config.enabled):
        free = free_raw & ~unknown
        wall = wall_raw_arr & ~unknown & ~free
        debug = _debug_payload(
            config=config,
            nav_available=nav_available,
            free_raw=free_raw,
            nav_free=nav_free if nav_available else None,
            outside_nav=np.zeros(shape, dtype=bool),
            conflict=free & wall_raw_arr,
            terminal_wall=np.zeros(shape, dtype=bool),
            free=free,
            wall=wall,
            unknown=unknown,
        )
        debug["roomseg_input_sanitizer_enabled"] = False
        return free.astype(bool), wall.astype(bool), unknown.astype(bool), debug

    nav_gate = nav_free
    if nav_available and int(config.navigation_free_dilation_cells) > 0:
        nav_gate = ndimage.binary_dilation(nav_free, structure=_disk(int(config.navigation_free_dilation_cells)))

    outside_nav = free_raw & ~nav_gate if nav_available else np.zeros(shape, dtype=bool)
    if nav_available and bool(config.clip_vertical_free_to_navigation):
        free = free_raw & nav_gate & ~unknown
    else:
        free = free_raw & ~unknown

    terminal_wall = _terminal_wall_mask(
        shape=shape,
        roomseg_ray_evidence=roomseg_ray_evidence,
        min_count=int(config.terminal_wall_min_count),
        radius_cells=int(config.terminal_wall_splat_radius_cells),
        enabled=bool(config.use_terminal_wall_splat_as_wall),
    )
    wall_candidate = (wall_raw_arr | terminal_wall) & ~unknown

    if bool(config.protect_wall_core_from_free_overlap):
        wall_core = wall_candidate.copy()
        removed = remove_tiny_components_but_keep_supported_terminal_wall(
            wall_core,
            terminal_wall=terminal_wall,
            min_cells=int(config.wall_core_min_component_cells),
        )
        conflict = free & wall_core
        free = free & ~wall_core
        wall = wall_core
    else:
        conflict = free & wall_candidate
        wall = wall_candidate & ~free
        removed = 0

    if navigation_obstacle_mask is not None:
        nav_obstacle = np.asarray(navigation_obstacle_mask, dtype=bool)
        if nav_obstacle.shape != shape:
            raise ValueError("navigation_obstacle_mask must have same HxW shape")
        wall = wall | (nav_obstacle & ~unknown)
        free = free & ~nav_obstacle
    else:
        nav_obstacle = np.zeros(shape, dtype=bool)

    if nav_available and bool(config.enforce_free_subset_of_navigation):
        leaked = free & ~nav_free
        if np.any(leaked):
            raise AssertionError("roomseg vertical free leaked outside navigation free")

    debug = _debug_payload(
        config=config,
        nav_available=nav_available,
        free_raw=free_raw,
        nav_free=nav_free if nav_available else None,
        outside_nav=outside_nav,
        conflict=conflict,
        terminal_wall=terminal_wall,
        free=free,
        wall=wall,
        unknown=unknown,
    )
    debug.update(
        {
            "roomseg_input_sanitizer_removed_tiny_wall_core_cells": int(removed),
            "roomseg_input_sanitizer_navigation_obstacle_cells": int(np.count_nonzero(nav_obstacle)),
        }
    )
    return free.astype(bool), wall.astype(bool), unknown.astype(bool), debug


def validate_roomseg_input_invariants(
    *,
    free: np.ndarray,
    navigation_free_mask: np.ndarray | None,
) -> dict:
    free_arr = np.asarray(free, dtype=bool)
    if navigation_free_mask is None:
        return {"sanitized_free_subset_navigation_ok": False, "navigation_free_available": False}
    nav = np.asarray(navigation_free_mask, dtype=bool)
    leaked = free_arr & ~nav
    return {
        "sanitized_free_subset_navigation_ok": bool(not np.any(leaked)),
        "navigation_free_available": True,
        "sanitized_free_outside_navigation_cells": int(np.count_nonzero(leaked)),
    }


def remove_tiny_components_but_keep_supported_terminal_wall(
    mask: np.ndarray,
    *,
    terminal_wall: np.ndarray,
    min_cells: int,
) -> int:
    if int(min_cells) <= 1:
        return 0
    arr = np.asarray(mask, dtype=bool)
    terminal = np.asarray(terminal_wall, dtype=bool)
    labels, count = ndimage.label(arr, structure=np.ones((3, 3), dtype=bool))
    removed = 0
    for label in range(1, int(count) + 1):
        comp = labels == int(label)
        cells = int(np.count_nonzero(comp))
        if cells >= int(min_cells) or np.any(comp & terminal):
            continue
        arr[comp] = False
        removed += cells
    mask[...] = arr
    return int(removed)


def _terminal_wall_mask(
    *,
    shape: tuple[int, int],
    roomseg_ray_evidence: Mapping[str, np.ndarray] | None,
    min_count: int,
    radius_cells: int,
    enabled: bool,
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if not enabled:
        return out
    evidence = dict(roomseg_ray_evidence or {})
    splat = _first_array(evidence, shape, "terminal_wall_splat", "roomseg_terminal_wall_splat", dtype=bool)
    if splat is not None:
        out |= np.asarray(splat, dtype=bool)
    count = _first_array(evidence, shape, "terminal_wall_count", "roomseg_terminal_wall_count", dtype=np.uint16)
    if count is not None:
        out |= np.asarray(count, dtype=np.uint16) >= int(min_count)
    if int(radius_cells) > 0 and np.any(out):
        out = ndimage.binary_dilation(out, structure=_disk(int(radius_cells)))
    return out.astype(bool)


def _first_array(
    evidence: Mapping[str, np.ndarray],
    shape: tuple[int, int],
    *names: str,
    dtype,
) -> np.ndarray | None:
    for name in names:
        value = evidence.get(name)
        if value is None:
            continue
        arr = np.asarray(value, dtype=dtype)
        if arr.shape == shape:
            return arr
    return None


def _debug_payload(
    *,
    config: RoomsegInputSanitizerConfig,
    nav_available: bool,
    free_raw: np.ndarray,
    nav_free: np.ndarray | None,
    outside_nav: np.ndarray,
    conflict: np.ndarray,
    terminal_wall: np.ndarray,
    free: np.ndarray,
    wall: np.ndarray,
    unknown: np.ndarray,
) -> dict:
    nav_cells = None if nav_free is None else int(np.count_nonzero(nav_free))
    raw_cells = int(np.count_nonzero(free_raw))
    outside_cells = int(np.count_nonzero(outside_nav))
    conflict_cells = int(np.count_nonzero(conflict))
    denom = max(raw_cells, 1)
    conflict_denom = max(raw_cells + int(np.count_nonzero(wall)), 1)
    return {
        "roomseg_input_sanitizer_enabled": bool(config.enabled),
        "roomseg_input_sanitizer_navigation_free_available": bool(nav_available),
        "vertical_free_raw_cells": int(raw_cells),
        "navigation_free_cells": nav_cells,
        "vertical_free_outside_navigation_cells": int(outside_cells),
        "vertical_free_outside_navigation_ratio": float(outside_cells / denom),
        "vertical_free_clipped_outside_navigation_map": np.asarray(outside_nav, dtype=bool),
        "vertical_free_outside_navigation_warning": bool(outside_cells / denom > float(config.outside_nav_warn_ratio)),
        "free_wall_conflict_cells_before_sanitize": int(conflict_cells),
        "free_wall_conflict_map_before_sanitize": np.asarray(conflict, dtype=bool),
        "free_wall_conflict_warning": bool(conflict_cells / conflict_denom > float(config.wall_free_conflict_warn_ratio)),
        "sanitized_free_cells": int(np.count_nonzero(free)),
        "sanitized_wall_cells": int(np.count_nonzero(wall)),
        "sanitized_unknown_cells": int(np.count_nonzero(unknown)),
        "sanitized_free_subset_navigation_ok": bool(not nav_available or not np.any(np.asarray(free, dtype=bool) & ~np.asarray(nav_free, dtype=bool))),
        "roomseg_sanitized_free": np.asarray(free, dtype=bool),
        "roomseg_sanitized_wall": np.asarray(wall, dtype=bool),
        "terminal_wall_splat_cells": int(np.count_nonzero(terminal_wall)),
        "terminal_wall_count_cells": int(np.count_nonzero(terminal_wall)),
        "terminal_wall_used_as_roomseg_wall": bool(np.any(terminal_wall)),
        "terminal_wall_roomseg_mask": np.asarray(terminal_wall, dtype=bool),
    }


def _disk(radius_cells: int) -> np.ndarray:
    r = int(radius_cells)
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r
