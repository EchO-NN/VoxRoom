from __future__ import annotations

from typing import Mapping

import numpy as np


RAY_VALID_WALL_INFERENCE_MODE = "ray_valid_terminal_wall"


def build_ray_valid_wall_inference(
    *,
    vertical_free: np.ndarray,
    vertical_occupied: np.ndarray,
    vertical_observed: np.ndarray | None = None,
    terminal_wall_count: np.ndarray | None = None,
    terminal_wall_splat: np.ndarray | None = None,
    ray_covered_count: np.ndarray | None = None,
    terminal_wall_height_min: np.ndarray | None = None,
    terminal_wall_height_max: np.ndarray | None = None,
    terminal_wall_depth_min: np.ndarray | None = None,
    config: object | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the roomseg input from vertical-free and valid depth-ray endpoints.

    This intentionally does not consume navigation free/occupied masks. A cell
    becomes roomseg occupied only from vertical endpoint evidence or terminal
    wall evidence from a valid depth ray. Vertical-free wins over everything.
    """

    vf = np.asarray(vertical_free, dtype=bool)
    vo = np.asarray(vertical_occupied, dtype=bool)
    if vf.shape != vo.shape:
        raise ValueError("vertical_free and vertical_occupied must have the same HxW shape")
    shape = vf.shape
    cfg = _config_dict(config)
    enabled = bool(cfg.get("enabled", True))
    min_terminal_wall_count = max(1, int(cfg.get("min_terminal_wall_count", 1)))
    splat_radius = max(0, int(cfg.get("terminal_wall_splat_radius_cells", 1)))
    require_no_vertical_free = bool(cfg.get("require_no_vertical_free", True))

    tw_count = _optional_uint(terminal_wall_count, shape)
    if terminal_wall_splat is None:
        terminal = tw_count >= min_terminal_wall_count
        tw_splat = _dilate_binary(terminal, splat_radius)
    else:
        tw_splat = np.asarray(terminal_wall_splat, dtype=bool)
        if tw_splat.shape != shape:
            raise ValueError("terminal_wall_splat has shape %s, expected %s" % (tw_splat.shape, shape))
        if np.any(tw_count):
            tw_splat |= _dilate_binary(tw_count >= min_terminal_wall_count, splat_radius)

    if require_no_vertical_free:
        tw_splat &= ~vf
    if not enabled:
        tw_splat = np.zeros(shape, dtype=bool)

    occupied_before_ray_wall = vo & ~vf
    unknown_before_ray_wall = ~(vf | occupied_before_ray_wall)
    ray_valid_wall = (occupied_before_ray_wall | tw_splat) & ~vf
    initial_free = vf.copy()
    initial_occupied = ray_valid_wall.copy()
    initial_unknown = ~(initial_free | initial_occupied)
    vertical_free_overridden = initial_free & initial_occupied
    if np.any(vertical_free_overridden):
        raise AssertionError("ray-valid roomseg wall inference overrode vertical-free cells")
    unknown_removed_by_ray_wall = unknown_before_ray_wall & initial_occupied
    if np.any(initial_free & ray_valid_wall):
        raise AssertionError("ray-valid wall mask overlaps vertical-free cells")

    observed = _optional_bool(vertical_observed, shape) | initial_free | initial_occupied
    ray_covered = _optional_uint(ray_covered_count, shape)
    hmin = _optional_float(terminal_wall_height_min, shape, fill=np.inf)
    hmax = _optional_float(terminal_wall_height_max, shape, fill=-np.inf)
    dmin = _optional_float(terminal_wall_depth_min, shape, fill=np.inf)

    debug = {
        "ray_valid_wall_inference_enabled": bool(enabled),
        "ray_valid_wall_inference_mode": str(cfg.get("mode", RAY_VALID_WALL_INFERENCE_MODE) or RAY_VALID_WALL_INFERENCE_MODE),
        "depth_max_m": float(cfg.get("depth_max_m", 3.0)),
        "vertical_profile_free_min_height_m": float(cfg.get("min_endpoint_height_m", 0.20)),
        "vertical_profile_free_max_height_m": float(cfg.get("max_endpoint_height_m", 2.00)),
        "min_terminal_wall_count": int(min_terminal_wall_count),
        "terminal_wall_splat_radius_cells": int(splat_radius),
        "require_no_vertical_free": bool(require_no_vertical_free),
        "strict_no_navigation_obstacle_overlay": bool(cfg.get("strict_no_navigation_obstacle_overlay", True)),
        "strict_no_navigation_free_overlay": bool(cfg.get("strict_no_navigation_free_overlay", True)),
        "vertical_free_cells": int(np.count_nonzero(vf)),
        "vertical_occupied_cells": int(np.count_nonzero(vo)),
        "vertical_observed_cells": int(np.count_nonzero(observed)),
        "roomseg_ray_covered_cells": int(np.count_nonzero(ray_covered)),
        "roomseg_ray_covered_count_sum": int(np.sum(ray_covered, dtype=np.uint64)),
        "terminal_wall_cells": int(np.count_nonzero(tw_count >= min_terminal_wall_count)),
        "terminal_wall_count_sum": int(np.sum(tw_count, dtype=np.uint64)),
        "terminal_wall_splat_cells": int(np.count_nonzero(tw_splat)),
        "unknown_before_cells": int(np.count_nonzero(unknown_before_ray_wall)),
        "unknown_after_cells": int(np.count_nonzero(initial_unknown)),
        "unknown_removed_by_ray_wall_cells": int(np.count_nonzero(unknown_removed_by_ray_wall)),
        "vertical_free_overridden_by_wall_cells": int(np.count_nonzero(vertical_free_overridden)),
    }
    return {
        "initial_roomseg_free": initial_free.astype(bool),
        "initial_roomseg_occupied": initial_occupied.astype(bool),
        "initial_roomseg_unknown": initial_unknown.astype(bool),
        "vertical_observed_map": observed.astype(bool),
        "vertical_occupied_0p2_2p0": vo.astype(bool),
        "vertical_observed_0p2_2p0": observed.astype(bool),
        "roomseg_ray_covered_count": ray_covered.astype(np.uint16),
        "roomseg_terminal_wall_count": tw_count.astype(np.uint16),
        "roomseg_terminal_wall_height_min": hmin.astype(np.float32),
        "roomseg_terminal_wall_height_max": hmax.astype(np.float32),
        "roomseg_terminal_wall_depth_min": dmin.astype(np.float32),
        "roomseg_terminal_wall_splat": tw_splat.astype(bool),
        "ray_valid_wall_inference": ray_valid_wall.astype(bool),
        "initial_roomseg_free_after_ray_wall": initial_free.astype(bool),
        "initial_roomseg_occupied_after_ray_wall": initial_occupied.astype(bool),
        "initial_roomseg_unknown_after_ray_wall": initial_unknown.astype(bool),
        "unknown_before_ray_wall": unknown_before_ray_wall.astype(bool),
        "unknown_after_ray_wall": initial_unknown.astype(bool),
        "unknown_removed_by_ray_wall": unknown_removed_by_ray_wall.astype(bool),
        "debug": debug,
    }


def _config_dict(config: object | Mapping[str, object] | None) -> dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        raw = dict(config)
        nested = raw.get("ray_valid_wall_inference")
        return dict(nested or raw)
    if hasattr(config, "ray_valid_wall_inference"):
        return dict(getattr(config, "ray_valid_wall_inference") or {})
    return {
        "enabled": getattr(config, "ray_valid_wall_inference_enabled", True),
        "mode": getattr(config, "ray_valid_wall_inference_mode", RAY_VALID_WALL_INFERENCE_MODE),
        "depth_max_m": getattr(config, "ray_valid_wall_depth_max_m", 3.0),
        "min_endpoint_height_m": getattr(config, "ray_valid_wall_min_endpoint_height_m", 0.20),
        "max_endpoint_height_m": getattr(config, "ray_valid_wall_max_endpoint_height_m", 2.00),
        "min_terminal_wall_count": getattr(config, "ray_valid_wall_min_terminal_wall_count", 1),
        "terminal_wall_splat_radius_cells": getattr(config, "ray_valid_wall_terminal_wall_splat_radius_cells", 1),
        "require_no_vertical_free": getattr(config, "ray_valid_wall_require_no_vertical_free", True),
        "mark_ray_covered_debug": getattr(config, "ray_valid_wall_mark_ray_covered_debug", True),
        "decouple_roomseg_rays_from_nav_clear": getattr(config, "ray_valid_wall_decouple_roomseg_rays_from_nav_clear", True),
        "strict_no_navigation_obstacle_overlay": getattr(config, "ray_valid_wall_strict_no_navigation_obstacle_overlay", True),
        "strict_no_navigation_free_overlay": getattr(config, "ray_valid_wall_strict_no_navigation_free_overlay", True),
    }


def _optional_bool(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        raise ValueError("optional roomseg ray mask has shape %s, expected %s" % (arr.shape, shape))
    return arr


def _optional_uint(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.uint16)
    arr = np.asarray(value, dtype=np.uint32)
    if arr.shape != shape:
        raise ValueError("optional roomseg ray count has shape %s, expected %s" % (arr.shape, shape))
    return np.minimum(arr, np.iinfo(np.uint16).max).astype(np.uint16)


def _optional_float(value: np.ndarray | None, shape: tuple[int, int], *, fill: float) -> np.ndarray:
    if value is None:
        return np.full(shape, float(fill), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError("optional roomseg ray float array has shape %s, expected %s" % (arr.shape, shape))
    return arr


def _dilate_binary(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    radius = int(max(0, radius_cells))
    if radius <= 0 or not np.any(src):
        return src.copy()
    out = src.copy()
    rows, cols = np.nonzero(src)
    h, w = src.shape
    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= radius * radius:
                offsets.append((dr, dc))
    for row, col in zip(rows, cols):
        for dr, dc in offsets:
            rr, cc = int(row + dr), int(col + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out
