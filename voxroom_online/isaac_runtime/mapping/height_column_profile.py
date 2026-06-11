from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import ndimage


HEIGHT_STATE_UNKNOWN = np.uint8(0)
HEIGHT_STATE_FREE = np.uint8(1)
HEIGHT_STATE_OCCUPIED = np.uint8(2)
HEIGHT_STATE_CONFLICT = np.uint8(3)


@dataclass
class HeightColumnProfileConfig:
    enabled: bool = True
    z_min_m: float = 0.10
    z_max_m: float = 3.20
    storage_z_max_m: float = 3.20
    z_bin_size_m: float = 0.05
    active_z_min_m: float = 0.10
    active_z_max_m: float | None = None
    active_z_max_fallback_m: float = 2.80
    active_z_max_ceiling_ratio: float = 0.85
    active_z_max_cap_m: float = 2.80
    min_free_rays_per_bin: int = 1
    min_occupied_points_per_bin: int = 1
    min_free_z_bins_for_xy_free: int = 3
    wall_occupied_ratio_min: float = 0.95
    wall_min_observed_z_bins: int = 8
    wall_min_occupied_z_bins: int = 8
    use_navigation_free_gate: bool = True
    navigation_free_gate_dilation_cells: int = 0
    hard_wall_overrides_free: bool = True
    wall_ratio_denominator: str = "active_bins"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "HeightColumnProfileConfig":
        if isinstance(data, cls):
            return data
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


def _active_z_cap_or_storage_max(config: HeightColumnProfileConfig) -> float:
    cap = float(getattr(config, "active_z_max_cap_m", 0.0))
    if np.isfinite(cap) and cap > 0.0:
        return float(cap)
    return float(getattr(config, "storage_z_max_m", getattr(config, "z_max_m", 3.20)))


@dataclass
class HeightColumnClassification:
    z_state: np.ndarray
    free_bin_mask: np.ndarray
    occupied_bin_mask: np.ndarray
    conflict_bin_mask: np.ndarray
    observed_bin_mask: np.ndarray
    active_z_bin_mask: np.ndarray
    vertical_free_raw_xy: np.ndarray
    vertical_free_xy: np.ndarray
    wall_xy: np.ndarray
    unknown_xy: np.ndarray
    free_z_bin_count_xy: np.ndarray
    occupied_z_bin_count_xy: np.ndarray
    conflict_z_bin_count_xy: np.ndarray
    observed_z_bin_count_xy: np.ndarray
    active_z_bin_count_xy: np.ndarray
    occupied_ratio_observed_xy: np.ndarray
    occupied_ratio_active_xy: np.ndarray
    vertical_free_outside_navigation_xy: np.ndarray
    free_wall_conflict_xy: np.ndarray
    navigation_gate_xy: np.ndarray
    debug: dict[str, object] = field(default_factory=dict)


@dataclass
class HeightColumnProfileMap:
    free_ray_count: np.ndarray
    occupied_count: np.ndarray
    observed_count: np.ndarray
    z_min_m: float
    z_max_m: float
    z_bin_size_m: float
    active_z_min_m: float = 0.10
    active_z_max_m: float | None = None

    @classmethod
    def zeros(cls, shape: tuple[int, int], cfg: HeightColumnProfileConfig | Mapping[str, object] | None = None) -> "HeightColumnProfileMap":
        config = cfg if isinstance(cfg, HeightColumnProfileConfig) else HeightColumnProfileConfig.from_mapping(cfg)
        storage_z_max = float(config.storage_z_max_m if config.storage_z_max_m is not None else config.z_max_m)
        bins = _bin_count(float(config.z_min_m), storage_z_max, float(config.z_bin_size_m))
        h, w = int(shape[0]), int(shape[1])
        zeros = np.zeros((bins, h, w), dtype=np.uint16)
        return cls(
            free_ray_count=zeros.copy(),
            occupied_count=zeros.copy(),
            observed_count=zeros.copy(),
            z_min_m=float(config.z_min_m),
            z_max_m=storage_z_max,
            z_bin_size_m=float(config.z_bin_size_m),
            active_z_min_m=float(config.active_z_min_m),
            active_z_max_m=float(config.active_z_max_m if config.active_z_max_m is not None else _active_z_cap_or_storage_max(config)),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.free_ray_count.shape[1]), int(self.free_ray_count.shape[2])

    @property
    def z_bin_count(self) -> int:
        return int(self.free_ray_count.shape[0])

    @property
    def bin_edges_m(self) -> np.ndarray:
        edges = float(self.z_min_m) + np.arange(self.z_bin_count + 1, dtype=np.float32) * float(self.z_bin_size_m)
        edges[-1] = min(float(edges[-1]), float(self.z_max_m))
        return edges.astype(np.float32)

    @property
    def bin_centers_m(self) -> np.ndarray:
        edges = self.bin_edges_m
        return ((edges[:-1] + edges[1:]) * 0.5).astype(np.float32)

    @property
    def bin_ranges_m(self) -> tuple[tuple[float, float], ...]:
        edges = self.bin_edges_m
        return tuple((float(edges[idx]), float(edges[idx + 1])) for idx in range(len(edges) - 1))

    def reset_shape(self, shape: tuple[int, int]) -> None:
        fresh = HeightColumnProfileMap.zeros(
            shape,
            HeightColumnProfileConfig(
                z_min_m=float(self.z_min_m),
                z_max_m=float(self.z_max_m),
                storage_z_max_m=float(self.z_max_m),
                z_bin_size_m=float(self.z_bin_size_m),
                active_z_min_m=float(getattr(self, "active_z_min_m", self.z_min_m)),
                active_z_max_m=getattr(self, "active_z_max_m", None),
            ),
        )
        self.free_ray_count = fresh.free_ray_count
        self.occupied_count = fresh.occupied_count
        self.observed_count = fresh.observed_count
        self.active_z_min_m = fresh.active_z_min_m
        self.active_z_max_m = fresh.active_z_max_m

    def bin_index_for_height(self, rel_z_m: float) -> int | None:
        z = float(rel_z_m)
        if z < float(self.z_min_m) or z >= float(self.z_max_m):
            return None
        idx = int(np.floor((z - float(self.z_min_m)) / max(float(self.z_bin_size_m), 1e-9)))
        if 0 <= idx < self.z_bin_count:
            return int(idx)
        return None

    def mark_occupied_points(self, rows_cols: np.ndarray, rel_z_m: np.ndarray) -> None:
        cells = np.asarray(rows_cols, dtype=np.int32).reshape(-1, 2)
        heights = np.asarray(rel_z_m, dtype=np.float32).reshape(-1)
        if cells.size == 0 or len(cells) != len(heights):
            return
        valid = np.isfinite(heights) & (heights >= float(self.z_min_m)) & (heights < float(self.z_max_m))
        if int(np.count_nonzero(valid)) == 0:
            return
        bin_idx = np.floor((heights[valid] - float(self.z_min_m)) / max(float(self.z_bin_size_m), 1e-9)).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, self.z_bin_count - 1)
        valid_cells = cells[valid]
        for idx in np.unique(bin_idx):
            self._increment_cells(self.occupied_count[int(idx)], valid_cells[bin_idx == int(idx)])
            self._increment_cells(self.observed_count[int(idx)], valid_cells[bin_idx == int(idx)])

    def mark_free_ray_cells_by_bin(self, flat_indices_by_bin: list[np.ndarray], weights_by_bin: list[np.ndarray]) -> None:
        h, w = self.shape
        total_cells = int(h * w)
        uint16_max = int(np.iinfo(np.uint16).max)
        for bin_idx, (flat_values, weights) in enumerate(zip(flat_indices_by_bin, weights_by_bin)):
            if int(bin_idx) >= self.z_bin_count:
                break
            flat, weight = _valid_flat_weight_arrays(flat_values, weights, total_cells)
            if flat.size == 0:
                continue
            counts = np.bincount(flat, weights=weight, minlength=total_cells).reshape(h, w).astype(np.uint32)
            free_updated = np.asarray(self.free_ray_count[bin_idx], dtype=np.uint32) + counts
            observed_updated = np.asarray(self.observed_count[bin_idx], dtype=np.uint32) + counts
            self.free_ray_count[bin_idx][:, :] = np.minimum(free_updated, uint16_max).astype(np.uint16)
            self.observed_count[bin_idx][:, :] = np.minimum(observed_updated, uint16_max).astype(np.uint16)

    def classify_columns(
        self,
        *,
        navigation_free_mask: np.ndarray | None,
        cfg: HeightColumnProfileConfig | Mapping[str, object] | None = None,
        active_z_min_m: float | None = None,
        active_z_max_m: float | None = None,
    ) -> HeightColumnClassification:
        config = cfg if isinstance(cfg, HeightColumnProfileConfig) else HeightColumnProfileConfig.from_mapping(cfg)
        observed_bin = np.asarray(self.observed_count, dtype=np.uint32) > 0
        occupied_bin = np.asarray(self.occupied_count, dtype=np.uint32) >= max(1, int(config.min_occupied_points_per_bin))
        free_bin = np.asarray(self.free_ray_count, dtype=np.uint32) >= max(1, int(config.min_free_rays_per_bin))
        active_min = float(active_z_min_m if active_z_min_m is not None else getattr(self, "active_z_min_m", config.active_z_min_m))
        configured_active_max = active_z_max_m if active_z_max_m is not None else (config.active_z_max_m if config.active_z_max_m is not None else getattr(self, "active_z_max_m", None))
        active_max = float(configured_active_max if configured_active_max is not None else _active_z_cap_or_storage_max(config))
        active_min = max(float(self.z_min_m), active_min)
        active_max = min(float(self.z_max_m), max(active_min, active_max))
        centers = self.bin_centers_m
        active_1d = (centers >= active_min) & (centers <= active_max)
        active = np.broadcast_to(active_1d[:, None, None], self.free_ray_count.shape).astype(bool, copy=False)

        z_state = np.zeros(self.free_ray_count.shape, dtype=np.uint8)
        free_only = active & free_bin & ~occupied_bin
        occupied_only = active & occupied_bin & ~free_bin
        conflict_bin = active & free_bin & occupied_bin
        z_state[free_only] = HEIGHT_STATE_FREE
        z_state[occupied_only] = HEIGHT_STATE_OCCUPIED
        z_state[conflict_bin] = HEIGHT_STATE_CONFLICT

        free_count = np.sum(z_state == HEIGHT_STATE_FREE, axis=0).astype(np.uint16)
        occupied_count = np.sum(z_state == HEIGHT_STATE_OCCUPIED, axis=0).astype(np.uint16)
        conflict_count = np.sum(z_state == HEIGHT_STATE_CONFLICT, axis=0).astype(np.uint16)
        observed_count = np.sum(active & (free_bin | occupied_bin), axis=0).astype(np.uint16)
        active_count_scalar = int(np.count_nonzero(active_1d))
        active_count = np.full(self.shape, active_count_scalar, dtype=np.uint16)
        ratio_observed = np.zeros(self.shape, dtype=np.float32)
        np.divide(
            occupied_count.astype(np.float32),
            np.maximum(observed_count.astype(np.float32), 1.0),
            out=ratio_observed,
        )
        ratio_active = np.zeros(self.shape, dtype=np.float32)
        np.divide(
            occupied_count.astype(np.float32),
            np.maximum(active_count.astype(np.float32), 1.0),
            out=ratio_active,
        )

        vertical_free_raw = free_count >= int(config.min_free_z_bins_for_xy_free)
        occupied_required = int(np.ceil(max(0, active_count_scalar) * float(config.wall_occupied_ratio_min)))
        wall = (active_count_scalar > 0) & (occupied_count >= max(1, occupied_required))

        nav_gate = np.ones(self.shape, dtype=bool)
        if bool(config.use_navigation_free_gate):
            if navigation_free_mask is None:
                nav_gate = np.zeros(self.shape, dtype=bool)
            else:
                nav_gate = np.asarray(navigation_free_mask, dtype=bool).copy()
                if nav_gate.shape != self.shape:
                    raise ValueError("navigation_free_mask must match height profile shape")
                radius = int(config.navigation_free_gate_dilation_cells)
                if radius > 0:
                    nav_gate = ndimage.binary_dilation(nav_gate, structure=_disk(radius)).astype(bool)

        outside_nav = vertical_free_raw & ~nav_gate
        vertical_free = vertical_free_raw & nav_gate
        conflict = vertical_free_raw & wall
        if bool(config.hard_wall_overrides_free):
            vertical_free = vertical_free & ~wall
        unknown = ~(vertical_free | wall)

        summary = {
            "vertical_free_raw_cells": int(np.count_nonzero(vertical_free_raw)),
            "vertical_free_after_nav_gate_cells": int(np.count_nonzero(vertical_free)),
            "vertical_free_outside_navigation_cells": int(np.count_nonzero(outside_nav)),
            "wall_cells": int(np.count_nonzero(wall)),
            "unknown_cells": int(np.count_nonzero(unknown)),
            "free_wall_conflict_cells": int(np.count_nonzero(conflict)),
            "conflict_z_bin_cells": int(np.count_nonzero(conflict_bin)),
            "active_z_bins": int(active_count_scalar),
            "active_z_min_m": float(active_min),
            "active_z_max_m": float(active_max),
            "wall_occupied_required_bins": int(max(1, occupied_required)) if active_count_scalar > 0 else 0,
            "mean_observed_z_bins": float(np.mean(observed_count.astype(np.float32))) if observed_count.size else 0.0,
            "max_observed_z_bins": int(np.max(observed_count)) if observed_count.size else 0,
        }
        debug = {
            "height_profile_z_min_m": float(self.z_min_m),
            "height_profile_z_max_m": float(self.z_max_m),
            "height_profile_z_bin_size_m": float(self.z_bin_size_m),
            "height_profile_z_bin_count": int(self.z_bin_count),
            "height_profile_storage_z_max_m": float(self.z_max_m),
            "height_profile_active_z_min_m": float(active_min),
            "height_profile_active_z_max_m": float(active_max),
            "height_profile_active_z_bin_count": int(active_count_scalar),
            "height_profile_wall_occupied_required_bins": int(max(1, occupied_required)) if active_count_scalar > 0 else 0,
            "height_profile_debug_summary": summary,
        }
        return HeightColumnClassification(
            z_state=z_state.astype(np.uint8),
            free_bin_mask=free_bin.astype(bool),
            occupied_bin_mask=occupied_bin.astype(bool),
            conflict_bin_mask=conflict_bin.astype(bool),
            observed_bin_mask=observed_bin.astype(bool),
            active_z_bin_mask=active.astype(bool),
            vertical_free_raw_xy=vertical_free_raw.astype(bool),
            vertical_free_xy=vertical_free.astype(bool),
            wall_xy=wall.astype(bool),
            unknown_xy=unknown.astype(bool),
            free_z_bin_count_xy=free_count.astype(np.uint16),
            occupied_z_bin_count_xy=occupied_count.astype(np.uint16),
            conflict_z_bin_count_xy=conflict_count.astype(np.uint16),
            observed_z_bin_count_xy=observed_count.astype(np.uint16),
            active_z_bin_count_xy=active_count.astype(np.uint16),
            occupied_ratio_observed_xy=ratio_observed.astype(np.float32),
            occupied_ratio_active_xy=ratio_active.astype(np.float32),
            vertical_free_outside_navigation_xy=outside_nav.astype(bool),
            free_wall_conflict_xy=conflict.astype(bool),
            navigation_gate_xy=nav_gate.astype(bool),
            debug=debug,
        )

    @staticmethod
    def _increment_cells(layer: np.ndarray, rows_cols: np.ndarray) -> None:
        cells = np.asarray(rows_cols, dtype=np.int32).reshape(-1, 2)
        if cells.size == 0:
            return
        h, w = layer.shape
        valid = (cells[:, 0] >= 0) & (cells[:, 0] < h) & (cells[:, 1] >= 0) & (cells[:, 1] < w)
        cells = cells[valid]
        if cells.size == 0:
            return
        flat = cells[:, 0].astype(np.int64) * int(w) + cells[:, 1].astype(np.int64)
        counts = np.bincount(flat, minlength=int(h * w)).reshape(h, w)
        updated = np.asarray(layer, dtype=np.uint32) + counts.astype(np.uint32)
        layer[:, :] = np.minimum(updated, np.iinfo(np.uint16).max).astype(np.uint16)


def height_classification_debug_arrays(classification: HeightColumnClassification) -> dict[str, np.ndarray | object]:
    return {
        "height_profile_z_state": classification.z_state,
        "height_profile_free_bin_mask": classification.free_bin_mask,
        "height_profile_occupied_bin_mask": classification.occupied_bin_mask,
        "height_profile_conflict_bin_mask": classification.conflict_bin_mask,
        "height_profile_observed_bin_mask": classification.observed_bin_mask,
        "height_profile_active_z_bin_mask": classification.active_z_bin_mask,
        "height_profile_free_bin_count_xy": classification.free_z_bin_count_xy,
        "height_profile_occupied_bin_count_xy": classification.occupied_z_bin_count_xy,
        "height_profile_conflict_bin_count_xy": classification.conflict_z_bin_count_xy,
        "height_profile_observed_bin_count_xy": classification.observed_z_bin_count_xy,
        "height_profile_active_z_bin_count_xy": classification.active_z_bin_count_xy,
        "height_profile_occupied_ratio_observed_xy": classification.occupied_ratio_observed_xy,
        "height_profile_occupied_ratio_active_xy": classification.occupied_ratio_active_xy,
        "height_profile_vertical_free_raw_xy": classification.vertical_free_raw_xy,
        "height_profile_vertical_free_xy": classification.vertical_free_xy,
        "height_profile_vertical_free_outside_navigation_xy": classification.vertical_free_outside_navigation_xy,
        "height_profile_wall_xy": classification.wall_xy,
        "height_profile_unknown_xy": classification.unknown_xy,
        "height_profile_free_wall_conflict_xy": classification.free_wall_conflict_xy,
        "height_profile_navigation_gate_xy": classification.navigation_gate_xy,
        **classification.debug,
    }


def _bin_count(z_min_m: float, z_max_m: float, z_bin_size_m: float) -> int:
    if float(z_bin_size_m) <= 0:
        raise ValueError("z_bin_size_m must be positive")
    if float(z_max_m) <= float(z_min_m):
        raise ValueError("z_max_m must be greater than z_min_m")
    return int(np.ceil((float(z_max_m) - float(z_min_m)) / float(z_bin_size_m)))


def _valid_flat_weight_arrays(values: np.ndarray, weights: np.ndarray, total_cells: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    if arr.shape != weight.shape:
        raise ValueError("flat value and weight arrays must have the same shape")
    valid = (arr >= 0) & (arr < int(total_cells)) & np.isfinite(weight) & (weight > 0.0)
    return arr[valid], weight[valid]


def _disk(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((yy * yy + xx * xx) <= radius * radius).astype(bool)
