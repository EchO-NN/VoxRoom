from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


HEIGHT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("low", 0.10, 0.35),
    ("robot_body", 0.35, 0.80),
    ("mid", 0.80, 1.30),
    ("upper", 1.30, 2.00),
)


@dataclass
class VerticalCellEvidence:
    free_count: int
    occupied_count: int
    observed_count: int
    unknown_count: int
    min_free_z_m: float | None
    max_free_z_m: float | None
    min_occupied_z_m: float | None
    max_occupied_z_m: float | None
    has_floor_level_free: bool
    has_robot_body_free: bool
    has_upper_free: bool
    has_any_reliable_free_0p1_2p0: bool
    has_low_obstacle_or_sill: bool
    has_window_like_high_gap: bool


@dataclass
class VerticalProfileMap:
    occupied_count: np.ndarray
    free_ray_count: np.ndarray
    observed_count: np.ndarray
    unknown_count: np.ndarray
    band_names: tuple[str, ...] = tuple(item[0] for item in HEIGHT_BANDS)
    band_ranges_m: tuple[tuple[float, float], ...] = tuple((item[1], item[2]) for item in HEIGHT_BANDS)

    @classmethod
    def zeros(cls, shape: tuple[int, int]) -> "VerticalProfileMap":
        bands = len(HEIGHT_BANDS)
        h, w = int(shape[0]), int(shape[1])
        zeros = np.zeros((bands, h, w), dtype=np.uint16)
        unknown = np.ones((bands, h, w), dtype=np.uint16)
        return cls(
            occupied_count=zeros.copy(),
            free_ray_count=zeros.copy(),
            observed_count=zeros.copy(),
            unknown_count=unknown,
        )

    @classmethod
    def from_counts(
        cls,
        *,
        occupied_count: np.ndarray,
        free_ray_count: np.ndarray | None = None,
        observed_count: np.ndarray | None = None,
        unknown_count: np.ndarray | None = None,
    ) -> "VerticalProfileMap":
        occ = np.asarray(occupied_count, dtype=np.uint16)
        if occ.ndim != 3 or occ.shape[0] != len(HEIGHT_BANDS):
            raise ValueError("occupied_count must have shape (4, H, W)")
        free = np.zeros_like(occ) if free_ray_count is None else np.asarray(free_ray_count, dtype=np.uint16)
        observed = np.asarray(observed_count, dtype=np.uint16) if observed_count is not None else np.maximum(occ, free)
        unknown = np.asarray(unknown_count, dtype=np.uint16) if unknown_count is not None else (observed == 0).astype(np.uint16)
        for name, arr in {
            "free_ray_count": free,
            "observed_count": observed,
            "unknown_count": unknown,
        }.items():
            if arr.shape != occ.shape:
                raise ValueError("%s must have shape %s" % (name, occ.shape))
        return cls(
            occupied_count=occ.copy(),
            free_ray_count=free.copy(),
            observed_count=observed.copy(),
            unknown_count=unknown.copy(),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.occupied_count.shape[1]), int(self.occupied_count.shape[2])

    def reset_shape(self, shape: tuple[int, int]) -> None:
        fresh = VerticalProfileMap.zeros(shape)
        self.occupied_count = fresh.occupied_count
        self.free_ray_count = fresh.free_ray_count
        self.observed_count = fresh.observed_count
        self.unknown_count = fresh.unknown_count

    def band_index_for_height(self, rel_z_m: float) -> int | None:
        z = float(rel_z_m)
        for idx, (_name, lo, hi) in enumerate(HEIGHT_BANDS):
            if lo <= z < hi:
                return int(idx)
        return None

    def mark_occupied_points(self, rows_cols: np.ndarray, rel_z_m: np.ndarray) -> None:
        cells = np.asarray(rows_cols, dtype=np.int32)
        heights = np.asarray(rel_z_m, dtype=np.float32).reshape(-1)
        if cells.size == 0 or len(cells) != len(heights):
            return
        for band_idx, (_name, lo, hi) in enumerate(HEIGHT_BANDS):
            mask = (heights >= float(lo)) & (heights < float(hi))
            self._increment_cells(self.occupied_count[band_idx], cells[mask])
            self._increment_cells(self.observed_count[band_idx], cells[mask])
            self._clear_unknown(self.unknown_count[band_idx], cells[mask])

    def mark_free_ray_cells(self, cells: Iterable[Tuple[int, int]], rel_z_m: float) -> None:
        band_idx = self.band_index_for_height(float(rel_z_m))
        if band_idx is None:
            return
        arr = np.asarray(list(cells), dtype=np.int32)
        self._increment_cells(self.free_ray_count[band_idx], arr)
        self._increment_cells(self.observed_count[band_idx], arr)
        self._clear_unknown(self.unknown_count[band_idx], arr)

    def to_debug_dict(self) -> dict:
        out = {}
        for idx, name in enumerate(self.band_names):
            occ = self.occupied_count[idx]
            free = self.free_ray_count[idx]
            observed = self.observed_count[idx]
            unknown = self.unknown_count[idx]
            out[name] = {
                "occupied_cells": int(np.count_nonzero(occ)),
                "free_ray_cells": int(np.count_nonzero(free)),
                "observed_cells": int(np.count_nonzero(observed)),
                "unknown_cells": int(np.count_nonzero(unknown)),
                "occupied_count_sum": int(np.sum(occ, dtype=np.uint64)),
                "free_ray_count_sum": int(np.sum(free, dtype=np.uint64)),
            }
        return out

    def reliable_free_mask(
        self,
        *,
        min_free_rays: int = 1,
        min_observed_rays: int = 1,
        band_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        indices = self._band_indices(band_names)
        free = np.asarray(self.free_ray_count[indices], dtype=np.uint32)
        observed = np.asarray(self.observed_count[indices], dtype=np.uint32)
        return (np.sum(free, axis=0) >= int(min_free_rays)) & (np.sum(observed, axis=0) >= int(min_observed_rays))

    def reliable_occupied_mask(
        self,
        *,
        min_occupied_rays: int = 1,
        min_observed_rays: int = 1,
        band_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        indices = self._band_indices(band_names)
        occupied = np.asarray(self.occupied_count[indices], dtype=np.uint32)
        observed = np.asarray(self.observed_count[indices], dtype=np.uint32)
        return (np.sum(occupied, axis=0) >= int(min_occupied_rays)) & (np.sum(observed, axis=0) >= int(min_observed_rays))

    def evidence_fields(self, *, min_free_rays: int = 1, min_observed_rays: int = 1) -> dict[str, np.ndarray]:
        floor_free = self.reliable_free_mask(min_free_rays=min_free_rays, min_observed_rays=min_observed_rays, band_names=("low",))
        robot_free = self.reliable_free_mask(min_free_rays=min_free_rays, min_observed_rays=min_observed_rays, band_names=("robot_body",))
        mid_free = self.reliable_free_mask(min_free_rays=min_free_rays, min_observed_rays=min_observed_rays, band_names=("mid",))
        upper_free = self.reliable_free_mask(min_free_rays=min_free_rays, min_observed_rays=min_observed_rays, band_names=("upper",))
        any_free = floor_free | robot_free | mid_free | upper_free
        low_occ = self.reliable_occupied_mask(min_occupied_rays=1, min_observed_rays=min_observed_rays, band_names=("low",))
        robot_occ = self.reliable_occupied_mask(min_occupied_rays=1, min_observed_rays=min_observed_rays, band_names=("robot_body",))
        return {
            "has_floor_level_free": floor_free,
            "has_robot_body_free": robot_free,
            "has_upper_free": upper_free,
            "has_any_reliable_free_0p1_2p0": any_free,
            "has_low_obstacle_or_sill": low_occ & ~robot_free,
            "has_window_like_high_gap": (mid_free | upper_free) & ~(floor_free & robot_free),
            "has_floor_or_robot_free": floor_free | robot_free,
            "has_floor_and_robot_free": floor_free & robot_free,
            "has_mid_or_upper_free": mid_free | upper_free,
        }

    def cell_evidence(self, row: int, col: int, *, min_free_rays: int = 1, min_observed_rays: int = 1) -> VerticalCellEvidence:
        r, c = int(row), int(col)
        h, w = self.shape
        if not (0 <= r < h and 0 <= c < w):
            raise IndexError("vertical profile cell out of bounds")
        free_counts = np.asarray(self.free_ray_count[:, r, c], dtype=np.uint32)
        occ_counts = np.asarray(self.occupied_count[:, r, c], dtype=np.uint32)
        observed_counts = np.asarray(self.observed_count[:, r, c], dtype=np.uint32)
        unknown_counts = np.asarray(self.unknown_count[:, r, c], dtype=np.uint32)
        fields = self.evidence_fields(min_free_rays=min_free_rays, min_observed_rays=min_observed_rays)
        free_z = self._z_range_for_counts(free_counts)
        occ_z = self._z_range_for_counts(occ_counts)
        return VerticalCellEvidence(
            free_count=int(np.sum(free_counts, dtype=np.uint64)),
            occupied_count=int(np.sum(occ_counts, dtype=np.uint64)),
            observed_count=int(np.sum(observed_counts, dtype=np.uint64)),
            unknown_count=int(np.sum(unknown_counts, dtype=np.uint64)),
            min_free_z_m=free_z[0],
            max_free_z_m=free_z[1],
            min_occupied_z_m=occ_z[0],
            max_occupied_z_m=occ_z[1],
            has_floor_level_free=bool(fields["has_floor_level_free"][r, c]),
            has_robot_body_free=bool(fields["has_robot_body_free"][r, c]),
            has_upper_free=bool(fields["has_upper_free"][r, c]),
            has_any_reliable_free_0p1_2p0=bool(fields["has_any_reliable_free_0p1_2p0"][r, c]),
            has_low_obstacle_or_sill=bool(fields["has_low_obstacle_or_sill"][r, c]),
            has_window_like_high_gap=bool(fields["has_window_like_high_gap"][r, c]),
        )

    def _band_indices(self, band_names: Sequence[str] | None) -> list[int]:
        if band_names is None:
            return list(range(len(self.band_names)))
        lookup = {str(name): idx for idx, name in enumerate(self.band_names)}
        return [int(lookup[str(name)]) for name in band_names]

    def _z_range_for_counts(self, counts: np.ndarray) -> tuple[float | None, float | None]:
        active = [idx for idx, value in enumerate(counts) if int(value) > 0]
        if not active:
            return None, None
        lo = min(float(self.band_ranges_m[idx][0]) for idx in active)
        hi = max(float(self.band_ranges_m[idx][1]) for idx in active)
        return lo, hi

    @staticmethod
    def _increment_cells(layer: np.ndarray, rows_cols: np.ndarray) -> None:
        cells = np.asarray(rows_cols, dtype=np.int32)
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

    @staticmethod
    def _clear_unknown(layer: np.ndarray, rows_cols: np.ndarray) -> None:
        cells = np.asarray(rows_cols, dtype=np.int32)
        if cells.size == 0:
            return
        h, w = layer.shape
        valid = (cells[:, 0] >= 0) & (cells[:, 0] < h) & (cells[:, 1] >= 0) & (cells[:, 1] < w)
        cells = cells[valid]
        if cells.size:
            layer[cells[:, 0], cells[:, 1]] = 0


def ensure_vertical_profile(value: VerticalProfileMap | None, shape: tuple[int, int]) -> VerticalProfileMap:
    if value is not None:
        return value
    return VerticalProfileMap.zeros(shape)


def band_index(name: str) -> int:
    lookup = {band_name: idx for idx, (band_name, _lo, _hi) in enumerate(HEIGHT_BANDS)}
    return int(lookup[str(name)])
