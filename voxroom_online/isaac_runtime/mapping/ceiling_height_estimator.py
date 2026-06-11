from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class CeilingHeightEstimatorConfig:
    enabled: bool = True
    source: str = "voxel_occupied_layer_peak"
    candidate_min_z_m: float = 1.80
    candidate_max_z_m: float = 4.00
    histogram_bin_size_m: float = 0.05
    smooth_bins: int = 2
    min_points_per_frame: int = 80
    min_occupied_voxels_per_layer: int = 1
    min_stable_frames: int = 3
    max_frame_to_frame_jump_m: float = 0.20
    ema_alpha: float = 0.20
    lock_after_stable: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "CeilingHeightEstimatorConfig":
        if isinstance(data, cls):
            return data
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class CeilingHeightEstimate:
    height_m: float | None
    stable: bool
    locked: bool
    frame_candidate_m: float | None
    active_z_max_m: float
    debug: dict[str, object] = field(default_factory=dict)


class CeilingHeightEstimator:
    def __init__(
        self,
        config: CeilingHeightEstimatorConfig | Mapping[str, object] | None = None,
        *,
        active_z_min_m: float = 0.10,
        storage_z_max_m: float = 3.20,
        active_z_max_fallback_m: float = 2.80,
        active_z_max_ceiling_ratio: float = 0.85,
        active_z_max_cap_m: float = 2.80,
    ) -> None:
        self.config = config if isinstance(config, CeilingHeightEstimatorConfig) else CeilingHeightEstimatorConfig.from_mapping(config)
        self.active_z_min_m = float(active_z_min_m)
        self.storage_z_max_m = float(storage_z_max_m)
        self.active_z_max_fallback_m = float(active_z_max_fallback_m)
        self.active_z_max_ceiling_ratio = float(active_z_max_ceiling_ratio)
        self.active_z_max_cap_m = float(active_z_max_cap_m)
        self.height_m: float | None = None
        self.locked = False
        self._stable_frames = 0
        self.last_estimate = self._make_estimate(None, None, stable=False, reason="not_updated")

    def reset(self) -> None:
        self.height_m = None
        self.locked = False
        self._stable_frames = 0
        self.last_estimate = self._make_estimate(None, None, stable=False, reason="reset")

    def update(self, rel_z_m: np.ndarray, rows_cols: np.ndarray | None = None) -> CeilingHeightEstimate:
        _ = rows_cols
        cfg = self.config
        if not bool(cfg.enabled):
            self.last_estimate = self._make_estimate(self.height_m, None, stable=False, reason="disabled")
            return self.last_estimate
        z = np.asarray(rel_z_m, dtype=np.float32).reshape(-1)
        z = z[np.isfinite(z)]
        z = z[(z >= float(cfg.candidate_min_z_m)) & (z <= float(cfg.candidate_max_z_m))]
        if int(z.size) < int(cfg.min_points_per_frame):
            self._stable_frames = 0 if self.height_m is None else self._stable_frames
            self.last_estimate = self._make_estimate(
                self.height_m,
                None,
                stable=bool(self.locked),
                reason="insufficient_candidate_points",
                candidate_point_count=int(z.size),
            )
            return self.last_estimate

        frame_candidate, hist_debug = self._frame_candidate(z)
        if frame_candidate is None:
            self._stable_frames = 0 if self.height_m is None else self._stable_frames
            self.last_estimate = self._make_estimate(
                self.height_m,
                None,
                stable=bool(self.locked),
                reason="no_histogram_mode",
                candidate_point_count=int(z.size),
                **hist_debug,
            )
            return self.last_estimate

        jump_too_large = (
            self.height_m is not None
            and abs(float(frame_candidate) - float(self.height_m)) > float(cfg.max_frame_to_frame_jump_m)
            and not bool(self.locked)
        )
        if jump_too_large:
            self._stable_frames = 0
            self.last_estimate = self._make_estimate(
                self.height_m,
                frame_candidate,
                stable=False,
                reason="frame_candidate_jump_too_large",
                candidate_point_count=int(z.size),
                **hist_debug,
            )
            return self.last_estimate

        if not self.locked:
            if self.height_m is None:
                self.height_m = float(frame_candidate)
            else:
                alpha = float(np.clip(float(cfg.ema_alpha), 0.0, 1.0))
                self.height_m = float(alpha * float(frame_candidate) + (1.0 - alpha) * float(self.height_m))
            self._stable_frames += 1
            if bool(cfg.lock_after_stable) and self._stable_frames >= int(cfg.min_stable_frames):
                self.locked = True

        stable = bool(self.locked or self._stable_frames >= int(cfg.min_stable_frames))
        self.last_estimate = self._make_estimate(
            self.height_m,
            frame_candidate,
            stable=stable,
            reason="ok",
            candidate_point_count=int(z.size),
            stable_frames=int(self._stable_frames),
            **hist_debug,
        )
        return self.last_estimate

    def update_from_occupied_layers(
        self,
        state_zyx: np.ndarray,
        z_centers_m: np.ndarray,
        *,
        occupied_value: int = 2,
    ) -> CeilingHeightEstimate:
        cfg = self.config
        if not bool(cfg.enabled):
            self.last_estimate = self._make_estimate(self.height_m, None, stable=False, reason="disabled")
            return self.last_estimate

        state = np.asarray(state_zyx)
        z_centers = np.asarray(z_centers_m, dtype=np.float32).reshape(-1)
        if state.ndim != 3 or z_centers.size != int(state.shape[0]):
            self.last_estimate = self._make_estimate(
                self.height_m,
                None,
                stable=bool(self.locked),
                reason="invalid_occupied_layer_inputs",
                ceiling_height_source="voxel_occupied_layer_peak",
                voxel_state_ndim=int(state.ndim),
                z_center_count=int(z_centers.size),
            )
            return self.last_estimate

        z_mask = (
            np.isfinite(z_centers)
            & (z_centers >= float(cfg.candidate_min_z_m))
            & (z_centers <= float(cfg.candidate_max_z_m))
        )
        layer_indices = np.flatnonzero(z_mask)
        if layer_indices.size == 0:
            self._stable_frames = 0 if self.height_m is None else self._stable_frames
            self.last_estimate = self._make_estimate(
                self.height_m,
                None,
                stable=bool(self.locked),
                reason="no_occupied_layer_candidates",
                ceiling_height_source="voxel_occupied_layer_peak",
                occupied_layer_min_z_m=float(cfg.candidate_min_z_m),
                occupied_layer_max_z_m=float(cfg.candidate_max_z_m),
            )
            return self.last_estimate

        occupied = state[layer_indices] == int(occupied_value)
        layer_counts = np.count_nonzero(occupied, axis=(1, 2)).astype(np.int64)
        max_count = int(layer_counts.max()) if layer_counts.size else 0
        min_count = max(1, int(cfg.min_occupied_voxels_per_layer))
        if max_count < min_count:
            self._stable_frames = 0 if self.height_m is None else self._stable_frames
            self.last_estimate = self._make_estimate(
                self.height_m,
                None,
                stable=bool(self.locked),
                reason="insufficient_occupied_layer_count",
                ceiling_height_source="voxel_occupied_layer_peak",
                occupied_layer_peak_count=max_count,
                occupied_layer_min_count=int(min_count),
                occupied_layer_candidate_count=int(layer_indices.size),
            )
            return self.last_estimate

        tied = np.flatnonzero(layer_counts == max_count)
        best_rel_idx = int(tied[-1])
        best_layer_idx = int(layer_indices[best_rel_idx])
        frame_candidate = float(z_centers[best_layer_idx])

        if not self.locked:
            # The occupied-layer rule is discrete by definition: the ceiling is
            # the z layer with the most occupied cells, not a smoothed EMA.
            self.height_m = float(frame_candidate)
            self._stable_frames += 1
            if bool(cfg.lock_after_stable) and self._stable_frames >= int(cfg.min_stable_frames):
                self.locked = True

        stable = bool(self.locked or self._stable_frames >= int(cfg.min_stable_frames))
        self.last_estimate = self._make_estimate(
            self.height_m,
            frame_candidate,
            stable=stable,
            reason="occupied_layer_peak",
            ceiling_height_source="voxel_occupied_layer_peak",
            candidate_point_count=max_count,
            stable_frames=int(self._stable_frames),
            occupied_layer_peak_count=max_count,
            occupied_layer_peak_z_m=frame_candidate,
            occupied_layer_peak_index=best_layer_idx,
            occupied_layer_tie_count=int(tied.size),
            occupied_layer_candidate_count=int(layer_indices.size),
            occupied_layer_min_z_m=float(cfg.candidate_min_z_m),
            occupied_layer_max_z_m=float(cfg.candidate_max_z_m),
        )
        return self.last_estimate

    def active_z_max_for_height(self, height_m: float | None) -> float:
        if height_m is None:
            active = float(self.active_z_max_cap_m) if np.isfinite(self.active_z_max_cap_m) and self.active_z_max_cap_m > 0.0 else float(self.storage_z_max_m)
        else:
            active = float(self.active_z_max_ceiling_ratio) * float(height_m)
            if np.isfinite(self.active_z_max_cap_m) and self.active_z_max_cap_m > 0.0:
                active = min(float(active), float(self.active_z_max_cap_m))
        lo = float(self.active_z_min_m) + 0.30
        hi = float(self.storage_z_max_m)
        return float(np.clip(active, lo, hi))

    def _frame_candidate(self, z: np.ndarray) -> tuple[float | None, dict[str, object]]:
        cfg = self.config
        bin_size = max(float(cfg.histogram_bin_size_m), 1e-3)
        edges = np.arange(float(cfg.candidate_min_z_m), float(cfg.candidate_max_z_m) + bin_size, bin_size, dtype=np.float32)
        if edges.size < 2:
            return None, {"histogram_bin_count": 0}
        hist, edges = np.histogram(z, bins=edges)
        smooth = hist.astype(np.float32)
        radius = max(0, int(cfg.smooth_bins))
        if radius > 0:
            kernel = np.ones((2 * radius + 1,), dtype=np.float32)
            kernel /= float(np.sum(kernel))
            smooth = np.convolve(smooth, kernel, mode="same")
        if smooth.size == 0 or float(np.max(smooth)) <= 0.0:
            return None, {"histogram_bin_count": int(smooth.size), "histogram_max_count": 0.0}
        centers = ((edges[:-1] + edges[1:]) * 0.5).astype(np.float32)
        significant = smooth >= 0.5 * float(np.max(smooth))
        if np.any(significant):
            high_half = centers >= float(np.median(centers[significant]))
            candidates = np.flatnonzero(significant & high_half)
            if candidates.size == 0:
                candidates = np.flatnonzero(significant)
        else:
            candidates = np.asarray([int(np.argmax(smooth))], dtype=np.int64)
        best = int(candidates[np.argmax(smooth[candidates])])
        close = candidates[smooth[candidates] >= 0.5 * float(smooth[best])]
        if close.size:
            best = int(close[np.argmax(centers[close])])
        return float(centers[best]), {
            "histogram_bin_count": int(smooth.size),
            "histogram_max_count": float(np.max(smooth)),
            "histogram_peak_z_m": float(centers[best]),
        }

    def _make_estimate(
        self,
        height_m: float | None,
        frame_candidate_m: float | None,
        *,
        stable: bool,
        reason: str,
        **debug: object,
    ) -> CeilingHeightEstimate:
        active = self.active_z_max_for_height(height_m)
        payload = {
            "ceiling_height_estimator_enabled": bool(self.config.enabled),
            "ceiling_height_estimate_m": None if height_m is None else float(height_m),
            "ceiling_height_stable": bool(stable),
            "ceiling_height_locked": bool(self.locked),
            "ceiling_height_frame_candidate_m": None if frame_candidate_m is None else float(frame_candidate_m),
            "height_profile_active_z_min_m": float(self.active_z_min_m),
            "height_profile_active_z_max_m": float(active),
            "height_profile_storage_z_max_m": float(self.storage_z_max_m),
            "ceiling_height_reason": str(reason),
            **debug,
        }
        return CeilingHeightEstimate(
            height_m=None if height_m is None else float(height_m),
            stable=bool(stable),
            locked=bool(self.locked),
            frame_candidate_m=None if frame_candidate_m is None else float(frame_candidate_m),
            active_z_max_m=float(active),
            debug=payload,
        )
