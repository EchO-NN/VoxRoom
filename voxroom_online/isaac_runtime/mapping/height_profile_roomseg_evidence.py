from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import ndimage

from voxroom_online.isaac_runtime.mapping.height_column_profile import (
    HeightColumnClassification,
    HeightColumnProfileConfig,
    HeightColumnProfileMap,
    height_classification_debug_arrays,
)
from voxroom_online.isaac_runtime.mapping.online_roomseg.utils import conn, label_components


@dataclass
class HeightProfileRoomsegEvidenceConfig:
    enabled: bool = True
    resolution_m: float = 0.05
    height_profile: HeightColumnProfileConfig = field(default_factory=HeightColumnProfileConfig)
    free_remove_island_max_area_cells: int = 4
    free_close_radius_cells: int = 0
    wall_remove_island_max_area_cells: int = 0
    wall_micro_close_radius_cells: int = 0
    wall_min_component_cells: int = 1
    suppress_free_on_navigation_obstacle: bool = True
    promote_navigation_obstacle_to_wall: bool = False
    fill_unknown_as_wall: bool = False
    fill_unknown_as_free: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "HeightProfileRoomsegEvidenceConfig":
        if isinstance(data, cls):
            cfg = data
            for key, value in overrides.items():
                if value is not None and hasattr(cfg, key):
                    setattr(cfg, key, value)
            return cfg
        raw = dict(data or {})
        if "height_profile" in raw:
            raw["height_profile"] = HeightColumnProfileConfig.from_mapping(raw.get("height_profile"))
        for key, value in overrides.items():
            if value is not None:
                raw[key] = value
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class HeightProfileRoomsegEvidence:
    free_clean: np.ndarray
    wall_clean: np.ndarray
    unknown_clean: np.ndarray
    vertical_observed_xy: np.ndarray
    classification: HeightColumnClassification
    debug: dict[str, object] = field(default_factory=dict)


def build_height_profile_roomseg_evidence(
    *,
    height_profile: HeightColumnProfileMap,
    navigation_free_mask: np.ndarray,
    navigation_obstacle_mask: np.ndarray,
    unknown_mask_from_navigation: np.ndarray,
    resolution_m: float,
    config: HeightProfileRoomsegEvidenceConfig | Mapping[str, object] | None = None,
) -> HeightProfileRoomsegEvidence:
    cfg = config if isinstance(config, HeightProfileRoomsegEvidenceConfig) else HeightProfileRoomsegEvidenceConfig.from_mapping(config)
    cfg.resolution_m = float(resolution_m)
    cfg.height_profile.active_z_min_m = float(getattr(height_profile, "active_z_min_m", cfg.height_profile.active_z_min_m))
    if getattr(height_profile, "active_z_max_m", None) is not None:
        cfg.height_profile.active_z_max_m = float(getattr(height_profile, "active_z_max_m"))
    classification = height_profile.classify_columns(
        navigation_free_mask=np.asarray(navigation_free_mask, dtype=bool),
        cfg=cfg.height_profile,
    )
    free = np.asarray(classification.vertical_free_xy, dtype=bool).copy()
    wall = np.asarray(classification.wall_xy, dtype=bool).copy()
    observed_xy = np.asarray(classification.observed_z_bin_count_xy > 0, dtype=bool)
    nav_obstacle = np.asarray(navigation_obstacle_mask, dtype=bool)
    nav_unknown = np.asarray(unknown_mask_from_navigation, dtype=bool)
    if nav_obstacle.shape != free.shape:
        raise ValueError("navigation_obstacle_mask must match height profile shape")

    wall_before_nav = wall.copy()
    nav_obstacle_overlap_free = free & nav_obstacle
    if bool(cfg.suppress_free_on_navigation_obstacle):
        free &= ~nav_obstacle
    promoted_nav_obstacle = np.zeros_like(wall, dtype=bool)
    if bool(cfg.promote_navigation_obstacle_to_wall):
        promoted_nav_obstacle = nav_obstacle & ~wall
        wall |= nav_obstacle
    free = _remove_small_true_components(free, max_area_cells=int(cfg.free_remove_island_max_area_cells), connectivity=4, remove_leq=True)
    if int(cfg.free_close_radius_cells) > 0:
        free = ndimage.binary_closing(free, structure=_disk(int(cfg.free_close_radius_cells))).astype(bool)
        free &= ~wall

    wall = _remove_small_true_components(wall, max_area_cells=int(cfg.wall_remove_island_max_area_cells), connectivity=8, remove_leq=True)
    if int(cfg.wall_min_component_cells) > 1:
        wall = _remove_small_true_components(wall, max_area_cells=int(cfg.wall_min_component_cells) - 1, connectivity=8, remove_leq=True)
    if int(cfg.wall_micro_close_radius_cells) > 0:
        closed = ndimage.binary_closing(wall, structure=_disk(int(cfg.wall_micro_close_radius_cells))).astype(bool)
        strict_support = ndimage.binary_dilation(wall_before_nav, structure=_disk(1)).astype(bool)
        high_ratio = np.asarray(classification.occupied_ratio_active_xy, dtype=np.float32) >= max(0.0, float(cfg.height_profile.wall_occupied_ratio_min) - 0.05)
        wall |= closed & ~free & observed_xy & strict_support & high_ratio

    if bool(cfg.fill_unknown_as_wall) or bool(cfg.fill_unknown_as_free):
        raise ValueError("height-profile roomseg evidence must preserve unknown; fill_unknown_as_wall/free are forbidden")
    unknown = ~(free | wall)
    debug_summary = {
        "free_clean_cells": int(np.count_nonzero(free)),
        "wall_clean_cells": int(np.count_nonzero(wall)),
        "unknown_clean_cells": int(np.count_nonzero(unknown)),
        "vertical_observed_cells": int(np.count_nonzero(observed_xy)),
        "navigation_obstacle_cells": int(np.count_nonzero(nav_obstacle)),
        "nav_obstacle_overlap_free_cells": int(np.count_nonzero(nav_obstacle_overlap_free)),
        "nav_obstacle_suppressed_free_cells": int(np.count_nonzero(nav_obstacle_overlap_free)) if bool(cfg.suppress_free_on_navigation_obstacle) else 0,
        "nav_obstacle_added_wall_cells": int(np.count_nonzero(promoted_nav_obstacle)),
        "promote_navigation_obstacle_to_wall": bool(cfg.promote_navigation_obstacle_to_wall),
        "promote_navigation_obstacle_to_wall_warning": (
            "navigation obstacle promotion violates strict height-profile wall evidence"
            if bool(cfg.promote_navigation_obstacle_to_wall)
            else ""
        ),
        "navigation_unknown_cells": int(np.count_nonzero(nav_unknown)),
        "unknown_preserved_cells": int(np.count_nonzero(unknown & nav_unknown)),
    }
    debug = {
        **height_classification_debug_arrays(classification),
        "height_evidence_free_clean": free.astype(bool),
        "height_evidence_wall_clean": wall.astype(bool),
        "height_evidence_unknown_clean": unknown.astype(bool),
        "height_evidence_vertical_observed_xy": observed_xy.astype(bool),
        "height_evidence_nav_obstacle_overlap_free": nav_obstacle_overlap_free.astype(bool),
        "height_evidence_navigation_obstacle_promoted_wall": promoted_nav_obstacle.astype(bool),
        "height_evidence_wall_after_navigation_obstacle": wall.astype(bool),
        "height_evidence_free_after_navigation_gate": free.astype(bool),
        "height_evidence_unknown_preserved_cells": int(np.count_nonzero(unknown)),
        "height_evidence_debug_summary": debug_summary,
    }
    return HeightProfileRoomsegEvidence(
        free_clean=free.astype(bool),
        wall_clean=wall.astype(bool),
        unknown_clean=unknown.astype(bool),
        vertical_observed_xy=observed_xy.astype(bool),
        classification=classification,
        debug=debug,
    )


def _remove_small_true_components(mask: np.ndarray, *, max_area_cells: int, connectivity: int, remove_leq: bool) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if int(max_area_cells) < 0 or not np.any(src):
        return src.copy()
    labels, count = label_components(src, connectivity)
    out = src.copy()
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        area = int(np.count_nonzero(comp))
        if (area <= int(max_area_cells)) if remove_leq else (area < int(max_area_cells)):
            out[comp] = False
    return out.astype(bool)


def _disk(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return conn(4).astype(bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((yy * yy + xx * xx) <= radius * radius).astype(bool)
