from __future__ import annotations

from typing import Mapping

import numpy as np


EVIDENCE_FUSION_MODE = "vertical_free_nav_obstacle_wall"


def fuse_vertical_profile_with_navigation_obstacles(
    *,
    vertical_free: np.ndarray,
    vertical_observed: np.ndarray,
    nav_raw_obstacle: np.ndarray,
    static_structural_occupied: np.ndarray | None = None,
    navigation_free: np.ndarray | None = None,
    inflated_obstacle: np.ndarray | None = None,
    config: object | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fuse roomseg evidence without letting navigation free create rooms.

    Priority is intentionally strict:

    1. vertical free wins and always remains roomseg free;
    2. vertical observed non-free remains occupied/wall-like;
    3. vertical unknown plus raw/static obstacle is rescued as occupied;
    4. everything else remains unknown.

    Inflated obstacles and navigation free are accepted only for audit/debug and
    are never used as positive roomseg wall/free evidence.
    """

    vf = np.asarray(vertical_free, dtype=bool)
    vo = np.asarray(vertical_observed, dtype=bool)
    raw_obstacle = np.asarray(nav_raw_obstacle, dtype=bool)
    if vf.shape != vo.shape or vf.shape != raw_obstacle.shape:
        raise ValueError("vertical_free, vertical_observed, and nav_raw_obstacle must have the same HxW shape")

    cfg = _config_dict(config)
    enabled = bool(cfg.get("enabled", True))
    vertical_free_priority = bool(cfg.get("vertical_free_priority", True))
    use_nav_obstacle = bool(cfg.get("use_navigation_obstacle_for_vertical_unknown", True))
    use_static = bool(cfg.get("use_static_structural_occupied", True))
    use_navigation_free_for_roomseg_free = bool(cfg.get("use_navigation_free_for_roomseg_free", False))
    use_inflated_obstacle = bool(cfg.get("use_inflated_obstacle", False))
    use_depth_valid_as_wall = bool(cfg.get("use_depth_valid_as_wall", False))

    static_occ = _optional_bool(static_structural_occupied, vf.shape)
    nav_free = _optional_bool(navigation_free, vf.shape)
    inflated = _optional_bool(inflated_obstacle, vf.shape)

    if not enabled:
        obstacle_source = raw_obstacle.copy()
        if use_static:
            obstacle_source |= static_occ
        initial_free = vf.copy()
        initial_occupied = vo & ~initial_free
        initial_unknown = ~(initial_free | initial_occupied)
        overlay_candidate = np.zeros_like(vf, dtype=bool)
        overlay_accepted = np.zeros_like(vf, dtype=bool)
    else:
        obstacle_source = raw_obstacle.copy()
        if use_static:
            obstacle_source |= static_occ
        if use_inflated_obstacle:
            # This is deliberately unsupported for the benchmark path. Keep the
            # field visible for audits, but do not mix it into obstacle_source.
            obstacle_source = obstacle_source.copy()

        initial_free = vf.copy()
        if use_navigation_free_for_roomseg_free:
            # Kept only for explicit ablation/debug configs; defaults and strict
            # tests require this to stay false.
            initial_free |= nav_free
        if vertical_free_priority:
            initial_free |= vf

        vertical_unknown_before_overlay = ~vo & ~vf
        overlay_candidate = vertical_unknown_before_overlay & obstacle_source & ~vf
        overlay_accepted = overlay_candidate if use_nav_obstacle else np.zeros_like(vf, dtype=bool)
        vertical_nonfree_observed = vo & ~initial_free
        initial_occupied = vertical_nonfree_observed | overlay_accepted
        if vertical_free_priority:
            initial_occupied &= ~vf
        initial_unknown = ~(initial_free | initial_occupied)

    vertical_unknown_before_overlay = ~vo & ~vf
    walls_rescued = overlay_accepted.copy()
    vertical_free_over_nav_obstacle = vf & obstacle_source
    nav_obstacle_still_unknown = obstacle_source & initial_unknown

    debug = {
        "evidence_fusion_enabled": bool(enabled),
        "evidence_fusion_mode": str(cfg.get("mode", EVIDENCE_FUSION_MODE) or EVIDENCE_FUSION_MODE),
        "vertical_free_priority": bool(vertical_free_priority),
        "use_navigation_obstacle_for_vertical_unknown": bool(use_nav_obstacle),
        "use_navigation_free_for_roomseg_free": bool(use_navigation_free_for_roomseg_free),
        "use_inflated_obstacle": bool(use_inflated_obstacle),
        "use_depth_valid_as_wall": bool(use_depth_valid_as_wall),
        "use_static_structural_occupied": bool(use_static),
        "vertical_free_cells": int(np.count_nonzero(vf)),
        "vertical_observed_cells": int(np.count_nonzero(vo)),
        "vertical_unknown_before_overlay_cells": int(np.count_nonzero(vertical_unknown_before_overlay)),
        "nav_raw_obstacle_cells": int(np.count_nonzero(raw_obstacle)),
        "static_structural_occupied_cells": int(np.count_nonzero(static_occ)),
        "inflated_obstacle_cells_seen_but_ignored": int(np.count_nonzero(inflated)),
        "navigation_free_cells_seen_but_not_used_as_roomseg_free": int(np.count_nonzero(nav_free)),
        "nav_obstacle_overlay_candidate_cells": int(np.count_nonzero(overlay_candidate)),
        "nav_obstacle_overlay_accepted_cells": int(np.count_nonzero(overlay_accepted)),
        "walls_rescued_from_unknown_cells": int(np.count_nonzero(walls_rescued)),
        "vertical_free_over_nav_obstacle_cells": int(np.count_nonzero(vertical_free_over_nav_obstacle)),
        "nav_obstacle_still_unknown_after_fusion_cells": int(np.count_nonzero(nav_obstacle_still_unknown)),
        "initial_roomseg_free_cells": int(np.count_nonzero(initial_free)),
        "initial_roomseg_occupied_cells": int(np.count_nonzero(initial_occupied)),
        "initial_roomseg_unknown_cells": int(np.count_nonzero(initial_unknown)),
        "occupied_source_audit": {
            "occupied_param_source": "mapper.grid.occupied/raw_obstacle plus roomseg_static_structural_occupied",
            "is_inflated_obstacle": False,
            "inflated_obstacle_ignored": True,
            "has_static_structural_source": bool(np.any(static_occ)),
            "static_wall_written_to_grid_occupied": False,
            "static_wall_written_to_vertical_profile": False,
        },
    }
    return {
        "initial_roomseg_free": initial_free.astype(bool),
        "initial_roomseg_occupied": initial_occupied.astype(bool),
        "initial_roomseg_unknown": initial_unknown.astype(bool),
        "vertical_unknown_before_overlay": vertical_unknown_before_overlay.astype(bool),
        "nav_raw_obstacle": raw_obstacle.astype(bool),
        "roomseg_static_structural_occupied": static_occ.astype(bool),
        "nav_obstacle_overlay_candidate": overlay_candidate.astype(bool),
        "nav_obstacle_overlay_accepted": overlay_accepted.astype(bool),
        "walls_rescued_from_unknown": walls_rescued.astype(bool),
        "vertical_free_over_nav_obstacle": vertical_free_over_nav_obstacle.astype(bool),
        "nav_obstacle_still_unknown_after_fusion": nav_obstacle_still_unknown.astype(bool),
        "initial_roomseg_free_after_fusion": initial_free.astype(bool),
        "initial_roomseg_occupied_after_fusion": initial_occupied.astype(bool),
        "initial_roomseg_unknown_after_fusion": initial_unknown.astype(bool),
        "debug": debug,
    }


def _optional_bool(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        raise ValueError("optional roomseg evidence mask has shape %s, expected %s" % (arr.shape, shape))
    return arr


def _config_dict(config: object | Mapping[str, object] | None) -> dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "roomseg_evidence_fusion"):
        return dict(getattr(config, "roomseg_evidence_fusion") or {})
    return {
        "enabled": getattr(config, "roomseg_evidence_fusion_enabled", True),
        "mode": getattr(config, "roomseg_evidence_fusion_mode", EVIDENCE_FUSION_MODE),
        "vertical_free_priority": getattr(config, "roomseg_evidence_fusion_vertical_free_priority", True),
        "use_navigation_obstacle_for_vertical_unknown": getattr(
            config,
            "roomseg_use_navigation_obstacle_for_vertical_unknown",
            True,
        ),
        "use_navigation_free_for_roomseg_free": getattr(config, "roomseg_use_navigation_free_for_roomseg_free", False),
        "use_inflated_obstacle": getattr(config, "roomseg_use_inflated_obstacle", False),
        "use_depth_valid_as_wall": getattr(config, "roomseg_use_depth_valid_as_wall", False),
        "use_static_structural_occupied": getattr(config, "roomseg_use_static_structural_occupied", True),
    }
