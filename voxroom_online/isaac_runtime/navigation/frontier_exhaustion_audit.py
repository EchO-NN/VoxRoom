from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class FrontierExhaustionAudit:
    reachable_frontier_components: int
    recovery_approach_candidates: int
    navigation_unknown_adjacent_to_free_cells: int
    stop_allowed: bool
    stop_blocked_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "reachable_frontier_components": int(self.reachable_frontier_components),
            "recovery_approach_candidates": int(self.recovery_approach_candidates),
            "navigation_unknown_adjacent_to_free_cells": int(self.navigation_unknown_adjacent_to_free_cells),
            "stop_allowed": bool(self.stop_allowed),
            "stop_blocked_reason": self.stop_blocked_reason,
        }


def _count_frontiers(frontiers: Sequence[object] | None) -> int:
    return int(len(list(frontiers or [])))


def _nav_unknown_adjacent_to_free_cells(
    *,
    navigation_unknown_mask: np.ndarray | None,
    free_mask: np.ndarray | None,
) -> int:
    if navigation_unknown_mask is None or free_mask is None:
        return 0
    unknown = np.asarray(navigation_unknown_mask, dtype=bool)
    free = np.asarray(free_mask, dtype=bool)
    if unknown.shape != free.shape:
        return 0
    if not np.any(unknown) or not np.any(free):
        return 0
    adjacent_to_free = ndimage.binary_dilation(free, structure=np.ones((3, 3), dtype=bool)).astype(bool) & ~free
    return int(np.count_nonzero(unknown & adjacent_to_free))


def audit_frontier_exhaustion(
    *,
    real_frontiers: Sequence[object] | None,
    filtered_frontiers: Sequence[object] | None,
    near_fallback_frontiers: Sequence[object] | None,
    navigation_unknown_mask: np.ndarray | None,
    free_mask: np.ndarray | None,
    frontier_near_fallback_counts_as_exploration_frontier: bool = False,
    extra_recovery_candidates: int = 0,
) -> FrontierExhaustionAudit:
    reachable = _count_frontiers(filtered_frontiers)
    if reachable <= 0:
        reachable = _count_frontiers(real_frontiers)
    near = _count_frontiers(near_fallback_frontiers)
    recovery_candidates = int(max(0, int(extra_recovery_candidates)))
    if bool(frontier_near_fallback_counts_as_exploration_frontier):
        recovery_candidates += int(near)
    nav_unknown_adjacent = _nav_unknown_adjacent_to_free_cells(
        navigation_unknown_mask=navigation_unknown_mask,
        free_mask=free_mask,
    )
    stop_allowed = bool(reachable == 0 and recovery_candidates == 0 and nav_unknown_adjacent == 0)
    reason = None
    if reachable > 0:
        reason = "stop_blocked_reachable_frontiers_available"
    elif recovery_candidates > 0:
        reason = "stop_blocked_recovery_approach_candidates_available"
    elif nav_unknown_adjacent > 0:
        reason = "stop_blocked_frontier_detector_lost_nav_unknown_boundary"
    return FrontierExhaustionAudit(
        reachable_frontier_components=int(reachable),
        recovery_approach_candidates=int(recovery_candidates),
        navigation_unknown_adjacent_to_free_cells=int(nav_unknown_adjacent),
        stop_allowed=bool(stop_allowed),
        stop_blocked_reason=reason,
    )


def frontier_exhaustion_audit_from_mapping(data: Mapping[str, object] | None) -> FrontierExhaustionAudit:
    raw = dict(data or {})
    return FrontierExhaustionAudit(
        reachable_frontier_components=int(raw.get("reachable_frontier_components", 0) or 0),
        recovery_approach_candidates=int(raw.get("recovery_approach_candidates", 0) or 0),
        navigation_unknown_adjacent_to_free_cells=int(raw.get("navigation_unknown_adjacent_to_free_cells", 0) or 0),
        stop_allowed=bool(raw.get("stop_allowed", False)),
        stop_blocked_reason=None if raw.get("stop_blocked_reason") is None else str(raw.get("stop_blocked_reason")),
    )
