from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.navigation.astar import GridAStarPlanner, astar_distance_map

GridCell = Tuple[int, int]


@dataclass
class FrontierRecoveryConfig:
    enabled: bool = True
    local_search_radius_m: float = 1.50
    global_search_enabled: bool = True
    global_max_frontier_distance_m: float = 3.00
    min_clearance_m: float = 0.18
    min_approach_improvement_m: float = 0.20
    allow_current_as_partial_arrival: bool = False
    current_partial_arrival_radius_m: float = 0.35
    current_partial_arrival_min_motion_m: float = 0.30
    current_partial_arrival_min_steps_since_selection: int = 6
    max_recovery_attempts_per_frontier: int = 1
    max_debug_candidates: int = 64


@dataclass
class FrontierRecoveryTarget:
    target_cell: GridCell | None
    reachable: bool
    path: List[GridCell]
    mode: str
    reason: str
    frontier_center_grid: GridCell | None
    original_target_grid: GridCell | None
    distance_to_frontier_m: float
    current_distance_to_frontier_m: float
    improvement_m: float
    target_clearance_m: float
    path_length_m: float
    candidates_evaluated: int
    candidates_rejected_by_clearance: int
    candidates_rejected_by_unreachable: int
    debug_candidates: List[dict] = field(default_factory=list)

    def metadata(self) -> dict[str, object]:
        return {
            "frontier_target_planning_stage": "recovery" if self.reachable else "recovery_failed",
            "frontier_unreachable_recovery": bool(self.reachable),
            "frontier_unreachable_recovery_mode": str(self.mode),
            "frontier_recovery_target_grid": (
                [int(self.target_cell[0]), int(self.target_cell[1])] if self.target_cell is not None else None
            ),
            "frontier_recovery_reason": str(self.reason),
            "frontier_recovery_target_clearance_m": None if not math.isfinite(self.target_clearance_m) else float(self.target_clearance_m),
            "frontier_recovery_distance_to_frontier_m": None if not math.isfinite(self.distance_to_frontier_m) else float(self.distance_to_frontier_m),
            "frontier_recovery_current_distance_to_frontier_m": (
                None if not math.isfinite(self.current_distance_to_frontier_m) else float(self.current_distance_to_frontier_m)
            ),
            "frontier_recovery_improvement_m": None if not math.isfinite(self.improvement_m) else float(self.improvement_m),
            "frontier_recovery_path_length_m": None if not math.isfinite(self.path_length_m) else float(self.path_length_m),
            "frontier_recovery_candidates_evaluated": int(self.candidates_evaluated),
            "frontier_recovery_candidates_rejected_by_clearance": int(self.candidates_rejected_by_clearance),
            "frontier_recovery_candidates_rejected_by_unreachable": int(self.candidates_rejected_by_unreachable),
            "frontier_recovery_debug_candidates": list(self.debug_candidates),
        }


def find_best_reachable_frontier_approach(
    *,
    current_grid: GridCell,
    frontier_center: GridCell,
    frontier_members: Sequence[GridCell],
    original_target: GridCell | None,
    traversible: np.ndarray,
    clearance_m: np.ndarray,
    planner: GridAStarPlanner,
    resolution_m: float,
    config: FrontierRecoveryConfig,
    selected_step: int | None = None,
    current_step: int | None = None,
    selected_grid: GridCell | None = None,
) -> FrontierRecoveryTarget:
    nav = np.asarray(traversible, dtype=bool)
    clearance = np.asarray(clearance_m, dtype=np.float32)
    if nav.ndim != 2 or clearance.shape != nav.shape:
        raise ValueError("traversible and clearance_m must be same-shaped 2D arrays")
    resolution = max(float(resolution_m), 1e-6)
    cfg = config
    center = _cell(frontier_center)
    current = _cell(current_grid)
    members = _unique_cells(list(frontier_members or []) + [center])
    original = None if original_target is None else _cell(original_target)
    distance_to_frontier = _distance_to_frontier_map(nav.shape, members, resolution)
    current_d_frontier = _safe_map_value(distance_to_frontier, current, default=float("inf"))

    if not bool(cfg.enabled):
        return _none_target(
            "frontier_recovery_disabled",
            center,
            original,
            current_d_frontier,
            candidates_evaluated=0,
        )

    local = _evaluate_local_candidates(
        current=current,
        center=center,
        members=members,
        original=original,
        nav=nav,
        clearance=clearance,
        distance_to_frontier=distance_to_frontier,
        planner=planner,
        resolution_m=resolution,
        config=cfg,
        current_d_frontier=current_d_frontier,
    )
    if local.reachable:
        return local
    if bool(cfg.global_search_enabled):
        global_target = _evaluate_global_candidates(
            current=current,
            center=center,
            original=original,
            nav=nav,
            clearance=clearance,
            distance_to_frontier=distance_to_frontier,
            planner=planner,
            resolution_m=resolution,
            config=cfg,
            current_d_frontier=current_d_frontier,
            seed_debug=list(local.debug_candidates),
            seed_counts=(
                int(local.candidates_evaluated),
                int(local.candidates_rejected_by_clearance),
                int(local.candidates_rejected_by_unreachable),
            ),
        )
        if global_target.reachable:
            return global_target
        local = global_target
    partial_allowed, partial_reject = _current_cell_partial_arrival_allowed(
        current=current,
        selected_grid=selected_grid,
        current_d_frontier=current_d_frontier,
        current_step=current_step,
        selected_step=selected_step,
        resolution_m=resolution,
        config=cfg,
    )
    if partial_allowed:
        clear = _safe_map_value(clearance, current, default=0.0)
        return FrontierRecoveryTarget(
            target_cell=current,
            reachable=True,
            path=[current],
            mode="current_cell_partial_arrival",
            reason="frontier_current_cell_is_best_reachable_approach",
            frontier_center_grid=center,
            original_target_grid=original,
            distance_to_frontier_m=float(current_d_frontier),
            current_distance_to_frontier_m=float(current_d_frontier),
            improvement_m=0.0,
            target_clearance_m=float(clear),
            path_length_m=0.0,
            candidates_evaluated=int(local.candidates_evaluated),
            candidates_rejected_by_clearance=int(local.candidates_rejected_by_clearance),
            candidates_rejected_by_unreachable=int(local.candidates_rejected_by_unreachable),
            debug_candidates=list(local.debug_candidates[: int(cfg.max_debug_candidates)]),
        )
    if bool(cfg.allow_current_as_partial_arrival) and partial_reject:
        debug = list(local.debug_candidates[: int(cfg.max_debug_candidates)])
        if len(debug) < int(cfg.max_debug_candidates):
            debug.append(
                {
                    "cell": [int(current[0]), int(current[1])],
                    "mode": "current_cell_partial_arrival",
                    "distance_to_frontier_m": None if not math.isfinite(current_d_frontier) else float(current_d_frontier),
                    "reject_reason": str(partial_reject),
                }
            )
        return _none_target(
            "frontier_no_reachable_approach_target",
            center,
            original,
            current_d_frontier,
            candidates_evaluated=int(local.candidates_evaluated),
            candidates_rejected_by_clearance=int(local.candidates_rejected_by_clearance),
            candidates_rejected_by_unreachable=int(local.candidates_rejected_by_unreachable),
            debug_candidates=debug,
        )
    return local


def _current_cell_partial_arrival_allowed(
    *,
    current: GridCell,
    selected_grid: GridCell | None,
    current_d_frontier: float,
    current_step: int | None,
    selected_step: int | None,
    resolution_m: float,
    config: FrontierRecoveryConfig,
) -> tuple[bool, str]:
    if not bool(config.allow_current_as_partial_arrival):
        return False, "current_cell_partial_arrival_disabled"
    if not math.isfinite(current_d_frontier):
        return False, "current_cell_partial_arrival_distance_unknown"
    if float(current_d_frontier) > float(config.current_partial_arrival_radius_m):
        return False, "current_cell_partial_arrival_too_far"
    if current_step is None or selected_step is None:
        return False, "current_cell_partial_arrival_missing_step"
    age = int(current_step) - int(selected_step)
    if age < int(config.current_partial_arrival_min_steps_since_selection):
        return False, "current_cell_partial_arrival_too_early"
    if selected_grid is None:
        return False, "current_cell_partial_arrival_missing_selected_grid"
    motion_m = _cell_l2(current, _cell(selected_grid)) * float(resolution_m)
    if motion_m + 1e-6 < float(config.current_partial_arrival_min_motion_m):
        return False, "current_cell_partial_arrival_insufficient_motion"
    return True, ""


def _evaluate_local_candidates(
    *,
    current: GridCell,
    center: GridCell,
    members: Sequence[GridCell],
    original: GridCell | None,
    nav: np.ndarray,
    clearance: np.ndarray,
    distance_to_frontier: np.ndarray,
    planner: GridAStarPlanner,
    resolution_m: float,
    config: FrontierRecoveryConfig,
    current_d_frontier: float,
) -> FrontierRecoveryTarget:
    radius_cells = max(1, int(math.ceil(float(config.local_search_radius_m) / max(float(resolution_m), 1e-6))))
    seeds = _unique_cells(list(members) + ([original] if original is not None else []))
    candidates: dict[GridCell, str] = {}
    h, w = nav.shape
    for seed in seeds:
        sr, sc = seed
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                rr, cc = sr + dr, sc + dc
                if 0 <= rr < h and 0 <= cc < w:
                    source = "member_neighbor_approach" if seed in members else "original_target_approach"
                    candidates.setdefault((rr, cc), source)
    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            _safe_map_value(distance_to_frontier, item[0], default=float("inf")),
            _cell_l2(item[0], center),
        ),
    )
    return _select_candidate(
        current=current,
        center=center,
        original=original,
        candidate_items=ordered,
        nav=nav,
        clearance=clearance,
        distance_to_frontier=distance_to_frontier,
        dist_from_current=None,
        planner=planner,
        resolution_m=resolution_m,
        config=config,
        current_d_frontier=current_d_frontier,
        default_mode="local_approach_cell",
    )


def _evaluate_global_candidates(
    *,
    current: GridCell,
    center: GridCell,
    original: GridCell | None,
    nav: np.ndarray,
    clearance: np.ndarray,
    distance_to_frontier: np.ndarray,
    planner: GridAStarPlanner,
    resolution_m: float,
    config: FrontierRecoveryConfig,
    current_d_frontier: float,
    seed_debug: Sequence[dict],
    seed_counts: tuple[int, int, int],
) -> FrontierRecoveryTarget:
    dist_from_current = astar_distance_map(nav, current, resolution_m, allow_diagonal=planner.allow_diagonal)
    max_d = float(config.global_max_frontier_distance_m)
    valid = np.isfinite(dist_from_current) & nav & (clearance + 1e-6 >= float(config.min_clearance_m))
    near = valid & (distance_to_frontier <= max_d + 1e-6)
    if not np.any(near):
        near = valid
    coords = np.argwhere(near)
    if coords.size == 0:
        return _none_target(
            "frontier_no_reachable_approach_target",
            center,
            original,
            current_d_frontier,
            candidates_evaluated=int(seed_counts[0]),
            candidates_rejected_by_clearance=int(seed_counts[1]),
            candidates_rejected_by_unreachable=int(seed_counts[2]),
            debug_candidates=list(seed_debug[: int(config.max_debug_candidates)]),
        )
    d_frontier = distance_to_frontier[coords[:, 0], coords[:, 1]]
    d_current = dist_from_current[coords[:, 0], coords[:, 1]]
    clear = clearance[coords[:, 0], coords[:, 1]]
    improvement = float(current_d_frontier) - d_frontier
    score = -2.0 * d_frontier - 0.25 * d_current + 0.5 * clear
    eligible = improvement + 1e-6 >= float(config.min_approach_improvement_m)
    if not np.any(eligible):
        return _none_target(
            "frontier_no_recovery_candidate_with_required_improvement",
            center,
            original,
            current_d_frontier,
            candidates_evaluated=int(seed_counts[0] + coords.shape[0]),
            candidates_rejected_by_clearance=int(seed_counts[1]),
            candidates_rejected_by_unreachable=int(seed_counts[2]),
            debug_candidates=list(seed_debug[: int(config.max_debug_candidates)]),
        )
    eligible_idx = np.flatnonzero(eligible)
    ranked = eligible_idx[np.argsort(score[eligible_idx])[::-1]]
    candidate_items = [
        ((int(coords[idx, 0]), int(coords[idx, 1])), "reachable_region_nearest_frontier")
        for idx in ranked[: max(1, min(512, ranked.size))]
    ]
    selected = _select_candidate(
        current=current,
        center=center,
        original=original,
        candidate_items=candidate_items,
        nav=nav,
        clearance=clearance,
        distance_to_frontier=distance_to_frontier,
        dist_from_current=dist_from_current,
        planner=planner,
        resolution_m=resolution_m,
        config=config,
        current_d_frontier=current_d_frontier,
        default_mode="reachable_region_nearest_frontier",
        initial_debug=seed_debug,
        initial_counts=seed_counts,
    )
    if selected.reachable:
        return selected
    return _none_target(
        selected.reason,
        center,
        original,
        current_d_frontier,
        candidates_evaluated=int(seed_counts[0] + coords.shape[0]),
        candidates_rejected_by_clearance=int(seed_counts[1] + selected.candidates_rejected_by_clearance),
        candidates_rejected_by_unreachable=int(seed_counts[2] + selected.candidates_rejected_by_unreachable),
        debug_candidates=selected.debug_candidates,
    )


def _select_candidate(
    *,
    current: GridCell,
    center: GridCell,
    original: GridCell | None,
    candidate_items: Sequence[tuple[GridCell, str]],
    nav: np.ndarray,
    clearance: np.ndarray,
    distance_to_frontier: np.ndarray,
    dist_from_current: np.ndarray | None,
    planner: GridAStarPlanner,
    resolution_m: float,
    config: FrontierRecoveryConfig,
    current_d_frontier: float,
    default_mode: str,
    initial_debug: Sequence[dict] | None = None,
    initial_counts: tuple[int, int, int] = (0, 0, 0),
) -> FrontierRecoveryTarget:
    debug = list(initial_debug or [])
    evaluated, rejected_clearance, rejected_unreachable = (int(v) for v in initial_counts)
    accepted: list[tuple[float, GridCell, str, float, float, float, float, list[GridCell]]] = []
    h, w = nav.shape
    max_debug = int(config.max_debug_candidates)
    for cell, mode in candidate_items:
        evaluated += 1
        row, col = int(cell[0]), int(cell[1])
        clear = _safe_map_value(clearance, (row, col), default=0.0)
        d_frontier = _safe_map_value(distance_to_frontier, (row, col), default=float("inf"))
        improvement = float(current_d_frontier) - float(d_frontier)
        reject = None
        path: list[GridCell] = []
        path_length = float("inf")
        if row < 0 or row >= h or col < 0 or col >= w or not bool(nav[row, col]):
            reject = "not_traversible"
            rejected_clearance += 1
        elif clear + 1e-6 < float(config.min_clearance_m):
            reject = "low_clearance"
            rejected_clearance += 1
        elif improvement + 1e-6 < float(config.min_approach_improvement_m):
            reject = "insufficient_approach_improvement"
            rejected_unreachable += 1
        else:
            if dist_from_current is not None:
                path_length = _safe_map_value(dist_from_current, (row, col), default=float("inf"))
            if not math.isfinite(path_length):
                plan = planner.plan(current, [(row, col)])
                path = list(plan.path)
                path_length = float(plan.length_m)
            else:
                plan = planner.plan(current, [(row, col)])
                path = list(plan.path)
                path_length = float(plan.length_m)
            if not path:
                reject = "unreachable"
                rejected_unreachable += 1
        score = -2.0 * float(d_frontier) - 0.25 * float(path_length) + 0.5 * float(clear)
        if len(debug) < max_debug:
            debug.append(
                {
                    "cell": [row, col],
                    "mode": str(mode),
                    "clearance_m": float(clear),
                    "distance_to_frontier_m": None if not math.isfinite(d_frontier) else float(d_frontier),
                    "current_distance_to_frontier_m": None if not math.isfinite(current_d_frontier) else float(current_d_frontier),
                    "improvement_m": None if not math.isfinite(improvement) else float(improvement),
                    "path_length_m": None if not math.isfinite(path_length) else float(path_length),
                    "score": None if not math.isfinite(score) else float(score),
                    "reject_reason": reject,
                }
            )
        if reject is None:
            accepted.append((score, (row, col), mode, clear, d_frontier, improvement, path_length, path))
    if not accepted:
        return _none_target(
            "frontier_no_reachable_approach_target",
            center,
            original,
            current_d_frontier,
            candidates_evaluated=evaluated,
            candidates_rejected_by_clearance=rejected_clearance,
            candidates_rejected_by_unreachable=rejected_unreachable,
            debug_candidates=debug,
        )
    accepted.sort(key=lambda item: (item[0], item[3]), reverse=True)
    _score, cell, mode, clear, d_frontier, improvement, path_length, path = accepted[0]
    return FrontierRecoveryTarget(
        target_cell=cell,
        reachable=True,
        path=path,
        mode=str(mode or default_mode),
        reason="frontier_recovery_target_selected",
        frontier_center_grid=center,
        original_target_grid=original,
        distance_to_frontier_m=float(d_frontier),
        current_distance_to_frontier_m=float(current_d_frontier),
        improvement_m=float(improvement),
        target_clearance_m=float(clear),
        path_length_m=float(path_length),
        candidates_evaluated=evaluated,
        candidates_rejected_by_clearance=rejected_clearance,
        candidates_rejected_by_unreachable=rejected_unreachable,
        debug_candidates=debug[:max_debug],
    )


def _distance_to_frontier_map(shape: tuple[int, int], members: Sequence[GridCell], resolution_m: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for row, col in members:
        rr, cc = int(row), int(col)
        if 0 <= rr < h and 0 <= cc < w:
            mask[rr, cc] = True
    if not np.any(mask):
        return np.full(shape, float("inf"), dtype=np.float32)
    try:
        from scipy import ndimage

        return ndimage.distance_transform_edt(~mask).astype(np.float32) * float(resolution_m)
    except Exception:
        coords = np.argwhere(mask)
        rows = np.arange(h, dtype=np.float32)[:, None]
        cols = np.arange(w, dtype=np.float32)[None, :]
        out = np.full(shape, np.inf, dtype=np.float32)
        for start in range(0, coords.shape[0], 256):
            chunk = coords[start : start + 256].astype(np.float32)
            dr = rows[:, :, None] - chunk[:, 0][None, None, :]
            dc = cols[:, :, None] - chunk[:, 1][None, None, :]
            out = np.minimum(out, np.sqrt(np.min(dr * dr + dc * dc, axis=2)).astype(np.float32))
        return out * float(resolution_m)


def _none_target(
    reason: str,
    center: GridCell | None,
    original: GridCell | None,
    current_d_frontier: float,
    *,
    candidates_evaluated: int,
    candidates_rejected_by_clearance: int = 0,
    candidates_rejected_by_unreachable: int = 0,
    debug_candidates: Sequence[dict] | None = None,
) -> FrontierRecoveryTarget:
    return FrontierRecoveryTarget(
        target_cell=None,
        reachable=False,
        path=[],
        mode="none",
        reason=str(reason),
        frontier_center_grid=center,
        original_target_grid=original,
        distance_to_frontier_m=float("inf"),
        current_distance_to_frontier_m=float(current_d_frontier),
        improvement_m=0.0,
        target_clearance_m=float("nan"),
        path_length_m=float("inf"),
        candidates_evaluated=int(candidates_evaluated),
        candidates_rejected_by_clearance=int(candidates_rejected_by_clearance),
        candidates_rejected_by_unreachable=int(candidates_rejected_by_unreachable),
        debug_candidates=list(debug_candidates or []),
    )


def _cell(value: Sequence[int]) -> GridCell:
    return (int(value[0]), int(value[1]))


def _unique_cells(cells: Sequence[GridCell]) -> list[GridCell]:
    out: list[GridCell] = []
    seen: set[GridCell] = set()
    for cell in cells:
        c = _cell(cell)
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _safe_map_value(array: np.ndarray, cell: GridCell, *, default: float) -> float:
    row, col = int(cell[0]), int(cell[1])
    if row < 0 or row >= array.shape[0] or col < 0 or col >= array.shape[1]:
        return float(default)
    try:
        return float(array[row, col])
    except Exception:
        return float(default)


def _cell_l2(a: GridCell, b: GridCell) -> float:
    return float(math.hypot(int(a[0]) - int(b[0]), int(a[1]) - int(b[1])))
