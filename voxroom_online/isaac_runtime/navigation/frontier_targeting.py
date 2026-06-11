from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.mapping.frontier import FrontierCluster
from voxroom_online.isaac_runtime.navigation.astar import GridAStarPlanner

GridCell = Tuple[int, int]


@dataclass
class FrontierTargetCandidate:
    cell: GridCell
    source: str
    distance_to_center_cells: float
    clearance_m: float
    path_length_m: float
    reachable: bool
    score: float
    reject_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "cell": [int(self.cell[0]), int(self.cell[1])],
            "source": str(self.source),
            "distance_to_center_cells": float(self.distance_to_center_cells),
            "clearance_m": float(self.clearance_m),
            "path_length_m": float(self.path_length_m),
            "reachable": bool(self.reachable),
            "score": float(self.score),
            "reject_reason": self.reject_reason,
        }


@dataclass
class FrontierTargetResolution:
    target_cells: List[GridCell]
    actual_target_grid: GridCell | None
    center_grid: GridCell
    mode: str
    reachable: bool
    reason: str
    candidates_evaluated: int = 0
    candidates_rejected_by_clearance: int = 0
    candidates_rejected_by_reachability: int = 0
    selected_clearance_m: float | None = None
    selected_path_length_m: float | None = None
    debug_candidates: List[dict] = field(default_factory=list)

    def metadata(self) -> dict:
        return {
            "frontier_target_mode": str(self.mode),
            "frontier_center_grid": [int(self.center_grid[0]), int(self.center_grid[1])],
            "frontier_actual_target_grid": (
                [int(self.actual_target_grid[0]), int(self.actual_target_grid[1])]
                if self.actual_target_grid is not None
                else None
            ),
            "frontier_target_reachable": bool(self.reachable),
            "frontier_target_reason": str(self.reason),
            "frontier_target_clearance_m": self.selected_clearance_m,
            "frontier_target_path_length_m": self.selected_path_length_m,
            "frontier_target_candidates_evaluated": int(self.candidates_evaluated),
            "frontier_target_rejected_by_clearance": int(self.candidates_rejected_by_clearance),
            "frontier_target_rejected_by_reachability": int(self.candidates_rejected_by_reachability),
            "frontier_target_debug_candidates": list(self.debug_candidates[:32]),
        }


def resolve_frontier_target(
    frontier: FrontierCluster,
    current_grid: GridCell,
    planner: GridAStarPlanner,
    traversible: np.ndarray,
    clearance_m: np.ndarray,
    resolution_m: float,
    *,
    min_clearance_m: float,
    search_radius_m: float,
    reached_radius_m: float,
    max_candidates: int = 128,
    require_reachable: bool = True,
) -> FrontierTargetResolution:
    nav = np.asarray(traversible, dtype=bool)
    clearance = np.asarray(clearance_m, dtype=np.float32)
    if nav.ndim != 2 or clearance.shape != nav.shape:
        raise ValueError("traversible and clearance_m must be same-shaped 2D arrays")
    h, w = nav.shape
    center = tuple(int(v) for v in frontier.center_grid)
    min_clearance = max(0.0, float(min_clearance_m))
    resolution = max(float(resolution_m), 1e-6)
    search_radius_cells = max(0, int(math.ceil(float(search_radius_m) / resolution)))
    reached_radius_cells = max(0, int(math.ceil(float(reached_radius_m) / resolution)))

    candidates = _frontier_candidate_cells(
        center,
        list(frontier.members),
        nav,
        search_radius_cells=max(search_radius_cells, reached_radius_cells),
        max_candidates=max(1, int(max_candidates)),
    )
    accepted: list[FrontierTargetCandidate] = []
    debug_candidates: list[dict] = []
    rejected_by_clearance = 0
    rejected_by_reachability = 0
    center_arr = np.asarray(center, dtype=np.float32)

    for cell, source in candidates:
        row, col = int(cell[0]), int(cell[1])
        center_dist_cells = float(np.linalg.norm(np.asarray((row, col), dtype=np.float32) - center_arr))
        clear = float(clearance[row, col]) if 0 <= row < h and 0 <= col < w else 0.0
        if not (0 <= row < h and 0 <= col < w) or not bool(nav[row, col]):
            rejected_by_clearance += 1
            candidate = FrontierTargetCandidate((row, col), source, center_dist_cells, clear, math.inf, False, -math.inf, "not_traversible")
            debug_candidates.append(candidate.to_dict())
            continue
        if clear + 1e-6 < min_clearance:
            rejected_by_clearance += 1
            candidate = FrontierTargetCandidate((row, col), source, center_dist_cells, clear, math.inf, False, -math.inf, "low_clearance")
            debug_candidates.append(candidate.to_dict())
            continue
        plan = planner.plan(tuple(int(v) for v in current_grid), [(row, col)])
        reachable = bool(plan.path)
        if require_reachable and not reachable:
            rejected_by_reachability += 1
            candidate = FrontierTargetCandidate((row, col), source, center_dist_cells, clear, math.inf, False, -math.inf, "unreachable")
            debug_candidates.append(candidate.to_dict())
            continue
        path_len = float(plan.length_m) if reachable else math.inf
        distance_to_center_m = center_dist_cells * resolution
        score = 2.0 * clear - 0.5 * distance_to_center_m - 0.2 * path_len
        source_bonus = 0.10 if source in {"approach_cell", "member_neighbor"} else 0.0
        candidate = FrontierTargetCandidate((row, col), source, center_dist_cells, clear, path_len, reachable, score + source_bonus, None)
        accepted.append(candidate)
        debug_candidates.append(candidate.to_dict())

    if not accepted:
        return FrontierTargetResolution(
            target_cells=[],
            actual_target_grid=None,
            center_grid=center,
            mode="unreachable",
            reachable=False,
            reason="frontier_no_clearance_safe_target",
            candidates_evaluated=int(len(candidates)),
            candidates_rejected_by_clearance=int(rejected_by_clearance),
            candidates_rejected_by_reachability=int(rejected_by_reachability),
            debug_candidates=debug_candidates,
        )

    accepted.sort(key=lambda item: (item.score, item.clearance_m, -item.distance_to_center_cells), reverse=True)
    selected = accepted[0]
    return FrontierTargetResolution(
        target_cells=[selected.cell],
        actual_target_grid=selected.cell,
        center_grid=center,
        mode=str(selected.source if selected.source != "member_neighbor" else "approach_cell"),
        reachable=True,
        reason="frontier_clearance_safe_target",
        candidates_evaluated=int(len(candidates)),
        candidates_rejected_by_clearance=int(rejected_by_clearance),
        candidates_rejected_by_reachability=int(rejected_by_reachability),
        selected_clearance_m=float(selected.clearance_m),
        selected_path_length_m=float(selected.path_length_m),
        debug_candidates=debug_candidates,
    )


def _frontier_candidate_cells(
    center: GridCell,
    members: Sequence[GridCell],
    traversible: np.ndarray,
    *,
    search_radius_cells: int,
    max_candidates: int,
) -> list[tuple[GridCell, str]]:
    nav = np.asarray(traversible, dtype=bool)
    h, w = nav.shape
    center_arr = np.asarray(center, dtype=np.float32)
    member_set = {tuple(int(v) for v in cell) for cell in members}
    member_set.add(tuple(int(v) for v in center))
    ordered_members = sorted(
        member_set,
        key=lambda cell: float(np.linalg.norm(np.asarray(cell, dtype=np.float32) - center_arr)),
    )
    out: dict[GridCell, str] = {}

    def add(cell: GridCell, source: str) -> None:
        if len(out) >= int(max_candidates):
            return
        row, col = int(cell[0]), int(cell[1])
        if 0 <= row < h and 0 <= col < w and (row, col) not in out:
            out[(row, col)] = source

    add(center, "center")
    for cell in ordered_members:
        add(cell, "member")
    radius = max(0, int(search_radius_cells))
    frontier_seeds = ordered_members[: max(1, min(len(ordered_members), max_candidates))]
    neighbor_candidates: list[tuple[float, GridCell, str]] = []
    for seed in frontier_seeds:
        sr, sc = int(seed[0]), int(seed[1])
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc > radius * radius:
                    continue
                rr, cc = sr + dr, sc + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not bool(nav[rr, cc]):
                    continue
                cell = (rr, cc)
                dist = float(np.linalg.norm(np.asarray(cell, dtype=np.float32) - center_arr))
                source = "approach_cell" if (rr, cc) not in member_set else "member_neighbor"
                neighbor_candidates.append((dist, cell, source))
    neighbor_candidates.sort(key=lambda item: item[0])
    for _dist, cell, source in neighbor_candidates:
        add(cell, source)
        if len(out) >= int(max_candidates):
            break
    return [(cell, source) for cell, source in out.items()]
