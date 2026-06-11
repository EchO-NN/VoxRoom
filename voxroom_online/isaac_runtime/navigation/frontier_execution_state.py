from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.graph.decision import NavigationDecision

GridCell = Tuple[int, int]


def _grid_cell(value) -> Optional[GridCell]:
    if value is None:
        return None
    try:
        if len(value) < 2:
            return None
        return (int(value[0]), int(value[1]))
    except Exception:
        return None


def _grid_cells(values: Sequence[Sequence[int]] | None) -> List[GridCell]:
    out: List[GridCell] = []
    for value in values or []:
        cell = _grid_cell(value)
        if cell is not None:
            out.append(cell)
    return out


def _distance_to_cells_m(current_grid: GridCell, cells: Sequence[GridCell], resolution_m: float) -> float:
    if not cells:
        return float("inf")
    cur = np.asarray(current_grid, dtype=np.float32)
    arr = np.asarray(cells, dtype=np.float32)
    return float(np.min(np.linalg.norm(arr - cur[None, :], axis=1)) * float(resolution_m))


def _extract_frontier_center(nav_decision: NavigationDecision) -> Optional[GridCell]:
    meta = dict(getattr(nav_decision, "metadata", None) or {})
    center = _grid_cell(meta.get("frontier_center_grid"))
    if center is not None:
        return center
    if nav_decision.frontier_decision is not None and nav_decision.frontier_decision.selected_frontier is not None:
        return _grid_cell(nav_decision.frontier_decision.selected_frontier.center_grid)
    cells = _grid_cells(nav_decision.target_cells)
    return cells[0] if cells else None


def _extract_actual_target(nav_decision: NavigationDecision) -> Optional[GridCell]:
    meta = dict(getattr(nav_decision, "metadata", None) or {})
    target = _grid_cell(meta.get("frontier_actual_target_grid"))
    if target is not None:
        return target
    cells = _grid_cells(nav_decision.target_cells)
    return cells[0] if cells else None


def _frontier_id_from_metadata(nav_decision: NavigationDecision, commitment_metadata: dict) -> Optional[int]:
    meta = dict(getattr(nav_decision, "metadata", None) or {})
    commitment = dict(meta.get("frontier_commitment") or {})
    value = (
        commitment_metadata.get("active_frontier_id")
        if commitment_metadata is not None
        else None
    )
    if value is None:
        value = meta.get("active_frontier_id", commitment.get("active_frontier_id"))
    try:
        return None if value is None else int(value)
    except Exception:
        return None


@dataclass
class CommittedFrontierExecutionState:
    active: bool = False
    frontier_id: Optional[int] = None
    selected_step: int = -1
    selected_robot_grid: Optional[GridCell] = None
    selected_center_grid: Optional[GridCell] = None
    actual_target_grid: Optional[GridCell] = None
    target_cells: List[GridCell] = field(default_factory=list)
    selected_reason: str = ""

    last_plan_step: int = -1
    last_replan_step: int = -1
    best_distance_m: float = float("inf")
    no_progress_steps: int = 0
    guard_blocked_steps: int = 0
    no_path_replans: int = 0

    arrival_candidate_steps: int = 0
    arrival_pending_update: bool = False
    arrival_confirmed_step: int = -1
    arrival_consumed_step: int = -1
    arrival_key: tuple | None = None

    failed: bool = False
    failure_reason: str = ""
    blacklisted: bool = False

    recovery_active: bool = False
    recovery_reason: str = ""
    recovery_started_step: int = -1
    recovery_target_grid: Optional[GridCell] = None
    recovery_path: List[GridCell] = field(default_factory=list)
    recovery_attempts: int = 0
    original_actual_target_grid: Optional[GridCell] = None
    partial_arrival_pending_update: bool = False
    partial_arrival_reason: str = ""

    def exists(self) -> bool:
        return bool(self.active) and not bool(self.failed) and bool(self.target_cells)

    def start_from_decision(
        self,
        nav_decision: NavigationDecision,
        step: int,
        commitment_metadata: dict | None = None,
        current_grid: GridCell | None = None,
    ) -> None:
        is_frontier = nav_decision is not None and str(nav_decision.mode) == "frontier" and bool(nav_decision.target_cells)
        if not is_frontier:
            self.clear()
            return
        metadata = dict(commitment_metadata or {})
        self.active = True
        self.frontier_id = _frontier_id_from_metadata(nav_decision, metadata)
        self.selected_robot_grid = _grid_cell(current_grid)
        self.selected_center_grid = _extract_frontier_center(nav_decision)
        self.actual_target_grid = _extract_actual_target(nav_decision)
        self.target_cells = _grid_cells(nav_decision.target_cells)
        self.selected_step = int(step)
        self.selected_reason = str(nav_decision.reason or metadata.get("frontier_commitment_reason") or "selected_frontier")
        self.last_plan_step = int(step)
        self.last_replan_step = int(step)
        self.best_distance_m = float("inf")
        self.no_progress_steps = 0
        self.guard_blocked_steps = 0
        self.no_path_replans = 0
        self.arrival_candidate_steps = 0
        self.arrival_pending_update = False
        self.arrival_confirmed_step = -1
        self.arrival_consumed_step = -1
        self.arrival_key = None
        self.failed = False
        self.failure_reason = ""
        self.blacklisted = False
        self.recovery_active = False
        self.recovery_reason = ""
        self.recovery_started_step = -1
        self.recovery_target_grid = None
        self.recovery_path = []
        self.recovery_attempts = 0
        self.original_actual_target_grid = None
        self.partial_arrival_pending_update = False
        self.partial_arrival_reason = ""

    def mark_replanned(self, step: int) -> None:
        self.last_replan_step = int(step)
        self.last_plan_step = int(step)

    def sync_actual_target(self, target: GridCell, step: int) -> None:
        cell = (int(target[0]), int(target[1]))
        if self.actual_target_grid == cell and self.target_cells == [cell]:
            self.mark_replanned(step)
            return
        self.actual_target_grid = cell
        self.target_cells = [cell]
        self.best_distance_m = float("inf")
        self.no_progress_steps = 0
        self.arrival_candidate_steps = 0
        self.arrival_pending_update = False
        self.arrival_confirmed_step = -1
        self.arrival_key = None
        self.mark_replanned(step)

    def start_recovery(
        self,
        *,
        step: int,
        target_grid: GridCell,
        path: Sequence[GridCell],
        reason: str,
    ) -> None:
        if self.original_actual_target_grid is None:
            self.original_actual_target_grid = self.actual_target_grid
        target = (int(target_grid[0]), int(target_grid[1]))
        self.recovery_active = True
        self.recovery_reason = str(reason)
        self.recovery_started_step = int(step)
        self.recovery_target_grid = target
        self.recovery_path = [tuple(int(x) for x in cell) for cell in path]
        self.recovery_attempts += 1
        self.actual_target_grid = target
        self.target_cells = [target]
        self.no_path_replans = 0
        self.no_progress_steps = 0
        self.guard_blocked_steps = 0
        self.best_distance_m = float("inf")
        self.arrival_candidate_steps = 0
        self.arrival_pending_update = False
        self.partial_arrival_pending_update = False
        self.partial_arrival_reason = ""
        self.mark_replanned(step)

    def mark_arrival_pending(self, step: int, key: tuple) -> None:
        self.arrival_pending_update = True
        self.arrival_confirmed_step = int(step)
        self.arrival_key = key

    def mark_partial_arrival_pending(self, step: int, key: tuple, reason: str) -> None:
        self.arrival_pending_update = True
        self.partial_arrival_pending_update = True
        self.partial_arrival_reason = str(reason)
        self.arrival_confirmed_step = int(step)
        self.arrival_key = key

    def consume_arrival_update(self, step: int) -> None:
        self.arrival_pending_update = False
        self.partial_arrival_pending_update = False
        self.arrival_consumed_step = int(step)
        self.active = False
        self.recovery_active = False

    def mark_failed(self, reason: str, blacklist: bool) -> None:
        self.failed = True
        self.failure_reason = str(reason)
        self.blacklisted = bool(blacklist)
        self.active = False
        self.recovery_active = False

    def clear(self) -> None:
        self.active = False
        self.frontier_id = None
        self.selected_step = -1
        self.selected_robot_grid = None
        self.selected_center_grid = None
        self.actual_target_grid = None
        self.target_cells = []
        self.selected_reason = ""
        self.last_plan_step = -1
        self.last_replan_step = -1
        self.best_distance_m = float("inf")
        self.no_progress_steps = 0
        self.guard_blocked_steps = 0
        self.no_path_replans = 0
        self.arrival_candidate_steps = 0
        self.arrival_pending_update = False
        self.arrival_confirmed_step = -1
        self.arrival_consumed_step = -1
        self.arrival_key = None
        self.failed = False
        self.failure_reason = ""
        self.blacklisted = False
        self.recovery_active = False
        self.recovery_reason = ""
        self.recovery_started_step = -1
        self.recovery_target_grid = None
        self.recovery_path = []
        self.recovery_attempts = 0
        self.original_actual_target_grid = None
        self.partial_arrival_pending_update = False
        self.partial_arrival_reason = ""

    def distance_to_target_m(self, current_grid: GridCell, resolution_m: float) -> float:
        if self.actual_target_grid is not None:
            return _distance_to_cells_m(current_grid, [self.actual_target_grid], resolution_m)
        return _distance_to_cells_m(current_grid, self.target_cells, resolution_m)

    def update_progress(self, current_grid: GridCell, resolution_m: float, progress_min_delta_m: float) -> float:
        distance_m = self.distance_to_target_m(current_grid, resolution_m)
        if not math.isfinite(distance_m):
            return distance_m
        if distance_m + float(progress_min_delta_m) < float(self.best_distance_m):
            self.best_distance_m = float(distance_m)
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
        return distance_m

    def to_navigation_decision(self, last_nav_decision: NavigationDecision | None) -> NavigationDecision:
        metadata = dict(getattr(last_nav_decision, "metadata", None) or {})
        metadata.update(self.debug_metadata(step=None, current_grid=None, resolution_m=1.0))
        metadata.update(
            {
                "frontier_commitment_reason": "continue_committed_frontier_execution",
                "frontier_execution_locked": True,
                "frontier_center_grid": list(self.selected_center_grid) if self.selected_center_grid is not None else None,
                "frontier_actual_target_grid": list(self.actual_target_grid) if self.actual_target_grid is not None else None,
                "frontier_original_target_grid": list(self.original_actual_target_grid) if self.original_actual_target_grid is not None else None,
                "frontier_recovery_target_grid": list(self.recovery_target_grid) if self.recovery_target_grid is not None else None,
            }
        )
        return NavigationDecision(
            "frontier",
            list(self.target_cells),
            False,
            None if last_nav_decision is None else last_nav_decision.selected_candidate,
            None if last_nav_decision is None else last_nav_decision.frontier_decision,
            "continue_committed_frontier_execution",
            state="" if last_nav_decision is None else last_nav_decision.state,
            metadata=metadata,
        )

    def debug_metadata(
        self,
        *,
        step: int | None,
        current_grid: GridCell | None,
        resolution_m: float,
    ) -> dict[str, object]:
        distance_m = None
        if current_grid is not None and self.exists():
            distance_m = self.distance_to_target_m(current_grid, resolution_m)
        age = 0 if step is None or self.selected_step < 0 else int(step) - int(self.selected_step)
        return {
            "frontier_exec_active": bool(self.exists()),
            "frontier_exec_id": self.frontier_id,
            "frontier_exec_selected_step": int(self.selected_step),
            "frontier_exec_age": int(age),
            "frontier_exec_selected_robot_grid": list(self.selected_robot_grid) if self.selected_robot_grid is not None else None,
            "frontier_exec_center_grid": list(self.selected_center_grid) if self.selected_center_grid is not None else None,
            "frontier_exec_actual_target_grid": list(self.actual_target_grid) if self.actual_target_grid is not None else None,
            "frontier_exec_original_actual_target_grid": (
                list(self.original_actual_target_grid) if self.original_actual_target_grid is not None else None
            ),
            "frontier_exec_distance_to_target_m": None if distance_m is None else float(distance_m),
            "frontier_exec_arrival_candidate_steps": int(self.arrival_candidate_steps),
            "frontier_exec_arrival_pending_update": bool(self.arrival_pending_update),
            "frontier_exec_arrival_confirmed_step": int(self.arrival_confirmed_step),
            "frontier_exec_arrival_consumed_step": int(self.arrival_consumed_step),
            "frontier_exec_no_progress_steps": int(self.no_progress_steps),
            "frontier_exec_guard_blocked_steps": int(self.guard_blocked_steps),
            "frontier_exec_no_path_replans": int(self.no_path_replans),
            "frontier_exec_failed": bool(self.failed),
            "frontier_exec_failure_reason": str(self.failure_reason),
            "frontier_exec_blacklisted": bool(self.blacklisted),
            "frontier_exec_recovery_active": bool(self.recovery_active),
            "frontier_exec_recovery_reason": str(self.recovery_reason),
            "frontier_exec_recovery_started_step": int(self.recovery_started_step),
            "frontier_exec_recovery_target_grid": list(self.recovery_target_grid) if self.recovery_target_grid is not None else None,
            "frontier_exec_recovery_attempts": int(self.recovery_attempts),
            "frontier_exec_recovery_path_len": int(len(self.recovery_path)),
            "frontier_exec_partial_arrival_pending": bool(self.partial_arrival_pending_update),
            "frontier_exec_partial_arrival_reason": str(self.partial_arrival_reason),
        }


def make_committed_frontier_arrival_key(execution: CommittedFrontierExecutionState) -> tuple:
    return (
        "frontier",
        execution.frontier_id,
        tuple(execution.selected_center_grid or (-1, -1)),
        tuple(execution.actual_target_grid or (-1, -1)),
    )


def frontier_arrival_status(
    *,
    step: int,
    current_grid: GridCell,
    execution: CommittedFrontierExecutionState,
    resolution_m: float,
    reached_radius_m: float,
    min_steps_since_selection: int,
    confirm_steps: int,
    current_path: Sequence[GridCell] | None = None,
) -> tuple[bool, float, str]:
    _ = current_path
    if not execution.exists():
        return False, float("inf"), "no_active_committed_frontier"
    distance_m = execution.distance_to_target_m(current_grid, resolution_m)
    age = int(step) - int(execution.selected_step)
    if age < int(min_steps_since_selection):
        execution.arrival_candidate_steps = 0
        return False, distance_m, "arrival_too_early"
    if distance_m <= float(reached_radius_m):
        execution.arrival_candidate_steps += 1
    else:
        execution.arrival_candidate_steps = 0
        return False, distance_m, "arrival_not_in_radius"
    if execution.arrival_candidate_steps >= max(1, int(confirm_steps)):
        return True, distance_m, "frontier_arrival_confirmed"
    return False, distance_m, "frontier_arrival_candidate"
