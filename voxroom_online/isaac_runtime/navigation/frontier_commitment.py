from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.mapping.frontier import FrontierCluster
from voxroom_online.isaac_runtime.navigation.astar import GridAStarPlanner

GridCell = Tuple[int, int]


@dataclass
class ActiveFrontierTarget:
    stable_id: int
    center_grid: GridCell
    target_cells: List[GridCell]
    selected_step: int
    last_seen_step: int
    selected_score: float
    best_distance_m: float = float("inf")
    no_progress_steps: int = 0
    reached: bool = False
    invalid_reason: str = ""
    target_clearance_m: Optional[float] = None


@dataclass
class FrontierCommitmentDecision:
    frontier: Optional[FrontierCluster]
    target_cells: List[GridCell]
    selected_stable_id: Optional[int]
    keep_existing: bool
    reason: str
    metadata: Dict[str, object] = field(default_factory=dict)


def _center_dist(a: Sequence[int], b: Sequence[int]) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(aa - bb))


def _metadata_float(metadata: Optional[Dict[str, object]], key: str) -> Optional[float]:
    if not metadata:
        return None
    value = metadata.get(key)
    try:
        return None if value is None else float(value)
    except Exception:
        return None


class FrontierCommitmentManager:
    def __init__(
        self,
        resolution_m: float,
        match_radius_m: float = 0.75,
        reached_radius_m: float = 0.60,
        min_commit_steps: int = 12,
        max_commit_steps: int = 0,
        switch_margin: float = 0.25,
        switch_ratio: float = 1.15,
        no_progress_steps: int = 25,
        progress_min_delta_m: float = 0.10,
        blacklist_ttl_steps: int = 100,
        allow_score_switch: bool = False,
        switch_requires_refresh: bool = False,
    ):
        self.resolution_m = float(resolution_m)
        self.match_radius_cells = max(1, int(round(float(match_radius_m) / max(self.resolution_m, 1e-6))))
        self.reached_radius_m = float(reached_radius_m)
        self.min_commit_steps = int(min_commit_steps)
        self.max_commit_steps = int(max_commit_steps)
        self.switch_margin = float(switch_margin)
        self.switch_ratio = float(switch_ratio)
        self.no_progress_steps = int(no_progress_steps)
        self.progress_min_delta_m = float(progress_min_delta_m)
        self.blacklist_ttl_steps = int(blacklist_ttl_steps)
        self.allow_score_switch = bool(allow_score_switch)
        self.switch_requires_refresh = bool(switch_requires_refresh)
        self.active: Optional[ActiveFrontierTarget] = None
        self.next_id = 1
        self.blacklist: List[Tuple[GridCell, int, str]] = []

    def reset(self) -> None:
        self.active = None
        self.next_id = 1
        self.blacklist = []

    def invalidate_active(self, step: int, reason: str, blacklist: bool = True) -> None:
        if self.active is None:
            return
        self.active.invalid_reason = str(reason)
        if blacklist:
            self._blacklist(self.active.center_grid, int(step), str(reason))
        self.active = None

    def mark_active_reached(self, step: int, reason: str = "frontier_reached") -> None:
        _ = step
        if self.active is None:
            return
        self.active.reached = True
        self.active.invalid_reason = str(reason)
        self.active = None

    def mark_active_failed(self, step: int, reason: str, blacklist: bool = True) -> None:
        self.invalidate_active(step, reason, blacklist=blacklist)

    def blacklist_frontier(self, frontier_or_cell, step: int, reason: str) -> None:
        center = getattr(frontier_or_cell, "center_grid", frontier_or_cell)
        self._blacklist(tuple(int(v) for v in center), int(step), str(reason))

    def select(
        self,
        frontiers: Sequence[FrontierCluster],
        proposed_frontier: Optional[FrontierCluster],
        proposed_score: float,
        current_grid: GridCell,
        step: int,
        planner: Optional[GridAStarPlanner] = None,
        target_cells: Optional[Sequence[GridCell]] = None,
        target_frontier: Optional[FrontierCluster] = None,
        target_metadata: Optional[Dict[str, object]] = None,
        scores_by_index: Optional[Sequence[float]] = None,
    ) -> FrontierCommitmentDecision:
        self._expire_blacklist(int(step))
        clusters = list(frontiers)
        scores = [float(x) for x in (scores_by_index or [])]
        stable_ids = self._assign_stable_ids(clusters)
        score_by_id = {
            id(cluster): float(scores[idx])
            for idx, cluster in enumerate(clusters)
            if idx < len(scores) and np.isfinite(float(scores[idx]))
        }
        matched_active = self._matched_active_frontier(clusters, stable_ids)
        self._update_active_state(matched_active, current_grid, step, score_by_id)

        proposed = proposed_frontier if proposed_frontier in clusters else None
        if proposed is None:
            proposed = self._best_nonblacklisted_frontier(clusters, current_grid, int(step), scores)
            proposed_score = float(score_by_id.get(id(proposed), 0.0)) if proposed is not None else 0.0
        if proposed is not None and self._is_blacklisted(proposed.center_grid, int(step)):
            proposed = self._best_nonblacklisted_frontier(clusters, current_grid, int(step), scores)
            proposed_score = float(score_by_id.get(id(proposed), 0.0)) if proposed is not None else 0.0

        if self.active is not None:
            active_age = int(step) - int(self.active.selected_step)
            active_score = float(score_by_id.get(id(matched_active), self.active.selected_score))
            switch_reason = self._switch_reason(self.active, active_age, proposed, proposed_score, active_score)
            if not switch_reason:
                if matched_active is not None:
                    self.active.center_grid = tuple(int(v) for v in matched_active.center_grid)
                    target_belongs_to_active = (
                        target_cells is not None
                        and target_frontier is not None
                        and _center_dist(target_frontier.center_grid, matched_active.center_grid) <= self.match_radius_cells
                    )
                    if target_belongs_to_active:
                        self.active.target_cells = [tuple(int(x) for x in cell) for cell in target_cells]
                        self.active.target_clearance_m = _metadata_float(target_metadata, "frontier_target_clearance_m")
                    self.active.last_seen_step = int(step)
                    self.active.selected_score = active_score
                return FrontierCommitmentDecision(
                    frontier=matched_active,
                    target_cells=list(self.active.target_cells),
                    selected_stable_id=int(self.active.stable_id),
                    keep_existing=True,
                    reason="continue_committed_frontier",
                    metadata=self._metadata(step, active_score=active_score, proposed_score=proposed_score),
                )
            if self.switch_requires_refresh and switch_reason != "frontier_reached":
                metadata = self._metadata(step, active_score=active_score, proposed_score=proposed_score)
                metadata.update(
                    {
                        "pending_switch_reason": str(switch_reason),
                        "frontier_refresh_required_before_reselect": True,
                    }
                )
                return FrontierCommitmentDecision(
                    frontier=matched_active,
                    target_cells=list(self.active.target_cells),
                    selected_stable_id=int(self.active.stable_id),
                    keep_existing=True,
                    reason="frontier_switch_requires_refresh",
                    metadata=metadata,
                )
            if switch_reason in {"frontier_reached", "frontier_unmatched", "frontier_no_progress", "frontier_max_commit_steps"}:
                if switch_reason in {"frontier_no_progress", "frontier_max_commit_steps"}:
                    self._blacklist(self.active.center_grid, int(step), switch_reason)
                self.active = None
                if proposed is not None and self._is_blacklisted(proposed.center_grid, int(step)):
                    proposed = self._best_nonblacklisted_frontier(clusters, current_grid, int(step), scores)
                    proposed_score = float(score_by_id.get(id(proposed), 0.0)) if proposed is not None else 0.0

        if proposed is None:
            return FrontierCommitmentDecision(None, [], None, False, "no_available_frontier", self._metadata(step))
        proposed_targets = [tuple(int(x) for x in cell) for cell in (target_cells if target_cells is not None else proposed.members)]
        if planner is not None:
            result = planner.plan(current_grid, proposed_targets)
            if not result.path:
                self._blacklist(proposed.center_grid, int(step), "frontier_unreachable")
                return FrontierCommitmentDecision(None, [], None, False, "frontier_unreachable", self._metadata(step))
        stable_id = stable_ids.get(id(proposed), self._new_stable_id())
        chosen_targets = proposed_targets
        self.active = ActiveFrontierTarget(
            stable_id=int(stable_id),
            center_grid=tuple(int(v) for v in proposed.center_grid),
            target_cells=chosen_targets,
            selected_step=int(step),
            last_seen_step=int(step),
            selected_score=float(proposed_score),
            target_clearance_m=_metadata_float(target_metadata, "frontier_target_clearance_m"),
        )
        self._update_progress(current_grid, int(step))
        return FrontierCommitmentDecision(
            frontier=proposed,
            target_cells=list(self.active.target_cells),
            selected_stable_id=int(self.active.stable_id),
            keep_existing=False,
            reason="selected_new_frontier",
            metadata=self._metadata(step, proposed_score=proposed_score),
        )

    def _assign_stable_ids(self, clusters: Sequence[FrontierCluster]) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for cluster in clusters:
            if self.active is not None and _center_dist(cluster.center_grid, self.active.center_grid) <= self.match_radius_cells:
                out[id(cluster)] = int(self.active.stable_id)
            else:
                out[id(cluster)] = self._new_stable_id()
        return out

    def _new_stable_id(self) -> int:
        stable_id = int(self.next_id)
        self.next_id += 1
        return stable_id

    def _matched_active_frontier(
        self,
        clusters: Sequence[FrontierCluster],
        stable_ids: Dict[int, int],
    ) -> Optional[FrontierCluster]:
        if self.active is None:
            return None
        matches = [
            cluster
            for cluster in clusters
            if stable_ids.get(id(cluster)) == int(self.active.stable_id)
        ]
        if not matches:
            return None
        matches.sort(key=lambda cluster: _center_dist(cluster.center_grid, self.active.center_grid))
        return matches[0]

    def _distance_to_cells_m(self, current_grid: GridCell, cells: Sequence[GridCell]) -> float:
        if not cells:
            return float("inf")
        cur = np.asarray(current_grid, dtype=np.float32)
        arr = np.asarray(cells, dtype=np.float32)
        return float(np.min(np.linalg.norm(arr - cur[None, :], axis=1)) * self.resolution_m)

    def _update_active_state(
        self,
        matched_active: Optional[FrontierCluster],
        current_grid: GridCell,
        step: int,
        score_by_id: Dict[int, float],
    ) -> None:
        if self.active is None:
            return
        if matched_active is None:
            self._update_progress(current_grid, step)
            if self.active.no_progress_steps >= self.no_progress_steps:
                self.active.invalid_reason = "frontier_no_progress"
            return
        # Commit/reached semantics are tied to the selected frontier center.
        # The controller may use a fallback member if the center is hard to
        # plan to, but reaching any member of a wide frontier band would make
        # exploration abandon the chosen frontier halfway.
        if self._distance_to_cells_m(current_grid, [self.active.center_grid]) <= self.reached_radius_m:
            self.active.reached = True
            return
        self._update_progress(current_grid, step)
        if self.active.no_progress_steps >= self.no_progress_steps:
            self.active.invalid_reason = "frontier_no_progress"
        score = score_by_id.get(id(matched_active))
        if score is not None:
            self.active.selected_score = float(score)

    def _update_progress(self, current_grid: GridCell, step: int) -> None:
        _ = step
        if self.active is None:
            return
        d = self._distance_to_cells_m(current_grid, self.active.target_cells)
        if d + self.progress_min_delta_m < self.active.best_distance_m:
            self.active.best_distance_m = d
            self.active.no_progress_steps = 0
        else:
            self.active.no_progress_steps += 1

    def _switch_reason(
        self,
        active: ActiveFrontierTarget,
        active_age: int,
        proposed: Optional[FrontierCluster],
        proposed_score: float,
        active_score: float,
    ) -> str:
        if active.reached:
            return "frontier_reached"
        if active.invalid_reason:
            return active.invalid_reason
        if self.max_commit_steps > 0 and active_age > self.max_commit_steps:
            return "frontier_max_commit_steps"
        if proposed is None:
            return ""
        proposed_is_active = _center_dist(proposed.center_grid, active.center_grid) <= self.match_radius_cells
        if proposed_is_active:
            return ""
        enough_age = active_age >= self.min_commit_steps
        better_margin = float(proposed_score) > float(active_score) + self.switch_margin
        better_ratio = float(proposed_score) > float(active_score) * self.switch_ratio
        if self.allow_score_switch and enough_age and better_margin and better_ratio:
            return "frontier_score_hysteresis_switch"
        return ""

    def _best_nonblacklisted_frontier(
        self,
        clusters: Sequence[FrontierCluster],
        current_grid: GridCell,
        step: int,
        scores: Sequence[float],
    ) -> Optional[FrontierCluster]:
        candidates = []
        for idx, cluster in enumerate(clusters):
            if self._is_blacklisted(cluster.center_grid, step):
                continue
            score = float(scores[idx]) if idx < len(scores) and np.isfinite(float(scores[idx])) else float(cluster.distance_inverse)
            dist = self._distance_to_cells_m(current_grid, cluster.members)
            candidates.append((score, -dist, cluster))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    def _blacklist(self, center_grid: GridCell, step: int, reason: str) -> None:
        if self.blacklist_ttl_steps <= 0:
            return
        self.blacklist.append((tuple(int(v) for v in center_grid), int(step) + self.blacklist_ttl_steps, str(reason)))

    def _expire_blacklist(self, step: int) -> None:
        self.blacklist = [(cell, until, reason) for cell, until, reason in self.blacklist if int(step) <= int(until)]

    def _is_blacklisted(self, center_grid: GridCell, step: int) -> bool:
        for cell, until, _reason in self.blacklist:
            if int(step) <= int(until) and _center_dist(center_grid, cell) <= self.match_radius_cells:
                return True
        return False

    def _metadata(
        self,
        step: int,
        active_score: Optional[float] = None,
        proposed_score: Optional[float] = None,
    ) -> Dict[str, object]:
        if self.active is None:
            return {
                "active_frontier_id": None,
                "active_frontier_age": 0,
                "active_frontier_distance_m": None,
                "active_frontier_center_grid": None,
                "active_frontier_actual_target_grid": None,
                "active_frontier_target_cells_count": 0,
                "active_frontier_target_clearance_m": None,
                "frontier_commitment_target_consistent": True,
                "active_frontier_no_progress_steps": 0,
                "frontier_blacklist_count": int(len(self.blacklist)),
                "frontier_active_score": active_score,
                "frontier_proposed_score": proposed_score,
            }
        return {
            "active_frontier_id": int(self.active.stable_id),
            "active_frontier_center_grid": [int(v) for v in self.active.center_grid],
            "active_frontier_actual_target_grid": (
                [int(v) for v in self.active.target_cells[0]]
                if self.active.target_cells
                else None
            ),
            "active_frontier_target_cells_count": int(len(self.active.target_cells)),
            "active_frontier_target_clearance_m": self.active.target_clearance_m,
            "frontier_commitment_target_consistent": True,
            "active_frontier_age": int(step) - int(self.active.selected_step),
            "active_frontier_distance_m": float(self.active.best_distance_m),
            "active_frontier_no_progress_steps": int(self.active.no_progress_steps),
            "active_frontier_invalid_reason": str(self.active.invalid_reason),
            "frontier_blacklist_count": int(len(self.blacklist)),
            "frontier_active_score": active_score,
            "frontier_proposed_score": proposed_score,
        }
