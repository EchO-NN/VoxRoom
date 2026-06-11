from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.dataset.category_normalizer import normalize_category
from voxroom_online.isaac_runtime.graph.reperception import GraphReperceptionManager, compute_goal_candidate_credibility
from voxroom_online.isaac_runtime.graph.sgnav_scenegraph_adapter import SGNavSceneGraphAdapter
from voxroom_online.isaac_runtime.graph.subgraph_builder import build_object_centered_subgraphs
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo, grid_to_world_xy
from voxroom_online.isaac_runtime.mapping.frontier import FrontierCluster
from voxroom_online.isaac_runtime.navigation.astar import GridAStarPlanner, astar_distance_map
from voxroom_online.isaac_runtime.perception.object_memory import ObjectMemory, ObjectNode


@dataclass
class DecisionResult:
    selected_index: Optional[int]
    selected_frontier: Optional[FrontierCluster]
    scenegraph_scores: List[float]
    distance_scores: List[float]
    total_scores: List[float]
    reason: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class NavigationDecision:
    mode: str
    target_cells: List[Tuple[int, int]]
    stop: bool
    selected_candidate: Optional[ObjectNode]
    frontier_decision: Optional[DecisionResult]
    reason: str
    state: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class CandidateTrack:
    node_id: int
    score_sum: float = 0.0
    observation_count: int = 0
    steps: int = 0
    accepted: bool = False
    rejected: bool = False
    last_counted_observed_count: int = 0
    last_seen_step: int = -1
    reperception_debug: Dict[str, object] = field(default_factory=dict)


def normalize_scores(values: Sequence[float], mode: str = "minmax") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return arr
    norm_mode = str(mode or "minmax").strip().lower()
    if norm_mode == "none":
        return arr
    if norm_mode == "minmax":
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo < 1e-6:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)
    if norm_mode == "zscore":
        std = float(np.std(arr))
        if std < 1e-6:
            return np.zeros_like(arr)
        return (arr - float(np.mean(arr))) / std
    raise ValueError("unsupported score normalization mode: %s" % mode)


class SGNavDecision:
    def __init__(
        self,
        scenegraph: SGNavSceneGraphAdapter,
        frontier_distance_weight: float = 0.2,
        frontier_min_select_distance_m: float = 1.0,
        frontier_distance_score_span_m: float = 10.0,
        frontier_allow_near_fallback: bool = False,
        candidate_min_hits: int = 2,
        candidate_start_min_confidence: float = 0.55,
        candidate_start_min_hits: int = 2,
        candidate_recent_max_age_steps: int = 30,
        candidate_match_substring: bool = False,
        candidate_accept_requires_reperception: bool = True,
        candidate_reject_ttl_steps: int = 80,
        candidate_accept_threshold: float = 0.65,
        candidate_stop_distance_m: float = 1.0,
        candidate_standoff_min_m: float = 0.65,
        candidate_standoff_max_m: float = 1.80,
        candidate_standoff_max_cells: int = 16,
        candidate_standoff_ideal_m: float = 1.0,
        reperception_enabled: bool = True,
        reperception_min_observations: int = 3,
        reperception_max_steps: int = 10,
        reperception_same_goal_radius_m: float = 0.8,
        stop_verification_steps: int = 4,
        stop_verification_min_hits: int = 2,
        found_goal_stop_distance_m: float = 0.35,
        score_frontiers_before_candidate: bool = False,
        frontier_scenegraph_score_norm: str = "minmax",
        frontier_selection_mode: str = "sgnav",
        frontier_random_seed: int = 0,
    ):
        self.scenegraph = scenegraph
        self.frontier_distance_weight = float(frontier_distance_weight)
        self.frontier_min_select_distance_m = max(0.0, float(frontier_min_select_distance_m))
        self.frontier_distance_score_span_m = max(1e-6, float(frontier_distance_score_span_m))
        self.frontier_allow_near_fallback = bool(frontier_allow_near_fallback)
        self.candidate_min_hits = max(1, int(candidate_min_hits))
        self.candidate_start_min_confidence = float(candidate_start_min_confidence)
        self.candidate_start_min_hits = max(1, int(candidate_start_min_hits))
        self.candidate_recent_max_age_steps = max(0, int(candidate_recent_max_age_steps))
        self.candidate_match_substring = bool(candidate_match_substring)
        self.candidate_accept_requires_reperception = bool(candidate_accept_requires_reperception)
        self.candidate_reject_ttl_steps = max(0, int(candidate_reject_ttl_steps))
        self.candidate_accept_threshold = float(candidate_accept_threshold)
        self.candidate_stop_distance_m = float(candidate_stop_distance_m)
        self.candidate_standoff_min_m = float(candidate_standoff_min_m)
        self.candidate_standoff_max_m = max(float(candidate_standoff_max_m), self.candidate_standoff_min_m)
        self.candidate_standoff_max_cells = max(1, int(candidate_standoff_max_cells))
        self.candidate_standoff_ideal_m = float(candidate_standoff_ideal_m)
        self.reperception_enabled = bool(reperception_enabled)
        self.reperception_min_observations = max(1, int(reperception_min_observations))
        self.reperception_max_steps = max(0, int(reperception_max_steps))
        self.reperception_same_goal_radius_m = max(0.05, float(reperception_same_goal_radius_m))
        self.stop_verification_steps = max(0, int(stop_verification_steps))
        self.stop_verification_min_hits = max(1, int(stop_verification_min_hits))
        self.found_goal_stop_distance_m = max(0.05, float(found_goal_stop_distance_m))
        self.score_frontiers_before_candidate = bool(score_frontiers_before_candidate)
        self.frontier_scenegraph_score_norm = str(frontier_scenegraph_score_norm or "minmax").strip().lower()
        self.frontier_selection_mode = str(frontier_selection_mode or "sgnav").strip().lower()
        if self.frontier_selection_mode not in {"sgnav", "nearest", "random"}:
            raise ValueError("unsupported frontier_selection_mode: %s" % frontier_selection_mode)
        self.frontier_random_seed = int(frontier_random_seed)
        self.frontier_rng = np.random.default_rng(self.frontier_random_seed)
        self.state = "frontier"
        self.reperception_candidate_id: Optional[int] = None
        self.reperception_steps = 0
        self.stop_candidate_id: Optional[int] = None
        self.stop_verification_steps_taken = 0
        self.active_candidate_track: Optional[CandidateTrack] = None
        self.rejected_candidates: Dict[int, int] = {}
        self.graph_reperception = GraphReperceptionManager(n_max=self.reperception_max_steps, s_thres=self.candidate_accept_threshold)
        self.last_reason = ""

    def choose_frontier(self, frontier_clusters: List[FrontierCluster]) -> DecisionResult:
        if not frontier_clusters:
            return DecisionResult(None, None, [], [], [], "no_frontiers")
        if self.frontier_selection_mode in {"nearest", "random"}:
            return self._choose_frontier_without_scenegraph(frontier_clusters)
        locs = np.asarray([f.center_grid for f in frontier_clusters], dtype=np.int32)
        raw_sg_scores = np.asarray(self.scenegraph.score(locs, len(frontier_clusters)), dtype=np.float32)
        sg_scores = normalize_scores(raw_sg_scores, self.frontier_scenegraph_score_norm)
        dist_scores = np.asarray([getattr(f, "distance_inverse", 0.0) for f in frontier_clusters], dtype=np.float32)
        if len(dist_scores) != len(frontier_clusters) or not np.all(np.isfinite(dist_scores)):
            dists = np.asarray([f.path_distance_from_agent for f in frontier_clusters], dtype=np.float32)
            min_d = self.frontier_min_select_distance_m
            clipped = np.clip(dists, min_d, min_d + self.frontier_distance_score_span_m)
            dist_scores = 1.0 - (clipped - min_d) / self.frontier_distance_score_span_m
        else:
            dists = np.asarray([f.path_distance_from_agent for f in frontier_clusters], dtype=np.float32)
        total = sg_scores + self.frontier_distance_weight * dist_scores
        eligible = [
            idx
            for idx, dist in enumerate(dists)
            if np.isfinite(float(dist)) and float(dist) >= self.frontier_min_select_distance_m
        ]
        filtered_near = int(len(frontier_clusters) - len(eligible))
        used_near_fallback = False
        if not eligible:
            if self.frontier_allow_near_fallback:
                eligible = list(range(len(frontier_clusters)))
                used_near_fallback = True
                reason = "near_frontier_fallback"
            else:
                return DecisionResult(
                    None,
                    None,
                    [float(x) for x in sg_scores],
                    [float(x) for x in dist_scores],
                    [float(x) for x in total],
                    "all_frontiers_within_min_distance",
                    metadata={
                        "raw_scenegraph_scores": [float(x) for x in raw_sg_scores],
                        "normalized_scenegraph_scores": [float(x) for x in sg_scores],
                        "distance_scores": [float(x) for x in dist_scores],
                        "total_scores": [float(x) for x in total],
                        "scenegraph_score_norm": self.frontier_scenegraph_score_norm,
                        "frontier_distance_weight": float(self.frontier_distance_weight),
                        "frontier_min_select_distance_m": float(self.frontier_min_select_distance_m),
                        "frontier_distance_score_span_m": float(self.frontier_distance_score_span_m),
                        "filtered_near_frontiers": filtered_near,
                        "eligible_frontier_indices": [],
                        "used_near_frontier_fallback": False,
                    },
                )
        else:
            reason = "selected_frontier"
        eligible_total = np.asarray([float(total[idx]) for idx in eligible], dtype=np.float32)
        idx = int(eligible[int(np.argmax(eligible_total))])
        return DecisionResult(
            idx,
            frontier_clusters[idx],
            [float(x) for x in sg_scores],
            [float(x) for x in dist_scores],
            [float(x) for x in total],
            reason,
            metadata={
                "raw_scenegraph_scores": [float(x) for x in raw_sg_scores],
                "normalized_scenegraph_scores": [float(x) for x in sg_scores],
                "distance_scores": [float(x) for x in dist_scores],
                "total_scores": [float(x) for x in total],
                "scenegraph_score_norm": self.frontier_scenegraph_score_norm,
                "frontier_distance_weight": float(self.frontier_distance_weight),
                "frontier_min_select_distance_m": float(self.frontier_min_select_distance_m),
                "frontier_distance_score_span_m": float(self.frontier_distance_score_span_m),
                "filtered_near_frontiers": filtered_near,
                "eligible_frontier_indices": [int(item) for item in eligible],
                "used_near_frontier_fallback": bool(used_near_fallback),
            },
        )

    def _choose_frontier_without_scenegraph(self, frontier_clusters: List[FrontierCluster]) -> DecisionResult:
        dists = np.asarray([float(f.path_distance_from_agent) for f in frontier_clusters], dtype=np.float32)
        dist_scores = np.asarray([getattr(f, "distance_inverse", 0.0) for f in frontier_clusters], dtype=np.float32)
        if len(dist_scores) != len(frontier_clusters) or not np.all(np.isfinite(dist_scores)):
            clipped = np.clip(dists, self.frontier_min_select_distance_m, self.frontier_min_select_distance_m + self.frontier_distance_score_span_m)
            dist_scores = 1.0 - (clipped - self.frontier_min_select_distance_m) / self.frontier_distance_score_span_m
        sg_scores = np.zeros((len(frontier_clusters),), dtype=np.float32)
        eligible = [
            idx
            for idx, dist in enumerate(dists)
            if np.isfinite(float(dist)) and float(dist) >= self.frontier_min_select_distance_m
        ]
        filtered_near = int(len(frontier_clusters) - len(eligible))
        used_near_fallback = False
        if not eligible:
            if not self.frontier_allow_near_fallback:
                return DecisionResult(
                    None,
                    None,
                    [float(x) for x in sg_scores],
                    [float(x) for x in dist_scores],
                    [float(x) for x in dist_scores],
                    "all_frontiers_within_min_distance",
                    metadata={
                        "frontier_selection_mode": self.frontier_selection_mode,
                        "frontier_random_seed": int(self.frontier_random_seed),
                        "frontier_min_select_distance_m": float(self.frontier_min_select_distance_m),
                        "filtered_near_frontiers": filtered_near,
                        "eligible_frontier_indices": [],
                        "used_near_frontier_fallback": False,
                        "scenegraph_scoring_skipped": True,
                    },
                )
            eligible = list(range(len(frontier_clusters)))
            used_near_fallback = True
        if self.frontier_selection_mode == "nearest":
            idx = min(eligible, key=lambda item: float(dists[int(item)]))
            reason = "selected_nearest_frontier"
        else:
            idx = int(self.frontier_rng.choice(np.asarray(eligible, dtype=np.int32)))
            reason = "selected_random_frontier"
        total = np.zeros((len(frontier_clusters),), dtype=np.float32)
        total[idx] = 1.0
        return DecisionResult(
            int(idx),
            frontier_clusters[int(idx)],
            [float(x) for x in sg_scores],
            [float(x) for x in dist_scores],
            [float(x) for x in total],
            reason,
            metadata={
                "frontier_selection_mode": self.frontier_selection_mode,
                "frontier_random_seed": int(self.frontier_random_seed),
                "distance_scores": [float(x) for x in dist_scores],
                "total_scores": [float(x) for x in total],
                "frontier_min_select_distance_m": float(self.frontier_min_select_distance_m),
                "filtered_near_frontiers": filtered_near,
                "eligible_frontier_indices": [int(item) for item in eligible],
                "used_near_frontier_fallback": bool(used_near_fallback),
                "scenegraph_scoring_skipped": True,
            },
        )

    def choose_navigation_target(
        self,
        object_memory: ObjectMemory,
        goal_category: str,
        current_grid: Tuple[int, int],
        frontiers: List[FrontierCluster],
        planner: GridAStarPlanner,
        map_info: MapInfo,
        current_pose: Sequence[float],
        allow_frontier: bool = True,
        current_step: Optional[int] = None,
    ) -> NavigationDecision:
        prefetched_frontier_decision: Optional[DecisionResult] = None
        if allow_frontier and self.score_frontiers_before_candidate and frontiers:
            prefetched_frontier_decision = self.choose_frontier(frontiers)

        rejected_metadata: Dict[str, object] = {}
        candidate = self.select_goal_candidate(object_memory, goal_category, current_pose, current_step=current_step)
        if candidate is not None:
            distance_to_candidate = float(
                np.linalg.norm(
                    np.asarray(candidate.center_world[:2], dtype=np.float32)
                    - np.asarray([float(current_pose[0]), float(current_pose[1])], dtype=np.float32)
                )
            )
            track = self._get_or_create_candidate_track(candidate)
            credibility = self._update_candidate_track(track, candidate, current_pose, current_step)
            candidate_metadata = self._candidate_metadata(candidate, track, credibility, distance_to_candidate)

            if self.reperception_enabled and self.candidate_accept_requires_reperception and not track.accepted:
                enough_observations = (
                    int(candidate.observed_count) >= self.reperception_min_observations
                    and int(track.observation_count) >= self.reperception_min_observations
                )
                if enough_observations and credibility >= self.candidate_accept_threshold:
                    track.accepted = True
                    candidate_metadata["candidate_accepted"] = True
                elif track.steps >= self.reperception_max_steps:
                    self._reject_candidate(candidate, current_step)
                    rejected_metadata = dict(candidate_metadata)
                    rejected_metadata["candidate_rejected"] = True
                    rejected_metadata["candidate_reject_until_step"] = self.rejected_candidates.get(int(candidate.node_id))
                    self._clear_candidate_track(candidate.node_id)
                    candidate = None
                    self.state = "frontier" if allow_frontier else "none"
                    self.last_reason = "candidate_rejected_after_reperception"
                else:
                    self.state = "reperception"
                    self.last_reason = "reperception_collecting" if not enough_observations else "reperception_scanning"
                    return NavigationDecision(
                        "reperception",
                        [current_grid],
                        False,
                        candidate,
                        prefetched_frontier_decision,
                        self.last_reason,
                        state=self.state,
                        metadata=candidate_metadata,
                    )

        if candidate is not None:
            close_for_stop = distance_to_candidate <= self.candidate_stop_distance_m
            very_close = distance_to_candidate <= self.found_goal_stop_distance_m
            enough_candidate_hits = candidate.observed_count >= self.candidate_min_hits
            enough_verification_hits = candidate.observed_count >= max(self.candidate_min_hits, self.stop_verification_min_hits)
            if close_for_stop and enough_candidate_hits:
                if self.stop_candidate_id != candidate.node_id:
                    self.stop_candidate_id = candidate.node_id
                    self.stop_verification_steps_taken = 0
                if (
                    self.stop_verification_steps <= 0
                    or self.stop_verification_steps_taken >= self.stop_verification_steps
                    or (very_close and enough_verification_hits)
                ):
                    self.state = "stop"
                    self.last_reason = "stop_verification_confirmed"
                    return NavigationDecision(
                        "stop",
                        [],
                        True,
                        candidate,
                        prefetched_frontier_decision,
                        "stop_verification_confirmed",
                        state=self.state,
                        metadata={
                            **candidate_metadata,
                            "candidate_distance_m": distance_to_candidate,
                            "candidate_observed_count": int(candidate.observed_count),
                            "stop_verification_steps_taken": int(self.stop_verification_steps_taken),
                        },
                    )
                self.stop_verification_steps_taken += 1
                self.state = "stop_verification"
                self.last_reason = "stop_verification_scanning"
                return NavigationDecision(
                    "reperception",
                    [current_grid],
                    False,
                    candidate,
                    prefetched_frontier_decision,
                    "stop_verification_scanning",
                    state=self.state,
                    metadata={
                        **candidate_metadata,
                        "candidate_distance_m": distance_to_candidate,
                        "candidate_observed_count": int(candidate.observed_count),
                        "stop_verification_steps_taken": int(self.stop_verification_steps_taken),
                    },
                )
            if (
                self.reperception_enabled
                and not self.candidate_accept_requires_reperception
                and distance_to_candidate <= self.reperception_same_goal_radius_m
                and candidate.observed_count < self.reperception_min_observations
                and self.reperception_steps < self.reperception_max_steps
            ):
                if self.reperception_candidate_id != candidate.node_id:
                    self.reperception_candidate_id = candidate.node_id
                    self.reperception_steps = 0
                self.reperception_steps += 1
                self.state = "reperception"
                self.last_reason = "reperception_scanning"
                return NavigationDecision(
                    "reperception",
                    [current_grid],
                    False,
                    candidate,
                    prefetched_frontier_decision,
                    "reperception_scanning",
                    state=self.state,
                    metadata={
                        **candidate_metadata,
                        "candidate_distance_m": distance_to_candidate,
                        "candidate_observed_count": int(candidate.observed_count),
                        "reperception_steps": int(self.reperception_steps),
                    },
                )
            standoff = self.candidate_standoff_cells(candidate, planner.traversible, map_info, current_grid=current_grid)
            standoff = self._targets_with_min_progress(standoff, planner, current_grid, map_info)
            if standoff:
                self.state = "candidate"
                self.last_reason = "navigate_to_goal_candidate"
                return NavigationDecision(
                    "candidate",
                    standoff,
                    False,
                    candidate,
                    prefetched_frontier_decision,
                    "navigate_to_goal_candidate",
                    state=self.state,
                    metadata={
                        **candidate_metadata,
                        "candidate_distance_m": distance_to_candidate,
                        "candidate_observed_count": int(candidate.observed_count),
                        "target_cells_count": int(len(standoff)),
                    },
                )

        if allow_frontier:
            if not rejected_metadata:
                self.reperception_candidate_id = None
                self.reperception_steps = 0
            self.stop_candidate_id = None
            self.stop_verification_steps_taken = 0
            frontier_decision = prefetched_frontier_decision or self.choose_frontier(frontiers)
            if frontier_decision.selected_frontier is not None:
                target_cells, target_meta = self._frontier_target_cells(
                    frontier_decision.selected_frontier,
                    planner,
                    current_grid,
                )
                self.state = "frontier"
                self.last_reason = "frontier_unreachable" if not target_cells else frontier_decision.reason
                return NavigationDecision(
                    "frontier",
                    target_cells,
                    False,
                    None,
                    frontier_decision,
                    self.last_reason,
                    state=self.state,
                    metadata={
                        **rejected_metadata,
                        **target_meta,
                        "target_cells_count": int(len(target_cells)),
                    },
                )
            self.state = "none"
            self.last_reason = frontier_decision.reason
            return NavigationDecision(
                "none",
                [],
                False,
                None,
                frontier_decision,
                frontier_decision.reason,
                state=self.state,
                metadata=rejected_metadata,
            )
        self.state = "none"
        self.last_reason = "no_candidate"
        return NavigationDecision("none", [], False, None, None, "no_candidate", state=self.state, metadata=rejected_metadata)

    def _frontier_target_cells(
        self,
        frontier: FrontierCluster,
        planner: GridAStarPlanner,
        current_grid: Tuple[int, int],
    ) -> Tuple[List[Tuple[int, int]], Dict[str, object]]:
        center = tuple(int(v) for v in frontier.center_grid)
        center_reachable = bool(planner.plan(current_grid, [center]).path)
        meta: Dict[str, object] = {
            "frontier_target_mode": "center" if center_reachable else "fallback_member",
            "frontier_center_grid": [int(center[0]), int(center[1])],
            "frontier_actual_target_grid": None,
            "frontier_unreachable_recovery": False,
            "frontier_unreachable_reason": None,
        }
        if center_reachable:
            meta["frontier_actual_target_grid"] = [int(center[0]), int(center[1])]
            return [center], meta
        center_arr = np.asarray(center, dtype=np.float32)
        members = sorted(
            {tuple(int(v) for v in cell) for cell in frontier.members},
            key=lambda cell: float(np.linalg.norm(np.asarray(cell, dtype=np.float32) - center_arr)),
        )
        for member in members:
            if planner.plan(current_grid, [member]).path:
                meta["frontier_actual_target_grid"] = [int(member[0]), int(member[1])]
                meta["frontier_unreachable_recovery"] = True
                return [member], meta
        meta["frontier_target_mode"] = "unreachable"
        meta["frontier_unreachable_recovery"] = True
        meta["frontier_unreachable_reason"] = "frontier_center_and_members_unreachable"
        return [], meta

    def select_goal_candidate(
        self,
        object_memory: ObjectMemory,
        goal_category: str,
        current_pose: Sequence[float],
        current_step: Optional[int] = None,
    ) -> Optional[ObjectNode]:
        candidates = []
        agent_xy = np.asarray([float(current_pose[0]), float(current_pose[1])], dtype=np.float32)
        for node in object_memory.nodes:
            if not self._category_matches_goal(node.category, goal_category):
                continue
            if float(node.confidence) < self.candidate_start_min_confidence:
                continue
            if int(node.observed_count) < self.candidate_start_min_hits:
                continue
            if current_step is not None:
                age = int(current_step) - int(getattr(node, "last_seen_step", current_step))
                if age > self.candidate_recent_max_age_steps:
                    continue
            if self._is_candidate_rejected(node, current_step):
                continue
            dist = float(np.linalg.norm(np.asarray(node.center_world[:2], dtype=np.float32) - agent_xy))
            score = float(node.confidence) + 0.25 * min(float(node.observed_count), 5.0) - 0.05 * dist
            candidates.append((score, node))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def candidate_standoff_cells(
        self,
        candidate: ObjectNode,
        traversible: np.ndarray,
        map_info: MapInfo,
        current_grid: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        cx, cy = float(candidate.center_world[0]), float(candidate.center_world[1])
        max_radius_cells = int(np.ceil(self.candidate_standoff_max_m / map_info.resolution_m))
        r0, c0 = int(candidate.center_grid[0]), int(candidate.center_grid[1])
        traversible_bool = np.asarray(traversible).astype(bool)
        reachable = traversible_bool
        if current_grid is not None:
            dist_map = astar_distance_map(traversible_bool, current_grid, map_info.resolution_m, allow_diagonal=True)
            reachable = np.isfinite(dist_map)
            if not np.any(reachable):
                return []
        cells: List[Tuple[float, Tuple[int, int]]] = []
        nearest: List[Tuple[float, Tuple[int, int]]] = []
        h, w = traversible_bool.shape
        for r in range(max(0, r0 - max_radius_cells), min(h, r0 + max_radius_cells + 1)):
            for c in range(max(0, c0 - max_radius_cells), min(w, c0 + max_radius_cells + 1)):
                if not bool(reachable[r, c]):
                    continue
                wx, wy = grid_to_world_xy(r, c, map_info)
                dist = float(np.hypot(wx - cx, wy - cy))
                nearest.append((dist, (int(r), int(c))))
                if self.candidate_standoff_min_m <= dist <= self.candidate_standoff_max_m:
                    path_dist = 0.0
                    if current_grid is not None:
                        if dist_map is None:
                            dist_map = astar_distance_map(
                                traversible_bool,
                                current_grid,
                                map_info.resolution_m,
                                allow_diagonal=True,
                            )
                        path_dist = float(dist_map[r, c])
                        if not np.isfinite(path_dist):
                            continue
                    ideal = self.candidate_standoff_ideal_m
                    if ideal <= 0.0:
                        ideal = 0.5 * (self.candidate_standoff_min_m + self.candidate_standoff_max_m)
                    cost = float(path_dist) + 0.25 * abs(dist - ideal)
                    cells.append((cost, (int(r), int(c))))
        if cells:
            cells.sort(key=lambda item: item[0])
            return [cell for _cost, cell in cells[: self.candidate_standoff_max_cells]]
        nearest.sort(key=lambda item: item[0])
        if nearest:
            return [cell for _dist, cell in nearest[: self.candidate_standoff_max_cells]]
        return self.nearest_reachable_cells_to_candidate(
            candidate,
            traversible,
            map_info,
            current_grid=current_grid,
            max_cells=self.candidate_standoff_max_cells,
        )

    def nearest_reachable_cells_to_candidate(
        self,
        candidate: ObjectNode,
        traversible: np.ndarray,
        map_info: MapInfo,
        current_grid: Optional[Tuple[int, int]] = None,
        max_cells: int = 16,
    ) -> List[Tuple[int, int]]:
        traversible_bool = np.asarray(traversible).astype(bool)
        rows, cols = np.nonzero(traversible_bool)
        if rows.size == 0:
            return []
        reachable = traversible_bool
        if current_grid is not None:
            dist_map = astar_distance_map(traversible_bool, current_grid, map_info.resolution_m, allow_diagonal=True)
            reachable = np.isfinite(dist_map)
            rows, cols = np.nonzero(reachable)
            if rows.size == 0:
                return []
        cx, cy = float(candidate.center_world[0]), float(candidate.center_world[1])
        ranked: List[Tuple[float, Tuple[int, int]]] = []
        current = (int(current_grid[0]), int(current_grid[1])) if current_grid is not None else None
        for row, col in zip(rows, cols):
            cell = (int(row), int(col))
            wx, wy = grid_to_world_xy(cell[0], cell[1], map_info)
            candidate_dist = float(np.hypot(wx - cx, wy - cy))
            current_penalty = 0.0
            if current is not None and cell == current:
                current_penalty = 1e3
            ranked.append((candidate_dist + current_penalty, cell))
        ranked.sort(key=lambda item: item[0])
        return [cell for _dist, cell in ranked[: max(1, int(max_cells))]]

    def _targets_with_min_progress(
        self,
        targets: Sequence[Tuple[int, int]],
        planner: GridAStarPlanner,
        current_grid: Tuple[int, int],
        map_info: MapInfo,
    ) -> List[Tuple[int, int]]:
        if not targets:
            return []
        min_progress_m = max(float(map_info.resolution_m) * 3.0, 0.25)
        dist_map = astar_distance_map(planner.traversible, current_grid, map_info.resolution_m, allow_diagonal=planner.allow_diagonal)
        filtered: List[Tuple[int, int]] = []
        seen = set()
        for target in targets:
            cell = (int(target[0]), int(target[1]))
            if cell in seen or not planner.inside(cell):
                continue
            seen.add(cell)
            dist = float(dist_map[cell])
            if np.isfinite(dist) and dist >= min_progress_m:
                filtered.append(cell)
                if len(filtered) >= self.candidate_standoff_max_cells:
                    break
        return filtered

    def _category_matches_goal(self, category: str, goal_category: str) -> bool:
        cat = normalize_category(category)
        goal = normalize_category(goal_category)
        if not goal or not cat:
            return False
        if cat == goal:
            return True
        if self.candidate_match_substring:
            return goal in cat or cat in goal
        return False

    def _get_or_create_candidate_track(self, candidate: ObjectNode) -> CandidateTrack:
        node_id = int(candidate.node_id)
        if self.active_candidate_track is None or int(self.active_candidate_track.node_id) != node_id:
            self.active_candidate_track = CandidateTrack(node_id=node_id)
            self.reperception_candidate_id = node_id
            self.reperception_steps = 0
        return self.active_candidate_track

    def _clear_candidate_track(self, node_id: Optional[int] = None) -> None:
        if node_id is None or self.active_candidate_track is None or int(self.active_candidate_track.node_id) == int(node_id):
            self.active_candidate_track = None
            self.reperception_candidate_id = None
            self.reperception_steps = 0

    def _candidate_observation_score(
        self,
        candidate: ObjectNode,
        current_pose: Sequence[float],
        frontier_decision: Optional[DecisionResult] = None,
    ) -> float:
        _ = current_pose, frontier_decision
        if str(getattr(self.scenegraph, "sgnav_mode", "")).strip().lower() == "paper":
            paper_graph = getattr(self.scenegraph, "paper_graph", None)
            paper_candidate = None if paper_graph is None else paper_graph.object_nodes.get("object:%s" % int(candidate.node_id))
            if paper_candidate is not None:
                subgraphs = build_object_centered_subgraphs(paper_graph)
                subgraph_scores = self.scenegraph.hcot_scorer.score(
                    subgraphs,
                    getattr(self.scenegraph, "obj_goal_sg", normalize_category(candidate.category)),
                    graph_version=getattr(paper_graph, "version", 0),
                )
                return float(compute_goal_candidate_credibility(paper_candidate, float(candidate.confidence), subgraph_scores))
        conf = float(candidate.confidence)
        hit_term = min(float(candidate.observed_count), 5.0) / 5.0
        return float(np.clip(conf * (0.5 + 0.5 * hit_term), 0.0, 1.0))

    def _update_candidate_track(
        self,
        track: CandidateTrack,
        candidate: ObjectNode,
        current_pose: Sequence[float],
        current_step: Optional[int],
    ) -> float:
        track.steps += 1
        self.reperception_steps = track.steps
        observed_count = max(0, int(candidate.observed_count))
        delta = max(0, observed_count - int(track.last_counted_observed_count))
        last_seen_step = int(getattr(candidate, "last_seen_step", current_step if current_step is not None else -1))
        if delta > 0:
            if str(getattr(self.scenegraph, "sgnav_mode", "")).strip().lower() == "paper":
                paper_result = self._update_graph_reperception(candidate)
                if paper_result is not None:
                    track.score_sum = float(paper_result["accumulated_credibility"])
                    track.observation_count = int(paper_result["num_reperception_steps"])
                    track.reperception_debug = dict(paper_result)
                else:
                    score = self._candidate_observation_score(candidate, current_pose)
                    track.score_sum += float(score) * float(delta)
                    track.observation_count += int(delta)
            else:
                score = self._candidate_observation_score(candidate, current_pose)
                track.score_sum += float(score) * float(delta)
                track.observation_count += int(delta)
            track.last_counted_observed_count = observed_count
            track.last_seen_step = last_seen_step
        if track.observation_count <= 0:
            return 0.0
        if str(getattr(self.scenegraph, "sgnav_mode", "")).strip().lower() == "paper":
            return float(track.score_sum)
        return float(track.score_sum / max(1, track.observation_count))

    def _update_graph_reperception(self, candidate: ObjectNode) -> Optional[Dict[str, object]]:
        paper_graph = getattr(self.scenegraph, "paper_graph", None)
        paper_candidate = None if paper_graph is None else paper_graph.object_nodes.get("object:%s" % int(candidate.node_id))
        if paper_candidate is None:
            return None
        subgraphs = build_object_centered_subgraphs(paper_graph)
        subgraph_scores = self.scenegraph.hcot_scorer.score(
            subgraphs,
            getattr(self.scenegraph, "obj_goal_sg", normalize_category(candidate.category)),
            graph_version=getattr(paper_graph, "version", 0),
        )
        result = self.graph_reperception.update(paper_candidate, float(candidate.confidence), subgraph_scores)
        return result.to_dict()

    def _candidate_metadata(
        self,
        candidate: ObjectNode,
        track: CandidateTrack,
        credibility: float,
        distance_to_candidate: float,
    ) -> Dict[str, object]:
        return {
            "candidate_credibility": float(credibility),
            "candidate_credibility_method": "graph_based" if str(getattr(self.scenegraph, "sgnav_mode", "")).strip().lower() == "paper" else "confidence_hits",
            "candidate_observed_count": int(candidate.observed_count),
            "candidate_track_observation_count": int(track.observation_count),
            "candidate_reperception_steps": int(track.steps),
            "candidate_rejected": bool(track.rejected),
            "candidate_accepted": bool(track.accepted),
            "candidate_distance_m": float(distance_to_candidate),
            "selected_candidate_id": int(candidate.node_id),
            "reperception": dict(track.reperception_debug),
        }

    def _is_candidate_rejected(self, candidate: ObjectNode, current_step: Optional[int]) -> bool:
        node_id = int(candidate.node_id)
        if node_id not in self.rejected_candidates:
            return False
        reject_until = int(self.rejected_candidates[node_id])
        if current_step is None:
            return True
        if int(current_step) <= reject_until:
            return True
        self.rejected_candidates.pop(node_id, None)
        return False

    def _reject_candidate(self, candidate: ObjectNode, current_step: Optional[int]) -> None:
        node_id = int(candidate.node_id)
        if current_step is None:
            reject_until = 2**31 - 1
        else:
            reject_until = int(current_step) + self.candidate_reject_ttl_steps
        self.rejected_candidates[node_id] = int(reject_until)
        if self.active_candidate_track is not None and int(self.active_candidate_track.node_id) == node_id:
            self.active_candidate_track.rejected = True
