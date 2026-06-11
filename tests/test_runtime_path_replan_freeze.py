from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.graph.decision import NavigationDecision
from voxroom_online.isaac_runtime.scripts.run_one_episode import (
    active_navigation_decision_for_path_replan,
    active_target_replan_trace_extra,
    astar_no_path_failure_reason,
    frontier_exact_target_locked_for_planning,
    path_trim_debug_metadata,
    resolve_frontier_source_layers,
    trim_path_to_nearest_with_info,
)
from voxroom_online.isaac_runtime.navigation.frontier_execution_state import CommittedFrontierExecutionState


def test_stale_path_trim_failure_requests_replan_and_clears_path() -> None:
    path = [(10, 10), (10, 11), (10, 12)]
    current = (20, 20)

    trim = trim_path_to_nearest_with_info(path, current, max_dist_cells=3)
    debug = path_trim_debug_metadata(
        trim=trim,
        before_head=path[0],
        after_head=None,
        replan_requested=True,
        replan_reason="path_trim_failed_%s" % trim.reason,
        cleared_stale_path=True,
    )

    assert not trim.success
    assert trim.reason == "nearest_too_far"
    assert trim.suffix == []
    assert debug["path_replan_requested"] is True
    assert debug["path_replan_current_path_cleared_as_stale"] is True
    assert debug["path_replan_suppressed_by_roomseg_freeze"] is False
    assert debug["path_trim_failed"] is True
    assert debug["current_path_head_grid"] is None


def test_frozen_roomseg_reuses_active_target_for_path_replan() -> None:
    decision = NavigationDecision(
        mode="frontier",
        target_cells=[(12, 34)],
        stop=False,
        selected_candidate=None,
        frontier_decision=None,
        reason="continue_committed_frontier",
    )

    active = active_navigation_decision_for_path_replan(
        decision,
        update_roomseg_frontiers=False,
        allow_path_replan_during_roomseg_freeze=True,
    )

    assert active is decision
    assert active.target_cells == [(12, 34)]


def test_frozen_roomseg_does_not_reuse_when_target_reselection_is_allowed_or_disabled() -> None:
    decision = NavigationDecision(
        mode="frontier",
        target_cells=[(12, 34)],
        stop=False,
        selected_candidate=None,
        frontier_decision=None,
        reason="continue_committed_frontier",
    )

    assert active_navigation_decision_for_path_replan(decision, update_roomseg_frontiers=True) is None
    assert active_navigation_decision_for_path_replan(
        decision,
        update_roomseg_frontiers=False,
        allow_path_replan_during_roomseg_freeze=False,
    ) is None
    empty = NavigationDecision(
        mode="none",
        target_cells=[],
        stop=False,
        selected_candidate=None,
        frontier_decision=None,
        reason="empty",
    )
    assert active_navigation_decision_for_path_replan(empty, update_roomseg_frontiers=False) is None


def test_voxel_frontier_source_treats_outside_as_observed_non_frontier_non_wall() -> None:
    nav_free = np.zeros((3, 3), dtype=bool)
    nav_observed = np.zeros((3, 3), dtype=bool)
    nav_occupancy = np.zeros((3, 3), dtype=bool)
    traversible = np.ones((3, 3), dtype=bool)
    vfree = np.ones((3, 3), dtype=bool)
    vobserved = np.zeros((3, 3), dtype=bool)
    vwall = np.zeros((3, 3), dtype=bool)
    outside = np.zeros((3, 3), dtype=bool)
    outside[1, 1] = True
    vwall[1, 1] = True

    frontier_free, frontier_observed, frontier_occupancy, meta = resolve_frontier_source_layers(
        room_debug={
            "voxel_vertical_free_xy": vfree,
            "voxel_vertical_observed_xy": vobserved,
            "voxel_wall_xy": vwall,
            "voxel_outside_xy": outside,
        },
        mapper=None,
        navigation_free=nav_free,
        navigation_observed=nav_observed,
        navigation_occupancy=nav_occupancy,
        frontier_traversible=traversible,
        source="voxel_vertical_free",
        require_navigation_reachable=True,
    )

    assert not bool(frontier_free[1, 1])
    assert bool(frontier_observed[1, 1])
    assert not bool(frontier_occupancy[1, 1])
    assert meta["frontier_cells_removed_by_voxel_outside"] == 1


def test_active_target_replan_trace_extra_serializes_required_fields() -> None:
    extra = active_target_replan_trace_extra(
        {
            "path_replan_requested": True,
            "path_trim_failed": True,
            "path_trim_reason": "nearest_too_far",
        }
    )

    assert extra["path_replan_requested"] is True
    assert extra["path_trim_failed"] is True
    assert extra["path_replan_only_active_target"] is True
    assert extra["target_reselection_suppressed_by_roomseg_freeze"] is True
    assert extra["path_replan_suppressed_by_roomseg_freeze"] is False
    assert extra["path_replan_reused_committed_target"] is True


def test_active_target_astar_no_path_uses_specific_failure_reason() -> None:
    assert astar_no_path_failure_reason(path_replan_only_active_target=True) == "astar_no_path_to_active_target"
    assert astar_no_path_failure_reason(path_replan_only_active_target=False) == "astar_no_path"


def test_committed_frontier_locks_exact_target_for_planning() -> None:
    decision = NavigationDecision(
        mode="frontier",
        target_cells=[(288, 376)],
        stop=False,
        selected_candidate=None,
        frontier_decision=None,
        reason="continue_committed_frontier_execution",
        metadata={
            "frontier_execution_locked": True,
            "frontier_commitment_reason": "continue_committed_frontier_execution",
        },
    )
    execution = CommittedFrontierExecutionState()
    execution.active = True
    execution.target_cells = [(288, 376)]
    execution.actual_target_grid = (288, 376)

    assert frontier_exact_target_locked_for_planning(decision, execution) is True


def test_uncommitted_frontier_can_use_radius_goal_planning() -> None:
    decision = NavigationDecision(
        mode="frontier",
        target_cells=[(288, 376)],
        stop=False,
        selected_candidate=None,
        frontier_decision=None,
        reason="selected_new_frontier",
        metadata={},
    )
    execution = CommittedFrontierExecutionState()

    assert frontier_exact_target_locked_for_planning(decision, execution) is False
