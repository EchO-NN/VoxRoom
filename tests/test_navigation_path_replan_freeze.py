from __future__ import annotations

from voxroom_online.isaac_runtime.scripts.run_one_episode import (
    path_trim_debug_metadata,
    trim_path_to_nearest_with_info,
)


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


def test_near_path_trim_keeps_suffix_without_replan() -> None:
    path = [(10, 10), (10, 11), (10, 12)]

    trim = trim_path_to_nearest_with_info(path, (10, 11), max_dist_cells=3)

    assert trim.success
    assert trim.reason == "exact_match"
    assert trim.suffix == [(10, 11), (10, 12)]
