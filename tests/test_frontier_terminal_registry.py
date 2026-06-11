from __future__ import annotations

from dataclasses import dataclass

from voxroom_online.isaac_runtime.navigation.frontier_terminal_registry import FrontierTerminalRegistry


@dataclass
class _Frontier:
    center_grid: tuple[int, int]


def test_targeted_record_not_suppressed_by_targetless_query() -> None:
    reg = FrontierTerminalRegistry(
        match_radius_cells=5,
        target_match_radius_cells=2,
        require_target_match_when_record_has_target=True,
        ignore_targetless_query_for_targeted_record=True,
    )
    reg.mark_terminal(
        center_grid=(10, 10),
        target_grid=(20, 20),
        robot_grid=(0, 0),
        reason="partial",
        status="partial_reached",
        step=0,
        suppress_steps=100,
    )

    assert not reg.is_suppressed((11, 10), None, step=1)
    assert reg.is_suppressed((11, 10), (21, 20), step=1)
    assert not reg.is_suppressed((11, 10), (30, 30), step=1)


def test_partial_reached_ttl_expires() -> None:
    reg = FrontierTerminalRegistry(match_radius_cells=5, target_match_radius_cells=2)
    reg.mark_terminal(
        center_grid=(10, 10),
        target_grid=(20, 20),
        robot_grid=(0, 0),
        reason="partial",
        status="partial_reached",
        step=10,
        suppress_steps=3,
    )

    assert reg.is_suppressed((10, 10), (20, 20), step=12)
    assert not reg.is_suppressed((10, 10), (20, 20), step=14)


def test_filter_relaxes_partial_when_all_frontiers_suppressed() -> None:
    reg = FrontierTerminalRegistry(match_radius_cells=2, target_match_radius_cells=1)
    frontiers = [_Frontier((10, 10)), _Frontier((30, 30))]
    targets = {(10, 10): (20, 20), (30, 30): (40, 40)}
    for frontier in frontiers:
        reg.mark_terminal(
            center_grid=frontier.center_grid,
            target_grid=targets[frontier.center_grid],
            robot_grid=(0, 0),
            reason="partial",
            status="partial_reached",
            step=0,
            suppress_steps=100,
        )

    result = reg.filter_frontiers_with_debug(
        frontiers,
        target_resolver=lambda f: targets[f.center_grid],
        step=1,
    )
    assert len(result.filtered) == 0
    assert result.suppressed_by_status == {"partial_reached": 2}

    relaxed = reg.filter_frontiers_with_debug(
        frontiers,
        target_resolver=lambda f: targets[f.center_grid],
        step=1,
        ignore_statuses={"partial_reached"},
    )
    assert relaxed.filtered == frontiers


def test_raw_alive_checker_unsuppresses_partial_record() -> None:
    reg = FrontierTerminalRegistry(match_radius_cells=5, target_match_radius_cells=2)
    reg.mark_terminal(
        center_grid=(10, 10),
        target_grid=(20, 20),
        robot_grid=(0, 0),
        reason="partial",
        status="partial_reached",
        step=0,
        suppress_steps=100,
    )
    frontiers = [_Frontier((10, 10))]

    result = reg.filter_frontiers_with_debug(
        frontiers,
        target_resolver=lambda _f: (20, 20),
        step=1,
        raw_alive_checker=lambda _record: True,
    )

    assert result.filtered == frontiers
    assert result.suppressed == []
