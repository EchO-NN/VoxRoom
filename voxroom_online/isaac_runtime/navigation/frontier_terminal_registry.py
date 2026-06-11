from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

GridCell = tuple[int, int]
FrontierTerminalStatus = Literal["reached", "partial_reached", "failed_unreachable", "stale_near_fallback"]


def _grid_cell(value: object) -> GridCell | None:
    if value is None:
        return None
    try:
        if len(value) < 2:  # type: ignore[arg-type]
            return None
        return (int(value[0]), int(value[1]))  # type: ignore[index]
    except Exception:
        return None


def _dist_cells(a: GridCell, b: GridCell) -> float:
    dr = int(a[0]) - int(b[0])
    dc = int(a[1]) - int(b[1])
    return float((dr * dr + dc * dc) ** 0.5)


@dataclass
class FrontierTerminalRecord:
    key: tuple
    center_grid: GridCell
    target_grid: GridCell | None
    reason: str
    status: FrontierTerminalStatus
    step: int
    robot_grid: GridCell
    refresh_consumed: bool = False
    refresh_consumed_step: int = -1
    suppress_until_step: int = 10**9
    count: int = 1


@dataclass
class FrontierTerminalFilterResult:
    filtered: list[object] = field(default_factory=list)
    suppressed: list[object] = field(default_factory=list)
    suppressed_records: list[FrontierTerminalRecord] = field(default_factory=list)
    suppressed_by_status: dict[str, int] = field(default_factory=dict)
    target_resolved_count: int = 0
    target_missing_count: int = 0
    targetless_query_ignored_count: int = 0


@dataclass
class FrontierTerminalRegistry:
    records: dict[tuple, FrontierTerminalRecord] = field(default_factory=dict)
    match_radius_cells: int = 4
    target_match_radius_cells: int = 2
    require_target_match_when_record_has_target: bool = True
    allow_center_only_match_for_targetless_records: bool = True
    ignore_targetless_query_for_targeted_record: bool = True
    duplicate_mark_count: int = 0
    suppressed_count: int = 0
    last_key: tuple | None = None
    last_filter_result: FrontierTerminalFilterResult | None = None

    def make_key(self, center_grid: GridCell, target_grid: GridCell | None) -> tuple:
        center = _grid_cell(center_grid)
        if center is None:
            raise ValueError("center_grid is required")
        target = _grid_cell(target_grid)
        if target is None:
            return ("frontier", int(center[0]), int(center[1]), None, None)
        return ("frontier", int(center[0]), int(center[1]), int(target[0]), int(target[1]))

    def mark_terminal(
        self,
        *,
        center_grid: GridCell,
        target_grid: GridCell | None,
        robot_grid: GridCell,
        reason: str,
        status: str,
        step: int,
        suppress_steps: int = 10**9,
    ) -> tuple:
        center = _grid_cell(center_grid)
        robot = _grid_cell(robot_grid)
        if center is None or robot is None:
            raise ValueError("center_grid and robot_grid are required")
        target = _grid_cell(target_grid)
        normalized_status = str(status)
        if normalized_status not in {"reached", "partial_reached", "failed_unreachable", "stale_near_fallback"}:
            raise ValueError(f"unsupported frontier terminal status {status!r}")
        key = self.make_key(center, target)
        existing = self.records.get(key)
        if existing is None:
            self.records[key] = FrontierTerminalRecord(
                key=key,
                center_grid=center,
                target_grid=target,
                reason=str(reason),
                status=normalized_status,  # type: ignore[arg-type]
                step=int(step),
                robot_grid=robot,
                suppress_until_step=int(step) + max(0, int(suppress_steps)),
            )
        else:
            existing.count += 1
            existing.reason = str(reason)
            existing.status = normalized_status  # type: ignore[assignment]
            existing.step = int(step)
            existing.robot_grid = robot
            existing.suppress_until_step = max(
                int(existing.suppress_until_step),
                int(step) + max(0, int(suppress_steps)),
            )
            self.duplicate_mark_count += 1
        self.last_key = key
        return key

    def mark_refresh_consumed(self, key: tuple | None, step: int) -> None:
        if key is None:
            return
        record = self.records.get(tuple(key))
        if record is None:
            return
        record.refresh_consumed = True
        record.refresh_consumed_step = int(step)

    def matching_record(
        self,
        center_grid: GridCell,
        target_grid: GridCell | None,
        step: int,
        *,
        ignore_statuses: set[str] | None = None,
        raw_alive_checker: Callable[[FrontierTerminalRecord], bool] | None = None,
    ) -> FrontierTerminalRecord | None:
        center = _grid_cell(center_grid)
        if center is None:
            return None
        target = _grid_cell(target_grid)
        max_center_dist = max(0, int(self.match_radius_cells))
        max_target_dist = max(0, int(self.target_match_radius_cells))
        ignored = {str(v) for v in (ignore_statuses or set())}
        for record in self.records.values():
            if int(step) > int(record.suppress_until_step):
                continue
            if str(record.status) in ignored:
                continue
            if raw_alive_checker is not None and bool(raw_alive_checker(record)) and str(record.status) in {
                "partial_reached",
                "failed_unreachable",
                "stale_near_fallback",
            }:
                continue
            if _dist_cells(center, record.center_grid) > max_center_dist:
                continue
            if record.target_grid is not None:
                if target is None:
                    if bool(self.ignore_targetless_query_for_targeted_record):
                        continue
                    if bool(self.require_target_match_when_record_has_target):
                        continue
                    return record
                if _dist_cells(target, record.target_grid) <= max_target_dist:
                    return record
                continue
            if target is None:
                return record
            if bool(self.allow_center_only_match_for_targetless_records):
                return record
        return None

    def is_suppressed(self, center_grid: GridCell, target_grid: GridCell | None, step: int) -> bool:
        return self.matching_record(center_grid, target_grid, step) is not None

    def filter_frontiers(
        self,
        frontiers: Sequence[object],
        target_resolver: Callable[[object], GridCell | None] | None = None,
        step: int = 0,
    ) -> list[object]:
        return self.filter_frontiers_with_debug(frontiers, target_resolver=target_resolver, step=step).filtered

    def filter_frontiers_with_debug(
        self,
        frontiers: Sequence[object],
        target_resolver: Callable[[object], GridCell | None] | None = None,
        step: int = 0,
        *,
        ignore_statuses: set[str] | None = None,
        raw_alive_checker: Callable[[FrontierTerminalRecord], bool] | None = None,
    ) -> FrontierTerminalFilterResult:
        filtered: list[object] = []
        suppressed: list[object] = []
        suppressed_records: list[FrontierTerminalRecord] = []
        suppressed_by_status = {
            "reached": 0,
            "partial_reached": 0,
            "failed_unreachable": 0,
            "stale_near_fallback": 0,
        }
        target_resolved_count = 0
        target_missing_count = 0
        targetless_query_ignored_count = 0
        for frontier in frontiers:
            center = _grid_cell(getattr(frontier, "center_grid", None))
            if center is None:
                filtered.append(frontier)
                continue
            target = target_resolver(frontier) if target_resolver is not None else None
            if target is None:
                target_missing_count += 1
                if any(
                    int(step) <= int(record.suppress_until_step)
                    and record.target_grid is not None
                    and _dist_cells(center, record.center_grid) <= max(0, int(self.match_radius_cells))
                    for record in self.records.values()
                ):
                    targetless_query_ignored_count += 1
            else:
                target_resolved_count += 1
            record = self.matching_record(
                center,
                target,
                int(step),
                ignore_statuses=ignore_statuses,
                raw_alive_checker=raw_alive_checker,
            )
            if record is not None:
                self.suppressed_count += 1
                suppressed.append(frontier)
                suppressed_records.append(record)
                status = str(record.status)
                suppressed_by_status[status] = int(suppressed_by_status.get(status, 0)) + 1
                continue
            filtered.append(frontier)
        result = FrontierTerminalFilterResult(
            filtered=filtered,
            suppressed=suppressed,
            suppressed_records=suppressed_records,
            suppressed_by_status={key: int(value) for key, value in suppressed_by_status.items() if int(value) > 0},
            target_resolved_count=int(target_resolved_count),
            target_missing_count=int(target_missing_count),
            targetless_query_ignored_count=int(targetless_query_ignored_count),
        )
        self.last_filter_result = result
        return result

    def status_counts(self) -> dict[str, int]:
        counts = {
            "reached": 0,
            "partial_reached": 0,
            "failed_unreachable": 0,
            "stale_near_fallback": 0,
        }
        for record in self.records.values():
            counts[str(record.status)] = int(counts.get(str(record.status), 0)) + 1
        return counts

    def debug_metadata(self) -> dict[str, object]:
        counts = self.status_counts()
        last = self.records.get(self.last_key) if self.last_key is not None else None
        meta: dict[str, object] = {
            "frontier_terminal_registry_size": int(len(self.records)),
            "frontier_terminal_reached_count": int(counts.get("reached", 0)),
            "frontier_terminal_partial_reached_count": int(counts.get("partial_reached", 0)),
            "frontier_terminal_failed_count": int(counts.get("failed_unreachable", 0)),
            "frontier_terminal_near_fallback_count": int(counts.get("stale_near_fallback", 0)),
            "frontier_terminal_suppressed_total": int(self.suppressed_count),
            "frontier_terminal_duplicate_mark_count": int(self.duplicate_mark_count),
            "frontier_terminal_last_key": list(self.last_key) if self.last_key is not None else None,
            "frontier_terminal_last_status": None if last is None else str(last.status),
            "frontier_terminal_last_reason": None if last is None else str(last.reason),
            "frontier_terminal_target_aware_matching": bool(self.require_target_match_when_record_has_target),
        }
        if self.last_filter_result is not None:
            result = self.last_filter_result
            meta.update(
                {
                    "frontier_terminal_suppressed_by_status": dict(result.suppressed_by_status),
                    "frontier_terminal_target_resolved_count": int(result.target_resolved_count),
                    "frontier_terminal_target_missing_count": int(result.target_missing_count),
                    "frontier_terminal_targetless_query_ignored_count": int(result.targetless_query_ignored_count),
                }
            )
        return meta
