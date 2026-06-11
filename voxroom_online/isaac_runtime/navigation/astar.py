from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

GridCell = Tuple[int, int]


@dataclass
class AStarResult:
    path: List[GridCell]
    length_m: float
    reached_goal: Optional[GridCell] = None


class GridAStarPlanner:
    def __init__(self, traversible: np.ndarray, resolution_m: float, allow_diagonal: bool = True):
        self.traversible = np.asarray(traversible).astype(bool)
        self.resolution_m = float(resolution_m)
        self.allow_diagonal = bool(allow_diagonal)
        self.height, self.width = self.traversible.shape
        self.neighbors = self._build_neighbors()

    def _build_neighbors(self) -> List[Tuple[int, int, float]]:
        steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
        if self.allow_diagonal:
            root2 = math.sqrt(2.0)
            steps += [(-1, -1, root2), (-1, 1, root2), (1, -1, root2), (1, 1, root2)]
        return [(dr, dc, cost * self.resolution_m) for dr, dc, cost in steps]

    def can_step(self, cell: GridCell, dr: int, dc: int) -> bool:
        nbr = (int(cell[0]) + int(dr), int(cell[1]) + int(dc))
        if not self.is_free(nbr):
            return False
        if int(dr) != 0 and int(dc) != 0:
            # Prevent diagonal corner cutting through two touching obstacles/walls.
            if not self.is_free((int(cell[0]) + int(dr), int(cell[1]))):
                return False
            if not self.is_free((int(cell[0]), int(cell[1]) + int(dc))):
                return False
        return True

    def inside(self, cell: GridCell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, cell: GridCell) -> bool:
        r, c = cell
        return self.inside(cell) and bool(self.traversible[r, c])

    def snap_to_free(self, cell: GridCell, max_radius: int = 20) -> Optional[GridCell]:
        if self.is_free(cell):
            return int(cell[0]), int(cell[1])
        r0, c0 = int(cell[0]), int(cell[1])
        best = None
        best_dist = float("inf")
        for radius in range(1, int(max_radius) + 1):
            for r in range(r0 - radius, r0 + radius + 1):
                for c in range(c0 - radius, c0 + radius + 1):
                    if not self.is_free((r, c)):
                        continue
                    dist = (r - r0) ** 2 + (c - c0) ** 2
                    if dist < best_dist:
                        best = (r, c)
                        best_dist = dist
            if best is not None:
                return best
        return None

    def _goal_set(self, goal: Union[GridCell, Iterable[GridCell]]) -> set:
        if isinstance(goal, tuple) and len(goal) == 2 and isinstance(goal[0], (int, np.integer)):
            cells = [goal]
        else:
            cells = list(goal)  # type: ignore[arg-type]
        return {(int(r), int(c)) for r, c in cells if self.is_free((int(r), int(c)))}

    @staticmethod
    def _heuristic(cell: GridCell, goals: Sequence[GridCell], resolution_m: float) -> float:
        if not goals:
            return 0.0
        r, c = cell
        return min(math.hypot(r - gr, c - gc) * resolution_m for gr, gc in goals)

    def plan(self, start: GridCell, goal: Union[GridCell, Iterable[GridCell]]) -> AStarResult:
        start_free = self.snap_to_free((int(start[0]), int(start[1])))
        goals = self._goal_set(goal)
        if start_free is None or not goals:
            return AStarResult([], float("inf"), None)
        if start_free in goals:
            return AStarResult([start_free], 0.0, start_free)

        goal_list = list(goals)
        open_heap: List[Tuple[float, int, GridCell]] = []
        seq = 0
        heapq.heappush(open_heap, (self._heuristic(start_free, goal_list, self.resolution_m), seq, start_free))
        came_from: Dict[GridCell, GridCell] = {}
        g_score: Dict[GridCell, float] = {start_free: 0.0}
        closed: set[GridCell] = set()

        reached: Optional[GridCell] = None
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current in goals:
                reached = current
                break
            closed.add(current)

            for dr, dc, step_cost in self.neighbors:
                nbr = (current[0] + dr, current[1] + dc)
                if not self.can_step(current, dr, dc) or nbr in closed:
                    continue
                tentative = g_score[current] + step_cost
                if tentative >= g_score.get(nbr, float("inf")):
                    continue
                came_from[nbr] = current
                g_score[nbr] = tentative
                seq += 1
                f_score = tentative + self._heuristic(nbr, goal_list, self.resolution_m)
                heapq.heappush(open_heap, (f_score, seq, nbr))

        if reached is None:
            return AStarResult([], float("inf"), None)

        path = [reached]
        while path[-1] != start_free:
            path.append(came_from[path[-1]])
        path.reverse()
        return AStarResult(path, g_score[reached], reached)

    def distance(self, start: GridCell, goal: Union[GridCell, Iterable[GridCell]]) -> float:
        return self.plan(start, goal).length_m


class ClearanceAStarPlanner(GridAStarPlanner):
    def __init__(
        self,
        traversible: np.ndarray,
        resolution_m: float,
        occupied: np.ndarray | None = None,
        allow_diagonal: bool = True,
        clearance_desired_m: float = 0.25,
        clearance_weight: float = 3.0,
        clearance_power: float = 2.0,
        clearance_hard_min_m: float = 0.0,
    ):
        super().__init__(traversible, resolution_m, allow_diagonal=allow_diagonal)
        self.clearance_desired_m = max(1e-6, float(clearance_desired_m))
        self.clearance_weight = max(0.0, float(clearance_weight))
        self.clearance_power = max(0.1, float(clearance_power))
        self.clearance_hard_min_m = max(0.0, float(clearance_hard_min_m))
        hard_free = np.asarray(self.traversible, dtype=bool)
        if occupied is not None:
            hard_free = hard_free & ~np.asarray(occupied, dtype=bool)
        self.clearance_m = _distance_transform_clearance_m(hard_free, self.resolution_m)

    def clearance_penalty(self, cell: GridCell) -> float:
        if not self.inside(cell):
            return float("inf")
        d = float(self.clearance_m[int(cell[0]), int(cell[1])])
        if self.clearance_hard_min_m > 0.0 and d < self.clearance_hard_min_m:
            return float("inf")
        deficit = max(0.0, self.clearance_desired_m - d) / self.clearance_desired_m
        if deficit <= 0.0 or self.clearance_weight <= 0.0:
            return 0.0
        return float(self.clearance_weight * (deficit ** self.clearance_power))

    def plan(self, start: GridCell, goal: Union[GridCell, Iterable[GridCell]]) -> AStarResult:
        start_free = self.snap_to_free((int(start[0]), int(start[1])))
        goals = self._goal_set(goal)
        if start_free is None or not goals:
            return AStarResult([], float("inf"), None)
        if start_free in goals:
            return AStarResult([start_free], 0.0, start_free)

        goal_list = list(goals)
        open_heap: List[Tuple[float, int, GridCell]] = []
        seq = 0
        heapq.heappush(open_heap, (self._heuristic(start_free, goal_list, self.resolution_m), seq, start_free))
        came_from: Dict[GridCell, GridCell] = {}
        g_score: Dict[GridCell, float] = {start_free: 0.0}
        closed: set[GridCell] = set()

        reached: Optional[GridCell] = None
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current in goals:
                reached = current
                break
            closed.add(current)

            for dr, dc, step_cost in self.neighbors:
                nbr = (current[0] + dr, current[1] + dc)
                if not self.can_step(current, dr, dc) or nbr in closed:
                    continue
                penalty = self.clearance_penalty(nbr)
                if not math.isfinite(penalty):
                    continue
                tentative = g_score[current] + step_cost * (1.0 + penalty)
                if tentative >= g_score.get(nbr, float("inf")):
                    continue
                came_from[nbr] = current
                g_score[nbr] = tentative
                seq += 1
                f_score = tentative + self._heuristic(nbr, goal_list, self.resolution_m)
                heapq.heappush(open_heap, (f_score, seq, nbr))

        if reached is None:
            return AStarResult([], float("inf"), None)

        path = [reached]
        while path[-1] != start_free:
            path.append(came_from[path[-1]])
        path.reverse()
        return AStarResult(path, g_score[reached], reached)


def _distance_transform_clearance_m(traversible: np.ndarray, resolution_m: float) -> np.ndarray:
    free = np.asarray(traversible, dtype=bool)
    try:
        from scipy import ndimage

        return ndimage.distance_transform_edt(free).astype(np.float32) * float(resolution_m)
    except Exception:
        dist = np.full(free.shape, np.inf, dtype=np.float32)
        queue: list[GridCell] = []
        h, w = free.shape
        for r in range(h):
            for c in range(w):
                if not free[r, c]:
                    dist[r, c] = 0.0
                    queue.append((r, c))
        if not queue:
            return np.full(free.shape, max(h, w) * float(resolution_m), dtype=np.float32)
        head = 0
        while head < len(queue):
            r, c = queue[head]
            head += 1
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] > dist[r, c] + float(resolution_m):
                    dist[nr, nc] = dist[r, c] + float(resolution_m)
                    queue.append((nr, nc))
        return dist


def astar_distance_map(traversible: np.ndarray, start: GridCell, resolution_m: float, allow_diagonal: bool = True) -> np.ndarray:
    planner = GridAStarPlanner(traversible, resolution_m, allow_diagonal=allow_diagonal)
    start_free = planner.snap_to_free(start)
    dist = np.full(planner.traversible.shape, np.inf, dtype=np.float64)
    if start_free is None:
        return dist
    heap: List[Tuple[float, GridCell]] = [(0.0, start_free)]
    dist[start_free] = 0.0
    while heap:
        cur_dist, cell = heapq.heappop(heap)
        if cur_dist > float(dist[cell]) + 1e-9:
            continue
        for dr, dc, step_cost in planner.neighbors:
            nbr = (cell[0] + dr, cell[1] + dc)
            if not planner.can_step(cell, dr, dc):
                continue
            nd = cur_dist + step_cost
            if nd < dist[nbr]:
                dist[nbr] = nd
                heapq.heappush(heap, (nd, nbr))
    return dist


def astar_distance_from_region_map(traversible: np.ndarray, goals: Iterable[GridCell], resolution_m: float, allow_diagonal: bool = True) -> np.ndarray:
    planner = GridAStarPlanner(traversible, resolution_m, allow_diagonal=allow_diagonal)
    dist = np.full(planner.traversible.shape, np.inf, dtype=np.float64)
    heap: List[Tuple[float, GridCell]] = []
    for goal in goals:
        cell = (int(goal[0]), int(goal[1]))
        if not planner.is_free(cell):
            continue
        if dist[cell] == 0.0:
            continue
        dist[cell] = 0.0
        heapq.heappush(heap, (0.0, cell))
    while heap:
        cur_dist, cell = heapq.heappop(heap)
        if cur_dist > float(dist[cell]) + 1e-9:
            continue
        for dr, dc, step_cost in planner.neighbors:
            nbr = (cell[0] + dr, cell[1] + dc)
            if not planner.can_step(cell, dr, dc):
                continue
            nd = cur_dist + step_cost
            if nd < dist[nbr]:
                dist[nbr] = nd
                heapq.heappush(heap, (nd, nbr))
    return dist
