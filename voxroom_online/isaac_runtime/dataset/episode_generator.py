from __future__ import annotations

import json
import math
import random
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from voxroom_online.isaac_runtime.dataset.category_normalizer import normalize_category
from voxroom_online.isaac_runtime.mapping.coordinate_transform import MapInfo, grid_to_world_xy, is_inside_grid, world_xy_to_grid
from voxroom_online.isaac_runtime.navigation.astar import GridAStarPlanner

GridCell = Tuple[int, int]
START_CLEARANCE_EXCLUDE_CATEGORIES = {
    "floor",
    "ceiling",
    "roof",
    "light_switch",
    "outlet",
    "door_frame",
    "doorway",
    "doorsill",
    "door_handle",
    "entrance",
    "entryway",
    "opening",
    "passage",
}


@dataclass
class EpisodeGenerationConfig:
    episodes_per_scene: int = 100
    min_start_goal_distance_m: float = 1.5
    max_start_goal_distance_m: float = 25.0
    min_start_object_clearance_m: float = 0.80
    min_start_goal_bbox_distance_m: float = 1.5
    success_distance_m: float = 1.0
    max_steps: int = 500
    max_time_s: float = 300.0
    max_attempts_per_episode: int = 2000
    seed: int = 123


def load_map_info(path: str) -> MapInfo:
    with open(path, "r", encoding="utf-8") as handle:
        return MapInfo.from_dict(json.load(handle))


def point_to_bbox_2d_distance(x: float, y: float, bbox_min: Sequence[float], bbox_max: Sequence[float]) -> float:
    dx = max(float(bbox_min[0]) - x, 0.0, x - float(bbox_max[0]))
    dy = max(float(bbox_min[1]) - y, 0.0, y - float(bbox_max[1]))
    return math.hypot(dx, dy)


def min_bbox_2d_distance(x: float, y: float, objects: Sequence[Mapping]) -> float:
    distances = []
    for obj in objects:
        bbox_min = obj.get("bbox_min_world")
        bbox_max = obj.get("bbox_max_world")
        if bbox_min is None or bbox_max is None:
            continue
        distances.append(point_to_bbox_2d_distance(x, y, bbox_min, bbox_max))
    return min(distances) if distances else math.inf


def filter_start_clearance_objects(objects: Sequence[Mapping], min_z: float = 0.05, max_z: float = 1.50) -> List[Mapping]:
    filtered = []
    for obj in objects:
        category = normalize_category(obj.get("category", "unknown"))
        if category in START_CLEARANCE_EXCLUDE_CATEGORIES:
            continue
        bbox_min = obj.get("bbox_min_world")
        bbox_max = obj.get("bbox_max_world")
        if bbox_min is None or bbox_max is None:
            continue
        if float(bbox_max[2]) < min_z or float(bbox_min[2]) > max_z:
            continue
        filtered.append(obj)
    return filtered


def build_clearance_mask(
    navigable: np.ndarray,
    map_info: MapInfo,
    clearance_objects: Sequence[Mapping],
    min_clearance_m: float,
) -> np.ndarray:
    base = np.asarray(navigable).astype(bool)
    if float(min_clearance_m) <= 0.0 or not clearance_objects:
        return base
    safe = np.zeros_like(base, dtype=bool)
    for r, c in np.argwhere(base):
        rr, cc = int(r), int(c)
        x, y = grid_to_world_xy(rr, cc, map_info)
        if min_bbox_2d_distance(x, y, clearance_objects) >= float(min_clearance_m):
            safe[rr, cc] = True
    return safe


def build_goal_region(
    obj: Mapping,
    navigable: np.ndarray,
    map_info: MapInfo,
    success_distance_m: float = 1.0,
    clearance_objects: Optional[Sequence[Mapping]] = None,
    min_goal_clearance_m: float = 0.0,
) -> List[GridCell]:
    center = obj["center_world"]
    center_rc = world_xy_to_grid(float(center[0]), float(center[1]), map_info)
    max_radius = int(math.ceil(max(success_distance_m, 1.5) / map_info.resolution_m))
    cells: List[GridCell] = []
    bbox_min = obj["bbox_min_world"]
    bbox_max = obj["bbox_max_world"]
    for dr in range(-max_radius, max_radius + 1):
        for dc in range(-max_radius, max_radius + 1):
            r, c = center_rc[0] + dr, center_rc[1] + dc
            if not is_inside_grid(r, c, map_info) or not bool(navigable[r, c]):
                continue
            wx, wy = grid_to_world_xy(r, c, map_info)
            if clearance_objects and min_bbox_2d_distance(wx, wy, clearance_objects) < min_goal_clearance_m:
                continue
            dist = point_to_bbox_2d_distance(wx, wy, bbox_min, bbox_max)
            if dist <= success_distance_m:
                cells.append((r, c))
    if cells:
        return cells
    # Controlled fallback for big/odd assets: accept cells in an expanded shell.
    expanded = success_distance_m + 0.5
    for dr in range(-max_radius, max_radius + 1):
        for dc in range(-max_radius, max_radius + 1):
            r, c = center_rc[0] + dr, center_rc[1] + dc
            if not is_inside_grid(r, c, map_info) or not bool(navigable[r, c]):
                continue
            wx, wy = grid_to_world_xy(r, c, map_info)
            if clearance_objects and min_bbox_2d_distance(wx, wy, clearance_objects) < min_goal_clearance_m:
                continue
            if point_to_bbox_2d_distance(wx, wy, bbox_min, bbox_max) <= expanded:
                cells.append((r, c))
    return cells


def sample_start_pose(
    navigable: np.ndarray,
    goal_cells: Sequence[GridCell],
    planner: GridAStarPlanner,
    map_info: MapInfo,
    rng: random.Random,
    cfg: EpisodeGenerationConfig,
    robot_spawn_height_m: float = 0.05,
    distance_map: Optional[np.ndarray] = None,
    all_objects: Optional[Sequence[Mapping]] = None,
    goal_objects: Optional[Sequence[Mapping]] = None,
) -> Optional[Tuple[List[float], GridCell, float, float, float]]:
    free = np.argwhere(navigable.astype(bool))
    if len(free) == 0:
        return None
    for _ in range(cfg.max_attempts_per_episode):
        idx = rng.randrange(len(free))
        r, c = int(free[idx][0]), int(free[idx][1])
        dist = float(distance_map[r, c]) if distance_map is not None else planner.distance((r, c), goal_cells)
        if not math.isfinite(dist):
            continue
        if dist < cfg.min_start_goal_distance_m or dist > cfg.max_start_goal_distance_m:
            continue
        x, y = grid_to_world_xy(r, c, map_info)
        object_clearance = min_bbox_2d_distance(x, y, all_objects or [])
        if object_clearance < cfg.min_start_object_clearance_m:
            continue
        goal_bbox_distance = min_bbox_2d_distance(x, y, goal_objects or [])
        if goal_bbox_distance < cfg.min_start_goal_bbox_distance_m:
            continue
        yaw = rng.uniform(-math.pi, math.pi)
        return [float(x), float(y), float(robot_spawn_height_m), float(yaw)], (r, c), float(dist), float(object_clearance), float(goal_bbox_distance)
    return None


def group_goal_objects(objects: Sequence[Mapping], exclude_categories: Optional[Iterable[str]] = None) -> Dict[str, List[Mapping]]:
    exclude = set(exclude_categories or [])
    groups: Dict[str, List[Mapping]] = defaultdict(list)
    for obj in objects:
        cat = str(obj.get("category", "unknown"))
        if cat in exclude or cat == "unknown":
            continue
        groups[cat].append(obj)
    return groups


def bfs_distance_from_region(traversible: np.ndarray, goals: Sequence[GridCell], resolution_m: float) -> np.ndarray:
    steps = np.full(traversible.shape, -1, dtype=np.int32)
    q = deque()
    h, w = traversible.shape
    for r, c in goals:
        r, c = int(r), int(c)
        if 0 <= r < h and 0 <= c < w and bool(traversible[r, c]):
            if steps[r, c] < 0:
                steps[r, c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        ns = int(steps[r, c]) + 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(traversible[rr, cc]) and steps[rr, cc] < 0:
                steps[rr, cc] = ns
                q.append((rr, cc))
    dist = steps.astype(np.float32) * float(resolution_m)
    dist[steps < 0] = np.inf
    return dist


def dijkstra_distance_from_region(traversible: np.ndarray, goals: Sequence[GridCell], resolution_m: float, allow_diagonal: bool = True) -> np.ndarray:
    from voxroom_online.isaac_runtime.navigation.astar import GridAStarPlanner

    planner = GridAStarPlanner(traversible, resolution_m, allow_diagonal=allow_diagonal)
    dist = np.full(traversible.shape, np.inf, dtype=np.float64)
    heap: List[Tuple[float, GridCell]] = []
    for r, c in goals:
        r, c = int(r), int(c)
        if planner.is_free((r, c)) and not np.isfinite(dist[r, c]):
            dist[r, c] = 0.0
            heapq.heappush(heap, (0.0, (r, c)))
    while heap:
        cost, (r, c) = heapq.heappop(heap)
        if cost > float(dist[r, c]):
            continue
        for dr, dc, step_cost in planner.neighbors:
            rr, cc = r + dr, c + dc
            if not planner.can_step((r, c), dr, dc):
                continue
            new_cost = cost + step_cost
            if new_cost < float(dist[rr, cc]):
                dist[rr, cc] = new_cost
                heapq.heappush(heap, (new_cost, (rr, cc)))
    return dist


def generate_scene_episodes(
    scene_dir: str,
    episodes_per_scene: int = 100,
    seed: int = 123,
    exclude_categories: Optional[Iterable[str]] = None,
    success_distance_m: float = 1.0,
    min_start_goal_distance_m: float = 1.5,
    max_start_goal_distance_m: float = 25.0,
    min_start_object_clearance_m: float = 0.80,
    min_start_goal_bbox_distance_m: float = 1.5,
    min_goal_clearance_m: float = 0.0,
    min_planning_clearance_m: float = 0.0,
    max_steps: int = 500,
    max_time_s: float = 300.0,
    max_attempts_per_episode: int = 2000,
    robot_spawn_height_m: float = 0.05,
) -> List[dict]:
    scene_path = Path(scene_dir)
    scene_id = scene_path.name
    with open(scene_path / "objects.json", "r", encoding="utf-8") as handle:
        objects = json.load(handle)
    objects_all_path = scene_path / "objects_all.json"
    if objects_all_path.exists():
        with open(objects_all_path, "r", encoding="utf-8") as handle:
            objects_for_start_clearance = filter_start_clearance_objects(json.load(handle))
    else:
        objects_for_start_clearance = filter_start_clearance_objects(objects)
    if not objects_for_start_clearance:
        objects_for_start_clearance = objects
    with open(scene_path / "scene_metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    map_info = load_map_info(str(scene_path / "map_info.json"))
    navigable = np.load(scene_path / "navigable.npy").astype(bool)
    navigable = build_clearance_mask(navigable, map_info, objects_for_start_clearance, min_planning_clearance_m)
    planner = GridAStarPlanner(navigable, map_info.resolution_m, allow_diagonal=True)
    cfg = EpisodeGenerationConfig(
        episodes_per_scene=episodes_per_scene,
        min_start_goal_distance_m=min_start_goal_distance_m,
        max_start_goal_distance_m=max_start_goal_distance_m,
        min_start_object_clearance_m=min_start_object_clearance_m,
        min_start_goal_bbox_distance_m=min_start_goal_bbox_distance_m,
        success_distance_m=success_distance_m,
        max_steps=max_steps,
        max_time_s=max_time_s,
        max_attempts_per_episode=max_attempts_per_episode,
        seed=seed,
    )
    scene_seed = seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(scene_id))
    rng = random.Random(scene_seed)
    groups = group_goal_objects(objects, exclude_categories=exclude_categories)

    goal_regions_by_category: Dict[str, List[GridCell]] = {}
    valid_objects_by_category: Dict[str, List[Mapping]] = {}
    distance_map_by_category: Dict[str, np.ndarray] = {}
    for cat, cat_objects in groups.items():
        cat_cells: List[GridCell] = []
        valid_objects: List[Mapping] = []
        for obj in cat_objects:
            cells = build_goal_region(
                obj,
                navigable,
                map_info,
                success_distance_m,
                clearance_objects=objects_for_start_clearance,
                min_goal_clearance_m=min_goal_clearance_m,
            )
            if not cells:
                continue
            cat_cells.extend(cells)
            valid_objects.append(obj)
        if cat_cells:
            # Deduplicate while preserving order.
            seen = set()
            deduped = []
            for cell in cat_cells:
                if cell not in seen:
                    seen.add(cell)
                    deduped.append(cell)
            goal_regions_by_category[cat] = deduped
            valid_objects_by_category[cat] = valid_objects
            distance_map_by_category[cat] = bfs_distance_from_region(navigable, deduped, map_info.resolution_m)

    categories = sorted(goal_regions_by_category)
    if not categories:
        return []
    episodes: List[dict] = []
    category_queue = deque(categories)
    attempts = 0
    while len(episodes) < episodes_per_scene and attempts < episodes_per_scene * max_attempts_per_episode:
        attempts += 1
        cat = category_queue[0]
        category_queue.rotate(-1)
        goal_cells = goal_regions_by_category[cat]
        start = sample_start_pose(
            navigable,
            goal_cells,
            planner,
            map_info,
            rng,
            cfg,
            robot_spawn_height_m=robot_spawn_height_m,
            distance_map=distance_map_by_category[cat],
            all_objects=objects_for_start_clearance,
            goal_objects=valid_objects_by_category[cat],
        )
        if start is None:
            continue
        start_pose, start_grid, shortest, object_clearance, goal_bbox_distance = start
        exact_shortest = planner.distance(start_grid, goal_cells)
        if not math.isfinite(exact_shortest):
            continue
        if exact_shortest < cfg.min_start_goal_distance_m or exact_shortest > cfg.max_start_goal_distance_m:
            continue
        cat_objects = valid_objects_by_category[cat]
        nearest_instance = None
        nearest_center_dist = float("inf")
        sx, sy = grid_to_world_xy(start_grid[0], start_grid[1], map_info)
        for obj in cat_objects:
            center = obj.get("center_world", [0.0, 0.0, 0.0])
            dist = math.hypot(float(center[0]) - sx, float(center[1]) - sy)
            if dist < nearest_center_dist:
                nearest_center_dist = dist
                nearest_instance = obj.get("instance_id")
        episode_idx = len(episodes)
        episodes.append(
            {
                "version": "interioragent_objectnav_episode_v1",
                "episode_id": "%s_%06d" % (scene_id, episode_idx),
                "scene_id": scene_id,
                "usd_path": metadata["usd_path"],
                "rooms_json_path": metadata["rooms_json_path"],
                "preprocessed_scene_dir": str(scene_path),
                "goal_category": cat,
                "goal_instance_ids": [str(obj.get("instance_id")) for obj in cat_objects],
                "goal_centers_world": [obj.get("center_world") for obj in cat_objects],
                "goal_regions_grid": [[int(r), int(c)] for r, c in goal_cells],
                "start_pose_world": start_pose,
                "start_grid": [int(start_grid[0]), int(start_grid[1])],
                "start_object_clearance_m": float(object_clearance),
                "start_goal_bbox_distance_m": float(goal_bbox_distance),
                "nearest_goal_instance_id": nearest_instance,
                "shortest_path_distance_m": float(exact_shortest),
                "success_distance_m": float(success_distance_m),
                "max_steps": int(max_steps),
                "max_time_s": float(max_time_s),
                "metadata": {
                    "map_resolution_m": float(map_info.resolution_m),
                    "generator_seed": int(seed),
                    "min_start_object_clearance_m": float(min_start_object_clearance_m),
                    "min_start_goal_bbox_distance_m": float(min_start_goal_bbox_distance_m),
                    "min_goal_clearance_m": float(min_goal_clearance_m),
                    "min_planning_clearance_m": float(min_planning_clearance_m),
                    "start_clearance_object_count": int(len(objects_for_start_clearance)),
                    "start_clearance_excluded_categories": sorted(START_CLEARANCE_EXCLUDE_CATEGORIES),
                },
            }
        )
    return episodes


def select_longest_instance_astar_episode(
    scene_dir: str,
    seed: int = 123,
    exclude_categories: Optional[Iterable[str]] = None,
    success_distance_m: float = 1.0,
    min_start_goal_distance_m: float = 0.0,
    max_start_goal_distance_m: float = 100.0,
    min_start_object_clearance_m: float = 0.80,
    min_start_goal_bbox_distance_m: float = 1.5,
    min_goal_clearance_m: float = 0.80,
    min_planning_clearance_m: float = 0.80,
    max_steps: int = 500,
    max_time_s: float = 300.0,
    robot_spawn_height_m: float = 0.05,
    top_k: int = 10,
) -> Tuple[dict, List[dict]]:
    scene_path = Path(scene_dir)
    scene_id = scene_path.name
    with open(scene_path / "objects.json", "r", encoding="utf-8") as handle:
        objects = json.load(handle)
    objects_all_path = scene_path / "objects_all.json"
    if objects_all_path.exists():
        with open(objects_all_path, "r", encoding="utf-8") as handle:
            objects_for_start_clearance = filter_start_clearance_objects(json.load(handle))
    else:
        objects_for_start_clearance = filter_start_clearance_objects(objects)
    if not objects_for_start_clearance:
        objects_for_start_clearance = objects
    with open(scene_path / "scene_metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    map_info = load_map_info(str(scene_path / "map_info.json"))
    navigable = np.load(scene_path / "navigable.npy").astype(bool)
    navigable = build_clearance_mask(navigable, map_info, objects_for_start_clearance, min_planning_clearance_m)
    planner = GridAStarPlanner(navigable, map_info.resolution_m, allow_diagonal=True)
    exclude = set(exclude_categories or [])

    free_cells = [(int(r), int(c)) for r, c in np.argwhere(navigable)]
    start_clearance_by_cell: Dict[GridCell, float] = {}
    for r, c in free_cells:
        x, y = grid_to_world_xy(r, c, map_info)
        start_clearance_by_cell[(r, c)] = min_bbox_2d_distance(x, y, objects_for_start_clearance)

    rng = random.Random(seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(scene_id)) + 7919)
    candidates: List[dict] = []
    for obj in objects:
        category = str(obj.get("category", "unknown"))
        if category in exclude or category == "unknown":
            continue
        goal_cells = build_goal_region(
            obj,
            navigable,
            map_info,
            success_distance_m,
            clearance_objects=objects_for_start_clearance,
            min_goal_clearance_m=min_goal_clearance_m,
        )
        if not goal_cells:
            continue
        distance_map = dijkstra_distance_from_region(navigable, goal_cells, map_info.resolution_m, allow_diagonal=True)
        best_start: Optional[GridCell] = None
        best_distance = -math.inf
        best_object_clearance = math.inf
        best_goal_bbox_distance = math.inf
        shuffled_free = list(free_cells)
        rng.shuffle(shuffled_free)
        for r, c in shuffled_free:
            distance = float(distance_map[r, c])
            if not math.isfinite(distance) or distance < min_start_goal_distance_m or distance > max_start_goal_distance_m:
                continue
            object_clearance = start_clearance_by_cell[(r, c)]
            if object_clearance < min_start_object_clearance_m:
                continue
            x, y = grid_to_world_xy(r, c, map_info)
            goal_bbox_distance = min_bbox_2d_distance(x, y, [obj])
            if goal_bbox_distance < min_start_goal_bbox_distance_m:
                continue
            if distance > best_distance:
                best_start = (r, c)
                best_distance = distance
                best_object_clearance = object_clearance
                best_goal_bbox_distance = goal_bbox_distance
        if best_start is None:
            continue
        plan = planner.plan(best_start, goal_cells)
        if not plan.path or not math.isfinite(plan.length_m):
            continue
        exact_distance = float(plan.length_m)
        if exact_distance < min_start_goal_distance_m or exact_distance > max_start_goal_distance_m:
            continue
        center = obj.get("center_world", [0.0, 0.0, 0.0])
        sx, sy = grid_to_world_xy(best_start[0], best_start[1], map_info)
        yaw = math.atan2(float(center[1]) - sy, float(center[0]) - sx)
        candidates.append(
            {
                "scene_id": scene_id,
                "category": category,
                "instance_id": str(obj.get("instance_id")),
                "start_grid": [int(best_start[0]), int(best_start[1])],
                "start_pose_world": [float(sx), float(sy), float(robot_spawn_height_m), float(yaw)],
                "shortest_path_distance_m": exact_distance,
                "dijkstra_distance_m": float(best_distance),
                "start_object_clearance_m": float(best_object_clearance),
                "start_goal_bbox_distance_m": float(best_goal_bbox_distance),
                "goal_cells_count": int(len(goal_cells)),
                "goal_center_world": center,
                "goal_regions_grid": [[int(r), int(c)] for r, c in goal_cells],
                "path_grid": [[int(r), int(c)] for r, c in plan.path],
            }
        )
    candidates.sort(key=lambda row: (float(row["shortest_path_distance_m"]), float(row["start_goal_bbox_distance_m"])), reverse=True)
    if not candidates:
        raise ValueError("No valid instance/start pair found for longest-instance A* selection")
    selected = candidates[0]
    episode = {
        "version": "interioragent_objectnav_episode_v1",
        "episode_id": "%s_longest_instance_astar_000000" % scene_id,
        "scene_id": scene_id,
        "usd_path": metadata["usd_path"],
        "rooms_json_path": metadata["rooms_json_path"],
        "preprocessed_scene_dir": str(scene_path),
        "goal_category": selected["category"],
        "goal_instance_ids": [selected["instance_id"]],
        "goal_centers_world": [selected["goal_center_world"]],
        "goal_regions_grid": selected["goal_regions_grid"],
        "start_pose_world": selected["start_pose_world"],
        "start_grid": selected["start_grid"],
        "start_object_clearance_m": selected["start_object_clearance_m"],
        "start_goal_bbox_distance_m": selected["start_goal_bbox_distance_m"],
        "nearest_goal_instance_id": selected["instance_id"],
        "shortest_path_distance_m": float(selected["shortest_path_distance_m"]),
        "success_distance_m": float(success_distance_m),
        "max_steps": int(max_steps),
        "max_time_s": float(max_time_s),
        "metadata": {
            "map_resolution_m": float(map_info.resolution_m),
            "generator_seed": int(seed),
            "selection_mode": "longest_instance_astar",
            "evaluated_instance_count": int(len(candidates)),
            "min_start_object_clearance_m": float(min_start_object_clearance_m),
            "min_start_goal_bbox_distance_m": float(min_start_goal_bbox_distance_m),
            "min_goal_clearance_m": float(min_goal_clearance_m),
            "min_planning_clearance_m": float(min_planning_clearance_m),
            "min_start_goal_distance_m": float(min_start_goal_distance_m),
            "max_start_goal_distance_m": float(max_start_goal_distance_m),
            "start_clearance_object_count": int(len(objects_for_start_clearance)),
            "start_clearance_excluded_categories": sorted(START_CLEARANCE_EXCLUDE_CATEGORIES),
            "top_candidates": [
                {key: value for key, value in candidate.items() if key not in {"goal_regions_grid", "path_grid"}}
                for candidate in candidates[: max(1, int(top_k))]
            ],
        },
    }
    return episode, candidates


def write_jsonl(rows: Iterable[Mapping], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
