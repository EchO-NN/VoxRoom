from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HoughRawDoorSeedResult:
    raw_points_xy: tuple[tuple[int, int], ...]
    paired_lines_xyxy: tuple[tuple[int, int, int, int], ...]
    centerline_mask: np.ndarray
    raw_seed_mask: np.ndarray
    debug: dict[str, object]


def reconstruct_hough_raw_door_seed(
    *,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    agent_rc: tuple[int, int],
    yaw_deg: float,
    resolution_m: float,
    seed_width_cells: int = 3,
) -> HoughRawDoorSeedResult:
    """Reconstruct TVARS range-jump proposals and their pre-filter door lines.

    This deliberately stops after the original ``door_presentation`` stage.
    Later size, topology, waypoint, RGB, and persistent-door filters are not
    raw-candidate generation and therefore must not remove cells here.
    """

    obstacle = np.asarray(obstacle_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if obstacle.ndim != 2 or unknown.shape != obstacle.shape:
        raise ValueError("obstacle_mask and unknown_mask must share one 2D shape")
    if float(resolution_m) <= 0.0:
        raise ValueError("resolution_m must be positive")
    row, col = int(agent_rc[0]), int(agent_rc[1])
    if not (0 <= row < obstacle.shape[0] and 0 <= col < obstacle.shape[1]):
        raise ValueError("agent_rc lies outside the saved map")

    points = range_jump_door_points(
        obstacle_mask=obstacle,
        unknown_mask=unknown,
        agent_rc=(row, col),
        yaw_deg=float(yaw_deg),
        resolution_m=float(resolution_m),
    )
    lines = pair_raw_door_lines(
        obstacle_mask=obstacle,
        candidate_points_xy=points,
    )
    centerline = rasterize_lines(
        lines,
        shape=obstacle.shape,
        width_cells=1,
    )
    raw_seed = rasterize_lines(
        lines,
        shape=obstacle.shape,
        width_cells=max(1, int(seed_width_cells)),
    )
    return HoughRawDoorSeedResult(
        raw_points_xy=tuple(points),
        paired_lines_xyxy=tuple(lines),
        centerline_mask=centerline,
        raw_seed_mask=raw_seed,
        debug={
            "algorithm": "tvars_original_360deg_range_jump_then_prefilter_pairing",
            "ray_loop_count": 720,
            "unique_heading_count": 360,
            "ray_max_range_m": 6.0,
            "range_jump_cells": 15,
            "range_jump_m": 15.0 * float(resolution_m),
            "raw_point_count": len(points),
            "paired_line_count": len(lines),
            "centerline_cells": int(np.count_nonzero(centerline)),
            "raw_seed_cells": int(np.count_nonzero(raw_seed)),
            "seed_width_cells": max(1, int(seed_width_cells)),
        },
    )


def range_jump_door_points(
    *,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    agent_rc: tuple[int, int],
    yaw_deg: float,
    resolution_m: float,
    max_range_m: float = 6.0,
    bot_near_range_cells: int = 30,
    gap_threshold_cells: float = 15.0,
    boundary_margin_cells: int = 22,
) -> tuple[tuple[int, int], ...]:
    """Port of the original ``convert_2_laser`` point-generation path."""

    import cv2

    obstacle = np.asarray(obstacle_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    shape = obstacle.shape
    agent_r, agent_c = int(agent_rc[0]), int(agent_rc[1])
    explored = (~unknown).astype(np.float32)
    bot_mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(
        bot_mask,
        (agent_c, agent_r),
        int(bot_near_range_cells),
        1,
        thickness=-1,
    )
    explored += bot_mask
    state = np.full(shape, 0.5, dtype=np.float32)
    state[explored != 0] = 0.0
    state[obstacle] = 1.0

    max_range_cells = float(max_range_m) / float(resolution_m)
    laser_points: list[tuple[int, int]] = []
    laser_distances: list[float] = []
    bot_out_of_map = False
    # Keep the source's 720-loop integer-angle behavior.  ``//`` means the
    # actual headings are 0,0,1,1,...,359,359 degrees.
    for index in range(360 * 2):
        angle = np.deg2rad(float(yaw_deg) + float(index * 360 // (360 * 2)))
        goal_x = round(float(agent_c) + max_range_cells * float(np.cos(angle)))
        goal_y = round(float(agent_r) + max_range_cells * float(np.sin(angle)))
        line_rows, line_cols = _rasterized_ray_cells(
            start_xy=(agent_c, agent_r),
            end_xy=(int(goal_x), int(goal_y)),
            shape=shape,
        )
        if line_rows.size == 0:
            bot_out_of_map = True
            continue
        line_distances = np.hypot(line_rows - agent_r, line_cols - agent_c)
        pairs = sorted(
            zip(
                line_distances.tolist(),
                state[line_rows, line_cols].tolist(),
            )
        )
        stop_index = len(pairs) - 1
        for pair_index, (_distance, cell_state) in enumerate(pairs):
            stop_index = pair_index
            if float(cell_state) != 0.0:
                break
        stop_distance = float(pairs[stop_index][0])
        # ``np.where`` in the source selects the first matching global cell in
        # row-major order.  ``_rasterized_ray_cells`` preserves that order.
        matching = np.flatnonzero(line_distances == stop_distance)
        match_index = int(matching[0])
        laser_points.append(
            (int(line_cols[match_index]), int(line_rows[match_index]))
        )
        laser_distances.append(stop_distance)

    raw_points: list[tuple[int, int]] = []
    if not bot_out_of_map:
        for index, current_distance in enumerate(laser_distances):
            next_index = 0 if index == len(laser_distances) - 1 else index + 1
            next_distance = laser_distances[next_index]
            if abs(float(current_distance) - float(next_distance)) <= float(gap_threshold_cells):
                continue
            raw_points.append(
                laser_points[index]
                if float(current_distance) < float(next_distance)
                else laser_points[next_index]
            )

    h, w = shape
    margin = int(boundary_margin_cells)
    clipped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in raw_points:
        point = (int(x), int(y))
        if not (margin <= point[0] < w - margin and margin <= point[1] < h - margin):
            continue
        if point in seen:
            continue
        seen.add(point)
        clipped.append(point)
    return tuple(clipped)


def _rasterized_ray_cells(
    *,
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one OpenCV ray without scanning the entire global grid."""

    import cv2

    h, w = int(shape[0]), int(shape[1])
    start_x, start_y = int(start_xy[0]), int(start_xy[1])
    end_x, end_y = int(end_xy[0]), int(end_xy[1])
    c0 = max(0, min(start_x, end_x))
    c1 = min(w - 1, max(start_x, end_x))
    r0 = max(0, min(start_y, end_y))
    r1 = min(h - 1, max(start_y, end_y))
    if c0 > c1 or r0 > r1:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.uint8)
    cv2.line(
        local,
        (start_x - c0, start_y - r0),
        (end_x - c0, end_y - r0),
        1,
        thickness=1,
    )
    local_rows, local_cols = np.where(local == 1)
    return (
        np.asarray(local_rows + r0, dtype=np.int32),
        np.asarray(local_cols + c0, dtype=np.int32),
    )


def pair_raw_door_lines(
    *,
    obstacle_mask: np.ndarray,
    candidate_points_xy: Iterable[tuple[int, int]],
    checking_size: int = 2,
    detection_range: int = 20,
    noisy_component_cells: int = 5,
    direction_threshold: float = 0.7,
    wall_direction_threshold: float = 0.4,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return original pre-filter ``door_presentation`` start/end pairs."""

    import cv2

    obstacle = np.asarray(obstacle_mask, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.erode(
        cv2.dilate(obstacle.astype(np.uint8), kernel, iterations=1),
        kernel,
        iterations=1,
    ).astype(bool)
    checking_distance = _distance_mask(int(checking_size))
    detection_distance = _distance_mask(int(detection_range))
    circle = np.zeros(detection_distance.shape, dtype=np.uint8)
    cv2.circle(
        circle,
        (int(detection_range), int(detection_range)),
        int(detection_range),
        1,
        thickness=-1,
    )
    h, w = closed.shape
    unique_lines: list[tuple[int, int, int, int]] = []
    seen_lines: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for raw_x, raw_y in candidate_points_xy:
        x, y = int(raw_x), int(raw_y)
        local = closed[
            y - checking_size : y + checking_size + 1,
            x - checking_size : x + checking_size + 1,
        ]
        if local.shape != checking_distance.shape or not np.any(local):
            continue
        if not bool(local[checking_size, checking_size]):
            weighted = np.where(local, checking_distance, np.inf)
            nearest = np.argwhere(weighted == np.min(weighted))
            nearest_r, nearest_c = (int(nearest[0, 0]), int(nearest[0, 1]))
            door_x = x + nearest_c - checking_size
            door_y = y + nearest_r - checking_size
        else:
            door_x, door_y = x, y
        if not (
            detection_range <= door_x < w - detection_range
            and detection_range <= door_y < h - detection_range
        ):
            continue
        local_detection = closed[
            door_y - detection_range : door_y + detection_range + 1,
            door_x - detection_range : door_x + detection_range + 1,
        ].copy()
        local_detection &= circle.astype(bool)
        original_detection = local_detection.copy()
        center = (int(detection_range), int(detection_range))
        start_component = _consume_component(local_detection, center)
        if len(start_component) < int(noisy_component_cells):
            continue
        wall_vectors = _wall_direction_vectors(
            start_component,
            original_detection,
            center=center,
            detection_range=int(detection_range),
            wall_direction_threshold=float(wall_direction_threshold),
        )
        while np.any(local_detection):
            weighted = np.where(local_detection, detection_distance, np.inf)
            nearest = np.argwhere(weighted == np.min(weighted))
            nearest_r, nearest_c = int(nearest[0, 0]), int(nearest[0, 1])
            component = _consume_component(
                local_detection,
                (nearest_r, nearest_c),
            )
            if len(component) < int(noisy_component_cells):
                continue
            delta = np.asarray(
                [nearest_c - detection_range, nearest_r - detection_range],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-9:
                continue
            direction = delta / norm
            if any(
                float(np.dot(direction, wall_direction))
                > float(direction_threshold)
                for wall_direction in wall_vectors
            ):
                continue
            end_x = door_x + int(delta[0])
            end_y = door_y + int(delta[1])
            start = (int(door_x), int(door_y))
            end = (int(end_x), int(end_y))
            identity = tuple(sorted((start, end)))
            if identity in seen_lines:
                continue
            seen_lines.add(identity)
            unique_lines.append((start[0], start[1], end[0], end[1]))
    return tuple(unique_lines)


def rasterize_lines(
    lines_xyxy: Iterable[tuple[int, int, int, int]],
    *,
    shape: tuple[int, int],
    width_cells: int,
) -> np.ndarray:
    import cv2

    out = np.zeros(tuple(shape), dtype=np.uint8)
    for x0, y0, x1, y1 in lines_xyxy:
        cv2.line(
            out,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            1,
            thickness=max(1, int(width_cells)),
        )
    return out.astype(bool)


def _consume_component(
    mutable_mask: np.ndarray,
    start_rc: tuple[int, int],
) -> list[tuple[int, int]]:
    start = (int(start_rc[0]), int(start_rc[1]))
    if not bool(mutable_mask[start]):
        return []
    queue: deque[tuple[int, int]] = deque([start])
    mutable_mask[start] = False
    component: list[tuple[int, int]] = []
    h, w = mutable_mask.shape
    while queue:
        row, col = queue.popleft()
        component.append((row, col))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if bool(mutable_mask[nr, nc]):
                    mutable_mask[nr, nc] = False
                    queue.append((nr, nc))
    return component


def _wall_direction_vectors(
    component: list[tuple[int, int]],
    original_mask: np.ndarray,
    *,
    center: tuple[int, int],
    detection_range: int,
    wall_direction_threshold: float,
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    high = 2 * int(detection_range)
    center_xy = np.asarray([center[1], center[0]], dtype=np.float64)
    for row, col in component:
        if row in {0, high} or col in {0, high}:
            continue
        neighbor = original_mask[row - 1 : row + 2, col - 1 : col + 2].astype(np.uint8).copy()
        neighbor[0, 0] = 1
        neighbor[0, 2] = 1
        neighbor[1, 1] = 1
        neighbor[2, 0] = 1
        neighbor[2, 2] = 1
        if int(np.abs(neighbor.astype(np.int16) - 1).sum()) == 0:
            continue
        if float(np.hypot(row - center[0], col - center[1])) <= 10.0:
            continue
        candidate = np.asarray([col, row], dtype=np.float64) - center_xy
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-9:
            continue
        candidate /= norm
        if any(
            float(np.dot(existing, candidate)) > float(wall_direction_threshold)
            for existing in vectors
        ):
            continue
        vectors.append(candidate)
    return vectors


def _distance_mask(radius: int) -> np.ndarray:
    rr, cc = np.indices((2 * int(radius) + 1, 2 * int(radius) + 1))
    return np.hypot(rr - int(radius), cc - int(radius))
