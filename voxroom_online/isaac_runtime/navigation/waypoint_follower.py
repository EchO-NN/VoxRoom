from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def clamp(value: float, limit: float) -> float:
    return max(-float(limit), min(float(limit), float(value)))


def limit_vector(x: float, y: float, limit: float) -> Tuple[float, float]:
    norm = math.hypot(float(x), float(y))
    if norm <= float(limit) or norm <= 1e-9:
        return float(x), float(y)
    scale = float(limit) / norm
    return float(x) * scale, float(y) * scale


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class HolonomicWaypointFollower:
    def __init__(self, max_vx: float = 0.15, max_vy: float = 0.15, max_wz: float = 0.35, lookahead_m: float = 0.15):
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_wz = float(max_wz)
        self.lookahead_m = float(lookahead_m)
        self.kp_xy = 1.2
        self.kp_yaw = 1.5

    def select_lookahead(self, pose_world: Sequence[float], path_world: List[Tuple[float, float]]) -> Tuple[float, float]:
        x, y = float(pose_world[0]), float(pose_world[1])
        if not path_world:
            return x, y
        for wx, wy in path_world:
            if math.hypot(wx - x, wy - y) >= self.lookahead_m:
                return wx, wy
        return path_world[-1]

    def compute_cmd(self, pose_world: Sequence[float], path_world: List[Tuple[float, float]]) -> Tuple[float, float, float]:
        x, y, yaw = float(pose_world[0]), float(pose_world[1]), float(pose_world[3])
        tx, ty = self.select_lookahead(pose_world, path_world)
        dx_w, dy_w = tx - x, ty - y
        vx_w, vy_w = limit_vector(dx_w * self.kp_xy, dy_w * self.kp_xy, min(self.max_vx, self.max_vy))
        vx_body = math.cos(yaw) * vx_w + math.sin(yaw) * vy_w
        vy_body = -math.sin(yaw) * vx_w + math.cos(yaw) * vy_w
        desired_yaw = math.atan2(dy_w, dx_w) if abs(dx_w) + abs(dy_w) > 1e-6 else yaw
        wz = wrap_angle(desired_yaw - yaw) * self.kp_yaw
        return (
            clamp(vx_body, self.max_vx),
            clamp(vy_body, self.max_vy),
            clamp(wz, self.max_wz),
        )

    @staticmethod
    def reached(pose_world: Sequence[float], target_world: Tuple[float, float], tolerance_m: float = 0.2) -> bool:
        return math.hypot(float(pose_world[0]) - target_world[0], float(pose_world[1]) - target_world[1]) <= tolerance_m
