from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def clamp(value: float, limit: float) -> float:
    return max(-float(limit), min(float(limit), float(value)))


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def limit_translation_command_displacement(
    cmd: Sequence[float],
    *,
    dt: float,
    max_displacement_m: float,
) -> tuple[Tuple[float, float, float], float]:
    if len(cmd) < 3:
        raise ValueError("kinematic command must contain vx, vy, and wz")
    if float(dt) <= 0.0:
        raise ValueError("kinematic command dt must be positive")
    if float(max_displacement_m) < 0.0:
        raise ValueError("maximum translation displacement must be non-negative")
    vx, vy, wz = (float(cmd[0]), float(cmd[1]), float(cmd[2]))
    commanded_displacement_m = math.hypot(vx, vy) * float(dt)
    if commanded_displacement_m <= float(max_displacement_m) + 1e-12:
        return (vx, vy, wz), 1.0
    if commanded_displacement_m <= 1e-12:
        return (vx, vy, wz), 1.0
    scale = float(max_displacement_m) / commanded_displacement_m
    return (vx * scale, vy * scale, wz), float(scale)


class ForwardOnlyWaypointFollower:
    def __init__(
        self,
        max_vx: float = 0.15,
        max_wz: float = math.radians(30.0) / 0.2,
        lookahead_m: float = 0.15,
        translation_command_scale: float = 1.0,
        control_dt: float = 0.2,
        forward_heading_tolerance_rad: float = math.radians(75.0),
    ):
        self.max_vx = float(max_vx)
        self.max_wz = float(max_wz)
        self.lookahead_m = float(lookahead_m)
        self.translation_command_scale = float(translation_command_scale)
        self.control_dt = float(control_dt)
        self.forward_heading_tolerance_rad = float(forward_heading_tolerance_rad)
        if self.translation_command_scale <= 0.0:
            raise ValueError("translation_command_scale must be positive")
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if self.max_wz <= 0.0:
            raise ValueError("max_wz must be positive")
        if not 0.0 < self.forward_heading_tolerance_rad < math.pi / 2.0:
            raise ValueError("forward_heading_tolerance_rad must be in (0, pi/2)")
        self.kp_xy = 1.2 * self.translation_command_scale
        self.max_turn_step_rad = self.max_wz * self.control_dt

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
        desired_yaw = math.atan2(dy_w, dx_w) if abs(dx_w) + abs(dy_w) > 1e-6 else yaw
        heading_error = wrap_angle(desired_yaw - yaw)
        turn_step = math.copysign(
            min(abs(heading_error), self.max_turn_step_rad),
            heading_error,
        )
        wz = turn_step / self.control_dt

        # Keep differential-drive semantics: only rotate in place for targets
        # outside the forward arc, and never command reverse or lateral motion.
        if abs(heading_error) > self.forward_heading_tolerance_rad:
            return 0.0, 0.0, wz

        alignment_scale = max(0.0, math.cos(heading_error))
        forward_speed = (
            min(self.max_vx, math.hypot(dx_w, dy_w) * self.kp_xy)
            * alignment_scale
        )
        return max(0.0, forward_speed), 0.0, wz

    @staticmethod
    def reached(pose_world: Sequence[float], target_world: Tuple[float, float], tolerance_m: float = 0.2) -> bool:
        return math.hypot(float(pose_world[0]) - target_world[0], float(pose_world[1]) - target_world[1]) <= tolerance_m
