from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple


class PathLengthAccumulator:
    def __init__(self, teleport_reject_m: float = 0.5):
        self.teleport_reject_m = float(teleport_reject_m)
        self.prev_xy: Optional[Tuple[float, float]] = None
        self.total_m = 0.0

    def update(self, xy: Sequence[float]) -> float:
        cur = (float(xy[0]), float(xy[1]))
        if self.prev_xy is None:
            self.prev_xy = cur
            return self.total_m
        step = math.hypot(cur[0] - self.prev_xy[0], cur[1] - self.prev_xy[1])
        if step < self.teleport_reject_m:
            self.total_m += step
        self.prev_xy = cur
        return self.total_m

