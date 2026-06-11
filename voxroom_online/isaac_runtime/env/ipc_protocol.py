from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class IsaacCommand:
    command: str
    payload: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IsaacObservation:
    rgb: Any
    depth: Any
    pose_world: Any
    camera_pose_world: Any
    sim_time: float
    collided: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

