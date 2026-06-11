from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


GridRC = tuple[int, int]


@dataclass(frozen=True)
class DoorPointCandidate:
    rc: GridRC
    source: Literal["raycast", "vision"]
    score: float
    yaw_rad: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DoorPair:
    door_id: int
    p0_rc: GridRC
    p1_rc: GridRC
    midpoint_rc: tuple[float, float]
    width_cells: float
    normal_rc: tuple[float, float]
    waypoint_a_rc: GridRC
    waypoint_b_rc: GridRC
    score: float
    sources: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class FrontierCluster:
    cluster_id: int
    mask: np.ndarray
    size: int
    centroid_rc: tuple[float, float]
    representative_rc: GridRC
    distance_to_robot_cells: float


@dataclass(frozen=True)
class MCLResult:
    mcl_mask: np.ndarray
    frontier_mask: np.ndarray
    frontier_clusters: list[FrontierCluster]


@dataclass
class RoomNode:
    node_id: int
    status: Literal["Unexplored", "Exploring", "Explored"]
    entry_waypoint_rc: GridRC | None
    mcl_mask: np.ndarray
    first_seen_step: int
    last_seen_step: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DoorEdge:
    edge_id: int
    door_id: int
    from_node: int
    to_node: int
    waypoint_from_rc: GridRC
    waypoint_to_rc: GridRC
    midpoint_rc: tuple[float, float]
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    target_rc: GridRC | None
    reason: Literal["escape", "current_room_frontier", "nearest_unexplored", "done"]
    metadata: dict = field(default_factory=dict)

