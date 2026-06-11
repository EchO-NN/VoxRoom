from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .frontier_mcl import MCLResult
from .schema import GridRC
from .topomap import TopologicalRoomMap


@dataclass(frozen=True)
class PolicyDecision:
    target_rc: GridRC | None
    reason: str
    done: bool = False


def compute_original_style_policy_target(
    topomap: TopologicalRoomMap,
    current_node_id: int | None,
    mcl_result: MCLResult,
    agent_rc: Sequence[int],
) -> PolicyDecision:
    if bool(getattr(topomap, "escape_detected", False)):
        waypoint = _nearest_edge_waypoint(topomap, agent_rc)
        return PolicyDecision(target_rc=waypoint, reason="escape", done=waypoint is None)
    if mcl_result.frontier_clusters:
        cluster = max(
            mcl_result.frontier_clusters,
            key=lambda item: (int(item.size), -float(item.distance_to_robot_cells)),
        )
        return PolicyDecision(target_rc=cluster.representative_rc, reason="current_room_frontier", done=False)
    unexplored = [
        node
        for node in getattr(topomap, "nodes", {}).values()
        if getattr(node, "status", None) == "Unexplored" and getattr(node, "entry_waypoint_rc", None) is not None
    ]
    if unexplored:
        node = min(
            unexplored,
            key=lambda item: _dist(item.entry_waypoint_rc, agent_rc),
        )
        return PolicyDecision(target_rc=node.entry_waypoint_rc, reason="nearest_unexplored", done=node.entry_waypoint_rc is None)
    _ = current_node_id
    return PolicyDecision(target_rc=None, reason="done", done=True)


def _dist(rc: GridRC | None, agent_rc: Sequence[int]) -> float:
    if rc is None:
        return math.inf
    return float(math.hypot(float(rc[0] - int(agent_rc[0])), float(rc[1] - int(agent_rc[1]))))


def _nearest_edge_waypoint(topomap: TopologicalRoomMap, agent_rc: Sequence[int]) -> GridRC | None:
    best: GridRC | None = None
    best_dist = math.inf
    for edge in topomap.edges.values():
        for waypoint in (edge.waypoint_from_rc, edge.waypoint_to_rc):
            dist = _dist(waypoint, agent_rc)
            if dist < best_dist:
                best = waypoint
                best_dist = dist
    return best
