from __future__ import annotations

import numpy as np

from .schema import DoorEdge, DoorPair, MCLResult, PolicyDecision, RoomNode


class TopologicalRoomMap:
    def __init__(self) -> None:
        self.nodes: dict[int, RoomNode] = {}
        self.edges: dict[int, DoorEdge] = {}
        self.door_pairs: dict[int, DoorPair] = {}
        self.current_node_id: int | None = None
        self.escape_detected = False
        self._next_node_id = 1
        self._next_edge_id = 1

    def update(
        self,
        *,
        step: int,
        mcl_result: MCLResult,
        door_pairs: list[DoorPair],
        agent_rc: tuple[int, int],
        obstacle_mask: np.ndarray,
    ) -> int | None:
        if self.current_node_id is None:
            node_id = self._allocate_node_id()
            self.nodes[node_id] = RoomNode(
                node_id=node_id,
                status="Exploring",
                entry_waypoint_rc=tuple(int(v) for v in agent_rc),
                mcl_mask=np.asarray(mcl_result.mcl_mask, dtype=bool).copy(),
                first_seen_step=int(step),
                last_seen_step=int(step),
                metadata={},
            )
            self.current_node_id = node_id
        current = self.nodes[int(self.current_node_id)]
        current.mcl_mask = (np.asarray(current.mcl_mask, dtype=bool) | np.asarray(mcl_result.mcl_mask, dtype=bool)) & ~np.asarray(obstacle_mask, dtype=bool)
        current.last_seen_step = int(step)
        current.status = "Exploring" if mcl_result.frontier_clusters else "Explored"
        self._register_doors(door_pairs, current_node_id=int(current.node_id), step=int(step))
        self.escape_detected = False
        return self.current_node_id

    def unexplored_nodes(self) -> list[RoomNode]:
        return sorted(
            (node for node in self.nodes.values() if node.status == "Unexplored"),
            key=lambda node: int(node.node_id),
        )

    def to_metadata(self) -> dict:
        return {
            "num_nodes": int(len(self.nodes)),
            "num_doors": int(len(self.door_pairs)),
            "num_edges": int(len(self.edges)),
            "current_node_id": -1 if self.current_node_id is None else int(self.current_node_id),
            "escape_detected": bool(self.escape_detected),
            "node_status_counts": {
                status: int(sum(1 for node in self.nodes.values() if node.status == status))
                for status in ("Unexplored", "Exploring", "Explored")
            },
        }

    def policy_decision(self, *, mcl_result: MCLResult, agent_rc: tuple[int, int]) -> PolicyDecision:
        if mcl_result.frontier_clusters:
            cluster = mcl_result.frontier_clusters[0]
            return PolicyDecision(
                target_rc=cluster.representative_rc,
                reason="current_room_frontier",
                metadata={"frontier_cluster_id": int(cluster.cluster_id), "frontier_cluster_size": int(cluster.size)},
            )
        unexplored = [node for node in self.nodes.values() if node.status == "Unexplored" and node.entry_waypoint_rc is not None]
        if unexplored:
            node = sorted(unexplored, key=lambda n: abs(int(n.entry_waypoint_rc[0]) - int(agent_rc[0])) + abs(int(n.entry_waypoint_rc[1]) - int(agent_rc[1])))[0]
            return PolicyDecision(target_rc=node.entry_waypoint_rc, reason="nearest_unexplored", metadata={"node_id": int(node.node_id)})
        return PolicyDecision(target_rc=None, reason="done", metadata={})

    def _register_doors(self, door_pairs: list[DoorPair], *, current_node_id: int, step: int) -> None:
        for pair in door_pairs:
            if int(pair.door_id) in self.door_pairs:
                continue
            self.door_pairs[int(pair.door_id)] = pair
            other_id = self._allocate_node_id()
            self.nodes[other_id] = RoomNode(
                node_id=other_id,
                status="Unexplored",
                entry_waypoint_rc=pair.waypoint_b_rc,
                mcl_mask=np.zeros_like(self.nodes[current_node_id].mcl_mask, dtype=bool),
                first_seen_step=int(step),
                last_seen_step=int(step),
                metadata={"created_from_door_id": int(pair.door_id)},
            )
            edge_id = self._next_edge_id
            self._next_edge_id += 1
            self.edges[edge_id] = DoorEdge(
                edge_id=edge_id,
                door_id=int(pair.door_id),
                from_node=int(current_node_id),
                to_node=int(other_id),
                waypoint_from_rc=pair.waypoint_a_rc,
                waypoint_to_rc=pair.waypoint_b_rc,
                midpoint_rc=pair.midpoint_rc,
                metadata={},
            )

    def _allocate_node_id(self) -> int:
        node_id = int(self._next_node_id)
        self._next_node_id += 1
        return node_id
