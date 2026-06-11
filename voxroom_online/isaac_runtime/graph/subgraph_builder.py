from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from voxroom_online.isaac_runtime.graph.paper_scene_graph import PaperSceneGraph


@dataclass
class Subgraph:
    id: str
    central_object_id: str
    central_object_category: str
    parent_room_id: Optional[str]
    parent_group_id: Optional[str]
    directly_connected_object_ids: List[str]
    nodes: List[dict]
    edges: List[dict]
    central_world: np.ndarray


def build_object_centered_subgraphs(graph: PaperSceneGraph) -> List[Subgraph]:
    subgraphs: List[Subgraph] = []
    group_by_object = {}
    for group in graph.group_nodes.values():
        for object_id in group.object_ids:
            group_by_object.setdefault(object_id, group.id)
    for obj_id in sorted(graph.object_nodes):
        obj = graph.object_nodes[obj_id]
        neighbors = _object_neighbors(graph, obj_id)
        parent_group_id = group_by_object.get(obj_id)
        included_ids = {obj_id, *neighbors}
        if obj.room_id:
            included_ids.add(obj.room_id)
        if parent_group_id:
            included_ids.add(parent_group_id)
        nodes = []
        for node_id in sorted(included_ids):
            if node_id in graph.object_nodes:
                node = graph.object_nodes[node_id]
                nodes.append(
                    {
                        "id": node.id,
                        "type": "object",
                        "category": node.category,
                        "confidence": float(node.confidence),
                        "mean_confidence": float(getattr(node, "mean_confidence", node.confidence)),
                        "detection_count": int(getattr(node, "detection_count", node.observed_count)),
                        "winner_detection_count": int(getattr(node, "winner_detection_count", node.observed_count)),
                    }
                )
            elif node_id in graph.room_nodes:
                node = graph.room_nodes[node_id]
                nodes.append(
                    {
                        "id": node.id,
                        "type": "room",
                        "category": node.room_type,
                        "confidence": float(node.confidence),
                        "mask_id": getattr(node, "mask_id", None),
                        "mask_source": getattr(node, "mask_source", "unknown"),
                        "category_source": getattr(node, "category_source", "unknown"),
                        "area_m2": float(getattr(node, "area_m2", 0.0) or 0.0),
                        "centroid_xy": list(getattr(node, "centroid_xy", []) or []),
                        "boundary_unknown_fraction": float(getattr(node, "boundary_unknown_fraction", 0.0) or 0.0),
                        "is_partial": bool(getattr(node, "is_partial", False)),
                    }
                )
            elif node_id in graph.group_nodes:
                node = graph.group_nodes[node_id]
                nodes.append({"id": node.id, "type": "group", "category": node.category_summary, "object_ids": list(node.object_ids)})
        edges = []
        for edge in graph.object_edges:
            if edge.src_id in included_ids and edge.dst_id in included_ids:
                edges.append({"src": edge.src_id, "dst": edge.dst_id, "relation": edge.relation, "confidence": float(edge.confidence)})
        for edge in graph.affiliation_edges:
            if edge.src_id in included_ids and edge.dst_id in included_ids:
                edges.append(
                    {
                        "src": edge.src_id,
                        "dst": edge.dst_id,
                        "relation": edge.relation,
                        "confidence": float(edge.confidence),
                        "metadata": dict(getattr(edge, "metadata", {}) or {}),
                    }
                )
        subgraphs.append(
            Subgraph(
                id="sg_%s" % obj_id.replace(":", "_"),
                central_object_id=obj_id,
                central_object_category=obj.category,
                parent_room_id=obj.room_id,
                parent_group_id=parent_group_id,
                directly_connected_object_ids=neighbors,
                nodes=nodes,
                edges=edges,
                central_world=np.asarray(obj.center_world, dtype=np.float32).copy(),
            )
        )
    return subgraphs


def _object_neighbors(graph: PaperSceneGraph, object_id: str) -> List[str]:
    neighbors = set()
    for edge in graph.object_edges:
        if edge.src_id == object_id:
            neighbors.add(edge.dst_id)
        elif edge.dst_id == object_id:
            neighbors.add(edge.src_id)
    return sorted(neighbors)
