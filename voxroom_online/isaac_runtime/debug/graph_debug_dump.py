from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np


def save_graph_debug_dump(
    out_dir: str,
    step: int,
    goal: str,
    scenegraph,
    frontiers: Iterable,
    frontier_decision=None,
    nav_decision=None,
    commitment_metadata: Optional[Mapping[str, object]] = None,
) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "goal": str(goal),
        "score_backend": _score_backend(scenegraph),
        "objects": _objects(scenegraph),
        "rooms": _rooms(scenegraph),
        "room_context": dict(getattr(scenegraph, "room_context_debug", {}) or {}),
        "room_segmentation": dict(getattr(scenegraph, "room_segmentation_debug", {}) or {}),
        "room_semantics": dict(getattr(scenegraph, "room_semantics_debug", {}) or {}),
        "groups": _groups(scenegraph),
        "edges": _edges(scenegraph),
        "frontiers": _frontiers(frontiers, frontier_decision),
        "decision": _decision(nav_decision),
        "commitment": dict(commitment_metadata or {}),
        "score_debug": dict(getattr(scenegraph, "last_score_debug", {}) or {}),
    }
    out = path / ("graph_step_%06d.json" % int(step))
    out.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _score_backend(scenegraph) -> str:
    debug = getattr(scenegraph, "last_score_debug", {}) or {}
    mode = str(debug.get("mode", "fallback"))
    if mode == "vllm_frontier":
        return "vllm_frontier_direct"
    return mode


def _objects(scenegraph) -> list[dict]:
    out = []
    for node in getattr(scenegraph, "runtime_nodes", {}).values():
        if getattr(node, "kind", "") != "object":
            continue
        out.append(
            {
                "id": node.node_id,
                "cat": node.caption,
                "category": node.caption,
                "center_grid": list(node.center_grid) if node.center_grid is not None else None,
                "center_world": list(node.center_world) if node.center_world is not None else None,
                "conf": float(node.confidence),
                "mean_confidence": float(getattr(node, "mean_confidence", node.confidence)),
                "detection_count": int(getattr(node, "detection_count", node.observed_count)),
                "winner_detection_count": int(getattr(node, "winner_detection_count", node.observed_count)),
                "hits": int(node.observed_count),
                "room": node.room,
            }
        )
    return out


def _rooms(scenegraph) -> list[dict]:
    out = []
    for node in getattr(scenegraph, "runtime_rooms", {}).values():
        out.append(
            {
                "id": node.node_id,
                "caption": node.caption,
                "center_grid": list(node.center_grid) if node.center_grid is not None else None,
                "area": int(node.observed_count),
                "metadata": dict(getattr(node, "metadata", {}) or {}),
            }
        )
    return out


def _groups(scenegraph) -> list[dict]:
    out = []
    for node in getattr(scenegraph, "runtime_groups", []):
        out.append(
            {
                "id": node.node_id,
                "caption": node.caption,
                "center_grid": list(node.center_grid) if node.center_grid is not None else None,
                "room": node.room,
                "members": list(node.members),
            }
        )
    return out


def _edges(scenegraph) -> list[dict]:
    return [
        {
            "src": edge.source,
            "dst": edge.target,
            "rel": edge.relation,
            "weight": float(edge.weight),
        }
        for edge in getattr(scenegraph, "runtime_edges", [])
    ]


def _frontiers(frontiers: Iterable, frontier_decision) -> list[dict]:
    clusters = list(frontiers)
    sg_scores = list(getattr(frontier_decision, "scenegraph_scores", []) or [])
    dist_scores = list(getattr(frontier_decision, "distance_scores", []) or [])
    total_scores = list(getattr(frontier_decision, "total_scores", []) or [])
    selected_index = getattr(frontier_decision, "selected_index", None)
    out = []
    for idx, cluster in enumerate(clusters):
        out.append(
            {
                "index": int(idx),
                "selected": selected_index == idx,
                "center": list(cluster.center_grid),
                "size": int(cluster.size),
                "path_distance_m": float(cluster.path_distance_from_agent),
                "sg_score": float(sg_scores[idx]) if idx < len(sg_scores) else None,
                "dist_score": float(dist_scores[idx]) if idx < len(dist_scores) else None,
                "total": float(total_scores[idx]) if idx < len(total_scores) else None,
            }
        )
    return out


def _decision(nav_decision) -> dict:
    if nav_decision is None:
        return {}
    return {
        "mode": getattr(nav_decision, "mode", None),
        "reason": getattr(nav_decision, "reason", None),
        "target_cells": [list(cell) for cell in (getattr(nav_decision, "target_cells", []) or [])[:64]],
        "metadata": dict(getattr(nav_decision, "metadata", {}) or {}),
    }


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
