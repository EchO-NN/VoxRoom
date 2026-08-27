from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

from voxroom_online.isaac_runtime.graph.paper_scene_graph import ObjectEdge, ObjectNode, PaperSceneGraph


class LLMClient(Protocol):
    def complete_json(self, prompt: str) -> object:
        ...


class VLMClient(Protocol):
    def complete_json(self, prompt: str, image: Optional[np.ndarray] = None) -> object:
        ...


@dataclass
class ObjectEdgeProposal:
    src_id: str
    dst_id: str
    relation: str
    confidence: float
    reason: str
    is_short_edge: bool
    source: str = "llm_dense_connect"
    pruning_debug: Dict[str, object] = field(default_factory=dict)


def propose_object_edges_with_llm(
    new_objects: Sequence[ObjectNode],
    all_objects: Sequence[ObjectNode],
    llm_client: Optional[LLMClient] = None,
    max_retries: int = 2,
    batch_size: int = 12,
) -> List[ObjectEdgeProposal]:
    pairs = _candidate_pairs(new_objects, all_objects)
    if not pairs:
        return []
    if llm_client is None:
        return [
            ObjectEdgeProposal(src.id, dst.id, _fallback_relation(src, dst), 0.35, "deterministic fallback without llm_client", False)
            for src, dst in pairs
            if _fallback_relation(src, dst) != "none"
        ]
    proposals: List[ObjectEdgeProposal] = []
    chunk_size = max(1, int(batch_size))
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start : start + chunk_size]
        parsed = _call_edge_json(llm_client, chunk, max_retries=max_retries)
        proposals.extend(_edge_proposals_from_items(parsed, chunk))
    return proposals


def _edge_proposals_from_items(items: Sequence[object], pairs) -> List[ObjectEdgeProposal]:
    proposals: List[ObjectEdgeProposal] = []
    valid_pair_ids = {(src.id, dst.id) for src, dst in pairs}
    valid_pair_ids.update((dst.id, src.id) for src, dst in pairs)
    pair_lookup = {"pair_%03d" % idx: (src.id, dst.id) for idx, (src, dst) in enumerate(pairs)}
    for item in items:
        if not isinstance(item, dict):
            continue
        pair_id = str(item.get("pair_id") or item.get("pair") or "")
        pair_src_dst = pair_lookup.get(pair_id)
        src_id = str(item.get("src") or item.get("object1_id") or item.get("object1") or (pair_src_dst[0] if pair_src_dst else ""))
        dst_id = str(item.get("dst") or item.get("object2_id") or item.get("object2") or (pair_src_dst[1] if pair_src_dst else ""))
        relation = str(item.get("relation") or item.get("relationships") or "none").strip().lower()
        if not src_id or not dst_id or (src_id, dst_id) not in valid_pair_ids or relation in {"", "none", "no relation"}:
            continue
        proposals.append(
            ObjectEdgeProposal(
                src_id=src_id,
                dst_id=dst_id,
                relation=relation,
                confidence=float(np.clip(float(item.get("confidence", 0.5)), 0.0, 1.0)),
                reason=str(item.get("reason", "")),
                is_short_edge=False,
            )
        )
    return proposals


def _call_edge_json(llm_client: LLMClient, pairs, max_retries: int = 2) -> List[object]:
    prompt = _edge_prompt(pairs)
    last_error: Optional[Exception] = None
    attempts = max(1, int(max_retries) + 1)
    for attempt in range(attempts):
        try:
            return _ensure_json_list(llm_client.complete_json(prompt))
        except Exception as exc:
            last_error = exc
            prompt = _edge_prompt(pairs, previous_error=str(exc), repair_attempt=attempt + 1)
    raise ValueError("LLM edge proposal response failed strict JSON schema after %d attempts" % attempts) from last_error


def verify_short_edge_with_vlm(
    image: np.ndarray,
    object_a_mask: np.ndarray,
    object_b_mask: np.ndarray,
    proposal: ObjectEdgeProposal,
    vlm_client: Optional[VLMClient] = None,
) -> bool:
    if vlm_client is None:
        proposal.pruning_debug = {"kept": True, "reason": "no_vlm_client"}
        return True
    prompt = (
        "In this image, object A is %s, object B is %s.\n"
        "The proposed relation is: %s.\n"
        "Does this relation visually exist in the image?\n"
        "Return strict JSON: {\"exists\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"short explanation\"}."
        % (proposal.src_id, proposal.dst_id, proposal.relation)
    )
    _ = object_a_mask, object_b_mask
    parsed = _ensure_json_object(vlm_client.complete_json(prompt, image=image))
    kept = bool(parsed.get("exists", False)) and float(parsed.get("confidence", 0.0)) >= 0.5
    proposal.pruning_debug = {"kept": kept, "confidence": float(parsed.get("confidence", 0.0)), "reason": str(parsed.get("reason", ""))}
    return kept


def verify_long_edge_geometrically(
    proposal: ObjectEdgeProposal,
    graph: PaperSceneGraph,
    occupancy_map: np.ndarray,
    resolution_m: float = 0.05,
    origin_xy: Sequence[float] = (0.0, 0.0),
) -> bool:
    src = graph.object_nodes.get(proposal.src_id)
    dst = graph.object_nodes.get(proposal.dst_id)
    if src is None or dst is None:
        proposal.pruning_debug = {"kept": False, "reason": "missing_node"}
        return False
    same_room = bool(src.room_id and src.room_id == dst.room_id)
    unobstructed = _line_unobstructed(src.center_world, dst.center_world, occupancy_map, resolution_m, origin_xy)
    parallel_to_wall = _parallel_to_dominant_axis(src.center_world, dst.center_world, occupancy_map, resolution_m)
    kept = bool(same_room and unobstructed and parallel_to_wall)
    proposal.pruning_debug = {
        "same_room": same_room,
        "unobstructed": unobstructed,
        "parallel_to_wall": parallel_to_wall,
        "kept": kept,
    }
    return kept


def apply_edge_proposals(graph: PaperSceneGraph, proposals: Iterable[ObjectEdgeProposal]) -> None:
    existing = {tuple(sorted((edge.src_id, edge.dst_id))) for edge in graph.object_edges}
    for proposal in proposals:
        key = tuple(sorted((proposal.src_id, proposal.dst_id)))
        if key in existing:
            continue
        graph.object_edges.append(
            ObjectEdge(
                src_id=proposal.src_id,
                dst_id=proposal.dst_id,
                relation=proposal.relation,
                confidence=float(proposal.confidence),
                is_short_edge=bool(proposal.is_short_edge),
                source=proposal.source,
                pruning_debug=dict(proposal.pruning_debug),
            )
        )
        existing.add(key)


def _candidate_pairs(new_objects: Sequence[ObjectNode], all_objects: Sequence[ObjectNode]):
    new_ids = {obj.id for obj in new_objects}
    pairs = []
    seen = set()
    for src in new_objects:
        for dst in all_objects:
            if src.id == dst.id:
                continue
            if src.id not in new_ids and dst.id not in new_ids:
                continue
            key = tuple(sorted((src.id, dst.id)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((src, dst))
    return pairs


def _edge_prompt(pairs, previous_error: Optional[str] = None, repair_attempt: int = 0) -> str:
    payload = [
        {
            "pair_id": "pair_%03d" % idx,
            "src": src.id,
            "dst": dst.id,
            "src_category": src.category,
            "dst_category": dst.category,
        }
        for idx, (src, dst) in enumerate(pairs)
    ]
    repair = ""
    if previous_error:
        repair = (
            "The previous response failed JSON/schema validation: %s\n"
            "Repair the answer by returning one valid JSON object only.\n"
        ) % previous_error
    return repair + (
        "Predict the most likely relationships between these pairs of objects.\n"
        "Return exactly one JSON object with this schema:\n"
        "{\"edges\":[{\"pair_id\":\"pair_000\",\"src\":\"object:id\",\"dst\":\"object:id\",\"relation\":\"near\",\"confidence\":0.0,\"reason\":\"short\"}]}\n"
        "Rules:\n"
        "- Return no markdown, no prose, no code fence, and no trailing text.\n"
        "- Include only edges whose relation is not none; use {\"edges\":[]} when no relation exists.\n"
        "- Use only pair_id/src/dst values from the provided pairs.\n"
        "- Allowed relation values: next to, opposite to, on, under, near, functionally related.\n"
        "- confidence must be a number from 0.0 to 1.0.\n"
        "- reason must be at most 12 words.\n"
        "- Keep the response compact enough to fit in the token budget.\n"
        "Repair attempt: %d\n"
        "Pairs:\n%s" % (repair_attempt, json.dumps(payload, ensure_ascii=False))
    )


def _fallback_relation(src: ObjectNode, dst: ObjectNode) -> str:
    dist = float(np.linalg.norm(np.asarray(src.center_world[:2], dtype=np.float32) - np.asarray(dst.center_world[:2], dtype=np.float32)))
    if dist <= 1.25:
        return "near"
    if src.room_id and src.room_id == dst.room_id and dist <= 3.0:
        return "functionally related"
    return "none"


def _ensure_json_list(value: object) -> List[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("edges", "relationships", "relations", "items"):
            items = value.get(key)
            if isinstance(items, list):
                return items
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return _ensure_json_list(parsed)
    raise ValueError("LLM response must be a JSON list")


def _ensure_json_object(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("VLM response must be a JSON object")


def _line_unobstructed(
    a_world: Sequence[float],
    b_world: Sequence[float],
    occupancy_map: np.ndarray,
    resolution_m: float,
    origin_xy: Sequence[float],
) -> bool:
    occ = np.asarray(occupancy_map).astype(bool)
    if occ.size == 0:
        return True
    a = np.asarray(a_world[:2], dtype=np.float32)
    b = np.asarray(b_world[:2], dtype=np.float32)
    dist = float(np.linalg.norm(b - a))
    steps = max(2, int(math.ceil(dist / max(float(resolution_m), 1e-6))))
    origin = np.asarray(origin_xy[:2], dtype=np.float32)
    for point in np.linspace(a, b, steps):
        col = int(round((float(point[0]) - float(origin[0])) / float(resolution_m)))
        row = int(round((float(point[1]) - float(origin[1])) / float(resolution_m)))
        if 0 <= row < occ.shape[0] and 0 <= col < occ.shape[1] and occ[row, col]:
            return False
    return True


def _parallel_to_dominant_axis(
    a_world: Sequence[float],
    b_world: Sequence[float],
    occupancy_map: np.ndarray,
    resolution_m: float,
) -> bool:
    _ = resolution_m
    direction = np.asarray(b_world[:2], dtype=np.float32) - np.asarray(a_world[:2], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return False
    direction = direction / norm
    free_rows, free_cols = np.where(~np.asarray(occupancy_map).astype(bool))
    if len(free_rows) < 2:
        return True
    coords = np.stack([free_cols, free_rows], axis=1).astype(np.float32)
    coords -= np.mean(coords, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    axis = vh[0]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        return True
    axis = axis / axis_norm
    cos = abs(float(np.dot(direction, axis)))
    return cos >= math.cos(math.radians(30.0))
