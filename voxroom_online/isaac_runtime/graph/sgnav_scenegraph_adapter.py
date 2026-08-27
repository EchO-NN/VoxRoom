from __future__ import annotations

import math
import os
import sys
import base64
from collections.abc import Mapping as ABCMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from urllib import request as urllib_request
from urllib.error import URLError

import numpy as np

from voxroom_online.isaac_runtime.config import load_yaml, repo_root
from voxroom_online.isaac_runtime.dataset.category_normalizer import normalize_category
from voxroom_online.isaac_runtime.graph.edge_builder import apply_edge_proposals, propose_object_edges_with_llm, verify_long_edge_geometrically
from voxroom_online.isaac_runtime.graph.frontier_interpolation import frontier_debug_payload, score_frontiers_by_subgraphs
from voxroom_online.isaac_runtime.graph.hcot_scorer import HCoTSubgraphScorer, OpenAICompatibleJSONClient
from voxroom_online.isaac_runtime.graph.paper_scene_graph import PaperSceneGraph
from voxroom_online.isaac_runtime.graph.subgraph_builder import build_object_centered_subgraphs
from voxroom_online.isaac_runtime.mapping.coordinate_transform import grid_to_world_xy
from voxroom_online.isaac_runtime.mapping.frontier import FrontierCluster
from voxroom_online.isaac_runtime.mapping.room_segmentation import RoomMask, room_mask_to_dict
from voxroom_online.isaac_runtime.perception.detection_types import Detection2D
from voxroom_online.isaac_runtime.perception.object_memory import ObjectMemory


SGNAV_ROOM_NAMES = [
    "bedroom",
    "living room",
    "bathroom",
    "kitchen",
    "dining room",
    "office room",
    "gym",
    "lounge",
    "laundry room",
]


RELATED_CATEGORY_PAIRS = {
    tuple(sorted(pair))
    for pair in [
        ("bed", "nightstand"),
        ("wardrobe", "dresser"),
        ("bookshelf", "chair"),
        ("counter", "stove"),
        ("table", "chair"),
        ("sink", "mirror"),
        ("shower", "bathtub"),
        ("refrigerator", "freezer"),
        ("oven", "microwave"),
        ("washing machine", "dryer"),
        ("sofa", "table"),
        ("desk", "chair"),
        ("computer", "tv"),
        ("tv", "sofa"),
        ("tv", "table"),
        ("vase", "table"),
        ("lamp", "table"),
    ]
}


GOAL_ROOM_PRIORS: Dict[str, Tuple[str, ...]] = {
    "bed": ("bedroom",),
    "nightstand": ("bedroom",),
    "dresser": ("bedroom",),
    "wardrobe": ("bedroom",),
    "sofa": ("living room", "lounge"),
    "tv": ("living room", "lounge", "bedroom"),
    "table": ("dining room", "living room", "kitchen", "office room"),
    "chair": ("dining room", "living room", "office room"),
    "desk": ("office room", "bedroom"),
    "bookshelf": ("office room", "living room", "bedroom"),
    "sink": ("bathroom", "kitchen"),
    "toilet": ("bathroom",),
    "bathtub": ("bathroom",),
    "shower": ("bathroom",),
    "refrigerator": ("kitchen",),
    "stove": ("kitchen",),
    "oven": ("kitchen",),
    "microwave": ("kitchen",),
    "cabinet": ("kitchen", "bathroom", "bedroom"),
    "vase": ("living room", "dining room", "bedroom"),
    "lamp": ("living room", "bedroom", "office room"),
}


@dataclass
class RuntimeGraphNode:
    node_id: str
    kind: str
    caption: str
    center_grid: Optional[Tuple[int, int]] = None
    center_world: Optional[Tuple[float, float, float]] = None
    confidence: float = 0.0
    mean_confidence: float = 0.0
    detection_count: int = 0
    winner_detection_count: int = 0
    observed_count: int = 0
    members: List[str] = field(default_factory=list)
    room: Optional[str] = None
    source_instance_id: str = ""
    parent_track_id: Optional[str] = None
    child_track_ids: List[str] = field(default_factory=list)


@dataclass
class RuntimeGraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


class _SimplePointCloud:
    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=np.float32)


class VLLMFrontierScorer:
    """OpenAI-compatible vLLM scorer for frontier ranking.

    The scorer is deliberately optional: if no vLLM endpoint is reachable, the
    adapter falls back to deterministic graph scores so Isaac runs stay usable.
    """

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.base_url = str(cfg.get("base_url") or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.model = str(cfg.get("model") or os.environ.get("VLLM_MODEL", "qwen3-vl-8b-instruct"))
        self.api_key = str(cfg.get("api_key") or os.environ.get("VLLM_API_KEY", "EMPTY"))
        self.timeout_s = float(cfg.get("timeout_s", os.environ.get("VLLM_TIMEOUT", 8.0)))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_frontiers = int(cfg.get("max_frontiers", 32))
        self.include_image = bool(cfg.get("include_image", True))
        self.image_max_width = int(cfg.get("image_max_width", 640))
        self.image_jpeg_quality = int(cfg.get("image_jpeg_quality", 75))
        self._disabled_reason: Optional[str] = None
        self._cache: Dict[str, np.ndarray] = {}
        self.last_used_image = False
        self.request_count = 0
        self.cache_hit_count = 0
        self.last_skip_reason = "not_called"
        self.last_error: Optional[str] = None
        self.last_request_frontiers = 0
        self.last_response_chars = 0

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    def score(
        self,
        goal: str,
        frontier_locations: np.ndarray,
        graph_summary: str,
        graph_version: int,
        rgb_image: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if not self.enabled:
            self.last_skip_reason = "disabled"
            return None
        if self._disabled_reason:
            self.last_skip_reason = "disabled_after_error"
            return None
        frontiers = np.asarray(frontier_locations, dtype=np.int32)
        if frontiers.size == 0:
            self.last_skip_reason = "no_frontiers"
            return np.zeros((0,), dtype=np.float32)
        key = "%s|%d|%s" % (goal, int(graph_version), frontiers[: self.max_frontiers].tolist())
        if key in self._cache:
            self.cache_hit_count += 1
            self.last_skip_reason = "cache_hit"
            return self._cache[key].copy()

        clipped = frontiers[: self.max_frontiers]
        prompt = (
            "You are scoring exploration frontiers for an indoor object navigation robot.\n"
            "Return only JSON: {\"scores\":[{\"index\":0,\"score\":0.0,\"reason\":\"short\"}, ...]}.\n"
            "Scores must be floats in [0, 1]. Higher means more promising for finding the goal.\n"
            f"Goal object: {goal}\n"
            f"Scene graph evidence:\n{graph_summary}\n"
            f"Frontiers as grid row/col with zero-based indices:\n{clipped.tolist()}\n"
        )
        data_url = self._image_to_data_url(rgb_image) if self.include_image else None
        self.last_used_image = data_url is not None
        user_content: object
        if data_url is not None:
            user_content = [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt + "\nUse the image as current visual context."},
            ]
        else:
            user_content = prompt
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict JSON-only robotics frontier scorer.",
                },
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        try:
            self.last_request_frontiers = int(len(clipped))
            self.request_count += 1
            self.last_skip_reason = "request_sent"
            print(
                "[sgnav-vllm] POST %s/chat/completions model=%s frontiers=%d image=%s"
                % (self.base_url, self.model, len(clipped), bool(data_url is not None)),
                flush=True,
            )
            raw = self._post_chat(payload)
            self.last_response_chars = int(len(raw))
            scores = self._parse_scores(raw, len(clipped))
        except Exception as exc:
            self._disabled_reason = "%s" % exc
            self.last_error = self._disabled_reason
            print("[sgnav-adapter] vLLM frontier scorer unavailable: %s" % exc, flush=True)
            return None
        if scores is None:
            self._disabled_reason = "invalid score response"
            self.last_error = self._disabled_reason
            print("[sgnav-adapter] vLLM frontier scorer returned invalid JSON scores", flush=True)
            return None
        out = np.zeros((len(frontiers),), dtype=np.float32)
        out[: len(scores)] = scores
        self._cache[key] = out.copy()
        return out

    def _post_chat(self, payload: Mapping[str, object]) -> str:
        data = __import__("json").dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % self.api_key,
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError("cannot reach %s: %s" % (self.base_url, exc)) from exc
        parsed = __import__("json").loads(body)
        return str(parsed["choices"][0]["message"].get("content", ""))

    def _image_to_data_url(self, image: Optional[np.ndarray]) -> Optional[str]:
        if image is None:
            return None
        try:
            from PIL import Image

            arr = np.asarray(image)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return None
            arr = arr[:, :, :3].astype(np.uint8, copy=False)
            pil = Image.fromarray(arr, mode="RGB")
            if self.image_max_width > 0 and pil.width > self.image_max_width:
                scale = float(self.image_max_width) / float(pil.width)
                pil = pil.resize((self.image_max_width, max(1, int(round(pil.height * scale)))))
            buffer = BytesIO()
            pil.save(buffer, format="JPEG", quality=max(1, min(95, self.image_jpeg_quality)))
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return "data:image/jpeg;base64,%s" % encoded
        except Exception:
            return None

    @staticmethod
    def _parse_scores(raw: str, n: int) -> Optional[np.ndarray]:
        try:
            parsed = __import__("json").loads(raw)
        except Exception:
            return None
        items = parsed.get("scores", parsed if isinstance(parsed, list) else [])
        if not isinstance(items, list):
            return None
        scores = np.zeros((n,), dtype=np.float32)
        seen = 0
        for pos, item in enumerate(items[:n]):
            if isinstance(item, ABCMapping):
                idx = int(item.get("index", pos))
                val = float(item.get("score", 0.0))
            else:
                idx = pos
                val = float(item)
            if 0 <= idx < n:
                scores[idx] = float(np.clip(val, 0.0, 1.0))
                seen += 1
        return scores if seen > 0 else None


class SGNavSceneGraphAdapter:
    """Lightweight room/object context adapter retained for runtime compatibility."""

    def __init__(
        self,
        sgnav_repo: str = "",
        use_original: bool = False,
        semantic_priors_path: Optional[str] = None,
        vllm_config: Optional[Mapping[str, object]] = None,
        llm_config: Optional[Mapping[str, object]] = None,
        sgnav_mode: str = "legacy",
    ):
        self.sgnav_repo = sgnav_repo
        self.use_original = bool(use_original)
        self.sgnav_mode = str(sgnav_mode or "legacy").strip().lower()
        self.scenegraph = None
        self.obj_goal_sg = ""
        self.object_memory: Optional[ObjectMemory] = None
        self.room_map = None
        self.room_masks: List[RoomMask] = []
        self.room_semantic_labels: Dict[str, object] = {}
        self.room_segmentation_debug: Dict[str, object] = {}
        self.room_semantics_debug: Dict[str, object] = {}
        priors = self._load_semantic_priors(semantic_priors_path)
        self.room_names = list(priors["room_names"])
        self.related_category_pairs = set(priors["related_category_pairs"])
        self.goal_room_priors = dict(priors["goal_room_priors"])
        self.category_aliases = dict(priors["aliases"])
        self.last_score_debug: Dict[str, object] = {}
        self.runtime_nodes: Dict[str, RuntimeGraphNode] = {}
        self.runtime_edges: List[RuntimeGraphEdge] = []
        self.runtime_groups: List[RuntimeGraphNode] = []
        self.runtime_rooms: Dict[str, RuntimeGraphNode] = {}
        self._runtime_edge_keys: set = set()
        self.frame_observations: Dict[str, object] = {}
        self.latest_rgb_image: Optional[np.ndarray] = None
        self.latest_map_info = None
        self.graph_version = 0
        paper_llm_config = dict(llm_config or {})
        allow_hcot_fallback = not bool(paper_llm_config.get("strict_benchmark", False))
        self.paper_llm_client = (
            OpenAICompatibleJSONClient(paper_llm_config)
            if self.sgnav_mode == "paper" and bool(paper_llm_config.get("enabled", False))
            else None
        )
        self.vllm_scorer = VLLMFrontierScorer(vllm_config)
        if self.sgnav_mode == "paper":
            self.vllm_scorer.enabled = False
        self.paper_graph = PaperSceneGraph(related_category_pairs=self.related_category_pairs)
        self.hcot_scorer = HCoTSubgraphScorer(
            llm_client=self.paper_llm_client,
            allow_deterministic_fallback=allow_hcot_fallback,
        )
        self.max_hcot_subgraphs_per_decision = int(paper_llm_config.get("max_hcot_subgraphs_per_decision", 8))
        if self.use_original:
            self._try_init_original()

    def _load_semantic_priors(self, path: Optional[str]) -> Dict[str, object]:
        data: Dict[str, object] = {
            "room_names": list(SGNAV_ROOM_NAMES),
            "related_category_pairs": set(RELATED_CATEGORY_PAIRS),
            "goal_room_priors": {
                normalize_category(key): tuple(normalize_category(room).replace("_", " ") for room in rooms)
                for key, rooms in GOAL_ROOM_PRIORS.items()
            },
            "aliases": {},
        }
        priors_path = Path(path) if path else repo_root() / "voxroom_online" / "isaac_runtime" / "configs" / "room_semantic_priors.yaml"
        if not priors_path.is_absolute():
            priors_path = repo_root() / priors_path
        if not priors_path.exists():
            return data
        try:
            raw = load_yaml(priors_path)
            aliases = {
                normalize_category(str(key)): normalize_category(str(value))
                for key, value in dict(raw.get("aliases", {}) or {}).items()
            }
            if raw.get("room_names"):
                data["room_names"] = [str(name) for name in raw.get("room_names", [])]
            if aliases:
                data["aliases"] = aliases
            pairs = set()
            for pair in raw.get("related_category_pairs", []) or []:
                if len(pair) < 2:
                    continue
                a = self._canonical_category_static(str(pair[0]), aliases)
                b = self._canonical_category_static(str(pair[1]), aliases)
                pairs.add(tuple(sorted((a, b))))
            if pairs:
                data["related_category_pairs"] = pairs
            goal_priors = {}
            for goal, rooms in dict(raw.get("goal_room_priors", {}) or {}).items():
                goal_priors[self._canonical_category_static(str(goal), aliases)] = tuple(
                    normalize_category(str(room)).replace("_", " ") for room in rooms
                )
            if goal_priors:
                data["goal_room_priors"] = goal_priors
        except Exception as exc:
            print("[sgnav-adapter] semantic priors unavailable, using defaults: %s" % exc, flush=True)
        return data

    @staticmethod
    def _canonical_category_static(category: str, aliases: Mapping[str, str]) -> str:
        cat = normalize_category(category)
        return str(aliases.get(cat, cat))

    def _canonical_category(self, category: str) -> str:
        return self._canonical_category_static(category, self.category_aliases)

    @contextmanager
    def _sgnav_cwd(self):
        old_cwd = os.getcwd()
        try:
            if self.sgnav_repo and os.path.isdir(self.sgnav_repo):
                os.chdir(self.sgnav_repo)
            yield
        finally:
            os.chdir(old_cwd)

    def _try_init_original(self) -> None:
        if self.sgnav_repo and self.sgnav_repo not in sys.path:
            sys.path.insert(0, self.sgnav_repo)
        try:
            with self._sgnav_cwd():
                from scenegraph import SceneGraph  # type: ignore

                camera_matrix = np.eye(3, dtype=np.float32)
                agent_stub = SimpleNamespace(
                    args=SimpleNamespace(
                        sgnav_score_mode="paper_object",
                        frontier_score_norm="weighted_mean",
                        disable_llm_edges=True,
                    )
                )
                self.scenegraph = SceneGraph(
                    map_resolution=5,
                    map_size_cm=4000,
                    map_size=800,
                    camera_matrix=camera_matrix,
                    is_navigation=True,
                    agent=agent_stub,
                )
        except Exception as exc:
            print("[sgnav-adapter] original SceneGraph unavailable, using fallback: %s" % exc, flush=True)
            self.scenegraph = None

    def reset(self, goal_category: str) -> None:
        self.obj_goal_sg = self._canonical_category(str(goal_category))
        self.runtime_nodes.clear()
        self.runtime_edges.clear()
        self.runtime_groups.clear()
        self.runtime_rooms.clear()
        self.room_masks = []
        self.room_semantic_labels = {}
        self.room_segmentation_debug = {}
        self.room_semantics_debug = {}
        self.frame_observations = {}
        self.latest_rgb_image = None
        self.latest_map_info = None
        self.graph_version = 0
        self.last_score_debug = {}
        self.paper_graph.reset()
        if self.scenegraph is not None:
            try:
                with self._sgnav_cwd():
                    self.scenegraph.reset()
                    self.scenegraph.set_obj_goal(self.obj_goal_sg, self.obj_goal_sg)
            except Exception:
                pass

    def update(self, object_memory: ObjectMemory, room_map=None) -> None:
        self.update_from_frame(object_memory=object_memory, room_map=room_map)

    def update_scenegraph(self, *args, **kwargs) -> None:
        """Compatibility wrapper for older callers."""
        self.update_from_frame(*args, **kwargs)

    def update_from_frame(
        self,
        object_memory: ObjectMemory,
        room_map=None,
        room_masks: Optional[Sequence[RoomMask]] = None,
        room_semantic_labels: Optional[Mapping[str, object]] = None,
        room_segmentation_debug: Optional[Mapping[str, object]] = None,
        room_semantics_debug: Optional[Mapping[str, object]] = None,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        detections_2d: Optional[Sequence[Detection2D]] = None,
        map_info=None,
        occupancy: Optional[np.ndarray] = None,
        free: Optional[np.ndarray] = None,
        navigable: Optional[np.ndarray] = None,
        observed: Optional[np.ndarray] = None,
        pose_world: Optional[Sequence[float]] = None,
        camera_pose_world: Optional[Sequence[float]] = None,
        step_id: int = 0,
    ) -> None:
        self.object_memory = object_memory
        if hasattr(object_memory, "dedupe"):
            object_memory.dedupe(map_info=map_info)
        self.room_map = room_map
        self.room_masks = list(room_masks or [])
        self.room_semantic_labels = dict(room_semantic_labels or {})
        self.room_segmentation_debug = dict(room_segmentation_debug or {})
        self.room_semantics_debug = dict(room_semantics_debug or {})
        self.latest_map_info = map_info
        if rgb is not None:
            self.latest_rgb_image = np.asarray(rgb, dtype=np.uint8).copy()
        self.frame_observations = {
            "has_rgb": rgb is not None,
            "has_depth": depth is not None,
            "num_detections_2d": int(len(detections_2d or [])),
            "map_shape": tuple(int(v) for v in occupancy.shape) if occupancy is not None else None,
            "observed_cells": int(np.count_nonzero(observed)) if observed is not None else 0,
            "free_cells": int(np.count_nonzero(free if free is not None else navigable)) if (free is not None or navigable is not None) else 0,
            "step_id": int(step_id),
            "pose_world": tuple(float(v) for v in pose_world) if pose_world is not None else None,
            "camera_pose_world": tuple(float(v) for v in camera_pose_world) if camera_pose_world is not None else None,
            "room_mask_count": len([room for room in self.room_masks if not room.stale]),
            "room_semantic_count": len(self.room_semantic_labels),
        }
        self._rebuild_runtime_graph(map_info=map_info)
        if self.sgnav_mode == "paper":
            if self.room_masks and map_info is not None:
                self.paper_graph.update_room_nodes_from_room_masks(self.room_masks, self.room_semantic_labels, map_info=map_info)
            elif room_map is not None and map_info is not None:
                self.paper_graph.update_room_nodes_from_room_map(room_map, map_info, self.room_names)
            self._update_paper_graph_from_memory(object_memory, occupancy=occupancy, map_info=map_info)
        if self.scenegraph is not None:
            try:
                with self._sgnav_cwd():
                    self.scenegraph.set_room_map(room_map)
                    self._sync_original_nodes(map_info=map_info)
                    if hasattr(self.scenegraph, "set_full_map") and occupancy is not None:
                        self.scenegraph.set_full_map(np.asarray(occupancy, dtype=np.float32)[None, None])
                    fbe_free_map = free if free is not None else navigable
                    if hasattr(self.scenegraph, "set_fbe_free_map") and fbe_free_map is not None:
                        self.scenegraph.set_fbe_free_map(np.asarray(fbe_free_map, dtype=np.float32)[None, None])
                    if hasattr(self.scenegraph, "set_full_pose") and pose_world is not None:
                        self.scenegraph.set_full_pose(np.asarray(pose_world, dtype=np.float32))
                    if hasattr(self.scenegraph, "update_group"):
                        self.scenegraph.update_group()
            except Exception:
                pass
        self.graph_version += 1

    def score(self, frontier_locations: np.ndarray, num_frontiers: int) -> np.ndarray:
        if num_frontiers <= 0:
            return np.zeros((0,), dtype=np.float32)
        if self.sgnav_mode == "paper":
            return self._paper_score(frontier_locations, num_frontiers)
        if self.scenegraph is not None:
            try:
                with self._sgnav_cwd():
                    scores = np.asarray(self.scenegraph.score(frontier_locations, num_frontiers), dtype=np.float32)
                    self.last_score_debug = dict(getattr(self.scenegraph, "last_score_debug", {}) or {})
                    return scores
            except Exception as exc:
                print("[sgnav-adapter] original score failed, using fallback: %s" % exc, flush=True)
        vllm_scores = self.vllm_scorer.score(
            self.obj_goal_sg,
            np.asarray(frontier_locations[:num_frontiers], dtype=np.int32),
            self.graph_summary(max_objects=24, max_edges=32),
            self.graph_version,
            rgb_image=self.latest_rgb_image,
        )
        if vllm_scores is not None:
            self.last_score_debug = {
                "mode": "vllm_frontier",
                "used_image": bool(self.vllm_scorer.last_used_image),
                "num_objects": len(self.object_memory.nodes) if self.object_memory is not None else 0,
                "num_edges": len(self.runtime_edges),
                "num_groups": len(self.runtime_groups),
                "frontier_score_min": float(np.min(vllm_scores)) if len(vllm_scores) else 0.0,
                "frontier_score_max": float(np.max(vllm_scores)) if len(vllm_scores) else 0.0,
                "frontier_score_mean": float(np.mean(vllm_scores)) if len(vllm_scores) else 0.0,
            }
            return vllm_scores
        return self._fallback_score(frontier_locations, num_frontiers)

    def _update_paper_graph_from_memory(self, object_memory: ObjectMemory, occupancy: Optional[np.ndarray] = None, map_info=None) -> None:
        self.paper_graph.update_from_object_memory(object_memory, map_info=map_info)
        new_ids = set(getattr(self.paper_graph, "new_object_ids", []) or [])
        if not new_ids:
            self.paper_graph.update_group_nodes()
            return
        all_objects = list(self.paper_graph.object_nodes.values())
        new_objects = [node for node in all_objects if node.id in new_ids]
        proposals = propose_object_edges_with_llm(new_objects, all_objects, llm_client=self.paper_llm_client)
        proposals = self._prune_paper_edge_proposals(proposals, occupancy=occupancy, map_info=map_info)
        apply_edge_proposals(self.paper_graph, proposals)
        self.paper_graph.update_group_nodes()

    def _paper_score(self, frontier_locations: np.ndarray, num_frontiers: int) -> np.ndarray:
        subgraphs_all = build_object_centered_subgraphs(self.paper_graph)
        subgraphs = self._select_relevant_subgraphs(subgraphs_all, self.obj_goal_sg)
        subgraph_scores = self.hcot_scorer.score(subgraphs, self.obj_goal_sg, graph_version=self.paper_graph.version)
        frontiers = []
        for idx, loc in enumerate(np.asarray(frontier_locations[:num_frontiers], dtype=np.int32)):
            row, col = int(loc[0]), int(loc[1])
            if self.latest_map_info is not None:
                wx, wy = grid_to_world_xy(row, col, self.latest_map_info)
            else:
                wx, wy = float(row), float(col)
            frontiers.append(FrontierCluster((row, col), (float(wx), float(wy)), [(row, col)], 1, float(idx + 1)))
        frontier_scores = score_frontiers_by_subgraphs(frontiers, subgraph_scores)
        payload = frontier_debug_payload(frontier_scores)
        payload.update(
            {
                "mode": "paper_subgraph_interpolation",
                "num_object_nodes": len(self.paper_graph.object_nodes),
                "num_room_nodes": len(self.paper_graph.room_nodes),
                "num_group_nodes": len(self.paper_graph.group_nodes),
                "num_object_edges": len(self.paper_graph.object_edges),
                "num_subgraphs": len(subgraphs),
                "num_subgraphs_total": len(subgraphs_all),
                "num_subgraphs_scored": len(subgraphs),
                "max_hcot_subgraphs_per_decision": self.max_hcot_subgraphs_per_decision,
            }
        )
        self.last_score_debug = payload
        return np.asarray([item.score for item in frontier_scores], dtype=np.float32)

    def _select_relevant_subgraphs(self, subgraphs, goal: str):
        max_k = max(0, int(getattr(self, "max_hcot_subgraphs_per_decision", 0)))
        if max_k <= 0 or len(subgraphs) <= max_k:
            return subgraphs
        goal = self._canonical_category(goal)

        def score_sg(sg) -> float:
            central = self._canonical_category(getattr(sg, "central_object_category", ""))
            value = 0.0
            if central == goal:
                value += 100.0
            for node in getattr(sg, "nodes", []):
                if not isinstance(node, ABCMapping):
                    continue
                cat = self._canonical_category(str(node.get("category", "")))
                if cat == goal:
                    value += 50.0
                if tuple(sorted((cat, goal))) in self.related_category_pairs:
                    value += 10.0
                value += float(node.get("confidence", 0.0)) * 0.1
            return value

        return sorted(subgraphs, key=score_sg, reverse=True)[:max_k]

    def _prune_paper_edge_proposals(self, proposals, occupancy: Optional[np.ndarray] = None, map_info=None):
        pruned = []
        for proposal in proposals:
            if proposal.is_short_edge:
                proposal.pruning_debug = {
                    **dict(proposal.pruning_debug),
                    "kept": True,
                    "reason": "short_edge_vlm_not_enabled_kept",
                }
                pruned.append(proposal)
                continue
            if occupancy is None or map_info is None:
                proposal.pruning_debug = {
                    **dict(proposal.pruning_debug),
                    "kept": True,
                    "reason": "no_occupancy_for_long_edge_pruning",
                }
                pruned.append(proposal)
                continue
            try:
                keep = verify_long_edge_geometrically(
                    proposal,
                    self.paper_graph,
                    np.asarray(occupancy).astype(bool),
                    resolution_m=float(map_info.resolution_m),
                    origin_xy=(float(getattr(map_info, "min_x", 0.0)), float(getattr(map_info, "min_y", 0.0))),
                )
            except Exception as exc:
                proposal.pruning_debug = {
                    **dict(proposal.pruning_debug),
                    "kept": False,
                    "reason": "long_edge_pruning_error: %s" % exc,
                }
                keep = False
            if keep:
                pruned.append(proposal)
        return pruned

    def _sync_original_nodes(self, map_info=None) -> None:
        if self.scenegraph is None or self.object_memory is None:
            return
        try:
            from scenegraph import Edge, ObjectNode  # type: ignore
        except Exception:
            return
        if hasattr(self.scenegraph, "refresh_room_nodes_from_room_map"):
            try:
                self.scenegraph.refresh_room_nodes_from_room_map()
            except Exception:
                pass
        nodes = []
        mem_to_node = {}
        for fallback_id, mem_node in enumerate(self.object_memory.nodes):
            mem_id = int(getattr(mem_node, "node_id", fallback_id))
            node = ObjectNode()
            node.set_caption(self._canonical_category(mem_node.category))
            node.set_center([int(mem_node.center_grid[0]), int(mem_node.center_grid[1])])
            node.is_new_node = False
            node.is_goal_node = self._category_matches_goal(mem_node.category)
            node.score = float(mem_node.confidence)
            object_payload = {
                "captions": [self._canonical_category(mem_node.category)],
                "conf": [float(mem_node.confidence)],
                "image_idx": list(range(int(mem_node.observed_count))),
                "mask_idx": [0] * int(mem_node.observed_count),
                "num_detections": int(mem_node.observed_count),
                "center_world": list(mem_node.center_world),
                "pcd": _SimplePointCloud(self._object_point_cloud(mem_node)),
            }
            node.set_object(object_payload)
            room_name = self.room_name_at_grid(mem_node.center_grid)
            room_node = self._original_room_node(room_name)
            if room_node is not None:
                self.scenegraph.set_node_room(node, room_node, 1.0, 1.0)
            nodes.append(node)
            mem_to_node[mem_id] = node
        self.scenegraph.nodes = nodes
        for edge in self.runtime_edges:
            if not edge.source.startswith("object:") or not edge.target.startswith("object:"):
                continue
            src_id = int(edge.source.split(":", 1)[1])
            dst_id = int(edge.target.split(":", 1)[1])
            src = mem_to_node.get(src_id)
            dst = mem_to_node.get(dst_id)
            if src is None or dst is None:
                continue
            try:
                new_edge = Edge(src, dst)
                new_edge.set_relation(edge.relation)
            except Exception:
                continue
        self.scenegraph._subgraph_score_cache_by_key.clear()
        self.scenegraph._subgraph_score_cache = []
        self.scenegraph._subgraph_score_cache_key = None

    def _original_room_node(self, room_name: Optional[str]):
        if not room_name or self.scenegraph is None:
            return None
        target = normalize_category(room_name).replace("_", " ")
        for room_node in getattr(self.scenegraph, "room_nodes", []):
            if normalize_category(getattr(room_node, "caption", "")) == target:
                return room_node
        return None

    def _fallback_score(self, frontier_locations: np.ndarray, num_frontiers: int) -> np.ndarray:
        scores = np.zeros((num_frontiers,), dtype=np.float32)
        if self.object_memory is None or len(self.object_memory.nodes) == 0:
            self.last_score_debug = {"mode": "fallback", "num_objects": 0}
            return scores
        goal_nodes = [node for node in self.object_memory.nodes if self._category_matches_goal(node.category)]
        context_nodes = goal_nodes or self.object_memory.nodes
        for idx, frontier in enumerate(frontier_locations[:num_frontiers]):
            best = 0.0
            for node in context_nodes:
                center = np.asarray(node.center_grid, dtype=np.float32)
                dist = float(np.linalg.norm(np.asarray(frontier, dtype=np.float32) - center))
                related = self._node_relevance(node)
                best = max(best, related * float(node.confidence) * math.log1p(node.observed_count) / max(dist, 1.0))
            best += self._room_prior_score(tuple(int(v) for v in frontier))
            scores[idx] = best
        self.last_score_debug = {
            "mode": "fallback",
            "num_objects": len(self.object_memory.nodes),
            "num_goal_nodes": len(goal_nodes),
            "frontier_score_min": float(np.min(scores)) if len(scores) else 0.0,
            "frontier_score_max": float(np.max(scores)) if len(scores) else 0.0,
            "frontier_score_mean": float(np.mean(scores)) if len(scores) else 0.0,
        }
        return scores

    def _category_matches_goal(self, category: str) -> bool:
        cat = self._canonical_category(category)
        goal = self._canonical_category(self.obj_goal_sg)
        return cat == goal or (goal and (goal in cat or cat in goal))

    def _node_relevance(self, node) -> float:
        if self._category_matches_goal(node.category):
            return 8.0
        cat = self._canonical_category(node.category)
        goal = self._canonical_category(self.obj_goal_sg)
        if tuple(sorted((cat, goal))) in self.related_category_pairs:
            return 2.0
        room_name = self.room_name_at_grid(node.center_grid)
        if room_name and normalize_category(room_name).replace("_", " ") in self.goal_room_priors.get(goal, ()):
            return 1.25
        return 0.35

    def room_name_at_grid(self, grid: Sequence[int]) -> Optional[str]:
        if self.room_masks:
            r, c = int(grid[0]), int(grid[1])
            for room in self.room_masks:
                if room.stale:
                    continue
                if 0 <= r < room.mask.shape[0] and 0 <= c < room.mask.shape[1] and bool(room.mask[r, c]):
                    label = self.room_semantic_labels.get(room.room_id)
                    category = getattr(label, "category", None)
                    return str(category).replace("_", " ") if category else "unknown"
        if self.room_map is None:
            return None
        arr = np.asarray(self.room_map)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[0] == 0:
            return None
        r, c = int(grid[0]), int(grid[1])
        if not (0 <= r < arr.shape[1] and 0 <= c < arr.shape[2]):
            return None
        idx = int(np.argmax(arr[:, r, c]))
        if float(arr[idx, r, c]) <= 0.0 or idx >= len(self.room_names):
            return None
        return self.room_names[idx]

    def _room_prior_score(self, frontier: Tuple[int, int]) -> float:
        goal = self._canonical_category(self.obj_goal_sg)
        priors = self.goal_room_priors.get(goal, ())
        if not priors:
            return 0.0
        room_name = self.room_name_at_grid(frontier)
        if room_name is None:
            return 0.0
        return 0.6 if normalize_category(room_name).replace("_", " ") in priors else 0.0

    def _rebuild_runtime_graph(self, map_info=None) -> None:
        self.runtime_nodes = {}
        self.runtime_edges = []
        self.runtime_groups = []
        self.runtime_rooms = {}
        self._runtime_edge_keys = set()
        if self.room_masks:
            self._rebuild_room_nodes_from_masks()
        elif self.room_map is not None:
            self._rebuild_room_nodes()
        if self.object_memory is None:
            return
        track_to_node_id: Dict[str, str] = {}
        for fallback_id, mem_node in enumerate(self.object_memory.nodes):
            room_name = self.room_name_at_grid(mem_node.center_grid)
            node_id = "object:%d" % int(getattr(mem_node, "node_id", fallback_id))
            source_instance_id = str(getattr(mem_node, "source_instance_id", "") or "")
            self.runtime_nodes[node_id] = RuntimeGraphNode(
                node_id=node_id,
                kind="object",
                caption=self._canonical_category(mem_node.category),
                center_grid=(int(mem_node.center_grid[0]), int(mem_node.center_grid[1])),
                center_world=tuple(float(v) for v in mem_node.center_world),
                confidence=float(mem_node.confidence),
                mean_confidence=float(getattr(mem_node, "mean_confidence", mem_node.confidence)),
                detection_count=int(getattr(mem_node, "valid_detection_count", mem_node.observed_count) or mem_node.observed_count),
                winner_detection_count=int(getattr(mem_node, "winner_detection_count", mem_node.observed_count)),
                observed_count=int(mem_node.observed_count),
                room=room_name,
                source_instance_id=source_instance_id,
                parent_track_id=getattr(mem_node, "parent_track_id", None),
                child_track_ids=list(getattr(mem_node, "child_track_ids", ()) or ()),
            )
            if source_instance_id:
                track_to_node_id[source_instance_id] = node_id
            if room_name:
                room_id = self._runtime_room_id_for_name(room_name)
                if room_id in self.runtime_rooms:
                    self._append_runtime_edge(node_id, room_id, "belongs to", 1.0)
        self._rebuild_object_edges_and_groups()
        self._append_parent_child_edges(track_to_node_id)
        self._append_room_doorway_edges()

    def _append_parent_child_edges(self, track_to_node_id: Mapping[str, str]) -> None:
        if not track_to_node_id:
            return
        for node_id, node in list(self.runtime_nodes.items()):
            if node.kind != "object" or not node.parent_track_id:
                continue
            parent_id = track_to_node_id.get(str(node.parent_track_id))
            if parent_id is None or parent_id == node_id:
                continue
            self._append_runtime_edge(node_id, parent_id, "supported by", 1.0)
            self._append_runtime_edge(node_id, parent_id, "inside or on", 1.0)

    def _runtime_room_id_for_name(self, room_name: str) -> str:
        if str(room_name).startswith("room_"):
            return "room:%s" % room_name
        return "room:%s" % normalize_category(room_name).replace("_", " ")

    def _rebuild_room_nodes_from_masks(self) -> None:
        for room in self.room_masks:
            if room.stale:
                continue
            label = self.room_semantic_labels.get(room.room_id)
            caption = str(getattr(label, "category", "unknown")).replace("_", " ")
            centroid_grid = None
            if isinstance(room.metadata, dict) and room.metadata.get("centroid_grid"):
                values = room.metadata["centroid_grid"]
                centroid_grid = (int(round(float(values[0]))), int(round(float(values[1]))))
            if centroid_grid is None:
                rr, cc = np.nonzero(room.mask)
                centroid_grid = (int(np.mean(rr)), int(np.mean(cc))) if rr.size else None
            node_id = "room:%s" % room.room_id
            node = RuntimeGraphNode(
                node_id=node_id,
                kind="room",
                caption=caption,
                center_grid=centroid_grid,
                center_world=(float(room.centroid_xy[0]), float(room.centroid_xy[1]), 0.0),
                confidence=float(getattr(label, "confidence", room.confidence)),
                observed_count=int(room.observed_free_cells),
            )
            setattr(
                node,
                "metadata",
                {
                    **room_mask_to_dict(room, include_mask=False),
                    "semantic_label": getattr(label, "to_dict", lambda: {})() if label is not None else {},
                },
            )
            self.runtime_rooms[node_id] = node

    def _rebuild_room_nodes(self) -> None:
        arr = np.asarray(self.room_map)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            return
        for idx in range(min(arr.shape[0], len(self.room_names))):
            mask = arr[idx] > 0.0
            if not np.any(mask):
                continue
            rr, cc = np.where(mask)
            room_name = self.room_names[idx]
            node_id = "room:%s" % normalize_category(room_name).replace("_", " ")
            self.runtime_rooms[node_id] = RuntimeGraphNode(
                node_id=node_id,
                kind="room",
                caption=room_name,
                center_grid=(int(np.mean(rr)), int(np.mean(cc))),
                observed_count=int(np.count_nonzero(mask)),
            )

    def _rebuild_object_edges_and_groups(self) -> None:
        object_items = [(node_id, node) for node_id, node in self.runtime_nodes.items() if node.kind == "object"]
        adjacency: Dict[str, set] = {node_id: set() for node_id, _node in object_items}
        for idx, (src_id, src) in enumerate(object_items):
            for dst_id, dst in object_items[idx + 1 :]:
                if src.center_world is None or dst.center_world is None:
                    continue
                dist = float(np.linalg.norm(np.asarray(src.center_world[:2]) - np.asarray(dst.center_world[:2])))
                related_pair = tuple(sorted((self._canonical_category(src.caption), self._canonical_category(dst.caption))))
                is_related = related_pair in self.related_category_pairs
                same_room = bool(src.room and src.room == dst.room)
                if is_related and dist <= 3.0:
                    relation = "related near"
                    weight = max(0.1, 1.0 / max(dist, 0.25))
                elif same_room and dist <= 2.0:
                    relation = "near"
                    weight = max(0.1, 0.5 / max(dist, 0.25))
                elif dist <= 1.25:
                    relation = "near"
                    weight = max(0.1, 0.35 / max(dist, 0.25))
                else:
                    continue
                self._append_runtime_edge(src_id, dst_id, relation, float(weight))
                adjacency[src_id].add(dst_id)
                adjacency[dst_id].add(src_id)

        visited = set()
        for node_id, node in object_items:
            if node_id in visited or not adjacency.get(node_id):
                continue
            stack = [node_id]
            visited.add(node_id)
            members = []
            while stack:
                cur = stack.pop()
                members.append(cur)
                for nxt in adjacency.get(cur, set()):
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            if len(members) < 2:
                continue
            centers = [self.runtime_nodes[mid].center_grid for mid in members if self.runtime_nodes[mid].center_grid is not None]
            rooms = [self.runtime_nodes[mid].room for mid in members if self.runtime_nodes[mid].room]
            room = rooms[0] if rooms and all(r == rooms[0] for r in rooms) else None
            captions = [self.runtime_nodes[mid].caption for mid in members]
            member_key = "-".join(sorted(mid.split(":", 1)[1] for mid in members))
            group_id = "group:%s" % member_key
            group_node = RuntimeGraphNode(
                node_id=group_id,
                kind="group",
                caption=", ".join(sorted(set(captions))),
                center_grid=(int(np.mean([c[0] for c in centers])), int(np.mean([c[1] for c in centers]))) if centers else None,
                members=list(members),
                room=room,
            )
            self.runtime_groups.append(group_node)
            for member_id in members:
                self._append_runtime_edge(group_id, member_id, "contains", 1.0)
            if room:
                room_id = self._runtime_room_id_for_name(room)
                if room_id in self.runtime_rooms:
                    self._append_runtime_edge(group_id, room_id, "belongs to", 1.0)

    def _append_room_doorway_edges(self) -> None:
        for room in self.room_masks:
            if room.stale:
                continue
            for edge in room.doorway_edges:
                room_a = edge.get("room_a") or edge.get("room_a_id")
                room_b = edge.get("room_b") or edge.get("room_b_id")
                if room_a is None or room_b is None:
                    continue
                src = "room:%s" % str(room_a).replace("room:", "")
                dst = "room:%s" % str(room_b).replace("room:", "")
                if src == dst or src not in self.runtime_rooms or dst not in self.runtime_rooms:
                    continue
                self._append_runtime_edge(src, dst, "adjacent via doorway", float(edge.get("confidence", 1.0)))

    def _append_runtime_edge(self, source: str, target: str, relation: str, weight: float = 1.0) -> None:
        relation_key = normalize_category(relation).replace("_", " ")
        if relation_key in {"near", "related near"}:
            edge_ends = tuple(sorted((str(source), str(target))))
        else:
            edge_ends = (str(source), str(target))
        key = (edge_ends, relation_key)
        if key in self._runtime_edge_keys:
            return
        self._runtime_edge_keys.add(key)
        self.runtime_edges.append(RuntimeGraphEdge(str(source), str(target), relation_key, float(weight)))

    def _object_point_cloud(self, mem_node) -> np.ndarray:
        x, y, z = [float(v) for v in mem_node.center_world]
        radius = 0.08 + min(0.20, 0.02 * max(1, int(mem_node.observed_count)))
        offsets = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [radius, 0.0, 0.0],
                [-radius, 0.0, 0.0],
                [0.0, radius, 0.0],
                [0.0, -radius, 0.0],
                [radius, radius, 0.0],
                [-radius, -radius, 0.0],
                [0.0, 0.0, radius],
            ],
            dtype=np.float32,
        )
        return offsets + np.asarray([x, y, z], dtype=np.float32)

    def graph_summary(self, max_objects: int = 16, max_edges: int = 24) -> str:
        objects = [node for node in self.runtime_nodes.values() if node.kind == "object"]
        objects.sort(key=lambda n: (self._category_matches_goal(n.caption), n.observed_count, n.confidence), reverse=True)
        object_text = []
        for node in objects[:max_objects]:
            loc = "rc=%s" % (list(node.center_grid),) if node.center_grid else "rc=?"
            room = " room=%s" % node.room if node.room else ""
            object_text.append(
                "%s mean_conf=%.2f detections=%d %s%s"
                % (
                    node.caption,
                    float(node.mean_confidence or node.confidence),
                    int(node.detection_count or node.observed_count),
                    loc,
                    room,
                )
            )
        edge_text = []
        for edge in self.runtime_edges[:max_edges]:
            src = self.runtime_nodes.get(edge.source) or self.runtime_rooms.get(edge.source)
            dst = self.runtime_nodes.get(edge.target) or self.runtime_rooms.get(edge.target)
            if edge.source.startswith("group:"):
                src = next((g for g in self.runtime_groups if g.node_id == edge.source), None)
            if edge.target.startswith("group:"):
                dst = next((g for g in self.runtime_groups if g.node_id == edge.target), None)
            if src is None or dst is None:
                continue
            edge_text.append("(%s, %s, %s)" % (src.caption, edge.relation, dst.caption))
        group_text = ["[%s] room=%s" % (g.caption, g.room or "?") for g in self.runtime_groups[:8]]
        return (
            "objects: " + ("; ".join(object_text) if object_text else "none") + "\n"
            "groups: " + ("; ".join(group_text) if group_text else "none") + "\n"
            "edges: " + ("; ".join(edge_text) if edge_text else "none") + "\n"
            "frame: " + str(self.frame_observations)
        )
