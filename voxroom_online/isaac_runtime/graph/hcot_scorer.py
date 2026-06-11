from __future__ import annotations

import json
import re
from urllib import error as urllib_error
from urllib import request as urllib_request
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from voxroom_online.isaac_runtime.graph.edge_builder import LLMClient
from voxroom_online.isaac_runtime.graph.subgraph_builder import Subgraph


@dataclass
class SubgraphScore:
    subgraph_id: str
    central_object_id: str
    goal_category: str
    estimated_distance_m: float
    p_sub: float
    summary_reason: str
    raw_llm_response: dict
    central_world: np.ndarray


class HCoTSubgraphScorer:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        min_distance_m: float = 0.25,
        max_retries: int = 2,
        allow_deterministic_fallback: bool = True,
    ):
        self.llm_client = llm_client
        self.min_distance_m = max(1e-3, float(min_distance_m))
        self.max_retries = max(0, int(max_retries))
        self.allow_deterministic_fallback = bool(allow_deterministic_fallback)
        self._cache: Dict[str, SubgraphScore] = {}
        self.llm_request_count = 0
        self.llm_failure_count = 0
        self.llm_fallback_count = 0
        self.last_error = None

    def score(self, subgraphs: List[Subgraph], goal_category: str, graph_version: int = 0) -> List[SubgraphScore]:
        scores = [self.score_subgraph(subgraph, goal_category, graph_version=graph_version) for subgraph in subgraphs]
        if not scores:
            return []
        raw = np.asarray([score.p_sub for score in scores], dtype=np.float32)
        hi = float(np.max(raw))
        if hi > 0.0:
            for score in scores:
                score.p_sub = float(score.p_sub / hi)
        return scores

    def score_subgraph(self, subgraph: Subgraph, goal_category: str, graph_version: int = 0) -> SubgraphScore:
        key = "%s|%s|%s" % (graph_version, goal_category, subgraph.id)
        if key in self._cache:
            return self._cache[key]
        if self.llm_client is None:
            if not self.allow_deterministic_fallback:
                raise RuntimeError("strict graph reasoning requires an enabled OpenAI-compatible LLM backend")
            result = self._fallback_score(subgraph, goal_category)
        else:
            self.llm_request_count += 1
            try:
                result = score_subgraph_with_hcot(subgraph, goal_category, self.llm_client, self.min_distance_m, self.max_retries)
            except Exception as exc:
                self.llm_failure_count += 1
                self.last_error = str(exc)
                if not self.allow_deterministic_fallback:
                    raise RuntimeError("strict graph reasoning LLM failed: %s" % exc) from exc
                self.llm_fallback_count += 1
                result = self._fallback_score(subgraph, goal_category)
                result.raw_llm_response = {
                    **dict(result.raw_llm_response or {}),
                    "fallback": True,
                    "llm_error": str(exc),
                }
        self._cache[key] = result
        return result

    def _fallback_score(self, subgraph: Subgraph, goal_category: str) -> SubgraphScore:
        same_category = subgraph.central_object_category == goal_category
        related = any(goal_category in str(node.get("category", "")) for node in subgraph.nodes)
        distance = self.min_distance_m if same_category else (1.0 if related else 4.0)
        return SubgraphScore(
            subgraph_id=subgraph.id,
            central_object_id=subgraph.central_object_id,
            goal_category=goal_category,
            estimated_distance_m=float(distance),
            p_sub=float(1.0 / max(distance, self.min_distance_m)),
            summary_reason="deterministic fallback subgraph distance estimate",
            raw_llm_response={
                "paper_hcot": False,
                "central_object_category": subgraph.central_object_category,
                "goal_category": goal_category,
                "stage1_prior_distance_m": float(distance),
                "stage2_questions": [],
                "stage3_answers": [],
                "stage4_final_distance_m": float(distance),
                "summary_reason": "deterministic fallback subgraph distance estimate",
                "p_sub_formula": "1 / max(stage4_final_distance_m, min_distance_m)",
                "fallback": True,
            },
            central_world=np.asarray(subgraph.central_world, dtype=np.float32).copy(),
        )


class OpenAICompatibleJSONClient:
    def __init__(self, config: Optional[Mapping[str, object]] = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.base_url = str(cfg.get("base_url") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = str(cfg.get("model") or "qwen3-vl-8b-instruct")
        self.api_key = str(cfg.get("api_key") or "EMPTY")
        self.timeout_s = float(cfg.get("timeout_s", 8.0))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_tokens = int(cfg.get("max_tokens", 512))
        self.response_format_json = bool(cfg.get("response_format_json", True))
        self.request_count = 0

    def complete_json(self, prompt: str) -> object:
        if not self.enabled:
            raise RuntimeError("OpenAI-compatible JSON client is disabled")
        max_tokens = max(64, int(self.max_tokens))
        last_error: Optional[Exception] = None
        for _ in range(4):
            try:
                content = self._complete_json_once(prompt, max_tokens=max_tokens)
                return _parse_json_value(str(content))
            except RuntimeError as exc:
                last_error = exc
                if max_tokens <= 128 or not _is_context_length_error(str(exc)):
                    raise
                max_tokens = max(128, max_tokens // 2)
        raise RuntimeError("OpenAI-compatible JSON request failed after reducing max_tokens") from last_error

    def _complete_json_once(self, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON API. Return exactly one valid JSON object. "
                        "Do not include markdown, comments, prose, code fences, or trailing text. "
                        "Every string must be closed and the JSON must parse with json.loads."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": int(max_tokens),
        }
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % self.api_key,
            },
            method="POST",
        )
        self.request_count += 1
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "OpenAI-compatible JSON request failed: HTTP %s %s: %s"
                % (exc.code, exc.reason, detail[:2000])
            ) from exc
        content = body["choices"][0]["message"].get("content", "")
        return str(content)


def score_subgraph_with_hcot(
    subgraph: Subgraph,
    goal_category: str,
    llm_client: LLMClient,
    min_distance_m: float = 0.25,
    max_retries: int = 2,
) -> SubgraphScore:
    """Score one object-centered subgraph with an optional LLM reasoning chain.

    The four calls intentionally follow the paper's sequence: prior object-goal
    distance, questions, subgraph-grounded answers, and final distance summary.
    The LLM never scores frontiers directly on this path; frontier interpolation
    consumes the inverse-distance ``P_sub`` returned here.
    """
    prior = _call_json(
        llm_client,
        build_hcot_prior_distance_prompt(subgraph, goal_category),
        max_retries,
    )
    prior_distance = _safe_float(prior.get("prior_distance_m", prior.get("distance_m", 4.0)), 4.0)
    questions = _call_json(
        llm_client,
        build_hcot_question_prompt(subgraph, goal_category, prior_distance),
        max_retries,
    )
    question_list = _json_string_list(questions.get("questions", []))
    answers = _call_json(
        llm_client,
        build_hcot_answer_prompt(subgraph, goal_category, question_list),
        max_retries,
    )
    answer_list = _answer_string_list(answers.get("answers", []))
    final = _call_json(
        llm_client,
        build_hcot_final_distance_prompt(subgraph, goal_category, prior_distance, question_list, answer_list),
        max_retries,
    )
    estimated_distance = _safe_float(
        final.get("estimated_distance_m", final.get("final_distance_m", final.get("distance", prior_distance))),
        prior_distance,
    )
    p_sub = 1.0 / max(float(estimated_distance), float(min_distance_m))
    summary_reason = str(final.get("summary_reason", final.get("reason", "")))
    return SubgraphScore(
        subgraph_id=subgraph.id,
        central_object_id=subgraph.central_object_id,
        goal_category=goal_category,
        estimated_distance_m=float(estimated_distance),
        p_sub=float(p_sub),
        summary_reason=summary_reason,
        raw_llm_response={
            "paper_hcot": True,
            "central_object_id": subgraph.central_object_id,
            "central_object_category": subgraph.central_object_category,
            "goal_category": goal_category,
            "stage1_prior_distance_m": float(prior_distance),
            "stage1_raw": prior,
            "stage2_questions": question_list,
            "stage2_raw": questions,
            "stage3_answers": answer_list,
            "stage3_raw": answers,
            "stage4_final_distance_m": float(estimated_distance),
            "stage4_raw": final,
            "summary_reason": summary_reason,
            "p_sub_formula": "1 / max(stage4_final_distance_m, min_distance_m)",
            "fallback": False,
        },
        central_world=np.asarray(subgraph.central_world, dtype=np.float32).copy(),
    )


def build_hcot_prior_distance_prompt(subgraph: Subgraph, goal_category: str) -> str:
    context = _subgraph_context_payload(subgraph, goal_category)
    return (
        "Graph reasoning stage 1/4: prior object-goal distance.\n"
        "Predict the most likely distance between the central object and the goal object in an indoor environment.\n"
        "Use only common indoor spatial priors for this prior stage; do not choose or score frontiers.\n"
        "Return exactly one strict JSON object with schema:\n"
        "{\"prior_distance_m\": 0.0, \"reason\": \"short explanation\"}\n"
        "Context:\n%s" % json.dumps(context, ensure_ascii=False, sort_keys=True)
    )


def build_hcot_question_prompt(subgraph: Subgraph, goal_category: str, prior_distance_m: float) -> str:
    context = _subgraph_context_payload(subgraph, goal_category)
    context["stage1_prior_distance_m"] = float(prior_distance_m)
    return (
        "Graph reasoning stage 2/4: ask distance-prediction questions.\n"
        "Ask useful questions about the central object and goal object for predicting their distance.\n"
        "Questions should refer to the room, group, direct object neighbors, and edges when present.\n"
        "Return exactly one strict JSON object with schema:\n"
        "{\"questions\": [\"question 1\", \"question 2\", \"question 3\"]}\n"
        "Context:\n%s" % json.dumps(context, ensure_ascii=False, sort_keys=True)
    )


def build_hcot_answer_prompt(subgraph: Subgraph, goal_category: str, questions: Sequence[str]) -> str:
    context = _subgraph_context_payload(subgraph, goal_category)
    context["questions"] = list(questions)
    return (
        "Graph reasoning stage 3/4: answer questions from the object-centered subgraph.\n"
        "Answer the questions using only the provided subgraph nodes and edges. Do not invent unseen objects.\n"
        "Return exactly one strict JSON object with schema:\n"
        "{\"answers\": [{\"question\": \"...\", \"answer\": \"...\"}]}\n"
        "Context:\n%s" % json.dumps(context, ensure_ascii=False, sort_keys=True)
    )


def build_hcot_final_distance_prompt(
    subgraph: Subgraph,
    goal_category: str,
    prior_distance_m: float,
    questions: Sequence[str],
    answers: Sequence[str],
) -> str:
    context = _subgraph_context_payload(subgraph, goal_category)
    context["stage1_prior_distance_m"] = float(prior_distance_m)
    context["stage2_questions"] = list(questions)
    context["stage3_answers"] = list(answers)
    return (
        "Graph reasoning stage 4/4: summarize and output final subgraph-goal distance.\n"
        "Determine the most likely distance between this object-centered subgraph and the goal object.\n"
        "Return exactly one strict JSON object with schema:\n"
        "{\"estimated_distance_m\": 0.0, \"confidence\": 0.0, \"summary_reason\": \"short explanation\"}\n"
        "The downstream probability is P_sub = 1 / max(estimated_distance_m, min_distance_m); do not score frontiers.\n"
        "Context:\n%s" % json.dumps(context, ensure_ascii=False, sort_keys=True)
    )


def _subgraph_context_payload(subgraph: Subgraph, goal_category: str) -> dict:
    nodes = [dict(node) for node in (subgraph.nodes or []) if isinstance(node, Mapping)]
    edges = [dict(edge) for edge in (subgraph.edges or []) if isinstance(edge, Mapping)]

    def find_node(node_id: Optional[str]) -> Optional[dict]:
        if not node_id:
            return None
        for node in nodes:
            if str(node.get("id")) == str(node_id):
                return dict(node)
        return None

    central = find_node(subgraph.central_object_id) or {
        "id": subgraph.central_object_id,
        "type": "object",
        "category": subgraph.central_object_category,
    }
    parent_room = find_node(subgraph.parent_room_id)
    if parent_room is not None:
        parent_room = {
            "id": parent_room.get("id"),
            "type": "room",
            "mask_id": parent_room.get("mask_id", parent_room.get("id")),
            "label": parent_room.get("category", parent_room.get("label", "unknown")),
            "label_confidence": float(parent_room.get("confidence", parent_room.get("label_confidence", 0.0)) or 0.0),
            "is_unknown": str(parent_room.get("category", parent_room.get("label", "unknown"))).strip().lower()
            in {"unknown", "unknown_room"},
            "source": parent_room.get("source", parent_room.get("mask_source", "unknown")),
            "is_partial": bool(parent_room.get("is_partial", False)),
        }
    parent_group = find_node(subgraph.parent_group_id)
    direct_objects = []
    direct_ids = {str(item) for item in (subgraph.directly_connected_object_ids or [])}
    for node in nodes:
        if str(node.get("id")) in direct_ids:
            direct_objects.append(dict(node))
    return {
        "paper_hcot": True,
        "central_object": central,
        "central_object_id": subgraph.central_object_id,
        "central_object_category": subgraph.central_object_category,
        "goal_category": goal_category,
        "parent_room": parent_room
        or {
            "id": None,
            "mask_id": None,
            "label": "unknown",
            "label_confidence": 0.0,
            "is_unknown": True,
            "source": "unassigned",
        },
        "parent_group": parent_group,
        "direct_objects": direct_objects,
        "nodes": nodes,
        "edges": edges,
        "central_world": [float(v) for v in np.asarray(subgraph.central_world, dtype=np.float32).reshape(-1)[:3]],
    }


def _call_json(llm_client: LLMClient, prompt: str, max_retries: int) -> dict:
    last_error: Optional[Exception] = None
    for _ in range(max(1, int(max_retries) + 1)):
        try:
            return _parse_json_object(llm_client.complete_json(prompt))
        except Exception as exc:
            last_error = exc
            prompt = "Repair the previous answer and return strict JSON only.\n" + prompt
    raise ValueError("failed to parse strict JSON response") from last_error


def _parse_json_object(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = _parse_json_value(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("response must be a JSON object")


def _parse_json_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = _strip_json_fence(value.strip())
        decoder = json.JSONDecoder()
        parsed, end = decoder.raw_decode(text)
        trailing = text[end:].strip()
        if trailing:
            raise json.JSONDecodeError("Extra data after JSON value", text, end)
        return parsed
    raise ValueError("response must be JSON")


def _safe_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _json_string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, Mapping):
            text = item.get("question", item.get("text", item.get("answer", "")))
        else:
            text = item
        text = str(text).strip()
        if text:
            out.append(text)
    return out


def _answer_string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, Mapping):
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", item.get("text", ""))).strip()
            out.append(("%s %s" % (q, a)).strip() if q else a)
        else:
            out.append(str(item).strip())
    return [item for item in out if item]


def _strip_json_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match is not None:
        return match.group(1).strip()
    return text


def _is_context_length_error(text: str) -> bool:
    lowered = text.lower()
    return "maximum context length" in lowered or "context length" in lowered or "input tokens" in lowered
