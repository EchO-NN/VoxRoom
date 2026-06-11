from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Optional

VOXROOM_ROOMSEG_MODES = {
    "",
    "none",
    "disabled",
    "voxel_occupancy_door_wall",
    "voxel_occupancy_door_wall_vlm",
    "voxel_occupancy_door_wall_v9",
    "voxel_occupancy_door_wall_v9_vlm",
    "voxel_occupancy_door_wall_v29",
    "voxel_occupancy_door_wall_v29_vlm",
    "voxel_occupancy_door_wall_v32",
    "voxel_occupancy_door_wall_v32_vlm",
    "voxel_occupancy_door_wall_v33",
    "voxel_occupancy_door_wall_v33_vlm",
}

REQUIRED_RESULT_FIELDS = [
    "metric_valid",
    "strict_benchmark",
    "fallbacks_used",
    "detector_backend",
    "segmenter_backend",
    "llm_backend",
    "sim_backend",
    "map_source",
    "policy_name",
    "sgnav_decision_mode",
    "sgnav_decision_reason",
    "object_memory_count",
    "goal_candidate_count",
    "frontier_count",
    "selected_frontier",
    "stop_reason",
    "success",
    "spl",
    "softspl",
    "distance_to_goal",
    "path_length",
    "steps",
    "collisions",
    "stuck_events",
    "timeout",
    "perception_latency_ms",
    "mapping_latency_ms",
    "graph_latency_ms",
    "llm_latency_ms",
    "planning_latency_ms",
]


SGNAV_STEP_DUMP_KEYS = [
    "objects",
    "groups",
    "rooms",
    "edges",
    "room_context",
    "room_segmentation",
    "room_semantics",
    "subgraphs",
    "subgraph_texts_or_payloads",
    "llm_scores",
    "subgraph_probabilities",
    "frontier_scores",
    "selected_frontier",
    "candidate_goals",
    "reperception_state",
    "stop_state",
]


class BenchmarkAssetError(RuntimeError):
    """Raised when a strict benchmark run is missing a required asset."""


def complete_result_row(row: Mapping[str, object], args: object | None = None) -> dict:
    out = dict(row)
    strict_benchmark = bool(_get_arg(args, "strict_benchmark", out.get("strict_benchmark", False)))
    ablation_name = _get_arg(args, "ablation_name", out.get("ablation_name"))
    detector_backend = str(_get_arg(args, "detector", out.get("detector_backend", out.get("detector", "none"))))
    segmenter_backend = str(_get_arg(args, "segmenter", out.get("segmenter_backend", out.get("segmenter", "none"))))
    sim_backend = str(_get_arg(args, "sim_backend", out.get("sim_backend", "map")))
    llm_backend = _infer_llm_backend(out, args)
    map_source = _infer_map_source(out, sim_backend)
    policy_name = _infer_policy_name(out, args, sim_backend)

    out["strict_benchmark"] = strict_benchmark
    out["ablation_name"] = ablation_name
    out["detector_backend"] = detector_backend
    out["segmenter_backend"] = segmenter_backend
    out["llm_backend"] = llm_backend
    out["sim_backend"] = sim_backend
    out["map_source"] = map_source
    out["policy_name"] = policy_name
    out["steps"] = int(out.get("steps", out.get("num_steps", 0)) or 0)
    out["collisions"] = int(out.get("collisions", out.get("num_collisions", 0)) or 0)
    out["stuck_events"] = int(out.get("stuck_events", out.get("num_stuck_events", 0)) or 0)
    out["timeout"] = bool(
        out.get("timeout", False)
        or out.get("failure_reason") == "max_control_steps"
        or out.get("terminal_reason") == "timeout"
    )
    out["object_memory_count"] = int(out.get("object_memory_count", 0) or 0)
    out["goal_candidate_count"] = int(
        out.get("goal_candidate_count", len(out.get("object_memory_goal_candidates", []) or [])) or 0
    )
    out["frontier_count"] = int(out.get("frontier_count", out.get("frontier_clusters", 0)) or 0)
    out["selected_frontier"] = out.get("selected_frontier", out.get("frontier_center_grid"))
    out["stop_reason"] = _infer_stop_reason(out)

    for key in (
        "sgnav_decision_mode",
        "sgnav_decision_reason",
        "success",
        "spl",
        "softspl",
        "distance_to_goal",
        "path_length",
    ):
        out.setdefault(key, None)
    for key in (
        "perception_latency_ms",
        "mapping_latency_ms",
        "graph_latency_ms",
        "llm_latency_ms",
        "planning_latency_ms",
    ):
        out[key] = float(out.get(key, 0.0) or 0.0)

    if (
        bool(out.get("success", False))
        and bool(out.get("success_requires_sgnav_stop", False))
        and not bool(out.get("policy_stop_confirmed", False))
    ):
        out["success"] = False
        out["spl"] = 0.0
        out["failure_reason"] = out.get("failure_reason") or "sgnav_stop_required"
        out["stop_reason"] = "sgnav_stop_required"

    fallbacks = _stable_unique([*list(out.get("fallbacks_used", []) or []), *_infer_fallbacks(out, args)])
    out["fallbacks_used"] = fallbacks
    explicit_invalid = out.get("metric_valid") is False
    out["metric_valid"] = bool((not fallbacks) and not explicit_invalid)
    if strict_benchmark and fallbacks:
        out["metric_valid"] = False

    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in out]
    if missing:
        raise ValueError("result row missing required fields after completion: %s" % ", ".join(missing))
    return out

def validate_strict_benchmark_assets(args: object) -> None:
    if not bool(_get_arg(args, "strict_benchmark", False)):
        return
    detector = str(_get_arg(args, "detector", "") or "").strip().lower()
    segmenter = str(_get_arg(args, "segmenter", "") or "").strip().lower()
    if detector not in {"", "none", "dry_run"}:
        raise BenchmarkAssetError("VoxRoom-Online room segmentation runs do not use open-vocabulary detectors")
    if segmenter not in {"", "none"}:
        raise BenchmarkAssetError("VoxRoom-Online room segmentation runs do not use external mask segmenters")
    sgnav_mode = str(_get_arg(args, "sgnav_mode", "") or "").strip().lower()
    if sgnav_mode == "paper" and not bool(_get_arg(args, "llm_enabled", False)):
        raise BenchmarkAssetError("strict VoxRoom paper mode requires --llm-enabled true and a reachable LLM/VLM endpoint")
    room_map_mode = str(_get_arg(args, "room_map_mode", "") or "").strip().lower()
    if room_map_mode not in VOXROOM_ROOMSEG_MODES:
        raise BenchmarkAssetError("strict VoxRoom runs require voxel occupancy room segmentation, got %s" % room_map_mode)


def empty_voxroom_step_dump(metadata: Optional[Mapping[str, object]] = None) -> dict:
    payload = {
        "objects": [],
        "groups": [],
        "rooms": [],
        "room_context": {},
        "room_segmentation": {
            "source": "voxel_occupancy_door_wall_v33",
            "algorithm": "voxel_occupancy_door_wall_v33",
            "room_count": 0,
            "rooms": [],
        },
        "object_memory": {
            "raw_detection_count": 0,
            "stable_track_count": 0,
            "tentative_track_count": 0,
            "mask_association_count": 0,
            "partial_edge_count": 0,
            "contained_child_count": 0,
        },
        "room_semantics": {
            "backend": "unavailable",
            "allowed_categories": [],
            "labels": [],
        },
        "edges": [],
        "subgraphs": [],
        "subgraph_texts_or_payloads": [],
        "llm_scores": [],
        "subgraph_probabilities": [],
        "frontier_scores": [],
        "selected_frontier": None,
        "candidate_goals": [],
        "reperception_state": {},
        "stop_state": {},
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    payload["contract_version"] = "voxroom_step_dump_v1"
    return payload


def _get_arg(args: object | None, name: str, default: object = None) -> object:
    if args is None:
        return default
    return getattr(args, name, default)


def _infer_llm_backend(row: Mapping[str, object], args: object | None) -> str:
    if row.get("llm_backend"):
        return str(row["llm_backend"])
    if bool(_get_arg(args, "vllm_frontier_scoring", row.get("vllm_frontier_scoring", False))):
        return "vllm_frontier"
    if bool(_get_arg(args, "llm_enabled", row.get("hcot_llm_enabled", row.get("paper_llm_enabled", False)))):
        return "openai_compatible"
    return "deterministic_local"


def _infer_map_source(row: Mapping[str, object], sim_backend: str) -> str:
    if row.get("map_source"):
        return str(row["map_source"])
    if row.get("mapping_source"):
        return str(row["mapping_source"])
    if sim_backend == "isaac":
        return "online_rgbd_depth"
    return "static_preprocessed"


def _infer_policy_name(row: Mapping[str, object], args: object | None, sim_backend: str) -> str:
    if row.get("policy_name"):
        return str(row["policy_name"])
    policy = _get_arg(args, "policy", None)
    if policy:
        return str(policy)
    planner = str(_get_arg(args, "planner", row.get("planner", "astar")))
    sgnav_mode = str(_get_arg(args, "sgnav_mode", row.get("sgnav_mode", "legacy")))
    if sim_backend == "isaac":
        return "sgnav_%s_%s" % (sgnav_mode, planner)
    return "%s_static_map_baseline" % planner


def _infer_stop_reason(row: Mapping[str, object]) -> str:
    if row.get("stop_reason"):
        return str(row["stop_reason"])
    if row.get("failure_reason"):
        return str(row["failure_reason"])
    if bool(row.get("success", False)):
        return "success"
    if row.get("terminal_reason"):
        return str(row["terminal_reason"])
    if bool(row.get("timeout", False)):
        return "timeout"
    return "not_success"


def _infer_fallbacks(row: Mapping[str, object], args: object | None) -> list[str]:
    detector = str(row.get("detector_backend", "")).strip().lower()
    segmenter = str(row.get("segmenter_backend", "")).strip().lower()
    llm_backend = str(row.get("llm_backend", "")).strip().lower()
    sim_backend = str(row.get("sim_backend", "")).strip().lower()
    ablation_name = row.get("ablation_name")
    fallbacks: list[str] = []
    if detector not in {"", "none", "dry_run"}:
        fallbacks.append("unsupported_detector")
    if segmenter not in {"", "none"}:
        fallbacks.append("unsupported_segmenter")
    if bool(_get_arg(args, "seed_gt_object_memory", row.get("seed_gt_object_memory", False))):
        fallbacks.append("seeded_gt_object_memory")
    if int(row.get("seeded_object_memory_count", 0) or 0) > 0:
        fallbacks.append("seeded_gt_object_memory")
    if bool(_get_arg(args, "allow_gt_goal_fallback", row.get("allow_gt_goal_fallback", False))):
        fallbacks.append("gt_goal_fallback_allowed")
    if row.get("sgnav_decision_mode") == "gt_goal_fallback":
        fallbacks.append("gt_goal_fallback_used")
    if llm_backend == "deterministic_local" and not ablation_name:
        fallbacks.append("llm_deterministic_local")
    if int(row.get("hcot_llm_fallback_count", 0) or 0) > 0:
        fallbacks.append("llm_deterministic_local")
    room_vlm_backend = str(row.get("room_vlm_backend", "") or "").strip().lower()
    if room_vlm_backend in {"deterministic_debug", "unavailable"} and not ablation_name:
        fallbacks.append("room_vlm_unavailable" if room_vlm_backend == "unavailable" else "room_vlm_deterministic_debug")
    if bool(row.get("room_vlm_invalid_json", False)):
        fallbacks.append("room_vlm_invalid_json")
    room_map_mode = str(row.get("room_map_mode", _get_arg(args, "room_map_mode", "")) or "").strip().lower()
    if room_map_mode not in VOXROOM_ROOMSEG_MODES:
        fallbacks.append("unsupported_room_segmentation_mode")
    if sim_backend == "map":
        fallbacks.append("static_map_planning")
    if bool(_get_arg(args, "static_nearfield_map", row.get("static_nearfield_map", False))):
        fallbacks.append("static_nearfield_map")
    if bool(_get_arg(args, "frontier_allow_near_fallback", row.get("frontier_allow_near_fallback", False))):
        fallbacks.append("frontier_near_fallback")
    return fallbacks


def _stable_unique(values: Iterable[object]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out
