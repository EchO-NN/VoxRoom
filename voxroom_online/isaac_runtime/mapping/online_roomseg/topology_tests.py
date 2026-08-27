from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from .separator_candidates import SeparatorAnchorConfig, SeparatorCandidate, extend_candidate_to_anchors, separator_mask_for_candidate
from .utils import component_metrics, conn, dilate, label_components, relabel_compact


@dataclass
class CompoundSeparatorConfig:
    enabled: bool = False
    max_group_size: int = 2
    max_candidates_per_parent_component: int = 16
    accept_only_best_per_parent_component: bool = True
    source_reject_reasons: tuple[str, ...] = (
        "reject_no_topology_gain",
        "reject_no_two_sides",
        "reject_small_known_side_low_unknown",
        "reject_split_main_corridor_axis",
    )
    allowed_kinds: tuple[str, ...] = (
        "line_extension_corridor_separator",
        "step2_corridor_separator",
        "line_extension_door_neck",
        "doorway_virtual_cut",
        "physical_wall_completion",
    )
    min_candidate_confidence: float = 0.45
    min_anchor_score: float = 1.0
    require_same_parent_component: bool = True
    require_both_candidates_have_two_anchors: bool = True
    max_angle_diff_deg: float = 18.0
    min_pair_spacing_m: float = 0.15
    max_pair_spacing_m: float = 6.00
    min_pair_combined_length_m: float = 0.80
    require_combined_global_gain: bool = True
    required_global_gain: int = 1
    max_new_components_delta: int = 1
    min_major_side_area_m2: float = 1.50
    min_major_side_width_m: float = 0.60
    max_major_area_ratio: float = 12.0
    max_tiny_fragment_delta: int = 0
    reject_if_intersects_existing_separator: bool = True
    reject_if_crosses_structural_wall: bool = True
    penalty_per_cut: float = 0.35
    debug_max_rejected_pairs_to_store: int = 128
    debug_max_excluded_candidates_to_store: int = 128

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "CompoundSeparatorConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        values: dict[str, object] = {}
        for key, value in raw.items():
            if key not in fields:
                continue
            if key in {"source_reject_reasons", "allowed_kinds"}:
                values[key] = tuple(str(v) for v in (value or ()))
            else:
                values[key] = value
        return cls(**values)


@dataclass
class TopologyTestConfig:
    enabled: bool = True
    min_split_area_m2: float = 1.0
    per_kind_min_split_area_m2: dict[str, float] | None = None
    min_new_component_width_m: float = 0.15
    min_new_component_area_cells: int = 4
    allow_partial_room_split: bool = True
    max_tiny_fragment_count: int = 3
    tiny_fragment_count_mode: str = "local_delta"
    local_fragment_radius_cells: int = 12
    centered_extension_allow_small_gaps: bool = True
    centered_extension_max_bridge_gap_cells: int = 2
    centered_extension_bridge_unknown: bool = True
    centered_extension_bridge_nonfree_if_between_free: bool = False
    require_cut_adjacent_to_wall_or_barrier: bool = True
    reject_corridor_split: bool = True
    open_room_width_min_m: float = 1.80
    open_room_min_side_area_m2: float = 2.00
    narrow_neck_width_max_m: float = 1.60
    reject_open_living_room_internal_split: bool = True
    reject_if_side_width_cells_leq: int = 0
    reject_small_known_side_for_line_extensions: bool = True
    small_known_side_area_m2: float = 2.00
    small_known_side_unknown_ratio_max: float = 0.20
    small_known_side_boundary_dilation_cells: int = 1
    corridor_min_split_area_m2: float = 0.05
    corridor_min_new_component_width_m: float = 0.10
    corridor_reject_tiny_side_width_cells_leq: int = 2
    corridor_tiny_side_min_area_m2: float = 0.03
    corridor_tiny_side_min_length_m: float = 0.35
    corridor_accept_long_narrow_side: bool = True
    corridor_local_topology_radius_cells: int = 20
    separator: SeparatorAnchorConfig | None = None
    compound_separator: CompoundSeparatorConfig = field(default_factory=CompoundSeparatorConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "TopologyTestConfig":
        raw = dict(data or {})
        separator_raw = dict(raw.pop("separator", {}) or {})
        compound_raw = dict(raw.pop("compound_separator", {}) or {})
        fields = {name for name in cls.__dataclass_fields__}
        values = {key: raw[key] for key in raw if key in fields}
        values["separator"] = SeparatorAnchorConfig.from_mapping(separator_raw or raw.get("separator_config"))
        values["compound_separator"] = CompoundSeparatorConfig.from_mapping(compound_raw)
        return cls(**values)

    def min_area_for_kind(self, kind: str) -> float:
        mapping = dict(self.per_kind_min_split_area_m2 or {})
        return float(mapping.get(str(kind), self.min_split_area_m2))

    def thickness_for_kind(self, kind: str) -> int:
        sep = self.separator or SeparatorAnchorConfig()
        if str(kind) == "doorway_virtual_cut":
            return int(sep.doorway_thickness_cells)
        if str(kind) in {"physical_wall_completion", "missed_scan_gap_closure", "short_unknown_gap_closure", "single_sided_wall_extension"}:
            return int(sep.wall_completion_thickness_cells)
        if str(kind) == "corridor_room_neck_cut":
            return int(sep.corridor_neck_thickness_cells)
        if str(kind) == "line_extension_door_neck":
            return int(sep.doorway_thickness_cells)
        if str(kind) in {"line_extension_corridor_separator", "step2_corridor_separator"}:
            return int(sep.doorway_thickness_cells)
        return int(sep.thickness_cells)


@dataclass
class TopologyFragmentStats:
    before_component_count: int
    after_component_count: int
    before_tiny_count: int
    after_tiny_count: int
    new_tiny_count: int
    local_before_tiny_count: int
    local_after_tiny_count: int
    local_new_tiny_count: int
    touched_labels: list[int]
    component_info: list[dict]

    def to_dict(self) -> dict:
        return {
            "before_component_count": int(self.before_component_count),
            "after_component_count": int(self.after_component_count),
            "before_tiny_count": int(self.before_tiny_count),
            "after_tiny_count": int(self.after_tiny_count),
            "new_tiny_count": int(self.new_tiny_count),
            "local_before_tiny_count": int(self.local_before_tiny_count),
            "local_after_tiny_count": int(self.local_after_tiny_count),
            "local_new_tiny_count": int(self.local_new_tiny_count),
            "touched_labels": [int(v) for v in self.touched_labels],
            "component_info": list(self.component_info),
        }


def greedily_select_separators(
    candidates: Sequence[SeparatorCandidate],
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    wall_candidate_clean: np.ndarray | None = None,
    corridor_skeleton: np.ndarray | None = None,
    resolution_m: float,
    config: TopologyTestConfig | Mapping[str, object] | None = None,
) -> tuple[list[SeparatorCandidate], list[SeparatorCandidate], np.ndarray, np.ndarray, dict]:
    cfg = config if isinstance(config, TopologyTestConfig) else TopologyTestConfig.from_mapping(config)
    free = np.asarray(free_clean, dtype=bool)
    accepted_map = np.zeros_like(free, dtype=bool)
    accepted: list[SeparatorCandidate] = []
    rejected: list[SeparatorCandidate] = []
    rejected_maps: dict[int, np.ndarray] = {}
    ordered = sorted(list(candidates), key=_candidate_sort_key)
    for candidate in ordered:
        ok, reason, candidate_map, metrics = evaluate_candidate(
            candidate,
            free_clean=free,
            unknown_clean=unknown_clean,
            wall_candidate_clean=wall_candidate_clean,
            corridor_skeleton=corridor_skeleton,
            current_separator_map=accepted_map,
            resolution_m=float(resolution_m),
            config=cfg,
        )
        candidate.debug.update(metrics)
        if ok:
            candidate.accepted = True
            candidate.reject_reason = ""
            accepted_map |= candidate_map
            accepted.append(candidate)
        else:
            candidate.accepted = False
            candidate.reject_reason = reason
            rejected_maps[int(candidate.candidate_id)] = np.asarray(candidate_map, dtype=bool)
            rejected.append(candidate)
    compound_debug: dict[str, object] = {"compound_separator_enabled": bool(cfg.compound_separator.enabled)}
    if bool(cfg.compound_separator.enabled):
        compound_accepted, compound_map, compound_debug = select_compound_separator_pairs(
            rejected,
            rejected_maps=rejected_maps,
            free_clean=free,
            unknown_clean=unknown_clean,
            wall_candidate_clean=wall_candidate_clean,
            current_separator_map=accepted_map,
            resolution_m=float(resolution_m),
            config=cfg,
        )
        if compound_accepted:
            compound_ids = {int(item.candidate_id) for item in compound_accepted}
            for candidate in compound_accepted:
                candidate.accepted = True
                candidate.reject_reason = ""
            accepted.extend(compound_accepted)
            rejected = [item for item in rejected if int(item.candidate_id) not in compound_ids]
            accepted_map |= compound_map
    final_labels, _ = label_components(free & ~accepted_map, 4)
    final_labels = relabel_compact(final_labels)
    debug = {
        "topology_test_enabled": bool(cfg.enabled),
        "tiny_fragment_count_mode": str(cfg.tiny_fragment_count_mode),
        "local_fragment_radius_cells": int(cfg.local_fragment_radius_cells),
        "accepted_separator_count": int(len(accepted)),
        "rejected_separator_count": int(len(rejected)),
        "accepted_separator_kinds": _kind_counts(accepted),
        "rejected_separator_reasons": _reason_counts(rejected),
        "topology_fragment_stats_per_candidate": [
            item.debug.get("topology_fragment_stats", {}) for item in [*accepted, *rejected]
        ],
        "candidate_cut_not_adjacent_to_barrier_count": int(
            sum(1 for item in [*accepted, *rejected] if not bool(item.debug.get("candidate_cut_touch_wall_a", True)) or not bool(item.debug.get("candidate_cut_touch_wall_b", True)))
        ),
        "corridor_to_corridor_rejected_count": int(sum(1 for item in rejected if item.reject_reason in {"reject_split_main_corridor_axis", "reject_corridor_to_corridor_split"})),
        "corridor_to_room_accepted_count": int(sum(1 for item in accepted if (item.debug.get("corridor_side_classification") or {}).get("one_corridor_one_room"))),
        **compound_debug,
    }
    return accepted, rejected, accepted_map.astype(bool), final_labels.astype(np.int32), debug


def select_compound_separator_pairs(
    candidates: Sequence[SeparatorCandidate],
    *,
    rejected_maps: Mapping[int, np.ndarray],
    free_clean: np.ndarray,
    unknown_clean: np.ndarray | None,
    wall_candidate_clean: np.ndarray | None,
    current_separator_map: np.ndarray,
    resolution_m: float,
    config: TopologyTestConfig,
) -> tuple[list[SeparatorCandidate], np.ndarray, dict]:
    cfg = config.compound_separator
    free = np.asarray(free_clean, dtype=bool)
    current = np.asarray(current_separator_map, dtype=bool)
    wall = np.zeros_like(free, dtype=bool) if wall_candidate_clean is None else np.asarray(wall_candidate_clean, dtype=bool)
    before = free & ~current
    parent_labels, parent_count = label_components(before, 4)
    out_map = np.zeros_like(free, dtype=bool)
    debug_rejected: list[dict[str, object]] = []
    max_debug = max(0, int(cfg.debug_max_rejected_pairs_to_store))
    max_excluded = max(0, int(cfg.debug_max_excluded_candidates_to_store))
    pre_filter_reason_counts: Counter[str] = Counter()
    excluded_filter_counts: Counter[str] = Counter()
    excluded_by_reject_reason_counts: Counter[str] = Counter()
    excluded_by_kind_counts: Counter[str] = Counter()
    excluded_by_confidence_count = 0
    excluded_by_anchor_count = 0
    excluded_by_empty_map_count = 0
    excluded_by_parent_count = 0
    excluded_candidates_sample: list[dict[str, object]] = []
    eligible_by_parent: dict[int, list[tuple[SeparatorCandidate, np.ndarray]]] = {}
    allowed_reasons = {str(v) for v in cfg.source_reject_reasons}
    allowed_kinds = {str(v) for v in cfg.allowed_kinds}

    def record_excluded(candidate: SeparatorCandidate, gate: str, extra: Mapping[str, object] | None = None) -> None:
        nonlocal excluded_by_confidence_count, excluded_by_anchor_count, excluded_by_empty_map_count, excluded_by_parent_count
        excluded_filter_counts[str(gate)] += 1
        if gate == "reject_reason_not_pairable":
            excluded_by_reject_reason_counts[str(candidate.reject_reason or "")] += 1
        elif gate == "kind_not_allowed":
            excluded_by_kind_counts[str(candidate.kind)] += 1
        elif gate == "confidence_too_low":
            excluded_by_confidence_count += 1
        elif gate == "anchor_score_too_low":
            excluded_by_anchor_count += 1
        elif gate == "empty_candidate_map":
            excluded_by_empty_map_count += 1
        elif gate == "not_single_parent_component":
            excluded_by_parent_count += 1
        if len(excluded_candidates_sample) < max_excluded:
            item = {
                "candidate_id": int(candidate.candidate_id),
                "kind": str(candidate.kind),
                "reject_reason": str(candidate.reject_reason or ""),
                "confidence": float(candidate.confidence),
                "anchor_score": float(candidate.debug.get("anchor_score", 0.0)),
                "excluded_gate": str(gate),
                "filter_reason": str(gate),
                "touch_wall_a": bool(candidate.debug.get("candidate_cut_touch_wall_a", False)),
                "touch_wall_b": bool(candidate.debug.get("candidate_cut_touch_wall_b", False)),
                "theta": float(candidate.theta),
                "length_m": float(candidate.length_m),
            }
            if extra:
                item.update(dict(extra))
            excluded_candidates_sample.append(item)

    for candidate in candidates:
        reason = str(candidate.reject_reason or "")
        pre_filter_reason_counts[reason] += 1
        if reason not in allowed_reasons:
            record_excluded(candidate, "reject_reason_not_pairable")
            continue
        if str(candidate.kind) not in allowed_kinds:
            record_excluded(candidate, "kind_not_allowed")
            continue
        if float(candidate.confidence) < float(cfg.min_candidate_confidence):
            record_excluded(candidate, "confidence_too_low")
            continue
        if float(candidate.debug.get("anchor_score", 0.0)) < float(cfg.min_anchor_score):
            record_excluded(candidate, "anchor_score_too_low")
            continue
        if bool(cfg.require_both_candidates_have_two_anchors) and not (
            bool(candidate.debug.get("candidate_cut_touch_wall_a", False))
            and bool(candidate.debug.get("candidate_cut_touch_wall_b", False))
        ):
            record_excluded(
                candidate,
                "missing_two_anchors",
                {
                    "candidate_cut_touch_wall_a": bool(candidate.debug.get("candidate_cut_touch_wall_a", False)),
                    "candidate_cut_touch_wall_b": bool(candidate.debug.get("candidate_cut_touch_wall_b", False)),
                },
            )
            continue
        if int(candidate.candidate_id) not in rejected_maps:
            record_excluded(candidate, "missing_rejected_map")
            continue
        cand_map = np.asarray(rejected_maps.get(int(candidate.candidate_id), np.zeros_like(free, dtype=bool)), dtype=bool)
        cand_map = cand_map & before
        if not np.any(cand_map):
            record_excluded(candidate, "empty_candidate_map")
            continue
        parent_values = {int(v) for v in np.unique(parent_labels[cand_map]) if int(v) > 0}
        if len(parent_values) != 1:
            record_excluded(candidate, "not_single_parent_component", {"parent_values": sorted(parent_values)})
            continue
        parent = int(next(iter(parent_values)))
        eligible_by_parent.setdefault(parent, []).append((candidate, cand_map))

    accepted: list[SeparatorCandidate] = []
    accepted_pairs: list[dict[str, object]] = []
    evaluated_pairs = 0
    reject_reason_counts: dict[str, int] = {}
    ok_but_not_selected_count = 0
    for parent, items in sorted(eligible_by_parent.items(), key=lambda kv: int(kv[0])):
        items.sort(key=lambda item: _candidate_sort_key(item[0]))
        items = items[: max(2, int(cfg.max_candidates_per_parent_component))]
        best: tuple[float, SeparatorCandidate, SeparatorCandidate, np.ndarray, dict] | None = None
        ok_pair_count_for_parent = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if int(cfg.max_group_size) != 2:
                    continue
                cand_a, map_a = items[i]
                cand_b, map_b = items[j]
                evaluated_pairs += 1
                ok, reason, pair_map, pair_debug = evaluate_compound_separator_pair(
                    cand_a,
                    cand_b,
                    map_a=map_a,
                    map_b=map_b,
                    free_clean=free,
                    unknown_clean=unknown_clean,
                    wall_candidate_clean=wall,
                    current_separator_map=current,
                    parent_labels=parent_labels,
                    parent_label=int(parent),
                    resolution_m=float(resolution_m),
                    config=config,
                )
                pair_debug = {"reason": str(reason), **pair_debug}
                if not ok:
                    reject_reason_counts[str(reason)] = int(reject_reason_counts.get(str(reason), 0)) + 1
                    if len(debug_rejected) < max_debug:
                        debug_rejected.append(pair_debug)
                    continue
                ok_pair_count_for_parent += 1
                score = _compound_pair_score(cand_a, cand_b, pair_debug, cfg)
                if best is None or score > best[0]:
                    best = (float(score), cand_a, cand_b, pair_map, pair_debug)
        if best is None:
            continue
        if ok_pair_count_for_parent > 1:
            ok_but_not_selected_count += int(ok_pair_count_for_parent - 1)
        _score, cand_a, cand_b, pair_map, pair_debug = best
        out_map |= pair_map
        for candidate in (cand_a, cand_b):
            candidate.debug.update(
                {
                    "compound_separator_accepted": True,
                    "compound_separator_pair_id": [int(cand_a.candidate_id), int(cand_b.candidate_id)],
                    **{f"compound_{key}": value for key, value in pair_debug.items() if key not in {"candidate_a_id", "candidate_b_id"}},
                }
            )
        accepted.extend([cand_a, cand_b])
        accepted_pairs.append(pair_debug)
        if not bool(cfg.accept_only_best_per_parent_component):
            # Current implementation still limits to the best non-overlapping pair per parent for safety.
            pass

    rejected_pair_count = max(0, int(evaluated_pairs) - int(len(accepted_pairs)))
    if ok_but_not_selected_count:
        reject_reason_counts["not_selected_best_pair"] = int(ok_but_not_selected_count)
    parent_candidate_counts = {str(int(parent)): int(len(items)) for parent, items in eligible_by_parent.items()}
    debug = {
        "compound_separator_enabled": bool(cfg.enabled),
        "compound_separator_pool_count": int(sum(len(items) for items in eligible_by_parent.values())),
        "compound_separator_parent_component_count": int(parent_count),
        "compound_separator_eligible_parent_count": int(len(eligible_by_parent)),
        "compound_separator_pre_filter_reason_counts": dict(pre_filter_reason_counts),
        "compound_separator_allowed_source_reject_reasons": sorted(str(v) for v in cfg.source_reject_reasons),
        "compound_separator_allowed_kinds": sorted(str(v) for v in cfg.allowed_kinds),
        "compound_separator_excluded_filter_counts": dict(excluded_filter_counts),
        "compound_separator_excluded_reject_reason_counts": dict(excluded_by_reject_reason_counts),
        "compound_separator_excluded_kind_counts": dict(excluded_by_kind_counts),
        "compound_separator_excluded_by_reject_reason_counts": dict(excluded_by_reject_reason_counts),
        "compound_separator_excluded_by_kind_counts": dict(excluded_by_kind_counts),
        "compound_separator_excluded_by_confidence_count": int(excluded_by_confidence_count),
        "compound_separator_excluded_by_anchor_count": int(excluded_by_anchor_count),
        "compound_separator_excluded_by_empty_map_count": int(excluded_by_empty_map_count),
        "compound_separator_excluded_by_parent_count": int(excluded_by_parent_count),
        "compound_separator_excluded_candidates_sample": excluded_candidates_sample,
        "compound_separator_parent_candidate_counts": parent_candidate_counts,
        "compound_separator_pair_evaluated_count": int(evaluated_pairs),
        "compound_separator_pair_accepted_count": int(len(accepted_pairs)),
        "compound_separator_pair_rejected_count": int(rejected_pair_count),
        "compound_separator_reject_reason_counts": dict(reject_reason_counts),
        "compound_separator_evaluated_pair_count": int(evaluated_pairs),
        "compound_separator_accepted_pair_count": int(len(accepted_pairs)),
        "compound_separator_rejected_pair_count": int(rejected_pair_count),
        "compound_separator_accepted_candidate_count": int(len(accepted)),
        "compound_separator_accepted_pairs": accepted_pairs,
        "compound_separator_rejected_pairs_sample": debug_rejected,
    }
    return accepted, out_map.astype(bool), debug


def evaluate_compound_separator_pair(
    cand_a: SeparatorCandidate,
    cand_b: SeparatorCandidate,
    *,
    map_a: np.ndarray,
    map_b: np.ndarray,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray | None,
    wall_candidate_clean: np.ndarray,
    current_separator_map: np.ndarray,
    parent_labels: np.ndarray,
    parent_label: int,
    resolution_m: float,
    config: TopologyTestConfig,
) -> tuple[bool, str, np.ndarray, dict]:
    cfg = config.compound_separator
    free = np.asarray(free_clean, dtype=bool)
    current = np.asarray(current_separator_map, dtype=bool)
    before = free & ~current
    pair_map = (np.asarray(map_a, dtype=bool) | np.asarray(map_b, dtype=bool)) & before
    debug: dict[str, object] = {
        "candidate_a_id": int(cand_a.candidate_id),
        "candidate_b_id": int(cand_b.candidate_id),
        "parent_label": int(parent_label),
        "candidate_a_kind": str(cand_a.kind),
        "candidate_b_kind": str(cand_b.kind),
        "candidate_a_reason": str(cand_a.reject_reason),
        "candidate_b_reason": str(cand_b.reject_reason),
        "candidate_a_anchor_score": float(cand_a.debug.get("anchor_score", 0.0)),
        "candidate_b_anchor_score": float(cand_b.debug.get("anchor_score", 0.0)),
        "compound_pair_map_cell_count": int(np.count_nonzero(pair_map)),
        "compound_pair_contains_small_known_side_candidate": bool(
            str(cand_a.reject_reason) == "reject_small_known_side_low_unknown"
            or str(cand_b.reject_reason) == "reject_small_known_side_low_unknown"
        ),
        "compound_pair_contains_small_known_side_low_unknown": bool(
            str(cand_a.reject_reason) == "reject_small_known_side_low_unknown"
            or str(cand_b.reject_reason) == "reject_small_known_side_low_unknown"
        ),
        "compound_pair_contains_main_corridor_axis_candidate": bool(
            str(cand_a.reject_reason) == "reject_split_main_corridor_axis"
            or str(cand_b.reject_reason) == "reject_split_main_corridor_axis"
        ),
        "compound_pair_contains_split_main_corridor_axis": bool(
            str(cand_a.reject_reason) == "reject_split_main_corridor_axis"
            or str(cand_b.reject_reason) == "reject_split_main_corridor_axis"
        ),
        "compound_pair_reason_specific_safety_gate": "combined_gain_one_no_tiny_fragments_major_sides",
    }
    if not np.any(pair_map):
        return False, "reject_compound_empty_pair_map", pair_map, debug
    if bool(cfg.require_same_parent_component):
        parent_values = {int(v) for v in np.unique(parent_labels[pair_map]) if int(v) > 0}
        if parent_values != {int(parent_label)}:
            debug["compound_parent_values"] = sorted(parent_values)
            return False, "reject_compound_not_same_parent_component", pair_map, debug
    if bool(cfg.reject_if_intersects_existing_separator) and np.any(pair_map & current):
        return False, "reject_compound_intersects_existing_separator", pair_map, debug
    if bool(cfg.reject_if_crosses_structural_wall) and np.any(pair_map & np.asarray(wall_candidate_clean, dtype=bool)):
        return False, "reject_compound_crosses_structural_wall", pair_map, debug
    angle_diff = _angle_diff_deg(float(cand_a.theta), float(cand_b.theta))
    spacing_m = float(np.linalg.norm(_candidate_center(cand_a) - _candidate_center(cand_b)) * float(resolution_m))
    combined_length_m = float(cand_a.length_m) + float(cand_b.length_m)
    debug.update(
        {
            "compound_angle_diff_deg": float(angle_diff),
            "compound_pair_spacing_m": float(spacing_m),
            "compound_combined_length_m": float(combined_length_m),
        }
    )
    if angle_diff > float(cfg.max_angle_diff_deg):
        return False, "reject_compound_angle_diff", pair_map, debug
    if spacing_m < float(cfg.min_pair_spacing_m) or spacing_m > float(cfg.max_pair_spacing_m):
        return False, "reject_compound_spacing", pair_map, debug
    if combined_length_m < float(cfg.min_pair_combined_length_m):
        return False, "reject_compound_combined_length", pair_map, debug

    before_labels, before_count = label_components(before, 4)
    after = before & ~pair_map
    after_labels, after_count = label_components(after, 4)
    gain = int(after_count) - int(before_count)
    debug.update({"compound_before_components": int(before_count), "compound_after_components": int(after_count), "compound_global_gain": int(gain)})
    if bool(cfg.require_combined_global_gain) and gain <= 0:
        return False, "reject_compound_no_global_gain", pair_map, debug
    if gain != int(cfg.required_global_gain):
        return False, "reject_compound_wrong_global_gain", pair_map, debug
    if gain > int(cfg.max_new_components_delta):
        return False, "reject_compound_too_many_new_components", pair_map, debug

    adjacent = dilate(pair_map, 1) & after
    touched = sorted({int(v) for v in np.unique(after_labels[adjacent]) if int(v) > 0})
    debug["compound_touched_labels"] = [int(v) for v in touched]
    if len(touched) < 2:
        return False, "reject_compound_no_two_sides", pair_map, debug
    distance_cells = ndimage.distance_transform_edt(before)
    component_info = _component_info_for_labels(after_labels, touched, resolution_m=float(resolution_m), distance_cells=distance_cells)
    component_info.sort(key=lambda item: int(item.get("area_cells", 0)), reverse=True)
    main = component_info[:2]
    debug["compound_components"] = main
    if len(main) < 2:
        return False, "reject_compound_missing_major_sides", pair_map, debug
    min_area_m2 = min(float(item.get("area_m2", 0.0)) for item in main)
    min_width_m = min(float(item.get("thickness_m", 0.0)) for item in main)
    area_ratio = max(float(item.get("area_m2", 0.0)) for item in main) / max(min_area_m2, 1.0e-9)
    debug.update(
        {
            "compound_min_major_side_area_m2": float(min_area_m2),
            "compound_min_major_side_width_m": float(min_width_m),
            "compound_major_area_ratio": float(area_ratio),
        }
    )
    if min_area_m2 < float(cfg.min_major_side_area_m2):
        return False, "reject_compound_tiny_major_side_area", pair_map, debug
    if min_width_m < float(cfg.min_major_side_width_m):
        return False, "reject_compound_tiny_major_side_width", pair_map, debug
    if area_ratio > float(cfg.max_major_area_ratio):
        return False, "reject_compound_unbalanced_major_sides", pair_map, debug
    min_cells = max(int(config.min_new_component_area_cells), int(round(float(cfg.min_major_side_area_m2) / max(float(resolution_m) ** 2, 1.0e-9))))
    fragment_stats = _fragment_stats(
        before_labels=before_labels,
        after_labels=after_labels,
        before_count=int(before_count),
        after_count=int(after_count),
        candidate_map=pair_map,
        min_cells=int(min_cells),
        local_radius_cells=int(config.local_fragment_radius_cells),
        touched_labels=touched,
        component_info=component_info[:8],
    )
    tiny_delta = _tiny_fragment_reject_count(fragment_stats, str(config.tiny_fragment_count_mode))
    debug["compound_tiny_fragment_delta"] = int(tiny_delta)
    debug["compound_fragment_stats"] = fragment_stats.to_dict()
    if tiny_delta > int(cfg.max_tiny_fragment_delta):
        return False, "reject_compound_tiny_fragment_delta", pair_map, debug
    side_class = classify_cut_sides_local(
        pair_map,
        free_clean=before,
        after_labels=after_labels,
        touched_labels=touched,
        component_info=main,
        distance_transform=distance_cells,
        corridor_skeleton=np.zeros_like(free, dtype=bool),
        resolution_m=float(resolution_m),
    )
    debug["compound_side_classification"] = side_class
    return True, "", pair_map, debug


def evaluate_candidate(
    candidate: SeparatorCandidate,
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray | None = None,
    wall_candidate_clean: np.ndarray | None = None,
    corridor_skeleton: np.ndarray | None = None,
    current_separator_map: np.ndarray,
    resolution_m: float,
    config: TopologyTestConfig,
) -> tuple[bool, str, np.ndarray, dict]:
    free = np.asarray(free_clean, dtype=bool)
    current = np.asarray(current_separator_map, dtype=bool)
    wall = np.zeros_like(free, dtype=bool) if wall_candidate_clean is None else np.asarray(wall_candidate_clean, dtype=bool)
    unknown = np.zeros_like(free, dtype=bool) if unknown_clean is None else np.asarray(unknown_clean, dtype=bool)
    skeleton = np.zeros_like(free, dtype=bool) if corridor_skeleton is None else np.asarray(corridor_skeleton, dtype=bool)
    sep_cfg = config.separator or SeparatorAnchorConfig()
    candidate, anchor_debug = extend_candidate_to_anchors(
        candidate,
        free_clean=free,
        wall_candidate_clean=wall,
        unknown_clean=unknown,
        existing_separator_map=current,
        resolution_m=float(resolution_m),
        max_anchor_extension_m=float(sep_cfg.max_anchor_extension_m),
    )
    if bool(sep_cfg.require_two_anchors) and float(anchor_debug.get("anchor_score", 0.0)) < 1.0:
        reason = "reject_unanchored_separator" if float(anchor_debug.get("anchor_score", 0.0)) <= 0.0 else "reject_one_sided_unanchored"
        return False, reason, np.zeros_like(free, dtype=bool), anchor_debug
    thickness = config.thickness_for_kind(str(candidate.kind))
    candidate_map, cut_debug = _candidate_separator_map_for_topology(
        candidate,
        free_clean=free,
        unknown_clean=unknown,
        wall_candidate_clean=wall,
        current_separator_map=current,
        resolution_m=float(resolution_m),
        thickness_cells=thickness,
        config=config,
    )
    anchor_debug = {**anchor_debug, **cut_debug}
    if int(np.count_nonzero(candidate_map)) <= 0:
        return False, "reject_candidate_not_on_free_space", candidate_map, {**anchor_debug, "candidate_mask_cell_count": 0}
    if not bool(config.enabled):
        candidate.topology_gain_score = 1.0
        return True, "", candidate_map, {**anchor_debug, "topology_test_skipped": True}
    if str(candidate.kind) in {"line_extension_corridor_separator", "step2_corridor_separator"}:
        return evaluate_corridor_separator_candidate_v16(
            candidate,
            free_clean=free,
            unknown_clean=unknown,
            current_separator_map=current,
            candidate_map=candidate_map,
            corridor_skeleton=skeleton,
            resolution_m=float(resolution_m),
            config=config,
            anchor_debug=anchor_debug,
        )
    before = free & ~current
    after = before & ~candidate_map
    before_labels, before_count = label_components(before, 4)
    after_labels, after_count = label_components(after, 4)
    if int(after_count) <= int(before_count):
        return False, "reject_no_topology_gain", candidate_map, {**anchor_debug, "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)), "before_components": int(before_count), "after_components": int(after_count)}
    adjacent = dilate(candidate_map, 1) & after
    touched = sorted({int(v) for v in np.unique(after_labels[adjacent]) if int(v) > 0})
    if len(touched) < 2:
        return False, "reject_no_two_sides", candidate_map, {**anchor_debug, "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)), "touched_components": touched}
    component_info = []
    is_corridor_extension = str(candidate.kind) in {"line_extension_corridor_separator", "step2_corridor_separator"}
    min_area_m2 = float(config.corridor_min_split_area_m2) if is_corridor_extension else float(config.min_area_for_kind(str(candidate.kind)))
    min_new_width_m = float(config.corridor_min_new_component_width_m) if is_corridor_extension else float(config.min_new_component_width_m)
    reject_width_cells = int(config.corridor_reject_tiny_side_width_cells_leq) if is_corridor_extension else int(config.reject_if_side_width_cells_leq)
    min_cells = max(int(config.min_new_component_area_cells), int(round(min_area_m2 / max(float(resolution_m) ** 2, 1e-9))))
    distance_cells = ndimage.distance_transform_edt(before)
    for label in touched:
        mask = after_labels == int(label)
        info = component_metrics(mask, float(resolution_m), distance_cells)
        info["label"] = int(label)
        component_info.append(info)
    component_info.sort(key=lambda item: int(item["area_cells"]), reverse=True)
    main = component_info[:2]
    min_width = min(float(item.get("thickness_m", 0.0)) for item in main) if len(main) >= 2 else 0.0
    min_side_width_cells = int(round(float(min_width) / max(float(resolution_m), 1e-9))) if len(main) >= 2 else 0
    if str(candidate.kind) in {"line_extension_door_neck", "line_extension_corridor_separator", "step2_corridor_separator"}:
        small_known_side_reject, small_known_side_debug = _small_known_side_reject_for_line_extension(
            component_info=main,
            labels=after_labels,
            unknown_clean=unknown,
            resolution_m=float(resolution_m),
            config=config,
        )
        if small_known_side_reject:
            return False, "reject_small_known_side_low_unknown", candidate_map, {
                **anchor_debug,
                "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)),
                "components": component_info[:8],
                "topology_touched_labels": touched,
                **small_known_side_debug,
            }
    if int(reject_width_cells) > 0 and min_side_width_cells <= int(reject_width_cells) and (
        not is_corridor_extension
        or _is_true_sliver_split(
            main,
            candidate_map=candidate_map,
            after_labels=after_labels,
            touched_labels=[int(item.get("label", 0)) for item in main],
            resolution_m=float(resolution_m),
            width_threshold_cells=int(reject_width_cells),
            area_threshold_m2=float(config.corridor_tiny_side_min_area_m2),
            long_axis_threshold_m=float(config.corridor_tiny_side_min_length_m),
            contact_length_threshold_cells=3,
        )
    ):
        return False, "reject_split_tiny_side_width_1_to_3_cells", candidate_map, {
            **anchor_debug,
            "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)),
            "components": component_info[:8],
            "min_side_width_cells": int(min_side_width_cells),
            "reject_if_side_width_cells_leq": int(reject_width_cells),
            "topology_touched_labels": touched,
        }
    width_reject = (not is_corridor_extension) and min_width < float(min_new_width_m)
    if len(main) < 2 or min(int(item["area_cells"]) for item in main) < min_cells or width_reject:
        return False, "reject_tiny_split", candidate_map, {
            **anchor_debug,
            "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)),
            "components": component_info[:8],
            "min_split_cells": int(min_cells),
            "topology_min_area_m2": float(min_area_m2),
            "topology_min_new_component_width_m": float(min_new_width_m),
            "topology_touched_labels": touched,
            "topology_min_side_width_cells": int(min_side_width_cells),
            "topology_min_side_area_cells": int(min(int(item["area_cells"]) for item in main)) if main else 0,
        }
    fragment_stats = _fragment_stats(
        before_labels=before_labels,
        after_labels=after_labels,
        before_count=int(before_count),
        after_count=int(after_count),
        candidate_map=candidate_map,
        min_cells=int(min_cells),
        local_radius_cells=int(config.local_fragment_radius_cells),
        touched_labels=touched,
        component_info=component_info[:8],
    )
    tiny_count = _tiny_fragment_reject_count(fragment_stats, str(config.tiny_fragment_count_mode))
    if tiny_count > int(config.max_tiny_fragment_count):
        candidate.fragmentation_penalty = float(tiny_count)
        return False, "reject_too_many_tiny_fragments", candidate_map, {
            **anchor_debug,
            "tiny_fragment_count": int(tiny_count),
            "tiny_fragment_count_mode": str(config.tiny_fragment_count_mode),
            "before_tiny_count": int(fragment_stats.before_tiny_count),
            "after_tiny_count": int(fragment_stats.after_tiny_count),
            "new_tiny_count": int(fragment_stats.new_tiny_count),
            "local_before_tiny_count": int(fragment_stats.local_before_tiny_count),
            "local_after_tiny_count": int(fragment_stats.local_after_tiny_count),
            "local_new_tiny_count": int(fragment_stats.local_new_tiny_count),
            "topology_fragment_stats": fragment_stats.to_dict(),
            "components": component_info[:8],
        }
    side_class = classify_cut_sides_local(
        candidate_map,
        free_clean=before,
        after_labels=after_labels,
        touched_labels=touched,
        component_info=main,
        distance_transform=distance_cells,
        corridor_skeleton=skeleton,
        resolution_m=float(resolution_m),
    )
    both_corridor = bool(side_class.get("side_a_corridor_like") and side_class.get("side_b_corridor_like"))
    one_corridor = bool(side_class.get("side_a_corridor_like") or side_class.get("side_b_corridor_like"))
    one_room = bool(side_class.get("side_a_room_like") or side_class.get("side_b_room_like"))
    if bool(config.reject_corridor_split) and both_corridor:
        candidate.corridor_split_penalty = 1.0
        return False, "reject_split_main_corridor_axis", candidate_map, {**anchor_debug, "components": main, "corridor_side_classification": side_class, "both_sides_corridor_like": True}
    if bool(config.reject_open_living_room_internal_split) and bool(side_class.get("both_open_living_room_like", False)):
        return False, "reject_open_living_room_internal_split", candidate_map, {**anchor_debug, "components": main, "corridor_side_classification": side_class}
    if candidate.kind == "corridor_room_neck_cut" and not (one_corridor and one_room):
        return False, "reject_not_corridor_room_neck", candidate_map, {**anchor_debug, "components": main, "corridor_side_classification": side_class, "one_corridor": one_corridor, "one_room": one_room}
    candidate.topology_gain_score = float(max(0, int(after_count) - int(before_count)))
    candidate.corridor_preservation_score = 1.0 if not both_corridor else 0.0
    return True, "", candidate_map, {
        **anchor_debug,
        "before_components": int(before_count),
        "after_components": int(after_count),
        "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)),
        "touched_components": touched,
        "topology_touched_labels": touched,
        "topology_min_side_width_cells": int(min_side_width_cells),
        "topology_min_side_area_cells": int(min(int(item["area_cells"]) for item in main)) if main else 0,
        "components": main,
        "tiny_fragment_count_mode": str(config.tiny_fragment_count_mode),
        "before_tiny_count": int(fragment_stats.before_tiny_count),
        "after_tiny_count": int(fragment_stats.after_tiny_count),
        "new_tiny_count": int(fragment_stats.new_tiny_count),
        "local_before_tiny_count": int(fragment_stats.local_before_tiny_count),
        "local_after_tiny_count": int(fragment_stats.local_after_tiny_count),
        "local_new_tiny_count": int(fragment_stats.local_new_tiny_count),
        "topology_fragment_stats": fragment_stats.to_dict(),
        "corridor_side_classification": side_class,
        "topology_min_area_m2": float(min_area_m2),
        "topology_gain_score": float(candidate.topology_gain_score),
        "corridor_preservation_score": float(candidate.corridor_preservation_score),
    }


def classify_cut_sides_local(
    candidate_map: np.ndarray,
    *,
    free_clean: np.ndarray,
    after_labels: np.ndarray,
    touched_labels: Sequence[int],
    component_info: Sequence[dict],
    distance_transform: np.ndarray,
    corridor_skeleton: np.ndarray,
    resolution_m: float,
) -> dict:
    _ = free_clean, after_labels, touched_labels, distance_transform, resolution_m
    first = dict(component_info[0]) if len(component_info) > 0 else {}
    second = dict(component_info[1]) if len(component_info) > 1 else {}
    skel = np.asarray(corridor_skeleton, dtype=bool)
    around_cut = dilate(np.asarray(candidate_map, dtype=bool), 2)
    axis_crossed = bool(np.any(around_cut & skel))
    side_a_corridor = bool(first.get("corridor_like", False))
    side_b_corridor = bool(second.get("corridor_like", False))
    side_a_room = bool(first.get("room_like", False)) or (float(first.get("area_m2", 0.0)) >= 0.30 and not side_a_corridor)
    side_b_room = bool(second.get("room_like", False)) or (float(second.get("area_m2", 0.0)) >= 0.30 and not side_b_corridor)
    side_a_open = bool(float(first.get("median_width_m", 0.0)) >= 1.80 and float(first.get("area_m2", 0.0)) >= 2.00 and not side_a_corridor)
    side_b_open = bool(float(second.get("median_width_m", 0.0)) >= 1.80 and float(second.get("area_m2", 0.0)) >= 2.00 and not side_b_corridor)
    return {
        "side_a_corridor_like": bool(side_a_corridor),
        "side_b_corridor_like": bool(side_b_corridor),
        "side_a_room_like": bool(side_a_room),
        "side_b_room_like": bool(side_b_room),
        "one_corridor_one_room": bool((side_a_corridor and side_b_room) or (side_b_corridor and side_a_room)),
        "side_a_open_living_room_like": bool(side_a_open),
        "side_b_open_living_room_like": bool(side_b_open),
        "both_open_living_room_like": bool(side_a_open and side_b_open),
        "corridor_axis_crossed": bool(axis_crossed),
        "corridor_axis_preserved": bool(not (side_a_corridor and side_b_corridor and axis_crossed)),
    }


def evaluate_corridor_separator_candidate_v16(
    candidate: SeparatorCandidate,
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    current_separator_map: np.ndarray,
    candidate_map: np.ndarray,
    corridor_skeleton: np.ndarray,
    resolution_m: float,
    config: TopologyTestConfig,
    anchor_debug: Mapping[str, object] | None = None,
) -> tuple[bool, str, np.ndarray, dict]:
    anchor = dict(anchor_debug or {})
    free = np.asarray(free_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    current = np.asarray(current_separator_map, dtype=bool)
    cut = np.asarray(candidate_map, dtype=bool) & free & ~current
    before = free & ~current
    after = before & ~cut
    before_labels, before_count = label_components(before, 4)
    after_labels, after_count = label_components(after, 4)
    distance_cells = ndimage.distance_transform_edt(before)
    global_gain = int(after_count) > int(before_count)

    local_radius = int(max(1, int(config.corridor_local_topology_radius_cells)))
    local_region = dilate(cut, local_radius) & before
    if not np.any(local_region):
        local_region = dilate(cut, max(1, int(config.local_fragment_radius_cells))) & before
    local_before_labels, local_before_count = label_components(before & local_region, 4)
    local_after_labels, local_after_count = label_components(after & local_region, 4)
    local_gain = int(local_after_count) > int(local_before_count)

    adjacent = dilate(cut, 1) & after
    global_touched = sorted({int(v) for v in np.unique(after_labels[adjacent]) if int(v) > 0})
    local_adjacent = dilate(cut, 1) & (after & local_region)
    local_touched = sorted({int(v) for v in np.unique(local_after_labels[local_adjacent]) if int(v) > 0})
    use_local_sides = len(global_touched) < 2 and len(local_touched) >= 2
    side_labels = local_touched if use_local_sides else global_touched
    labels_for_sides = local_after_labels if use_local_sides else after_labels

    if not (global_gain or local_gain):
        return False, "reject_no_topology_gain", cut, {
            **anchor,
            "corridor_topology_v16": True,
            "candidate_mask_cell_count": int(np.count_nonzero(cut)),
            "before_components": int(before_count),
            "after_components": int(after_count),
            "local_before_components": int(local_before_count),
            "local_after_components": int(local_after_count),
            "topology_touched_labels": [int(v) for v in side_labels],
        }
    if len(side_labels) < 2:
        return False, "reject_no_two_sides", cut, {
            **anchor,
            "corridor_topology_v16": True,
            "candidate_mask_cell_count": int(np.count_nonzero(cut)),
            "before_components": int(before_count),
            "after_components": int(after_count),
            "local_before_components": int(local_before_count),
            "local_after_components": int(local_after_count),
            "touched_components": [int(v) for v in side_labels],
        }

    component_info = _component_info_for_labels(
        labels_for_sides,
        side_labels,
        resolution_m=float(resolution_m),
        distance_cells=distance_cells,
    )
    component_info.sort(key=lambda item: int(item["area_cells"]), reverse=True)
    main = component_info[:2]
    min_area_m2 = float(config.corridor_min_split_area_m2)
    min_cells = max(int(config.min_new_component_area_cells), int(round(min_area_m2 / max(float(resolution_m) ** 2, 1e-9))))
    min_width = min(float(item.get("thickness_m", 0.0)) for item in main) if len(main) >= 2 else 0.0
    min_side_width_cells = int(round(float(min_width) / max(float(resolution_m), 1e-9))) if len(main) >= 2 else 0
    min_side_area_cells = int(min(int(item.get("area_cells", 0)) for item in main)) if main else 0
    reject_width_cells = int(config.corridor_reject_tiny_side_width_cells_leq)
    small_known_side_reject, small_known_side_debug = _small_known_side_reject_for_line_extension(
        component_info=main,
        labels=labels_for_sides,
        unknown_clean=unknown,
        resolution_m=float(resolution_m),
        config=config,
    )
    if small_known_side_reject:
        return False, "reject_small_known_side_low_unknown", cut, {
            **anchor,
            "corridor_topology_v16": True,
            "candidate_mask_cell_count": int(np.count_nonzero(cut)),
            "components": component_info[:8],
            "topology_touched_labels": [int(v) for v in side_labels],
            "topology_min_side_width_cells": int(min_side_width_cells),
            "topology_min_side_area_cells": int(min_side_area_cells),
            **small_known_side_debug,
        }
    true_sliver = False
    if reject_width_cells > 0 and min_side_width_cells <= reject_width_cells:
        true_sliver = _corridor_true_sliver_v16(
            main,
            candidate_map=cut,
            after_labels=labels_for_sides,
            touched_labels=[int(item.get("label", 0)) for item in main],
            resolution_m=float(resolution_m),
            width_threshold_cells=reject_width_cells,
            area_threshold_m2=float(config.corridor_tiny_side_min_area_m2),
            long_axis_threshold_m=float(config.corridor_tiny_side_min_length_m),
            contact_length_threshold_cells=3,
            accept_long_narrow_side=bool(config.corridor_accept_long_narrow_side),
        )
    if true_sliver:
        return False, "reject_split_tiny_side_width_1_to_3_cells", cut, {
            **anchor,
            "corridor_topology_v16": True,
            "candidate_mask_cell_count": int(np.count_nonzero(cut)),
            "components": component_info[:8],
            "min_side_width_cells": int(min_side_width_cells),
            "topology_min_side_width_cells": int(min_side_width_cells),
            "topology_min_side_area_cells": int(min_side_area_cells),
            "reject_if_side_width_cells_leq": int(reject_width_cells),
            "topology_touched_labels": [int(v) for v in side_labels],
            "corridor_long_narrow_side_accepted": False,
        }
    if len(main) < 2 or min_side_area_cells < min_cells:
        return False, "reject_tiny_split", cut, {
            **anchor,
            "corridor_topology_v16": True,
            "candidate_mask_cell_count": int(np.count_nonzero(cut)),
            "components": component_info[:8],
            "min_split_cells": int(min_cells),
            "topology_min_area_m2": float(min_area_m2),
            "topology_min_side_width_cells": int(min_side_width_cells),
            "topology_min_side_area_cells": int(min_side_area_cells),
            "topology_touched_labels": [int(v) for v in side_labels],
        }

    fragment_stats = _fragment_stats(
        before_labels=before_labels,
        after_labels=after_labels,
        before_count=int(before_count),
        after_count=int(after_count),
        candidate_map=cut,
        min_cells=int(min_cells),
        local_radius_cells=int(config.local_fragment_radius_cells),
        touched_labels=global_touched,
        component_info=component_info[:8],
    )
    tiny_count = _tiny_fragment_reject_count(fragment_stats, str(config.tiny_fragment_count_mode))
    if tiny_count > int(config.max_tiny_fragment_count):
        candidate.fragmentation_penalty = float(tiny_count)
        return False, "reject_too_many_tiny_fragments", cut, {
            **anchor,
            "corridor_topology_v16": True,
            "tiny_fragment_count": int(tiny_count),
            "tiny_fragment_count_mode": str(config.tiny_fragment_count_mode),
            "topology_fragment_stats": fragment_stats.to_dict(),
            "components": component_info[:8],
        }

    side_class = classify_cut_sides_local(
        cut,
        free_clean=before,
        after_labels=labels_for_sides,
        touched_labels=side_labels,
        component_info=main,
        distance_transform=distance_cells,
        corridor_skeleton=corridor_skeleton,
        resolution_m=float(resolution_m),
    )
    both_corridor = bool(side_class.get("side_a_corridor_like") and side_class.get("side_b_corridor_like"))
    if bool(config.reject_corridor_split) and both_corridor:
        candidate.corridor_split_penalty = 1.0
        return False, "reject_split_main_corridor_axis", cut, {
            **anchor,
            "corridor_topology_v16": True,
            "components": main,
            "corridor_side_classification": side_class,
            "both_sides_corridor_like": True,
        }

    candidate.topology_gain_score = float(max(0, int(after_count) - int(before_count), int(local_after_count) - int(local_before_count)))
    candidate.corridor_preservation_score = 1.0 if not both_corridor else 0.0
    long_narrow_accepted = bool(
        reject_width_cells > 0
        and min_side_width_cells <= reject_width_cells
        and not true_sliver
        and bool(config.corridor_accept_long_narrow_side)
    )
    return True, "", cut, {
        **anchor,
        "corridor_topology_v16": True,
        "before_components": int(before_count),
        "after_components": int(after_count),
        "local_before_components": int(local_before_count),
        "local_after_components": int(local_after_count),
        "candidate_mask_cell_count": int(np.count_nonzero(cut)),
        "touched_components": [int(v) for v in side_labels],
        "topology_touched_labels": [int(v) for v in side_labels],
        "topology_min_side_width_cells": int(min_side_width_cells),
        "topology_min_side_area_cells": int(min_side_area_cells),
        "components": main,
        "tiny_fragment_count_mode": str(config.tiny_fragment_count_mode),
        "topology_fragment_stats": fragment_stats.to_dict(),
        "corridor_side_classification": side_class,
        "topology_min_area_m2": float(min_area_m2),
        "topology_gain_score": float(candidate.topology_gain_score),
        "corridor_preservation_score": float(candidate.corridor_preservation_score),
        "corridor_long_narrow_side_accepted": bool(long_narrow_accepted),
        "corridor_local_topology_radius_cells": int(local_radius),
        **small_known_side_debug,
    }


def _candidate_separator_map_for_topology(
    candidate: SeparatorCandidate,
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    wall_candidate_clean: np.ndarray,
    current_separator_map: np.ndarray,
    resolution_m: float,
    thickness_cells: int,
    config: TopologyTestConfig,
) -> tuple[np.ndarray, dict]:
    free = np.asarray(free_clean, dtype=bool)
    wall = np.asarray(wall_candidate_clean, dtype=bool)
    current = np.asarray(current_separator_map, dtype=bool)
    if str(candidate.kind) not in {"line_extension_door_neck", "line_extension_corridor_separator", "step2_corridor_separator"}:
        mask = separator_mask_for_candidate(candidate, free.shape, thickness_cells) & free & ~current
        return mask.astype(bool), {"centered_extension_cut_enabled": False}
    mask, debug = _centered_line_extension_free_run_mask(
        candidate,
        free_clean=free,
        unknown_clean=unknown_clean,
        wall_candidate_clean=wall,
        current_separator_map=current,
        resolution_m=float(resolution_m),
        thickness_cells=int(thickness_cells),
        config=config,
    )
    return mask.astype(bool), debug


def _centered_line_extension_free_run_mask(
    candidate: SeparatorCandidate,
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    wall_candidate_clean: np.ndarray,
    current_separator_map: np.ndarray,
    resolution_m: float,
    thickness_cells: int,
    config: TopologyTestConfig,
) -> tuple[np.ndarray, dict]:
    free = np.asarray(free_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    wall = np.asarray(wall_candidate_clean, dtype=bool)
    current = np.asarray(current_separator_map, dtype=bool)
    cells = _ordered_line_cells(candidate.p0_rc, candidate.p1_rc, free.shape)
    valid = np.asarray([bool(free[int(r), int(c)] and not current[int(r), int(c)]) for r, c in cells], dtype=bool)
    bridgeable = np.asarray(
        [
            bool(
                not current[int(r), int(c)]
                and not wall[int(r), int(c)]
                and (
                    (bool(config.centered_extension_bridge_unknown) and unknown[int(r), int(c)])
                    or (bool(config.centered_extension_bridge_nonfree_if_between_free) and not free[int(r), int(c)])
                )
            )
            for r, c in cells
        ],
        dtype=bool,
    )
    bridged_gap_count = 0
    bridged_indices: list[int] = []
    if bool(config.centered_extension_allow_small_gaps):
        valid, bridged_gap_count, bridged_indices = _bridge_small_line_gaps(
            valid,
            bridgeable,
            int(config.centered_extension_max_bridge_gap_cells),
        )
    if not np.any(valid):
        mask = separator_mask_for_candidate(candidate, free.shape, thickness_cells) & free & ~current
        return mask.astype(bool), {
            "centered_extension_cut_enabled": True,
            "centered_extension_cut_applied": False,
            "centered_extension_cut_reason": "no_free_cells_on_extension",
            "candidate_mask_cell_count": int(np.count_nonzero(mask)),
        }

    midpoint = 0.5 * (np.asarray(candidate.p0_rc, dtype=np.float32) + np.asarray(candidate.p1_rc, dtype=np.float32))
    valid_indices = np.flatnonzero(valid)
    distances = np.sum((cells[valid_indices].astype(np.float32) - midpoint[None, :]) ** 2, axis=1)
    center_idx = int(valid_indices[int(np.argmin(distances))])
    lo = center_idx
    while lo - 1 >= 0 and bool(valid[lo - 1]):
        lo -= 1
    hi = center_idx
    while hi + 1 < len(valid) and bool(valid[hi + 1]):
        hi += 1

    cut_cells = cells[lo : hi + 1]
    base = np.zeros_like(free, dtype=bool)
    base[cut_cells[:, 0], cut_cells[:, 1]] = True
    mask = dilate(base, int(thickness_cells)) if int(thickness_cells) > 0 else base
    mask = mask & free & ~current
    wall_dilated = dilate(wall, 1)
    p0_touch = bool(np.any(dilate(_single_cell_mask(tuple(int(v) for v in cut_cells[0]), free.shape), 1) & wall_dilated))
    p1_touch = bool(np.any(dilate(_single_cell_mask(tuple(int(v) for v in cut_cells[-1]), free.shape), 1) & wall_dilated))
    p0_cut = cut_cells[0].astype(np.float32)
    p1_cut = cut_cells[-1].astype(np.float32)
    original_p0 = np.asarray(candidate.p0_rc, dtype=np.float32).copy()
    original_p1 = np.asarray(candidate.p1_rc, dtype=np.float32).copy()
    candidate.p0_rc = p0_cut
    candidate.p1_rc = p1_cut
    candidate.length_m = float(max(1, int(np.max(np.abs(p1_cut - p0_cut)) + 1)) * float(resolution_m))
    debug = {
        "centered_extension_cut_enabled": True,
        "centered_extension_cut_applied": True,
        "centered_extension_cut_mode": "middle_contiguous_free_run",
        "centered_extension_allow_small_gaps": bool(config.centered_extension_allow_small_gaps),
        "centered_extension_bridge_gap_count": int(bridged_gap_count),
        "centered_extension_bridge_cells": [[int(cells[idx][0]), int(cells[idx][1])] for idx in bridged_indices[:64]],
        "centered_extension_cut_original_p0": [int(round(float(v))) for v in original_p0.tolist()],
        "centered_extension_cut_original_p1": [int(round(float(v))) for v in original_p1.tolist()],
        "centered_extension_cut_p0": [int(v) for v in p0_cut.astype(np.int32).tolist()],
        "centered_extension_cut_p1": [int(v) for v in p1_cut.astype(np.int32).tolist()],
        "centered_extension_cut_cell_count": int(len(cut_cells)),
        "centered_extension_cut_mask_cell_count": int(np.count_nonzero(mask)),
        "candidate_mask_cell_count": int(np.count_nonzero(mask)),
        "candidate_cut_touch_wall_a": bool(p0_touch),
        "candidate_cut_touch_wall_b": bool(p1_touch),
        "candidate_cut_not_adjacent_to_barrier": bool(not (p0_touch and p1_touch)),
        "centered_extension_cut_trimmed": bool(
            not np.array_equal(np.rint(original_p0).astype(np.int32), p0_cut.astype(np.int32))
            or not np.array_equal(np.rint(original_p1).astype(np.int32), p1_cut.astype(np.int32))
        ),
    }
    candidate.debug.update(debug)
    return mask.astype(bool), debug


def _ordered_line_cells(p0_rc: np.ndarray, p1_rc: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    p0 = np.rint(np.asarray(p0_rc, dtype=np.float32)).astype(np.int32)
    p1 = np.rint(np.asarray(p1_rc, dtype=np.float32)).astype(np.int32)
    dr = int(p1[0] - p0[0])
    dc = int(p1[1] - p0[1])
    steps = max(abs(dr), abs(dc))
    if steps <= 0:
        rows = np.asarray([int(p0[0])], dtype=np.int32)
        cols = np.asarray([int(p0[1])], dtype=np.int32)
    else:
        rows = np.rint(np.linspace(int(p0[0]), int(p1[0]), steps + 1)).astype(np.int32)
        cols = np.rint(np.linspace(int(p0[1]), int(p1[1]), steps + 1)).astype(np.int32)
    inside = (rows >= 0) & (rows < int(shape[0])) & (cols >= 0) & (cols < int(shape[1]))
    coords = np.stack([rows[inside], cols[inside]], axis=1)
    if len(coords) <= 1:
        return coords.astype(np.int32)
    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
    return coords[keep].astype(np.int32)


def _single_cell_mask(cell: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r, c = int(cell[0]), int(cell[1])
    if 0 <= r < shape[0] and 0 <= c < shape[1]:
        out[r, c] = True
    return out


def _bridge_small_false_gaps(values: np.ndarray, max_gap_cells: int) -> tuple[np.ndarray, int]:
    src = np.asarray(values, dtype=bool).copy()
    if int(max_gap_cells) <= 0 or src.size <= 2:
        return src, 0
    count = 0
    idx = 0
    while idx < src.size:
        if bool(src[idx]):
            idx += 1
            continue
        start = idx
        while idx < src.size and not bool(src[idx]):
            idx += 1
        end = idx
        if start == 0 or end >= src.size:
            continue
        if bool(src[start - 1]) and bool(src[end]) and (end - start) <= int(max_gap_cells):
            src[start:end] = True
            count += 1
    return src, int(count)


def _bridge_small_line_gaps(values: np.ndarray, bridgeable: np.ndarray, max_gap_cells: int) -> tuple[np.ndarray, int, list[int]]:
    src = np.asarray(values, dtype=bool).copy()
    bridge = np.asarray(bridgeable, dtype=bool)
    if src.shape != bridge.shape:
        raise ValueError("line bridge arrays must have the same shape")
    bridged: list[int] = []
    if int(max_gap_cells) <= 0 or src.size <= 2:
        return src, 0, bridged
    count = 0
    idx = 0
    while idx < src.size:
        if bool(src[idx]):
            idx += 1
            continue
        start = idx
        while idx < src.size and not bool(src[idx]):
            idx += 1
        end = idx
        if start == 0 or end >= src.size:
            continue
        if bool(src[start - 1]) and bool(src[end]) and (end - start) <= int(max_gap_cells) and bool(np.all(bridge[start:end])):
            src[start:end] = True
            bridged.extend(range(start, end))
            count += 1
    return src, int(count), bridged


def _component_info_for_labels(labels: np.ndarray, touched_labels: Sequence[int], *, resolution_m: float, distance_cells: np.ndarray) -> list[dict]:
    arr = np.asarray(labels, dtype=np.int32)
    out: list[dict] = []
    for label in touched_labels:
        mask = arr == int(label)
        if not np.any(mask):
            continue
        info = component_metrics(mask, float(resolution_m), distance_cells)
        info["label"] = int(label)
        out.append(info)
    return out


def _corridor_true_sliver_v16(
    component_info: Sequence[dict],
    *,
    candidate_map: np.ndarray,
    after_labels: np.ndarray,
    touched_labels: Sequence[int],
    resolution_m: float,
    width_threshold_cells: int,
    area_threshold_m2: float,
    long_axis_threshold_m: float,
    contact_length_threshold_cells: int,
    accept_long_narrow_side: bool,
) -> bool:
    if len(component_info) < 2:
        return True
    by_label = {int(item.get("label", 0)): item for item in component_info if int(item.get("label", 0)) > 0}
    contact = _candidate_contact_lengths(candidate_map, after_labels, touched_labels)
    for label in touched_labels[:2]:
        info = by_label.get(int(label))
        if not info:
            continue
        width_cells = int(round(float(info.get("thickness_m", 0.0)) / max(float(resolution_m), 1e-9)))
        if width_cells > int(width_threshold_cells):
            continue
        area_m2 = float(info.get("area_m2", 0.0))
        long_axis_m = float(info.get("length_m", 0.0))
        contact_cells = int(contact.get(int(label), 0))
        if bool(accept_long_narrow_side) and area_m2 >= float(area_threshold_m2) and long_axis_m >= float(long_axis_threshold_m):
            continue
        if area_m2 < float(area_threshold_m2) and long_axis_m < float(long_axis_threshold_m) and contact_cells <= int(contact_length_threshold_cells):
            return True
    return False


def _is_true_sliver_split(
    component_info: Sequence[dict],
    *,
    candidate_map: np.ndarray,
    after_labels: np.ndarray,
    touched_labels: Sequence[int],
    resolution_m: float,
    width_threshold_cells: int,
    area_threshold_m2: float,
    long_axis_threshold_m: float,
    contact_length_threshold_cells: int,
) -> bool:
    if len(component_info) < 2:
        return True
    by_label = {int(item.get("label", 0)): item for item in component_info if int(item.get("label", 0)) > 0}
    contact = _candidate_contact_lengths(candidate_map, after_labels, touched_labels)
    for label in touched_labels[:2]:
        info = by_label.get(int(label))
        if not info:
            continue
        width_cells = int(round(float(info.get("thickness_m", 0.0)) / max(float(resolution_m), 1e-9)))
        area_m2 = float(info.get("area_m2", 0.0))
        long_axis_m = float(info.get("length_m", 0.0))
        contact_cells = int(contact.get(int(label), 0))
        if (
            width_cells <= int(width_threshold_cells)
            and area_m2 < float(area_threshold_m2)
            and long_axis_m < float(long_axis_threshold_m)
            and contact_cells <= int(contact_length_threshold_cells)
        ):
            return True
    return False


def _candidate_contact_lengths(candidate_map: np.ndarray, labels: np.ndarray, touched_labels: Sequence[int]) -> dict[int, int]:
    contact_region = dilate(np.asarray(candidate_map, dtype=bool), 1) & ~np.asarray(candidate_map, dtype=bool)
    label_arr = np.asarray(labels, dtype=np.int32)
    out: dict[int, int] = {}
    for label in touched_labels:
        out[int(label)] = int(np.count_nonzero(contact_region & (label_arr == int(label))))
    return out


def _candidate_center(candidate: SeparatorCandidate) -> np.ndarray:
    return 0.5 * (np.asarray(candidate.p0_rc, dtype=np.float32) + np.asarray(candidate.p1_rc, dtype=np.float32))


def _angle_diff_deg(theta_a: float, theta_b: float) -> float:
    diff = abs(float(theta_a) - float(theta_b)) % np.pi
    if diff > np.pi * 0.5:
        diff = np.pi - diff
    return float(np.degrees(diff))


def _compound_pair_score(
    cand_a: SeparatorCandidate,
    cand_b: SeparatorCandidate,
    debug: Mapping[str, object],
    config: CompoundSeparatorConfig,
) -> float:
    return float(
        float(cand_a.confidence)
        + float(cand_b.confidence)
        + 0.25 * float(debug.get("compound_global_gain", 0.0))
        + 0.05 * float(cand_a.length_m + cand_b.length_m)
        - float(config.penalty_per_cut) * 2.0
    )


def _small_known_side_reject_for_line_extension(
    *,
    component_info: Sequence[dict],
    labels: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    config: TopologyTestConfig,
) -> tuple[bool, dict]:
    enabled = bool(config.reject_small_known_side_for_line_extensions)
    area_threshold = float(config.small_known_side_area_m2)
    unknown_threshold = float(config.small_known_side_unknown_ratio_max)
    dilation_cells = max(1, int(config.small_known_side_boundary_dilation_cells))
    debug: dict[str, object] = {
        "small_known_side_gate_enabled": bool(enabled),
        "small_known_side_area_threshold_m2": float(area_threshold),
        "small_known_side_unknown_ratio_threshold": float(unknown_threshold),
        "small_known_side_boundary_dilation_cells": int(dilation_cells),
        "small_known_side_components": [],
    }
    if not enabled or area_threshold <= 0.0 or len(component_info) < 2:
        debug["small_known_side_rejected"] = False
        return False, debug

    label_arr = np.asarray(labels, dtype=np.int32)
    unknown = np.asarray(unknown_clean, dtype=bool)
    rejected = False
    entries: list[dict[str, object]] = []
    for info in component_info[:2]:
        label = int(info.get("label", 0))
        if label <= 0:
            continue
        comp = label_arr == int(label)
        if not np.any(comp):
            continue
        area_m2 = float(info.get("area_m2", float(np.count_nonzero(comp)) * float(resolution_m) ** 2))
        ring = dilate(comp, dilation_cells) & ~comp
        ring_cells = int(np.count_nonzero(ring))
        unknown_cells = int(np.count_nonzero(ring & unknown))
        unknown_ratio = 0.0 if ring_cells <= 0 else float(unknown_cells) / float(ring_cells)
        entry = {
            "label": int(label),
            "area_m2": float(area_m2),
            "boundary_cells": int(ring_cells),
            "boundary_unknown_cells": int(unknown_cells),
            "boundary_unknown_ratio": float(unknown_ratio),
            "rejected": bool(area_m2 < area_threshold and unknown_ratio < unknown_threshold),
        }
        if bool(entry["rejected"]):
            rejected = True
        entries.append(entry)
    debug["small_known_side_components"] = entries
    debug["small_known_side_rejected"] = bool(rejected)
    debug["small_known_side_reject_reason"] = "reject_small_known_side_low_unknown" if rejected else None
    return bool(rejected), debug


def _fragment_stats(
    *,
    before_labels: np.ndarray,
    after_labels: np.ndarray,
    before_count: int,
    after_count: int,
    candidate_map: np.ndarray,
    min_cells: int,
    local_radius_cells: int,
    touched_labels: Sequence[int],
    component_info: Sequence[dict],
) -> TopologyFragmentStats:
    before_tiny = _tiny_labels(before_labels, int(before_count), int(min_cells))
    after_tiny = _tiny_labels(after_labels, int(after_count), int(min_cells))
    local_region = dilate(np.asarray(candidate_map, dtype=bool), int(local_radius_cells))
    local_before = _labels_touching_region(before_labels, before_tiny, local_region)
    local_after = _labels_touching_region(after_labels, after_tiny, local_region)
    return TopologyFragmentStats(
        before_component_count=int(before_count),
        after_component_count=int(after_count),
        before_tiny_count=int(len(before_tiny)),
        after_tiny_count=int(len(after_tiny)),
        new_tiny_count=max(0, int(len(after_tiny)) - int(len(before_tiny))),
        local_before_tiny_count=int(len(local_before)),
        local_after_tiny_count=int(len(local_after)),
        local_new_tiny_count=max(0, int(len(local_after)) - int(len(local_before))),
        touched_labels=[int(v) for v in touched_labels],
        component_info=list(component_info),
    )


def _tiny_labels(labels: np.ndarray, count: int, min_cells: int) -> set[int]:
    out: set[int] = set()
    arr = np.asarray(labels, dtype=np.int32)
    for label in range(1, int(count) + 1):
        if int(np.count_nonzero(arr == int(label))) < int(min_cells):
            out.add(int(label))
    return out


def _labels_touching_region(labels: np.ndarray, label_set: set[int], region: np.ndarray) -> set[int]:
    if not label_set:
        return set()
    arr = np.asarray(labels, dtype=np.int32)
    values = {int(v) for v in np.unique(arr[np.asarray(region, dtype=bool)]) if int(v) > 0}
    return {int(v) for v in values if int(v) in label_set}


def _tiny_fragment_reject_count(stats: TopologyFragmentStats, mode: str) -> int:
    mode = str(mode or "local_delta").strip().lower()
    if mode == "absolute":
        return int(stats.after_tiny_count)
    if mode == "global_delta":
        return int(stats.new_tiny_count)
    return int(stats.local_new_tiny_count)


def _kind_order(kind: str) -> int:
    return {
        "physical_wall_completion": 0,
        "missed_scan_gap_closure": 1,
        "short_unknown_gap_closure": 2,
        "doorway_virtual_cut": 3,
        "line_extension_door_neck": 3,
        "single_sided_wall_extension": 4,
        "corridor_room_neck_cut": 5,
        "line_extension_corridor_separator": 6,
    }.get(str(kind), 99)


def _candidate_sort_key(item: SeparatorCandidate) -> tuple:
    kind = str(item.kind)
    if kind == "corridor_room_neck_cut":
        return (
            _kind_order(kind),
            -float(item.confidence),
            -float(item.length_m),
            int(item.candidate_id),
        )
    return (
        _kind_order(kind),
        -float(item.confidence),
        -float(item.debug.get("anchor_score", 0.0)),
        float(item.length_m),
        int(item.candidate_id),
    )


def _kind_counts(candidates: Sequence[SeparatorCandidate]) -> dict:
    out: dict[str, int] = {}
    for candidate in candidates:
        out[str(candidate.kind)] = out.get(str(candidate.kind), 0) + 1
    return out


def _reason_counts(candidates: Sequence[SeparatorCandidate]) -> dict:
    out: dict[str, int] = {}
    for candidate in candidates:
        reason = str(candidate.reject_reason or "")
        out[reason] = out.get(reason, 0) + 1
    return out
