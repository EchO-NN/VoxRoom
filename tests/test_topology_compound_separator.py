from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.mapping.online_roomseg.separator_candidates import SeparatorCandidate
from voxroom_online.isaac_runtime.mapping.online_roomseg.topology_tests import (
    CompoundSeparatorConfig,
    TopologyTestConfig,
    select_compound_separator_pairs,
)


def _candidate(
    candidate_id: int,
    p0: tuple[int, int],
    p1: tuple[int, int],
    *,
    reject_reason: str = "reject_no_topology_gain",
) -> SeparatorCandidate:
    item = SeparatorCandidate(
        candidate_id=int(candidate_id),
        kind="line_extension_corridor_separator",
        p0_rc=np.asarray(p0, dtype=np.float32),
        p1_rc=np.asarray(p1, dtype=np.float32),
        theta=float(np.pi * 0.5),
        length_m=0.20,
        confidence=0.90,
        source_segment_ids=[],
    )
    item.reject_reason = str(reject_reason)
    item.debug.update(
        {
            "anchor_score": 1.0,
            "candidate_cut_touch_wall_a": True,
            "candidate_cut_touch_wall_b": True,
        }
    )
    return item


def test_compound_separator_pair_accepts_two_half_cuts_that_split_together() -> None:
    free = np.ones((7, 7), dtype=bool)
    map_a = np.zeros_like(free)
    map_b = np.zeros_like(free)
    map_a[0:4, 3] = True
    map_b[3:7, 3] = True
    cand_a = _candidate(1, (0, 3), (3, 3))
    cand_b = _candidate(2, (3, 3), (6, 3))
    cfg = TopologyTestConfig.from_mapping(
        {
            "min_new_component_area_cells": 1,
            "compound_separator": {
                "enabled": True,
                "min_candidate_confidence": 0.5,
                "min_anchor_score": 1.0,
                "min_pair_spacing_m": 0.0,
                "max_pair_spacing_m": 1.0,
                "min_pair_combined_length_m": 0.1,
                "min_major_side_area_m2": 0.01,
                "min_major_side_width_m": 0.01,
                "max_major_area_ratio": 20.0,
                "max_tiny_fragment_delta": 0,
                "reject_if_crosses_structural_wall": False,
            },
        }
    )

    accepted, accepted_map, debug = select_compound_separator_pairs(
        [cand_a, cand_b],
        rejected_maps={1: map_a, 2: map_b},
        free_clean=free,
        unknown_clean=np.zeros_like(free),
        wall_candidate_clean=np.zeros_like(free),
        current_separator_map=np.zeros_like(free),
        resolution_m=0.05,
        config=cfg,
    )

    assert [item.candidate_id for item in accepted] == [1, 2]
    assert int(np.count_nonzero(accepted_map)) == 7
    assert debug["compound_separator_pair_evaluated_count"] == 1
    assert debug["compound_separator_pair_accepted_count"] == 1
    assert debug["compound_separator_pair_rejected_count"] == 0
    assert debug["compound_separator_pool_count"] == 2
    assert debug["compound_separator_reject_reason_counts"] == {}
    assert debug["compound_separator_accepted_pair_count"] == 1
    assert cand_a.debug["compound_separator_accepted"] is True


def test_compound_separator_pool_includes_small_known_and_corridor_axis_rejects() -> None:
    free = np.ones((7, 7), dtype=bool)
    map_a = np.zeros_like(free)
    map_b = np.zeros_like(free)
    map_a[0:4, 3] = True
    map_b[3:7, 3] = True
    cand_a = _candidate(10, (0, 3), (3, 3), reject_reason="reject_small_known_side_low_unknown")
    cand_b = _candidate(11, (3, 3), (6, 3), reject_reason="reject_split_main_corridor_axis")
    cfg = TopologyTestConfig.from_mapping(
        {
            "min_new_component_area_cells": 1,
            "compound_separator": {
                "enabled": True,
                "min_candidate_confidence": 0.5,
                "min_anchor_score": 1.0,
                "min_pair_spacing_m": 0.0,
                "max_pair_spacing_m": 1.0,
                "min_pair_combined_length_m": 0.1,
                "min_major_side_area_m2": 0.01,
                "min_major_side_width_m": 0.01,
                "max_major_area_ratio": 20.0,
                "max_tiny_fragment_delta": 0,
                "reject_if_crosses_structural_wall": False,
            },
        }
    )

    accepted, accepted_map, debug = select_compound_separator_pairs(
        [cand_a, cand_b],
        rejected_maps={10: map_a, 11: map_b},
        free_clean=free,
        unknown_clean=np.zeros_like(free),
        wall_candidate_clean=np.zeros_like(free),
        current_separator_map=np.zeros_like(free),
        resolution_m=0.05,
        config=cfg,
    )

    assert [item.candidate_id for item in accepted] == [10, 11]
    assert int(np.count_nonzero(accepted_map)) == 7
    assert debug["compound_separator_pool_count"] == 2
    assert debug["compound_separator_pre_filter_reason_counts"]["reject_small_known_side_low_unknown"] == 1
    assert debug["compound_separator_pre_filter_reason_counts"]["reject_split_main_corridor_axis"] == 1
    pair = debug["compound_separator_accepted_pairs"][0]
    assert pair["compound_pair_contains_small_known_side_candidate"] is True
    assert pair["compound_pair_contains_main_corridor_axis_candidate"] is True
    assert pair["compound_pair_contains_split_main_corridor_axis"] is True
    assert pair["compound_pair_contains_small_known_side_low_unknown"] is True
    assert pair["compound_pair_reason_specific_safety_gate"] == "combined_gain_one_no_tiny_fragments_major_sides"


def test_compound_separator_default_pool_and_excluded_debug_fields() -> None:
    cfg_default = CompoundSeparatorConfig()
    assert cfg_default.max_candidates_per_parent_component == 16
    assert "reject_small_known_side_low_unknown" in cfg_default.source_reject_reasons
    assert "reject_split_main_corridor_axis" in cfg_default.source_reject_reasons

    free = np.ones((5, 5), dtype=bool)
    disallowed_reason = _candidate(20, (0, 2), (4, 2), reject_reason="reject_not_pairable_for_test")
    low_conf = _candidate(21, (0, 1), (4, 1))
    low_conf.confidence = 0.1
    bad_kind = _candidate(22, (0, 3), (4, 3))
    bad_kind.kind = "unsupported_kind"
    missing_anchor = _candidate(23, (1, 0), (1, 4))
    missing_anchor.debug["candidate_cut_touch_wall_b"] = False
    missing_map = _candidate(24, (3, 0), (3, 4))
    cfg = TopologyTestConfig.from_mapping(
        {
            "compound_separator": {
                "enabled": True,
                "min_candidate_confidence": 0.5,
                "min_anchor_score": 1.0,
                "debug_max_excluded_candidates_to_store": 8,
            },
        }
    )

    accepted, accepted_map, debug = select_compound_separator_pairs(
        [disallowed_reason, low_conf, bad_kind, missing_anchor, missing_map],
        rejected_maps={
            20: np.ones_like(free),
            21: np.ones_like(free),
            22: np.ones_like(free),
            23: np.ones_like(free),
        },
        free_clean=free,
        unknown_clean=np.zeros_like(free),
        wall_candidate_clean=np.zeros_like(free),
        current_separator_map=np.zeros_like(free),
        resolution_m=0.05,
        config=cfg,
    )

    assert accepted == []
    assert not np.any(accepted_map)
    assert debug["compound_separator_allowed_source_reject_reasons"]
    assert debug["compound_separator_allowed_kinds"]
    counts = debug["compound_separator_excluded_filter_counts"]
    assert counts["reject_reason_not_pairable"] == 1
    assert counts["confidence_too_low"] == 1
    assert counts["kind_not_allowed"] == 1
    assert counts["missing_two_anchors"] == 1
    assert counts["missing_rejected_map"] == 1
    sample = debug["compound_separator_excluded_candidates_sample"][0]
    assert "filter_reason" in sample
    assert "touch_wall_a" in sample
    assert "touch_wall_b" in sample
    assert "theta" in sample
    assert "length_m" in sample
