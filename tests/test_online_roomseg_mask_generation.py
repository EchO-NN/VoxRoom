from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.evaluation.online_roomseg.annotation_schema import LineAnnotation, MergeGroup
from voxroom_online.isaac_runtime.evaluation.online_roomseg.mask_generation import (
    GtGenerationConfig,
    bresenham_rc,
    generate_gt_from_annotation,
)


def test_bresenham_rc_vertical_line() -> None:
    assert bresenham_rc((0, 5), (3, 5)) == [(0, 5), (1, 5), (2, 5), (3, 5)]


def test_line_split_two_rooms_and_cut_pixels_filled() -> None:
    domain = np.ones((10, 10), dtype=bool)
    result = generate_gt_from_annotation(
        eval_domain=domain,
        split_lines=[LineAnnotation(id="line_0001", p0_rc=(0, 5), p1_rc=(9, 5), width_cells=1)],
        merge_groups=[],
        config=GtGenerationConfig(line_width_cells=1, preclose_radius_cells=0, connectivity=4),
    )

    assert result.metadata["room_count"] == 2
    assert result.metadata["unlabeled_domain_pixels"] == 0
    assert int(np.count_nonzero(result.labels > 0)) == 100


def test_merge_group_combines_two_components() -> None:
    domain = np.zeros((6, 7), dtype=bool)
    domain[:, :3] = True
    domain[:, 4:] = True

    split = generate_gt_from_annotation(
        eval_domain=domain,
        split_lines=[],
        merge_groups=[],
        config=GtGenerationConfig(preclose_radius_cells=0),
    )
    merged = generate_gt_from_annotation(
        eval_domain=domain,
        split_lines=[],
        merge_groups=[MergeGroup(id="merge_0001", component_ids=(1, 2))],
        config=GtGenerationConfig(preclose_radius_cells=0),
    )

    assert split.metadata["room_count"] == 2
    assert merged.metadata["room_count"] == 1


def test_vertical_domain_drives_split_then_projects_to_navigation_domain() -> None:
    navigation = np.ones((6, 7), dtype=bool)
    navigation[:, 3] = False
    vertical = np.ones((6, 7), dtype=bool)

    result = generate_gt_from_annotation(
        eval_domain=navigation,
        segmentation_domain=vertical,
        split_lines=[],
        merge_groups=[],
        config=GtGenerationConfig(preclose_radius_cells=0),
    )

    assert result.metadata["room_count"] == 1
    assert result.metadata["output_domain_pixels"] == int(np.count_nonzero(navigation))
    assert result.metadata["segmentation_domain_pixels"] == int(np.count_nonzero(vertical))
    assert int(np.count_nonzero(result.labels > 0)) == int(np.count_nonzero(navigation))


def test_wall_completion_line_is_a_red_barrier_in_vertical_domain() -> None:
    domain = np.ones((8, 8), dtype=bool)

    result = generate_gt_from_annotation(
        eval_domain=domain,
        segmentation_domain=domain,
        split_lines=[LineAnnotation(id="wall_0001", p0_rc=(0, 4), p1_rc=(7, 4), width_cells=1, kind="wall_completion")],
        merge_groups=[],
        config=GtGenerationConfig(line_width_cells=1, preclose_radius_cells=0),
    )

    assert result.metadata["room_count"] == 2
    assert result.metadata["wall_completion_line_count"] == 1
    assert result.metadata["separator_line_count"] == 0
