from __future__ import annotations

import pytest

from voxroom_online.isaac_runtime.comparison.metadata_gate import assert_main_experiment_metadata


def test_metadata_gate_accepts_original_dude() -> None:
    assert_main_experiment_metadata(
        {
            "method": "dude_incremental",
            "runner_type": "original_ros",
            "main_experiment_allowed": True,
            "original_repo": "lfermin77/Incremental_DuDe_ROS",
            "original_repo_commit": "abc123",
            "input_topic": "/map",
            "output_topic": "/tagged_image",
            "incremental_state_reset_per_scene": True,
            "incremental_order_enforced": True,
            "map_resolution_m": 0.05,
        },
        "dude_incremental",
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"method": "rose2", "runner_type": "python_smoke_fallback", "main_experiment_allowed": True},
        {"method": "rose2", "runner_type": "original_ros", "main_experiment_allowed": False},
        {"method": "rose2", "runner_type": "original_ros", "main_experiment_allowed": True, "approximation_note": "placeholder_not_canonical"},
    ],
)
def test_metadata_gate_rejects_non_main_markers(metadata: dict) -> None:
    with pytest.raises(ValueError):
        assert_main_experiment_metadata(metadata, str(metadata["method"]))


def test_metadata_gate_accepts_original_ipa_action() -> None:
    assert_main_experiment_metadata(
        {
            "method": "distance_transform",
            "runner_type": "original_ros_action",
            "main_experiment_allowed": True,
            "original_repo": "ipa320/ipa_coverage_planning",
            "original_repo_commit": "abc123",
            "original_package": "ipa_room_segmentation",
            "room_segmentation_algorithm": 2,
            "algorithm_id_verified": True,
            "action_type": "ipa_building_msgs/MapSegmentationAction",
            "action_server": "room_segmentation_server",
            "input_image_encoding": "mono8",
            "input_free_value": 255,
            "input_occupied_value": 0,
            "map_resolution_m": 0.05,
        },
        "distance_transform",
    )


def test_metadata_gate_rejects_wrong_ipa_algorithm_id() -> None:
    with pytest.raises(ValueError, match="room_segmentation_algorithm"):
        assert_main_experiment_metadata(
            {
                "method": "voronoi",
                "runner_type": "original_ros_action",
                "main_experiment_allowed": True,
                "original_repo": "ipa320/ipa_coverage_planning",
                "original_repo_commit": "abc123",
                "original_package": "ipa_room_segmentation",
                "room_segmentation_algorithm": 999,
                "algorithm_id_verified": True,
                "action_type": "ipa_building_msgs/MapSegmentationAction",
                "action_server": "room_segmentation_server",
                "input_image_encoding": "mono8",
                "input_free_value": 255,
                "input_occupied_value": 0,
                "map_resolution_m": 0.05,
            },
            "voronoi",
        )
