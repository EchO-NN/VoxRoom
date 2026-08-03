import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import quaternion

from env.habitat.exploration_env import (
    habitat_depth_to_meters,
    habitat_rotation_yaw_in_voxroom,
    habitat_states_to_voxroom,
)
from voxroom_sidecar import (
    _door_segments_rc,
    attach_voxroom_projection,
    load_navigation_projection,
)


class VoxRoomGeometryTests(unittest.TestCase):
    def test_coverage_evaluation_has_a_mode_specific_completion_contract(self):
        repository_root = Path(__file__).resolve().parents[1]
        runtime_source = (
            repository_root / "explorable_with_door_detection.py"
        ).read_text(encoding="utf-8")
        launcher_source = (
            repository_root / "scripts" / "run_habitat_test.sh"
        ).read_text(encoding="utf-8")
        validator_source = (
            repository_root / "scripts" / "validate_run.py"
        ).read_text(encoding="utf-8")

        for source in (runtime_source, launcher_source, validator_source):
            self.assertIn("coverage_episode_step_limit_reached", source)
            self.assertIn("coverage_exploration_completed", source)
        self.assertIn('"coverage_episode_completed"', runtime_source)
        self.assertIn("--roomseg-coverage-eval", launcher_source)
        self.assertIn("--roomseg-coverage-eval", validator_source)
        self.assertIn(
            "context_mode and not args.roomseg_coverage_eval",
            validator_source,
        )
        self.assertNotIn("confirm_pending_transition", runtime_source)
        self.assertIn('"room_transition_advanced"', runtime_source)
        self.assertIn(
            'transition_method="upstream_source_semantics"',
            runtime_source,
        )

    def test_missing_accepted_door_collection_is_rejected(self):
        with self.assertRaisesRegex(
            TypeError,
            "accepted door collection is required",
        ):
            _door_segments_rc(None)

    def test_original_xy_door_endpoints_are_converted_to_row_col(self):
        segments = _door_segments_rc(
            [{"start": [17, 23], "end": [41, 29]}]
        )

        self.assertEqual(segments, [[23, 17, 29, 41]])

    def test_active_room_keeps_native_mapping_pose_and_planner_pipeline(self):
        repository_root = Path(__file__).resolve().parents[1]
        runtime_source = (
            repository_root / "explorable_with_door_detection.py"
        ).read_text(encoding="utf-8")
        environment_source = (
            repository_root / "env" / "habitat" / "exploration_env.py"
        ).read_text(encoding="utf-8")
        sidecar_source = (
            repository_root / "voxroom_sidecar.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("navigation_free_pred", runtime_source)
        self.assertNotIn("navigation_start_pred", runtime_source)
        self.assertNotIn("map_start_pred", runtime_source)
        self.assertNotIn("active_room_pose_from_voxroom", runtime_source)
        self.assertNotIn("navigation_free_pred", environment_source)
        self.assertNotIn("strict_voxroom_navigation", environment_source)
        self.assertNotIn('info["gt_map"] =', sidecar_source)
        self.assertNotIn('info["gt_exp"] =', sidecar_source)
        self.assertIn(
            '"navigation_planner_source": "active_room_original_fmm"',
            runtime_source,
        )

    def test_voxroom_projection_is_parallel_and_preserves_active_room_maps(self):
        free = np.zeros((4, 5), dtype=bool)
        occupied = np.zeros_like(free)
        observed = np.zeros_like(free)
        free[0, 1] = True
        occupied[1, 3] = True
        observed[0, 1] = True
        observed[1, 3] = True
        observed[2, 4] = True
        unknown = ~observed

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "navigation.npz"
            with path.open("wb") as stream:
                np.savez(
                    stream,
                    format_version=np.asarray(1, dtype=np.int32),
                    step=np.asarray(7, dtype=np.int64),
                    shape_hw=np.asarray(free.shape, dtype=np.int32),
                    free_bits=np.packbits(free.reshape(-1), bitorder="little"),
                    occupied_bits=np.packbits(
                        occupied.reshape(-1),
                        bitorder="little",
                    ),
                    observed_bits=np.packbits(
                        observed.reshape(-1),
                        bitorder="little",
                    ),
                    unknown_bits=np.packbits(
                        unknown.reshape(-1),
                        bitorder="little",
                    ),
                    resolution_m=np.asarray(0.05, dtype=np.float64),
                    bounds_xyxy_m=np.asarray(
                        [-0.125, -0.1, 0.125, 0.1],
                        dtype=np.float64,
                    ),
                    source=np.asarray(
                        "mapper.last_voxel_navigation_projection"
                    ),
                )

            navigation = load_navigation_projection(path, 7, free.shape)

        self.assertTrue(navigation["free"][3, 1])
        self.assertTrue(navigation["occupied"][2, 3])
        self.assertTrue(navigation["observed"][1, 4])
        self.assertFalse(navigation["unknown"][1, 4])

        native_obstacles = np.arange(
            np.prod(free.shape),
            dtype=np.float32,
        ).reshape(free.shape)
        native_explored = np.flipud(native_obstacles).copy()
        info = {
            "gt_map": native_obstacles.copy(),
            "gt_exp": native_explored.copy(),
            "voxroom_base_pose_world_xyzyaw": np.asarray(
                [0.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            ),
        }
        attach_voxroom_projection(info, navigation)

        self.assertEqual(info["voxroom_navigation_map_step"], 7)
        self.assertEqual(
            info["voxroom_navigation_map_source"],
            "voxroom_last_voxel_navigation_projection",
        )
        np.testing.assert_array_equal(info["gt_map"], native_obstacles)
        np.testing.assert_array_equal(info["gt_exp"], native_explored)
        self.assertFalse(info["voxroom_navigation_free"][1, 4])
        self.assertTrue(info["voxroom_navigation_observed"][1, 4])
        self.assertFalse(info["voxroom_navigation_occupied"][1, 4])
        np.testing.assert_array_equal(
            info["voxroom_navigation_agent_cell"],
            [2, 2],
        )

        reset_info = {
            "voxroom_base_pose_world_xyzyaw": np.asarray(
                [0.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            ),
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "native obstacle and explored maps are required",
        ):
            attach_voxroom_projection(reset_info, navigation)

    def test_normalized_depth_is_restored_to_meters_without_downsampling(self):
        normalized = np.asarray(
            [[[0.0], [0.25]], [[0.5], [1.0]]],
            dtype=np.float32,
        )

        depth_m = habitat_depth_to_meters(normalized, 0.0, 10.0, True)

        self.assertEqual(depth_m.shape, (2, 2))
        np.testing.assert_allclose(depth_m, [[0.0, 2.5], [5.0, 10.0]])

    def test_exact_habitat_sensor_pose_maps_to_voxroom_flu(self):
        agent_state = SimpleNamespace(
            position=np.asarray([1.0, 0.2, -3.0], dtype=np.float64),
            rotation=np.quaternion(1.0, 0.0, 0.0, 0.0),
        )
        sensor_state = SimpleNamespace(
            position=np.asarray([1.0, 1.45, -3.0], dtype=np.float64),
            rotation=np.quaternion(1.0, 0.0, 0.0, 0.0),
        )

        base_pose, camera_transform = habitat_states_to_voxroom(
            agent_state,
            sensor_state,
            agent_state.position,
        )

        np.testing.assert_allclose(base_pose[:3], [0.0, 0.0, 0.0], atol=1.0e-8)
        self.assertAlmostEqual(float(base_pose[3]), 0.0)
        np.testing.assert_allclose(
            camera_transform[:3, 3],
            [0.0, 0.0, 1.25],
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            camera_transform[:3, :3].T @ camera_transform[:3, :3],
            np.eye(3),
            atol=1.0e-8,
        )
        self.assertAlmostEqual(float(np.linalg.det(camera_transform[:3, :3])), 1.0)

    def test_world_axis_heading_is_not_zeroed_at_episode_start(self):
        rotation = quaternion.from_rotation_vector(
            np.asarray([0.0, np.deg2rad(67.0), 0.0], dtype=np.float64)
        )

        yaw = habitat_rotation_yaw_in_voxroom(rotation)

        self.assertAlmostEqual(yaw, np.deg2rad(67.0), places=7)

if __name__ == "__main__":
    unittest.main()
