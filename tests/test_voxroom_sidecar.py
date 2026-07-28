import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import quaternion

from env.habitat.exploration_env import (
    habitat_depth_to_meters,
    habitat_states_to_voxroom,
)
from voxroom_sidecar import apply_navigation_projection, load_navigation_projection


class VoxRoomGeometryTests(unittest.TestCase):
    def test_navigation_projection_preserves_outside_and_flips_rows(self):
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

        info = {
            "gt_map": np.full(free.shape, 9.0, dtype=np.float32),
            "gt_exp": np.full(free.shape, 8.0, dtype=np.float32),
        }
        apply_navigation_projection(info, navigation)

        self.assertEqual(info["navigation_map_step"], 7)
        self.assertEqual(
            info["navigation_map_source"],
            "voxroom_last_voxel_navigation_projection",
        )
        self.assertTrue(info["gt_map"][1, 4])
        self.assertTrue(info["gt_exp"][1, 4])
        self.assertFalse(info["voxroom_navigation_free"][1, 4])

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


if __name__ == "__main__":
    unittest.main()
