import unittest
from types import SimpleNamespace

import numpy as np
import quaternion

from env.habitat.exploration_env import (
    habitat_depth_to_meters,
    habitat_states_to_voxroom,
)


class VoxRoomGeometryTests(unittest.TestCase):
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
