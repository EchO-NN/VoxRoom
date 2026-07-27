import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import gym
import numpy as np
import torch

from arguments import get_args
from detr_door_detection.run_detr import PostProcess, _resolve_device


class FakeVectorEnvironment(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (1,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(3)
    number_of_episodes = 1

    def reset(self):
        return np.array([0], dtype=np.uint8), {"reset": True}

    def step(self, action):
        return (
            np.array([action], dtype=np.uint8),
            float(action),
            False,
            {"action": action},
        )

    def get_short_term_goal(self, inputs):
        return np.asarray(inputs["goal"], dtype=np.float32)


def make_fake_vector_environment():
    return FakeVectorEnvironment()


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_single_process_defaults_do_not_expand_gpu_workers(self):
        with mock.patch.object(sys, "argv", ["active-room-seg", "--no_cuda"]):
            args = get_args()
        self.assertFalse(args.cuda)
        self.assertEqual(args.auto_gpu_config, 0)
        self.assertEqual(args.num_processes, 1)
        self.assertEqual(args.num_mini_batch, 1)
        self.assertEqual(args.detector_device, "cuda")

    def test_explicit_cpu_detector_device(self):
        self.assertEqual(_resolve_device("cpu"), torch.device("cpu"))

    def test_cuda_detector_request_fails_when_cuda_is_unavailable(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                _resolve_device("cuda")

    def test_postprocess_preserves_batch_and_query_count(self):
        outputs = {
            "pred_logits": torch.zeros((1, 10, 3), dtype=torch.float32),
            "pred_boxes": torch.full((1, 10, 4), 0.5, dtype=torch.float32),
        }
        result = PostProcess()(outputs, torch.tensor([[256, 320]]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["scores"].shape, (10,))
        self.assertEqual(result[0]["boxes"].shape, (10, 4))

    def test_habitat_vector_adapter_imports(self):
        from env.habitat.vector_env import VectorEnv
        from habitat.core.vector_env import VectorEnv as HabitatVectorEnv

        self.assertTrue(issubclass(VectorEnv, HabitatVectorEnv))

    def test_modern_habitat_custom_action_registration(self):
        from env.habitat.utils.noisy_actions import CustomActionSpaceConfiguration
        from habitat.sims.habitat_simulator.actions import HabitatSimActions

        action_names = ("NOISY_FORWARD", "NOISY_RIGHT", "NOISY_LEFT")
        for name in action_names:
            HabitatSimActions.extend_action_space(name)
        actions = CustomActionSpaceConfiguration(
            SimpleNamespace(FORWARD_STEP_SIZE=0.25, TURN_ANGLE=30)
        ).get()

        self.assertEqual(len(actions), 7)
        self.assertEqual(
            [actions[getattr(HabitatSimActions, name)].name for name in action_names],
            ["noisy_forward", "noisy_right", "noisy_left"],
        )

    def test_habitat_test_task_has_spl_dependencies(self):
        from habitat.config.default import get_config

        config = get_config(
            config_paths=[
                "env/habitat/habitat_api/configs/tasks/pointnav_habitat_test.yaml"
            ]
        )
        self.assertEqual(
            config.TASK.MEASUREMENTS,
            ["DISTANCE_TO_GOAL", "SUCCESS", "SPL"],
        )

    def test_scene_id_uses_current_episode_contract(self):
        from env.habitat.exploration_env import Exploration_Env

        environment = Exploration_Env.__new__(Exploration_Env)
        environment._env = SimpleNamespace(
            current_episode=SimpleNamespace(scene_id="test-scene.glb")
        )
        self.assertEqual(environment._current_scene_id(), "test-scene.glb")

    def test_habitat_vector_adapter_preserves_original_contract(self):
        from env.habitat.vector_env import VectorEnv

        environments = VectorEnv(
            make_env_fn=make_fake_vector_environment,
            env_fn_args=(tuple(),),
            auto_reset_done=False,
        )
        try:
            self.assertEqual(environments.observation_space, FakeVectorEnvironment.observation_space)
            self.assertEqual(environments.action_space, FakeVectorEnvironment.action_space)

            observations, infos = environments.reset()
            self.assertEqual(observations.tolist(), [[0]])
            self.assertTrue(infos[0]["reset"])

            observations, rewards, dones, infos = environments.step(np.asarray([2]))
            self.assertEqual(observations.tolist(), [[2]])
            self.assertEqual(rewards.tolist(), [2.0])
            self.assertEqual(dones.tolist(), [False])
            self.assertEqual(infos[0]["action"], 2)

            with self.assertRaises(TypeError):
                environments.step(torch.tensor([2]))

            goals = environments.get_short_term_goal([{"goal": [3, 4]}])
            self.assertEqual(goals.tolist(), [[3.0, 4.0]])
        finally:
            environments.close()

    def test_pytorch_wrapper_and_habitat_adapter_contract(self):
        from env import VecPyTorch
        from env.habitat.vector_env import VectorEnv

        adapter = VectorEnv(
            make_env_fn=make_fake_vector_environment,
            env_fn_args=(tuple(),),
            auto_reset_done=False,
        )
        environments = VecPyTorch(adapter, torch.device("cpu"))
        try:
            observations, infos = environments.reset()
            self.assertTrue(torch.is_tensor(observations))
            self.assertEqual(observations.tolist(), [[0.0]])
            self.assertTrue(infos[0]["reset"])

            observations, rewards, dones, infos = environments.step(torch.tensor([2]))
            self.assertTrue(torch.is_tensor(observations))
            self.assertTrue(torch.is_tensor(rewards))
            self.assertEqual(observations.tolist(), [[2.0]])
            self.assertEqual(rewards.tolist(), [2.0])
            self.assertEqual(dones.tolist(), [False])

            goals = environments.get_short_term_goal([{"goal": [3, 4]}])
            self.assertTrue(torch.is_tensor(goals))
            self.assertEqual(goals.tolist(), [[3.0, 4.0]])
        finally:
            environments.close()


if __name__ == "__main__":
    unittest.main()
