import numpy as np
from habitat.core.vector_env import VectorEnv as HabitatVectorEnv


class VectorEnv(HabitatVectorEnv):
    """Preserve the data contract expected by the original runner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_spaces[0]
        self.action_space = self.action_spaces[0]

    def reset(self):
        results = super().reset()
        observations, infos = zip(*results)
        return np.stack(observations), infos

    def step(self, actions):
        if not isinstance(actions, np.ndarray):
            raise TypeError("VectorEnv.step expects the NumPy actions emitted by VecPyTorch")
        actions = actions.reshape(-1).tolist()
        normalized_actions = [{"action": int(action)} for action in actions]
        results = super().step(normalized_actions)
        observations, rewards, dones, infos = zip(*results)
        return (
            np.stack(observations),
            np.stack(rewards),
            np.stack(dones),
            infos,
        )

    def get_short_term_goal(self, inputs):
        results = self.call(
            ["get_short_term_goal"] * len(inputs),
            [{"inputs": item} for item in inputs],
        )
        return np.stack(results)
