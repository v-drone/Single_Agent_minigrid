from gymnasium.wrappers import FrameStack
import gymnasium as gym
import numpy as np


class StackWrapper(FrameStack):
    def __init__(
            self,
            env: gym.Env,
            num_stack: int,
            lz4_compress: bool = False,
    ):
        # Rendering attributes for observations
        super().__init__(env, num_stack=num_stack, lz4_compress=lz4_compress)

    def observation(self, observation):
        lazy_frame = super().observation(observation)
        return np.array(lazy_frame)
