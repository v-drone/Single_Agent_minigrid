import numpy as np
from gymnasium import spaces
from functools import reduce
from minigrid.wrappers import ObservationWrapper


class AddEmptyWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=([reduce(lambda x, y: x * y, self.observation_space.shape) + 1]),  # number of cells
            dtype="uint8",
        )

    def observation(self, obs):
        img = obs.copy().flatten()

        return np.concatenate([img, np.array([0])])
