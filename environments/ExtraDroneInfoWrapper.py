import numpy as np
from gymnasium import spaces
from functools import reduce
from minigrid.wrappers import ObservationWrapper


class ExtraDroneInfoWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=([reduce(lambda x, y: x * y, self.observation_space.shape)
                    + 2]),
            dtype="uint8",
        )

    def observation(self, obs):
        extra_info = np.concatenate([np.array([self.info["yaw_degrees"]]).astype(np.uint8),
                                     np.array([self.battery]).astype(np.uint8)],
                                    axis=-1)
        return np.concatenate([obs, extra_info], axis=-1, dtype=np.uint8)
