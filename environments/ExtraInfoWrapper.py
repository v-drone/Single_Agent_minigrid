import cv2
import numpy as np
from gymnasium import spaces
from functools import reduce
from minigrid.wrappers import ObservationWrapper


class ExtraInfoWrapper(ObservationWrapper):
    def __init__(self, env, walked_size=64):
        super().__init__(env)
        self.walked_size = walked_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=([reduce(lambda x, y: x * y, self.observation_space.shape)
                    + self.walked_size * self.walked_size
                    + 1]),
            dtype="uint8",
        )

    def observation(self, obs):
        img = obs.copy().flatten()
        walked = self.walked.copy()
        resized_walked = cv2.resize(walked, [self.walked_size, self.walked_size], interpolation=cv2.INTER_LINEAR)

        return np.concatenate([img, resized_walked.flatten(), np.array([self.battery], dtype=np.uint8)],
                              dtype=self.observation_space.dtype)
