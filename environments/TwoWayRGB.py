import cv2
import numpy as np
from gymnasium import spaces
from minigrid.wrappers import ObservationWrapper


class RGBImgObsWrapper(ObservationWrapper):
    def __init__(self, env, tile_size=12, shape=None):
        super().__init__(env)
        if shape is None:
            shape = [[100, 100], [64, 64]]
        self.shape = shape
        self.tile_size = tile_size
        new_image_space = [np.prod(self.shape[0]) * 3 + np.prod(self.shape[1]) * 3, ]
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=np.array(new_image_space),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img = self.get_frame(highlight=False, tile_size=self.tile_size)
        rgb_img = cv2.resize(
            rgb_img, self.shape[0][::-1], interpolation=cv2.INTER_AREA
        ).flatten()
        rgb_view = self.get_frame(highlight=False, tile_size=self.tile_size, agent_pov=True)
        rgb_view = cv2.resize(
            rgb_view, self.shape[1][::-1], interpolation=cv2.INTER_AREA
        ).flatten()
        return {**obs, "image": np.concatenate([rgb_img, rgb_view], axis=0)}
