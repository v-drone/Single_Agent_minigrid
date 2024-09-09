import numpy as np
from gymnasium import spaces
from scipy.ndimage import zoom
from minigrid.wrappers import ObservationWrapper


class AddMapWrapper(ObservationWrapper):
    def __init__(self, env, zoom_size=3):
        super().__init__(env)
        self.zoom_size = zoom_size
        self.camera_shape = self.observation_space["image"]
        map_shape = spaces.Box(
            low=0,
            high=255,
            shape=np.array([self.size * self.zoom_size, self.size * self.zoom_size, 4]),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                **self.observation_space,
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=np.array([np.prod(self.camera_shape.shape) + np.prod(map_shape.shape), 1]),
                    dtype="uint8"
                )
            }
        )

    def observation(self, obs):
        map_obs = self.get_frame(tile_size=self.tile_size)
        walked = zoom(self.walked, (self.zoom_size, self.zoom_size), order=0)
        walked = np.expand_dims(walked, -1)
        map_obs = np.concatenate([map_obs, walked], axis=-1, dtype=np.uint8)
        next_obs = np.concatenate([obs["image"].flatten(),
                                   map_obs.flatten()], dtype=np.uint8)
        return {**obs, "image": next_obs}
