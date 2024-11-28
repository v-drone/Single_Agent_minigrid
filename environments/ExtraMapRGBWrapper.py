import numpy as np
from gymnasium import spaces
from scipy.ndimage import zoom
from minigrid.wrappers import ObservationWrapper

class AddMapWrapper(ObservationWrapper):
    def __init__(self, env, render_map_size=100):
        super().__init__(env)
        self.render_map_size = render_map_size
        self.camera_shape = self.observation_space["image"].shape

        map_shape = (self.render_map_size, self.render_map_size, 4)
        map_size = np.prod(map_shape)
        camera_size = np.prod(self.camera_shape)

        self.observation_space = spaces.Dict(
            {
                **self.observation_space,
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(camera_size + map_size,),
                    dtype="uint8"
                )
            }
        )

    def observation(self, obs):
        map_obs = self.get_frame(tile_size=self.tile_size)
        H, W = map_obs.shape[:2]
        zoom_factors = (self.render_map_size / H, self.render_map_size / W, 1)
        map_obs_resized = zoom(map_obs, zoom_factors, order=1)

        walked_zoom_factors = (self.render_map_size / H, self.render_map_size / W)
        walked_resized = zoom(self.walked, walked_zoom_factors, order=0)
        walked_resized = np.expand_dims(walked_resized, -1)

        map_obs_with_walked = np.concatenate([map_obs_resized, walked_resized], axis=-1)
        map_obs_with_walked = map_obs_with_walked.astype(np.uint8)

        next_obs = np.concatenate([obs["image"].flatten(), map_obs_with_walked.flatten()], dtype=np.uint8)
        return {**obs, "image": next_obs}
