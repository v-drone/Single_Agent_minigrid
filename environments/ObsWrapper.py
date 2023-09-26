from gymnasium import spaces
from minigrid.core.mission import MissionSpace
from minigrid.wrappers import RGBImgPartialObsWrapper
import numpy as np


class FullRGBImgPartialObsWrapper(RGBImgPartialObsWrapper):
    def __init__(self, env, tile_size=8, agent_pov=False):
        super().__init__(env, tile_size)
        self.agent_pov = agent_pov
        if self.agent_pov:
            image_observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.width, self.height, self.agent_view_size, 3),
                dtype="uint8",
            )
            self.observation_space["image"] = image_observation_space

    def observation(self, obs):
        img = self.get_frame(tile_size=self.tile_size, agent_pov=self.agent_pov)
        # Constants for the energy bar
        ENERGY_BAR_HEIGHT = self.tile_size  # Height of the energy bar in pixels
        ENERGY_BAR_COLOR_FULL = np.array([0, 255, 0])  # Green color
        ENERGY_BAR_COLOR_EMPTY = np.array([255, 0, 0])  # Red color
        cut_off = int(self.tile_size / 2)
        cut_off = (cut_off, self.tile_size - cut_off)
        # Calculate the width of the energy bar based on the remaining steps
        energy_fraction = self.battery / self.full_battery
        energy_bar_width = int(energy_fraction * img.shape[1])

        # Create an empty image for the energy bar
        energy_bar_img = np.ones((ENERGY_BAR_HEIGHT, img.shape[1], 3), dtype=np.uint8) * 255

        # Calculate the color of the energy bar based on the remaining steps
        energy_bar_color = (
                ENERGY_BAR_COLOR_FULL * energy_fraction + ENERGY_BAR_COLOR_EMPTY * (1 - energy_fraction)
        ).astype(np.uint8)

        # Draw the energy bar
        energy_bar_img[:, :energy_bar_width] = energy_bar_color

        # Concatenate the energy bar with the original image
        return {**obs, "image": np.concatenate((energy_bar_img, img[cut_off[0]:-cut_off[1], :, :]), axis=0)}
