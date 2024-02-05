import numpy as np
from gymnasium import spaces
from functools import reduce
from minigrid.wrappers import ObservationWrapper


class AddBatteryWrapper(ObservationWrapper):
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
        # ENERGY_BAR_COLOR_FULL = np.array([0, 255, 0])
        # ENERGY_BAR_COLOR_EMPTY = np.array([255, 0, 0])
        # ENERGY_BAR_HEIGHT = self.tile_size
        #
        # cut_off = int(ENERGY_BAR_HEIGHT / 2)
        # cut_off = (cut_off, ENERGY_BAR_HEIGHT - cut_off)
        #
        # energy_fraction = self.battery / self.full_battery
        # energy_bar_width = int(energy_fraction * img.shape[1])
        #
        # energy_bar_img = np.ones((ENERGY_BAR_HEIGHT, img.shape[1], 3), dtype=np.uint8) * 255
        #
        # energy_bar_color = (ENERGY_BAR_COLOR_FULL * energy_fraction +
        #                     ENERGY_BAR_COLOR_EMPTY * (1 - energy_fraction)).astype(np.uint8)
        # energy_bar_img[:, :energy_bar_width] = energy_bar_color
        #
        # img_ex = np.concatenate((energy_bar_img, img[cut_off[0]:-cut_off[1], :, :]), axis=0)
        return np.concatenate([img, np.array([self.battery])])
