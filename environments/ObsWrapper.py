from minigrid.wrappers import RGBImgPartialObsWrapper
from gymnasium import spaces
import numpy as np
import cv2

# Text settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.2
font_color = (0, 0, 0)  # Black color
line_type = 1


class FullRGBImgPartialObsWrapper(RGBImgPartialObsWrapper):
    def __init__(self, env, tile_size=8, img_size=150):
        # Rendering attributes for observations
        super().__init__(env, tile_size)
        self.tile_size = tile_size
        self.agent_pov = env.agent_pov
        self.img_size = img_size
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=(img_size, img_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": image_space}
        )

    def observation(self, obs):
        img = self.get_frame(tile_size=self.tile_size, agent_pov=self.agent_pov)
        # Constants for the energy bar
        # Height of the energy bar in pixels
        ENERGY_BAR_COLOR_FULL = np.array([0, 255, 0])  # Green color
        ENERGY_BAR_COLOR_EMPTY = np.array([255, 0, 0])  # Red color
        if self.agent_pov:
            ENERGY_BAR_HEIGHT = int(0.2 * self.tile_size)
        else:
            ENERGY_BAR_HEIGHT = self.tile_size
        cut_off = int(ENERGY_BAR_HEIGHT / 2)
        cut_off = (cut_off, ENERGY_BAR_HEIGHT - cut_off)
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

        # Calculate text size
        text = str(self.render_reward)
        text_size = cv2.getTextSize(text, font, font_scale, line_type)[0]
        text_x = (energy_bar_img.shape[1] - text_size[0]) // 2
        text_y = (energy_bar_img.shape[0] + text_size[1]) // 2
        cv2.putText(energy_bar_img, text, (text_x, text_y), font, font_scale, font_color, line_type)

        img = np.concatenate((energy_bar_img, img[cut_off[0]:-cut_off[1], :, :]), axis=0)

        # Create a blank image with the new frame size
        padded_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8) + 255

        # Calculate the position to place the original image
        y_center = (self.img_size - img.shape[0]) // 2
        x_center = (self.img_size - img.shape[1]) // 2

        # Place the original image in the center of the new frame
        padded_img[y_center:y_center + img.shape[0], x_center:x_center + img.shape[1]] = img
        # Concatenate the energy bar with the original image
        return {**obs, "image": padded_img}
