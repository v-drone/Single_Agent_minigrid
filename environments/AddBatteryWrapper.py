from minigrid.wrappers import Wrapper
import numpy as np
import cv2

# Text settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.28
font_color = (0, 0, 0)  # Black color
line_type = 1


class AddBatteryWrapper(Wrapper):
    def __init__(self, env):
        """Constructor for the imgervation wrapper."""
        super().__init__(env)

    def render(self):
        img = super().render()
        print(img.shape)
        ENERGY_BAR_COLOR_FULL = np.array([0, 255, 0])
        ENERGY_BAR_COLOR_EMPTY = np.array([255, 0, 0])
        ENERGY_BAR_HEIGHT = self.tile_size

        cut_off = int(ENERGY_BAR_HEIGHT / 2)
        cut_off = (cut_off, ENERGY_BAR_HEIGHT - cut_off)

        energy_fraction = self.battery / self.full_battery
        energy_bar_width = int(energy_fraction * img.shape[1])

        energy_bar_img = np.ones((ENERGY_BAR_HEIGHT, img.shape[1], 3), dtype=np.uint8) * 255

        energy_bar_color = (ENERGY_BAR_COLOR_FULL * energy_fraction +
                            ENERGY_BAR_COLOR_EMPTY * (1 - energy_fraction)).astype(np.uint8)
        energy_bar_img[:, :energy_bar_width] = energy_bar_color

        text = '{:.2f}, {:.2f}'.format(self.render_reward[0], self.render_reward[1])
        text_size = cv2.getTextSize(text, font, font_scale, line_type)[0]
        text_x = (energy_bar_img.shape[1] - text_size[0]) // 2
        text_y = (energy_bar_img.shape[0] + text_size[1]) // 2
        cv2.putText(energy_bar_img, text, (text_x, text_y), font, font_scale, font_color, line_type)

        img_ex = np.concatenate((energy_bar_img, img[cut_off[0]:-cut_off[1], :, :]), axis=0)
        return img_ex
