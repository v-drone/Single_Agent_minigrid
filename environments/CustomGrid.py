from minigrid.core.grid import Grid as OriginalGrid
from typing import Any
from minigrid.core.constants import TILE_PIXELS, COLORS, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.world_object import Lava, Floor
from minigrid.utils.rendering import downsample, fill_coords, point_in_rect, point_in_triangle, rotate_fn
import numpy as np
import colorsys
import math


def get_color(color, color_buffer):
    color = COLORS[color] / 2
    hsv_color = colorsys.rgb_to_hsv(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    buffer = color_buffer * 0.02
    new_h = np.clip(hsv_color[0] + buffer, 0, 1)
    new_s = np.clip(hsv_color[1] + buffer, 0, 1)
    new_v = np.clip(hsv_color[2] + buffer, 0, 1)

    new_rgb = colorsys.hsv_to_rgb(new_h, new_s, new_v)
    return [int(new_rgb[0] * 255), int(new_rgb[1] * 255), int(new_rgb[2] * 255)]


class BaseTile(Floor):
    def __init__(self, color="purple", color_buffer=0, label=1.0):
        super().__init__(color=color)
        self.color_buffer = color_buffer
        self.label = label

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1),
                    get_color(self.color, self.color_buffer))

    def encode(self):
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color] * 10 + self.color_buffer, 0


class RoadTile(BaseTile):
    """Custom world object to represent the path tiles."""

    def __init__(self, color_buffer=0, label=0.0):
        super().__init__("blue", color_buffer, label)

    def update_color(self):
        """Change color when agent steps on it."""
        self.color = "yellow"


class BuildTile(BaseTile):
    """Custom world object to represent the path tiles."""

    def __init__(self, color_buffer=0, label=0.0):
        super().__init__("grey", color_buffer, label)


class DamageTile(BaseTile):
    """Custom world object to represent the path tiles."""

    def __init__(self, color_buffer=0, label=0.0):
        super().__init__("blue", color_buffer=color_buffer, label=label)

    def update_color(self):
        """Change color when agent steps on it."""
        self.color = "purplew"
        self.label = 1


class StartPoint(Floor):
    """Custom world object to represent the path tiles."""

    def __init__(self):
        super().__init__(color="green")


class WallFail(Lava):
    """Custom world object to represent the path tiles."""

    def __init__(self, color: str = "grey"):
        super().__init__()
        self.color = color

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Grid(OriginalGrid):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)

    def render_tile(self, obj=None, agent_dir=None, highlight=False, tile_size=TILE_PIXELS, subdivs=3) -> np.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in self.tile_cache:
            return self.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        # fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        # fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # # Highlight the cell if needed
        # if highlight:
        #     highlight_img(img)

        img = downsample(img, subdivs)

        # Cache the rendered tile
        self.tile_cache[key] = img

        return img

    def wall_rect(self, x: int, y: int, w: int, h: int):
        self.horz_wall(x, y, w, WallFail)
        self.horz_wall(x, y + h - 1, w, WallFail)
        self.vert_wall(x, y, h, WallFail)
        self.vert_wall(x + w - 1, y, h, WallFail)

    def render(self, tile_size: int, agent_pos: tuple[int, int], agent_dir=None, highlight_mask=None):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels

        Args:
            agent_pos:
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                assert highlight_mask is not None
                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )
                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img
