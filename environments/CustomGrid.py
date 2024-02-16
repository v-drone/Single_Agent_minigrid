from minigrid.core.grid import Grid as OriginalGrid
import numpy as np
from typing import Any

import math
from minigrid.core.constants import TILE_PIXELS

from minigrid.utils.rendering import downsample, fill_coords, highlight_img, point_in_triangle, rotate_fn

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "white": np.array([255, 255, 255])
}

COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5, "white": 9}

CHECKED = 'yellow'


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

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        img = downsample(img, subdivs)

        # Cache the rendered tile
        self.tile_cache[key] = img

        return img

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
