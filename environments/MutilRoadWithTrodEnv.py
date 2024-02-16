from __future__ import annotations
from environments.CustomGrid import COLOR_TO_IDX, CHECKED
from environments.MutilRoadEnv import RouteEnv, get_color
from minigrid.core.world_object import Floor
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_line
from typing import Any
import numpy as np
import random


def point_on_line(x0, y0, x1, y1, x, y, r=0.01):
    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)
    dir_l = p1 - p0
    dist = np.linalg.norm(dir_l)
    dir_l = dir_l / dist

    q = np.array([x, y])
    pq = q - p0

    a = np.dot(pq, dir_l)
    a = np.clip(a, 0, dist)
    p = p0 + a * dir_l

    dist_to_line = np.linalg.norm(q - p)
    return dist_to_line <= r


class TrodTile(Floor):
    """Custom world object to represent the path tiles."""

    def __init__(self, color='purple', color_buffer=0):
        super().__init__(color)
        self.color_buffer = color_buffer
        self._color = get_color(self.color, self.color_buffer)
        self.cx = []
        self.cy = []
        for _ in range(self.color_buffer):
            cx = random.uniform(0.1, 0.9)
            cy = random.uniform(0.1, 0.9)
            self.cx.append(cx)
            self.cy.append(cy)

    def update_color(self):
        """Change color when agent steps on it."""
        self.color = CHECKED
        self._color = get_color(self.color, 0)

    def render(self, img):
        r = 0.02
        # Convert color to RGB and apply random variation
        # fill_coords(img, point_in_rect(0, 1, 0, 1), color)
        fill_coords(img, point_in_rect(0, 1, 0, 1), self._color)

        if self.color_buffer // 2 == 0:

            horizontal_lines = [(0.1, 0.33, 0.9, 0.33), (0.1, 0.66, 0.9, 0.66)]
            vertical_lines = [(0.33, 0.1, 0.33, 0.9), (0.66, 0.1, 0.66, 0.9)]

            for x0, y0, x1, y1 in horizontal_lines:
                fill_coords(img, lambda x, y: point_on_line(x0, y0, x1, y1, x, y, r), (0, 0, 0))

            for x0, y0, x1, y1 in vertical_lines:
                fill_coords(img, lambda x, y: point_on_line(x0, y0, x1, y1, x, y, r), (0, 0, 0))

        else:
            # Little waves
            for i in range(3):
                ylo = 0.3 + 0.2 * i
                yhi = 0.4 + 0.2 * i
                fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
                fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
                fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
                fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))

    def encode(self):
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color] * 10 + self.color_buffer, 0


# Update the RouteEnv class to use the new RoutePoint object
class RouteWithTrodEnv(RouteEnv):

    def __init__(self, size=20, max_steps=100, agent_view_size=3,
                 routes=(3, 5), trods=(3, 5),
                 battery=100, render_mode="human", **kwargs):
        super().__init__(size=size, max_steps=max_steps, routes=routes,
                         agent_view_size=agent_view_size,
                         battery=battery, render_mode=render_mode, **kwargs)
        self.trods = trods
        self.unvisited_trods = set()
        self.visited_trods = set()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, _ = super().reset()
        self.visited_trods = set()
        return obs, {}

    def _gen_grid(self, width, height):
        # Call the original _gen_grid method to generate the base grid
        super()._gen_grid(width, height)
        # Randomly decide the number of routes
        num_trods = random.randint(*self.trods)
        all_trod_cells = []

        # Determine the min and max trod length based on grid size
        min_trod_length = 5
        max_trod_length = min(width, height) - 2

        for _ in range(num_trods):
            # Randomly decide the length of the trod based on min and max values
            trod_length = random.randint(min_trod_length, max_trod_length)

            # Random starting point for the trod, ensuring it's not the agent's start/goal position
            while True:
                trod_start_x, trod_start_y = random.randint(1, width - 2), random.randint(1, height - 2)
                if (trod_start_x, trod_start_y) != self.start_pos:
                    break

            # List to store the trod's cells
            trod_cells = [(trod_start_x, trod_start_y)]

            # Choose a random direction for this trod
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])

            # Generate the trod
            for _ in range(trod_length - 1):
                next_x, next_y = trod_start_x + dx, trod_start_y + dy

                # Check if the next cell is within grid, is empty, and is not the agent's start/goal position
                if ((1 <= next_x < width - 1)
                        and (1 <= next_y < height - 1)
                        and self.grid.get(next_x, next_y) is None
                        and (next_x, next_y) != self.start_pos):
                    # Add the cell to the trod and update the start cell
                    trod_cells.append((next_x, next_y))
                    trod_start_x, trod_start_y = next_x, next_y
                else:
                    # Stop generating the trod if we hit a wall, another trod, or the agent's start/goal position
                    break
            # Mark the trod cells in the grid
            for x, y in trod_cells:
                if self.grid.get(x, y) is None:
                    _ = TrodTile(color_buffer=self._rand_int(0, 5))

                    self.grid.set(x, y, _)
                    all_trod_cells.append((x, y))
        self.unvisited_trods = set(all_trod_cells)

    def step(self, action):
        # Record the agent's current position before executing the action
        self.prev_pos = np.copy(self.agent_pos)

        # Execute the agent's action
        obs, reward, terminated, truncated, info = super().step(action)
        # Check if agent stepped on a path tile and update its color
        # Ensure the agent has actually moved
        if not np.equal(self.agent_pos, self.prev_pos).all():
            cell = self.grid.get(*self.agent_pos)
            if isinstance(cell, TrodTile) and cell.color != CHECKED:
                cell.update_color()
                self.unvisited_trods.remove(self.agent_pos)
                self.visited_trods.add(self.agent_pos)
        reward = self._reward()
        return obs, reward, terminated, truncated, info

    def reward_breakdown(self):
        return super().reward_breakdown(), self._reward()

    def _reward(self) -> float:
        reward = super()._reward()
        if not self.unvisited_tiles and self.agent_pos == self.start_pos:
            # Provide a positive reward for completing the task
            return reward + 0.05 * len(self.visited_trods)
        else:
            return reward
