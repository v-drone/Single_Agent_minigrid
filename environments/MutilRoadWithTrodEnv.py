from __future__ import annotations
from environments.MutilRoadEnv import RouteEnv, PathTile
from minigrid.core.world_object import Floor
from typing import Any
import numpy as np
import random


class TrodTile(Floor):
    """Custom world object to represent the path tiles."""

    def __init__(self, color='yellow'):
        super().__init__(color)
        self.view = False

    def purple(self):
        """Change color when agent steps on it."""
        self.color = 'purple'


# Update the RouteEnv class to use the new RoutePoint object
class RouteWithTrodEnv(RouteEnv):

    def __init__(self, size=20, max_steps=100, routes=(3, 5), trods=(3, 5),
                 battery=100, render_mode="human", **kwargs):
        super().__init__(size=size, max_steps=max_steps, routes=routes,
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
                    self.grid.set(x, y, TrodTile())
                    all_trod_cells.append((x, y))
        self.unvisited_trods = set(all_trod_cells)

    def step(self, action):
        # Record the agent's current position before executing the action
        self.prev_pos = np.copy(self.agent_pos)

        # Execute the agent's action
        obs, reward, terminated, truncated, info = super().step(action)
        reward = self._reward()
        self.render_reward[0] = reward
        self.render_reward[1] += reward
        # Check if agent stepped on a path tile and update its color
        # Ensure the agent has actually moved
        if not np.equal(self.agent_pos, self.prev_pos).all():
            cell = self.grid.get(*self.agent_pos)
            if isinstance(cell, TrodTile) and cell.color != 'purple':
                cell.purple()
                self.unvisited_trods.remove(self.agent_pos)
                self.visited_trods.add(self.agent_pos)
        return obs, reward, terminated, truncated, info

    def _reward(self) -> float:
        reward = super()._reward()
        if not np.equal(self.agent_pos, self.prev_pos).all():
            cell = self.grid.get(*self.agent_pos)
            if isinstance(cell, TrodTile) and cell.color != 'purple':
                # Reward for visiting a route tile
                reward += 1
        return reward