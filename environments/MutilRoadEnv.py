from __future__ import annotations
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import Floor, Goal
from minigrid.core.actions import IntEnum
from gymnasium.envs.registration import EnvSpec
from gymnasium import spaces
from typing import Any
from environments.CustomGrid import Grid, COLOR_TO_IDX, COLORS, CHECKED
import numpy as np
import random
import colorsys
from minigrid.utils.rendering import fill_coords, point_in_rect
from minigrid.core.constants import OBJECT_TO_IDX


def get_color(color, color_buffer):
    color = COLORS[color] / 2
    hsv_color = colorsys.rgb_to_hsv(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    buffer = color_buffer * 0.02
    new_h = np.clip(hsv_color[0] + buffer, 0, 1)
    new_s = np.clip(hsv_color[1] + buffer, 0, 1)
    new_v = np.clip(hsv_color[2] + buffer, 0, 1)

    new_rgb = colorsys.hsv_to_rgb(new_h, new_s, new_v)
    return [int(new_rgb[0] * 255), int(new_rgb[1] * 255), int(new_rgb[2] * 255)]


class PathTile(Floor):
    """Custom world object to represent the path tiles."""

    def __init__(self, color='blue', color_buffer=0, label=1.0):
        super().__init__(color)
        self.color_buffer = color_buffer
        self._color = get_color(self.color, self.color_buffer)
        self.label = label

    def update_color(self):
        """Change color when agent steps on it."""
        self.color = CHECKED
        self._color = get_color(self.color, 0)

    def render(self, img):
        # Convert color to RGB and apply random variation
        # fill_coords(img, point_in_rect(0, 1, 0, 1), color)

        fill_coords(img, point_in_rect(0, 1, 0, 1), self._color)

    def encode(self):
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color] * 10 + self.color_buffer, 0


# Update the RouteEnv class to use the new RoutePoint object
class RouteEnv(EmptyEnv):
    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self, size=20, max_steps=100, routes=(3, 5), battery=100, agent_view_size=3,
                 basic_coefficient=0.1,
                 render_mode="human", **kwargs):

        super().__init__(size=size, max_steps=max_steps, agent_view_size=agent_view_size,
                         render_mode=render_mode)
        self.spec = EnvSpec("RouteEnv-v0", max_episode_steps=self.max_steps)
        self.routes = routes
        # To track tiles that are not yet visited by the agent
        self.unvisited_tiles = set()
        self.visited_tiles = set()
        self.actions = self.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.full_battery = battery
        self.battery = battery
        self.prev_pos = None
        self.basic_coefficient = basic_coefficient
        self.prev_distance = size * 2
        self.current_distance = size * 2

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, _ = super().reset()
        self.battery = self.full_battery
        self.prev_distance = self.distance_to_closest_blue(self.agent_pos)
        self.current_distance = self.distance_to_closest_blue(self.agent_pos)
        self.visited_tiles = set()
        self.prev_pos = np.copy(self.agent_pos)
        return obs, {}

    def _gen_grid(self, width, height):
        # Call the original _gen_grid method to generate the base grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Random starting/goal point for the agent
        start_x, start_y = random.randint(1, width - 2), random.randint(1, height - 2)
        self.start_pos = (start_x, start_y)
        self.agent_pos = self.start_pos
        self.agent_dir = self._rand_int(0, 4)
        self.grid.set(start_x, start_y, Goal())

        # Randomly decide the number of routes
        num_routes = random.randint(*self.routes)
        all_route_cells = []

        # Determine the min and max route length based on grid size
        min_route_length = 5
        max_route_length = min(width, height) - 2

        for _ in range(num_routes):
            # Randomly decide the length of the route based on min and max values
            route_length = random.randint(min_route_length, max_route_length)

            # Random starting point for the route, ensuring it's not the agent's start/goal position
            while True:
                route_start_x, route_start_y = random.randint(1, width - 2), random.randint(1, height - 2)
                if (route_start_x, route_start_y) != (start_x, start_y):
                    break

            # List to store the route's cells
            route_cells = [(route_start_x, route_start_y)]

            # Choose a random direction for this route
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])

            # Generate the route
            for _ in range(route_length - 1):
                next_x, next_y = route_start_x + dx, route_start_y + dy

                # Check if the next cell is within grid, is empty, and is not the agent's start/goal position
                if ((1 <= next_x < width - 1)
                        and (1 <= next_y < height - 1)
                        and self.grid.get(next_x, next_y) is None
                        and (next_x, next_y) != (start_x, start_y)):
                    # Add the cell to the route and update the start cell
                    route_cells.append((next_x, next_y))
                    route_start_x, route_start_y = next_x, next_y
                else:
                    # Stop generating the route if we hit a wall, another route, or the agent's start/goal position
                    break
            # Mark the route cells in the grid
            for o, (x, y) in enumerate(route_cells):
                _ = PathTile(color_buffer=self._rand_int(0, 4))
                self.grid.set(x, y, _)

            all_route_cells.extend(route_cells)
        # Remember the initial position as the final goal
        self.agent_pos = self.start_pos
        self.all_route_cells = all_route_cells

        # Initialize the set of unvisited tiles
        self.unvisited_tiles = set(self.all_route_cells)
        self.mission = self._gen_mission()

    @staticmethod
    def _gen_mission():
        return ""

    def step(self, action):
        # Record the agent's current position before executing the action
        self.prev_pos = np.copy(self.agent_pos)

        # Execute the agent's action
        obs, reward, terminated, truncated, info = super().step(action)
        # Update distance
        if self.unvisited_tiles:
            self.prev_distance = self.distance_to_closest_blue(self.prev_pos)
            self.current_distance = self.distance_to_closest_blue(self.agent_pos)
        else:
            self.current_distance = 0
            self.prev_distance = 0

        if self.agent_pos == self.start_pos:
            self.battery = self.full_battery
        else:
            self.battery -= 1

        if self.battery <= 0:
            truncated = True

        if not np.equal(self.agent_pos, self.prev_pos).all():
            cell = self.grid.get(*self.agent_pos)
            if isinstance(cell, PathTile) and cell.color != CHECKED:
                cell.update_color()
                self.unvisited_tiles.remove(self.agent_pos)
                self.visited_tiles.add(self.agent_pos)

        reward = self._reward()
        # Check if agent stepped on a path tile and update its color
        # Ensure the agent has actually moved

        # Check the game ending conditions
        if not self.unvisited_tiles and self.agent_pos == self.start_pos:
            terminated = True
        else:
            terminated = False

        return obs, reward, terminated, truncated, info

    def reward_breakdown(self):
        return super()._reward(), super()._reward() * self.basic_coefficient + 0.05 * len(self.visited_tiles)

    def distance_to_closest_blue(self, pos):
        # Calculate the Manhattan distance to the closest blue tile
        return min(abs(pos[0] - x) + abs(pos[1] - y) for (x, y) in self.unvisited_tiles)

    def _reward(self) -> float:
        if not self.unvisited_tiles and self.agent_pos == self.start_pos:
            # Provide a positive reward for completing the task
            return super()._reward() * self.basic_coefficient + 0.05 * len(self.visited_tiles)
        else:
            return 0
