from __future__ import annotations

from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import Floor, Goal
from minigrid.core.actions import IntEnum
from gymnasium.envs.registration import EnvSpec
from gymnasium import spaces
from typing import Any
import numpy as np
import random


class PathTile(Floor):
    """Custom world object to represent the path tiles."""

    def __init__(self, color='blue'):
        super().__init__(color)

    def toggle(self, agent, pos):
        """Change color when agent steps on it."""
        self.color = 'purple'


# Update the RouteEnv class to use the new RoutePoint object
class RouteEnv(EmptyEnv):
    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        forward = 0
        left = 1
        right = 2

    def __init__(self, size=20, max_steps=100, roads=(3, 5), battery=100, render_mode="human", agent_pov=True):

        super().__init__(size=size, max_steps=max_steps, render_mode=render_mode)
        self.spec = EnvSpec("RouteEnv-v0", max_episode_steps=self.max_steps)
        self.screen_size = 300
        self.roads = roads
        # To track tiles that are not yet visited by the agent
        self.unvisited_tiles = set()
        self.visited_tiles = set()
        self.actions = self.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.full_battery = battery
        self.battery = battery
        self.prev_pos = None
        self.prev_distance = 0
        self.current_distance = 0
        self.render_reward = [0, 0]
        self.agent_pov = False
        self.agent_pov = agent_pov
        if not agent_pov:
            self.observation_space["image"] = spaces.Box(
                low=0,
                high=255,
                shape=(self.width, self.height, 3),
                dtype=np.uint8
            )
        else:
            self.observation_space["image"] = spaces.Box(
                low=0,
                high=255,
                shape=(7, 7, 3),
                dtype=np.uint8
            )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, _ = super().reset()
        self.render_reward = [0, 0]
        self.battery = self.full_battery
        self.prev_distance = self.distance_to_closest_blue(self.agent_pos)
        self.current_distance = self.distance_to_closest_blue(self.agent_pos)
        self.visited_tiles = set()
        return obs, {}

    def _gen_grid(self, width, height):
        # Call the original _gen_grid method to generate the base grid
        super()._gen_grid(width, height)

        # Clear any existing goals from the grid
        for i in range(width):
            for j in range(height):
                if isinstance(self.grid.get(i, j), Goal):
                    self.grid.set(i, j, None)

        # Random starting/goal point for the agent
        start_x, start_y = random.randint(1, width - 2), random.randint(1, height - 2)
        self.start_pos = (start_x, start_y)
        self.grid.set(start_x, start_y, Goal())

        # Randomly decide the number of routes (3 to 5)
        num_routes = random.randint(*self.roads)
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
            for x, y in route_cells:
                self.grid.set(x, y, PathTile())

            all_route_cells.extend(route_cells)

        # Remember the initial position as the final goal
        self.agent_pos = self.start_pos
        self.all_route_cells = all_route_cells

        # Initialize the set of unvisited tiles
        self.unvisited_tiles = set(self.all_route_cells)
        self.prev_pos = None

        self.mission = self._gen_mission()

    @staticmethod
    def _gen_mission():
        return ""

    def step(self, action):
        # Record the agent's current position before executing the action
        self.prev_pos = self.agent_pos
        if self.prev_pos is not None:
            self.prev_distance = self.distance_to_closest_blue(self.prev_pos)

        # Execute the agent's action
        obs, reward, terminated, truncated, info = super().step(action)

        self.current_distance = self.distance_to_closest_blue(self.agent_pos)

        terminated = False
        if self.agent_pos == self.start_pos:
            self.battery = self.full_battery
        else:
            self.battery -= 1

        if self.battery <= 0:
            truncated = True

        reward = self._reward()
        self.render_reward[0] = reward
        self.render_reward[1] += reward
        # Check if agent stepped on a path tile and update its color
        # Ensure the agent has actually moved
        if self.agent_pos != self.prev_pos:
            cell = self.grid.get(*self.agent_pos)
            if isinstance(cell, PathTile) and cell.color != 'purple':
                cell.toggle(None, self.agent_pos)
                self.unvisited_tiles.remove(self.agent_pos)
                self.visited_tiles.add(self.agent_pos)

        # Check the game ending conditions
        # terminated, truncated
        if not self.unvisited_tiles and self.agent_pos == self.start_pos:
            terminated = True

        return obs, reward, terminated, truncated, info

    def get_frame(
            self,
            highlight: bool = True,
            tile_size: int = 3,
            agent_pov: bool = False,
    ):
        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def distance_to_closest_blue(self, pos):
        # Calculate the Manhattan distance to the closest blue tile
        return min(abs(pos[0] - x) + abs(pos[1] - y) for (x, y) in self.unvisited_tiles)

    def _reward(self) -> float:
        # Basic small negative reward for each step
        reward = -0.01

        # Calculate the change in distance to the closest blue tile
        if self.current_distance < self.prev_distance:
            reward += 0.05

        # Check if agent stepped on a path tile and update its color
        # Ensure the agent has actually moved
        if self.agent_pos != self.prev_pos:
            cell = self.grid.get(*self.agent_pos)
            if isinstance(cell, PathTile) and cell.color != 'purple':
                # Reward for visiting a route tile
                reward += 0.1
        else:
            # If agent didn't move and tried a forward action, then it likely hit a wall or obstacle
            reward -= 0.05

        # Check the game ending conditions
        # terminated, truncated
        if not self.unvisited_tiles and self.agent_pos == self.start_pos:
            # Provide a positive reward for completing the task
            reward += 10

        return reward
