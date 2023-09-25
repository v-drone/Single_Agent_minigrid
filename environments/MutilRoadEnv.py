from typing import Any
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import Floor, Goal
from minigrid.core.actions import IntEnum
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.spaces.box import Box
import numpy as np
import pygame
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

    def __init__(self, size=20, max_steps=100, roads=(3, 5), battery=100, render_mode="human"):
        super().__init__(size=size, max_steps=max_steps, render_mode=render_mode)
        self.tile_size = 8
        self.screen_size = 300
        self.roads = roads
        # To track tiles that are not yet visited by the agent
        self.unvisited_tiles = set()
        self.actions = self.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = Box(0, 255, (size * self.tile_size, size * self.tile_size, 3), np.uint8)
        self.full_battery = battery
        self.battery = battery
        self.prev_pos = None

    def reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset()
        return self._gen_obs(), {}

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
                if (1 <= next_x < width - 1) and (1 <= next_y < height - 1) and self.grid.get(next_x, next_y) is None and (next_x, next_y) != (start_x, start_y):
                    # Add the cell to the route and update the start cell
                    route_cells.append((next_x, next_y))
                    route_start_x, route_start_y = next_x, next_y
                else:
                    break  # Stop generating the route if we hit a wall, another route, or the agent's start/goal position

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
        return "get to the green goal after get to all blue square"

    def step(self, action):
        # Record the agent's current position before executing the action
        self.prev_pos = self.agent_pos

        # Execute the agent's action
        obs, reward, terminated, truncated, info = super().step(action)
        terminated = False
        if self.agent_pos == self.start_pos:
            self.battery = self.full_battery
        else:
            self.battery -= 1

        if self.battery <= 0:
            truncated = True

        reward = self._reward()
        # Check if agent stepped on a path tile and update its color
        if self.agent_pos != self.prev_pos:  # Ensure the agent has actually moved
            cell = self.grid.get(*self.agent_pos)
            if isinstance(cell, PathTile) and cell.color != 'purple':
                cell.toggle(None, self.agent_pos)
                self.unvisited_tiles.remove(self.agent_pos)

        # Check the game ending conditions
        # terminated, truncated
        if not self.unvisited_tiles and self.agent_pos == self.start_pos:
            terminated = True

        return self._gen_obs(), reward, terminated, truncated, info

    def _gen_obs(self):
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

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
        return  np.concatenate((energy_bar_img, img[cut_off[0]:-cut_off[1], :, :]), axis=0)


    def render(self):
        img = self._gen_obs()

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 10
            text = self.mission
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def _reward(self) -> float:
        # Basic small negative reward for each step
        reward = -0.01

        # Check if agent stepped on a path tile and update its color
        if self.agent_pos != self.prev_pos:  # Ensure the agent has actually moved
            cell = self.grid.get(*self.agent_pos)
            if isinstance(cell, PathTile) and cell.color != 'purple':
                reward += 0.1  # Reward for visiting a route tile
        else:
            # If agent didn't move and tried a forward action, then it likely hit a wall or obstacle
            reward -= 0.1  # Penalty for collision

        # Check the game ending conditions
        # terminated, truncated
        if not self.unvisited_tiles and self.agent_pos == self.start_pos:
            reward += 1  # Provide a positive reward for completing the task

        return reward
