from __future__ import annotations
from environments.AirSimException import AirSimResponseError, AirSimConnectionError
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.utils.rendering import downsample, fill_coords, point_in_rect, point_in_triangle, rotate_fn
from environments.CustomGrid import Grid, Lava, StartPoint, BuildTile, RoadTile, DamageTile
from environments.EmptyAndMap import EmptyWithMapEmpty
from minigrid.core.world_object import Goal, Floor
from minigrid.core.actions import IntEnum
from minigrid.envs import EmptyEnv
from minigrid.core.mission import MissionSpace
from gymnasium.envs.registration import EnvSpec
from gymnasium import spaces
from typing import Any
import numpy as np
import itertools
import logging
import requests
import random
import math
import json
import time

ID_TO_OBJ = {
    0: None,
    1: Lava,
    2: StartPoint,
    3: BuildTile,
    4: RoadTile,
}

OBJ_TO_ID = {
    type(None): 0,
    Lava: 1,
    StartPoint: 2,
    BuildTile: 3,
    RoadTile: 4,
}


class RoadNetworkAndMap(EmptyWithMapEmpty):
    # Enumeration of possible actions
    class Actions(IntEnum):
        forward_0 = 0
        forward_1 = 1
        forward_2 = 2
        # yaw
        yaw_45 = 3
        yaw_90 = 4
        yaw_n_45 = 5
        yaw_n_90 = 6

    def __init__(self, size=200, max_steps=1000, battery=500,
                 agent_view_size=5, port=7575, camera=100,
                 render_mode="human", render_rate=3, **kwargs):
        self.whole_grid = Grid(512, 512)

        with open("./map.txt", "r") as f:
            whole_map = json.load(f)["map"]
            for each in whole_map:
                obj_type = ID_TO_OBJ[each["type"]]
                if obj_type is not None:
                    self.whole_grid.set(each["x"], each["y"], obj_type())
        super().__init__(size=size, max_steps=max_steps, battery=battery,
                         agent_view_size=agent_view_size, camera=camera,
                         port=port,
                         render_mode=render_mode,
                         render_rate=render_rate,
                         tile_size=kwargs.get("tile_size", 5))
        self.sliced_map = {}
        self.sliced_array = []
        self.sliced_info = {
            "start_pos": None,
            "damages": {}
        }

    def to_json(self):
        return {
            "height": self.height,
            "width": self.width,
            "start": self.sliced_info["start_pos"],
            "damages": self.sliced_info["damages"]
        }

    def _gen_grid(self, width, height):
        self.sliced_map = {}
        self.sliced_info = {
            "start_pos": None,
            "damages": {},
        }
        # Generate random center coordinates within the specified range
        center_x = np.random.randint(40 + int(self.size / 2), 480 - int(self.size / 2))
        center_y = np.random.randint(40 + int(self.size / 2), 480 - int(self.size / 2))
        self.sliced_info["center_x"] = center_x
        self.sliced_info["center_y"] = center_y
        # Calculate the starting and ending indices for the slice
        x_range = list(range(center_x - int(self.size / 2), center_x + int(self.size / 2)))
        y_range = list(range(center_y - int(self.size / 2), center_y + int(self.size / 2)))
        self.sliced_info["x_range"] = (center_x - int(self.size / 2), center_x + int(self.size / 2))
        self.sliced_info["y_range"] = (center_y - int(self.size / 2), center_y + int(self.size / 2))
        locations = list(itertools.product(x_range, y_range))
        # Call the original _gen_grid method to generate the base grid
        self.grid = Grid(width, height)
        slice_array = []
        for i, (x, y) in enumerate(locations):
            new_x = int(i / self.size)
            new_y = i % self.size
            self.grid.set(new_x, new_y, self.whole_grid.get(x, y))
            self.sliced_map[(new_x, new_y)] = (x, y)
            slice_array.append(OBJ_TO_ID[type(self.whole_grid.get(x, y))])
        self.slice_array = np.array(slice_array).reshape([height, width]).T
        # Random starting/goal point for the agent
        road_positions = np.argwhere(self.slice_array == OBJ_TO_ID[RoadTile])
        # To store the positions of zeros near any RoadTile
        edge_positions = []

        # Find edge RoadTile positions
        for (y, x) in road_positions:
            neighbors = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
            if any((ny < 0 or ny >= self.size or
                    nx < 0 or nx >= self.size or
                    self.slice_array[ny, nx] != OBJ_TO_ID[RoadTile]) for ny, nx in neighbors):
                edge_positions.append((x, y))
        self.edge_positions = edge_positions
        nearby_zeros = []

        # Create a ring of potential centers around each edge position
        for (x, y) in edge_positions:
            for dy, dx in [(-4, 0), (4, 0), (0, -4), (0, 4)]:
                ny, nx = y + dy, x + dx
                # Correct boundary checks to ensure the whole 5x5 block is within array bounds
                if 0 <= ny - 2 and ny + 2 < self.size and 0 <= nx - 2 and nx + 2 < self.size:
                    block = self.slice_array[ny - 2:ny + 3, nx - 2:nx + 3]
                    if np.all(block == 0):
                        nearby_zeros.append((nx, ny))
        self.start_pos = random.choice(nearby_zeros)
        self.agent_pos = self.start_pos
        self.agent_dir = 3
        # Calculate the boundaries for the 5x5 block centered on (center_y, center_x)
        start_x = self.start_pos[0] - 2
        end_x = self.start_pos[0] + 2
        start_y = self.start_pos[1] - 2
        end_y = self.start_pos[1] + 2
        # Ensure the boundaries are within the limits of the slice_array dimensions
        start_y = max(start_y, 0)
        end_y = min(end_y, self.slice_array.shape[0] - 2)
        start_x = max(start_x, 0)
        end_x = min(end_x, self.slice_array.shape[1] - 2)
        for (x, y) in itertools.product(list(range(start_x, end_x)), list(range(start_y, end_y))):
            self.grid.set(x, y, StartPoint())
        self.sliced_info["start_pos"] = (int(self.start_pos[0]), int(self.start_pos[1]))
        # Set goal
        nearby_roads = []
        for (x, y) in edge_positions:
            for dy, dx in [(-4, 0), (4, 0), (0, -4), (0, 4)]:
                ny, nx = y + dy, x + dx
                # Correct boundary checks to ensure the whole 5x5 block is within array bounds
                if 0 <= ny - 2 and ny + 2 < self.size and 0 <= nx - 2 and nx + 2 < self.size:
                    block = self.slice_array[ny - 2:ny + 3, nx - 2:nx + 3]
                    if np.all(block == 4):
                        nearby_roads.append((nx, ny, 2))

            for dy, dx in [(-6, 0), (6, 0), (0, -6), (0, 6)]:
                ny, nx = y + dy, x + dx
                # Correct boundary checks to ensure the whole 6x6 block is within array bounds
                if 0 <= ny - 3 and ny + 3 < self.size and 0 <= nx - 3 and nx + 3 < self.size:
                    block = self.slice_array[ny - 3:ny + 4, nx - 3:nx + 4]
                    if np.all(block == 4):
                        nearby_roads.append((nx, ny, 3))

            for dy, dx in [(-8, 0), (8, 0), (0, -8), (0, 8)]:
                ny, nx = y + dy, x + dx
                # Correct boundary checks to ensure the whole 7x7 block is within array bounds
                if 0 <= ny - 4 and ny + 4 < self.size and 0 <= nx - 4 and nx + 4 < self.size:
                    block = self.slice_array[ny - 4:ny + 5, nx - 4:nx + 5]
                    if np.all(block == 4):
                        nearby_roads.append((nx, ny, 4))
        damages = set()
        for each in range(random.randint(2, 7)):
            damages.add(random.choice(nearby_roads))
        for number, (cen_x, cen_y, size) in enumerate(damages):
            start_x = cen_x - size
            end_x = cen_x + size
            start_y = cen_y - size
            end_y = cen_y + size
            # Ensure the boundaries are within the limits of the slice_array dimensions
            start_y = max(start_y, 0)
            end_y = min(end_y, self.slice_array.shape[0] - 2)
            start_x = max(start_x, 0)
            end_x = min(end_x, self.slice_array.shape[1] - 2)
            for (x, y) in itertools.product(list(range(start_x, end_x)), list(range(start_y, end_y))):
                self.grid.set(x, y, DamageTile())
            self.sliced_info["damages"][number] = (int(cen_x), int(cen_y), int(size), False)

