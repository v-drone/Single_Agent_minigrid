from __future__ import annotations

import random

from environments.MutilRoadWithTrodEnv import RouteWithTrodEnv, TrodTile
from environments.MutilRoadEnv import PathTile, Goal
from minigrid.core.world_object import Floor, Lava
from minigrid.core.grid import Grid
from typing import Any
import numpy as np


# Update the RouteEnv class to use the new RoutePoint object
class RouteByMapEnv(RouteWithTrodEnv):

    def __init__(self, size=20, max_steps=100, routes=(3, 5), trods=(3, 5),
                 agent_view_size=7,
                 battery=100, render_mode="human", maps=None, **kwargs):
        super().__init__(size=size, max_steps=max_steps, routes=routes,
                         agent_view_size=agent_view_size,
                         battery=battery, render_mode=render_mode, **kwargs)
        self.trods = trods
        self.unvisited_trods = set()
        self.visited_trods = set()
        self.maps = maps

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, _ = super().reset()
        self.visited_trods = set()
        return obs, {}

    def _gen_grid(self, width, height):
        if self.maps is not None:
            self.grid = Grid(width, height)
            map_for_use = random.sample(self.maps, 1)[0]
            self._load_map_from_file(map_for_use)
            self.agent_pos = self.start_pos
            self.agent_dir = self.agent_start_dir
            self.mission = ""
        else:
            # Call the original _gen_grid method to generate the base grid
            super()._gen_grid(width, height)

    def _load_map_from_file(self, map_texts_y):
        for y, line in enumerate(map_texts_y):
            for x, char in enumerate(line.strip()):
                self._place_object_from_char(x, y, char)

    def _place_object_from_char(self, x, y, char):
        if char == '.':
            pass
        elif char == 'P':
            self.grid.set(x, y, PathTile())
            self.unvisited_tiles.add((x, y))
        elif char == 'T':
            self.grid.set(x, y, TrodTile())
            self.unvisited_trods.add((x, y))
        elif char == 'G':
            self.grid.set(x, y, Goal())
            self.start_pos = (x, y)
        elif char == 'L':
            self.grid.set(x, y, Lava())
