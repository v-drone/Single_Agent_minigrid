from __future__ import annotations

import traceback
from typing import Any, Optional
from environments.AirSimException import GenException, AirSimRetryError
from environments.CustomGrid import Grid, Lava, StartPoint, BuildTile, RoadTile, DamageTile, WalkWayTile
from environments.EmptyAndMap import EmptyWithMapEmpty
from minigrid.core.actions import IntEnum
from gymnasium import spaces
import numpy as np
import itertools
import random
import math
import json

ID_TO_OBJ_DICT = {
    0: None,
    1: Lava,
    2: StartPoint,
    3: BuildTile,
    4: RoadTile,
    5: DamageTile,
    6: WalkWayTile,
}

OBJ_TO_ID_DICT = {
    type(None): 0,
    Lava: 1,
    StartPoint: 2,
    BuildTile: 3,
    RoadTile: 4,
    DamageTile: 5,
    WalkWayTile: 6
}


class RoadNetworkAndMap(EmptyWithMapEmpty):
    # Enumeration of possible actions
    class Actions(IntEnum):
        # forward
        forward_0 = 0
        forward_1 = 1
        forward_2 = 2
        # yaw
        yaw_45 = 3
        yaw_90 = 4
        yaw_n_45 = 5
        yaw_n_90 = 6

    def __init__(self, size=50, max_steps=500, battery=200,
                 agent_view_size=5, port=7575, camera=100,
                 render_mode="human",
                 render_rate=3,
                 agent_size=3,
                 map_path="./map.txt",
                 flip_y=True,
                 **kwargs):
        """
        Initialize the downstream environment with a large map.

        Args:
            size: The subgrid dimension to be sliced from the whole map.
            max_steps: Maximum steps in an episode.
            battery: Initial battery capacity.
            agent_view_size: View range if needed.
            port: Port for AirSim communication.
            camera: Camera resolution.
            render_mode: Render mode (e.g. 'human').
            render_rate: The factor used to convert position from AirSim to grid coordinates.
            agent_size: Agent size used in CustomGrid.
            map_path: Path to the JSON map file.
            **kwargs: Additional arguments passed to the parent init.
        """
        self.agent_size = agent_size
        self.whole_grid = Grid(512, 512, self.agent_size)
        self.whole_grid_start = np.array([-322.5, -324.5])  # Offset in the simulation world
        self.reset_start = np.array([100, 100])  # Reference start offset
        self.map_path = map_path
        self.flip_y = flip_y
        self.load_whole_map()

        # Call parent constructor
        super().__init__(size=size, max_steps=max_steps, battery=battery,
                         agent_view_size=agent_view_size, camera=camera,
                         port=port, render_mode=render_mode,
                         render_rate=render_rate,
                         tile_size=kwargs.get("render_rate", 5),
                         **kwargs)

        # Discrete action space
        self.actions = self.Actions
        self.action_space = spaces.Discrete(7)

        # Additional attributes
        self.sliced_info = {"damages": {}}
        self.movement = []
        self.reward = 0

    def to_json(self):
        """
        Convert the environment's current state (e.g., subgrid, start, damages)
        into a JSON-compatible dict for sending to the simulator.
        """
        doc = {
            "height": self.height,
            "width": self.width,
            "start": list(np.array(self.start_pos) + self.sliced_info["top_left"] + self.whole_grid_start),
            "top_left": list(self.sliced_info["top_left"] + self.whole_grid_start),
            "damages": {
                "A": (0, 0, 0),
                "B": (0, 0, 0),
                "C": (0, 0, 0),
                "D": (0, 0, 0),
                "E": (0, 0, 0),
            }
        }
        mapper = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
        for i, damage in self.sliced_info["damages"].items():
            doc["damages"][mapper[i]] = (
                int(damage[0] - self.start_pos[0]),
                int(damage[1] - self.start_pos[1]),
                damage[2]
            )
        return doc

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None, retry=5):
        """
        Reset the environment.
        - Clear the tile states in the large grid (whole_grid).
        - Attempt to call parent reset(), which in turn calls _gen_grid().
        - If GenException is raised, retry until `retry` times are exceeded.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options, not used in this environment.
            retry: How many times to retry generating a valid subgrid.

        Returns:
            obs, info: Observation and info from the parent environment's reset.

        Raises:
            AirSimRetryError: If all retries fail to generate a valid map.
        """
        for tile in self.whole_grid.grid:
            if tile is not None:
                tile.reset_tile()

        if retry <= 0:
            raise AirSimRetryError("Exceeded max attempts to generate map. Check map data.")

        try:
            obs, info = super().reset(seed=seed, options=options)
        except GenException:
            self.gym_logger.warning("GenException encountered during reset")
            return self.reset(seed=seed, options=options, retry=retry - 1)

        self.movement = []
        self.agent_dir = 3
        self.reward = 0
        return obs, info

    def step(self, action):
        """
        Take one step in the environment using the given action.
        After calling the parent's step, compute custom reward logic here.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        if isinstance(self.grid.get(*self.agent_pos), StartPoint):
            self.battery = self.full_battery

        # Recompute reward
        reward = self._reward()

        return obs, reward, terminated, truncated, info

    def _check_fail(self):
        """
        Override the parent's _get_fail to check additional conditions:
        collision, battery, or high altitude (z > 100).
        """
        if self.info["collision"]:
            return True
        if self.battery <= 0 or self.step_count > self.max_steps:
            return True
        if self.info["position"]["z"] > 100:
            return True
        return False

    def _check_success(self):
        """
        Override the parent's _get_success to check if >80% RoadTile is visited
        and the agent is on a StartPoint tile.
        """
        visited = 0
        total = 0
        for tile in self.grid.grid:
            if isinstance(tile, RoadTile):
                total += 1
                if tile.visit == 1:
                    visited += 1
        if total == 0:
            return False

        visited_rate = visited / total
        if visited_rate > 0.8 and isinstance(self.grid.get(*self.agent_pos), StartPoint):
            return True
        return False

    def _reward(self):
        """
        Compute the final reward for the current step, incorporating
        the step-based reward (self.reward) and success/fail bonuses.
        """
        terminated, truncated = self._check_status()
        base_reward = self.reward

        if truncated:
            bonus = -5
        elif terminated:
            bonus = 10
        else:
            bonus = 0

        return base_reward + bonus

    def _gen_grid(self, width, height):
        """
        Modified to attempt multiple tries of random center.
        If none succeed, raise GenException.
        """
        max_tries = 5
        for attempt in range(max_tries):
            self.gym_logger.debug(f"Attempt {attempt + 1}/{max_tries} to slice subgrid.")
            try:
                self._try_slice_and_place(width, height)
                # If successful, break out
                self.gym_logger.debug("Successfully sliced subgrid and placed start_pos/damages.")
                return
            except GenException as e:
                self.gym_logger.debug(f"Attempt {attempt + 1} failed: {e}")
        # If all attempts fail
        raise GenException(f"Failed to generate a valid subgrid after {max_tries} attempts.")

    def _try_slice_and_place(self, width, height):
        """
        Actually pick a random center, slice the subgrid, place start pos, place damages.
        May raise GenException if no valid start_pos or other failure.
        """
        self.sliced_info = {"damages": {}}
        subgrid = Grid(width, height, self.whole_grid.agent_size)

        # Random center
        center_x = np.random.randint(80 + int(self.size / 2), 420 - int(self.size / 2))
        center_y = np.random.randint(80 + int(self.size / 2), 420 - int(self.size / 2))
        self.gym_logger.debug(f"_try_slice_and_place: center=({center_x}, {center_y})")

        # top-left corner
        top_left = np.array([int(center_x - (self.size / 2)), int(center_y - int(self.size / 2))])
        top_left_unity = top_left + self.whole_grid_start

        # copy tiles from whole_grid to subgrid
        x_range = range(center_x - int(self.size / 2), center_x + int(self.size / 2))
        y_range = range(center_y - int(self.size / 2), center_y + int(self.size / 2))
        locations = list(itertools.product(x_range, y_range))

        slice_array = []
        for i, (x, y) in enumerate(locations):
            new_x = int(i / self.size)
            new_y = i % self.size
            tile = self.whole_grid.get(x, y)
            subgrid.set(new_x, new_y, tile)
            slice_array.append(OBJ_TO_ID_DICT[type(tile)])

        slice_array = np.array(slice_array).reshape([height, width]).T

        self.grid = subgrid
        self.slice_array = slice_array
        self.sliced_info["top_left"] = top_left

        self.gym_logger.debug(f"RoadTile count={np.sum(slice_array == 4)}, NoneTile count={np.sum(slice_array == 0)}")

        # place start pos and damages
        self._place_start_pos()  # may raise GenException
        self._place_damages()  # might fail to place damages, but won't raise GenException unless you want it to

    def _update_grid(self):
        """
        Update agent position in subgrid based on info from AirSim,
        then compute step-based reward, update direction etc.
        """
        relative_y = int(self.info["position"]["x"] / self.render_rate - self.reset_start[1])
        relative_x = int(self.info["position"]["y"] / self.render_rate - self.reset_start[0])
        x = self.start_pos[0] + relative_x
        x = max(0, min(x, self.width - 1))
        y = self.start_pos[1] + relative_y
        y = max(0, min(y, self.height - 1))
        self.agent_pos = [x, y]

        # previous
        prev_relative_y = int(self.info["prev_position"]["x"] / self.render_rate - self.reset_start[1])
        prev_relative_x = int(self.info["prev_position"]["y"] / self.render_rate - self.reset_start[0])
        prev_x = self.start_pos[0] + prev_relative_x
        prev_x = max(0, min(prev_x, self.width - 1))
        prev_y = self.start_pos[1] + prev_relative_y
        prev_y = max(0, min(prev_y, self.height - 1))

        # step-based reward
        self.reward = self._mark_path(prev_x, prev_y, x, y)

        # update direction
        roll, pitch, yaw = self.info["orientation"]
        yaw_degrees = math.degrees(yaw)
        if yaw_degrees < 0:
            yaw_degrees += 360
        self.info["yaw_degrees"] = yaw_degrees

        if 45 <= yaw_degrees < 135:
            self.agent_dir = 2  # West
        elif 135 <= yaw_degrees < 225:
            self.agent_dir = 1  # South
        elif 225 <= yaw_degrees < 315:
            self.agent_dir = 0  # East
        else:
            self.agent_dir = 3  # North

    def _mark_path(self, start_x, start_y, end_x, end_y):
        """
        Mark the path from (start_x, start_y) to (end_x, end_y),
        accumulate reward for each walked cell.
        """
        reward = 0
        steps = max(abs(end_x - start_x), abs(end_y - start_y)) + 1
        for step in range(steps + 1):
            t = step / steps
            interp_x = round(start_x + t * (end_x - start_x))
            interp_y = round(start_y + t * (end_y - start_y))
            reward += self._walk(interp_x, interp_y)
        return reward

    def _walk(self, x, y):
        """
        Increase 'walked' counter for nearby cells; accumulate tile-based rewards.
        """
        reward = 0
        for i in range(max(0, y - 1), min(y + 2, self.height)):
            for j in range(max(0, x - 1), min(x + 2, self.width)):
                self.walked[i][j] += 1
                self.movement.append([j, i])
                tile = self.grid.get(j, i)
                if tile is not None:
                    reward += tile.reward
                    tile.update_color()
        return reward

    def load_whole_map(self):
        """
        Load the entire map from the specified file into self.whole_grid.
        Only once in __init__.
        """
        with open(self.map_path, "r") as f:
            data = json.load(f)
            whole_map = data["map"]
            if self.flip_y:
                whole_map = [{**i, "y": 511 - i["y"]} for i in whole_map]

        for each in whole_map:
            obj_type = ID_TO_OBJ_DICT[each["type"]]
            if obj_type is not None:
                if "mark" in each:
                    obj = obj_type(mark_type=each["mark"])
                else:
                    obj = obj_type()
                self.whole_grid.set(each["x"], each["y"], obj)

    def _place_start_pos(self):
        """
        Find a valid start_pos on a RoadTile area and set it in the subgrid.
        Raises GenException if no valid start can be placed.
        """
        road_positions = np.argwhere(self.slice_array == OBJ_TO_ID_DICT[RoadTile])
        start_pos_list = []

        for (y, x) in road_positions:
            # Instead of checking a 3x3 block,
            # we'll just look at some neighbors where slice_array == 0
            neighbors = [
                (y + dy, x + dx)
                for dy in (-3, -2, -1, 0, 1, 2, 3)
                for dx in (-3, -2, -1, 0, 1, 2, 3)
                if (dy, dx) != (0, 0)
            ]
            for ny, nx in neighbors:
                # Make sure in range
                if 0 <= ny < self.size and 0 <= nx < self.size:
                    # Check 1x1 is empty => slice_array[ny, nx] == 0
                    if self.slice_array[ny, nx] == 0:
                        start_pos_list.append((nx, ny))

        if not start_pos_list:
            raise GenException("No valid start_pos found in the sliced map (1x1 check).")

        self.start_pos_list = start_pos_list
        self.start_pos = random.choice(start_pos_list)
        self.agent_pos = self.start_pos
        self.agent_dir = 1

        # Now place a 5×5 block of StartPoint around that start
        sx = self.start_pos[0] - 2
        ex = self.start_pos[0] + 3
        sy = self.start_pos[1] - 2
        ey = self.start_pos[1] + 3
        sx = max(sx, 0)
        ex = min(ex, self.slice_array.shape[1])
        sy = max(sy, 0)
        ey = min(ey, self.slice_array.shape[0])

        for (x, y) in itertools.product(range(sx, ex), range(sy, ey)):
            obj = StartPoint()
            obj.unity_pos = [
                x + (self.sliced_info["top_left"][0] + self.whole_grid_start[0]),
                y + (self.sliced_info["top_left"][1] + self.whole_grid_start[1])
            ]
            self.grid.set(x, y, obj)

    def _place_damages(self):
        """
        Place 2-5 damage areas near RoadTile.
        We keep your existing logic checking 3x3,5x5,7x7.
        If random.choice fails, we skip.
        """
        road_positions = np.argwhere(self.slice_array == OBJ_TO_ID_DICT[RoadTile])
        damaged_list = []

        for (y, x) in road_positions:
            # 3x3
            block_3x3 = self.slice_array[y - 1:y + 2, x - 1:x + 2] \
                if (y - 1 >= 0 and x - 1 >= 0
                    and y + 2 <= self.slice_array.shape[0]
                    and x + 2 <= self.slice_array.shape[1]) else None
            if block_3x3 is not None and (block_3x3.shape == (3, 3)) and (np.all(block_3x3 == 4)):
                damaged_list.append((x, y, 1))

            # 5x5
            block_5x5 = self.slice_array[y - 2:y + 3, x - 2:x + 3] \
                if (y - 2 >= 0 and x - 2 >= 0
                    and y + 3 <= self.slice_array.shape[0]
                    and x + 3 <= self.slice_array.shape[1]) else None
            if block_5x5 is not None and (block_5x5.shape == (5, 5)) and (np.all(block_5x5 == 4)):
                damaged_list.append((x, y, 2))

            # 7x7
            block_7x7 = self.slice_array[y - 3:y + 4, x - 3:x + 4] \
                if (y - 3 >= 0 and x - 3 >= 0
                    and y + 4 <= self.slice_array.shape[0]
                    and x + 4 <= self.slice_array.shape[1]) else None
            if block_7x7 is not None and (block_7x7.shape == (7, 7)) and (np.all(block_7x7 == 4)):
                damaged_list.append((x, y, 3))

        # Randomly pick 2-5 damage positions
        damages = set()
        for _ in range(random.randint(2, 5)):
            try:
                damages.add(random.choice(damaged_list))
            except IndexError:
                pass

        for number, (cen_x, cen_y, size) in enumerate(damages):
            self.sliced_info["damages"][number + 1] = (cen_x, cen_y, size, False)
            sx = cen_x - size
            ex = cen_x + size
            sy = cen_y - size
            ey = cen_y + size
            sx = max(sx, 0)
            ex = min(ex, self.slice_array.shape[1] - 1)
            sy = max(sy, 0)
            ey = min(ey, self.slice_array.shape[0] - 1)

            for (x, y) in itertools.product(range(sx, ex + 1), range(sy, ey + 1)):
                obj = DamageTile()
                obj.unity_pos = [
                    x + (self.sliced_info["top_left"][0] + self.whole_grid_start[0]),
                    y + (self.sliced_info["top_left"][1] + self.whole_grid_start[1])
                ]
                self.grid.set(x, y, obj)
