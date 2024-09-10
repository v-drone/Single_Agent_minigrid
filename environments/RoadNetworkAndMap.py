from __future__ import annotations

from typing import Any

from environments.CustomGrid import Grid, Lava, StartPoint, BuildTile, RoadTile, DamageTile, WalkWayTile
from environments.EmptyAndMap import EmptyWithMapEmpty
from minigrid.core.actions import IntEnum
import numpy as np
import itertools
import random
import copy
import math
import json

ID_TO_OBJ = {
    0: None,
    1: Lava,
    2: StartPoint,
    3: BuildTile,
    4: RoadTile,
    5: DamageTile,
    6: WalkWayTile,
}

OBJ_TO_ID = {
    type(None): 0,
    Lava: 1,
    StartPoint: 2,
    BuildTile: 3,
    RoadTile: 4,
    DamageTile: 5,
    WalkWayTile: 6
}


class GenException(Exception):
    def __init__(self, message="Gen Error", e=None):
        self.message = message
        if e is not None:
            self.message += f"; {e}"
        super().__init__(self.message)


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

    def __init__(self, size=50, max_steps=500, battery=200,
                 agent_view_size=5, port=7575, camera=100,
                 render_mode="human",
                 render_rate=3,
                 agent_size=3,
                 **kwargs):
        self.agent_size = agent_size
        self.whole_grid = Grid(512, 512, self.agent_size)
        with open("./map.txt", "r") as f:
            whole_map = json.load(f)["map"]
            whole_map = [{**i, "y": 511 - i["y"]} for i in whole_map]
            for each in whole_map:
                obj_type = ID_TO_OBJ[each["type"]]
                if obj_type is not None:
                    if "mark" in each:
                        obj = obj_type(mark_type=each["mark"])
                    else:
                        obj = obj_type()
                    self.whole_grid.set(each["x"], each["y"], obj)
                self.whole_grid_start = np.array([-322.5, -324.5])
                self.reset_start = np.array([100, 100])
        super().__init__(size=size, max_steps=max_steps, battery=battery,
                         agent_view_size=agent_view_size, camera=camera,
                         port=port,
                         render_mode=render_mode,
                         render_rate=render_rate,
                         tile_size=kwargs.get("tile_size", 5))
        self.sliced_info = {
            "damages": {}
        }
        self.movement = []
        self.reward = 0

    def to_json(self):
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
        mapper = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E"
        }
        for i, damage in self.sliced_info["damages"].items():
            doc["damages"][mapper[i]] = (int(damage[0] - self.start_pos[0]),
                                         int(damage[1] - self.start_pos[1]),
                                         damage[2])
        return doc

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, retry=5):
        for each in self.whole_grid.grid:
            if each is not None:
                each.reset_tile()
        try:
            obs, _ = super().reset()
        except GenException:
            return self.reset(retry=retry)
        self.movement = []
        self.agent_dir = 3
        self.reward = 0
        return obs, _

    def step(self, action):
        obs, reward, terminated, truncated, _ = super().step(action)
        if type(self.grid.get(*self.agent_pos)) == StartPoint:
            self.battery = self.full_battery
        reward = self._reward()
        return obs, reward, terminated, truncated, _

    def _get_fail(self):
        if self.info["collision"]:
            return True
        if self.battery <= 0 or self.step_count > self.max_steps:
            return True
        if self.info["position"]["z"] > 100:
            return True
        return False

    def _get_success(self):
        visited = 0
        total = 0
        for each in self.grid.grid:
            if type(each) == RoadTile:
                total += 1
                if each.visit == 1:
                    visited += 1
        visited_rate = visited / total
        if visited_rate > 0.8 and type(self.grid.get(*self.agent_pos)) == StartPoint:
            return True
        else:
            return False

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        terminated, truncated = self._check_status()
        basic_reward = self.reward
        if truncated:
            bonus = -100
        elif terminated:
            bonus = 100
        else:
            bonus = 0
        return basic_reward + bonus

    def _gen_grid(self, width, height):
        self.sliced_info = {
            "damages": {}
        }
        self.grid = Grid(width, height, self.agent_size)
        ### generate random center coordinates within the specified range
        center_x = np.random.randint(100 + int(self.size / 2), 400 - int(self.size / 2))
        center_y = np.random.randint(100 + int(self.size / 2), 400 - int(self.size / 2))
        top_left = np.array([int(center_x - (self.size / 2)), int(center_y - int(self.size / 2))])
        top_left_unity = copy.copy(top_left + self.whole_grid_start)
        self.sliced_info["top_left"] = top_left
        # Copy map
        x_range = list(range(center_x - int(self.size / 2), center_x + int(self.size / 2)))
        y_range = list(range(center_y - int(self.size / 2), center_y + int(self.size / 2)))
        locations = list(itertools.product(x_range, y_range))
        slice_array = []
        for i, (x, y) in enumerate(locations):
            new_x = int(i / self.size)
            new_y = i % self.size
            self.grid.set(new_x, new_y, self.whole_grid.get(x, y))
            slice_array.append(OBJ_TO_ID[type(self.whole_grid.get(x, y))])
        slice_array = np.array(slice_array).reshape([height, width]).T
        self.slice_array = slice_array
        # Set start
        road_positions = np.argwhere(slice_array == OBJ_TO_ID[RoadTile])
        start_pos_list = []
        for (y, x) in road_positions:
            neighbors = [(y + dy, x + dx) for dy in (-3, -2, -1, 0, 1, 2, 3) for dx in (-3, -2, -1, 0, 1, 2, 3) if
                         (dy, dx) != (0, 0)]
            for ny, nx in neighbors:
                block = slice_array[ny - 1:ny + 2, nx - 1:nx + 2]
                if (
                        (3 < ny < self.size - 3) and
                        (3 < nx < self.size - 3) and
                        (block.shape == (3, 3)) and
                        (np.sum(block) == 0)
                ):
                    start_pos_list.append((nx, ny))
        if len(start_pos_list) == 0:
            raise GenException()
        self.start_pos_list = start_pos_list
        self.start_pos = random.choice(start_pos_list)
        self.agent_pos = self.start_pos
        self.agent_dir = 1
        ## set grid map
        ### calculate the boundaries for the 5x5 block centered on (center_y, center_x)
        start_x = self.start_pos[0] - 2
        end_x = self.start_pos[0] + 3
        start_y = self.start_pos[1] - 2
        end_y = self.start_pos[1] + 3
        ### ensure the boundaries are within the limits of the slice_array dimensions
        start_y = max(start_y, 0)
        end_y = min(end_y, slice_array.shape[0] - 2)
        start_x = max(start_x, 0)
        end_x = min(end_x, slice_array.shape[1] - 2)
        for (x, y) in itertools.product(list(range(start_x, end_x)), list(range(start_y, end_y))):
            obj = StartPoint()
            obj.unity_pos = [x + top_left_unity[0], y + top_left_unity[1]]
            self.grid.set(x, y, obj)

        # Set damaged point
        ## find edge RoadTile positions
        damaged_list = []
        for (y, x) in road_positions:
            block_3x3 = slice_array[y - 1:y + 2, x - 1:x + 2]
            if (
                    (1 <= y < self.size - 1) and
                    (1 <= x < self.size - 1) and
                    (block_3x3.shape == (3, 3)) and
                    (np.all(block_3x3 == 4))
            ):
                damaged_list.append((x, y, 1))

            block_5x5 = slice_array[y - 2:y + 3, x - 2:x + 3]
            if (
                    (2 <= y < self.size - 2) and
                    (2 <= x < self.size - 2) and
                    (block_5x5.shape == (5, 5)) and
                    (np.all(block_5x5 == 4))
            ):
                damaged_list.append((x, y, 2))

            block_7x7 = slice_array[y - 3:y + 4, x - 3:x + 4]
            if (
                    (3 <= y < self.size - 3) and
                    (3 <= x < self.size - 3) and
                    (block_7x7.shape == (7, 7)) and
                    (np.all(block_7x7 == 4))
            ):
                damaged_list.append((x, y, 3))

        ## choice damaged point
        damages = set()
        for each in range(random.randint(2, 5)):
            damages.add(random.choice(damaged_list))
        ### damaged point in the map
        for number, (cen_x, cen_y, size) in enumerate(damages):
            self.sliced_info["damages"][number + 1] = (copy.copy(cen_x), copy.copy(cen_y), int(size), False)
            start_x = cen_x - size
            end_x = cen_x + size
            start_y = cen_y - size
            end_y = cen_y + size
            ### ensure the boundaries are within the limits of the slice_array dimensions
            start_y = max(start_y, 0)
            end_y = min(end_y, slice_array.shape[0] - 1)
            start_x = max(start_x, 0)
            end_x = min(end_x, slice_array.shape[1] - 1)
            for (x, y) in itertools.product(list(range(start_x, end_x + 1)), list(range(start_y, end_y + 1))):
                obj = DamageTile()
                obj.unity_pos = [x + top_left_unity[0], y + top_left_unity[1]]
                self.grid.set(x, y, obj)

    def _update_grid(self):
        # current pos
        relative_y = int(self.info["position"]["x"] / self.render_rate - self.reset_start[1])
        relative_x = int(self.info["position"]["y"] / self.render_rate - self.reset_start[0])
        x = self.start_pos[0] + relative_x
        x = max(0, min(x, self.width - 1))
        y = self.start_pos[1] + relative_y
        y = max(0, min(y, self.height - 1))
        self.agent_pos = [x, y]
        # last pos
        prev_relative_y = int(self.info["prev_position"]["x"] / self.render_rate - self.reset_start[1])
        prev_relative_x = int(self.info["prev_position"]["y"] / self.render_rate - self.reset_start[0])
        prev_x = self.start_pos[0] + prev_relative_x
        prev_x = max(0, min(prev_x, self.width - 1))
        prev_y = self.start_pos[1] + prev_relative_y
        prev_y = max(0, min(prev_y, self.height - 1))
        self.reward = self._mark_path(prev_x, prev_y, x, y)
        roll, pitch, yaw = self.info["orientation"]
        yaw_degrees = math.degrees(yaw)

        # Normalize the yaw to [0, 360)
        if yaw_degrees < 0:
            yaw_degrees += 360
        self.info["yaw_degrees"] = yaw_degrees
        # Divide the circle into 4 quadrants
        if 45 <= yaw_degrees < 135:
            self.agent_dir = 2  # West
        elif 135 <= yaw_degrees < 225:
            self.agent_dir = 1  # South
        elif 225 <= yaw_degrees < 315:
            self.agent_dir = 0  # East
        else:
            self.agent_dir = 3  # North

    def _mark_path(self, start_x, start_y, end_x, end_y):
        reward = 0
        steps = max(abs(end_x - start_x), abs(end_y - start_y)) + 1
        for step in range(steps + 1):
            t = step / steps
            interp_x = round(start_x + t * (end_x - start_x))
            interp_y = round(start_y + t * (end_y - start_y))
            reward += self._walk(interp_x, interp_y)
        return reward

    def _walk(self, x, y):
        reward = 0
        for i in range(max(0, y - 1), min(y + 2, self.height)):
            for j in range(max(0, x - 1), min(x + 2, self.width)):
                self.walked[i][j] += 1
                self.movement.append([j, i])
                if self.grid.get(j, i) is not None:
                    reward += self.grid.get(j, i).reward
                    self.grid.get(j, i).update_color()
        return reward
