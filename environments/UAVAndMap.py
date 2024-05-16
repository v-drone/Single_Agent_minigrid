from __future__ import annotations
from environments.CustomGrid import Grid
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import Goal
from minigrid.core.actions import IntEnum
from gymnasium.envs.registration import EnvSpec
from gymnasium import spaces
from typing import Any
import numpy as np
import requests
import random
import math
import json

mapper = {
    "lava": 1,
    "goal": 2,
}


def _organize_obs(obs, grid, info):
    obs_flatten = np.array(obs).flatten()


class UAVWithMapEmpty(EmptyEnv):
    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        throttle = 0
        brake = 1
        steering_0 = 2
        steering_left_25 = 3
        steering_left_50 = 4
        steering_right_25 = 5
        steering_right_50 = 6
        reset = 7

    def __init__(self, size=30, max_steps=400, battery=100, agent_view_size=3,
                 basic_coefficient=0.1, port=6000,
                 render_mode="human", exist=0, render_rate=3, **kwargs):

        super().__init__(size=size, max_steps=max_steps, agent_view_size=agent_view_size,
                         render_mode=render_mode, **kwargs)
        self.spec = EnvSpec("UAVWithMapEnv-v0", max_episode_steps=self.max_steps)
        # self.client = AirSimEnv()
        # To track tiles that are not yet visited by the agent
        self.actions = self.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.full_battery = battery
        self.battery = battery
        self.info = {}
        self.prev_pos = None
        self.basic_coefficient = basic_coefficient
        self.prev_distance = size * 2
        self.current_distance = size * 2
        self.walked = np.zeros(shape=[size, size], dtype=np.uint8)
        self.visited_tiles = set()
        self.unvisited_tiles = set()
        self.local_port = port
        self.local_client_id = None
        self.prev_transitions = None
        self.exist = exist
        self.render_rate = render_rate

    def connect_local_airsim_server(self, tried):
        if self.exist == 1:
            self.local_client_id = 0
            return
        if tried >= 1:
            raise Exception
        response = requests.post("http://127.0.0.1:%d/restart" % self.local_port, json={})
        if response.status_code == 200:
            self.local_client_id = response.json()["local_client_id"]
        else:
            tried += 1
            self.connect_local_airsim_server(tried)

    def kill_connect(self):
        response = requests.post("http://127.0.0.1:%d/close/" % self.local_port,
                                 json={"local_client_id": self.local_client_id})
        if response.status_code == 200:
            self.local_client_id = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.local_client_id is None:
            self.connect_local_airsim_server(0)
        super().reset()
        self.agent_dir = 3
        self.info = {}
        self.visited_tiles = set()
        self.unvisited_tiles = set()
        self.battery = self.full_battery
        self.walked = np.zeros(shape=[self.width, self.height], dtype=np.uint8)
        response = requests.post("http://127.0.0.1:%d/reset" % self.local_port,
                                 json={"local_client_id": self.local_client_id,
                                       "map": self.to_json()})
        if response.status_code == 200:
            obs, info = self.get_info()
            return obs, info
        else:
            raise Exception

    def to_json(self):
        json_return = {"mission": self.mission,
                       "height": self.height,
                       "width": self.width,
                       "map": [],
                       "start": self.start_pos,
                       }
        for y, y_d in enumerate(np.array(self.grid.grid).reshape([self.height, self.width])):
            for x, x_d in enumerate(y_d):
                if x_d is not None:
                    json_return["map"].append({
                        "x": x,
                        "y": y,
                        "type": mapper[x_d.type]
                    })
                else:
                    json_return["map"].append({
                        "x": x,
                        "y": y,
                        "type": 0
                    })
        return json_return

    def get_info(self):
        obs, self.info = self._call_airsim_info()
        self._update_grid()
        return obs, self.info

    def step(self, action):
        # Record the agent's current position before executing the action
        self.prev_pos = np.copy(self.agent_pos)
        # Update battery
        self.battery -= 1
        self.step_count += 1
        # Execute the agent's action
        self._call_airsim_step(action)
        # Update map
        obs, self.info = self.get_info()
        self.walked[self.agent_pos[1]][self.agent_pos[0]] += 1

        terminated = bool(self.info["gear"])
        truncated = self._get_fail()
        if terminated:
            reward = self._reward()
        else:
            reward = 0

        return obs, reward, terminated, truncated, {}

    def get_map_render(self):

        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
                self.agent_pos
                + f_vec * (self.agent_view_size - 1)
                - r_vec * (self.agent_view_size // 2)
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        img = self.grid.render(
            100,
            (999, 999),
            None,
            highlight_mask=highlight_mask if True else None,
        )
        return img

    def reward_breakdown(self):
        return (super()._reward() * self.basic_coefficient,
                super()._reward() * self.basic_coefficient + 0.05 * len(self.visited_tiles))

    def distance_to_closest_blue(self, pos):
        # Calculate the Manhattan distance to the closest blue tile
        return min(abs(pos[0] - x) + abs(pos[1] - y) for (x, y) in self.unvisited_tiles)

    def _reward(self) -> float:
        if not self.unvisited_tiles and self.agent_pos == self.start_pos:
            # Provide a positive reward for completing the task
            return super()._reward() * self.basic_coefficient + 0.05 * len(self.visited_tiles)
        else:
            return 0

    def _gen_grid(self, width, height):
        # Call the original _gen_grid method to generate the base grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # Random starting/goal point for the agent
        start_x, start_y = random.randint(10, width - 10), random.randint(10, height - 10)
        self.start_pos = (start_x, start_y)
        self.agent_pos = self.start_pos
        self.agent_dir = self._rand_int(0, 4)

        #  Set goal
        self.put_obj(Goal(), random.randint(1, width - 2), random.randint(1, height - 2))
        # self.grid.set(start_x, start_y, Goal())

    def _update_grid(self):
        y = int(- self.info["position"]["x"] / self.render_rate)
        x = int(self.info["position"]["y"] / self.render_rate)
        self.agent_pos = [x, y]
        self.walked[self.agent_pos[1]][self.agent_pos[0]] += 1
        roll, pitch, yaw = self.info["orientation"]
        yaw_degrees = math.degrees(yaw)

        # Normalize the yaw to [0, 360)
        if yaw_degrees < 0:
            yaw_degrees += 360

        # Divide the circle into 4 quadrants
        if 45 <= yaw_degrees < 135:
            self.agent_dir = 2  # West
        elif 135 <= yaw_degrees < 225:
            self.agent_dir = 1  # South
        elif 225 <= yaw_degrees < 315:
            self.agent_dir = 0  # East
        else:
            self.agent_dir = 3  # North

    def _call_airsim_step(self, action):
        response = requests.post("http://127.0.0.1:%d/step" % self.local_port,
                                 json={"local_client_id": self.local_client_id,
                                       "action": action})
        return response.json()["obs"], json.loads(response.json()["info"])

    def _call_airsim_info(self):
        response = requests.post("http://127.0.0.1:%d/info" % self.local_port,
                                 json={"local_client_id": self.local_client_id})
        return response.json()["obs"], json.loads(response.json()["info"])

    # def _get_obs(self, air_sim_obs):
    #     obs = [np.array(air_sim_obs).flatten().astype(np.uint8),
    #            self.get_frame(tile_size=self.tile_size).flatten(),
    #            ]

    def _get_fail(self):
        if self.battery <= 0:
            return True
        else:
            if self.info["position"]["z"] > 10:
                return True
            else:
                return False

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
