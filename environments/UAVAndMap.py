from __future__ import annotations
from environments.CustomGrid import Grid
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import Goal
from minigrid.core.actions import IntEnum
from minigrid.core.mission import MissionSpace
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


class UAVWithMapEmpty(EmptyEnv):
    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        throttle = 0
        brake = 1
        steering_right_half = 2
        steering_right_full = 3
        steering_left_half = 4
        steering_left_full = 5
        reset_car = 6

    def __init__(self, size=30, max_steps=400, battery=100,
                 agent_view_size=3, port=5000, camera=100,
                 render_mode="human", render_rate=3, **kwargs):

        super().__init__(size=size, max_steps=max_steps, agent_view_size=agent_view_size,
                         render_mode=render_mode, tile_size=kwargs.get("tile_size", 5))

        self.spec = EnvSpec("UAVWithMapEnv-v0", max_episode_steps=self.max_steps)
        self.size = size
        self.actions = self.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.full_battery = battery
        self.battery = battery
        self.info = {}
        self.prev_pos = None
        self.camera = camera
        self.walked = np.zeros(shape=[size, size], dtype=np.uint8)
        self.visited_tiles = set()
        self.unvisited_tiles = set()
        self.local_port = port
        self.prev_transitions = None
        self.render_rate = render_rate
        self.goal = [0, 0]
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=np.array([self.camera, self.camera, 3]),
            dtype="uint8",
        )
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.observation_space = spaces.Dict(
            {
                "image": image_space,
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            }
        )

    def reset_airsim_win(self, tried=2):
        if tried >= 1:
            return False
        response = requests.post("http://127.0.0.1:%d/restart" % self.local_port, json={})
        if response.status_code == 200:
            return True
        else:
            tried += 1
            self.reset_airsim_win(tried)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.local_client_id is None:
            self.reset_airsim_win(0)
        super().reset()
        self.agent_dir = 3
        self.info = {}
        self.visited_tiles = set()
        self.unvisited_tiles = set()
        self.battery = self.full_battery
        self.walked = np.zeros(shape=[self.width, self.height], dtype=np.uint8)
        response = requests.post("http://127.0.0.1:%d/reset" % self.local_port,
                                 json={"map": self.to_json()})
        if response.status_code == 200:
            obs = self._get_info()
            return obs, self.info
        elif response.status_code == 500:
            self.reset_airsim_win(2)

    def step(self, action):
        # Record the agent's current position before executing the action
        self.prev_pos = np.copy(self.agent_pos)
        # Update battery
        self.battery -= 1
        self.step_count += 1
        # Execute the agent's action
        _, _, exception = self._call_airsim_step(action)
        if exception:
            self.reset()
        # Update map
        obs = self._get_info()
        self.walked[self.agent_pos[1]][self.agent_pos[0]] += 1

        terminated = bool(self.info["gear"])
        truncated = self._get_fail()
        if terminated:
            reward = self._reward()
        else:
            reward = 0

        return obs, reward, terminated, truncated, {}

    def render(self):
        view_obs, self.info = self._call_airsim_info()
        self._update_grid()
        return np.array(view_obs)

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

    def _get_info(self):
        view_obs, self.info, exception = self._call_airsim_info()
        self._update_grid()
        obs = {"image": np.array(view_obs, dtype=np.uint8), "direction": self.agent_dir, "mission": self.mission}

        return obs

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
        # Set goal
        goal = [random.randint(2, width - 3), random.randint(2, height - 3)]
        self.goal = goal
        self.put_obj(Goal(), goal[0], goal[1])

    def _update_grid(self):
        y = int(- self.info["position"]["x"] / self.render_rate)
        x = int(self.info["position"]["y"] / self.render_rate)
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        self.agent_pos = [x, y]
        self.walked[self.agent_pos[1]][self.agent_pos[0]] += 1
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

    def _call_airsim_step(self, action):
        response = requests.post("http://127.0.0.1:%d/step" % self.local_port,
                                 json={"local_client_id": self.local_client_id,
                                       "action": int(action)})
        if response.status_code == 200:
            return response.json()["obs"], json.loads(response.json()["info"]), False
        elif response.status_code == 500:
            self.reset_airsim_win(2)
            return {}, {}, True

    def _call_airsim_info(self):
        response = requests.post("http://127.0.0.1:%d/info" % self.local_port,
                                 json={"local_client_id": self.local_client_id})
        if response.status_code == 200:
            return response.json()["obs"], json.loads(response.json()["info"]), False
        elif response.status_code == 500:
            self.reset_airsim_win(2)
            return {}, {}, True

    def _get_fail(self):
        if self.battery <= 0:
            return True
        else:
            if self.info["position"]["z"] > 100 or self.agent_pos[0] == 29 or self.agent_pos[0] == 0 or \
                    self.agent_pos[1] == 29 or self.agent_pos[1] == 0:
                return True
            else:
                return False

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
