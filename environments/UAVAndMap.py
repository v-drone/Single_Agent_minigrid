from __future__ import annotations
from environments.AirSimException import AirSimResponseError, AirSimConnectionError, AirSimRestartFailed
from environments.CustomGrid import Grid
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import Goal
from minigrid.core.actions import IntEnum
from minigrid.core.mission import MissionSpace
from gymnasium.envs.registration import EnvSpec
from gymnasium import spaces
from typing import Any
import numpy as np
import logging
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
        throttle = 0
        brake = 1
        steering_right_half = 2
        steering_right_full = 3
        steering_left_half = 4
        steering_left_full = 5
        reset_car = 6

    def __init__(self, size=30, max_steps=400, battery=100,
                 agent_view_size=3, port=7575, camera=100,
                 render_mode="human", render_rate=3, **kwargs):
        super().__init__(size=size, max_steps=max_steps, agent_view_size=agent_view_size,
                         render_mode=render_mode, tile_size=kwargs.get("tile_size", 5))
        self.spec = EnvSpec("UAVWithMapEnv-v0", max_episode_steps=self.max_steps)
        self.action_space = spaces.Discrete(len(self.actions))
        self.actions = self.Actions
        self.size = size
        self.full_battery = battery
        self.battery = battery
        self.info = {}
        self.prev_pos = None
        self.camera = camera
        self.visited_tiles = set()
        self.unvisited_tiles = set()
        self.walked = np.zeros(shape=[self.size, self.size], dtype=np.uint8)

        # Logging setup
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initialized UAVWithMapEmpty")

        self.manager_port = port
        self.local_port = None
        self._set_local_port()

        self.prev_transitions = None
        self.render_rate = render_rate
        self.goal = [0, 0]
        image_space = spaces.Box(low=0, high=255, shape=np.array([self.camera, self.camera, 3]), dtype="uint8")
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.observation_space = spaces.Dict(
            {"image": image_space, "direction": spaces.Discrete(4), "mission": mission_space})

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, _ = super().reset()
        try:
            self._check_airsim()
            self.agent_dir = 3
            self.info = {}
            self.visited_tiles = set()
            self.unvisited_tiles = set()
            self.battery = self.full_battery
            self.walked = np.zeros(shape=[self.width, self.height], dtype=np.uint8)
            self._call_airsim_reset()
            return self._update_info(*self._call_airsim_info()), {}
        except Exception as e:
            self.logger.error(f"Whole Restart failed: {str(e)}")
            self.close()

    def step(self, action):
        self.prev_pos = np.copy(self.agent_pos)
        self.battery -= 1
        self.step_count += 1
        try:
            obs, info = self._call_airsim_step(action)
            obs = self._update_info(obs, info)
            self.walked[self.agent_pos[1]][self.agent_pos[0]] += 1
            terminated = bool(self.info.get("gear", False))
            truncated = self._get_fail()
            reward = self._reward() if terminated else 0
            self.prev_transitions = (obs, reward, terminated, truncated, {})
        except Exception as e:
            self.logger.error(f"Exception during AirSim step call: {str(e)}")
            self.reset()
            obs, reward, terminated, truncated, _ = self.prev_transitions
        return obs, reward, terminated, truncated, {}

    def render(self):
        view_obs, self.info = self._update_info(*self._call_airsim_info())
        return view_obs["image"]

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

    def close(self):
        requests.post("http://127.0.0.1:7575/release", timeout=5, json={"port": self.local_port})
        super().close()

    def _update_info(self, obs, info):
        self.info = json.loads(info)
        self._update_grid()
        obs = {"image": np.array(obs, dtype=np.uint8), "direction": self.agent_dir, "mission": self.mission}
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
        y = int(- int(self.info["position"]["x"]) / self.render_rate)
        x = int(int(self.info["position"]["y"]) / self.render_rate)
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

    def _call_airsim_restart(self, tried=2, failed=True):
        if tried <= 0:
            self.logger.error("Maximum retries reached for resetting AirSim window.")
            raise AirSimRestartFailed(f"Maximum retries reached for resetting AirSim window.")
        try:
            response = requests.get(f"http://127.0.0.1:{self.local_port}/restart", timeout=20)
            response.raise_for_status()
            failed = False
        except requests.RequestException or AirSimConnectionError as e:
            self.logger.error(f"Failed to reset AirSim window on try {tried}: {str(e)}")
            self._call_airsim_restart(tried - 1, failed)
        except Exception as e:
            raise e
        finally:
            return failed

    def _call_airsim_reset(self):
        try:
            reset_response = requests.post(f"http://127.0.0.1:{self.local_port}/reset", timeout=5,
                                           json={"map": self.to_json()})
            reset_response.raise_for_status()
        except Exception as e:
            self.logger.error(f"Failed to reset environment state: {str(e)}")
            raise AirSimConnectionError(f"Failed to reset environment state: {str(e)}")

    def _call_airsim_step(self, action):
        try:
            response = requests.post(f"http://127.0.0.1:{self.local_port}/step", timeout=5,
                                     json={"action": int(action)})
            response.raise_for_status()
            data = response.json()
            return np.array(data.get("obs")), data.get("info")
        except Exception as e:
            self.logger.error(f"Request to AirSim step failed: {str(e)}")
            raise AirSimConnectionError()

    def _call_airsim_info(self):
        try:
            response = requests.get(f"http://127.0.0.1:{self.local_port}/info", timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("obs"), data.get("info")
        except Exception as e:
            self.logger.error(f"Failed to retrieve information from AirSim: {str(e)}")
            raise AirSimResponseError(f"Failed to retrieve information from AirSim: {str(e)}")

    def _get_fail(self):
        if self.battery <= 0:
            return True
        else:
            if self.info["position"]["z"] > 100 or self.agent_pos[0] == 29 or self.agent_pos[0] == 0 or \
                    self.agent_pos[1] == 29 or self.agent_pos[1] == 0:
                return True
            else:
                return False

    def _check_airsim(self):
        try:
            response = requests.get(f"http://127.0.0.1:{self.local_port}/ping", timeout=5)
            response.raise_for_status()
            if not response.status_code == 200:
                self.logger.error(f"AirSim ping failed")
                raise AirSimConnectionError("AirSim ping failed.")
        except Exception as e:
            self.logger.error(f"Request to AirSim ping: {str(e)}")
            self._set_local_port_died()
            # if self._call_airsim_restart(5):
            #     raise AirSimRestartFailed()

    def _set_local_port_died(self):
        self.logger.info(f"Port {self.local_port} Died")
        requests.post(f"http://127.0.0.1:{self.manager_port}/set_died",
                      json={"port": self.local_port}, timeout=5)
        self._set_local_port()

    def _set_local_port(self):
        self.local_port = requests.get(f"http://127.0.0.1:{self.manager_port}/handshake", timeout=5).json()["port"]
        self.logger.info(f"Set New Local Port {self.local_port}")

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
