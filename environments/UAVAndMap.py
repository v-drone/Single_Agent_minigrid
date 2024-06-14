from __future__ import annotations
from environments.AirSimException import AirSimResponseError, AirSimConnectionError
from requests.exceptions import RequestException
from environments.CustomGrid import Grid
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import Goal
from minigrid.core.actions import IntEnum
from minigrid.core.mission import MissionSpace
from gymnasium.envs.registration import EnvSpec
from gymnasium import spaces
from typing import Any
import traceback
import numpy as np
import logging
import requests
import random
import math
import json
import time
import ray
import os

mapper = {
    "lava": 1,
    "goal": 2,
}


class UAVWithMapEmpty(EmptyEnv):
    # Enumeration of possible actions
    class Actions(IntEnum):
        slow_throttle = 0
        fast_throttle = 1
        steering_right_half = 2
        steering_right_full = 3
        steering_left_half = 4
        steering_left_full = 5
        brake = 6

    def __init__(self, size=30, max_steps=400, battery=100,
                 agent_view_size=3, port=7575, camera=100,
                 render_mode="human", render_rate=3, **kwargs):
        super().__init__(size=size, max_steps=max_steps, agent_view_size=agent_view_size,
                         render_mode=render_mode, tile_size=kwargs.get("tile_size", 5))

        # Logging setup
        log_directory = '/home/seventheli/data/logging/'
        log_filename = os.path.join(log_directory, f'{ray.get_runtime_context().get_job_id()}.txt')
        logging.basicConfig(filename=log_filename,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        # Set local port
        self.manager_port = port
        self.local_port = None
        self._set_local_port()
        # Set basic infos
        self.size = size
        self.camera = camera
        self.full_battery = battery
        self.render_rate = render_rate
        # Set env spec
        self.spec = EnvSpec("UAVWithMapEnv-v0", max_episode_steps=self.max_steps)
        self.actions = self.Actions
        self.action_space = spaces.Discrete(8, seed=np.random.randint(1000))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=np.array([self.camera, self.camera, 3]), dtype="uint8"),
            "direction": spaces.Discrete(4),
            "mission": MissionSpace(mission_func=self._gen_mission)
        })
        # Set updatable infos
        self.info = {}
        self.goal = [0, 0]
        self.battery = battery
        self.prev_pos = np.array([self.size, self.size])
        self.walked = np.zeros(shape=[self.size, self.size], dtype=np.uint8)
        self.prev_transitions = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, retry=5):
        obs, _ = super().reset()
        self.action_space.seed(np.random.randint(1000))
        if retry <= 0:
            self.close()
            self._set_local_port(10)
            retry = 5
            self.logger.error(f"Restarted at {str(self.local_port)}")
        try:
            self._check_airsim()
            self.agent_dir = 3
            self.info = {}
            self.battery = self.full_battery
            self.walked = np.zeros(shape=[self.width, self.height], dtype=np.uint8)
            self._call_airsim_reset()
            return self._update_info(*self._call_airsim_info()), {}
        except Exception as e:
            _ = e
            retry -= 1
            time.sleep(2)
            return self.reset(retry=retry)

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
        except AirSimConnectionError or AirSimResponseError or RequestException as e:
            self.logger.error(f"Airsim/Network Exception during AirSim step call: ``{str(e)}``")
            time.sleep(10)
            self.reset()
            obs, reward, terminated, truncated, _ = self.prev_transitions
        except Exception as e:
            self.logger.error(f"Unknown Exception during AirSim step call: {str(e)}")
            time.sleep(10)
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
        requests.post("http://127.0.0.1:7575/release", timeout=10, json={"port": self.local_port})
        self.local_port = None
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

    def _get_fail(self):
        if self.battery <= 0:
            return True
        else:
            if self.info["position"]["z"] > 100 or self.agent_pos[0] == 29 or self.agent_pos[0] == 0 or \
                    self.agent_pos[1] == 29 or self.agent_pos[1] == 0:
                return True
            else:
                return False

    def _call_airsim_reset(self, retry=3):
        if retry <= 0:
            raise AirSimConnectionError(f"Failed to reset environment, {self.local_port}")
        try:
            response = requests.post(f"http://127.0.0.1:{self.local_port}/reset", timeout=10,
                                     json={"map": self.to_json()})
            if not response.status_code == 200:
                raise AirSimConnectionError(f"Failed to reset environment, {self.local_port}")
        except Exception as e:
            _ = e
            retry -= 1
            return self._call_airsim_reset(retry)

    def _call_airsim_step(self, action, retry=5):
        if retry <= 0:
            raise AirSimConnectionError(f"Failed to request to step, {self.local_port}")
        try:
            response = requests.post(f"http://127.0.0.1:{self.local_port}/step", timeout=10,
                                     json={"action": int(action)})
            data = response.json()
            return np.array(data.get("obs")), data.get("info")
        except Exception as e:
            _ = e
            retry -= 1
            return self._call_airsim_step(action, retry)

    def _call_airsim_info(self, retry=5):
        if retry <= 0:
            raise AirSimResponseError(f"Failed to retrieve information failed, {self.local_port}")
        try:
            response = requests.get(f"http://127.0.0.1:{self.local_port}/info", timeout=10)
            data = response.json()
            return data.get("obs"), data.get("info")
        except Exception as e:
            _ = e
            retry -= 1
            return self._call_airsim_info(retry)

    def _check_airsim(self, retry=2):
        if retry <= 0:
            try:
                self._set_local_port_died()
                raise AirSimConnectionError(f"Failed to ping AirSim, {self.local_port}")
            except Exception as e:
                _ = e
                raise AirSimConnectionError(f"Failed to ping AirSim, {self.local_port}, {traceback.format_exc()}")

        try:
            response = requests.get(f"http://127.0.0.1:{self.local_port}/ping", timeout=10)
            if not response.status_code == 200:
                raise AirSimConnectionError(f"Failed to reset environment, {self.local_port}")
        except Exception as e:
            _ = e
            retry -= 1
            return self._check_airsim(retry)

    def _set_local_port_died(self, retry=5):
        if retry <= 0:
            self.logger.error(f"Failed to set died AirSim after multiple retries, port: {self.local_port}")
            raise AirSimResponseError(f"Failed to set died AirSim, port: {self.local_port}")
        try:
            response = requests.post(f"http://127.0.0.1:{self.manager_port}/set_died",
                                     json={"port": self.local_port}, timeout=10)
            response.raise_for_status()  # Ensures we raise an HTTPError for bad responses
            self.logger.info(f"Successfully set port {self.local_port} as 'died'")
            return
        except requests.exceptions.HTTPError as he:
            self.logger.error(f"HTTP Error occurred while setting port as 'died': {he}")
        except requests.exceptions.ConnectionError as ce:
            self.logger.error(f"Connection Error occurred while setting port as 'died': {ce}")
        except requests.exceptions.Timeout as te:
            self.logger.error(f"Timeout occurred while setting port as 'died': {te}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while setting port as 'died': {e}")

        # Log retry and decrement retry count
        retry -= 1
        self.logger.info(f"Retrying to set port as 'died', {retry} retries left")
        self._set_local_port_died(retry)

    def _set_local_port(self, retry=5):
        if retry <= 0:
            self.logger.error("Failed to set local port after multiple retries")
            raise AirSimResponseError("Failed to set local port")
        try:
            response = requests.get(f"http://127.0.0.1:{self.manager_port}/handshake", timeout=10)
            response.raise_for_status()
            if response.status_code == 200 and response.json().get("port", None) is not None:
                self.local_port = response.json()["port"]
                self.logger.info(f"Successfully set new local port: {self.local_port}")
                return
        except requests.exceptions.HTTPError as he:
            self.logger.error(f"HTTP Error occurred: {he}")
        except requests.exceptions.ConnectionError as ce:
            self.logger.error(f"Connection Error occurred: {ce}")
        except requests.exceptions.Timeout as te:
            self.logger.error(f"Timeout occurred: {te}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")

        # Retry logic
        retry -= 1
        self.logger.info(f"Retrying to set local port, {retry} retries left")
        self._set_local_port(retry)

    def _kill_airsim(self):
        try:
            response = requests.get(f"http://127.0.0.1:{self.local_port}/exit", timeout=10)
            response.raise_for_status()
            if response.status_code == 200:
                self.logger.info("Successfully requested server to terminate.")
        except Exception as e:
            self.logger.warning(f"Exception occurred while trying to kill AirSim: {e}")

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
