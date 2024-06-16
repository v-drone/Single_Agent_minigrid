from __future__ import annotations
from environments.AirSimException import AirSimResponseError, AirSimConnectionError, AirSimActionError
from environments.AirSimException import AirSimInfoError
from environments.CustomGrid import Grid
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import WorldObj, Goal
from minigrid.core.actions import IntEnum
from minigrid.core.mission import MissionSpace
from gymnasium.envs.registration import EnvSpec
from gymnasium import spaces
from typing import Any
import numpy as np
import base64
import logging
import requests
import random
import zlib
import math
import time
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
        log_filename = os.path.join(log_directory, f'{os.getpid()}.txt')
        logging.basicConfig(filename=log_filename,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.ERROR)
        self.logger = logging.getLogger(__name__)
        # Set local port
        self.manager_port = port
        self.local_port = None
        self.session = requests.Session()
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
        self.error_counter = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, retry=5):
        obs, _ = super().reset()
        try:
            self._ping_airsim()
            self.agent_dir = 3
            self.info = {}
            self.battery = self.full_battery
            self.walked = np.zeros(shape=[self.width, self.height], dtype=np.uint8)
            self._reset_airsim()
            self._info_airsim()
            self.error_counter = 0
            return self._trans_obs(self.info["obs"]), {}
        except (AirSimConnectionError, AirSimInfoError, Exception) as e:
            if e is AirSimInfoError or e is AirSimConnectionError:
                self.logger.error(e.message)
            else:
                self.logger.error(f"Unknown Exception during restarted at {str(self.local_port)}, {str(e)}")
            retry -= 1
            time.sleep(2)
            if retry <= 0:
                self.logger.error(f"Failed to reset environment after final retry, port: {self.local_port}")
                self.release()
                self._set_local_port(3)
                return self.reset(retry=5)
            else:
                self.logger.warning(f"Retrying to reset AirSim, {retry} retries left")
                return self.reset(retry=retry)

    def step(self, action):
        self.prev_pos = np.copy(self.agent_pos)
        self.battery -= 1
        self.step_count += 1
        try:
            self._step_airsim(action)
            self._info_airsim()
            self.walked[self.agent_pos[1]][self.agent_pos[0]] += 1
        except AirSimActionError as e:
            self.logger.error(e.message)
            time.sleep(5)
            self.error_counter += 1
        except AirSimInfoError as e:
            self.logger.error(e.message)
            time.sleep(5)
            self.error_counter += 1
        except Exception as e:
            self.logger.error(f"Unknown Exception during AirSim step call: {str(e)}")
            time.sleep(5)
            self.error_counter += 1
        obs = self._trans_obs(self.info["obs"])
        if self.error_counter < 5:
            terminated = bool(self.info.get("gear", False))
            truncated = self._get_fail()
        else:
            terminated = False
            truncated = True
        reward = self._reward() if terminated else 0
        return obs, reward, terminated, truncated, {}

    def render(self):
        return np.array(self.info["obs"], dtype=np.uint8)

    def to_json(self):
        json_return = {
            "mission": self.mission,
            "height": self.height,
            "width": self.width,
            "map": [],
            "start": self.start_pos,
        }

        # Reshape the grid to match the specified height and width
        grid_array = np.array(self.grid.grid).reshape([self.height, self.width])

        for y, row in enumerate(grid_array):
            for x, cell in enumerate(row):
                # Check if the cell is an instance of WorldObj or None
                if isinstance(cell, WorldObj) or cell is None:
                    # If it's None or a valid WorldObj, append the appropriate type
                    json_return["map"].append({
                        "x": x,
                        "y": y,
                        "type": mapper[cell.type] if cell is not None else 0
                    })
                else:
                    # If cell contains an invalid type, raise an error
                    raise ValueError(f"Unexpected object type at position {(x, y)}: {type(cell)}")
        return json_return

    def release(self):
        try:
            self.session.post("http://127.0.0.1:7575/release", timeout=10, json={"port": self.local_port})
        except Exception as e:
            self.logger.warning(f"Error while release AirSim: {e}")
        self._kill_airsim()

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
        goal = [random.randint(5, width - 5), random.randint(5, height - 5)]
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
            if self.info["position"]["z"] > 100:
                return True
            else:
                return False

    def _trans_obs(self, obs):
        return {"image": np.array(obs, dtype=np.uint8),
                "direction": self.agent_dir,
                "mission": self.mission}

    def _reset_airsim(self, retry=3):
        try:
            response = self.session.post(f"http://127.0.0.1:{self.local_port}/reset", timeout=10,
                                     json={"map": self.to_json()})
            response.raise_for_status()
            if response.status_code != 200:
                raise AirSimConnectionError(f"Failed to reset environment, port: {self.local_port}")
            else:
                return True
        except (requests.exceptions.RequestException, Exception) as e:
            self.logger.error(f"Error while resetting AirSim: {e}")
            retry -= 1
            if retry <= 0:
                raise AirSimConnectionError(f"Failed to reset environment after final retry, port: {self.local_port}")
            else:
                self.logger.warning(f"Retrying to reset AirSim, {retry} retries left")
                return self._reset_airsim(retry)

    def _step_airsim(self, action):
        try:
            response = self.session.post(f"http://127.0.0.1:{self.local_port}/step", timeout=10,
                                     json={"action": int(action)})
            response.raise_for_status()  # Check for HTTP errors

            # Check for non-200 status code, even though raise_for_status() above should handle it
            if response.status_code != 200:
                raise AirSimResponseError(f"Failed to do action environment, port: {self.local_port}")
        except (requests.exceptions.RequestException, AirSimResponseError, Exception) as e:
            raise AirSimActionError(f"An error occurred, port: {self.local_port}", e=e)

    def _info_airsim(self, retry=5):
        try:
            response = self.session.get(f"http://127.0.0.1:{self.local_port}/info", timeout=10)
            response.raise_for_status()
            info = response.json()
            info["obs"] = base64.b64decode(info["obs"])
            info["obs"] = zlib.decompress(info["obs"])
            info["obs"] = np.frombuffer(info["obs"], dtype=np.dtype(info['dtype'])).reshape(info["shape"])
            # valid
            self._trans_obs(info["obs"])
            self.info = info
            self._update_grid()
        except (requests.exceptions.RequestException, Exception) as e:
            if retry <= 0:
                raise AirSimInfoError(f"Failed to retrieve information after retries,"
                                      f" port: {self.local_port}", e=e)
            else:
                time.sleep(2 ** (6 - retry) / 1000)
                return self._info_airsim(retry - 1)

    def _ping_airsim(self, retry=2):
        try:
            response = self.session.get(f"http://127.0.0.1:{self.local_port}/ping", timeout=10)
            response.raise_for_status()
            return True
        except (requests.exceptions.RequestException, Exception) as e:
            self.logger.warning(f"Failed to ping AirSim due to {type(e).__name__}: {e}")
            if retry > 0:
                retry -= 1
                self.logger.warning(f"Retrying ping AirSim, {retry} retries left")
                return self._ping_airsim(retry)
            else:
                self._handle_failed_ping()
                raise AirSimConnectionError(f"Failed to ping AirSim after all retries, port: {self.local_port}")

    def _handle_failed_ping(self):
        """Handles actions to take when pinging fails after all retries."""
        try:
            self._set_local_port_died()
            self._kill_airsim()
        except Exception as e:
            self.logger.error(f"Exception occurred while handling failed ping: {e}")

    def _set_local_port(self, retry=5):
        try:
            response = self.session.get(f"http://127.0.0.1:{self.manager_port}/handshake", timeout=10)
            response.raise_for_status()
            if response.status_code == 200 and response.json().get("port", None) is not None:
                self.local_port = response.json()["port"]
                self.logger.info(f"Successfully set new local port: {self.local_port}")
                self._change_port()
                return True
        except (requests.exceptions.RequestException, Exception) as e:
            self.logger.warning(f"Failed to set local port, {retry} retries left: {e}")
            if retry > 0:
                retry -= 1
                self.logger.warning(f"Retrying ping AirSim, {retry} retries left")
                return self._set_local_port(retry)
            else:
                raise AirSimConnectionError("Failed to set local port after multiple retries")

    def _set_local_port_died(self, retry=2):
        if retry <= 0:
            self.logger.error(f"Failed to set died AirSim after multiple retries, port: {self.local_port}")
            raise AirSimResponseError(f"Failed to set died AirSim, port: {self.local_port}")
        try:
            response = self.session.post(f"http://127.0.0.1:{self.manager_port}/set_died",
                                     json={"port": self.local_port}, timeout=10)
            response.raise_for_status()  # Ensures we raise an HTTPError for bad responses
            return True
        except (requests.exceptions.RequestException, Exception) as e:
            self.logger.error(f"RequestException occurred while setting port as 'died': {e}")
            if retry > 0:
                retry -= 1
                self.logger.warning(f"Retrying to set port as 'died', {retry} retries left")
                self._set_local_port_died(retry)
            else:
                self.logger.error("Failed to set local port as 'died' after multiple retries")

    def _kill_airsim(self):
        try:
            response = self.session.get(f"http://127.0.0.1:{self.local_port}/exit", timeout=10)
            response.raise_for_status()
        except Exception as e:
            self.logger.warning(f"Exception occurred while trying to kill AirSim: {e}")

    def _change_port(self):
        self.session.close()
        self.session = requests.Session()
        self.logger.info(f"Session and local port updated to: {self.local_port}")

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
