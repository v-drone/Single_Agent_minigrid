from environments.AirSimException import AirSimConnectionError, AirSimInfoError, AirSimRetryError
from environments.CustomGrid import Grid
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import WorldObj, Goal
from minigrid.core.actions import IntEnum
from minigrid.core.mission import MissionSpace
from gymnasium.envs.registration import EnvSpec
from gymnasium import spaces
from typing import Any, Optional
import copy
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


class RetryOperation:
    """Utility to handle retry logic."""

    @staticmethod
    def execute(operation, retries=3, delay=2, *args, **kwargs):
        print(operation)
        for attempt in range(retries):
            try:
                with requests.Session() as session:
                    return operation(session, *args, **kwargs)
            except AirSimConnectionError as e:
                logging.warning(f"Attempt {attempt + 1} failed due to network error: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
        raise AirSimRetryError(f"Operation failed after {retries} retries due to network issues")


class EmptyWithMapEmpty(EmptyEnv):
    # Enumeration of possible actions
    class Actions(IntEnum):
        slow_throttle = 0
        fast_throttle = 1
        steering_right_half = 2
        steering_right_full = 3
        steering_left_half = 4
        steering_left_full = 5
        mid_break = 6

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

        # Set local port and session
        self.manager_port = port
        self.local_port = None

        # Set basic infos
        self.size = size
        self.camera = camera
        self.full_battery = battery
        self.render_rate = render_rate

        # Set env spec
        self.spec = EnvSpec("UAVWithMapEnv-v0", max_episode_steps=self.max_steps)
        self.actions = self.Actions
        self.action_space = spaces.Discrete(7, seed=np.random.randint(1000))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=np.array([self.camera, self.camera, 3]),
                                dtype=np.uint8),
            "direction": spaces.Discrete(4),
            "mission": MissionSpace(mission_func=self._gen_mission)
        })

        # Set updatable infos
        self.info = {}
        self.goal = [0, 0]
        self.battery = battery
        self.prev_pos = np.array([self.size, self.size])
        self.prev_obs = None
        self.walked = np.zeros(shape=[self.size, self.size], dtype=np.uint8)
        self.error_counter = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None, retry=5):
        obs, _ = super().reset()
        try:
            if self.local_port is None:
                self._set_local_port(5)
            self._ping_airsim()
            self._reset_internal_state()
            self._reset_airsim()
            self._info_airsim()
            self.error_counter = 0
            obs = self._trans_obs(self.info["obs"])
            self.prev_obs = copy.copy(obs)
            return obs, {}
        except (AirSimRetryError, AirSimInfoError):
            if self.error_counter >= 5:
                self._handle_port_error()
                return self.reset(retry=5)
        except Exception as e:
            self.logger.warning(f"Error occurred during reset: {e}")
            retry -= 1
            time.sleep(2)
            if retry <= 0:
                self.logger.error(f"Failed to reset environment after final retry, port: {self.local_port}")
                self._handle_port_error()
                return self.reset(retry=5)
            else:
                self.logger.warning(f"Retrying to reset AirSim, {retry} retries left")
                return self.reset(retry=retry)

    def step(self, action):
        self.prev_pos = np.copy(self.agent_pos)
        self.battery -= 1
        self.step_count += 1
        try:
            self._step_airsim(action=action)
            self._info_airsim()
            self.walked[self.agent_pos[1]][self.agent_pos[0]] += 1
            obs = self._trans_obs(self.info["obs"])
            terminated, truncated = self._check_status()
            reward = self._reward()
            self.prev_obs = obs
            return obs, reward, terminated, truncated, {}
        except AirSimRetryError:
            self.logger.warning("Network-related error during step, resetting to previous position.")
            return self.prev_obs, 0, False, True, {"error": "Environment reset due to network error"}
        except AirSimInfoError as e:
            self.logger.warning(f"Failed to parse information during step: {e.message}")
            time.sleep(5)
            self.error_counter += 1
            return self.prev_obs, 0, False, True, {"error": "Environment reset due to info parsing error"}

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
                if isinstance(cell, WorldObj) or cell is None:
                    json_return["map"].append({
                        "x": x,
                        "y": y,
                        "type": mapper[cell.type] if cell is not None else 0
                    })
                else:
                    raise ValueError(f"Unexpected object type at position {(x, y)}: {type(cell)}")
        return json_return

    def release(self):
        try:
            with requests.Session() as session:
                session.post("http://127.0.0.1:{port}/release".format(port=self.manager_port)
                             , timeout=10, json={"port": self.local_port})
            self.local_port = None
        except Exception as e:
            self.logger.warning(f"Error while releasing AirSim: {e}")

    def _reset_internal_state(self):
        self.agent_dir = 3
        self.info = {}
        self.battery = self.full_battery
        self.walked = np.zeros(shape=[self.width, self.height], dtype=np.uint8)

    def _reward(self) -> float:
        terminated, _ = self._check_status()
        return super()._reward() if terminated else 0

    def _check_status(self):
        if self.error_counter < 5:
            terminated = self._get_success()
            truncated = self._get_fail()
        else:
            terminated = False
            truncated = True
        return terminated, truncated

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

    def _get_success(self):
        if self.error_counter < 5:
            terminated = bool(self.info.get("gear", False))
        else:
            terminated = False
        return terminated

    def _trans_obs(self, obs):
        try:
            return {
                "image": np.array(obs, dtype=np.uint8),
                "direction": self.agent_dir,
                "mission": MissionSpace(mission_func=self._gen_mission)
            }
        except Exception as e:
            raise AirSimInfoError(f"Failed to parse info response: {e}")

    def _reset_airsim(self, retry=3):
        def reset_operation(session):
            response = session.post(f"http://127.0.0.1:{self.local_port}/reset",
                                    timeout=10, json={"map": self.to_json()})
            response.raise_for_status()
            if response.status_code != 200:
                raise AirSimConnectionError(f"Failed to reset environment, port: {self.local_port}")
            return True

        RetryOperation.execute(reset_operation, retries=retry)

    def _step_airsim(self, action, retry=3):
        def step_operation(session):
            response = session.post(f"http://127.0.0.1:{self.local_port}/step",
                                    timeout=10, json={"action": int(action)})
            response.raise_for_status()
            if response.status_code != 200:
                raise AirSimConnectionError(f"Failed to perform action in environment, port: {self.local_port}")
            return True

        RetryOperation.execute(step_operation, retries=retry)

    def _info_airsim(self, retry=5):
        def info_operation(session):
            response = session.get(f"http://127.0.0.1:{self.local_port}/info", timeout=10)
            response.raise_for_status()
            if response.status_code != 200:
                raise AirSimConnectionError(f"Failed to get info, port: {self.local_port}")
            try:
                info = response.json()
                info["obs"] = self._decode_observation(info)
                self.info = info
                self._update_grid()
            except Exception as e:
                raise AirSimInfoError(f"Failed to parse info response: {e}")
            return True

        RetryOperation.execute(info_operation, retries=retry)

    def _ping_airsim(self, retry=2):
        def ping_operation(session):
            response = session.get(f"http://127.0.0.1:{self.local_port}/ping", timeout=10)
            response.raise_for_status()
            if response.status_code != 200:
                raise AirSimConnectionError(f"Failed to ping, port: {self.local_port}")
            return True

        RetryOperation.execute(ping_operation, retries=retry)

    def _set_local_port(self, retry=5):
        def set_local_port_operation(session):
            response = session.get(f"http://127.0.0.1:{self.manager_port}/handshake", timeout=10)
            response.raise_for_status()
            if response.status_code == 200 and response.json().get("port", None) is not None:
                self.local_port = response.json()["port"]
                self.logger.info(f"Successfully set new local port: {self.local_port}")
                return True
            else:
                raise AirSimConnectionError("Failed to receive a valid port during handshake.")

        RetryOperation.execute(set_local_port_operation, retries=retry)

    def _set_local_port_died(self, retry=3):
        def set_port_died_operation(session):
            response = session.post(f"http://127.0.0.1:{self.manager_port}/set_died",
                                    json={"port": self.local_port}, timeout=10)
            response.raise_for_status()
            return True

        try:
            RetryOperation.execute(set_port_died_operation, retries=retry)
        except Exception as e:
            self.logger.warning(f"Error while setting local port as dead: {e}")


    def _kill_airsim(self, retry=3):
        def kill_operation(session):
            response = session.get(f"http://127.0.0.1:{self.local_port}/exit", timeout=10)
            response.raise_for_status()
            return True

        RetryOperation.execute(kill_operation, retries=retry)

    def _handle_port_error(self):
        self.logger.error(f"Port error occurred, marking port as dead and resetting, port: {self.local_port}")
        self._set_local_port_died()
        self._kill_airsim()
        self.local_port = None

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    @staticmethod
    def _decode_observation(info):
        try:
            obs = base64.b64decode(info["obs"])
            obs = zlib.decompress(obs)
            obs = np.frombuffer(obs, dtype=np.dtype(info['dtype'])).reshape(info["shape"])
            return obs
        except Exception as e:
            raise AirSimInfoError(f"Failed to parse info response: {e}")

    def close(self):
        self.release()