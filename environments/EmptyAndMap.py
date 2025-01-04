from environments.AirSimException import AirSimInfoError, AirSimRetryError
from environments.AirSimClient import AirSimClient
from environments.CustomGrid import Grid
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import WorldObj, Goal
from minigrid.core.actions import IntEnum
from minigrid.core.mission import MissionSpace
from gymnasium import spaces
from typing import Any, Optional
import traceback
import copy
import uuid
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
        for attempt in range(retries):
            try:
                with requests.Session() as session:
                    return operation(session, *args, **kwargs)
            except requests.exceptions.ConnectionError as e:
                logging.warning(f"ConnectionError on attempt {attempt + 1}/{retries}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
            except Exception as e:
                logging.warning(f"Exception on attempt {attempt + 1}/{retries}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
        raise AirSimRetryError(f"Operation failed after {retries} retries due to issues")


class EmptyWithMapEmpty(EmptyEnv):
    # Enumeration of possible actions
    class Actions(IntEnum):
        slow_throttle = 0
        fast_throttle = 1
        steering_right_half = 2
        steering_right_full = 3
        steering_left_half = 4
        steering_left_full = 5

    def __init__(self, size=30, max_steps=400, battery=100,
                 agent_view_size=3, port=7575, camera=100,
                 render_mode="human", render_rate=3, remote_ip="192.168.3.10", **kwargs):
        super().__init__(size=size, max_steps=max_steps, agent_view_size=agent_view_size,
                         render_mode=render_mode, tile_size=kwargs.get("render_rate", 5))

        # Generate a unique environment ID (either from config or a random uuid)
        self.env_id = str(uuid.uuid4())

        # Create a logger name that includes the env_id
        logger_name = f"gym_logger_{self.env_id}"
        self.gym_logger = logging.getLogger(logger_name)
        self.gym_logger.setLevel(logging.DEBUG)

        # Construct a unique file path
        log_directory = "C:/Users/seven/Documents/UAV/Single_Agent_minigrid/Logs/"
        os.makedirs(log_directory, exist_ok=True)
        file_path = os.path.join(log_directory, f"gym_{self.env_id}.log")

        # Only add a FileHandler if there is none
        if not self.gym_logger.handlers:
            fh1 = logging.FileHandler(file_path, mode='a')
            fh1.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s [GYM %(name)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            fh1.setFormatter(formatter)
            self.gym_logger.addHandler(fh1)

        self.gym_logger.info(f"environment init, env_id={self.env_id}")

        # Similarly for AirSimClient
        airsim_logger_name = f"airsim_logger_{self.env_id}"
        self.airsim_logger = logging.getLogger(airsim_logger_name)
        self.airsim_logger.setLevel(logging.DEBUG)

        airsim_file_path = os.path.join(log_directory, f"airsim_{self.env_id}.log")
        if not self.airsim_logger.handlers:
            fh2 = logging.FileHandler(airsim_file_path, mode='a')
            fh2.setLevel(logging.DEBUG)
            formatter2 = logging.Formatter(
                '%(asctime)s [AIRSIM %(name)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            fh2.setFormatter(formatter2)
            self.airsim_logger.addHandler(fh2)

        # Initialize AirSimClient
        self.airsim_client = AirSimClient(manager_port=port, remote_ip=remote_ip, logger=self.airsim_logger)

        # Set local port and session
        self.manager_port = port
        self.local_port = None
        self.remote_ip = remote_ip

        # Set basic infos
        self.size = size
        self.camera = camera
        self.full_battery = battery
        self.render_rate = render_rate

        self.actions = self.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.camera, self.camera, 3),
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

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed, options=options)
        try:
            if self.airsim_client.local_port is None:
                self.airsim_client.handshake()
            self.airsim_client.ping()
            self.reset_internal_state()
            self.airsim_client.reset_env(self.to_json())
            self.info_airsim()
            obs = self._trans_obs(self.info["obs"])
            self.prev_obs = copy.copy(obs)
            self.error_counter = 0
            return obs, {}
        except (AirSimRetryError, AirSimInfoError) as e:
            self.gym_logger.warning(f"Error during reset: {e}")
            time.sleep(2)
            self.error_counter += 1
            if self.error_counter >= 2:
                self._handle_port_error()
            return self.reset()
        except Exception as e:
            self.gym_logger.warning(traceback.format_exc())
            self.gym_logger.warning(f"Unexpected error during reset: {e}")
            self.error_counter += 1
            time.sleep(2)
            if self.error_counter >= 2:
                self._handle_port_error()
            return self.reset()

    def step(self, action):
        self.prev_pos = np.copy(self.agent_pos)
        self.battery -= 1
        self.step_count += 1
        try:
            self.airsim_client.step_env(action=action)
            self.info_airsim()
            self.walked[self.agent_pos[1]][self.agent_pos[0]] += 1
            obs = self._trans_obs(self.info["obs"])
            terminated, truncated = self._check_status()
            reward = self._reward()
            self.prev_obs = obs
            return obs, reward, terminated, truncated, {}
        except AirSimRetryError:
            self.gym_logger.warning("Network-related error during step, resetting to previous position.")
            return self.prev_obs, 0, False, True, {"error": "Environment reset due to network error"}
        except AirSimInfoError as e:
            self.gym_logger.warning(f"Failed to parse information during step: {e}")
            time.sleep(5)
            self.error_counter += 1
            return self.prev_obs, 0, False, True, {"error": "Environment reset due to info parsing error"}
        except Exception as e:
            self.gym_logger.warning()
            self.gym_logger.warning(f"Unexpected error during step: {e}")
            self.error_counter += 1
            time.sleep(2)
            if self.error_counter >= 2:
                self._handle_port_error()
            return self.reset()

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
        self.airsim_client.release()

    def reset_internal_state(self):
        self.agent_dir = 3
        self.info = {}
        self.battery = self.full_battery
        self.walked = np.zeros(shape=[self.width, self.height], dtype=np.uint8)

    def close(self):
        self.release()

    def _reward(self) -> float:
        terminated, truncated = self._check_status()
        if terminated:
            return super()._reward()
        elif truncated:
            return -0.5
        else:
            return 0

    def _check_status(self):
        # check network
        if self.error_counter >= 5:
            # truncated
            return False, True

        # check success
        if self._check_success():
            return True, False

        # check fail
        if self._check_fail():
            return False, True

        return False, False

    def _check_success(self):
        return bool(self.info.get("gear", False))

    def _check_fail(self):
        if self.battery <= 0:
            return True
        if self.info["position"]["z"] >= -1.49:
            return True
        return False

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
        self.airsim_client.reset_env(self.to_json(), retry=retry)

    def _step_airsim(self, action, retry=3):
        self.airsim_client.step_env(action=action, retry=retry)

    def info_airsim(self):
        info = self.airsim_client.get_env_info()
        info["obs"] = self.airsim_client.decode_observation(info)
        self.info = info
        self._update_grid()

    def _ping_airsim(self, retry=2):
        self.airsim_client.ping(retry=retry)

    def _handle_port_error(self):
        if self.airsim_client.local_port is not None:
            self.gym_logger.error(f"Port error occurred, marking port as dead and resetting, port: {self.airsim_client.local_port}")
            self.airsim_client.set_local_port_died()
            self.airsim_client.kill_airsim()
            self.airsim_client.local_port = None

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
