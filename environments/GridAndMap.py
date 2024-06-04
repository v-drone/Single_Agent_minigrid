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
import logging
import random
import cv2

mapper = {
    "lava": 1,
    "goal": 2,
}


class GridWithMapEmpty(EmptyEnv):
    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self, size=30, max_steps=400, battery=100,
                 agent_view_size=3, port=7575, camera=100,
                 render_mode="human", render_rate=3, **kwargs):
        super().__init__(size=size, max_steps=max_steps, agent_view_size=agent_view_size,
                         render_mode=render_mode, tile_size=kwargs.get("tile_size", 5))
        self.spec = EnvSpec("UAVWithMapEnv-v0", max_episode_steps=self.max_steps)
        self.actions = self.Actions
        self.action_space = spaces.Discrete(3, seed=np.random.randint(1000))
        self.size = size
        self.full_battery = battery
        self.battery = battery
        self.info = {}
        self.prev_pos = None
        self.camera = camera
        self.step_count = 0
        self.walked = np.zeros(shape=[self.size, self.size], dtype=np.uint8)

        # Logging setup
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initialized UAVWithMapEmpty")

        self.manager_port = port
        self.local_port = None
        self.info = {
            "yaw_degrees" : 0,
            "speed": 1,
        }
        self.prev_transitions = None
        self.render_rate = render_rate
        self.goal = [0, 0]
        image_space = spaces.Box(low=0, high=255, shape=np.array([self.camera, self.camera, 3]), dtype="uint8")
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.observation_space = spaces.Dict(
            {"image": image_space, "direction": spaces.Discrete(4), "mission": mission_space})

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, _ = super().reset()
        obs["image"] = self.get_obs()
        self.battery = self.full_battery
        self.prev_pos = np.copy(self.agent_pos)
        self.walked = np.zeros(shape=[self.width, self.height], dtype=np.uint8)
        return obs, {}

    def get_obs(self):
        obs = self.get_frame(highlight=False, tile_size=30, agent_pov=True)
        return cv2.resize(
            obs, (self.camera, self.camera), interpolation=cv2.INTER_AREA
        )

    def step(self, action):
        # Record the agent's current position before executing the action
        self.prev_pos = np.copy(self.agent_pos)
        obs, reward, terminated, truncated, info = super().step(action)
        obs["image"] = self.get_obs()
        if self.agent_pos == self.start_pos:
            self.battery = self.full_battery
        else:
            self.battery -= 1

        if self.battery <= 0:
            truncated = True

        self.walked[self.agent_pos[1]][self.agent_pos[0]] += 1

        return obs, reward, terminated, truncated, info

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

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
