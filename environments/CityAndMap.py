from __future__ import annotations
from environments.AirSimException import AirSimResponseError, AirSimConnectionError
from requests.exceptions import RequestException
from environments.CustomGrid import Grid
from environments.UAVAndMap import UAVWithMapEmpty
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
import time

mapper = {
    "lava": 1,
    "goal": 2,
}


class CityWithMapEmpty(UAVWithMapEmpty):
    # Enumeration of possible actions
    class Actions(IntEnum):
        slow_throttle = 0
        fast_throttle = 1
        steering_right_half = 2
        steering_right_full = 3
        steering_left_half = 4
        steering_left_full = 5
        brake = 6

    def __init__(self, size=100, max_steps=1000, battery=500,
                 agent_view_size=5, port=7575, camera=100,
                 render_mode="human", render_rate=3, **kwargs):
        super().__init__(size=size, max_steps=max_steps, agent_view_size=agent_view_size,
                         render_mode=render_mode, tile_size=kwargs.get("tile_size", 5))
        self.spec = EnvSpec("UAVWithMapEnv-v0", max_episode_steps=self.max_steps)
        self.actions = self.Actions
        self.action_space = spaces.Discrete(8, seed=np.random.randint(1000))
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
