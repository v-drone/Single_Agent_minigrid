import time

from airsim import CarClient
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import os
import airsim
import subprocess


class AirSimEnv(gym.Env):
    def __init__(self):
        super(AirSimEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.unity_process = None
        self.client = None

    def start_unity_environment(self):
        # unity_executable_path = "/home/seventheli/data/Airsim/Blocks/Blocks.sh"
        #
        # batch_mode_args = ['-batchmode', '-nographics']
        #
        # if os.path.isfile(unity_executable_path) and os.access(unity_executable_path, os.X_OK):
        #     command = [unity_executable_path] + batch_mode_args
        #     self.unity_process = subprocess.Popen(command)
        #     print("Unity process started:", self.unity_process)
        # else:
        #     raise Exception(f"Unity executable not found or not executable: {unity_executable_path}")
        # time.sleep(10)
        # print("Process is running:", self.unity_process.poll() is None)

        self.client = CarClient()
        self.client.confirmConnection()

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        if self.unity_process:
            self.unity_process.terminate()
            self.unity_process = None


env = os.environ.copy()
env["DISPLAY"] = ":99"

client = AirSimEnv()
client.start_unity_environment()
