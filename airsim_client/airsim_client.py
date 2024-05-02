import abc

import requests
from airsim import CarClient
from gymnasium import spaces
import numpy as np
import os


class AirSimEnv(abc.ABC):
    def __init__(self):
        super(AirSimEnv, self).__init__()
        self.remote_ip = "192.168.0.101:5000"
        self.unity_process = None
        self.client = None
        self.client_id = None

    def restart_unity_environment(self):
        client_info = requests.get(self.remote_ip + "/get_client").json()
        client_port = client_info["port"]
        self.client_id = client_info["client_id"]
        self.client = CarClient(self.remote_ip, port=client_port)
        self.client.confirmConnection()

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
