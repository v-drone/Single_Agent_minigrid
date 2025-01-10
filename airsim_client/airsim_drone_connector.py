import time
import copy
import airsim
import numpy as np
from airsim_connector import BaseConnector
from airsim_utils import drone_state_to_dict


class DroneConnector(BaseConnector):
    def __init__(self, ip, port):
        super().__init__(ip, port)
        # Initialize MultirotorClient
        self.client = airsim.MultirotorClient(ip=self.ip, port=self.port)
        self.client.confirmConnection()
        # Define state properties specific to a drone
        self.client_state = {
            "position": np.zeros(3),
            "orientation": np.zeros(3),
            "prev_position": np.zeros(3),
            "prev_orientation": np.zeros(3),
            "collision": False,
            "damage": []
        }
        self.img_shape = 100
        self.speed = 3
        self.height = - 22

    def do_action(self, action):
        """
        Executes a discrete action for the drone environment:
        - 0: Move forward short
        - 1: Move forward long
        - 2, 3: Rotate right at increasing yaw rates
        - 4, 5: Rotate left at increasing yaw rates
        """
        if action == 0:
            # Move forward short
            self.client.moveByVelocityZBodyFrameAsync(self.speed, 0, self.height, 2).join()
        elif action == 1:
            # Move forward long
            self.client.moveByVelocityZBodyFrameAsync(self.speed * 2, 0, self.height, 2).join()
        elif action == 2 or action == 3:
            # Rotate right (45 degrees per second for action=2, 90 degrees per second for action=3)
            yaw_rate = 45 if action == 2 else 90
            self.client.rotateByYawRateAsync(yaw_rate, 1).join()
        elif action == 4 or action == 5:
            # Rotate left (-45 degrees per second for action=4, -90 degrees per second for action=5)
            yaw_rate = -45 if action == 4 else -90
            self.client.rotateByYawRateAsync(yaw_rate, 1).join()
        else:
            raise Exception("Invalid action for DroneConnector")

    def get_info(self):
        """
        Retrieves the drone state and updates self.client_state.
        """
        drone_state = drone_state_to_dict(self.client.getMultirotorState())
        # position
        self.client_state["prev_position"] = copy.copy(self.client_state["position"])
        self.client_state["position"] = drone_state["position"]
        self.client_state["prev_orientation"] = copy.copy(self.client_state["orientation"])
        self.client_state["orientation"] = drone_state["orientation"]
        # collision
        if 1.05 > self.client.simGetCameraInfo("0").fov > 0.95:
            self.client_state["collision"] = True
        else:
            self.client_state["collision"] = False
        # other info
        damage_value = int(self.client.simGetCameraInfo("0").pose.position.x_val)
        if damage_value not in self.client_state["damage"] and int(damage_value) != 0:
            self.client_state["damage"].append(damage_value)
        return self._get_obs(), self.client_state

    def _setup_client(self):
        """
        Resets the drone environment, arms the drone, and takes off to the specified height.
        """
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simGetSegmentationObjectID("")
        self.client.moveToZAsync(self.height, 1).join()
        time.sleep(2)
