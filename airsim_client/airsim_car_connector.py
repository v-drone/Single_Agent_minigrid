import time
import airsim
import numpy as np
from airsim_connector import BaseConnector
from airsim_utils import car_state_to_dict


class CarConnector(BaseConnector):
    def __init__(self, ip, port):
        super().__init__(ip, port)
        # Initialize CarClient
        self.client = airsim.CarClient(ip=self.ip, port=self.port, timeout_value=self.timeout_value)
        self.client_controls = airsim.CarControls()

        # Define state properties specific to a car
        self.client_state = {
            "position": np.zeros(3),
            "orientation": np.zeros(3),
            "collision": False,
            "speed": 0,
            "gear": False
        }

    def do_action(self, action):
        """
        Executes a discrete action for the car environment (e.g., accelerate, steer).
        """
        if action == 0:
            # Slow forward throttle
            self.client_controls.throttle = 0.5
            self.client_controls.steering = 0
        elif action == 1:
            # Faster forward throttle
            self.client_controls.throttle = 1
            self.client_controls.steering = 0
        elif action == 2:
            # Left steering with throttle
            self.client_controls.throttle = 0.33
            self.client_controls.steering = 0.5
        elif action == 3:
            # Sharp left steering
            self.client_controls.throttle = 0.33
            self.client_controls.steering = 1
        elif action == 4:
            # Right steering with throttle
            self.client_controls.throttle = 0.33
            self.client_controls.steering = -0.5
        elif action == 5:
            # Sharp right steering
            self.client_controls.throttle = 0.33
            self.client_controls.steering = -1
        else:
            self.client_controls.throttle = 0
            self.client_controls.steering = 0

        # Apply the current action
        self.client.setCarControls(self.client_controls)
        time.sleep(0.02)

        # Attempt to bring the car to a near stop to simulate one-step action
        self._ensure_stopped()

        # Reset controls after the action
        self.client_controls.steering = 0
        self.client_controls.throttle = 0
        self.client_controls.brake = 0
        self.client.setCarControls(self.client_controls)
        time.sleep(0.01)

    def get_info(self):
        """
        Retrieves the car state and updates self.client_state.
        """
        car_state = car_state_to_dict(self.client.getCarState())
        self.client_state["position"] = car_state["position"]
        self.client_state["orientation"] = car_state["orientation"]
        self.client_state["collision"] = self.client.simGetCollisionInfo().has_collided
        self.client_state["speed"] = car_state["speed"]
        self.client_state["gear"] = car_state["gear"]

    def _setup_client(self):
        """
        Resets the car environment and returns it to the initial state.
        """
        self.client.reset()
        self.client.enableApiControl(False)
        self.client.enableApiControl(True)
        self.client.setCarControls(self.client_controls)
        time.sleep(0.5)

    def _ensure_stopped(self):
        """
        After each action, attempt to reduce the car's speed to near 0.
        """
        for _ in range(10):
            self.get_info()
            speed = self.client_state.speed
            if 1 >= speed > 0:
                return
            elif speed < 0:
                # The car is moving backward, apply some forward throttle
                self.client_controls.brake = 0
                self.client_controls.throttle = 0.2
            else:
                # The car is moving forward too fast, apply brake
                self.client_controls.brake = -0.1
                self.client_controls.throttle = 0
            self.client.setCarControls(self.client_controls)
            time.sleep(0.01)
