import io
import time
import airsim
import numpy as np
from PIL import Image
from airsim import MultirotorClient, Pose, Vector3r, Quaternionr
from airsim_utils import drone_state_to_dict


class DroneConnector(object):
    def __init__(self, ip, port):
        self.client = MultirotorClient(ip, port=port, timeout_value=5)
        self.client.confirmConnection()
        self.client_state = {
            "position": np.zeros(3),
            "orientation": np.zeros(3),
            "prev_position": np.zeros(3),
            "prev_orientation": np.zeros(3),
            "collision": False,
        }
        self.img_shape = 100
        self.speed = 10
        self.height = -7

    def reset(self):
        self._setup_client()
        return self.get_info()

    def do_action(self, action):
        if action == 0:
            # Move forward at low speed
            self.client.moveByVelocityBodyFrameAsync(self.speed, 0, 0, 3).join()
        elif action == 1:
            # Move forward at medium speed in the current body frame
            self.client.moveByVelocityBodyFrameAsync(self.speed * 2, 0, 0, 3).join()
        elif action == 2:
            # Move forward at high speed in the current body frame
            self.client.moveByVelocityBodyFrameAsync(self.speed * 4, 0, 0, 3).join()
        elif 3 <= action <= 4:
            # Rotate in place at various yaw rates
            yaw_rate = 45 * (action - 2)  # Degrees per second
            self.client.rotateByYawRateAsync(yaw_rate, 1).join()
        elif 5 <= action <= 6:
            # Rotate in place at various yaw rates
            yaw_rate = - (180 - 45 * (action - 2))  # Degrees per second
            self.client.rotateByYawRateAsync(yaw_rate, 1).join()
        else:
            raise Exception("Invalid action")
        print(self.client.getMultirotorState())
        return self.get_info()

    def get_info(self):
        client_state = drone_state_to_dict(self.client.getMultirotorState())
        self.client_state["prev_position"] = self.client_state["position"]
        self.client_state["position"] = client_state["position"]
        self.client_state["prev_orientation"] = self.client_state["orientation"]
        self.client_state["orientation"] = client_state["orientation"]
        self.client_state["collision"] = self.client.simGetCollisionInfo().has_collided
        return self._get_obs(), self.client_state

    def ping(self):
        return self.client.ping()

    def _transform_obs(self, response):
        img = Image.open(io.BytesIO(response))
        img_resized = img.resize((self.img_shape, self.img_shape))
        img_resized = np.array(img_resized, dtype=np.uint8)
        return img_resized.reshape([self.img_shape, self.img_shape, 3])

    def _get_obs(self):
        responses = self.client.simGetImage('0', airsim.ImageType.Scene)
        image = self._transform_obs(responses)
        return image

    def _setup_client(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToZAsync(self.height, self.speed).join()
        time.sleep(0.01)


