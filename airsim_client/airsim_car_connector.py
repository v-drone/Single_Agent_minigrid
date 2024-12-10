import io
import time
import airsim
import numpy as np
from PIL import Image
from airsim import CarClient
from airsim_utils import car_state_to_dict


class CarConnector(object):
    def __init__(self, ip, port):
        self.client = CarClient(ip, port=port, timeout_value=5)
        self.client.confirmConnection()
        self.client_controls = airsim.CarControls()
        self.client_state = {
            "position": np.zeros(3),
            "orientation": np.zeros(3),
            "collision": False,
            "speed": 0,
            "gear": False
        }
        self.img_shape = 100

    def reset(self):
        self._setup_client()
        return self.get_info()

    def do_action(self, action):
        if action == 0:
            # slow front throttle
            self.client_controls.brake = 0
            self.client_controls.throttle = 0.5
            self.client_controls.steering = 0
            self.client.setCarControls(self.client_controls)
            time.sleep(np.random.randint(25, 50) / 100)
        elif action == 1:
            # faster front throttle
            self.client_controls.brake = 0
            self.client_controls.throttle = 1
            self.client_controls.steering = 0
            self.client.setCarControls(self.client_controls)
            time.sleep(np.random.randint(25, 50) / 100)
        elif action == 2:
            # 50% brake left steering
            self.client_controls.brake = 0
            self.client_controls.throttle = 0.5
            self.client_controls.steering = 0.25
            self.client.setCarControls(self.client_controls)
            time.sleep(np.random.randint(25, 50) / 100)
        elif action == 3:
            # 100% brake left steering
            self.client_controls.brake = 0
            self.client_controls.throttle = 0.5
            self.client_controls.steering = 0.5
            self.client.setCarControls(self.client_controls)
            time.sleep(np.random.randint(25, 50) / 100)
        elif action == 4:
            # 50% brake right steering
            self.client_controls.brake = 0
            self.client_controls.throttle = 0.5
            self.client_controls.steering = -0.25
            self.client.setCarControls(self.client_controls)
            time.sleep(np.random.randint(25, 50) / 100)
        elif action == 5:
            # 100% brake right steering
            self.client_controls.brake = 0
            self.client_controls.throttle = 0.5
            self.client_controls.steering = -0.5
            self.client.setCarControls(self.client_controls)
            time.sleep(np.random.randint(25, 50) / 100)
        else:
            # brake
            self.client_controls.brake = 1
            self.client_controls.throttle = 0
            self.client.setCarControls(self.client_controls)
            time.sleep(np.random.randint(25, 50) / 100)

        self.client_controls.brake = 1
        self.client_controls.throttle = 0
        self.client_controls.steering = 0
        self.client.setCarControls(self.client_controls)
        time.sleep(0.2)

        return self.get_info()

    def get_info(self):
        client_state = car_state_to_dict(self.client.getCarState())
        self.client_state["position"] = client_state["position"]
        self.client_state["orientation"] = client_state["orientation"]
        self.client_state["collision"] = self.client.simGetCollisionInfo().has_collided
        self.client_state["collision"] = client_state["speed"]
        self.client_state["gear"] = client_state["gear"]
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
        self.client_controls.throttle = 0
        self.client_controls.steering = 0
        self.client.setCarControls(self.client_controls)
        time.sleep(0.01)

# connector = CarConnector("127.0.0.1", 41530)
# print(connector.get_info())