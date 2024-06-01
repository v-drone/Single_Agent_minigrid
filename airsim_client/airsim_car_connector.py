import io
import time
import airsim
import numpy as np
from PIL import Image
from airsim import CarClient


class CarConnector(object):
    def __init__(self, ip, port):
        self.car = CarClient(ip, port=port, timeout_value=5)
        self.car.confirmConnection()
        self.car_controls = airsim.CarControls()
        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }
        self.img_shape = 100

    def reset(self):
        self._setup_car()
        self.do_action(-1)
        return self.get_info()

    def do_action(self, action):
        if action == 0:
            self.car_controls.throttle = 0.05
            self.car.setCarControls(self.car_controls)
            time.sleep(0.3)
        elif action == 1:
            self.car_controls.brake = -0.2
            self.car_controls.throttle = 0
            self.car.setCarControls(self.car_controls)
            time.sleep(0.3)
        elif action == 2:
            self.car_controls.brake = 0
            self.car_controls.steering = 0.5
            self.car.setCarControls(self.car_controls)
            time.sleep(0.3)
        elif action == 3:
            self.car_controls.brake = 0
            self.car_controls.steering = 1
            self.car.setCarControls(self.car_controls)
            time.sleep(0.3)
        elif action == 4:
            self.car_controls.brake = 0
            self.car_controls.steering = -0.5
            self.car.setCarControls(self.car_controls)
            time.sleep(0.3)
        elif action == 5:
            self.car_controls.brake = 0
            self.car_controls.steering = -1
            self.car.setCarControls(self.car_controls)
            time.sleep(0.3)
        elif action == 6:
            self.car_controls.throttle = 0
            self.car_controls.steering = 0
            self.car.setCarControls(self.car_controls)
            time.sleep(0.3)
        else:
            time.sleep(0.3)

        return self.get_info()

    def get_info(self):
        return self._get_obs(), self.car.getCarState()

    def ping(self):
        return self.car.ping()

    def _get_obs(self):
        responses = self.car.simGetImage('0', airsim.ImageType.Scene)
        image = self._transform_obs(responses)
        self.car_state = self.car.getCarState()
        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image

    def _transform_obs(self, response):
        img = Image.open(io.BytesIO(response))
        img_resized = img.resize((self.img_shape, self.img_shape))
        img_resized = np.array(img_resized, dtype=np.uint8)
        return img_resized.reshape([self.img_shape, self.img_shape, 3])

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        self.car_controls.throttle = 0
        self.car_controls.steering = 0
        self.car.setCarControls(self.car_controls)
        time.sleep(0.01)

    def _get_ex_reward(self):
        pass
