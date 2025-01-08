import io
import airsim
import numpy as np
from PIL import Image

class BaseConnector(object):
    def __init__(self, ip, port, timeout_value=5, img_shape=100):
        """
        Parent class constructor.
        """
        # The actual client (CarClient or MultirotorClient) should be initialized in the child class.
        self.client = None
        self.ip = ip
        self.port = port
        self.timeout_value = timeout_value

        # Common state dictionary
        self.client_state = {}
        self.img_shape = img_shape

    def ping(self):
        """
        Common ping method.
        """
        return self.client.ping()

    def reset(self):
        """
        Common reset method.
        Calls _setup_client(), then returns get_info().
        """
        self._setup_client()
        return self.get_info()

    def get_info(self):
        """
        Retrieves observation (image) and client state.
        Child classes should implement _get_state() to update self.client_state properly.
        """
        self._get_state()  # Update self.client_state
        return self._get_obs(), self.client_state

    def do_action(self, action):
        """
        Abstract method for taking an action.
        Different child classes (Car/Drone) have different logic, so override in child classes.
        """
        raise NotImplementedError("Please implement do_action in the child class.")

    def _setup_client(self):
        """
        Abstract method for resetting the environment.
        Each child class must implement its own client setup steps.
        """
        raise NotImplementedError("Please implement _setup_client in the child class.")

    def _get_state(self):
        """
        Abstract method for retrieving the state from the environment.
        Must update self.client_state in the child class.
        """
        raise NotImplementedError("Please implement _get_state in the child class.")

    def _get_obs(self):
        """
        Retrieves and processes the image from the simulator.
        """
        # If you use ImageRequest (airsim.ImageRequest), you will get an airsim.ImageResponse
        # which needs different handling. Adjust accordingly if needed.
        response = self.client.simGetImage('0', airsim.ImageType.Scene)
        return self._transform_obs(response)

    def _transform_obs(self, response):
        """
        Converts the raw image bytes into a processed numpy array.
        """
        img = Image.open(io.BytesIO(response))
        img_resized = img.resize((self.img_shape, self.img_shape))
        img_resized = np.array(img_resized, dtype=np.uint8)
        return img_resized.reshape([self.img_shape, self.img_shape, 3])
