import requests
import time
import base64
import zlib
import numpy as np
import logging
from environments.AirSimException import AirSimConnectionError, AirSimInfoError, AirSimRetryError


class RetryOperation:
    @staticmethod
    def execute(operation, retries=3, delay=2, *args, **kwargs):
        for attempt in range(retries):
            try:
                with requests.Session() as session:
                    return operation(session, *args, **kwargs)
            except requests.exceptions.ConnectionError as e:
                _e = e
                if attempt < retries - 1:
                    time.sleep(delay)
            except Exception as e:
                _e = e
                if attempt < retries - 1:
                    time.sleep(delay)
        raise AirSimRetryError(f"Operation failed after {retries} retries")


class AirSimClient:
    def __init__(self, manager_port=7575, remote_ip="127.0.0.1", logger=None):
        self.manager_port = manager_port
        self.remote_ip = remote_ip
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.local_port = None

    def handshake(self, retry=5):
        def operation(session):
            url = f"http://127.0.0.1:{self.manager_port}/handshake"
            response = session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if response.status_code == 200 and data.get("port"):
                self.local_port = data["port"]
                self.logger.info(f"[AirSimClient] Handshake success, local_port={self.local_port}")
                return True
            else:
                raise AirSimConnectionError("Failed to get valid port from handshake")

        RetryOperation.execute(operation, retries=retry)

    def release(self):
        if self.local_port is None:
            return

        def operation(session):
            url = f"http://127.0.0.1:{self.manager_port}/release"
            session.post(url, json={"port": self.local_port}, timeout=10)
            self.logger.info(f"[AirSimClient] Released port={self.local_port}")
            self.local_port = None
            return True

        try:
            RetryOperation.execute(operation, retries=3)
        except Exception as e:
            self.logger.warning(f"Release failed: {e}")

    def set_local_port_died(self):
        if self.local_port is None:
            return

        def operation(session):
            url = f"http://127.0.0.1:{self.manager_port}/set_died"
            session.post(url, json={"port": self.local_port}, timeout=10)
            self.logger.error(f"[AirSimClient] Mark port as died: {self.local_port}")
            return True

        try:
            RetryOperation.execute(operation, retries=3)
        except Exception as e:
            self.logger.warning(f"Set port died failed: {e}")

    def kill_airsim(self):
        if self.local_port is None:
            return

        def operation(session):
            url = f"http://{self.remote_ip}:{self.local_port}/exit"
            response = session.get(url, timeout=10)
            response.raise_for_status()
            self.logger.error(f"[AirSimClient] Killed AirSim at {self.local_port}")
            return True

        try:
            RetryOperation.execute(operation, retries=3)
        except Exception as e:
            self.logger.warning(f"Kill AirSim failed: {e}")

    def ping(self, retry=2):
        def operation(session):
            url = f"http://{self.remote_ip}:{self.local_port}/ping"
            response = session.get(url, timeout=10)
            response.raise_for_status()
            return True

        RetryOperation.execute(operation, retries=retry)

    def reset_env(self, map_json, retry=3):
        def operation(session):
            url = f"http://{self.remote_ip}:{self.local_port}/reset"
            response = session.post(url, json={"map": map_json}, timeout=10)
            response.raise_for_status()
            if response.status_code != 200:
                raise AirSimConnectionError("Failed to reset environment")
            return True

        RetryOperation.execute(operation, retries=retry)

    def step_env(self, action, retry=3):
        def operation(session):
            url = f"http://{self.remote_ip}:{self.local_port}/step"
            response = session.post(url, json={"action": int(action)}, timeout=10)
            response.raise_for_status()
            if response.status_code != 200:
                raise AirSimConnectionError("Failed to step environment")
            return True

        RetryOperation.execute(operation, retries=retry)

    def get_env_info(self, retry=5):
        def operation(session):
            url = f"http://{self.remote_ip}:{self.local_port}/info"
            response = session.get(url, timeout=10)
            response.raise_for_status()
            info = response.json()
            return info

        return RetryOperation.execute(operation, retries=retry)

    @staticmethod
    def decode_observation(info):
        try:
            obs = base64.b64decode(info["obs"])
            obs = zlib.decompress(obs)
            obs = np.frombuffer(obs, dtype=np.dtype(info['dtype'])).reshape(info["shape"])
            return obs
        except Exception as e:
            raise AirSimInfoError(f"Failed to decode obs: {e}")
