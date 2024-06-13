import json
import time
import argparse
import logging
from flask import Flask, request, jsonify, abort
from airsim_car_connector import CarConnector
from airsim_utils import car_state_to_json, start_airsim, kill_airsim, load_config

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config", dest="config", type=str)
parser.add_argument("-l", "--log_path", dest="log", type=str)
# Configure logging to write to a file
logging.basicConfig(level=logging.INFO, filename=parser.parse_args().log, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class AirSimClient:
    def __init__(self, config):
        self.config = config
        self.pid = None
        self.car_connector = None
        logging.info("AirSimClient initialized with config.")

    def start_airsim(self):
        self.pid = start_airsim(self.config["path"], self.config["setting"])
        self.car_connector = CarConnector("127.0.0.1", self.config["port"])
        logging.info(f"Airsim started with PID: {self.pid}")

    def kill_airsim(self):
        kill_airsim(self.pid)
        logging.info(f"Airsim killed with PID: {self.pid}")
        self.pid = None
        self.car_connector = None

    def restart(self):
        logging.info("Restarting Airsim.")
        self.kill_airsim()
        self.start_airsim()


app = Flask(__name__)
airsim_config = load_config(parser.parse_args().config)
airsim_client = AirSimClient(airsim_config)
logging.info(f"Configuration loaded: {airsim_config}")

try:
    airsim_client.start_airsim()
except Exception as ex:
    logging.error(f"Failed to start Airsim: {ex}")
    exit()


@app.route('/reset', methods=['POST'])
def reset():
    data = request.get_json()
    if not data or not data.get('map'):
        abort(500, "Map data not provided")
    try:
        with open(airsim_client.config["map"], "w") as f:
            json.dump(data['map'], f)
        time.sleep(1)  # simulate map reset delay
        airsim_client.car_connector.reset()
        obs, info = airsim_client.car_connector.get_info()
        logging.info("Map reset successfully.")
        return jsonify({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        logging.error(f"Reset failed: {exc}")
        abort(500, f"Reset failed: {exc}")


# Similar error handling should be applied to other endpoints like '/step', '/restart', '/info', '/ping', '/exit'

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=airsim_config["server_port"])
