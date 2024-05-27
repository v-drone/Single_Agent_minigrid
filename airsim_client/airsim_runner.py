import json
import time
import argparse
import logging
from flask import Flask, request, jsonify, abort
from airsim_car_connector import CarConnector
from airsim_utils import car_state_to_json, start_airsim, kill_airsim, load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AirSimClient:
    def __init__(self, config):
        self.config = config
        self.pid = None
        self.car_connector = None
        logging.info("AirSimClient initialized with config.")

    def start_airsim(self):
        self.pid = start_airsim(self.config["path"], self.config["setting"])
        time.sleep(5)  # simulate startup time synchronously
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
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config", dest="config", type=str)
airsim_config = load_config(parser.parse_args().config)
airsim_client = AirSimClient(airsim_config)
logging.info(f"Configuration loaded: {airsim_config}")
airsim_client.start_airsim()


@app.route('/reset', methods=['POST'])
def reset():
    data = request.get_json()
    if not data or not data.get('map'):
        abort(500, "Map data not provided")
    try:
        with open(airsim_client.config["map"], "w") as f:
            json.dump(data['map'], f)
        time.sleep(0.5)  # simulate map reset delay
        airsim_client.car_connector.reset()
        obs, info = airsim_client.car_connector.get_info()
        logging.info("Map reset successfully.")
        return jsonify({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        logging.error(f"Reset failed: {str(exc)}")
        abort(500, f"Reset failed: {str(exc)}")


@app.route('/step', methods=['POST'])
def step():
    data = request.get_json()
    try:
        airsim_client.car_connector.do_action(data['action'])
        obs, info = airsim_client.car_connector.get_info()
        return jsonify({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        logging.error(f"Action failed: {str(exc)}")
        abort(500, f"Action failed: {str(exc)}")


@app.route('/restart', methods=['GET'])
def restart():
    try:
        airsim_client.restart()
        airsim_client.car_connector.reset()
        obs, info = airsim_client.car_connector.get_info()
        logging.info("AirSim restarted successfully.")
        return jsonify({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        logging.error(f"Restart failed: {str(exc)}")
        abort(500, f"Restart failed: {str(exc)}")


@app.route('/info', methods=['GET'])
def get_info():
    try:
        obs, info = airsim_client.car_connector.get_info()
        logging.info("Information retrieved successfully.")
        return jsonify({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        logging.error(f"Info retrieval failed: {str(exc)}")
        abort(500, f"Info retrieval failed: {str(exc)}")


@app.route('/ping', methods=['GET'])
def ping():
    try:
        if airsim_client.car_connector.ping():
            logging.info("Ping successful.")
            return "Pong! CarConnector is active.", 200
        else:
            abort(500, "Failed to connect to CarConnector")
    except Exception as e:
        logging.error(f"Ping failed: {str(e)}")
        abort(500, f"Ping failed: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=airsim_config["server_port"])
