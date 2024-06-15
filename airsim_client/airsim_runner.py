import os
import json
import time
import signal
import logging
import argparse
import requests
import threading
from flask import Flask, request, jsonify, abort
from airsim_car_connector import CarConnector
from airsim_client.airsim_utils import car_state_to_dict, kill_airsim, load_config

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config", dest="config", type=str)
parser.add_argument("-l", "--log_path", dest="log", type=str)
parser.add_argument("-p", "--pid", dest="pid", type=str)

# Configure logging to write to a file
logging.basicConfig(level=logging.INFO, filename=parser.parse_args().log, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def shutdown_server():
    time.sleep(3)
    os.kill(os.getpid(), signal.SIGINT)


class AirSimClient:
    def __init__(self, config, pid):
        self.config = config
        self.pid = pid
        self.car_connector = CarConnector("127.0.0.1", self.config["port"])
        logging.info(f"Airsim started with PID: {self.pid}")
        logging.info("AirSimClient initialized with config.")

    def kill_airsim(self):
        kill_airsim(self.pid)
        logging.info(f"Airsim killed with PID: {self.pid}")

    def restart(self):
        logging.info("Restarting Airsim.")
        self.kill_airsim()


app = Flask(__name__)
airsim_config = load_config(parser.parse_args().config)
airsim_client = AirSimClient(airsim_config, parser.parse_args().pid)
logging.info(f"Configuration loaded: {airsim_config}")
requests.post('http://192.168.0.104:7575/add', json={'port': airsim_config["server_port"]})


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
        logging.info("Map reset successfully.")
        return jsonify({"signal": True})
    except Exception as exc:
        logging.error(f"Reset failed: {str(exc)}")
        abort(500, f"Reset failed: {str(exc)}")


@app.route('/step', methods=['POST'])
def step():
    data = request.get_json()
    try:
        airsim_client.car_connector.do_action(data['action'])
        logging.info("Action: %d" % data['action'])
        return jsonify({"signal": True})
    except Exception as exc:
        logging.error(f"Action failed: {str(exc)}")
        abort(500, f"Action failed: {str(exc)}")


@app.route('/info', methods=['GET'])
def get_info():
    try:
        obs, info = airsim_client.car_connector.get_info()
        logging.info("Information retrieved successfully.")
        return jsonify({
            "obs": obs.tolist(),
            **car_state_to_dict(info)
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
            abort(503, "Failed to connect to CarConnector")
    except Exception as e:
        logging.error(f"Ping failed: {str(e)}")
        abort(500, f"Ping failed: {str(e)}")


@app.route('/exit', methods=['GET'])
def exit_server():
    try:
        airsim_client.kill_airsim()
        response_info = {"info": "Kill Succeed"}
        threading.Thread(target=shutdown_server).start()
    except Exception as e:
        logging.error(f"Kill Airsim failed: {str(e)}")
        response_info = {"info": "Kill Failed", "error": str(e)}
    return jsonify(response_info)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=airsim_config["server_port"])
