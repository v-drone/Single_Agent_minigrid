import os
import zlib
import json
import time
import base64
import signal
import logging
import argparse
import requests
import threading
from flask import Flask, request, jsonify, abort
from airsim_connector import DroneConnector
from airsim_utils import kill_airsim, load_config

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config", dest="config", type=str)
parser.add_argument("-l", "--log_path", dest="log", type=str)
parser.add_argument("-p", "--pid", dest="pid", type=str)

# Configure logging to write to a file
logging.basicConfig(level=logging.INFO, filename=parser.parse_args().log, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('werkzeug')


def shutdown_server():
    time.sleep(3)
    os.kill(os.getpid(), signal.SIGINT)


class DroneClient:
    def __init__(self, config, pid):
        self.config = config
        self.pid = pid
        self.connector = DroneConnector("127.0.0.1", self.config["port"])
        logging.info(f"Airsim started with PID: {self.pid}")
        logging.info("DroneClient initialized with config.")

    def kill_airsim(self):
        kill_airsim(self.pid)
        logging.info(f"Airsim killed with PID: {self.pid}")

    def restart(self):
        logging.info("Restarting Airsim.")
        self.kill_airsim()


app = Flask(__name__)
airsim_config = load_config(parser.parse_args().config)
airsim_client = DroneClient(airsim_config, parser.parse_args().pid)
logging.info(f"Configuration loaded: {airsim_config}")
response = requests.post('http://192.168.0.104:7575/add', json={'port': airsim_config["server_port"]})
logging.info(f"Added: {response.status_code}, {airsim_config['server_port']}")


@app.route('/reset', methods=['POST'])
def reset():
    data = request.get_json()
    try:
        if data.get("map", None) is not None:
            with open(airsim_client.config["map"], "w") as f:
                json.dump(data['map'], f)
            airsim_client.connector.reset()
        elif data.get("start", None) is not None:
            # todo: update start point
            airsim_client.connector.reset()
        else:
            abort(501, "Map/Start data not provided")
        time.sleep(1)
        logging.info("Map reset successfully.")
        return jsonify({"signal": True})
    except Exception as exc:
        logging.error(f"Reset failed: {str(exc)}")
        abort(500, f"Reset failed: {str(exc)}")


@app.route('/step', methods=['POST'])
def step():
    data = request.get_json()
    try:
        airsim_client.connector.do_action(data['action'])
        logging.info("Action: %d" % data['action'])
        return jsonify({"signal": True})
    except Exception as exc:
        logging.error(f"Action failed: {str(exc)}")
        abort(500, f"Action failed: {str(exc)}")


@app.route('/info', methods=['GET'])
def get_info():
    try:
        obs, info = airsim_client.connector.get_info()
        compressed_obs = zlib.compress(obs.tobytes())
        b64_compressed_obs = base64.b64encode(compressed_obs).decode('utf-8')
        logging.info("Information retrieved successfully.")
        return jsonify({
            "obs": b64_compressed_obs,
            "dtype": str(obs.dtype),
            "shape": obs.shape,
            **info
        })
    except Exception as exc:
        logging.error(f"Info retrieval failed: {str(exc)}")
        abort(500, f"Info retrieval failed: {str(exc)}")


@app.route('/ping', methods=['GET'])
def ping():
    try:
        if airsim_client.connector.ping():
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
