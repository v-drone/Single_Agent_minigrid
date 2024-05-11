import time
import requests
from airsim_client.airsim_car_connector import CarConnector
from flask import Flask, request, jsonify

remote_ip = "127.0.0.1"
remote_address = "http://127.0.0.1:5000"
client_info = {}

app = Flask(__name__)


@app.route('/restart', methods=['POST'])
def restart_unity_environment():
    client_info_data = requests.get(remote_address + "/get_client").json()
    local_client_id = request.get_json().get("local_client_id", None)
    if local_client_id is None:
        local_client_id = len(client_info)
    client_info[local_client_id] = {
        "local_client_id": local_client_id,
        "client_port": client_info_data["port"],
        "client_id": client_info_data["client_id"],
        "car": CarConnector(remote_ip, int(client_info_data["port"]))
    }
    print(client_info[local_client_id])
    time.sleep(0.01)
    return jsonify({
        "local_client_id": client_info[local_client_id]["local_client_id"],
        "client_port": client_info[local_client_id]["client_port"],
        "client_id": client_info[local_client_id]["client_id"],
    })


@app.route('/step', methods=['POST'])
def step():
    """ Perform an action step for the given client """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input'}), 400
    else:
        local_client_id = data.get('local_client_id')
        action = data.get('action')
        if local_client_id is None or action is None:
            return jsonify({'error': 'Missing local_client_id or action'}), 400
        elif local_client_id not in client_info:
            return jsonify({'error': f'Client {local_client_id} not found'}), 404
        else:
            car = client_info[local_client_id]["car"]
            return jsonify({'message': f'Step action {action} executed for client {local_client_id}'})


@app.route('/reset', methods=['POST'])
def reset():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input'}), 400
    else:
        local_client_id = data.get('local_client_id')
        car = client_info[local_client_id].get("car", None)
        if car is not None:
            obs, _ = car.reset()
            return jsonify({"obs": obs.tolist(), "local_client_id": local_client_id})
        else:
            return jsonify({'error': 'Missing local_client_id or action'}), 400


@app.route('/render', methods=['POST'])
def render():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input'}), 400
    else:
        local_client_id = data.get('local_client_id')


@app.route('/close/', methods=['POST'])
def close():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input'}), 400
    else:
        local_client_id = data.get('local_client_id')
    client_id = client_info[int(local_client_id)]["client_id"]
    print(client_id)
    client_info_data = requests.post(remote_address + "/cleanup_client", json={"client_id": client_id})
    print(client_info_data.json())
    return client_info_data.json()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
