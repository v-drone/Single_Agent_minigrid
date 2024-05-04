import time

from flask import Flask, request, jsonify
import requests
from airsim import CarClient

remote_ip = "192.168.0.102"
remote_address = "http://192.168.0.102:5000"
client_info = {}
client = {}

app = Flask(__name__)


@app.route('/restart', methods=['POST'])
def restart_unity_environment():
    client_info_data = requests.get(remote_address + "/get_client").json()
    local_client_id = request.get_json().get("local_client_id", None)
    if local_client_id is None:
        local_client_id = len(client_info)
    client_info[local_client_id] = {
        "client_port": client_info_data["port"],
        "client_id": client_info_data["client_id"],
    }
    print(client_info[local_client_id])
    client_port = client_info[local_client_id]["client_port"]
    client[local_client_id] = CarClient(remote_ip)
    client[local_client_id].confirmConnection()
    print(client[local_client_id])
    time.sleep(5)
    _ = {
        'local_client_id': local_client_id,
        'client_id': client_info_data["client_id"],
        'client_port': client_info_data["port"]
    }
    print(_)
    return jsonify(
        {
            'local_client_id': local_client_id,
            'client_id': client_info_data["client_id"],
            'client_port': client_info_data["port"]
        }
    )


@app.route('/step/<local_client_id>', methods=['POST'])
def step(local_client_id):
    pass


@app.route('/reset/<local_client_id>', methods=['POST'])
def reset(local_client_id):
    # 请在这里添加重置环境的逻辑
    pass


@app.route('/render/<local_client_id>', methods=['POST'])
def render(local_client_id, mode='human'):
    # 请在这里添加渲染环境的逻辑
    pass


@app.route('/close/<local_client_id>', methods=['POST'])
def close(local_client_id):
    client_id = client_info[local_client_id]["client_id"]
    client_info_data = requests.post(remote_address + "/cleanup_client", json={"client_id": client_id})
    return client_info_data.json()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
