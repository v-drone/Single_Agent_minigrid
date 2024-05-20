import os
import signal
import subprocess
import time
import json
from flask import Flask, jsonify, request

app = Flask(__name__)

airsim_info = {
    0: {"port": 41451, "path": "c:\\Users\\Administrator\\Documents\\airsimcar41451\\AirSimAssets_41451.exe"},
    1: {"port": 41452, "path": "c:\\Users\\Administrator\\Documents\\airsimcar41452\\AirSimAssets_41452.exe"},
    2: {"port": 41453, "path": "c:\\Users\\Administrator\\Documents\\airsimcar41453\\AirSimAssets_41453.exe"},
}

map_info = {
    0: {"path": "c:\\Users\\Administrator\\Documents\\airsimcar41451"
                "\\AirSimAssets_41451_Data\\StreamingAssets\\Test1.json"},
    1: {"path": "c:\\Users\\Administrator\\Documents\\airsimcar41452"
                "\\AirSimAssets_41452_Data\\StreamingAssets\\Test1.json"},
    2: {"path": "c:\\Users\\Administrator\\Documents\\airsimcar41453"
                "\\AirSimAssets_41453_Data\\StreamingAssets\\Test1.json"}

}

used_ports = {}


def kill_airsim(port):
    try:
        os.kill(used_ports[port], signal.SIGTERM)
        del used_ports[port]
        return 0
    except ProcessLookupError:
        return 1


def get_available_port():
    print(used_ports)
    for info_id, info in airsim_info.items():
        if info['port'] not in used_ports:
            return info_id, info['port'], info['path']
    return None, None, None


def start_airsim(path):
    if os.path.isfile(path) and os.access(path, os.X_OK):
        # batch_mode_args = ['-batchmode', '-nographics']
        batch_mode_args = []
        command = [path] + batch_mode_args
        return subprocess.Popen(command)

    else:
        raise Exception(f"Unity executable not found or not executable: {path}")


# DONE
@app.route('/get_client', methods=['GET'])
def get_client():
    info_id, port, path = get_available_port()
    if info_id is not None and port is not None and path is not None:
        process = start_airsim(path)
        time.sleep(10)
        used_ports[port] = process.pid
        print(port)
        return jsonify({'port': port, 'client_id': info_id, 'message': 'AirSim Client started'})
    else:
        return jsonify({'message': 'No available AirSim environment'})


@app.route('/cleanup_client', methods=['POST'])
def cleanup_port():
    data = request.json
    if 'client_id' in data:
        client_id = data['client_id']
        port = airsim_info[client_id]["port"]
        code = kill_airsim(port)
        return jsonify({'message': f'Resources cleaned up for client_id {client_id}', 'code': code})

    else:
        return jsonify({'message': f'Port not found in used ports', 'code': 2})


@app.route('/update_client', methods=['POST'])
def reset_map_json():
    data = request.json
    if 'client_id' in data:
        client_id = data['client_id']
        map_dict = data["map"]
        with open(map_info[client_id]["path"], "w") as f:
            json.dump(map_dict, f)
            return jsonify({'message': f'JSON file updated client_id {client_id}', 'code': 0})

    else:
        return jsonify({'message': f'client  not found in used ports', 'code': 2})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
