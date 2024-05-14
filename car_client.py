import time
import json
import math
import requests
from airsim_client.airsim_car_connector import CarConnector
from flask import Flask, request, jsonify

remote_ip = "127.0.0.1"
remote_address = "http://127.0.0.1:5000"
client_info = {}

app = Flask(__name__)


def _vector3r_to_dict(vector3r):
    return {
        'x_val': vector3r['x_val'],
        'y_val': vector3r['y_val'],
        'z_val': vector3r['z_val']
    }


def _quaternionr_to_dict(quaternionr):
    return {
        'w_val': quaternionr['w_val'],
        'x_val': quaternionr['x_val'],
        'y_val': quaternionr['y_val'],
        'z_val': quaternionr['z_val']
    }


def _kinematics_to_dict(kinematics):
    return {
        'angular_acceleration': _vector3r_to_dict(kinematics['angular_acceleration']),
        'angular_velocity': _vector3r_to_dict(kinematics['angular_velocity']),
        'linear_acceleration': _vector3r_to_dict(kinematics['linear_acceleration']),
        'linear_velocity': _vector3r_to_dict(kinematics['linear_velocity']),
        'orientation': _quaternionr_to_dict(kinematics['orientation']),
        'position': _vector3r_to_dict(kinematics['position'])
    }


def _quaternion_to_euler(orientation):
    w, x, y, z = orientation['w'], orientation['x'], orientation['y'], orientation['z']

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _car_state_to_json(car_state):
    # Extract orientation and position data from the KinematicsState object within CarState
    orientation = {
        "w": car_state.kinematics_estimated.orientation.w_val,
        "x": car_state.kinematics_estimated.orientation.x_val,
        "y": car_state.kinematics_estimated.orientation.y_val,
        "z": car_state.kinematics_estimated.orientation.z_val
    }

    position = {
        "x": car_state.kinematics_estimated.position.x_val,
        "y": car_state.kinematics_estimated.position.y_val,
        "z": car_state.kinematics_estimated.position.z_val
    }

    # Create a dictionary containing the orientation and position
    result = {
        "orientation": _quaternion_to_euler(orientation),
        "position": position,
        'gear': car_state.gear,
        'handbrake': car_state.handbrake,
        'maxrpm': car_state.maxrpm,
        'rpm': car_state.rpm,
        'speed': car_state.speed,
        'timestamp': car_state.timestamp

    }

    # Convert the dictionary to a JSON string
    json_result = json.dumps(result, indent=4)
    return json_result


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
            obs, info = car.do_action(action)
            info = _car_state_to_json(info)
            return jsonify({"obs": obs.tolist(),
                            "info": info,
                            'message': f'Step action {action} executed for client {local_client_id}'})


@app.route('/reset', methods=['POST'])
def reset():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input'}), 400
    else:
        local_client_id = data.get('local_client_id')
        car = client_info[local_client_id].get("car", None)
        remote_client_id = client_info[local_client_id].get("client_id", None)
        map_json = data.get("map")
        if car is not None and remote_client_id is not None:
            requests.post(remote_address + "/update_client", json={
                "client_id": remote_client_id,
                "map": map_json
            })
            time.sleep(2)
            obs, info = car.reset()
            info = _car_state_to_json(info)
            return jsonify({"obs": obs.tolist(), "info": info, "local_client_id": local_client_id})
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
    client_info_data = requests.post(remote_address + "/cleanup_client", json={"client_id": client_id})
    return client_info_data.json()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
