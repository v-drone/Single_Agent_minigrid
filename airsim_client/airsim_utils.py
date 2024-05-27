import logging
import os
import math
import json
import signal
import subprocess
import time

from pydantic import BaseModel


class ActionData(BaseModel):
    action: int


class PortData(BaseModel):
    port: int


class MapData(BaseModel):
    map: dict


class ClientConfig(BaseModel):
    port: int
    path: str
    setting: str


def load_config(path):
    with open(path, "r") as file:
        return json.load(file)


def vector3r_to_dict(vector3r):
    return {
        'x_val': vector3r.x_val,  # Make sure these fields match your actual object structure
        'y_val': vector3r.y_val,
        'z_val': vector3r.z_val
    }


def quaternionr_to_dict(quaternionr):
    return {
        'w_val': quaternionr.w_val,
        'x_val': quaternionr.x_val,
        'y_val': quaternionr.y_val,
        'z_val': quaternionr.z_val
    }


def kinematics_to_dict(kinematics):
    return {
        'angular_acceleration': vector3r_to_dict(kinematics.angular_acceleration),
        'angular_velocity': vector3r_to_dict(kinematics.angular_velocity),
        'linear_acceleration': vector3r_to_dict(kinematics.linear_acceleration),
        'linear_velocity': vector3r_to_dict(kinematics.linear_velocity),
        'orientation': quaternionr_to_dict(kinematics.orientation),
        'position': vector3r_to_dict(kinematics.position)
    }


def quaternion_to_euler(orientation):
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


def car_state_to_json(car_state):
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
        "orientation": quaternion_to_euler(orientation),
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


def kill_airsim(process_id):
    try:
        os.kill(process_id, signal.SIGTERM)
    except ProcessLookupError:
        return "Process not found"


def start_airsim(path, setting):
    if os.path.isfile(path) and os.access(path, os.X_OK):
        command = f'"{path}" "{setting}"'
        logging.info(f"Executing command: {command}")

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       cwd=os.path.dirname(path))

            time.sleep(10)
            if process.poll() is None:
                return process.pid
            else:
                _, stderr = process.communicate()
                error_message = stderr.decode().strip()
                logging.error(f"Process error output: {error_message}")
                raise Exception(f"Process terminated prematurely with error: {error_message}")

        except Exception as e:
            raise Exception(f"Failed to start process: {str(e)}")
    else:
        raise Exception(f"Executable not found or not executable: {path}")
