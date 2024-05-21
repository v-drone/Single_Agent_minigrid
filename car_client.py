import json
import math
import requests
import asyncio
from airsim_client.airsim_car_connector import CarConnector
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

remote_airsim_ip = "127.0.0.1"
remote_address = "http://127.0.0.1:5000"
client_info = {}

app = FastAPI()


class ActionData(BaseModel):
    local_client_id: int
    action: Optional[str] = None
    map: Optional[dict] = None


def _vector3r_to_dict(vector3r):
    return {
        'x_val': vector3r.x_val,  # Make sure these fields match your actual object structure
        'y_val': vector3r.y_val,
        'z_val': vector3r.z_val
    }


def _quaternionr_to_dict(quaternionr):
    return {
        'w_val': quaternionr.w_val,
        'x_val': quaternionr.x_val,
        'y_val': quaternionr.y_val,
        'z_val': quaternionr.z_val
    }


def _kinematics_to_dict(kinematics):
    return {
        'angular_acceleration': _vector3r_to_dict(kinematics.angular_acceleration),
        'angular_velocity': _vector3r_to_dict(kinematics.angular_velocity),
        'linear_acceleration': _vector3r_to_dict(kinematics.linear_acceleration),
        'linear_velocity': _vector3r_to_dict(kinematics.linear_velocity),
        'orientation': _quaternionr_to_dict(kinematics.orientation),
        'position': _vector3r_to_dict(kinematics.position)
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


@app.post('/restart')
async def restart_unity_environment(request: Request):
    client_info_data = requests.get(remote_address + "/get_client").json()
    body = await request.json()
    local_client_id = body.get("local_client_id", len(client_info))
    client_info[local_client_id] = {
        "local_client_id": local_client_id,
        "client_port": client_info_data["port"],
        "client_id": client_info_data["client_id"],
        "car": CarConnector(remote_airsim_ip, int(client_info_data["port"]))
    }
    return JSONResponse(client_info[local_client_id])


@app.post('/step')
async def step(data: ActionData):
    if data.local_client_id not in client_info or not data.action:
        raise HTTPException(status_code=404, detail='Client or action not found')
    car = client_info[data.local_client_id]["car"]
    obs, info = car.do_action(data.action)
    return JSONResponse({"obs": obs.tolist(), "info": info})


@app.post('/reset')
async def reset(data: ActionData):
    if data.local_client_id not in client_info:
        raise HTTPException(status_code=404, detail='Client not found')
    car = client_info[data.local_client_id].get("car")
    remote_client_id = client_info[data.local_client_id].get("client_id")
    if car is None or remote_client_id is None:
        raise HTTPException(status_code=400, detail='Missing car or remote client ID')

    if data.map:
        # Post the map to the remote server
        requests.post(remote_address + "/update_client", json={
            "client_id": remote_client_id,
            "map": data.map
        })
        await asyncio.sleep(0.5)

    # Reset the car and get the new state
    obs, info = car.reset()
    return JSONResponse({
        "obs": obs.tolist(),
        "info": _car_state_to_json(info),
        "local_client_id": data.local_client_id
    })


@app.post('/info')
async def get_info(data: ActionData):
    if data.local_client_id not in client_info:
        raise HTTPException(status_code=404, detail='Client not found')
    car = client_info[data.local_client_id].get("car")
    if car is None:
        raise HTTPException(status_code=400, detail='Car instance not found for the given client ID')

    # Get current state information from the car
    obs, info = car.get_info()
    return JSONResponse({
        "obs": obs.tolist(),
        "info": _car_state_to_json(info),
        "local_client_id": data.local_client_id
    })


@app.post('/close')
async def close(data: ActionData):
    client_id = client_info[data.local_client_id]["client_id"]
    response = requests.post(remote_address + "/cleanup_client", json={"client_id": client_id})
    return response.json()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=6000)
