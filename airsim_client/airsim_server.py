import os
import json
import signal
import asyncio
import subprocess
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI()

airsim_info = {
    0: {"port": 41451, "path": "c:\\Users\\Administrator\\Documents\\airsimcar41451\\AirSimAssets_41451.exe"},
    1: {"port": 41452, "path": "c:\\Users\\Administrator\\Documents\\airsimcar41452\\AirSimAssets_41452.exe"},
    2: {"port": 41453, "path": "c:\\Users\\Administrator\\Documents\\airsimcar41453\\AirSimAssets_41453.exe"},
}

map_info = {
    0: {'path': "c:\\Users\\Administrator\\Documents\\airsimcar41451\\AirSimAssets_41451_Data\\StreamingAssets\\Test1"
                ".json"},
    1: {'path': "c:\\Users\\Administrator\\Documents\\airsimcar41452\\AirSimAssets_41452_Data\\StreamingAssets\\Test1"
                ".json"},
    2: {'path': "c:\\Users\\Administrator\\Documents\\airsimcar41453\\AirSimAssets_41453_Data\\StreamingAssets\\Test1"
                ".json"},
}

used_ports = {}


class CleanupData(BaseModel):
    client_id: int


class UpdateData(BaseModel):
    client_id: int
    map: dict


def kill_airsim(port):
    try:
        os.kill(used_ports[port], signal.SIGTERM)
        del used_ports[port]
        return 0
    except ProcessLookupError:
        return 1


def get_available_port():
    for info_id, info in airsim_info.items():
        if info['port'] not in used_ports:
            return info_id, info['port'], info['path']
    return None, None, None


async def start_airsim(path):
    if os.path.isfile(path) and os.access(path, os.X_OK):
        command = [path]
        return subprocess.Popen(command)
    else:
        raise Exception(f"Unity executable not found or not executable: {path}")


@app.get("/get_client")
async def get_client():
    info_id, port, path = get_available_port()
    if info_id is not None:
        process = await start_airsim(path)
        # FastAPI can handle async operations but subprocess.Popen here is synchronous.
        used_ports[port] = process.pid
        await asyncio.sleep(1)
        print(process.pid)
        return {'port': port, 'client_id': info_id, 'message': 'AirSim Client started'}
    else:
        raise HTTPException(status_code=404, detail="No available AirSim environment")


@app.post("/cleanup_client")
async def cleanup_port(data: CleanupData):
    port = airsim_info[data.client_id]["port"]
    code = kill_airsim(port)
    return {'message': f'Resources cleaned up for client_id {data.client_id}', 'code': code}


@app.post("/update_client")
async def reset_map_json(data: UpdateData):
    map_dict = data.map
    path = map_info[data.client_id]["path"]
    with open(path, "w") as f:
        json.dump(map_dict, f)
    return {'message': f'JSON file updated for client_id {data.client_id}', 'code': 0}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
