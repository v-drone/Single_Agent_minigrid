import json
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from airsim_utils import ActionData, MapData
from airsim_car_connector import CarConnector
from airsim_utils import car_state_to_json, start_airsim, kill_airsim


class AirSimClient:
    def __init__(self, config):
        self.config = config
        self.pid = None
        self.car_connector = None

    async def start_airsim(self):
        self.pid = await start_airsim(self.config["path"], self.config["setting"])
        await asyncio.sleep(5)  # simulate startup time asynchronously
        self.car_connector = CarConnector("127.0.0.1", self.config["port"])

    def kill_airsim(self):
        if self.pid:
            kill_airsim(self.pid)
            self.pid = None
            self.car_connector = None

    async def restart(self):
        self.kill_airsim()
        await self.start_airsim()


airsim_config = {
    "port": 41451,
    "path": "c:\\Users\\Administrator\\Documents\\airsimcar41451\\AirSimAssets_41451.exe",
    "setting": "-settings='c:\\Users\\Administrator\\Documents\\airsimcar41451\\settings.json'"
}

app = FastAPI()
client_instance = AirSimClient(airsim_config)


async def get_airsim_client():
    client = AirSimClient(airsim_config)
    await client.start_airsim()
    return client


@app.on_event("startup")
async def startup_event():
    app.state.client_instance = await get_airsim_client()


@app.post('/reset')
async def reset(data: MapData, client: AirSimClient = Depends(get_airsim_client)):
    if not data.map:
        raise HTTPException(status_code=500, detail="Map data not provided")
    try:
        with open(client.config["setting"], "w") as f:
            json.dump(data.map, f)
        await asyncio.sleep(0.5)  # simulate map reset delay
        obs, info = client.car_connector.reset()
        return JSONResponse({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(exc)}")


@app.post('/step')
async def step(data: ActionData, airsim_client: AirSimClient = Depends(get_airsim_client)):
    try:
        obs, info = airsim_client.car_connector.do_action(data.action)
        return JSONResponse({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Action failed: {str(exc)}")


@app.post('/restart')
async def cleanup(airsim_client: AirSimClient = Depends(get_airsim_client)):
    try:
        await airsim_client.restart()
        obs, info = airsim_client.car_connector.reset()
        return JSONResponse({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Restart failed: {str(exc)}")


@app.get('/info')
async def get_info(airsim_client: AirSimClient = Depends(get_airsim_client)):
    try:
        obs, info = airsim_client.car_connector.get_info()
        return JSONResponse({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Info retrieval failed: {str(exc)}")


@app.get('/ping')
async def ping(airsim_client: AirSimClient = Depends(get_airsim_client)):
    try:
        status = airsim_client.car_connector.get_info()
        if status:
            return PlainTextResponse("Pong! CarConnector is active.", status_code=200)
        else:
            return PlainTextResponse("Pong! CarConnector is inactive.", status_code=503)
    except Exception as e:
        return PlainTextResponse(f"Pong! Failed to connect to CarConnector: {str(e)}", status_code=503)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
