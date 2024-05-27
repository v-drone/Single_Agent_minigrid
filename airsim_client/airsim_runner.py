import json
import asyncio
import uvicorn
import argparse
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from airsim_utils import ActionData, MapData
from airsim_car_connector import CarConnector
from airsim_utils import car_state_to_json, start_airsim, kill_airsim, load_config


class AirSimClient:
    def __init__(self, config):
        self.config = config
        self.pid = None
        self.car_connector = None

    async def start_airsim(self):
        self.pid = start_airsim(self.config["path"], self.config["setting"])
        await asyncio.sleep(5)  # simulate startup time asynchronously
        self.car_connector = CarConnector("127.0.0.1", self.config["port"])

    def kill_airsim(self):
        kill_airsim(self.pid)
        self.pid = None
        self.car_connector = None

    async def restart(self):
        self.kill_airsim()
        await self.start_airsim()


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config", dest="config", type=str)

airsim_config = load_config(parser.parse_args().config)
print(airsim_config)
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    app.state.airsim_client = AirSimClient(airsim_config)
    await app.state.airsim_client.start_airsim()


async def get_airsim_client():
    return app.state.airsim_client


@app.post('/reset')
async def reset(data: MapData, airsim_client: AirSimClient = Depends(get_airsim_client)):
    if not data.map:
        raise HTTPException(status_code=500, detail="Map data not provided")
    try:
        with open(airsim_client.config["map"], "w") as f:
            json.dump(data.map, f)
        await asyncio.sleep(0.5)  # simulate map reset delay
        airsim_client.car_connector.reset()
        obs, info = airsim_client.car_connector.get_info()
        return JSONResponse({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(exc)}")


@app.post('/step')
async def step(data: ActionData, airsim_client: AirSimClient = Depends(get_airsim_client)):
    try:
        airsim_client.car_connector.do_action(data.action)
        obs, info = airsim_client.car_connector.get_info()
        return JSONResponse({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Action failed: {str(exc)}")


@app.get('/restart')
async def restart(airsim_client: AirSimClient = Depends(get_airsim_client)):
    try:
        await airsim_client.restart()
        airsim_client.car_connector.reset()
        obs, info = airsim_client.car_connector.get_info()
        return JSONResponse({
            "obs": obs.tolist(),
            "info": car_state_to_json(info)
        })
    except Exception as exc:
        print(str(exc))
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
        if airsim_client.car_connector.ping():
            return PlainTextResponse("Pong! CarConnector is active.", status_code=200)
        else:
            raise HTTPException(status_code=500, detail=f"Pong! Failed to connect to CarConnector", )
    except Exception as e:
        # Log the error here if possible
        raise HTTPException(status_code=501, detail=f"Pong! Failed to connect to CarConnector, {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=airsim_config["server_port"])
