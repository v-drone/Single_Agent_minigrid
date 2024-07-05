from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from airsim_utils import PortData
import uvicorn


class AirSimManager:
    def __init__(self):
        self.available_ports = set()
        self.active_ports = set()
        self.died_ports = set()

    def get_available_port(self):
        available = self.available_ports - self.active_ports - self.died_ports
        return next(iter(available), None)

    def register_connection(self, port):
        if port in self.available_ports and port not in self.died_ports and port not in self.active_ports:
            self.active_ports.add(port)
            return True
        return False

    def release_connection(self, port):
        self.active_ports.discard(port)

    def set_died(self, port):
        if port in self.available_ports:
            self.available_ports.discard(port)
        self.active_ports.discard(port)
        if port not in self.died_ports:
            self.died_ports.add(port)

    def add_new(self, port):
        if port not in self.available_ports:
            self.available_ports.add(port)


airsim_manager = AirSimManager()
app = FastAPI()


@app.get("/handshake")
async def handshake():
    port = airsim_manager.get_available_port()
    if port:
        if airsim_manager.register_connection(port):
            return JSONResponse({"message": "Connection successful", "port": port})
        else:
            raise HTTPException(status_code=500, detail="Failed to register connection")
    else:
        raise HTTPException(status_code=404, detail="No available AirSim instances")


@app.post("/release")
async def release(data: PortData):
    try:
        airsim_manager.release_connection(data.port)
        return JSONResponse({"message": "Release successful", "port": data.port})
    except Exception:
        raise HTTPException(status_code=500, detail="Release Failed")


@app.post("/set_died")
async def set_died(data: PortData):
    try:
        airsim_manager.set_died(data.port)
        return JSONResponse({"message": "Set Died successful", "port": data.port})
    except Exception:
        raise HTTPException(status_code=500, detail="Release Failed")


@app.get("/info")
async def get_info():
    try:
        response = {
            "died": list(airsim_manager.died_ports),
            "live": list(airsim_manager.active_ports),
            "available": list(airsim_manager.available_ports),
        }
        print(response)
        return JSONResponse(response)
    except Exception:
        raise HTTPException(status_code=500, detail="Release Failed")


@app.post("/add")
async def add(data: PortData):
    try:
        airsim_manager.add_new(data.port)
        return JSONResponse({"message": "Add successful", "port": data.port})
    except Exception:
        raise HTTPException(status_code=500, detail="Add. Failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7575)
