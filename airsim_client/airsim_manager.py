from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from airsim_utils import PortData
import uvicorn

app = FastAPI()


class AirSimManager:
    def __init__(self):
        self.available_ports = [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009]
        self.died_port = {}
        self.active_connections = {}

    def get_available_port(self):
        for port in self.available_ports:
            if port not in self.active_connections or port not in self.died_port:
                return port
        return None

    def register_connection(self, port):
        if port in self.available_ports and port not in self.died_port:
            self.active_connections[port] = 'gym'
            return True
        return False

    def release_connection(self, port):
        if port in self.active_connections:
            del self.active_connections[port]
        if port in self.died_port:
            del self.died_port[port]

    def set_died(self, port):
        self.died_port[port] = 'gym'


airsim_manager = AirSimManager()


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
        print(airsim_manager.available_ports)
        return JSONResponse({"message": "Release successful", "port": data.port})
    except Exception:
        raise HTTPException(status_code=500, detail="Release Failed")


@app.post("/set_died")
async def set_died(data: PortData):
    try:
        airsim_manager.set_died(data.port)
        print(airsim_manager.available_ports)
        return JSONResponse({"message": "Set Died successful", "port": data.port})
    except Exception:
        raise HTTPException(status_code=500, detail="Release Failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7575)
