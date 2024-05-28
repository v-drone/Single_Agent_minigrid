from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from airsim_utils import PortData
import uvicorn

app = FastAPI()


class AirSimManager:
    def __init__(self):
        self.available_ports = [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007]
        self.died_port = {}
        self.active_port = {}

    def get_available_port(self):
        for port in self.available_ports:
            if port not in self.active_port and port not in self.died_port:
                return port
        return None

    def register_connection(self, port):
        if port in self.available_ports and port not in self.died_port:
            self.active_port[port] = 'gym'
            return True
        return False

    def release_connection(self, port):
        if port in self.active_port:
            del self.active_port[port]
        if port in self.died_port:
            del self.died_port[port]

    def set_died(self, port):
        self.died_port[port] = 'gym'
        del self.active_port[port]

    def add_new(self, port):
        self.available_ports.append(port)


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
            "died": airsim_manager.died_port,
            "live": airsim_manager.active_port,
            "available": airsim_manager.available_ports,
        }
        print(response)
        return JSONResponse(response)
    except Exception:
        raise HTTPException(status_code=500, detail="Release Failed")


@app.get("/add")
async def add(data: PortData):
    try:
        airsim_manager.add_new(data.port)
        return JSONResponse({"message": "Add successful", "port": data.port})
    except Exception:
        raise HTTPException(status_code=500, detail="Add. Failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7575)
