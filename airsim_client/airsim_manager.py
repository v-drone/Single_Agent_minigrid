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

    def modify_ports(self, action: str, port: str, set_name: str):
        """Modify the sets based on the action."""
        if set_name not in ['available_ports', 'active_ports', 'died_ports']:
            raise ValueError("Invalid set name")

        target_set = getattr(self, set_name)

        if action == "add":
            target_set.add(port)
        elif action == "remove":
            target_set.discard(port)
        else:
            raise ValueError("Invalid action. Use 'add' or 'remove'.")

        return True


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
        raise HTTPException(status_code=500, detail="No available AirSim instances")


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
        raise HTTPException(status_code=500, detail="Set Died Failed")


@app.get("/info")
async def get_info():
    try:
        response = {
            "died": list(airsim_manager.died_ports),
            "live": list(airsim_manager.active_ports),
            "available": list(airsim_manager.available_ports),
        }
        return JSONResponse(response)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to retrieve information")


@app.post("/add")
async def add(data: PortData):
    try:
        airsim_manager.add_new(data.port)
        return JSONResponse({"message": "Add successful", "port": data.port})
    except Exception:
        raise HTTPException(status_code=500, detail="Add Failed")


# New endpoints to control the sets directly without changing the original endpoints

@app.post("/modify/available_ports/{port}")
async def modify_available_ports(action: str, port: str):
    """Add or remove ports to/from the available_ports set."""
    try:
        if action not in ["add", "remove"]:
            raise HTTPException(status_code=400, detail="Action must be 'add' or 'remove'")
        if action == "add":
            airsim_manager.available_ports.add(port)
        elif action == "remove":
            airsim_manager.available_ports.discard(port)
        return JSONResponse({"message": f"Port {action}ed to available_ports", "port": port})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/modify/active_ports/{port}")
async def modify_active_ports(action: str, port: str):
    """Add or remove ports to/from the active_ports set."""
    try:
        if action not in ["add", "remove"]:
            raise HTTPException(status_code=400, detail="Action must be 'add' or 'remove'")
        if action == "add":
            airsim_manager.active_ports.add(port)
        elif action == "remove":
            airsim_manager.active_ports.discard(port)
        return JSONResponse({"message": f"Port {action}ed to active_ports", "port": port})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/modify/died_ports/{port}")
async def modify_died_ports(action: str, port: str):
    """Add or remove ports to/from the died_ports set."""
    try:
        if action not in ["add", "remove"]:
            raise HTTPException(status_code=400, detail="Action must be 'add' or 'remove'")
        if action == "add":
            airsim_manager.died_ports.add(port)
        elif action == "remove":
            airsim_manager.died_ports.discard(port)
        return JSONResponse({"message": f"Port {action}ed to died_ports", "port": port})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7575)
