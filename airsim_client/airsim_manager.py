from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()


class AirSimManager:
    def __init__(self):
        self.available_ports = [41451, 41452, 41453]
        self.active_connections = {}

    def get_available_port(self):
        for port in self.available_ports:
            if port not in self.active_connections:
                return port
        return None

    def register_connection(self, port):
        if port in self.available_ports:
            self.active_connections[port] = 'gym'
            return True
        return False

    def release_connection(self, port):
        if port in self.active_connections:
            del self.active_connections[port]


airsim_manager = AirSimManager()


@app.get("/handshake")
async def handshake():
    port = airsim_manager.get_available_port()
    if port:
        if airsim_manager.register_connection(port):
            return {"message": "Connection successful", "port": port}
        else:
            raise HTTPException(status_code=500, detail="Failed to register connection")
    else:
        raise HTTPException(status_code=404, detail="No available AirSim instances")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7575)
