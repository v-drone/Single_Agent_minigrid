class AirSimError(Exception):
    """Base class for AirSim errors."""
    pass


class AirSimConnectionError(AirSimError):
    """Exception raised when failing to connect to the AirSim server."""

    def __init__(self, message="Failed to connect to AirSim server", e=None):
        self.message = message
        self.e = e
        super().__init__(self.message)


class AirSimResponseError(AirSimError):
    """Exception raised for errors in the response from AirSim."""

    def __init__(self, message="Invalid response from AirSim server", e=None):
        self.message = message
        self.e = e
        super().__init__(self.message)


class AirSimRestartFailed(AirSimError):
    """Exception raised when AirSim restart attempts fail."""

    def __init__(self, message="Failed to restart AirSim server", e=None):
        self.message = message
        self.e = e
        super().__init__(self.message)
