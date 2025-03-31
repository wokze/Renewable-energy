class WindTurbine:
    def __init__(self, capacity: float, location: str):
        self.capacity = capacity  # Maximum power output in kW
        self.location = location    # Location of the wind turbine
        self.current_output = 0.0   # Current power output in kW

    def generate_power(self, wind_speed: float) -> float:
        """Calculate power generated based on wind speed."""
        if wind_speed < 3:  # Cut-in speed
            self.current_output = 0.0
        elif 3 <= wind_speed <= 15:  # Rated speed
            self.current_output = self.capacity * (wind_speed / 15) ** 3
        else:  # Cut-out speed
            self.current_output = self.capacity
        
        return self.current_output

    def get_status(self) -> str:
        """Return the status of the wind turbine."""
        return f"Wind Turbine at {self.location} - Current Output: {self.current_output} kW"