class Turbine:
    def __init__(
        self,
        model_name: str,
        capacity_per_turbine: float,
        rotor_diameter: float,
        hub_height: float,
        manufactured_date: str,
        technology: str,
        cut_in_wind_speed: float,
        cut_out_wind_speed: float,
        wind_speed_range: str,
        temperature_range: str,
        warranty: str,
        operating_temperature_range: str,
        weight: str,
        power_curve: dict,
    ):
        self.model_name = model_name
        self.capacity = capacity_per_turbine  # MW
        self.rotor_diameter = rotor_diameter  # m
        self.hub_height = hub_height  # m
        self.manufactured_date = manufactured_date
        self.technology = technology
        self.cut_in_speed = cut_in_wind_speed  # m/s
        self.cut_out_speed = cut_out_wind_speed  # m/s
        self.wind_speed_range = wind_speed_range
        self.temperature_range = temperature_range
        self.warranty = warranty
        self.operating_temperature_range = operating_temperature_range
        self.weight = weight  # tons
        self.power_curve = power_curve
        self.current_output = 0.0

    def calculate_output(self, wind_speed: float, temperature: float = 20.0) -> float:
        """Calculate power output based on wind speed and temperature"""
        if wind_speed < self.cut_in_speed or wind_speed > self.cut_out_speed:
            self.current_output = 0.0
        else:
            # Interpolate from power curve
            for speed, power in sorted(self.power_curve.items()):
                if float(speed) > wind_speed:
                    break
                self.current_output = power

        return self.current_output

    def get_status(self) -> dict:
        return {
            "model": self.model_name,
            "capacity": self.capacity,
            "current_output": self.current_output,
            "technology": self.technology,
            "rotor_diameter": self.rotor_diameter,
            "hub_height": self.hub_height,
            "operating_range": self.wind_speed_range,
        }
