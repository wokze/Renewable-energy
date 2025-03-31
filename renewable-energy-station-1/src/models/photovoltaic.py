class Photovoltaic:
    def __init__(
        self,
        model_name: str,
        capacity_per_panel: float,
        efficiency: float,
        manufactured_date: str,
        module_type: str,
        max_power: float,
        temperature_coefficient: str,
        warranty: str,
        operating_temperature_range: str,
        weight: float,
        dimensions: str,
        max_system_voltage: int,
    ):
        self.model_name = model_name
        self.capacity_per_panel = capacity_per_panel  # kW
        self.efficiency = efficiency  # as percentage
        self.manufactured_date = manufactured_date
        self.module_type = module_type
        self.max_power = max_power  # W
        self.temperature_coefficient = temperature_coefficient
        self.warranty = warranty
        self.operating_temperature_range = operating_temperature_range
        self.weight = weight  # kg
        self.dimensions = dimensions
        self.max_system_voltage = max_system_voltage  # V

    def calculate_energy_production(
        self, solar_irradiance: float, temperature: float = 25.0
    ) -> float:
        """
        Calculate energy production based on solar irradiance and temperature.

        Args:
            solar_irradiance: Solar irradiance in W/m²
            temperature: Ambient temperature in °C (default 25°C)

        Returns:
            Energy production in kWh
        """
        # Base production at standard conditions
        base_production = (
            solar_irradiance * self.efficiency / 100
        ) * self.capacity_per_panel

        # Temperature correction
        temp_coeff = float(self.temperature_coefficient.strip("%")) / 100
        temp_correction = 1 + (
            temp_coeff * (temperature - 25)
        )  # 25°C is standard test condition

        return base_production * temp_correction

    def get_panel_area(self) -> float:
        """Calculate panel area in m² from dimensions string"""
        try:
            length, width = self.dimensions.replace("m", "").split("x")
            return float(length) * float(width)
        except:
            return 0.0

    def __str__(self) -> str:
        return (
            f"{self.model_name}: {self.capacity_per_panel}kW, "
            f"{self.efficiency}% efficiency, {self.module_type}"
        )

    def get_status(self) -> dict:
        """Get current status and specifications of the PV panel"""
        return {
            "model": self.model_name,
            "capacity": self.capacity_per_panel,
            "efficiency": self.efficiency,
            "type": self.module_type,
            "max_power": self.max_power,
            "warranty": self.warranty,
            "temp_coefficient": self.temperature_coefficient,
            "dimensions": self.dimensions,
            "area": self.get_panel_area(),
        }

    def get_pvgis_forecast(
        self,
        latitude: float,
        longitude: float,
        installation_angle: float = 35.0,
        azimuth: float = 0.0,
        mounting_type: str = "free",
        system_losses: float = 14.0,
    ) -> dict:
        """Get hourly production forecast from PVGIS API"""
        from src.services.pvgis_service import PVGISService

        # Convert module_type to PVGIS format
        pv_tech = "crystSi"  # default
        if self.module_type.lower().find("cdte") >= 0:
            pv_tech = "CdTe"
        elif (
            self.module_type.lower().find("cis") >= 0
            or self.module_type.lower().find("cigs") >= 0
        ):
            pv_tech = "CIS"

        # Calculate total peak power in kWp
        peak_power = self.capacity_per_panel  # Already in kW

        # Create service and get forecast
        service = PVGISService()
        forecast = service.calculate_pv_generation(
            latitude=latitude,
            longitude=longitude,
            peak_power=peak_power,
            mounting_type=mounting_type,
            loss=system_losses,
            angle=installation_angle,
            aspect=azimuth,
            pv_technology=pv_tech,
            cache_key=f"{self.model_name}_{latitude}_{longitude}",
        )

        return forecast
