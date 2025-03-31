import json
from pathlib import Path
from .turbine import Turbine


class TurbineFactory:
    def __init__(self, config_path: str = "config/turbines.json"):
        self.config_path = Path(config_path)
        self.turbine_configs = self._load_configs()

    def _load_configs(self) -> dict:
        """Load turbine configurations from JSON file"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {}

    def get_available_turbines(self) -> list:
        """Return list of available turbine configurations"""
        return list(self.turbine_configs.keys())

    def create_turbine(self, turbine_type: str) -> Turbine:
        """Create a turbine instance from configuration"""
        if turbine_type not in self.turbine_configs:
            raise ValueError(f"Unknown turbine type: {turbine_type}")

        config = self.turbine_configs[turbine_type]
        return Turbine(**config)
