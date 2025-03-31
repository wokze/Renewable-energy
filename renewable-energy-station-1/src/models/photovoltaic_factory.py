# src/models/photovoltaic_factory.py
import json
from pathlib import Path
from .photovoltaic import Photovoltaic


class PhotovoltaicFactory:
    def __init__(self, config_path: str = "config/pv.json"):
        self.config_path = Path(config_path)
        self.pv_configs = self._load_configs()

    def _load_configs(self) -> dict:
        """Load photovoltaic configurations from JSON file"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {}

    def get_available_pvs(self) -> list:
        """Return list of available photovoltaic configurations"""
        return list(self.pv_configs.keys())

    def create_pv(self, pv_type: str) -> Photovoltaic:
        """Create a photovoltaic instance from configuration"""
        if pv_type not in self.pv_configs:
            raise ValueError(f"Unknown PV type: {pv_type}")

        config = self.pv_configs[pv_type]
        return Photovoltaic(**config)
