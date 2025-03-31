import json
from pathlib import Path
from .grid import Grid


class GridFactory:
    def __init__(self, config_path: str = "config/grids.json"):
        self.config_path = Path(config_path)
        self.grid_configs = self._load_configs()

    def _load_configs(self) -> dict:
        """Load grid configurations from JSON file"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {}

    def get_available_grids(self) -> list:
        """Return list of available grid configurations"""
        return list(self.grid_configs.keys())

    def create_grid(self, grid_type: str) -> Grid:
        """Create a grid instance from configuration"""
        if grid_type not in self.grid_configs:
            raise ValueError(f"Unknown grid type: {grid_type}")

        config = self.grid_configs[grid_type]
        return Grid(**config)
