from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

metadata = MetaData()
Base = declarative_base(metadata=metadata)

# Import all models to make them available
from .battery import BatteryStatus, BatteryMetrics, ScenarioBattery
from .scenario import Scenario
from .location import Location
from .grid import ScenarioGrid
from .photovoltaic import ScenarioPhotovoltaic
from .wind_turbine import ScenarioWindTurbine

__all__ = [
    "Base",
    "BatteryStatus",
    "BatteryMetrics",
    "ScenarioBattery",
    "Scenario",
    "Location",
    "ScenarioGrid",
    "ScenarioPhotovoltaic",
    "ScenarioWindTurbine",
]
