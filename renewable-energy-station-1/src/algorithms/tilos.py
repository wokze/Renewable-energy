# src/algorithms/tilos.py
import numpy as np
from datetime import datetime, timedelta
from src.models.battery import BatteryFactory
from src.models.photovoltaic_factory import PhotovoltaicFactory
from src.models.turbine_factory import TurbineFactory
from pathlib import Path


class TILOSOptimizer:
    def __init__(self, scenario, time_step=1, forecast_horizon=24):
        self.scenario = scenario
        self.time_step = time_step  # hours
        self.forecast_horizon = forecast_horizon  # hours

        # Initialize factories to get component details
        self.battery_factory = BatteryFactory("/app/config/battery_configs.json")
        self.pv_factory = PhotovoltaicFactory("/app/config/pv.json")
        self.turbine_factory = TurbineFactory("/app/config/turbines.json")

    def optimize(self, solar_forecast, wind_forecast, demand_forecast):
        """
        Run TILOS optimization algorithm for energy balancing

        Args:
            solar_forecast: Hourly solar generation forecast (array)
            wind_forecast: Hourly wind generation forecast (array)
            demand_forecast: Hourly energy demand forecast (array)

        Returns:
            Dict containing battery dispatch schedule, grid import/export
        """
        # Get scenario components
        batteries = self._get_batteries()
        solar_panels = self._get_solar_panels()
        wind_turbines = self._get_wind_turbines()
        grid = self._get_grid()

        # Get battery models and calculate total capacity
        battery_models = {}
        for b in batteries:
            if b.battery_type not in battery_models:
                battery_models[b.battery_type] = self.battery_factory.create_battery(
                    b.battery_type
                )

        # Initialize result arrays
        hours = len(solar_forecast)
        battery_dispatch = np.zeros(hours)
        grid_exchange = np.zeros(hours)
        battery_soc = np.zeros(hours + 1)  # State of charge

        # Set initial battery state using factory models
        total_capacity = sum(
            battery_models[b.battery_type].capacity * b.quantity for b in batteries
        )
        battery_soc[0] = total_capacity * 0.5  # Start at 50% capacity

        # Run optimization for each time step
        for t in range(hours):
            # Calculate net generation (renewable generation minus demand)
            renewable_gen = solar_forecast[t] + wind_forecast[t]
            net_generation = renewable_gen - demand_forecast[t]

            # TILOS decision logic
            if net_generation > 0:  # Excess generation
                # Charge batteries if possible, export remainder to grid
                max_charge = self._get_max_charge_rate(batteries, battery_models)
                battery_charge = min(net_generation, max_charge)
                grid_export = max(0, net_generation - battery_charge)

                battery_dispatch[t] = -battery_charge  # Negative = charging
                grid_exchange[t] = -grid_export  # Negative = export

                # Update battery state
                battery_soc[t + 1] = min(
                    total_capacity, battery_soc[t] + battery_charge
                )

            else:  # Generation deficit
                # Discharge batteries if possible, import remainder from grid
                deficit = -net_generation
                max_discharge = self._get_max_discharge_rate(batteries, battery_models)
                battery_discharge = min(deficit, max_discharge, battery_soc[t])
                grid_import = deficit - battery_discharge

                battery_dispatch[t] = battery_discharge  # Positive = discharging
                grid_exchange[t] = grid_import  # Positive = import

                # Update battery state
                battery_soc[t + 1] = battery_soc[t] - battery_discharge

        return {
            "battery_dispatch": battery_dispatch.tolist(),
            "grid_exchange": grid_exchange.tolist(),
            "battery_soc": battery_soc.tolist(),
            "renewable_generation": [
                solar_forecast[t] + wind_forecast[t] for t in range(hours)
            ],
            "demand": demand_forecast,
            "timestamps": [
                (datetime.now() + timedelta(hours=t)).isoformat() for t in range(hours)
            ],
        }

    def _get_batteries(self):
        return self.scenario.batteries

    def _get_solar_panels(self):
        return self.scenario.solar_panels

    def _get_wind_turbines(self):
        return self.scenario.wind_turbines

    def _get_grid(self):
        return (
            self.scenario.grid_connections[0]
            if self.scenario.grid_connections
            else None
        )

    def _get_max_charge_rate(self, batteries, battery_models):
        return sum(
            battery_models[b.battery_type].max_charge_rate * b.quantity
            for b in batteries
        )

    def _get_max_discharge_rate(self, batteries, battery_models):
        return sum(
            battery_models[b.battery_type].max_discharge_rate * b.quantity
            for b in batteries
        )
