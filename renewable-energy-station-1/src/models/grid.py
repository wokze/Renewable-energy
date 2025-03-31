from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class GridMode(Enum):
    EXPORT = "export"  # Normal mode - sending agreed power to grid
    IMPORT = "import"  # Emergency/EOC mode - drawing power for battery charging
    ISLAND = "island"  # Operating independently
    EMERGENCY = "emergency"  # Emergency mode when grid connection is lost
    FLEXIBLE = "flexible"  # Flexible mode - can accept excess energy


class Grid:
    def __init__(
        self,
        grid_id: str,
        capacity: float,  # MW
        voltage_level: float = 110.0,  # kV
        frequency: float = 50.0,  # Hz
        power_factor: float = 0.95,
        max_import: Optional[float] = None,  # MW, limited import for battery charging
        max_export: Optional[float] = None,  # MW, main operation mode
        initial_load_demand: float = 0.0,  # MW, initial external load demand
        flexible_capacity: Optional[float] = None,  # Additional MW that can be accepted
    ):
        # Existing initialization
        self.grid_id = grid_id
        self.capacity = capacity
        self.voltage_level = voltage_level
        self.frequency = frequency
        self.power_factor = power_factor
        self.max_import = max_import or (capacity * 0.2)
        self.max_export = max_export or capacity

        # Add flexible capacity handling
        self.flexible_capacity = flexible_capacity or (
            capacity * 0.3
        )  # Default 30% extra
        self.is_flexible_mode = False

        # Rest of initialization remains same
        self.external_load_demand = initial_load_demand
        self.total_load_demand = initial_load_demand
        self.current_load = 0.0
        self.current_mode = GridMode.EXPORT
        self.current_frequency = frequency
        self.current_voltage = voltage_level
        self.power_quality = 1.0
        self.load_history = []
        self.event_log = []

    def set_load_demand(self, new_load: float) -> None:
        """
        Set external load demand from the grid
        Args:
            new_load: Load demand in MW
        """
        if new_load < 0:
            raise ValueError("Load demand cannot be negative")
        if new_load > self.capacity:
            raise ValueError(
                f"Load demand cannot exceed grid capacity ({self.capacity} MW)"
            )

        self.external_load_demand = new_load
        self._update_grid_state()

    def _update_grid_state(self) -> None:
        """Update grid state after load changes"""
        # Update frequency based on load change
        load_ratio = self.external_load_demand / self.capacity
        freq_deviation = 0.5 * load_ratio  # Max 0.5 Hz deviation at full load
        self.current_frequency = self.frequency - freq_deviation

        # Update voltage based on load
        voltage_drop = 0.1 * load_ratio  # Max 10% voltage drop at full load
        self.current_voltage = self.voltage_level * (1 - voltage_drop)

        # Update power quality
        self.power_quality = max(0.6, 1 - (load_ratio * 0.4))  # Min 0.6 quality

        # Log the event
        self._log_event(
            renewable_generation=0,
            battery_power=0,
            load_demand=self.external_load_demand,
            event_type="load_change",
        )

    def set_flexible_mode(self, enabled: bool) -> None:
        """Enable or disable flexible load mode"""
        self.is_flexible_mode = enabled
        self.current_mode = GridMode.FLEXIBLE if enabled else GridMode.EXPORT
        self._log_event(
            0,
            0,
            self.external_load_demand,
            event_type=f"mode_change_{self.current_mode.value}",
        )

    def balance_power(
        self,
        renewable_generation: float,  # MW from PV and Wind
        battery_power: float,  # MW (positive for discharge, negative for charge)
        batteries_need_charging: bool = False,  # Flag for EOC charging mode
    ) -> Dict:
        """
        Balance power with total load demand (external + internal)
        """
        # Calculate net power (generation - total demand)
        net_power = renewable_generation + battery_power - self.external_load_demand

        grid_import = 0
        grid_export = 0
        excess_accepted = 0

        if batteries_need_charging and net_power < 0:
            # Only import if batteries need charging and we don't have enough renewable generation
            grid_import = min(-net_power, self.max_import)
            self.current_mode = GridMode.IMPORT
        else:
            if self.is_flexible_mode and net_power > self.max_export:
                # Handle excess power in flexible mode
                base_export = min(net_power, self.max_export)
                excess_power = net_power - base_export
                excess_accepted = min(excess_power, self.flexible_capacity)
                grid_export = base_export + excess_accepted
                self.current_mode = GridMode.FLEXIBLE
            else:
                # Normal operation - export excess power
                grid_export = min(net_power, self.max_export)
                self.current_mode = GridMode.EXPORT

        # Update current load (negative for export, positive for import)
        self.current_load = grid_import - grid_export

        # Log the power balance event
        self._log_event(renewable_generation, battery_power, self.external_load_demand)

        return {
            "timestamp": datetime.now().isoformat(),
            "mode": self.current_mode.value,
            "renewable_generation": renewable_generation,
            "battery_power": battery_power,
            "external_load": self.external_load_demand,
            "grid_import": grid_import,
            "grid_export": grid_export,
            "excess_accepted": excess_accepted,
            "net_power": net_power,
            "frequency": self.current_frequency,
            "voltage": self.current_voltage,
            "power_quality": self.power_quality,
        }

    def evaluate_grid_stability(self) -> Dict:
        """
        Evaluate grid stability based on current conditions
        """
        stability_score = 1.0

        # Check frequency deviation
        freq_deviation = abs(self.current_frequency - self.frequency)
        if freq_deviation > 0.5:  # Hz
            stability_score *= 0.8

        # Check voltage deviation
        voltage_deviation = (
            abs(self.current_voltage - self.voltage_level) / self.voltage_level
        )
        if voltage_deviation > 0.1:  # 10%
            stability_score *= 0.8

        # Check power factor
        if self.power_factor < 0.9:
            stability_score *= 0.9

        # Check loading
        loading_ratio = abs(self.current_load) / self.capacity
        if loading_ratio > 0.9:  # 90% capacity
            stability_score *= 0.7

        return {
            "stability_score": stability_score,
            "frequency_quality": 1 - (freq_deviation / 0.5),
            "voltage_quality": 1 - voltage_deviation,
            "loading_ratio": loading_ratio,
        }

    def simulate_grid_disturbance(
        self,
        frequency_drop: float = 0.0,
        voltage_drop: float = 0.0,
        duration: int = 10,  # seconds
    ) -> None:
        """
        Simulate grid disturbances to test system response
        """
        self.current_frequency -= frequency_drop
        self.current_voltage -= voltage_drop
        self.power_quality *= 0.8
        self._log_event(0, 0, 0, event_type="disturbance")

    def _log_event(
        self,
        renewable_generation: float,
        battery_power: float,
        load_demand: float,
        event_type: str = "balance",
    ) -> None:
        """Log grid events for analysis"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "mode": self.current_mode.value,
            "load": self.current_load,
            "renewable_generation": renewable_generation,
            "battery_power": battery_power,
            "load_demand": load_demand,
            "frequency": self.current_frequency,
            "voltage": self.current_voltage,
            "power_quality": self.power_quality,
        }
        self.event_log.append(event)

    def get_status(self) -> Dict:
        """Get current grid status including flexible mode info"""
        status = {
            "grid_id": self.grid_id,
            "mode": self.current_mode.value,
            "current_load": self.current_load,
            "capacity": self.capacity,
            "flexible_capacity": self.flexible_capacity,
            "is_flexible_mode": self.is_flexible_mode,
            "frequency": self.current_frequency,
            "voltage": self.current_voltage,
            "power_factor": self.power_factor,
            "power_quality": self.power_quality,
            "stability": self.evaluate_grid_stability(),
            "external_load_demand": self.external_load_demand,
            "total_load_demand": self.total_load_demand,
        }
        return status
