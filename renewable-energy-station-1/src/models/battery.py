import json
from pathlib import Path
from flask import Blueprint, render_template


class Battery:
    def __init__(
        self,
        model_name: str,
        capacity: float,
        max_charge_rate: float,
        max_discharge_rate: float,
        manufactured_date: str,
        eoc_voltage: float = 4.2,
        charge_level: float = 0.0,
        requires_eoc: bool = False,
        min_safe_charge: float = 0.15,
        max_safe_charge: float = 0.90,
        charge_rate_curve: dict = None,
        discharge_rate_curve: dict = None,
        temperature_limits: dict = None,
        cycle_life: int = 10000,
    ):

        self.model_name = model_name
        self.capacity = capacity  # GWh
        self.max_charge_rate = max_charge_rate  # GW
        self.max_discharge_rate = max_discharge_rate  # GW
        self.manufactured_date = manufactured_date
        self.eoc_voltage = eoc_voltage  # V
        self.charge_level = charge_level
        self.cycle_count = 0
        self.status = "Ready"

        # New attributes
        self.requires_eoc = requires_eoc
        self.min_safe_charge = min_safe_charge
        self.max_safe_charge = max_safe_charge
        self.charge_rate_curve = charge_rate_curve or {
            0.0: 1.0,  # 0% charge -> 100% rate
            0.5: 0.8,  # 50% charge -> 80% rate
            0.7: 0.6,  # 70% charge -> 60% rate
            0.8: 0.4,  # 80% charge -> 40% rate
            0.9: 0.2,  # 90% charge -> 20% rate
        }
        self.discharge_rate_curve = discharge_rate_curve or {
            0.2: 0.6,  # 20% charge -> 60% rate
            0.3: 0.8,  # 30% charge -> 80% rate
            0.5: 1.0,  # 50% charge -> 100% rate
            0.7: 0.9,  # 70% charge -> 90% rate
            0.9: 0.7,  # 90% charge -> 70% rate
        }
        self.temperature_limits = temperature_limits or {
            "min_operating": 0,
            "max_operating": 45,
            "optimal_min": 15,
            "optimal_max": 35,
        }
        self.cycle_life = cycle_life
        self.health = 100.0  # Battery health percentage

    def get_current_charge_rate(self) -> float:
        """Calculate current maximum charge rate based on charge level"""
        charge_percentage = self.charge_level / self.capacity

        # Find the appropriate rate multiplier
        rate_multiplier = 1.0
        for threshold, multiplier in sorted(self.charge_rate_curve.items()):
            if charge_percentage >= threshold:
                rate_multiplier = multiplier
            else:
                break

        return self.max_charge_rate * rate_multiplier

    def get_current_discharge_rate(self) -> float:
        """Calculate current maximum discharge rate based on charge level"""
        charge_percentage = self.charge_level / self.capacity

        # Find the appropriate rate multiplier
        rate_multiplier = 1.0
        for threshold, multiplier in sorted(self.discharge_rate_curve.items()):
            if charge_percentage >= threshold:
                rate_multiplier = multiplier
            else:
                break

        return self.max_discharge_rate * rate_multiplier

    def charge(self, amount: float) -> float:
        """Charge the battery with rate limiting and safety checks"""
        if amount < 0:
            raise ValueError("Charge amount must be positive")

        current_max_rate = self.get_current_charge_rate()
        if amount > current_max_rate:
            amount = current_max_rate

        prev_level = self.charge_level
        target_level = self.charge_level + amount

        # Check charging limits
        if self.requires_eoc and not self.status.startswith("EOC"):
            target_level = min(target_level, self.capacity * self.max_safe_charge)

        self.charge_level = min(self.capacity, target_level)

        # Update status and cycles
        if self.is_full():
            self.end_of_charge()
        elif prev_level < self.capacity and self.charge_level >= self.capacity:
            self.cycle_count += 1
            self.update_health()

        return self.charge_level - prev_level  # Return actual amount charged

    def discharge(self, amount: float) -> None:
        """Discharge the battery with rate limiting"""
        if amount < 0:
            raise ValueError("Discharge amount must be positive")
        if amount > self.max_discharge_rate:
            raise ValueError(
                f"Discharge rate cannot exceed {self.max_discharge_rate} GW"
            )
        if amount > self.charge_level:
            raise ValueError("Not enough charge")

        self.charge_level -= amount

    def get_charge_level(self) -> float:
        return self.charge_level

    def get_status(self) -> dict:
        return {
            "model": self.model_name,
            "capacity": self.capacity,
            "charge_level": self.charge_level,
            "charge_percentage": (self.charge_level / self.capacity) * 100,
            "cycles": self.cycle_count,
            "status": self.status,
        }

    def is_full(self) -> bool:
        return self.charge_level >= self.capacity

    def is_empty(self) -> bool:
        return self.charge_level <= 0.0

    def end_of_charge(self) -> None:
        """End of Charge management"""
        if self.is_full():
            self.status = "Fully Charged - EOC Active"
            # Additional EOC logic like voltage monitoring
            if self.get_voltage() >= self.eoc_voltage:
                self.status = "EOC Complete"

    def get_voltage(self) -> float:
        """Simulate voltage reading"""
        # Simple voltage simulation based on charge level
        min_voltage = 3.2
        voltage_range = self.eoc_voltage - min_voltage
        return min_voltage + (voltage_range * (self.charge_level / self.capacity))

    def update_health(self):
        """Update battery health based on cycles"""
        self.health = max(0, 100 * (1 - (self.cycle_count / self.cycle_life)))


class BatteryFactory:
    def __init__(self, config_path: str = "./config/battery_configs.json"):
        self.config_path = Path(config_path)
        self.battery_configs = self._load_configs()

    def _load_configs(self) -> dict:
        """Load battery configurations from JSON file"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {}

    def get_available_batteries(self) -> list:
        """Return list of available battery configurations"""
        return list(self.battery_configs.keys())

    def create_battery(self, battery_type: str) -> Battery:
        """Create a battery instance from configuration"""
        if battery_type not in self.battery_configs:
            raise ValueError(f"Unknown battery type: {battery_type}")

        config = self.battery_configs[battery_type]
        return Battery(**config)

    def add_battery_config(self, name: str, config: dict) -> None:
        """Add a new battery configuration"""
        self.battery_configs[name] = config
        with open(self.config_path, "w") as f:
            json.dump(self.battery_configs, f, indent=4)


# Example batteries:

# 1. Moss Landing Energy Storage (California)
moss_landing = Battery(
    model_name="Vistra Moss Landing",
    capacity=1.6,  # 1.6 GWh
    max_charge_rate=0.4,  # 400 MW
    max_discharge_rate=0.4,
    manufactured_date="2021-08",
    eoc_voltage=4.2,
)

# 2. Hornsdale Power Reserve (Australia)
hornsdale = Battery(
    model_name="Tesla Hornsdale",
    capacity=0.194,  # 194 MWh
    max_charge_rate=0.150,  # 150 MW
    max_discharge_rate=0.193,
    manufactured_date="2017-12",
    eoc_voltage=4.2,
)

bp = Blueprint("batteries", __name__, url_prefix="/batteries")

# Get absolute path to config file
config_path = Path("/app/config/battery_configs.json")
battery_factory = BatteryFactory(str(config_path))


@bp.route("/")
def list_batteries():
    try:
        available_batteries = battery_factory.get_available_batteries()
        print(f"Found batteries: {available_batteries}")  # Debug line
        return render_template(
            "batteries.html",
            batteries=available_batteries,
            battery_factory=battery_factory,
        )
    except Exception as e:
        print(f"Error loading batteries: {e}")
        return render_template(
            "batteries.html",
            batteries=[],
            battery_factory=battery_factory,
            error=str(e),
        )
