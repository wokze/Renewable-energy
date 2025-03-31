from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from . import Base


class BatteryStatus(Base):
    __tablename__ = "battery_status"

    id = Column(Integer, primary_key=True)
    battery_id = Column(Integer, ForeignKey("scenario_batteries.id"))
    last_eoc = Column(TIMESTAMP)
    next_eoc = Column(TIMESTAMP)
    status = Column(String)  # Ready/Charging/Discharging/EOC/Error
    is_online = Column(Boolean, default=True)
    last_updated = Column(TIMESTAMP, default=datetime.utcnow)

    battery = relationship("ScenarioBattery", back_populates="battery_status")


class BatteryMetrics(Base):
    __tablename__ = "battery_metrics"

    time = Column(TIMESTAMP, primary_key=True)
    battery_id = Column(Integer, ForeignKey("scenario_batteries.id"), primary_key=True)
    charge_level = Column(Float, nullable=False)
    charge_rate = Column(Float)
    discharge_rate = Column(Float)
    temperature = Column(Float)
    voltage = Column(Float)
    state_of_charge = Column(Float)  # Percentage
    health = Column(Float)
    cycle_count = Column(Integer)
    energy_in = Column(Float)  # Energy charged in this hour
    energy_out = Column(Float)  # Energy discharged in this hour
    efficiency = Column(Float)  # Charging/discharging efficiency


class ScenarioBattery(Base):
    __tablename__ = "scenario_batteries"

    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    battery_type = Column(String, nullable=False)  # References battery_configs.json
    quantity = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)

    scenario = relationship("Scenario", back_populates="batteries")
    battery_status = relationship(
        "BatteryStatus", back_populates="battery", uselist=False
    )
