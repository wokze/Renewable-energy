from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from . import Base


class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)

    # Relationships
    location = relationship("Location", backref="scenario")
    batteries = relationship("ScenarioBattery", back_populates="scenario")
    solar_panels = relationship("ScenarioPhotovoltaic", back_populates="scenario")
    wind_turbines = relationship("ScenarioWindTurbine", back_populates="scenario")
    grid_connections = relationship("ScenarioGrid", back_populates="scenario")
