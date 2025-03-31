from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from . import Base


class ScenarioWindTurbine(Base):
    __tablename__ = "scenario_wind_turbines"

    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    turbine_type = Column(String, nullable=False)  # Added this field
    capacity = Column(Float, nullable=False)  # MW
    cut_in_speed = Column(Float, default=3.0)  # m/s
    cut_out_speed = Column(Float, default=25.0)  # m/s
    rated_speed = Column(Float, default=15.0)  # m/s
    quantity = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)

    scenario = relationship("Scenario", back_populates="wind_turbines")
