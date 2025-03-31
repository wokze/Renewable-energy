from sqlalchemy import Column, Integer, Float, Boolean, ForeignKey, String
from sqlalchemy.orm import relationship
from . import Base


class ScenarioPhotovoltaic(Base):
    __tablename__ = "scenario_photovoltaics"

    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    model_type = Column(String, nullable=False)  # Changed from pv_type
    capacity = Column(Float, nullable=False)  # kW
    efficiency = Column(Float, nullable=False)
    quantity = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)

    scenario = relationship("Scenario", back_populates="solar_panels")
