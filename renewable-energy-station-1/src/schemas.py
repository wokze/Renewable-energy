class ScenarioPhotovoltaic(Base):
    __tablename__ = "scenario_photovoltaics"

    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    model_type = Column(String, nullable=False)  # Changed from pv_type
    quantity = Column(Integer, default=1)

    # Relationship
    scenario = relationship("Scenario", back_populates="solar_panels")
