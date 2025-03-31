from sqlalchemy import Column, Integer, Float, String
from . import Base


class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timezone = Column(String, nullable=False)
    average_solar_hours = Column(Float)
    average_wind_speed = Column(Float)
