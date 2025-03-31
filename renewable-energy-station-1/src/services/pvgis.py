# src/services/pvgis_service.py
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime


class PVGISService:
    """Service for interacting with the PVGIS API"""

    BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_2/"

    def __init__(self):
        self.cache = {}  # Simple cache to avoid repeated calls

    def get_hourly_radiation(
        self,
        latitude: float,
        longitude: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get hourly radiation data for a location

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Optional start date in format YYYY-MM-DD
            end_date: Optional end date in format YYYY-MM-DD
            cache_key: Optional key for caching results

        Returns:
            Dictionary containing hourly radiation data
        """
        # Use cache if available
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]

        # Set up parameters
        params = {
            "lat": latitude,
            "lon": longitude,
            "outputformat": "json",
            "usehorizon": 1,
            "userhorizon": "",
            "angle": 35,  # Default tilt angle
            "aspect": 0,  # Default azimuth (south)
            "startyear": datetime.now().year - 1,
            "endyear": datetime.now().year,
            "database": "PVGIS-SARAH2",
        }

        if start_date and end_date:
            params["startyear"] = start_date.split("-")[0]
            params["endyear"] = end_date.split("-")[0]

        # Make the API request
        response = requests.get(f"{self.BASE_URL}seriescalc", params=params)

        if response.status_code != 200:
            raise Exception(f"PVGIS API error: {response.text}")

        data = response.json()

        # Cache the result
        if cache_key:
            self.cache[cache_key] = data

        return data

    def calculate_pv_generation(
        self,
        latitude: float,
        longitude: float,
        peak_power: float,  # kWp
        mounting_type: str = "free",  # 'free' or 'building'
        loss: float = 14.0,  # System losses in %
        angle: float = 35.0,  # Installation angle
        aspect: float = 0.0,  # Azimuth (0=south, 90=west)
        pv_technology: str = "crystSi",  # Default to crystalline silicon
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate PV generation for a specific system

        Args:
            latitude: Location latitude
            longitude: Location longitude
            peak_power: Peak power of the PV system in kWp
            mounting_type: Mounting type ('free' or 'building')
            loss: System losses in %
            angle: Installation angle in degrees
            aspect: Azimuth angle in degrees (0=south, 90=west)
            pv_technology: PV technology type
            cache_key: Optional key for caching results

        Returns:
            Dictionary with PV generation data
        """
        # Map our module_type to PVGIS technology types
        tech_mapping = {
            "Monocrystalline": "crystSi",
            "Polycrystalline": "crystSi",
            "CdTe": "CdTe",
            "CIS": "CIS",
        }

        if pv_technology in tech_mapping:
            pv_technology = tech_mapping[pv_technology]

        # Use cache if available
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]

        # Set up parameters
        params = {
            "lat": latitude,
            "lon": longitude,
            "peakpower": peak_power,
            "mountingplace": mounting_type,
            "loss": loss,
            "angle": angle,
            "aspect": aspect,
            "pvtechchoice": pv_technology,
            "outputformat": "json",
            "startyear": datetime.now().year - 1,
            "endyear": datetime.now().year,
        }

        # Make the API request
        response = requests.get(f"{self.BASE_URL}PVcalc", params=params)

        if response.status_code != 200:
            raise Exception(f"PVGIS API error: {response.text}")

        data = response.json()

        # Cache the result
        if cache_key:
            self.cache[cache_key] = data

        return data
