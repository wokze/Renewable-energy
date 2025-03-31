# src/services/pvgis_service.py
import requests
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os
import time


class PVGISService:
    """Service for interacting with the PVGIS API"""

    BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_2/"

    def __init__(self):
        self.cache_dir = "/app/tmp/pvgis_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, cache_key):
        """Get the path for a cached file"""
        if not cache_key:
            return None
        return os.path.join(self.cache_dir, f"{cache_key.replace('/', '_')}.json")

    def _get_from_cache(self, cache_key):
        """Try to get data from cache"""
        if not cache_key:
            return None

        cache_path = self._get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None

        # Check if cache is valid (less than 1 day old)
        if time.time() - os.path.getmtime(cache_path) > 86400:
            return None

        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except:
            return None

    def _save_to_cache(self, cache_key, data):
        """Save data to cache"""
        if not cache_key:
            return

        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, "w") as f:
            json.dump(data, f)

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
        # Try to get from cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

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

        try:
            # Make the API request
            response = requests.get(f"{self.BASE_URL}PVcalc", params=params)

            if response.status_code != 200:
                raise Exception(f"PVGIS API error: {response.text}")

            data = response.json()

            # Save to cache
            self._save_to_cache(cache_key, data)

            return data

        except Exception as e:
            # If API request fails, return a simulated response for development
            print(f"PVGIS API error (using simulated data): {str(e)}")
            simulated_data = self._generate_simulated_data(peak_power, latitude)
            self._save_to_cache(cache_key, simulated_data)
            return simulated_data

    def _generate_simulated_data(self, peak_power, latitude):
        """Generate simulated PV data for development when API is unavailable"""
        import math
        import random
        from datetime import datetime, timedelta

        # Basic structure of PVGIS response
        data = {
            "inputs": {
                "location": {"latitude": latitude, "longitude": 0.0},
                "meteo_data": {
                    "radiation_db": "PVGIS-SARAH2",
                    "year_min": 2005,
                    "year_max": 2020,
                },
                "mounting_system": {
                    "fixed": {"slope": {"value": 35}, "azimuth": {"value": 0}}
                },
                "pv_module": {"technology": "c-Si", "peak_power": peak_power},
                "system_loss": 14,
            },
            "outputs": {"monthly": [], "hourly": []},
        }

        # Generate monthly data
        monthly_data = []
        for month in range(1, 13):
            # Solar irradiation is higher in summer months, lower in winter
            season_factor = 1.0 + 0.8 * math.sin((month - 3) * math.pi / 6.0)
            # Latitude affects seasonality (stronger seasons at higher latitudes)
            latitude_factor = min(1.0, abs(latitude) / 60.0)

            monthly_item = {
                "month": month,
                "E_d": round(
                    peak_power * 4.0 * season_factor * random.uniform(0.9, 1.1), 2
                ),  # Daily energy
                "E_m": 0,  # Will calculate after
                "H(i)_d": round(
                    4.5 * season_factor * random.uniform(0.9, 1.1), 2
                ),  # Irradiation on plane
                "H(i)_m": 0,  # Will calculate after
                "SD_m": round(random.uniform(5, 15), 2),  # Standard deviation
            }

            # Calculate monthly values
            days_in_month = (
                31 if month in [1, 3, 5, 7, 8, 10, 12] else (29 if month == 2 else 30)
            )
            monthly_item["E_m"] = round(monthly_item["E_d"] * days_in_month, 2)
            monthly_item["H(i)_m"] = round(monthly_item["H(i)_d"] * days_in_month, 2)

            monthly_data.append(monthly_item)

        # Generate hourly data for a sample day in each month
        hourly_data = []
        start_date = datetime(datetime.now().year, 1, 15)  # Jan 15th

        for month in range(12):
            sample_date = start_date + timedelta(days=30 * month)

            for hour in range(24):
                hour_angle = (hour - 12) * 15  # Solar hour angle

                if -90 <= hour_angle <= 90:  # Daytime
                    # Solar elevation model - highest at noon, zero at sunrise/sunset
                    elevation_factor = math.cos(math.radians(hour_angle))

                    # Seasonal factor from our monthly data
                    month_idx = sample_date.month - 1
                    season_factor = monthly_data[month_idx]["H(i)_d"] / 5.0

                    # Calculate simulated power (with some randomness)
                    if 6 <= hour <= 18:  # Typical daylight hours
                        power = (
                            peak_power
                            * season_factor
                            * elevation_factor
                            * random.uniform(0.7, 1.0)
                        )
                    else:
                        power = 0
                else:
                    power = 0

                hourly_item = {
                    "time": sample_date.strftime("%Y%m%d:%H") + "00",
                    "P": round(max(0, power), 3),  # Power in kW
                }
                hourly_data.append(hourly_item)

        data["outputs"]["monthly"] = monthly_data
        data["outputs"]["hourly"] = hourly_data

        return data
