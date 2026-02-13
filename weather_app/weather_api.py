from __future__ import annotations

import requests
import pandas as pd

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherAPIError(RuntimeError):
    pass


def geocode_city(city: str) -> tuple[float, float, str]:
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    response = requests.get(GEOCODE_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not results:
        raise WeatherAPIError(f"City not found: {city}")

    top = results[0]
    label = ", ".join(
        part for part in [top.get("name"), top.get("admin1"), top.get("country")] if part
    )
    return float(top["latitude"]), float(top["longitude"]), label


def fetch_hourly_temperature(lat: float, lon: float, past_days: int = 7, forecast_hours: int = 24) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "past_days": past_days,
        "forecast_hours": forecast_hours,
        "timezone": "auto",
    }
    response = requests.get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly")
    if not hourly:
        raise WeatherAPIError("Hourly data missing from Open-Meteo response")

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature_2m": hourly["temperature_2m"],
            "relative_humidity_2m": hourly["relative_humidity_2m"],
            "wind_speed_10m": hourly["wind_speed_10m"],
        }
    )
    return df.dropna().reset_index(drop=True)
