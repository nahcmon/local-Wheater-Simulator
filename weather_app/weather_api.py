from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    label: str


def geocode_city(city: str) -> Location:
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    response = requests.get(GEOCODE_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not results:
        raise WeatherAPIError(f"City not found: {city}")

    top = results[0]
    label = ", ".join(part for part in [top.get("name"), top.get("admin1"), top.get("country")] if part)
    return Location(latitude=float(top["latitude"]), longitude=float(top["longitude"]), label=label)


def fetch_hourly_weather(
    lat: float,
    lon: float,
    past_days: int = 10,
    forecast_hours: int = 48,
) -> pd.DataFrame:
    past_hours = past_days * 24
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "past_hours": past_hours,
        "forecast_hours": forecast_hours,
        "timezone": "auto",
    }
    response = requests.get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly")
    if not hourly:
        raise WeatherAPIError("Hourly data missing from Open-Meteo response")

    api_timezone = payload.get("timezone")
    timestamps = pd.to_datetime(hourly["time"])
    if api_timezone:
        # Open-Meteo returns local clock times for timezone="auto"; localize them so
        # "now" is computed in the same timezone regardless of the host machine timezone.
        timestamps = timestamps.tz_localize(api_timezone)
        now = pd.Timestamp.now(tz=api_timezone)
    else:
        now = pd.Timestamp.now()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_2m": hourly["temperature_2m"],
            "relative_humidity_2m": hourly["relative_humidity_2m"],
            "wind_speed_10m": hourly["wind_speed_10m"],
        }
    ).dropna()

    df["is_historical"] = df["timestamp"] <= now
    return df.reset_index(drop=True)
