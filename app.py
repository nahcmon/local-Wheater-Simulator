from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from weather_app.model import ForecastModel
from weather_app.weather_api import Location, WeatherAPIError, fetch_hourly_weather, geocode_city

st.set_page_config(page_title="AI Weather Forecaster", page_icon="🌦️", layout="wide")

st.title("🌦️ AI Weather Forecaster")
st.caption("Live weather observations in, local Chronos-Bolt probabilistic forecast out.")

with st.sidebar:
    st.header("Forecast Settings")
    city = st.text_input("City", value="Berlin")
    forecast_hours = st.slider("Forecast horizon (hours)", min_value=6, max_value=72, value=24, step=6)
    past_days = st.slider("History window (days)", min_value=5, max_value=14, value=10)
    run = st.button("Run Forecast", type="primary", use_container_width=True)

if "model" not in st.session_state:
    st.session_state.model = ForecastModel()


def _build_plot(history: pd.DataFrame, out_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["timestamp"],
            y=history["temperature_2m"],
            mode="lines",
            name="Observed Temp",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=out_df["timestamp"],
            y=out_df["p90"],
            mode="lines",
            line=dict(width=0),
            name="P90",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=out_df["timestamp"],
            y=out_df["p10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.2)",
            name="Uncertainty (P10-P90)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=out_df["timestamp"],
            y=out_df["p50"],
            mode="lines",
            name="AI Forecast (P50)",
            line=dict(color="#ff7f0e", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=out_df["timestamp"],
            y=out_df["api_forecast"],
            mode="lines",
            name="Open-Meteo Baseline",
            line=dict(color="#2ca02c", dash="dot"),
        )
    )
    fig.update_layout(
        title="Temperature Forecast",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


if not run:
    st.info("Choose a location and click **Run Forecast**.")
else:
    try:
        with st.spinner("Fetching weather + running local inference..."):
            location: Location = geocode_city(city)
            df = fetch_hourly_weather(location.latitude, location.longitude, past_days=past_days, forecast_hours=forecast_hours)
            history = df[df["is_historical"]].copy()
            api_future = df[~df["is_historical"]].copy().head(forecast_hours)

            if len(history) < 24:
                raise WeatherAPIError("Not enough historical points returned by API. Try a bigger history window.")
            if len(api_future) < forecast_hours:
                raise WeatherAPIError("Not enough future points returned by API for selected horizon.")

            context = history["temperature_2m"].to_numpy(dtype=np.float32)
            quantile_preds = st.session_state.model.predict_quantiles(
                context_values=context,
                prediction_length=forecast_hours,
            )

            out_df = pd.DataFrame(
                {
                    "timestamp": api_future["timestamp"].to_numpy(),
                    "p10": quantile_preds[0.1],
                    "p50": quantile_preds[0.5],
                    "p90": quantile_preds[0.9],
                    "api_forecast": api_future["temperature_2m"].to_numpy(),
                }
            )

        st.subheader(f"Forecast for {location.label} ({location.latitude:.2f}, {location.longitude:.2f})")
        c1, c2, c3, c4 = st.columns(4)
        mae = float(np.mean(np.abs(out_df["p50"] - out_df["api_forecast"])))
        c1.metric("MAE vs API Baseline", f"{mae:.2f} °C")
        c2.metric("GPU Available", "Yes" if torch.cuda.is_available() else "No")
        c3.metric("Inference Engine", "Fallback" if st.session_state.model.using_fallback else "Chronos-Bolt")
        c4.metric("Forecast Horizon", f"{forecast_hours}h")

        tab1, tab2 = st.tabs(["Chart", "Data"])
        with tab1:
            st.plotly_chart(_build_plot(history, out_df), use_container_width=True)
        with tab2:
            st.dataframe(out_df.round(2), use_container_width=True)

    except WeatherAPIError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)
