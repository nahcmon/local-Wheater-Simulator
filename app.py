from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from weather_app.model import ForecastModel
from weather_app.weather_api import WeatherAPIError, fetch_hourly_temperature, geocode_city

st.set_page_config(page_title="AI Weather Forecaster", page_icon="🌦️", layout="wide")

st.title("🌦️ AI Weather Forecaster (Chronos-Bolt)")
st.caption("Real-time weather data input + local AI forecast for the next 24 hours.")

with st.sidebar:
    st.header("Location")
    city = st.text_input("City", value="Berlin")
    forecast_hours = st.slider("Forecast horizon (hours)", min_value=6, max_value=72, value=24, step=6)
    past_days = st.slider("History window (days)", min_value=3, max_value=14, value=7)
    run = st.button("Run Forecast", type="primary")

if "model" not in st.session_state:
    st.session_state.model = ForecastModel()

if run:
    with st.spinner("Fetching live weather data and running model inference..."):
        try:
            lat, lon, label = geocode_city(city)
            df = fetch_hourly_temperature(lat, lon, past_days=past_days, forecast_hours=forecast_hours)

            history = df.iloc[:-forecast_hours].copy()
            future = df.iloc[-forecast_hours:].copy()

            context = history["temperature_2m"].to_numpy(dtype=np.float32)
            quantile_preds = st.session_state.model.predict_quantiles(
                context_values=context,
                prediction_length=forecast_hours,
            )

            pred_df = pd.DataFrame(
                {
                    "timestamp": future["timestamp"].to_numpy(),
                    "p10": quantile_preds[0.1],
                    "p50": quantile_preds[0.5],
                    "p90": quantile_preds[0.9],
                    "actual_from_api": future["temperature_2m"].to_numpy(),
                }
            )

            st.subheader(f"Forecast for {label} ({lat:.2f}, {lon:.2f})")

            col1, col2, col3 = st.columns(3)
            mae = float(np.mean(np.abs(pred_df["p50"] - pred_df["actual_from_api"])))
            col1.metric("Median Forecast MAE (vs API forecast)", f"{mae:.2f} °C")
            col2.metric("GPU Available", "Yes" if __import__("torch").cuda.is_available() else "No")
            col3.metric("Model Mode", "Fallback" if st.session_state.model.using_fallback else "Chronos")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=history["timestamp"],
                    y=history["temperature_2m"],
                    mode="lines",
                    name="Historical Temp",
                    line=dict(color="#1f77b4"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pred_df["timestamp"],
                    y=pred_df["p90"],
                    mode="lines",
                    name="P90",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pred_df["timestamp"],
                    y=pred_df["p10"],
                    mode="lines",
                    name="Uncertainty (P10-P90)",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(255, 127, 14, 0.25)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pred_df["timestamp"],
                    y=pred_df["p50"],
                    mode="lines",
                    name="AI Forecast (P50)",
                    line=dict(color="#ff7f0e", width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pred_df["timestamp"],
                    y=pred_df["actual_from_api"],
                    mode="lines",
                    name="Open-Meteo Forecast",
                    line=dict(color="#2ca02c", dash="dot"),
                )
            )
            fig.update_layout(
                title="Temperature Forecast",
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Forecast Table")
            st.dataframe(pred_df.round(2), use_container_width=True)

        except WeatherAPIError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)
else:
    st.info("Choose location and click **Run Forecast**.")
