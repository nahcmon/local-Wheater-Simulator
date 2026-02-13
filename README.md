# Local Weather Forecaster (Windows + CUDA)

This project builds a **local AI weather-forecast app** optimized for a 12GB RTX 3080 Ti environment.

## Model research (quality-first, single-GPU reality)

### Candidates reviewed

1. **GraphCast (DeepMind/ECMWF)**
   - Quality: state-of-the-art global medium-range skill in many benchmarks.
   - Constraint: inference/training stack is not straightforward for consumer Windows CUDA workflows; operational use is usually TPU/large infra centric.

2. **Pangu-Weather (Huawei)**
   - Quality: excellent global NWP benchmark performance.
   - Constraint: model variants and runtime stacks can be heavy for a straightforward 12GB desktop deployment, especially with full-resolution workflows.

3. **FourCastNet / FourCastNetV2 (NVIDIA/ECMWF ecosystem)**
   - Quality: strong global skill and fast autoregressive inference relative to classical NWP.
   - Constraint: practical deployment at high global resolution can still be memory and engineering intensive.

4. **Chronos-Bolt (Amazon) for local point forecasting from real-time observations**
   - Quality: very strong zero-shot forecasting quality across many real-world time-series tasks.
   - Advantage: easy local deployment from Hugging Face with PyTorch/CUDA, excellent fit for a 12GB 3080 Ti when forecasting **location-specific weather variables** (e.g., temperature, humidity, wind).

## Final model choice for this app

For the explicit constraint "best practical quality on a single 12GB 3080 Ti with easy Windows deployment", this app uses:

- **`amazon/chronos-bolt-small`**

Why:
- Delivers high-quality probabilistic forecasts for point time series.
- Fits comfortably in 12GB VRAM with room for UI/runtime overhead.
- Simple and robust deployment path on Windows CUDA via PyTorch.
- Enables real-time forecasting by ingesting live weather observations.

---

## What the app does

- Accepts a city name (or lat/lon).
- Pulls **live + recent hourly weather observations** from Open-Meteo.
- Runs local inference with Chronos-Bolt to forecast next 24 hours.
- Displays:
  - historical context,
  - median forecast,
  - uncertainty interval (10th–90th percentile),
  - data table for practical usage.

## Windows quick start

1. Run one-time setup:
   - `setup.bat`
2. Start app:
   - `start.bat`
3. Open browser at the Streamlit URL shown in terminal.

## Notes

- Setup will automatically use CUDA-enabled PyTorch when possible.
- If CUDA is unavailable, the app falls back to CPU.
- Data source: Open-Meteo public APIs (no key required).
