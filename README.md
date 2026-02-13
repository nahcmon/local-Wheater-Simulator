# Local Weather Forecaster (Windows + CUDA)

A practical, quality-first weather forecasting app for a **single RTX 3080 Ti 12GB** PC running Windows + CUDA.

## Model research (quality vs practical deployment)

For pure global benchmark leadership, models such as GraphCast and Pangu-Weather are excellent, but they are usually more complex to deploy and operate on consumer Windows setups.

For this project goal (single desktop GPU + robust local inference + real-time live input), the best practical choice is:

- **`amazon/chronos-bolt-small`**

Why this is the best fit here:
- Strong zero-shot probabilistic forecasting quality for time series.
- Runs comfortably on 12GB VRAM.
- Straightforward PyTorch + CUDA workflow.
- Works directly on live hourly weather observations to generate relevant local forecasts.

## App behavior

1. User enters city.
2. App geocodes location using Open-Meteo.
3. App fetches recent hourly weather + near-term future hours.
4. Historical segment feeds Chronos-Bolt locally.
5. App predicts future quantiles (P10/P50/P90) and shows:
   - nice forecast chart with uncertainty band,
   - comparison against Open-Meteo baseline,
   - tabular output.

## Windows installation

### One-time setup
Run:

```bat
setup.bat
```

This does all setup automatically:
- creates `.venv`,
- installs CUDA PyTorch (`cu121`),
- installs app dependencies,
- downloads Chronos-Bolt weights.

### Start app
Run:

```bat
start.bat
```

Then open the URL printed by Streamlit (typically `http://localhost:8501`).

## Data source

- Open-Meteo APIs (free, no key required).
