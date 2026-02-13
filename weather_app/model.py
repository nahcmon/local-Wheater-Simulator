from __future__ import annotations

import numpy as np
import torch
from chronos import ChronosPipeline


class ForecastModel:
    def __init__(self, model_id: str = "amazon/chronos-bolt-small") -> None:
        self.model_id = model_id
        self._pipeline = None
        self._using_fallback = False

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def load(self) -> None:
        if self._pipeline is not None or self._using_fallback:
            return

        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
        except Exception:
            self._using_fallback = True

    def predict_quantiles(
        self,
        context_values: np.ndarray,
        prediction_length: int = 24,
        quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
        num_samples: int = 100,
    ) -> dict[float, np.ndarray]:
        self.load()

        if len(context_values) < 24:
            raise ValueError("Need at least 24 hourly points for forecasting.")

        if self._using_fallback:
            base = float(context_values[-1])
            seasonal = context_values[-24:]
            repeated = np.resize(seasonal, prediction_length)
            forecasts = np.tile(repeated, (num_samples, 1))
            noise = np.random.normal(0.0, 0.8, size=forecasts.shape)
            forecasts = 0.7 * forecasts + 0.3 * base + noise
            return {q: np.quantile(forecasts, q, axis=0) for q in quantiles}

        context = torch.tensor(context_values, dtype=torch.float32)
        samples = self._pipeline.predict(
            context=context,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )

        samples_np = samples.detach().cpu().numpy()
        if samples_np.ndim == 3:
            samples_np = samples_np[0]

        return {q: np.quantile(samples_np, q, axis=0) for q in quantiles}
