"""Weather result viewer (SPEC §5): predicted | truth | error triptych.

The viewers (marimo + Streamlit) stay generic and dispatch here. Returns the
three fields a heatmap triptych draws — the naked-eye behavior delta is the
``error`` map collapsing toward zero as the model trains (verify-first #9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Triptych:
    """Three (H,W) fields for the predicted | truth | error heatmaps."""

    predicted: Any
    truth: Any
    error: Any
    metric_name: str


class WeatherViewer:
    """What marimo/Streamlit render for the weather head."""

    def render(self, checkpoint: str, x: Any) -> Triptych:
        """Restore field `x` under `checkpoint`; truth optional (passed via x).

        `x` may be a degraded input field, or a (input, truth) pair. Returns the
        triptych; if no truth is supplied, `truth`/`error` are None.
        """
        from heads.weather.head import HEAD

        if isinstance(x, tuple):
            inp, truth = x
        else:
            inp, truth = x, None
        pred = HEAD.predict(checkpoint, inp)
        error = None if truth is None else (pred - truth).abs()
        return Triptych(predicted=pred, truth=truth, error=error, metric_name="rmse")
