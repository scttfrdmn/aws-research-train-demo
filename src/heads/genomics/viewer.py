"""Genomics result viewer (SPEC §5): sequence saliency + motif logo.

The generic viewers dispatch here. Returns the per-position saliency track (from
the head's input-gradient) and a logomaker information-content matrix built from
the highest-saliency window — which reconstructs the implanted motif once
trained, and is flat noise when untrained (the naked-eye delta, #10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Saliency:
    """A per-position importance track + a logo matrix for the head's input."""

    sequence: str
    track: Any  # length-L importance (numpy)
    logo_matrix: Any  # pandas DataFrame (position × ACGT, information units) or None
    metric_name: str


class GenomicsViewer:
    """What marimo/Streamlit render for the genomics head."""

    def render(self, checkpoint: str, x: str) -> Saliency:
        from heads.genomics.head import HEAD

        track = HEAD.predict(checkpoint, x)
        return Saliency(
            sequence=x,
            track=track,
            logo_matrix=self._logo(x, track),
            metric_name="auroc",
        )

    def _logo(self, seq: str, track: Any, window: int = 6) -> Any:
        """Build a logomaker information matrix from the peak-saliency window.

        Kept import-local and best-effort: if logomaker isn't installed the
        viewer still returns the saliency track (logo_matrix=None).
        """
        try:
            import logomaker  # noqa: F401
            import numpy as np
            import pandas as pd
        except ImportError:
            return None
        if len(seq) < window:
            return None
        # center the window on the highest-saliency position
        peak = int(np.argmax(track))
        lo = max(0, min(peak - window // 2, len(seq) - window))
        sub = seq[lo : lo + window].upper()
        counts = pd.DataFrame(0.0, index=range(window), columns=list("ACGT"))
        for i, base in enumerate(sub):
            if base in "ACGT":
                counts.loc[i, base] = 1.0
        import logomaker as lm

        return lm.transform_matrix(counts, from_type="counts", to_type="information")
