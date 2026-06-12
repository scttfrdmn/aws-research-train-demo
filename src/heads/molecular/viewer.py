"""Molecular result viewer (SPEC §5).

Declares what the generic viewers (marimo + Streamlit, v0.3) render for this
head: a molecule render + a predicted-vs-actual scatter. The viewers stay
domain-blind and dispatch here. Kept thin until v0.3 — it exposes the render
contract and the data a scatter needs, computed via the head's ``predict``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ScatterData:
    """Predicted-vs-actual points for one checkpoint — the behavior-delta view.

    Untrained → a scattered cloud; trained → tight along y=x (SPEC §3,
    verify-first #2). The viewer plots ``actual`` vs ``predicted``.
    """

    actual: list[float]
    predicted: list[float]
    metric_name: str


class MolecularViewer:
    """What marimo/Streamlit render for the molecular head."""

    def render(self, checkpoint: str, x: Any) -> dict[str, Any]:
        """Head-native view of one SMILES ``x`` under ``checkpoint``.

        Returns the pieces the surface draws: the SMILES (for a 2D molecule
        render) and the model's predicted log-solubility. The full scatter is
        built by the surface over a held-out set via ``scatter`` below.
        """
        from heads.molecular.head import HEAD

        return {"smiles": x, "predicted": HEAD.predict(checkpoint, x)}

    def scatter(
        self, checkpoint: str, smiles: list[str], actual: list[float]
    ) -> ScatterData:
        """Predicted-vs-actual over a set — the naked-eye behavior delta."""
        from heads.molecular.head import HEAD

        predicted = [HEAD.predict(checkpoint, s) for s in smiles]
        return ScatterData(actual=actual, predicted=predicted, metric_name="rmse")
