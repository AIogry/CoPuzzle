"""Compatibility exports for Lights Out; use copuzzle.games.lights_out.diagnostics."""

from .games.lights_out.diagnostics import (
    prediction_metrics,
    composition_examples,
    EffectLookup,
)

__all__ = [
    "prediction_metrics",
    "composition_examples",
    "EffectLookup",
]
