"""Regime detection data models."""
from dataclasses import dataclass


@dataclass
class RegimeState:
    label: str          # TRENDING_BULLISH, TRENDING_BEARISH, RANGING, VOLATILE, INVALIDATED
    direction: str      # BULLISH, BEARISH, NEUTRAL
    confidence: float   # 0.0 - 1.0
    adx: float
    atr: float
    atr_percentile: float
    volatility_regime: str  # LOW, NORMAL, HIGH, EXTREME
