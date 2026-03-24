"""Optimal Stop Calculator — structural stop with sweep buffer, VIX scaling, and round-number avoidance."""
import structlog
from ghost.core.models import StopLevel
from typing import Optional

logger = structlog.get_logger()

ROUND_GRID = {
    "MNQ": [25, 50, 100, 250, 500, 1000],
    "NQ": [25, 50, 100, 250, 500, 1000],
    "ES": [5, 10, 25, 50, 100, 250],
    "MES": [5, 10, 25, 50, 100],
    "GC": [5, 10, 25, 50, 100],
    "CL": [0.25, 0.50, 1.00, 2.50, 5.00],
}

VIX_MULT = {
    (0, 20): 1.00,
    (20, 25): 1.15,
    (25, 35): 1.35,
    (35, 9999): 1.60,
}


class OptimalStopCalculator:
    """Calculates optimal stop placement using structural levels, sweep buffers, and VIX scaling."""

    def __init__(self, tick_sizes: dict, point_values: dict):
        self.tick_sizes = tick_sizes
        self.point_values = point_values

    def calculate(
        self, instrument: str, direction: str, fvg_low: float, ob_low: Optional[float],
        sweep_dist, current_atr: float, vix: float, max_risk_dollars: float,
    ) -> StopLevel:
        tick = self.tick_sizes.get(instrument, 0.25)
        pv = self.point_values.get(instrument, 2.0)

        structural = (
            min(fvg_low, ob_low if ob_low else fvg_low) if direction == "BULLISH"
            else max(fvg_low, ob_low if ob_low else fvg_low)
        )

        vm = next((m for (lo, hi), m in VIX_MULT.items() if lo <= vix < hi), 1.0)

        cand = (
            structural - sweep_dist.p75_sweep_ticks * tick * vm if direction == "BULLISH"
            else structural + sweep_dist.p75_sweep_ticks * tick * vm
        )

        # Round-number avoidance
        for ls in ROUND_GRID.get(instrument, [50, 100]):
            n = round(cand / ls) * ls
            if abs(cand - n) <= 3 * tick:
                cand = n - 4 * tick if direction == "BULLISH" else n + 4 * tick

        dr = (abs(cand - structural) / tick) * pv
        sm = min(1.0, max_risk_dollars / dr) if dr > max_risk_dollars else 1.0

        return StopLevel(
            cand, structural, int(sweep_dist.p75_sweep_ticks),
            dr > max_risk_dollars, sm, False, vm,
        )
