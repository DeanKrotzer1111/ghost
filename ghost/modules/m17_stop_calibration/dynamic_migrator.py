"""Dynamic Stop Migrator — ratchets stops using R-multiple thresholds and structural FVG shifts."""
import structlog
from dataclasses import dataclass
from typing import Optional

logger = structlog.get_logger()

RATCHET = [
    (0.5, 0.0), (1.0, 0.25), (1.5, 0.50),
    (2.0, 1.00), (3.0, 1.50), (4.0, 2.50),
]


@dataclass
class StopMigration:
    new_stop: float
    reason: str
    confidence: float


class DynamicStopMigrator:
    """Migrates stops using R-multiple ratcheting, new FVG structures, and higher-low detection."""

    def __init__(self, tick_sizes: dict):
        self.tick_sizes = tick_sizes

    def evaluate(
        self, instrument: str, direction: str, entry: float, current_stop: float,
        current_price: float, new_fvgs: list, recent_bars: list, sweep_p50_ticks: float,
    ) -> Optional[StopMigration]:
        tick = self.tick_sizes.get(instrument, 0.25)
        risk = abs(entry - current_stop)
        if risk <= 0:
            return None

        cr = (
            (current_price - entry) / risk if direction == "BULLISH"
            else (entry - current_price) / risk
        )

        # R-multiple ratcheting
        for rt, fr in reversed(RATCHET):
            if cr >= rt:
                ns = entry + fr * risk if direction == "BULLISH" else entry - fr * risk
                if direction == "BULLISH" and ns > current_stop:
                    return StopMigration(ns, f"RATCHET_{rt}R", 0.95)
                elif direction == "BEARISH" and ns < current_stop:
                    return StopMigration(ns, f"RATCHET_{rt}R", 0.95)
                break

        # New FVG structural migration
        if direction == "BULLISH":
            good = [f for f in new_fvgs if getattr(f, "low", 0) > current_stop + 5 * tick]
            if good:
                ns = max(good, key=lambda f: f.low).low - sweep_p50_ticks * tick
                if ns > current_stop:
                    return StopMigration(ns, "NEW_FVG_STRUCTURAL", 0.85)

        # Higher-low detection
        if direction == "BULLISH" and len(recent_bars) >= 5:
            if recent_bars[-1].low > min(b.low for b in recent_bars[-5:-1]):
                ns = recent_bars[-1].low - sweep_p50_ticks * tick
                if ns > current_stop:
                    return StopMigration(ns, "NEW_HIGHER_LOW", 0.78)

        return None
