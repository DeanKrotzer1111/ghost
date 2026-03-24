"""Fair Value Gap (FVG) detection for ICT signal engine."""
from dataclasses import dataclass, field
from typing import List, Optional

import structlog

from ghost.modules.m01_data.models import Bar

logger = structlog.get_logger(__name__)


@dataclass
class FVG:
    """Represents a Fair Value Gap."""
    high: float
    low: float
    direction: str  # "bullish" or "bearish"
    timeframe: str = ""
    formation_time: float = 0.0
    mitigated: bool = False
    mitigation_time: Optional[float] = None

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def size(self) -> float:
        return self.high - self.low


class FVGDetector:
    """Detects Fair Value Gaps across any timeframe of bars.

    Bullish FVG: bar[i-2].high < bar[i].low  (gap up)
    Bearish FVG: bar[i-2].low > bar[i].high  (gap down)
    """

    def __init__(self, timeframe: str = ""):
        self.timeframe = timeframe
        self._active_fvgs: List[FVG] = []
        self._mitigated_fvgs: List[FVG] = []
        self._log = logger.bind(component="FVGDetector", tf=timeframe)

    @property
    def active_fvgs(self) -> List[FVG]:
        return list(self._active_fvgs)

    @property
    def all_fvgs(self) -> List[FVG]:
        return list(self._active_fvgs) + list(self._mitigated_fvgs)

    def detect(self, bars: List[Bar]) -> List[FVG]:
        """Scan a list of bars and return all newly detected FVGs.

        Requires at least 3 bars. Scans the full list and also updates
        mitigation status for all previously-tracked FVGs.
        """
        if len(bars) < 3:
            return []

        new_fvgs: List[FVG] = []

        for i in range(2, len(bars)):
            bar_left = bars[i - 2]
            bar_mid = bars[i - 1]
            bar_right = bars[i]

            # Bullish FVG: gap between bar_left high and bar_right low
            if bar_right.low > bar_left.high:
                fvg = FVG(
                    high=bar_right.low,
                    low=bar_left.high,
                    direction="bullish",
                    timeframe=self.timeframe,
                    formation_time=bar_mid.timestamp,
                )
                if not self._is_duplicate(fvg):
                    new_fvgs.append(fvg)
                    self._active_fvgs.append(fvg)
                    self._log.debug("bullish_fvg_detected",
                                    high=fvg.high, low=fvg.low,
                                    size=fvg.size, ts=fvg.formation_time)

            # Bearish FVG: gap between bar_right high and bar_left low
            if bar_left.low > bar_right.high:
                fvg = FVG(
                    high=bar_left.low,
                    low=bar_right.high,
                    direction="bearish",
                    timeframe=self.timeframe,
                    formation_time=bar_mid.timestamp,
                )
                if not self._is_duplicate(fvg):
                    new_fvgs.append(fvg)
                    self._active_fvgs.append(fvg)
                    self._log.debug("bearish_fvg_detected",
                                    high=fvg.high, low=fvg.low,
                                    size=fvg.size, ts=fvg.formation_time)

        # Update mitigation for all active FVGs using the latest bar
        if bars:
            self._update_mitigation(bars[-1])

        return new_fvgs

    def update(self, bar: Bar) -> None:
        """Incrementally update mitigation status with a new bar."""
        self._update_mitigation(bar)

    def _update_mitigation(self, bar: Bar) -> None:
        """Check if the bar mitigates (fills) any active FVGs."""
        still_active: List[FVG] = []
        for fvg in self._active_fvgs:
            if fvg.mitigated:
                self._mitigated_fvgs.append(fvg)
                continue

            mitigated = False
            if fvg.direction == "bullish":
                # Price trades down into the gap — low touches or breaches fvg midpoint
                if bar.low <= fvg.midpoint:
                    mitigated = True
            elif fvg.direction == "bearish":
                # Price trades up into the gap — high touches or breaches fvg midpoint
                if bar.high >= fvg.midpoint:
                    mitigated = True

            if mitigated:
                fvg.mitigated = True
                fvg.mitigation_time = bar.timestamp
                self._mitigated_fvgs.append(fvg)
                self._log.debug("fvg_mitigated", direction=fvg.direction,
                                high=fvg.high, low=fvg.low, ts=bar.timestamp)
            else:
                still_active.append(fvg)

        self._active_fvgs = still_active

    def _is_duplicate(self, fvg: FVG) -> bool:
        """Check if an FVG with the same bounds already exists."""
        for existing in self._active_fvgs:
            if (existing.high == fvg.high and existing.low == fvg.low
                    and existing.direction == fvg.direction):
                return True
        for existing in self._mitigated_fvgs:
            if (existing.high == fvg.high and existing.low == fvg.low
                    and existing.direction == fvg.direction):
                return True
        return False

    def get_nearest_fvg(self, price: float, direction: str = "") -> Optional[FVG]:
        """Return the nearest unmitigated FVG to the given price.

        Args:
            price: Current price to measure distance from.
            direction: Filter by 'bullish' or 'bearish'. Empty = any.
        """
        candidates = self._active_fvgs
        if direction:
            candidates = [f for f in candidates if f.direction == direction]
        if not candidates:
            return None
        return min(candidates, key=lambda f: abs(f.midpoint - price))

    def reset(self) -> None:
        """Clear all tracked FVGs."""
        self._active_fvgs.clear()
        self._mitigated_fvgs.clear()
