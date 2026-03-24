"""Order Block (OB) detection for ICT signal engine."""
from dataclasses import dataclass, field
from typing import List, Optional

import structlog

from ghost.modules.m01_data.models import Bar

logger = structlog.get_logger(__name__)


@dataclass
class OrderBlock:
    """Represents an Order Block zone."""
    high: float
    low: float
    direction: str  # "bullish" or "bearish"
    timeframe: str = ""
    formation_time: float = 0.0
    swept: bool = False
    sweep_time: Optional[float] = None

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def size(self) -> float:
        return self.high - self.low


class OrderBlockDetector:
    """Detects Order Blocks based on ICT methodology.

    Bullish OB: the last bearish candle before a strong bullish move (> 1.5x ATR in 3 bars).
    Bearish OB: the last bullish candle before a strong bearish move (> 1.5x ATR in 3 bars).
    """

    def __init__(self, timeframe: str = "", atr_multiplier: float = 1.5,
                 move_lookforward: int = 3):
        self.timeframe = timeframe
        self.atr_multiplier = atr_multiplier
        self.move_lookforward = move_lookforward
        self._active_obs: List[OrderBlock] = []
        self._swept_obs: List[OrderBlock] = []
        self._log = logger.bind(component="OrderBlockDetector", tf=timeframe)

    @property
    def active_obs(self) -> List[OrderBlock]:
        return list(self._active_obs)

    def detect(self, bars: List[Bar], atr: Optional[float] = None) -> List[OrderBlock]:
        """Scan bars for order blocks. Returns newly detected OBs.

        Args:
            bars: List of Bar objects in chronological order.
            atr: Current ATR value. If None, computed from bars.
        """
        min_bars = self.move_lookforward + 2
        if len(bars) < min_bars:
            return []

        if atr is None:
            atr = self._compute_atr(bars)

        if atr <= 0:
            return []

        threshold = atr * self.atr_multiplier
        new_obs: List[OrderBlock] = []

        # We need at least move_lookforward bars after the candidate
        for i in range(len(bars) - self.move_lookforward):
            bar = bars[i]

            # --- Bullish OB: bearish candle followed by strong bullish move ---
            if bar.is_bearish:
                max_high = max(b.high for b in bars[i + 1: i + 1 + self.move_lookforward])
                move_up = max_high - bar.low
                if move_up > threshold:
                    ob = OrderBlock(
                        high=bar.high,
                        low=bar.low,
                        direction="bullish",
                        timeframe=self.timeframe,
                        formation_time=bar.timestamp,
                    )
                    if not self._is_duplicate(ob):
                        new_obs.append(ob)
                        self._active_obs.append(ob)
                        self._log.debug("bullish_ob_detected",
                                        high=ob.high, low=ob.low, move=move_up)

            # --- Bearish OB: bullish candle followed by strong bearish move ---
            if bar.is_bullish:
                min_low = min(b.low for b in bars[i + 1: i + 1 + self.move_lookforward])
                move_down = bar.high - min_low
                if move_down > threshold:
                    ob = OrderBlock(
                        high=bar.high,
                        low=bar.low,
                        direction="bearish",
                        timeframe=self.timeframe,
                        formation_time=bar.timestamp,
                    )
                    if not self._is_duplicate(ob):
                        new_obs.append(ob)
                        self._active_obs.append(ob)
                        self._log.debug("bearish_ob_detected",
                                        high=ob.high, low=ob.low, move=move_down)

        # Update sweep status
        if bars:
            self._update_sweeps(bars[-1])

        return new_obs

    def update(self, bar: Bar) -> None:
        """Incrementally check if a new bar sweeps any active OBs."""
        self._update_sweeps(bar)

    def _update_sweeps(self, bar: Bar) -> None:
        """Mark OBs as swept when price trades through them."""
        still_active: List[OrderBlock] = []
        for ob in self._active_obs:
            if ob.swept:
                self._swept_obs.append(ob)
                continue

            swept = False
            if ob.direction == "bullish":
                # Bullish OB is swept if price drops below its low
                if bar.low < ob.low:
                    swept = True
            elif ob.direction == "bearish":
                # Bearish OB is swept if price rises above its high
                if bar.high > ob.high:
                    swept = True

            if swept:
                ob.swept = True
                ob.sweep_time = bar.timestamp
                self._swept_obs.append(ob)
                self._log.debug("ob_swept", direction=ob.direction,
                                high=ob.high, low=ob.low)
            else:
                still_active.append(ob)

        self._active_obs = still_active

    def _is_duplicate(self, ob: OrderBlock) -> bool:
        for existing in self._active_obs + self._swept_obs:
            if (existing.high == ob.high and existing.low == ob.low
                    and existing.direction == ob.direction
                    and existing.formation_time == ob.formation_time):
                return True
        return False

    def get_nearest_ob(self, price: float, direction: str = "") -> Optional[OrderBlock]:
        """Return the nearest un-swept OB to the given price."""
        candidates = self._active_obs
        if direction:
            candidates = [o for o in candidates if o.direction == direction]
        if not candidates:
            return None
        return min(candidates, key=lambda o: abs(o.midpoint - price))

    @staticmethod
    def _compute_atr(bars: List[Bar], period: int = 14) -> float:
        """Simple ATR calculation from bars."""
        if len(bars) < 2:
            return 0.0
        trs: List[float] = []
        for i in range(1, len(bars)):
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i - 1].close),
                abs(bars[i].low - bars[i - 1].close),
            )
            trs.append(tr)
        n = min(period, len(trs))
        if n == 0:
            return 0.0
        return sum(trs[-n:]) / n

    def reset(self) -> None:
        """Clear all tracked OBs."""
        self._active_obs.clear()
        self._swept_obs.clear()
