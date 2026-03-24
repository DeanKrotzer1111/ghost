"""Market structure analysis for ICT signal engine."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import structlog

from ghost.modules.m01_data.models import Bar

logger = structlog.get_logger(__name__)


@dataclass
class SwingPoint:
    """A swing high or swing low."""
    price: float
    timestamp: float
    type: str  # "high" or "low"
    bar_index: int = 0


@dataclass
class StructureBreak:
    """Represents a Break of Structure (BOS) or Change of Character (CHoCH)."""
    price: float
    timestamp: float
    direction: str  # "bullish" or "bearish"
    break_type: str  # "BOS" or "CHoCH"
    swing_point: Optional[SwingPoint] = None


@dataclass
class StructureState:
    """Current market structure state."""
    direction: str = "UNKNOWN"  # "bullish", "bearish", "UNKNOWN"
    last_bos: Optional[StructureBreak] = None
    last_choch: Optional[StructureBreak] = None
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    premium_discount: float = 0.5  # 0 = deep discount, 1 = deep premium
    range_high: float = 0.0
    range_low: float = 0.0


class StructureDetector:
    """Detects market structure: swing points, BOS, CHoCH, premium/discount.

    Uses a 5-bar lookback for swing detection (the swing point must be the
    highest high / lowest low among 5 bars on each side).
    """

    def __init__(self, swing_lookback: int = 5):
        self.swing_lookback = swing_lookback
        self._state = StructureState()
        self._prev_direction: str = "UNKNOWN"
        self._all_breaks: List[StructureBreak] = []
        self._log = logger.bind(component="StructureDetector")

    @property
    def state(self) -> StructureState:
        return self._state

    def analyze(self, bars: List[Bar]) -> StructureState:
        """Full analysis of market structure from a list of bars.

        Detects swing points, BOS, CHoCH, and computes premium/discount zone.
        Returns the current StructureState.
        """
        if len(bars) < self.swing_lookback * 2 + 1:
            return self._state

        # Detect swing points
        swing_highs, swing_lows = self._detect_swings(bars)
        self._state.swing_highs = swing_highs
        self._state.swing_lows = swing_lows

        # Determine range for premium/discount
        if swing_highs and swing_lows:
            recent_highs = swing_highs[-5:]
            recent_lows = swing_lows[-5:]
            self._state.range_high = max(s.price for s in recent_highs)
            self._state.range_low = min(s.price for s in recent_lows)

        # Detect BOS and CHoCH
        self._detect_structure_breaks(bars, swing_highs, swing_lows)

        # Compute premium/discount
        if self._state.range_high > self._state.range_low:
            current_price = bars[-1].close
            total_range = self._state.range_high - self._state.range_low
            self._state.premium_discount = (
                (current_price - self._state.range_low) / total_range
            )
            self._state.premium_discount = max(0.0, min(1.0, self._state.premium_discount))

        return self._state

    def _detect_swings(self, bars: List[Bar]) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Detect swing highs and swing lows with the configured lookback."""
        swing_highs: List[SwingPoint] = []
        swing_lows: List[SwingPoint] = []
        lb = self.swing_lookback

        for i in range(lb, len(bars) - lb):
            # Swing high: bar[i].high is highest in the window
            window_highs = [bars[j].high for j in range(i - lb, i + lb + 1)]
            if bars[i].high == max(window_highs):
                # Ensure it's strictly the max (not a plateau at the edge)
                is_swing = True
                for j in range(i - lb, i + lb + 1):
                    if j != i and bars[j].high >= bars[i].high:
                        # Allow if the equal bar is closer to center
                        if abs(j - i) >= lb:
                            is_swing = False
                            break
                if is_swing:
                    swing_highs.append(SwingPoint(
                        price=bars[i].high,
                        timestamp=bars[i].timestamp,
                        type="high",
                        bar_index=i,
                    ))

            # Swing low: bar[i].low is lowest in the window
            window_lows = [bars[j].low for j in range(i - lb, i + lb + 1)]
            if bars[i].low == min(window_lows):
                is_swing = True
                for j in range(i - lb, i + lb + 1):
                    if j != i and bars[j].low <= bars[i].low:
                        if abs(j - i) >= lb:
                            is_swing = False
                            break
                if is_swing:
                    swing_lows.append(SwingPoint(
                        price=bars[i].low,
                        timestamp=bars[i].timestamp,
                        type="low",
                        bar_index=i,
                    ))

        return swing_highs, swing_lows

    def _detect_structure_breaks(self, bars: List[Bar],
                                 swing_highs: List[SwingPoint],
                                 swing_lows: List[SwingPoint]) -> None:
        """Detect BOS and CHoCH from swing points and current price action."""
        if not swing_highs or not swing_lows:
            return

        current_price = bars[-1].close
        current_high = bars[-1].high
        current_low = bars[-1].low
        current_ts = bars[-1].timestamp

        # Track the most recent unbroken swing high and swing low
        last_sh = swing_highs[-1]
        last_sl = swing_lows[-1]

        new_direction = self._state.direction

        # Bullish BOS: price breaks above the most recent swing high
        if current_high > last_sh.price:
            break_dir = "bullish"
            if self._state.direction == "bearish":
                # First bullish break against bearish trend = CHoCH
                brk = StructureBreak(
                    price=last_sh.price,
                    timestamp=current_ts,
                    direction=break_dir,
                    break_type="CHoCH",
                    swing_point=last_sh,
                )
                self._state.last_choch = brk
                self._all_breaks.append(brk)
                self._log.info("choch_detected", direction=break_dir,
                               price=last_sh.price)
            else:
                brk = StructureBreak(
                    price=last_sh.price,
                    timestamp=current_ts,
                    direction=break_dir,
                    break_type="BOS",
                    swing_point=last_sh,
                )
                self._state.last_bos = brk
                self._all_breaks.append(brk)
                self._log.debug("bos_detected", direction=break_dir,
                                price=last_sh.price)
            new_direction = "bullish"

        # Bearish BOS: price breaks below the most recent swing low
        if current_low < last_sl.price:
            break_dir = "bearish"
            if self._state.direction == "bullish":
                brk = StructureBreak(
                    price=last_sl.price,
                    timestamp=current_ts,
                    direction=break_dir,
                    break_type="CHoCH",
                    swing_point=last_sl,
                )
                self._state.last_choch = brk
                self._all_breaks.append(brk)
                self._log.info("choch_detected", direction=break_dir,
                               price=last_sl.price)
            else:
                brk = StructureBreak(
                    price=last_sl.price,
                    timestamp=current_ts,
                    direction=break_dir,
                    break_type="BOS",
                    swing_point=last_sl,
                )
                self._state.last_bos = brk
                self._all_breaks.append(brk)
                self._log.debug("bos_detected", direction=break_dir,
                                price=last_sl.price)
            new_direction = "bearish"

        self._state.direction = new_direction

    def is_premium(self) -> bool:
        """True if price is in the premium zone (above 50% of range)."""
        return self._state.premium_discount > 0.5

    def is_discount(self) -> bool:
        """True if price is in the discount zone (below 50% of range)."""
        return self._state.premium_discount < 0.5

    def reset(self) -> None:
        """Clear all state."""
        self._state = StructureState()
        self._prev_direction = "UNKNOWN"
        self._all_breaks.clear()
