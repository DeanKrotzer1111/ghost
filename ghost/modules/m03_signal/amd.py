"""AMD (Accumulation, Manipulation, Distribution) phase detection."""
from dataclasses import dataclass
from typing import List, Optional

import structlog

from ghost.modules.m01_data.models import Bar

logger = structlog.get_logger(__name__)


@dataclass
class AMDState:
    """Current AMD phase classification."""
    phase: str  # "ACCUMULATION", "MANIPULATION", "DISTRIBUTION", "UNKNOWN"
    confidence: float = 0.0
    phase_bars: int = 0  # how many bars into this phase
    range_high: float = 0.0
    range_low: float = 0.0


class AMDDetector:
    """Detects the current AMD phase from recent bar patterns.

    Accumulation: tight range, low volume relative to average, building liquidity.
    Manipulation: false breakout / stop hunt — price sweeps a range extreme then reverses.
    Distribution: strong directional move after manipulation.
    """

    def __init__(self, accumulation_window: int = 10, vol_low_threshold: float = 0.7,
                 range_tight_threshold: float = 0.5, atr_breakout_mult: float = 1.2):
        """
        Args:
            accumulation_window: Bars to look back for range/vol analysis.
            vol_low_threshold: Volume below this fraction of avg = "low volume".
            range_tight_threshold: Range below this fraction of ATR = "tight".
            atr_breakout_mult: Move > this * ATR = "strong directional move".
        """
        self.accumulation_window = accumulation_window
        self.vol_low_threshold = vol_low_threshold
        self.range_tight_threshold = range_tight_threshold
        self.atr_breakout_mult = atr_breakout_mult
        self._state = AMDState(phase="UNKNOWN")
        self._log = logger.bind(component="AMDDetector")

    @property
    def state(self) -> AMDState:
        return self._state

    def analyze(self, bars: List[Bar], atr: Optional[float] = None) -> AMDState:
        """Classify the current AMD phase from recent bars.

        Args:
            bars: Chronologically ordered bars (newest last).
            atr: External ATR value. Computed from bars if not provided.
        """
        if len(bars) < self.accumulation_window:
            self._state = AMDState(phase="UNKNOWN")
            return self._state

        window = bars[-self.accumulation_window:]

        if atr is None:
            atr = self._compute_atr(bars)
        if atr <= 0:
            self._state = AMDState(phase="UNKNOWN")
            return self._state

        # Compute window stats
        range_high = max(b.high for b in window)
        range_low = min(b.low for b in window)
        total_range = range_high - range_low
        avg_volume = sum(b.volume for b in window) / len(window) if window else 1
        avg_bar_range = sum(b.range for b in window) / len(window)

        # Recent bars (last 3) for manipulation/distribution check
        recent = bars[-3:] if len(bars) >= 3 else bars
        recent_high = max(b.high for b in recent)
        recent_low = min(b.low for b in recent)

        # Older window (excluding last 3) for the accumulation range
        older = window[:-3] if len(window) > 3 else window
        older_high = max(b.high for b in older)
        older_low = min(b.low for b in older)

        last_bar = bars[-1]
        prev_bar = bars[-2] if len(bars) >= 2 else bars[-1]

        # ---- Distribution: strong directional move after sweep ----
        # Latest bar or last 2 bars show a strong move
        recent_move = abs(last_bar.close - recent[0].open) if recent else 0
        if recent_move > atr * self.atr_breakout_mult:
            # Confirm it's directional (close near extreme)
            if last_bar.close > last_bar.open:
                body_pct = last_bar.body / max(last_bar.range, 1e-9)
            else:
                body_pct = last_bar.body / max(last_bar.range, 1e-9)

            if body_pct > 0.5:
                confidence = min(1.0, recent_move / (atr * 2))
                self._state = AMDState(
                    phase="DISTRIBUTION",
                    confidence=confidence,
                    phase_bars=len(recent),
                    range_high=range_high,
                    range_low=range_low,
                )
                self._log.debug("amd_phase", phase="DISTRIBUTION",
                                confidence=confidence)
                return self._state

        # ---- Manipulation: sweep of range boundary then reversal ----
        sweep_high = recent_high > older_high
        sweep_low = recent_low < older_low
        reversal = False

        if sweep_high and last_bar.is_bearish:
            # Swept highs then reversed down
            reversal = True
        elif sweep_low and last_bar.is_bullish:
            # Swept lows then reversed up
            reversal = True

        if (sweep_high or sweep_low) and reversal:
            confidence = 0.6
            # Higher confidence if the wick was large (stop hunt signature)
            if sweep_high and last_bar.upper_wick > last_bar.body:
                confidence = 0.8
            if sweep_low and last_bar.lower_wick > last_bar.body:
                confidence = 0.8

            self._state = AMDState(
                phase="MANIPULATION",
                confidence=confidence,
                phase_bars=len(recent),
                range_high=range_high,
                range_low=range_low,
            )
            self._log.debug("amd_phase", phase="MANIPULATION",
                            confidence=confidence,
                            sweep_high=sweep_high, sweep_low=sweep_low)
            return self._state

        # ---- Accumulation: tight range, relatively low volume ----
        range_is_tight = total_range < atr * self.range_tight_threshold * self.accumulation_window
        avg_range_is_tight = avg_bar_range < atr * self.range_tight_threshold

        # Volume check: recent volume below threshold of the lookback average
        recent_vol = sum(b.volume for b in recent) / max(len(recent), 1)
        vol_is_low = recent_vol < avg_volume * self.vol_low_threshold if avg_volume > 0 else False

        if range_is_tight or avg_range_is_tight:
            confidence = 0.5
            if vol_is_low:
                confidence = 0.7
            if range_is_tight and avg_range_is_tight:
                confidence = 0.8

            self._state = AMDState(
                phase="ACCUMULATION",
                confidence=confidence,
                phase_bars=self.accumulation_window,
                range_high=range_high,
                range_low=range_low,
            )
            self._log.debug("amd_phase", phase="ACCUMULATION",
                            confidence=confidence)
            return self._state

        # Default: unknown / transitional
        self._state = AMDState(
            phase="UNKNOWN",
            confidence=0.0,
            range_high=range_high,
            range_low=range_low,
        )
        return self._state

    @staticmethod
    def _compute_atr(bars: List[Bar], period: int = 14) -> float:
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
        self._state = AMDState(phase="UNKNOWN")
