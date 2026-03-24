"""Market Regime Detector — classifies market state using ADX, ATR percentile, and structure."""
import numpy as np
import structlog
from typing import List
from ghost.modules.m01_data.models import Bar
from ghost.modules.m02_regime.models import RegimeState

logger = structlog.get_logger()


class RegimeDetector:
    """Detects market regime across any timeframe using ADX trend strength,
    ATR volatility percentile, and higher-high/higher-low structure analysis.
    """

    def __init__(self, adx_period: int = 14, atr_period: int = 14, lookback: int = 30):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.lookback = lookback

    def detect(self, bars: List[Bar]) -> RegimeState:
        """Classify the current market regime from a list of bars."""
        if len(bars) < self.lookback:
            return RegimeState("UNKNOWN", "NEUTRAL", 0.0, 0.0, 0.0, 0.0, "UNKNOWN")

        recent = bars[-self.lookback:]
        highs = np.array([b.high for b in recent])
        lows = np.array([b.low for b in recent])
        closes = np.array([b.close for b in recent])

        atr = self._atr(highs, lows, closes, self.atr_period)
        adx, plus_di, minus_di = self._adx(highs, lows, closes, self.adx_period)

        # ATR percentile for volatility regime
        atr_series = self._atr_series(highs, lows, closes, self.atr_period)
        atr_pct = self._percentile_rank(atr_series, atr) if len(atr_series) > 0 else 0.5

        vol_regime = (
            "EXTREME" if atr_pct >= 0.90 else
            "HIGH" if atr_pct >= 0.70 else
            "NORMAL" if atr_pct >= 0.30 else
            "LOW"
        )

        # Structure analysis
        structure_dir = self._structure_direction(bars[-50:])

        # Regime classification
        if adx >= 25:
            if plus_di > minus_di:
                label = "TRENDING_BULLISH"
                direction = "BULLISH"
                confidence = min(1.0, adx / 50.0) * (0.8 + 0.2 * (1.0 if structure_dir == "BULLISH" else 0.0))
            else:
                label = "TRENDING_BEARISH"
                direction = "BEARISH"
                confidence = min(1.0, adx / 50.0) * (0.8 + 0.2 * (1.0 if structure_dir == "BEARISH" else 0.0))
        elif adx < 20:
            if atr_pct >= 0.80:
                label = "VOLATILE"
                direction = "NEUTRAL"
                confidence = atr_pct
            else:
                label = "RANGING"
                direction = "NEUTRAL"
                confidence = 1.0 - (adx / 20.0)
        else:
            # Transition zone (ADX 20-25)
            if structure_dir in ("BULLISH", "BEARISH"):
                label = f"TRENDING_{structure_dir}"
                direction = structure_dir
                confidence = 0.55 + (adx - 20) / 10.0
            else:
                label = "RANGING"
                direction = "NEUTRAL"
                confidence = 0.50

        return RegimeState(label, direction, round(confidence, 3), adx, atr, atr_pct, vol_regime)

    def _structure_direction(self, bars: List[Bar]) -> str:
        """Determine directional bias from swing high/low structure."""
        if len(bars) < 20:
            return "NEUTRAL"

        swing_highs = []
        swing_lows = []
        for i in range(5, len(bars) - 5):
            if bars[i].high == max(b.high for b in bars[i - 5:i + 6]):
                swing_highs.append(bars[i].high)
            if bars[i].low == min(b.low for b in bars[i - 5:i + 6]):
                swing_lows.append(bars[i].low)

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1] > swing_highs[-2]
            hl = swing_lows[-1] > swing_lows[-2]
            lh = swing_highs[-1] < swing_highs[-2]
            ll = swing_lows[-1] < swing_lows[-2]

            if hh and hl:
                return "BULLISH"
            if lh and ll:
                return "BEARISH"

        return "NEUTRAL"

    def _atr(self, highs, lows, closes, period):
        """Calculate current ATR value."""
        tr = self._true_range(highs, lows, closes)
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0
        return float(np.mean(tr[-period:]))

    def _atr_series(self, highs, lows, closes, period):
        """Calculate rolling ATR series."""
        tr = self._true_range(highs, lows, closes)
        if len(tr) < period:
            return np.array([])
        result = np.convolve(tr, np.ones(period) / period, mode="valid")
        return result

    def _true_range(self, highs, lows, closes):
        """Calculate true range array."""
        if len(highs) < 2:
            return highs - lows
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        return np.maximum(tr1, np.maximum(tr2, tr3))

    def _adx(self, highs, lows, closes, period):
        """Calculate ADX, +DI, -DI."""
        if len(highs) < period * 2:
            return 0.0, 0.0, 0.0

        up = highs[1:] - highs[:-1]
        down = lows[:-1] - lows[1:]
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        tr = self._true_range(highs, lows, closes)
        n = min(len(tr), len(plus_dm))
        tr = tr[:n]
        plus_dm = plus_dm[:n]
        minus_dm = minus_dm[:n]

        if len(tr) < period:
            return 0.0, 0.0, 0.0

        smoothed_tr = self._wilder_smooth(tr, period)
        smoothed_plus = self._wilder_smooth(plus_dm, period)
        smoothed_minus = self._wilder_smooth(minus_dm, period)

        if smoothed_tr == 0:
            return 0.0, 0.0, 0.0

        plus_di = (smoothed_plus / smoothed_tr) * 100
        minus_di = (smoothed_minus / smoothed_tr) * 100

        dx_sum = plus_di + minus_di
        dx = abs(plus_di - minus_di) / dx_sum * 100 if dx_sum > 0 else 0.0

        return float(dx), float(plus_di), float(minus_di)

    def _wilder_smooth(self, data, period):
        """Wilder's smoothing method."""
        if len(data) < period:
            return float(np.mean(data))
        result = float(np.sum(data[:period]))
        for i in range(period, len(data)):
            result = result - (result / period) + data[i]
        return result / period

    def _percentile_rank(self, series, value):
        """Calculate percentile rank of value within series."""
        if len(series) == 0:
            return 0.5
        return float(np.sum(series <= value)) / len(series)
