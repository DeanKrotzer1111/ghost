"""Liquidity Void Detection Engine — identifies unfilled price gaps and displacement candles."""
import structlog
from ghost.core.models import LiquidityVoid
from typing import Optional, List

logger = structlog.get_logger()


class LiquidityVoidEngine:
    """Detects liquidity voids (price gaps and displacement candles) that act as magnets or obstacles.

    Voids between entry and TP1 reduce TP confidence; voids beyond TP1 serve as extended targets.
    """

    def __init__(self, tick_sizes: dict):
        self.tick_sizes = tick_sizes

    def detect(self, instrument: str, bars: list, atr: float) -> List[LiquidityVoid]:
        if not bars or atr <= 0:
            return []

        tick = self.tick_sizes.get(instrument, 0.25)
        voids = []

        for i in range(1, len(bars)):
            p, c = bars[i - 1], bars[i]

            # Gap up
            if c.low > p.high and c.low - p.high >= atr * 0.5:
                voids.append(LiquidityVoid(
                    c.low, p.high, int((c.low - p.high) / tick),
                    "BULLISH", False, getattr(c, "timestamp", None),
                ))

            # Gap down
            if c.high < p.low and p.low - c.high >= atr * 0.5:
                voids.append(LiquidityVoid(
                    p.low, c.high, int((p.low - c.high) / tick),
                    "BEARISH", False, getattr(c, "timestamp", None),
                ))

            # Displacement candle (body >= 2.5x ATR, minimal wicks)
            body = abs(c.close - c.open)
            rng = c.high - c.low
            if body >= atr * 2.5 and rng > 0 and (rng - body) / body <= 0.20:
                d = "BULLISH" if c.close > c.open else "BEARISH"
                voids.append(LiquidityVoid(
                    max(c.open, c.close), min(c.open, c.close),
                    int(body / tick), d, False, getattr(c, "timestamp", None),
                ))

        return voids

    def void_between_entry_and_tp(
        self, entry: float, tp: float, voids: List[LiquidityVoid], direction: str,
    ) -> Optional[LiquidityVoid]:
        b = (
            [v for v in voids if entry < v.low and v.high < tp and not v.filled]
            if direction == "BULLISH"
            else [v for v in voids if tp < v.low and v.high < entry and not v.filled]
        )
        return max(b, key=lambda v: v.size_ticks) if b else None
