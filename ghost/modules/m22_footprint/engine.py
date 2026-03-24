"""Smart Money Footprint Engine — detects institutional activity via absorption, icebergs, stop raids, and divergence."""
import structlog
from ghost.core.models import FootprintScore

logger = structlog.get_logger()


class SmartMoneyFootprintEngine:
    """Composites 5 institutional footprint signals into a 0-1 score for TQS integration."""

    def __init__(self, tick_sizes: dict):
        self.tick_sizes = tick_sizes

    def analyze(
        self, instrument: str, direction: str, tick_buffer: list, bars: list,
        ofd_history: list, liquidity_pools: list, current_price: float,
        zone_low: float, zone_high: float,
        bid_size: float = 0.0, ask_size: float = 0.0,
    ) -> FootprintScore:
        ab = self._absorption(tick_buffer, zone_low, zone_high)
        ic = bid_size / ask_size >= 3.0 if ask_size > 0 else False
        sr = self._stop_raid(liquidity_pools, ofd_history, current_price, instrument, direction)
        dv = self._divergence(bars, ofd_history, direction)
        im = bid_size / ask_size if ask_size > 0 else 0.0

        w = {
            "stop_raid": 0.30 if sr else 0.0,
            "absorption": 0.25 if ab else 0.0,
            "iceberg": 0.20 if ic else 0.0,
            "divergence": 0.15 if dv else 0.0,
            "bid_imbalance": min(im / 3.0, 1.0) * 0.10,
        }

        comp = sum(w.values())
        dom = max(w, key=w.get)

        return FootprintScore(comp, sum(1 for v in w.values() if v > 0), dom, ab, ic, sr, dv, im)

    def _absorption(self, tb: list, zl: float, zh: float) -> bool:
        z = [t for t in tb if zl <= t.price <= zh]
        if len(z) < 15:
            return False
        sv = [getattr(t, "sell_volume", 0) for t in z[-15:]]
        t3 = len(sv) // 3
        return t3 > 0 and sum(sv[-t3:]) < sum(sv[:t3]) * 0.65

    def _stop_raid(self, pools: list, odh: list, price: float, instrument: str, direction: str) -> bool:
        tick = self.tick_sizes.get(instrument, 0.25)
        td = "SSL" if direction == "BULLISH" else "BSL"
        n = [p for p in pools if getattr(p, "direction", "") == td and 0 < abs(price - p.level) < 30 * tick]
        if not n or not odh:
            return False
        delta = odh[-1].cumulative_delta
        return (direction == "BULLISH" and delta < -300) or (direction == "BEARISH" and delta > 300)

    def _divergence(self, bars: list, odh: list, direction: str) -> bool:
        if len(bars) < 3 or len(odh) < 3:
            return False
        dv = [s.cumulative_delta for s in odh[-3:]]
        if direction == "BULLISH":
            return bars[-1].low < bars[-2].low < bars[-3].low and not (dv[-1] < dv[-2] < dv[-3])
        return bars[-1].high > bars[-2].high > bars[-3].high and not (dv[-1] > dv[-2] > dv[-3])
