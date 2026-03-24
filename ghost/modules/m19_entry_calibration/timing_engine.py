"""Entry Timing Engine — confirms entry via rejection wicks, engulfing patterns, OFD sign changes, and absorption."""
import structlog
from dataclasses import dataclass
from typing import List

logger = structlog.get_logger()


@dataclass
class ConfirmationResult:
    confirmed: bool
    signals: List[str]
    urgency: str = "NORMAL"


class EntryTimingEngine:
    """Checks for micro-structure confirmation signals before entry execution."""

    def check(self, bars: list, ofd_snapshots: list, zone_tests: int = 0) -> ConfirmationResult:
        signals = []

        if len(bars) >= 2:
            c, p = bars[-1], bars[-2]
            rng = c.high - c.low
            if rng > 0 and (min(c.open, c.close) - c.low) / rng >= 0.60:
                signals.append("REJECTION_WICK")
            if c.close > c.open and p.close < p.open and c.close > p.open and c.open < p.close:
                signals.append("BULLISH_ENGULFING")

        if len(ofd_snapshots) >= 3:
            pd = [s.cumulative_delta for s in ofd_snapshots[-3:-1]]
            if all(d < 0 for d in pd) and ofd_snapshots[-1].cumulative_delta > 0:
                signals.append("OFD_SIGN_CHANGE")

        if zone_tests >= 3:
            signals.append("BID_ABSORPTION")

        if not signals:
            return ConfirmationResult(False, [])

        return ConfirmationResult(True, signals, "HIGH" if len(signals) >= 2 else "NORMAL")
