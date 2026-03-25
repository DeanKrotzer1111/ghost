"""M12 News Sentiment — Analyzes news events and their impact on trading risk."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Eastern Time offset (fixed EST = UTC-5)
_ET_OFFSET = timedelta(hours=-5)
_ET_TZ = timezone(_ET_OFFSET)


@dataclass
class NewsEvent:
    """A single news event with impact assessment."""
    headline: str = ""
    source: str = ""
    timestamp: float = 0.0
    impact: str = "LOW"          # HIGH / MEDIUM / LOW
    sentiment_score: float = 0.0  # -1.0 (bearish) to 1.0 (bullish)


class NewsAnalyzer:
    """Analyzes upcoming news events to determine risk level for trading.
    Tracks a queue of news events sorted by time and provides risk assessment
    based on proximity to high-impact events."""

    def __init__(self):
        self.events: list[NewsEvent] = []

    def add_event(self, event: NewsEvent) -> None:
        """Add a news event to the tracked list."""
        self.events.append(event)
        self.events.sort(key=lambda e: e.timestamp)

    def add_events(self, events: list[NewsEvent]) -> None:
        """Add multiple news events."""
        self.events.extend(events)
        self.events.sort(key=lambda e: e.timestamp)

    def get_risk_level(self, minutes_to_next_event: int) -> str:
        """Determine trading risk level based on proximity to next news event.

        Args:
            minutes_to_next_event: Minutes until the next scheduled news event.

        Returns:
            Risk level string: HIGH / MEDIUM / LOW / NEUTRAL
        """
        if minutes_to_next_event <= 5:
            return "HIGH"
        elif minutes_to_next_event <= 15:
            return "MEDIUM"
        elif minutes_to_next_event <= 30:
            return "LOW"
        else:
            return "NEUTRAL"

    def get_sentiment(self, current_time: float, lookback_seconds: float = 3600.0) -> float:
        """Get average sentiment score from recent events.

        Args:
            current_time: Current timestamp.
            lookback_seconds: How far back to look for events (default 1 hour).

        Returns:
            Average sentiment score from -1.0 to 1.0, or 0.0 if no events.
        """
        cutoff = current_time - lookback_seconds
        recent = [e for e in self.events if cutoff <= e.timestamp <= current_time]
        if not recent:
            return 0.0

        # Weight by impact level
        weights = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}
        total_weight = 0.0
        weighted_sum = 0.0

        for event in recent:
            w = weights.get(event.impact, 1.0)
            weighted_sum += event.sentiment_score * w
            total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_next_event(self, current_time: float) -> Optional[NewsEvent]:
        """Get the next upcoming event after current_time."""
        for event in self.events:
            if event.timestamp > current_time:
                return event
        return None

    def minutes_until_next(self, current_time: float) -> int:
        """Calculate minutes until the next event."""
        nxt = self.get_next_event(current_time)
        if nxt is None:
            return 999
        return max(0, int((nxt.timestamp - current_time) / 60.0))

    def clear(self) -> None:
        """Clear all tracked events."""
        self.events.clear()


class MockNewsAnalyzer(NewsAnalyzer):
    """Mock news analyzer for backtesting. Always returns LOW risk and neutral sentiment."""

    def get_risk_level(self, minutes_to_next_event: int) -> str:
        """Always return LOW risk for backtesting."""
        return "LOW"

    def get_sentiment(self, current_time: float, lookback_seconds: float = 3600.0) -> float:
        """Always return neutral sentiment for backtesting."""
        return 0.0

    def get_next_event(self, current_time: float) -> Optional[NewsEvent]:
        """No events in mock mode."""
        return None

    def minutes_until_next(self, current_time: float) -> int:
        """No upcoming events in mock mode."""
        return 999


# ──────────────────────────────────────────────────────────────────────────────
# Major economic events: (day_of_week, hour_et, minute_et, name)
#   day_of_week: 0=Monday ... 4=Friday
#   Times are Eastern Time
# ──────────────────────────────────────────────────────────────────────────────

_MAJOR_EVENTS: List[Tuple[int, int, int, str]] = [
    # FOMC — typically Wednesday 14:00 ET (8 times/year, but we flag every Wed)
    (2, 14, 0, "FOMC"),
    # NFP — first Friday of the month, 8:30 ET
    (4, 8, 30, "NFP"),
    # CPI — typically Tuesday/Wednesday, 8:30 ET (monthly)
    (1, 8, 30, "CPI"),
    (2, 8, 30, "CPI"),
    # PPI — typically Tuesday/Thursday, 8:30 ET
    (1, 8, 30, "PPI"),
    (3, 8, 30, "PPI"),
    # GDP — typically Thursday, 8:30 ET (quarterly)
    (3, 8, 30, "GDP"),
    # Jobless Claims — every Thursday 8:30 ET
    (3, 8, 30, "JOBLESS_CLAIMS"),
    # ISM Manufacturing — first business day of month (Mon), 10:00 ET
    (0, 10, 0, "ISM_MFG"),
    # Retail Sales — typically Tuesday, 8:30 ET
    (1, 8, 30, "RETAIL_SALES"),
]

# High-impact event names that warrant the widest risk windows
_HIGH_IMPACT_NAMES = {"FOMC", "NFP", "CPI", "PPI", "GDP"}


class LiveNewsAnalyzer:
    """News risk analyzer for live trading.

    Uses a hardcoded economic calendar of major recurring events keyed by
    day-of-week and time-of-day.  Returns risk levels based on proximity
    to those events and a simple time-of-day sentiment bias.
    """

    def __init__(self, extra_events: Optional[List[Tuple[int, int, int, str]]] = None):
        """
        Args:
            extra_events: Additional (dow, hour_et, minute_et, name) tuples to add.
        """
        self.events = list(_MAJOR_EVENTS)
        if extra_events:
            self.events.extend(extra_events)

    # ── Risk level ────────────────────────────────────────────────────────

    def get_risk_level(self, timestamp: float) -> str:
        """Return HIGH / MEDIUM / LOW risk based on proximity to major events.

        HIGH  — within 30 min of FOMC, NFP, CPI, PPI, GDP
        MEDIUM — within 60 min of the above
        LOW   — otherwise
        """
        dt_et = datetime.fromtimestamp(timestamp, tz=_ET_TZ)
        dow = dt_et.weekday()  # 0=Mon
        current_minutes = dt_et.hour * 60 + dt_et.minute

        closest_gap = 9999  # minutes to closest high-impact event today

        for ev_dow, ev_h, ev_m, ev_name in self.events:
            if ev_dow != dow:
                continue
            if ev_name not in _HIGH_IMPACT_NAMES:
                continue
            ev_minutes = ev_h * 60 + ev_m
            gap = abs(current_minutes - ev_minutes)
            closest_gap = min(closest_gap, gap)

        if closest_gap <= 30:
            return "HIGH"
        elif closest_gap <= 60:
            return "MEDIUM"
        return "LOW"

    # ── Sentiment by time-of-day ──────────────────────────────────────────

    def get_sentiment_for_instrument(self, instrument: str, timestamp: float) -> float:
        """Return a basic sentiment score based on time-of-day patterns.

        Heuristic (all instruments):
          - NY morning open (9:30-11:30 ET):  slight bullish bias  +0.15
          - Midday (11:30-13:30 ET):          neutral              0.00
          - NY afternoon (13:30-16:00 ET):    slight bearish bias -0.10
          - Overnight / pre-market:           neutral              0.00

        For equity-index micros (MNQ, MES, M2K, MYM) the bias is amplified
        slightly because institutional activity is concentrated around the
        NY open.

        Returns:
            Sentiment score from -1.0 to 1.0.
        """
        dt_et = datetime.fromtimestamp(timestamp, tz=_ET_TZ)
        total_min = dt_et.hour * 60 + dt_et.minute

        # NY morning open: 9:30 - 11:30 ET  (570 - 690 min)
        if 570 <= total_min < 690:
            base = 0.15
        # Lunch: 11:30 - 13:30 ET  (690 - 810 min)
        elif 690 <= total_min < 810:
            base = 0.0
        # NY afternoon: 13:30 - 16:00 ET  (810 - 960 min)
        elif 810 <= total_min < 960:
            base = -0.10
        else:
            base = 0.0

        # Amplify for equity-index micros
        equity_micros = {"MNQ", "MES", "M2K", "MYM", "NQ", "ES", "RTY", "YM"}
        if instrument in equity_micros:
            base *= 1.5

        return round(base, 3)

    # ── Convenience ───────────────────────────────────────────────────────

    def minutes_to_next_high_impact(self, timestamp: float) -> int:
        """Minutes until the next high-impact event today (or 999 if none left)."""
        dt_et = datetime.fromtimestamp(timestamp, tz=_ET_TZ)
        dow = dt_et.weekday()
        current_minutes = dt_et.hour * 60 + dt_et.minute

        best = 9999
        for ev_dow, ev_h, ev_m, ev_name in self.events:
            if ev_dow != dow or ev_name not in _HIGH_IMPACT_NAMES:
                continue
            ev_minutes = ev_h * 60 + ev_m
            gap = ev_minutes - current_minutes
            if gap > 0:
                best = min(best, gap)

        return best if best < 9999 else 999
