"""M12 News Sentiment — Analyzes news events and their impact on trading risk."""
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


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
