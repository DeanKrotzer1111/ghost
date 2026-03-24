"""Weekly Profile Classifier — identifies bullish/bearish/balanced weekly structure from Monday's displacement."""
import structlog
from ghost.core.models import WeeklyProfile

logger = structlog.get_logger()


class WeeklyProfileClassifier:
    """Classifies weekly bias using Monday displacement, weekly open relation, and 4H structure.

    Run every Sunday at 8pm ET to set the weekly directional filter.
    Trades against the weekly profile receive a TQS penalty.
    """

    def classify(
        self, weekly_open: float, monday_high: float, monday_low: float,
        monday_high_time_et: int, monday_low_time_et: int,
        structure_4h: str, current_price: float,
    ) -> WeeklyProfile:
        bull = sum([
            monday_low_time_et <= 10,
            current_price > weekly_open,
            structure_4h == "TRENDING_BULLISH",
        ]) / 3.0

        bear = sum([
            monday_high_time_et <= 10,
            current_price < weekly_open,
            structure_4h == "TRENDING_BEARISH",
        ]) / 3.0

        if bull >= 0.67:
            return WeeklyProfile("BULLISH_WEEK", "BULLISH", "weekly_bsl", bull, "LOW_EARLY")
        if bear >= 0.67:
            return WeeklyProfile("BEARISH_WEEK", "BEARISH", "weekly_ssl", bear, "HIGH_EARLY")
        return WeeklyProfile("BALANCED_WEEK", "NEUTRAL", "none", 0.5)
