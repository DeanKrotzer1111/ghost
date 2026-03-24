"""TP Overshoot Analyzer — learns how far price overshoots TP1 by instrument and session."""
import numpy as np
import structlog

logger = structlog.get_logger()

DEFAULT_OVERSHOOT = {
    "MNQ": 6, "NQ": 8, "ES": 5, "MES": 4, "GC": 5, "MGC": 4,
    "CL": 7, "SI": 4, "NG": 6, "6E": 3, "ZB": 4,
}


class TPOvershootAnalyzer:
    """Computes median TP1 overshoot from historical trades, bucketed by 2-hour session windows."""

    MINIMUM_SAMPLE = 15

    def __init__(self, db):
        self.db = db

    async def get_overshoot(self, instrument: str, hour_et: int) -> float:
        bucket = (hour_et // 2) * 2
        rows = await self.db.fetch(
            """
            SELECT tp1_level, highest_price_reached, lowest_price_reached, tp1_hit, direction
            FROM trades WHERE instrument=$1 AND tp1_level IS NOT NULL
            AND EXTRACT(HOUR FROM entry_time AT TIME ZONE 'America/New_York') BETWEEN $2 AND $3
            """,
            instrument, bucket, bucket + 2,
        )

        if len(rows) < self.MINIMUM_SAMPLE:
            return float(DEFAULT_OVERSHOOT.get(instrument, 6))

        hr = sum(1 for r in rows if r["tp1_hit"]) / len(rows)
        if hr < 0.70:
            logger.warning("tp1_hit_rate_low", instrument=instrument, rate=round(hr, 2))

        os = [
            float(r["highest_price_reached"]) - float(r["tp1_level"]) if r["direction"] == "BULLISH"
            else float(r["tp1_level"]) - float(r["lowest_price_reached"])
            for r in rows if r["tp1_hit"]
        ]
        return max(0.0, float(np.median(os))) if os else float(DEFAULT_OVERSHOOT.get(instrument, 6))
