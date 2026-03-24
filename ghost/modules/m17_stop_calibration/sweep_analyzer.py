"""Stop Sweep Distribution Analyzer — learns per-instrument sweep behavior from historical stops."""
import numpy as np
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()

DEFAULT_SWEEP_TICKS = {
    "MNQ": 4, "NQ": 5, "ES": 3, "MES": 3, "RTY": 4, "M2K": 4, "YM": 4, "MYM": 3,
    "GC": 3, "MGC": 3, "SI": 3, "SIL": 3, "HG": 3, "PL": 3,
    "CL": 4, "MCL": 4, "NG": 5, "RB": 4, "HO": 4,
    "6E": 3, "6B": 3, "6J": 4, "6A": 3, "6C": 3, "M6E": 3,
    "ZB": 3, "ZN": 3, "ZF": 3, "ZT": 2, "ZC": 4, "ZS": 4, "ZW": 5,
}


@dataclass
class SweepDistribution:
    instrument: str
    setup_type: str
    p50_sweep_ticks: float
    p75_sweep_ticks: float
    p90_sweep_ticks: float
    recovery_rate_at_p75: float
    sample_size: int
    sufficient_data: bool


class StopSweepAnalyzer:
    """Analyzes historical stop-hit trades to compute sweep distributions per instrument."""

    MINIMUM_SAMPLE_SIZE = 20

    def __init__(self, db):
        self.db = db

    async def compute(self, instrument: str, setup_type: str = "FVG_OB") -> SweepDistribution:
        trades = await self.db.fetch(
            """
            SELECT stop_level, lowest_price_after_stop, did_price_recover, tick_size
            FROM trades WHERE instrument=$1 AND outcome='STOP_HIT' AND setup_type=$2
            AND lowest_price_after_stop IS NOT NULL
            """,
            instrument, setup_type,
        )

        if len(trades) < self.MINIMUM_SAMPLE_SIZE:
            d = DEFAULT_SWEEP_TICKS.get(instrument, 4)
            return SweepDistribution(
                instrument, setup_type, d * 0.7, float(d), d * 1.5,
                0.70, len(trades), False,
            )

        ts = float(trades[0]["tick_size"])
        sw = [
            (float(t["stop_level"]) - float(t["lowest_price_after_stop"])) / ts
            for t in trades
        ]
        p75 = float(np.percentile(sw, 75))
        rec = [int(t["did_price_recover"]) for i, t in enumerate(trades) if sw[i] <= p75]

        return SweepDistribution(
            instrument, setup_type,
            float(np.percentile(sw, 50)), p75, float(np.percentile(sw, 90)),
            float(np.mean(rec)) if rec else 0.70,
            len(trades), True,
        )
