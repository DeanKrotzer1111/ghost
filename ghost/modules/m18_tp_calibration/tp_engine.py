"""Take Profit Calibration Engine — multi-target TP with liquidity draws, session modifiers, and void awareness."""
import structlog
from ghost.core.models import TPLevels

logger = structlog.get_logger()

SESSION_MODS = {
    (2, 5): 0.90, (7, 9): 1.00, (9, 10): 0.85,
    (10, 12): 0.75, (12, 14): 0.65, (14, 16): 0.70,
}


class TakeProfitCalibrationEngine:
    """Calculates 3-tier TP levels using liquidity pools, overshoot data, and session timing."""

    def __init__(self, overshoot_analyzer, tick_sizes: dict):
        self.overshoot = overshoot_analyzer
        self.tick_sizes = tick_sizes

    async def calculate(
        self, instrument: str, direction: str, entry: float, stop: float,
        liquidity_pools: list, htf_draws: list, hour_et: int, void_present: bool,
    ) -> TPLevels:
        tick = self.tick_sizes.get(instrument, 0.25)
        risk = abs(entry - stop) or tick * 20
        op = (await self.overshoot.get_overshoot(instrument, hour_et)) * tick

        um = [
            p for p in liquidity_pools
            if not getattr(p, "mitigated", False)
            and (p.level > entry if direction == "BULLISH" else p.level < entry)
        ]

        if not um:
            tp1 = entry + risk * 2.0 if direction == "BULLISH" else entry - risk * 2.0
        else:
            n = (
                min(um, key=lambda p: p.level - entry) if direction == "BULLISH"
                else min(um, key=lambda p: entry - p.level)
            )
            tp1 = n.level + op if direction == "BULLISH" else n.level - op

        beyond = [p for p in um if (p.level > tp1 if direction == "BULLISH" else p.level < tp1)]
        ts = min(beyond, key=lambda p: p.level - tp1).level if beyond else None
        tr = entry + risk * 2.5 if direction == "BULLISH" else entry - risk * 2.5
        tp2 = max(ts, tr) if ts and direction == "BULLISH" else min(ts, tr) if ts else tr
        tp3 = (
            htf_draws[0].level if htf_draws
            else (entry + risk * 4.0 if direction == "BULLISH" else entry - risk * 4.0)
        )

        mod = next((m for (s, e), m in SESSION_MODS.items() if s <= hour_et < e), 0.75)

        return TPLevels(
            tp1, tp2, tp3,
            round(0.72 * mod, 3), round(0.45 * mod, 3), round(0.22 * mod, 3),
            0.50, 0.30, 0.20, void_present, mod,
        )
