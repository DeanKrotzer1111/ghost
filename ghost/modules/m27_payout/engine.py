"""Payout Optimization Engine — manages funded account payout phases with dynamic risk/TQS scaling."""
import structlog
from ghost.core.models import PayoutPhase, PayoutState

logger = structlog.get_logger()

PHASE_CONFIG = {
    PayoutPhase.APPROACH: {"risk_mult": 1.00, "tqs": 85, "max_trades": 4},
    PayoutPhase.ACCELERATION: {"risk_mult": 1.10, "tqs": 87, "max_trades": 3},
    PayoutPhase.FINAL_APPROACH: {"risk_mult": 0.75, "tqs": 90, "max_trades": 2},
    PayoutPhase.PAYOUT_TRIGGERED: {"risk_mult": 0.50, "tqs": 92, "max_trades": 1},
    PayoutPhase.RECOVERY: {"risk_mult": 0.60, "tqs": 88, "max_trades": 2},
}


class PayoutOptimizationEngine:
    """Manages funded account lifecycle through 5 payout phases.

    As P&L approaches the payout target, risk is progressively reduced
    and TQS requirements are raised to protect realized gains.
    """

    def get_state(
        self, current_pnl: float, payout_target: float,
        avg_expectancy: float, avg_trades_per_day: float,
    ) -> PayoutState:
        ph = self._phase(current_pnl, payout_target)
        cfg = PHASE_CONFIG[ph]
        rem = max(0.0, payout_target - current_pnl)
        de = avg_expectancy * avg_trades_per_day

        return PayoutState(
            current_pnl, payout_target, ph,
            int(rem / de) if de > 0 else 999,
            cfg["risk_mult"], cfg["tqs"], cfg["max_trades"],
            current_pnl / payout_target if payout_target > 0 else 0.0,
        )

    def _phase(self, pnl: float, target: float) -> PayoutPhase:
        if target <= 0:
            return PayoutPhase.APPROACH
        r = pnl / target
        if r < 0:
            return PayoutPhase.RECOVERY
        if r >= 1:
            return PayoutPhase.PAYOUT_TRIGGERED
        if r >= 0.833:
            return PayoutPhase.FINAL_APPROACH
        if r >= 0.500:
            return PayoutPhase.ACCELERATION
        return PayoutPhase.APPROACH

    def check_graduation(
        self, consecutive_payouts: int, max_dd_pct: float,
        wr100: float, exp100: float, violations: int,
    ) -> dict:
        c = {
            "three_consecutive_payouts": consecutive_payouts >= 3,
            "dd_never_exceeded_40pct": max_dd_pct <= 0.40,
            "win_rate_above_80pct": wr100 >= 0.80,
            "expectancy_above_150": exp100 >= 150.0,
            "zero_violations": violations == 0,
        }
        return {
            "eligible": all(c.values()),
            "conditions": c,
            "conditions_met": sum(c.values()),
        }
