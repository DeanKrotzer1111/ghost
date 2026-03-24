"""M05 — Risk Management Engine for Ghost v5.5."""
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AccountState:
    """Snapshot of the trading account and recent performance."""
    balance: float
    equity: float
    daily_pnl: float
    daily_loss_used_pct: float          # e.g. 0.015 = 1.5 % of balance lost today
    max_risk_dollars: float
    open_positions: int
    has_correlated_position: bool
    current_pnl: float
    avg_expectancy_20: float            # average P&L per trade over last 20
    avg_trades_per_day: float


@dataclass
class RiskCheck:
    """Result of a pre-trade risk evaluation."""
    approved: bool
    reason: str
    adjusted_size: float
    daily_budget_remaining: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_DAILY_LOSS_PCT = 0.03               # 3 % of balance
MAX_RISK_PER_TRADE_PCT = 0.01           # 1 % of balance
MAX_CORRELATED_POSITIONS = 2
CONSECUTIVE_LOSS_CIRCUIT_BREAKER = 3


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------
class RiskManager:
    """Evaluate whether a new trade is permissible given account state and
    internal loss tracking.

    The manager keeps a per-instrument consecutive-loss counter that is
    updated externally via :meth:`record_result`.
    """

    def __init__(self) -> None:
        # instrument -> consecutive loss count
        self._consecutive_losses: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def check_risk(
        self,
        account: AccountState,
        instrument: str,
        risk_per_trade: float,
    ) -> RiskCheck:
        """Run all risk rules and return a ``RiskCheck``.

        Parameters
        ----------
        account : AccountState
            Current account snapshot.
        instrument : str
            Instrument symbol for the proposed trade.
        risk_per_trade : float
            Desired risk in dollars for this trade.
        """
        balance = account.balance
        daily_budget_total = balance * MAX_DAILY_LOSS_PCT
        daily_budget_remaining = daily_budget_total * (1.0 - account.daily_loss_used_pct / MAX_DAILY_LOSS_PCT) \
            if account.daily_loss_used_pct < MAX_DAILY_LOSS_PCT else 0.0
        # Clamp to non-negative
        daily_budget_remaining = max(0.0, daily_budget_remaining)

        max_trade_risk = balance * MAX_RISK_PER_TRADE_PCT

        # ----- Rule 1: daily loss cap -----
        if account.daily_loss_used_pct >= MAX_DAILY_LOSS_PCT:
            return RiskCheck(
                approved=False,
                reason=f"Daily loss limit reached ({account.daily_loss_used_pct:.2%} >= {MAX_DAILY_LOSS_PCT:.0%}).",
                adjusted_size=0.0,
                daily_budget_remaining=0.0,
            )

        # ----- Rule 2: per-trade risk cap -----
        if risk_per_trade > max_trade_risk:
            risk_per_trade = max_trade_risk  # auto-adjust downward

        # ----- Rule 3: correlated position limit -----
        if account.has_correlated_position and account.open_positions >= MAX_CORRELATED_POSITIONS:
            return RiskCheck(
                approved=False,
                reason=f"Max correlated positions ({MAX_CORRELATED_POSITIONS}) already open.",
                adjusted_size=0.0,
                daily_budget_remaining=round(daily_budget_remaining, 2),
            )

        # ----- Rule 4: consecutive-loss circuit breaker -----
        consec = self._consecutive_losses.get(instrument, 0)
        if consec >= CONSECUTIVE_LOSS_CIRCUIT_BREAKER:
            return RiskCheck(
                approved=False,
                reason=f"Circuit breaker: {consec} consecutive losses on {instrument}.",
                adjusted_size=0.0,
                daily_budget_remaining=round(daily_budget_remaining, 2),
            )

        # ----- Rule 5: risk can't exceed remaining daily budget -----
        if risk_per_trade > daily_budget_remaining:
            risk_per_trade = daily_budget_remaining

        # If after all adjustments the risk is effectively zero, reject
        if risk_per_trade <= 0:
            return RiskCheck(
                approved=False,
                reason="No remaining risk budget for today.",
                adjusted_size=0.0,
                daily_budget_remaining=round(daily_budget_remaining, 2),
            )

        return RiskCheck(
            approved=True,
            reason="Trade approved.",
            adjusted_size=round(risk_per_trade, 2),
            daily_budget_remaining=round(daily_budget_remaining - risk_per_trade, 2),
        )

    # ------------------------------------------------------------------ #
    # Loss tracking
    # ------------------------------------------------------------------ #

    def record_result(self, instrument: str, is_win: bool) -> None:
        """Update the consecutive-loss counter for *instrument*.

        Call this after every closed trade so the circuit breaker stays
        accurate.
        """
        if is_win:
            self._consecutive_losses[instrument] = 0
        else:
            self._consecutive_losses[instrument] = self._consecutive_losses.get(instrument, 0) + 1

    def reset_consecutive_losses(self, instrument: str) -> None:
        """Manually reset the circuit breaker for *instrument*."""
        self._consecutive_losses[instrument] = 0

    def get_consecutive_losses(self, instrument: str) -> int:
        """Return the current consecutive-loss count for *instrument*."""
        return self._consecutive_losses.get(instrument, 0)
