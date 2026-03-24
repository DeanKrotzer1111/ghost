"""M06 — Position Sizing Engine for Ghost v5.5."""
import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SizeResult:
    """Outcome of a position-size calculation."""
    contracts: int
    risk_dollars: float
    risk_pct: float
    risk_reward_ratio: Optional[float] = None


# ---------------------------------------------------------------------------
# Position Sizer
# ---------------------------------------------------------------------------

class PositionSizer:
    """Calculate position sizes in contracts (or lots) given account and
    instrument parameters.
    """

    def calculate(
        self,
        instrument: str,
        entry: float,
        stop: float,
        account_balance: float,
        max_risk_pct: float,
        tick_size: float,
        point_value: float,
        size_multiplier: float = 1.0,
        take_profit: Optional[float] = None,
    ) -> SizeResult:
        """Compute the number of contracts for a single trade.

        Parameters
        ----------
        instrument : str
            Symbol name (informational; not used in math).
        entry : float
            Planned entry price.
        stop : float
            Stop-loss price.
        account_balance : float
            Current account balance in dollars.
        max_risk_pct : float
            Maximum fraction of account to risk (e.g. 0.01 = 1 %).
        tick_size : float
            Minimum price increment (e.g. 0.25 for ES futures).
        point_value : float
            Dollar value per full point move per contract (e.g. 50.0 for ES).
        size_multiplier : float
            Optional multiplier applied after sizing (e.g. for scaling).
        take_profit : float, optional
            Take-profit price — used to compute risk/reward ratio.

        Returns
        -------
        SizeResult
        """
        if account_balance <= 0 or max_risk_pct <= 0 or tick_size <= 0 or point_value <= 0:
            return SizeResult(contracts=0, risk_dollars=0.0, risk_pct=0.0)

        # Points of risk (always positive)
        stop_distance = abs(entry - stop)
        if stop_distance == 0:
            return SizeResult(contracts=0, risk_dollars=0.0, risk_pct=0.0)

        # Ticks of risk
        ticks = stop_distance / tick_size
        # Dollar risk per single contract
        risk_per_contract = ticks * tick_size * point_value

        # Max dollars we are willing to risk
        max_risk_dollars = account_balance * max_risk_pct

        # Raw contract count (float)
        raw_contracts = max_risk_dollars / risk_per_contract
        raw_contracts *= size_multiplier

        # Floor to whole contracts, minimum 1
        contracts = max(1, int(math.floor(raw_contracts)))

        # Actual risk with the chosen contract count
        actual_risk_dollars = contracts * risk_per_contract
        actual_risk_pct = actual_risk_dollars / account_balance if account_balance else 0.0

        # Enforce max risk: if 1 contract already exceeds the cap, still allow
        # it (minimum-1 rule) but flag the percentage correctly.
        # If more than 1 contract exceeds, scale back.
        if contracts > 1 and actual_risk_pct > max_risk_pct:
            contracts = max(1, int(math.floor(max_risk_dollars / risk_per_contract)))
            actual_risk_dollars = contracts * risk_per_contract
            actual_risk_pct = actual_risk_dollars / account_balance if account_balance else 0.0

        # Risk-reward ratio
        rr: Optional[float] = None
        if take_profit is not None:
            reward_distance = abs(take_profit - entry)
            if stop_distance > 0:
                rr = round(reward_distance / stop_distance, 2)

        return SizeResult(
            contracts=contracts,
            risk_dollars=round(actual_risk_dollars, 2),
            risk_pct=round(actual_risk_pct, 6),
            risk_reward_ratio=rr,
        )

    # ------------------------------------------------------------------ #
    # Kelly criterion helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def kelly_size(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_balance: float,
    ) -> float:
        """Return the Kelly-optimal fraction of the account to risk.

        Parameters
        ----------
        win_rate : float
            Historical win rate as a fraction (0.0 – 1.0).
        avg_win : float
            Average winning trade in dollars (positive).
        avg_loss : float
            Average losing trade in dollars (positive).
        account_balance : float
            Current account balance.

        Returns
        -------
        float
            Optimal fraction of balance to risk (0.0 – 1.0).  Returns 0.0
            when inputs are invalid or Kelly is negative (no edge).
        """
        if win_rate <= 0 or win_rate >= 1.0:
            return 0.0
        if avg_win <= 0 or avg_loss <= 0 or account_balance <= 0:
            return 0.0

        # Kelly formula: f* = (p * b - q) / b
        #   p = win_rate,  q = 1 - p,  b = avg_win / avg_loss
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly_fraction = (win_rate * b - q) / b

        # Clamp to [0, 1]
        kelly_fraction = max(0.0, min(1.0, kelly_fraction))

        return round(kelly_fraction, 6)
