"""M14 Quant Analytics — Advanced performance analytics using numpy."""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class QuantAnalytics:
    """Computes advanced quantitative metrics from trade results."""

    def __init__(self, risk_free_rate: float = 0.05, periods_per_year: float = 252.0):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate_metrics(self, trades: list) -> dict:
        """Calculate comprehensive quantitative metrics from a list of trades.

        Args:
            trades: List of trade objects or dicts. Each must have at minimum
                    a 'pnl' field. Optionally 'instrument' for per-instrument stats.

        Returns:
            Dictionary with Sharpe, Sortino, max drawdown, Calmar, win rate
            by instrument, and expectancy curve.
        """
        if not trades:
            return self._empty_metrics()

        pnls = np.array([self._get_pnl(t) for t in trades], dtype=np.float64)

        sharpe = self._sharpe_ratio(pnls)
        sortino = self._sortino_ratio(pnls)
        max_dd = self._max_drawdown(pnls)
        calmar = self._calmar_ratio(pnls, max_dd)
        win_rate_by_instrument = self._win_rate_by_instrument(trades)
        expectancy_curve = self._expectancy_curve(pnls)

        total_pnl = float(np.sum(pnls))
        win_rate = float(np.mean(pnls > 0)) if len(pnls) > 0 else 0.0
        avg_win = float(np.mean(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
        avg_loss = float(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else 0.0

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": total_pnl,
            "trade_count": len(pnls),
            "win_rate_by_instrument": win_rate_by_instrument,
            "expectancy_curve": expectancy_curve,
        }

    def _get_pnl(self, trade: Any) -> float:
        """Extract PnL from a trade object or dict."""
        if isinstance(trade, dict):
            return float(trade.get("pnl", 0.0))
        return float(getattr(trade, "pnl", 0.0))

    def _get_instrument(self, trade: Any) -> str:
        """Extract instrument from a trade object or dict."""
        if isinstance(trade, dict):
            return str(trade.get("instrument", "UNKNOWN"))
        return str(getattr(trade, "instrument", "UNKNOWN"))

    def _sharpe_ratio(self, pnls: np.ndarray) -> float:
        """Annualized Sharpe ratio.
        Sharpe = (mean_return - risk_free_per_period) / std_return * sqrt(periods)
        """
        if len(pnls) < 2:
            return 0.0

        mean_r = np.mean(pnls)
        std_r = np.std(pnls, ddof=1)
        if std_r == 0:
            return 0.0

        daily_rf = self.risk_free_rate / self.periods_per_year
        sharpe = (mean_r - daily_rf) / std_r * np.sqrt(self.periods_per_year)
        return float(sharpe)

    def _sortino_ratio(self, pnls: np.ndarray) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        if len(pnls) < 2:
            return 0.0

        mean_r = np.mean(pnls)
        daily_rf = self.risk_free_rate / self.periods_per_year
        downside = pnls[pnls < 0]

        if len(downside) == 0:
            return float('inf') if mean_r > 0 else 0.0

        downside_std = np.std(downside, ddof=1)
        if downside_std == 0:
            return 0.0

        sortino = (mean_r - daily_rf) / downside_std * np.sqrt(self.periods_per_year)
        return float(sortino)

    def _max_drawdown(self, pnls: np.ndarray) -> float:
        """Maximum drawdown from peak equity."""
        if len(pnls) == 0:
            return 0.0

        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = peak - cumulative
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _calmar_ratio(self, pnls: np.ndarray, max_dd: float) -> float:
        """Calmar ratio = annualized return / max drawdown."""
        if max_dd == 0 or len(pnls) == 0:
            return 0.0

        total_return = float(np.sum(pnls))
        n_periods = len(pnls)
        annualized = total_return * (self.periods_per_year / n_periods)
        return float(annualized / max_dd)

    def _win_rate_by_instrument(self, trades: list) -> dict[str, float]:
        """Calculate win rate grouped by instrument."""
        instrument_trades: dict[str, list[float]] = {}

        for trade in trades:
            inst = self._get_instrument(trade)
            pnl = self._get_pnl(trade)
            if inst not in instrument_trades:
                instrument_trades[inst] = []
            instrument_trades[inst].append(pnl)

        result = {}
        for inst, pnls in instrument_trades.items():
            arr = np.array(pnls)
            result[inst] = float(np.mean(arr > 0)) if len(arr) > 0 else 0.0

        return result

    def _expectancy_curve(self, pnls: np.ndarray) -> list[float]:
        """Running expectancy (cumulative average PnL) over the trade sequence."""
        if len(pnls) == 0:
            return []

        cumsum = np.cumsum(pnls)
        counts = np.arange(1, len(pnls) + 1, dtype=np.float64)
        running_expectancy = cumsum / counts
        return running_expectancy.tolist()

    def _empty_metrics(self) -> dict:
        """Return empty metrics dict."""
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_pnl": 0.0,
            "trade_count": 0,
            "win_rate_by_instrument": {},
            "expectancy_curve": [],
        }
