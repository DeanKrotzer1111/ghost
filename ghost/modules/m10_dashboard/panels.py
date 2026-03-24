"""M10 Dashboard — Formats trading data for display."""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DashboardData:
    """Aggregates and formats data from various modules for dashboard display.
    Pure data formatting — no web server, just dict output for backtesting."""

    def get_summary(
        self,
        journal_stats: Any = None,
        account_state: Any = None,
        regime_state: Any = None,
    ) -> dict:
        """Build a summary dict of dashboard metrics from module states.

        Args:
            journal_stats: JournalStats object (from M09) or dict with performance metrics.
            account_state: Dict or object with account info (balance, equity, margin, etc.).
            regime_state: Dict or object with current regime info (macro_regime, direction, etc.).

        Returns:
            Dictionary of formatted dashboard metrics.
        """
        summary: dict[str, Any] = {}

        # Performance metrics from journal
        summary["performance"] = self._format_performance(journal_stats)

        # Account state
        summary["account"] = self._format_account(account_state)

        # Regime state
        summary["regime"] = self._format_regime(regime_state)

        return summary

    def _format_performance(self, stats: Any) -> dict:
        """Extract performance metrics from JournalStats or dict."""
        if stats is None:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "expectancy": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
                "trade_count": 0,
            }

        if isinstance(stats, dict):
            return {
                "win_rate": stats.get("win_rate", 0.0),
                "avg_win": stats.get("avg_win", 0.0),
                "avg_loss": stats.get("avg_loss", 0.0),
                "expectancy": stats.get("expectancy", 0.0),
                "profit_factor": stats.get("profit_factor", 0.0),
                "max_drawdown": stats.get("max_drawdown", 0.0),
                "total_pnl": stats.get("total_pnl", 0.0),
                "trade_count": stats.get("trade_count", 0),
            }

        # Dataclass / object with attributes
        return {
            "win_rate": getattr(stats, "win_rate", 0.0),
            "avg_win": getattr(stats, "avg_win", 0.0),
            "avg_loss": getattr(stats, "avg_loss", 0.0),
            "expectancy": getattr(stats, "expectancy", 0.0),
            "profit_factor": getattr(stats, "profit_factor", 0.0),
            "max_drawdown": getattr(stats, "max_drawdown", 0.0),
            "total_pnl": getattr(stats, "total_pnl", 0.0),
            "trade_count": getattr(stats, "trade_count", 0),
        }

    def _format_account(self, account: Any) -> dict:
        """Extract account info."""
        if account is None:
            return {
                "balance": 0.0,
                "equity": 0.0,
                "margin_used": 0.0,
                "open_positions": 0,
            }

        if isinstance(account, dict):
            return {
                "balance": account.get("balance", 0.0),
                "equity": account.get("equity", 0.0),
                "margin_used": account.get("margin_used", 0.0),
                "open_positions": account.get("open_positions", 0),
            }

        return {
            "balance": getattr(account, "balance", 0.0),
            "equity": getattr(account, "equity", 0.0),
            "margin_used": getattr(account, "margin_used", 0.0),
            "open_positions": getattr(account, "open_positions", 0),
        }

    def _format_regime(self, regime: Any) -> dict:
        """Extract regime info."""
        if regime is None:
            return {
                "macro_regime": "UNKNOWN",
                "macro_direction": "UNKNOWN",
                "macro_confidence": 0.0,
                "vix": 0.0,
            }

        if isinstance(regime, dict):
            return {
                "macro_regime": regime.get("macro_regime", "UNKNOWN"),
                "macro_direction": regime.get("macro_direction", "UNKNOWN"),
                "macro_confidence": regime.get("macro_confidence", 0.0),
                "vix": regime.get("vix", 0.0),
            }

        return {
            "macro_regime": getattr(regime, "macro_regime", "UNKNOWN"),
            "macro_direction": getattr(regime, "macro_direction", "UNKNOWN"),
            "macro_confidence": getattr(regime, "macro_confidence", 0.0),
            "vix": getattr(regime, "vix", 0.0),
        }
