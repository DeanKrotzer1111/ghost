"""Parameter Optimizer — grid search over backtest parameters to find optimal config.

Takes a list of BacktestTrades and current config, computes optimal:
  - stop_multiplier (1.0 - 3.0)
  - tp_ratio (1.5 - 4.0)
  - confluence_minimum (0.2 - 0.6)

Uses simple grid search, scoring by Sharpe ratio.
"""
import copy
import numpy as np
import structlog
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

logger = structlog.get_logger()


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_stop_multiplier: float
    best_tp1_ratio: float
    best_confluence_minimum: float
    best_sharpe: float
    best_win_rate: float
    best_profit_factor: float
    best_expectancy: float
    total_combinations: int
    results_grid: List[Dict[str, Any]]


class ParameterOptimizer:
    """Grid-search optimizer for backtest parameters.

    Usage:
        optimizer = ParameterOptimizer()
        result = optimizer.optimize(bars, instrument, base_config)
    """

    def __init__(
        self,
        stop_range: tuple = (1.0, 1.5, 2.0, 2.5, 3.0),
        tp1_range: tuple = (1.5, 2.0, 2.5, 3.0, 4.0),
        confluence_range: tuple = (0.20, 0.30, 0.40, 0.50, 0.60),
    ):
        self.stop_range = stop_range
        self.tp1_range = tp1_range
        self.confluence_range = confluence_range
        self._log = logger.bind(component="ParameterOptimizer")

    def optimize(self, bars: list, instrument: str, base_config=None) -> OptimizationResult:
        """Run grid search over parameter combinations.

        Args:
            bars: 1-minute bars for backtesting.
            instrument: Instrument symbol.
            base_config: Base BacktestConfig to modify.

        Returns:
            OptimizationResult with best parameters.
        """
        from ghost.modules.m00_backtest.engine import BacktestEngine, BacktestConfig

        if base_config is None:
            base_config = BacktestConfig()

        best_sharpe = -999.0
        best_wr = 0.0
        best_pf = 0.0
        best_exp = 0.0
        best_stop = base_config.stop_multiplier
        best_tp1 = base_config.tp1_ratio
        best_conf = base_config.confluence_minimum
        results_grid = []

        total_combos = len(self.stop_range) * len(self.tp1_range) * len(self.confluence_range)
        combo_num = 0

        self._log.info("optimizer_start",
                        instrument=instrument,
                        total_combinations=total_combos)

        for stop_mult in self.stop_range:
            for tp1_r in self.tp1_range:
                for conf_min in self.confluence_range:
                    combo_num += 1
                    config = copy.deepcopy(base_config)
                    config.stop_multiplier = stop_mult
                    config.tp1_ratio = tp1_r
                    config.confluence_minimum = conf_min

                    engine = BacktestEngine(config=config)
                    result = engine.run(bars, instrument)

                    entry = {
                        "stop_multiplier": stop_mult,
                        "tp1_ratio": tp1_r,
                        "confluence_minimum": conf_min,
                        "trades": result.total_trades,
                        "win_rate": round(result.win_rate * 100, 1),
                        "sharpe": round(result.sharpe_ratio, 2),
                        "profit_factor": round(result.profit_factor, 2),
                        "total_pnl": round(result.total_pnl, 2),
                        "expectancy": round(result.expectancy, 2),
                    }
                    results_grid.append(entry)

                    # Score: prioritize win rate + Sharpe
                    if result.total_trades >= 3:
                        score = result.win_rate * 50 + result.sharpe_ratio * 10 + min(result.profit_factor, 5) * 5
                        best_score = best_wr * 50 + best_sharpe * 10 + min(best_pf, 5) * 5
                        if score > best_score:
                            best_sharpe = result.sharpe_ratio
                            best_wr = result.win_rate
                            best_pf = result.profit_factor
                            best_exp = result.expectancy
                            best_stop = stop_mult
                            best_tp1 = tp1_r
                            best_conf = conf_min

                    if combo_num % 25 == 0:
                        self._log.info("optimizer_progress",
                                        combo=combo_num,
                                        total=total_combos,
                                        current_best_wr=round(best_wr * 100, 1))

        self._log.info("optimizer_complete",
                        best_stop=best_stop,
                        best_tp1=best_tp1,
                        best_conf=best_conf,
                        best_sharpe=round(best_sharpe, 2),
                        best_wr=round(best_wr * 100, 1))

        return OptimizationResult(
            best_stop_multiplier=best_stop,
            best_tp1_ratio=best_tp1,
            best_confluence_minimum=best_conf,
            best_sharpe=round(best_sharpe, 2),
            best_win_rate=round(best_wr * 100, 1),
            best_profit_factor=round(best_pf, 2),
            best_expectancy=round(best_exp, 2),
            total_combinations=total_combos,
            results_grid=results_grid,
        )

    def optimize_from_trades(self, trades: list, current_config=None) -> Dict[str, Any]:
        """Quick optimization from existing trades without re-running backtests.

        Analyzes trade characteristics to suggest parameter adjustments.

        Args:
            trades: List of BacktestTrade objects.
            current_config: Current BacktestConfig.

        Returns:
            Dict with suggested parameters.
        """
        if not trades:
            return {}

        from ghost.modules.m00_backtest.engine import BacktestConfig
        config = current_config or BacktestConfig()

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        stop_losses = [t for t in trades if t.outcome == "LOSS_STOP"]

        suggestions = {}

        # Analyze stop distance effectiveness
        if stop_losses:
            # Calculate how much of risk the stops captured
            risk_ratios = []
            for t in stop_losses:
                risk = abs(t.entry_price - t.stop)
                actual_loss = abs(t.entry_price - t.exit_price)
                if risk > 0:
                    risk_ratios.append(actual_loss / risk)

            if risk_ratios:
                avg_ratio = np.mean(risk_ratios)
                # If stops are being hit exactly (ratio ~1.0), suggest widening
                if avg_ratio > 0.9:
                    suggestions["stop_multiplier"] = min(3.0, config.stop_multiplier + 0.5)
                elif avg_ratio < 0.5:
                    suggestions["stop_multiplier"] = max(1.0, config.stop_multiplier - 0.25)

        # Analyze TP hit rates
        tp1_hits = sum(1 for t in trades if t.outcome == "WIN_TP1")
        tp2_hits = sum(1 for t in trades if t.outcome == "WIN_TP2")
        tp3_hits = sum(1 for t in trades if t.outcome == "WIN_TP3")

        total = len(trades)
        if total > 0:
            tp1_rate = tp1_hits / total
            # If TP1 is rarely hit, lower the ratio to make it easier
            if tp1_rate < 0.15 and config.tp1_ratio > 1.5:
                suggestions["tp1_ratio"] = max(1.5, config.tp1_ratio - 0.5)
            # If TP1 hits almost always but TP2/3 never, we might be leaving money on table
            elif tp1_rate > 0.5 and (tp2_hits + tp3_hits) / total > 0.1:
                suggestions["tp1_ratio"] = min(4.0, config.tp1_ratio + 0.5)

        # Analyze confluence threshold
        if trades:
            winning_confluences = [t.confluence_score for t in wins]
            losing_confluences = [t.confluence_score for t in losses]
            if winning_confluences and losing_confluences:
                avg_win_conf = np.mean(winning_confluences)
                avg_loss_conf = np.mean(losing_confluences)
                # If winning trades have notably higher confluence, raise the bar
                if avg_win_conf > avg_loss_conf + 0.05:
                    midpoint = (avg_win_conf + avg_loss_conf) / 2
                    suggestions["confluence_minimum"] = round(
                        max(0.20, min(0.60, midpoint)), 2
                    )

        return suggestions
