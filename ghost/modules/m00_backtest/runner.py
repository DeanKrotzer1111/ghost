"""Backtest Runner — loads Databento data and runs the Ghost backtest engine."""
import sys
import time
import structlog
from ghost.modules.m01_data.loader import DatabentoLoader
from ghost.modules.m00_backtest.engine import BacktestEngine, BacktestConfig

logger = structlog.get_logger()


class BacktestRunner:
    """Convenience runner that loads data and executes backtests."""

    def __init__(self, data_dir: str, config: BacktestConfig = None):
        self.loader = DatabentoLoader(data_dir)
        self.config = config or BacktestConfig()

    def run_instrument(self, instrument: str) -> dict:
        """Run backtest for a single instrument."""
        logger.info("runner.loading_data", instrument=instrument)

        df = self.loader.load_instrument(instrument)
        if df.empty:
            logger.error("runner.no_data", instrument=instrument)
            return {"error": f"No data for {instrument}"}

        bars = self.loader.to_bars(df)
        logger.info("runner.bars_loaded", instrument=instrument, count=len(bars))

        engine = BacktestEngine(self.config)
        start = time.time()
        result = engine.run(bars, instrument)
        elapsed = time.time() - start

        summary = {
            "instrument": instrument,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": round(result.win_rate * 100, 1),
            "total_pnl": round(result.total_pnl, 2),
            "avg_win": round(result.avg_win, 2),
            "avg_loss": round(result.avg_loss, 2),
            "expectancy": round(result.expectancy, 2),
            "profit_factor": round(result.profit_factor, 2),
            "max_drawdown": round(result.max_drawdown, 2),
            "max_drawdown_pct": round(result.max_drawdown_pct * 100, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 2),
            "final_balance": round(result.final_balance, 2),
            "signals_generated": result.signals_generated,
            "signals_rejected": result.signals_rejected,
            "shadow_signals": result.shadow_signals,
            "elapsed_seconds": round(elapsed, 1),
        }

        self._print_report(summary)
        return summary

    def run_all(self) -> dict:
        """Run backtest across all available instruments."""
        instruments = self.loader.list_instruments()
        logger.info("runner.instruments_found", instruments=instruments)

        results = {}
        for inst in instruments:
            results[inst] = self.run_instrument(inst)

        self._print_portfolio_summary(results)
        return results

    def _print_report(self, s: dict):
        """Print a single-instrument backtest report."""
        print(f"\n{'='*60}")
        print(f"  GHOST BACKTEST — {s['instrument']}")
        print(f"{'='*60}")
        print(f"  Trades:         {s['total_trades']} ({s['winning_trades']}W / {s['losing_trades']}L)")
        print(f"  Win Rate:       {s['win_rate']}%")
        print(f"  Total P&L:      ${s['total_pnl']:,.2f}")
        print(f"  Avg Win:        ${s['avg_win']:,.2f}")
        print(f"  Avg Loss:       ${s['avg_loss']:,.2f}")
        print(f"  Expectancy:     ${s['expectancy']:,.2f} / trade")
        print(f"  Profit Factor:  {s['profit_factor']}")
        print(f"  Max Drawdown:   ${s['max_drawdown']:,.2f} ({s['max_drawdown_pct']}%)")
        print(f"  Sharpe Ratio:   {s['sharpe_ratio']}")
        print(f"  Final Balance:  ${s['final_balance']:,.2f}")
        print(f"  Signals:        {s['signals_generated']} gen / {s['signals_rejected']} rej / {s['shadow_signals']} shadow")
        print(f"  Time:           {s['elapsed_seconds']}s")
        print(f"{'='*60}\n")

    def _print_portfolio_summary(self, results: dict):
        """Print combined portfolio summary."""
        valid = {k: v for k, v in results.items() if "error" not in v}
        if not valid:
            print("No valid results.")
            return

        total_pnl = sum(v["total_pnl"] for v in valid.values())
        total_trades = sum(v["total_trades"] for v in valid.values())
        total_wins = sum(v["winning_trades"] for v in valid.values())
        wr = total_wins / total_trades * 100 if total_trades > 0 else 0

        print(f"\n{'='*60}")
        print(f"  GHOST PORTFOLIO BACKTEST SUMMARY")
        print(f"{'='*60}")
        print(f"  Instruments:    {len(valid)}")
        print(f"  Total Trades:   {total_trades} ({total_wins}W)")
        print(f"  Portfolio WR:   {wr:.1f}%")
        print(f"  Total P&L:      ${total_pnl:,.2f}")
        print(f"{'='*60}\n")
