"""Ghost Dashboard Runner — runs backtest then launches the Dash visualization."""
import time
import structlog
from typing import Dict, List, Optional

from ghost.modules.m01_data.loader import DatabentoLoader
from ghost.modules.m00_backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from ghost.modules.m10_dashboard.app import create_app

logger = structlog.get_logger()


def run_and_launch(
    data_dir: str,
    config: BacktestConfig = None,
    instrument: Optional[str] = None,
    port: int = 3000,
    debug: bool = False,
):
    """Run backtest(s) and launch the dashboard.

    Args:
        data_dir: Path to Databento data directory.
        config: Backtest configuration. Uses defaults if None.
        instrument: Single instrument to test, or None for all available.
        port: Port to run the dashboard on.
        debug: Enable Dash debug mode.
    """
    config = config or BacktestConfig()
    loader = DatabentoLoader(data_dir)

    if instrument:
        instruments = [instrument]
    else:
        instruments = loader.list_instruments()
        # Filter to common instruments if many are available
        preferred = {"MNQ", "MES", "MGC", "SIL"}
        available = [i for i in instruments if i in preferred]
        if available:
            instruments = available
        elif not instruments:
            logger.error("dashboard.no_instruments", data_dir=data_dir)
            print(f"No instruments found in {data_dir}")
            return

    logger.info("dashboard.instruments", instruments=instruments)

    backtest_results: Dict[str, BacktestResult] = {}
    bar_data: Dict[str, list] = {}
    equity_curves: Dict[str, List[float]] = {}

    for inst in instruments:
        print(f"\n  Running backtest for {inst}...")
        df = loader.load_instrument(inst)
        if df.empty:
            logger.warning("dashboard.no_data", instrument=inst)
            continue

        bars = loader.to_bars(df)
        logger.info("dashboard.bars_loaded", instrument=inst, count=len(bars))

        engine = BacktestEngine(config)
        start = time.time()
        result = engine.run(bars, inst)
        elapsed = time.time() - start

        backtest_results[inst] = result
        equity_curves[inst] = engine.equity_curve

        # Downsample bars for the chart — sending 100k+ bars to the browser is slow.
        # Keep every Nth bar so we have ~3000-5000 points max.
        step = max(1, len(bars) // 4000)
        sampled = bars[::step]
        bar_data[inst] = [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in sampled
        ]

        # Also downsample equity curve to match
        eq_step = max(1, len(engine.equity_curve) // 4000)
        equity_curves[inst] = engine.equity_curve[::eq_step]

        _print_summary(inst, result, elapsed)

    if not backtest_results:
        print("\nNo backtest results to display.")
        return

    print(f"\n  Launching dashboard at http://localhost:{port}")
    print(f"  Press Ctrl+C to stop.\n")

    app = create_app(
        backtest_results=backtest_results,
        bar_data=bar_data,
        equity_curves=equity_curves,
        initial_balance=config.initial_balance,
    )
    app.run(host="0.0.0.0", port=port, debug=debug)


def _print_summary(instrument: str, result: BacktestResult, elapsed: float):
    """Print a compact backtest summary."""
    pnl_sign = "+" if result.total_pnl >= 0 else ""
    print(f"  {instrument}: {result.total_trades} trades | "
          f"WR {result.win_rate * 100:.1f}% | "
          f"P&L {pnl_sign}${result.total_pnl:,.2f} | "
          f"PF {result.profit_factor:.2f} | "
          f"Sharpe {result.sharpe_ratio:.2f} | "
          f"{elapsed:.1f}s")
