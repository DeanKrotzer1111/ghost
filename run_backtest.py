#!/usr/bin/env python3
"""Ghost v5.5 Backtest Runner — run from project root."""
import sys
import os
import argparse

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from ghost.modules.m00_backtest.runner import BacktestRunner
from ghost.modules.m00_backtest.engine import BacktestConfig


def main():
    parser = argparse.ArgumentParser(description="Ghost v5.5 Backtester")
    parser.add_argument(
        "--data-dir",
        default="GLBX-20260307-9Y6LVS8EUA",
        help="Path to Databento data directory",
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help="Single instrument to test (e.g., MNQ). Omit to test all.",
    )
    parser.add_argument("--balance", type=float, default=50000.0, help="Starting balance")
    parser.add_argument("--risk-pct", type=float, default=0.01, help="Max risk per trade")
    parser.add_argument("--tqs-min", type=int, default=85, help="Minimum TQS score")
    parser.add_argument("--confluence-min", type=float, default=0.55, help="Min confluence score")

    args = parser.parse_args()

    config = BacktestConfig(
        initial_balance=args.balance,
        max_risk_pct=args.risk_pct,
        tqs_minimum=args.tqs_min,
        confluence_minimum=args.confluence_min,
    )

    runner = BacktestRunner(args.data_dir, config)

    if args.instrument:
        runner.run_instrument(args.instrument)
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
