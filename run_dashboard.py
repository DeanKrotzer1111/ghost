#!/usr/bin/env python3
"""Ghost v5.5 Dashboard — run backtest then launch interactive trade visualizer.

Usage:
    python run_dashboard.py --data-dir GLBX-20260307-9Y6LVS8EUA
    python run_dashboard.py --data-dir GLBX-20260307-9Y6LVS8EUA --instrument MNQ
    python run_dashboard.py --data-dir GLBX-20260307-9Y6LVS8EUA --port 3000 --debug
"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from ghost.modules.m00_backtest.engine import BacktestConfig
from ghost.modules.m10_dashboard.run_dashboard import run_and_launch


def main():
    parser = argparse.ArgumentParser(description="Ghost v5.5 Trade Visualization Dashboard")
    parser.add_argument(
        "--data-dir",
        default="GLBX-20260307-9Y6LVS8EUA",
        help="Path to Databento data directory",
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help="Single instrument to test (e.g., MNQ). Omit to test all available.",
    )
    parser.add_argument("--port", type=int, default=3000, help="Dashboard port (default: 3000)")
    parser.add_argument("--balance", type=float, default=50000.0, help="Starting balance")
    parser.add_argument("--risk-pct", type=float, default=0.01, help="Max risk per trade")
    parser.add_argument("--tqs-min", type=int, default=85, help="Minimum TQS score")
    parser.add_argument("--confluence-min", type=float, default=0.55, help="Min confluence score")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")

    args = parser.parse_args()

    config = BacktestConfig(
        initial_balance=args.balance,
        max_risk_pct=args.risk_pct,
        tqs_minimum=args.tqs_min,
        confluence_minimum=args.confluence_min,
    )

    run_and_launch(
        data_dir=args.data_dir,
        config=config,
        instrument=args.instrument,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
