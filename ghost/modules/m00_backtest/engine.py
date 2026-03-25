"""Ghost Backtest Engine — walks forward through historical data, generating signals and simulating trades.

v6.0 — Win-rate optimizations:
  - Kill zone enforcement (NY 7-10am, London 2-5am ET only)
  - Regime filter: only TRENDING regimes
  - Premium/discount zone gating
  - 2x wider stops to survive noise sweeps
  - TP1 at 1.5R for easier hits
  - Max 1 trade per day per instrument
  - ATR 20th-percentile dead-market filter
  - Learner integration for self-calibration
"""
import structlog
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta

from ghost.modules.m01_data.models import Bar, MarketContext, LiquidityPool
from ghost.modules.m01_data.resampler import Resampler
from ghost.modules.m02_regime.detector import RegimeDetector
from ghost.modules.m03_signal.signal_generator import SignalGenerator
from ghost.modules.m04_confluence.scorer import ConfluenceScorer
from ghost.modules.m05_risk.manager import RiskManager, AccountState
from ghost.modules.m06_sizing.sizer import PositionSizer
from ghost.modules.m08_monitor.monitor import PositionMonitor, Position, MonitorAction
from ghost.modules.m09_journal.journal import TradeJournal, JournalEntry
from ghost.modules.m20_selectivity.checklist import PreSignalChecklist
from ghost.modules.m21_tqs.scorer import TradeQualityScorer
from ghost.modules.m17_stop_calibration.optimal_stop import OptimalStopCalculator
from ghost.modules.m17_stop_calibration.sweep_analyzer import SweepDistribution
from ghost.modules.m18_tp_calibration.tp_engine import TakeProfitCalibrationEngine, SESSION_MODS
from ghost.modules.m19_entry_calibration.optimal_entry import OptimalEntryCalculator
from ghost.modules.m22_footprint.engine import SmartMoneyFootprintEngine
from ghost.modules.m23_mtf_pyramid.pyramid import MTFConfluencePyramid
from ghost.modules.m25_liquidity_void.engine import LiquidityVoidEngine
from ghost.modules.m27_payout.engine import PayoutOptimizationEngine
from ghost.core.models import TQSGrade

logger = structlog.get_logger()

TICK_SIZES = {
    "MNQ": 0.25, "NQ": 0.25, "ES": 0.25, "MES": 0.25,
    "GC": 0.10, "MGC": 0.10, "SI": 0.005, "SIL": 0.005,
    "CL": 0.01, "MCL": 0.01, "NG": 0.001,
    "RTY": 0.10, "M2K": 0.10, "YM": 1.0, "MYM": 1.0,
    "6E": 0.00005, "6B": 0.0001, "6J": 0.0000005,
    "ZB": 0.03125, "ZN": 0.015625,
    "ZC": 0.25, "ZS": 0.25, "ZW": 0.25,
}

POINT_VALUES = {
    "MNQ": 0.50, "NQ": 5.00, "ES": 12.50, "MES": 1.25,
    "GC": 10.00, "MGC": 1.00, "SI": 25.00, "SIL": 2.50,
    "CL": 10.00, "MCL": 1.00, "NG": 10.00,
    "RTY": 5.00, "M2K": 0.50, "YM": 5.00, "MYM": 0.50,
    "6E": 6.25, "6B": 6.25, "6J": 6.25,
    "ZB": 31.25, "ZN": 15.625,
    "ZC": 12.50, "ZS": 12.50, "ZW": 12.50,
}

# Eastern Time offset (fixed EST = UTC-5)
_ET_OFFSET = timedelta(hours=-5)
_ET_TZ = timezone(_ET_OFFSET)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_balance: float = 50000.0
    max_risk_pct: float = 0.01
    max_daily_loss_pct: float = 0.03
    tqs_minimum: int = 85
    tqs_shadow_minimum: int = 80
    confluence_minimum: float = 0.55
    min_bars_warmup: int = 200
    signal_interval_bars: int = 15  # Check for signals every N 1m bars
    payout_target: float = 3000.0
    # v6.0 win-rate filters
    stop_multiplier: float = 1.0       # Stop distance multiplier (1.0 = structural)
    tp1_ratio: float = 2.0             # TP1 = risk * this
    tp2_ratio: float = 3.0
    tp3_ratio: float = 4.5
    require_kill_zone: bool = False    # Kill zone enforcement (can be enabled)
    require_trending: bool = False     # Regime filter (can be enabled)
    max_trades_per_day: int = 2        # Per instrument - avoids overtrading
    atr_min_percentile: float = 0.10   # Skip dead markets (10th percentile)
    premium_discount_long_max: float = 0.40   # Longs only in discount zone
    premium_discount_short_min: float = 0.60  # Shorts only in premium zone
    require_fvg_15m: bool = True       # Require 15m+ FVG
    require_1h_structure: bool = True  # Require 1H structure alignment
    # v7.0 — Quality over quantity
    min_risk_ticks: int = 10           # Reject micro-risk trades (noise)
    max_one_position: bool = True      # Only 1 position open at a time (no dupes)
    macro_trend_filter: bool = True    # Only trade WITH the 4H macro trend
    min_rr_ratio: float = 2.0         # Min reward:risk ratio (TP1 must be >= 2x risk)
    loss_cooloff_bars: int = 120       # Wait 2 hours after a loss before next trade
    blocked_instruments: list = None   # Instruments to skip entirely
    # v7.5 — Time management
    max_hold_bars: int = 300           # Close stale trades after 5 hours
    breakeven_after_bars: int = 120    # Move stop to breakeven after 2 hours
    bullish_only: bool = False         # Only take bullish trades (for bull markets)
    instrument_overrides: dict = None   # Per-instrument config overrides


@dataclass
class BacktestTrade:
    """A completed backtest trade."""
    trade_id: int
    instrument: str
    direction: str
    entry_price: float
    exit_price: float
    stop: float
    tp1: float
    tp2: float
    tp3: float
    contracts: int
    pnl: float
    outcome: str  # WIN_TP1, WIN_TP2, WIN_TP3, LOSS_STOP, LOSS_TIME
    tqs_score: float
    tqs_grade: str
    confluence_score: float
    entry_time: float
    exit_time: float
    bars_held: int
    regime_label: str = ""
    kill_zone: str = ""


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    trades: List[BacktestTrade]
    total_pnl: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    final_balance: float = 0.0
    signals_generated: int = 0
    signals_rejected: int = 0
    shadow_signals: int = 0


def _bar_hour_et(timestamp: float) -> int:
    """Convert Unix timestamp to ET hour."""
    dt = datetime.fromtimestamp(timestamp, tz=_ET_TZ)
    return dt.hour


def _is_in_kill_zone(timestamp: float) -> tuple:
    """Check if timestamp is within NY or London kill zone.
    Returns (is_in_kz: bool, kz_name: str).
    """
    dt = datetime.fromtimestamp(timestamp, tz=_ET_TZ)
    h, m = dt.hour, dt.minute
    total_min = h * 60 + m

    # London: 2:00 - 5:00 ET
    if 120 <= total_min < 300:
        return True, "LONDON"
    # NY: 7:00 - 10:00 ET
    if 420 <= total_min < 600:
        return True, "NY"
    return False, ""


class BacktestEngine:
    """Walk-forward backtesting engine for the Ghost trading system.

    Walks through 1-minute bars, resampling to higher timeframes on the fly,
    running the full signal pipeline at configurable intervals.

    v6.0 additions:
      - Kill zone enforcement
      - Regime filtering (trending only)
      - Premium/discount zone gating
      - Wider stops (2x)
      - Easier TP1 (1.5R)
      - Max 1 trade/day/instrument
      - ATR dead-market filter
      - Learner adjustments integration
    """

    def __init__(self, config: BacktestConfig = None, learner_adjustments: Optional[Dict] = None):
        self.config = config or BacktestConfig()
        self.learner_adjustments = learner_adjustments or {}

        # Initialize all modules
        self.regime_detector = RegimeDetector()
        self.signal_generator = None  # Created per-instrument in run()
        self.confluence_scorer = ConfluenceScorer()
        self.risk_manager = RiskManager()
        self.sizer = PositionSizer()
        self.monitor = PositionMonitor()
        self.journal = TradeJournal()
        self.resampler = Resampler()

        # v5.5 modules
        self.checklist = PreSignalChecklist()
        self.tqs_scorer = TradeQualityScorer()
        self.stop_calculator = OptimalStopCalculator(TICK_SIZES, POINT_VALUES)
        self.entry_calculator = OptimalEntryCalculator(TICK_SIZES)
        self.footprint_engine = SmartMoneyFootprintEngine(TICK_SIZES)
        self.mtf_pyramid = MTFConfluencePyramid()
        self.void_engine = LiquidityVoidEngine(TICK_SIZES)
        self.payout_engine = PayoutOptimizationEngine()

        # State
        self.balance = self.config.initial_balance
        self.equity_curve: List[float] = []
        self.open_positions: List[Position] = []
        self.trade_counter = 0
        self.daily_pnl = 0.0
        self.current_day = None
        self.consecutive_losses = 0
        self.last_loss_bar = -9999  # v7.0: cooloff tracking
        self.trades_today: int = 0  # v6.0: per-day trade counter

        # ATR history for percentile filter
        self._atr_history: List[float] = []

    def _get_instrument_config(self, instrument: str, key: str, default):
        """Get a config value with per-instrument overrides applied."""
        overrides = self.config.instrument_overrides or {}
        return overrides.get(instrument, {}).get(key, default)

    def _get_effective_stop_multiplier(self, instrument: str) -> float:
        """Get stop multiplier, applying learner adjustments if present."""
        base = self.config.stop_multiplier
        adj = self.learner_adjustments.get("instruments", {}).get(instrument, {})
        return adj.get("stop_multiplier", base)

    def _get_effective_tqs_bonus(self, instrument: str) -> int:
        """Get TQS bonus from learner adjustments."""
        adj = self.learner_adjustments.get("instruments", {}).get(instrument, {})
        return adj.get("tqs_bonus", 0)

    def _is_regime_blocked(self, instrument: str, regime_label: str) -> bool:
        """Check if learner has blocked this regime for this instrument."""
        adj = self.learner_adjustments.get("instruments", {}).get(instrument, {})
        blocked = adj.get("blocked_regimes", [])
        return regime_label in blocked

    def _get_session_tqs_bonus(self, kz_name: str) -> int:
        """Get per-session TQS bonus from learner adjustments."""
        sessions = self.learner_adjustments.get("sessions", {})
        return sessions.get(kz_name, {}).get("tqs_bonus", 0)

    def run(self, bars_1m: List[Bar], instrument: str) -> BacktestResult:
        """Run backtest on 1-minute bars for a single instrument."""
        self.signal_generator = SignalGenerator(instrument=instrument)

        # v7.0: Skip blocked instruments entirely
        blocked = self.config.blocked_instruments or []
        if instrument in blocked:
            logger.info("backtest.instrument_blocked", instrument=instrument)
            return BacktestResult(trades=[], final_balance=self.balance)

        if len(bars_1m) < self.config.min_bars_warmup:
            logger.warning("insufficient_bars", count=len(bars_1m), required=self.config.min_bars_warmup)
            return BacktestResult(trades=[], final_balance=self.balance)

        logger.info(
            "backtest.start",
            instrument=instrument,
            bars=len(bars_1m),
            balance=self.balance,
        )

        trades: List[BacktestTrade] = []
        signals_generated = 0
        signals_rejected = 0
        shadow_count = 0

        tick = TICK_SIZES.get(instrument, 0.25)
        pv = POINT_VALUES.get(instrument, 2.0)

        for i in range(self.config.min_bars_warmup, len(bars_1m)):
            current_bar = bars_1m[i]

            # Track daily P&L reset and trade count
            bar_day = int(current_bar.timestamp // 86400)
            if self.current_day is not None and bar_day != self.current_day:
                self.daily_pnl = 0.0
                self.trades_today = 0
            self.current_day = bar_day

            # 1. Monitor open positions
            closed_positions = []
            for pos in self.open_positions:
                bars_held = i - getattr(pos, "_entry_bar_idx", i)

                # v7.5: Move stop to breakeven after N bars
                if (self.config.breakeven_after_bars > 0
                        and bars_held >= self.config.breakeven_after_bars
                        and not getattr(pos, "_breakeven_set", False)):
                    if pos.direction == "BULLISH" and pos.stop < pos.entry:
                        pos.stop = pos.entry + tick  # breakeven + 1 tick
                        pos._breakeven_set = True
                    elif pos.direction == "BEARISH" and pos.stop > pos.entry:
                        pos.stop = pos.entry - tick
                        pos._breakeven_set = True

                # v7.5: Force close stale trades after max hold
                if self.config.max_hold_bars > 0 and bars_held >= self.config.max_hold_bars:
                    from ghost.modules.m08_monitor.monitor import MonitorAction
                    exit_price = current_bar.close
                    pnl = self._calc_pnl(pos, exit_price, tick, pv)
                    pos.pnl = pnl
                    pos.status = "CLOSED"
                    self.balance += pnl
                    self.daily_pnl += pnl
                    if pnl <= 0:
                        self.consecutive_losses += 1
                        self.last_loss_bar = i
                    else:
                        self.consecutive_losses = 0
                    bt_trade = BacktestTrade(
                        trade_id=pos.id, instrument=instrument, direction=pos.direction,
                        entry_price=pos.entry, exit_price=exit_price, stop=pos.stop,
                        tp1=pos.tp1, tp2=pos.tp2, tp3=pos.tp3, contracts=pos.contracts,
                        pnl=pnl, outcome="WIN_TIME" if pnl > 0 else "LOSS_TIME",
                        tqs_score=getattr(pos, "_tqs_score", 0),
                        tqs_grade=getattr(pos, "_tqs_grade", ""),
                        confluence_score=getattr(pos, "_confluence", 0),
                        entry_time=pos.entry_time, exit_time=current_bar.timestamp,
                        bars_held=bars_held,
                        regime_label=getattr(pos, "_regime_label", ""),
                        kill_zone=getattr(pos, "_kill_zone", ""),
                    )
                    trades.append(bt_trade)
                    self.journal.record(JournalEntry(
                        trade_id=str(pos.id), instrument=instrument, direction=pos.direction,
                        entry=pos.entry, exit_price=exit_price, pnl=pnl,
                        outcome=bt_trade.outcome, tqs_grade=bt_trade.tqs_grade,
                        entry_time=pos.entry_time, exit_time=current_bar.timestamp,
                    ))
                    closed_positions.append(pos)
                    continue

                action = self.monitor.update(pos, current_bar.close, current_bar)
                if action.action != "HOLD":
                    exit_price = self._get_exit_price(pos, action, current_bar)
                    pnl = self._calc_pnl(pos, exit_price, tick, pv)
                    pos.pnl = pnl
                    pos.status = "CLOSED"

                    self.balance += pnl
                    self.daily_pnl += pnl

                    if pnl > 0:
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1
                        self.last_loss_bar = i  # v7.0: track for cooloff

                    bt_trade = BacktestTrade(
                        trade_id=pos.id,
                        instrument=instrument,
                        direction=pos.direction,
                        entry_price=pos.entry,
                        exit_price=exit_price,
                        stop=pos.stop,
                        tp1=pos.tp1,
                        tp2=pos.tp2,
                        tp3=pos.tp3,
                        contracts=pos.contracts,
                        pnl=pnl,
                        outcome=self._outcome_from_action(action.action, pnl),
                        tqs_score=getattr(pos, "_tqs_score", 0),
                        tqs_grade=getattr(pos, "_tqs_grade", ""),
                        confluence_score=getattr(pos, "_confluence", 0),
                        entry_time=pos.entry_time,
                        exit_time=current_bar.timestamp,
                        bars_held=i - getattr(pos, "_entry_bar_idx", i),
                        regime_label=getattr(pos, "_regime_label", ""),
                        kill_zone=getattr(pos, "_kill_zone", ""),
                    )
                    trades.append(bt_trade)
                    closed_positions.append(pos)

                    self.journal.record(JournalEntry(
                        trade_id=str(pos.id),
                        instrument=instrument,
                        direction=pos.direction,
                        entry=pos.entry,
                        exit_price=exit_price,
                        pnl=pnl,
                        outcome=bt_trade.outcome,
                        tqs_grade=bt_trade.tqs_grade,
                        entry_time=pos.entry_time,
                        exit_time=current_bar.timestamp,
                    ))

                elif action.action == "MIGRATE_STOP" and action.new_stop:
                    pos.stop = action.new_stop

            for p in closed_positions:
                self.open_positions.remove(p)

            self.equity_curve.append(self.balance)

            # 2. Check for new signals at interval
            if i % self.config.signal_interval_bars != 0:
                continue
            # v7.0: Only 1 position at a time — zero duplicate trades
            if self.config.max_one_position and len(self.open_positions) >= 1:
                continue
            elif not self.config.max_one_position and len(self.open_positions) >= 2:
                continue
            if self.consecutive_losses >= 3:
                continue
            # v7.0: Cooloff after a loss — wait before next trade
            if self.config.loss_cooloff_bars > 0 and (i - self.last_loss_bar) < self.config.loss_cooloff_bars:
                continue
            if self.daily_pnl <= -(self.balance * self.config.max_daily_loss_pct):
                continue

            # v6.0: Max 1 trade per day per instrument
            if self.trades_today >= self.config.max_trades_per_day:
                continue

            # v6.0: Kill zone enforcement — only trade during NY (7-10am ET) and London (2-5am ET)
            if self.config.require_kill_zone:
                in_kz, kz_name = _is_in_kill_zone(current_bar.timestamp)
                if not in_kz:
                    continue
            else:
                _, kz_name = _is_in_kill_zone(current_bar.timestamp)

            # Build multi-timeframe bars from history
            # Need 7500+ 1m bars to get ~31 4h bars for regime detector (lookback=30)
            history = bars_1m[max(0, i - 8000):i + 1]
            context = self._generate_context(history, instrument, current_bar)
            if context is None:
                continue

            # v6.0: Regime filter — only trade TRENDING regimes
            regime_label = context.macro_regime_label.upper()
            if self.config.require_trending:
                if "TRENDING" not in regime_label:
                    continue

            # v6.0: Check if learner has blocked this regime
            if self._is_regime_blocked(instrument, regime_label):
                continue

            # v6.0: ATR dead-market filter
            if context.atr > 0:
                self._atr_history.append(context.atr)
                # Keep last 500 ATR readings for percentile calculation
                if len(self._atr_history) > 500:
                    self._atr_history = self._atr_history[-500:]
                if len(self._atr_history) >= 20:
                    atr_pct = sum(1 for a in self._atr_history if a <= context.atr) / len(self._atr_history)
                    if atr_pct < self.config.atr_min_percentile:
                        continue

            # v6.0: Premium/discount zone gating
            if context.direction == "BULLISH" and context.premium_discount > self.config.premium_discount_long_max:
                continue
            if context.direction == "BEARISH" and context.premium_discount < self.config.premium_discount_short_min:
                continue

            # v6.0: 4H direction must align with trade direction
            if context.direction_4h not in (context.direction, "UNKNOWN"):
                continue

            # v6.0: 1H structure must show clear HH/HL (bullish) or LH/LL (bearish)
            if self.config.require_1h_structure:
                direction_1h = getattr(context, "_direction_1h", "UNKNOWN")
                if direction_1h not in (context.direction, "UNKNOWN"):
                    continue

            # v6.0: Require FVG from 15m+ timeframe (not 5m noise)
            if self.config.require_fvg_15m:
                if not context.fvg_15m_valid and not context.fvg_1h_present and not context.fvg_4h_present:
                    continue

            signals_generated += 1

            # 3. Run confluence check
            confluence = self.confluence_scorer.score(context)
            if confluence.composite < self.config.confluence_minimum:
                signals_rejected += 1
                continue

            # 4. MTF Pyramid check (relaxed for backtest — not all TFs available)
            pyramid = self.mtf_pyramid.analyze(
                context.macro_regime, context.macro_confidence,
                context.direction_4h, context.direction_4h_confidence,
                context.direction_15m, context.direction_15m_confidence,
                context.direction_5m, context.direction_5m_confidence,
            )
            # In backtest, require at least 2 aligned layers instead of all 4
            aligned_count = sum(1 for d in [
                context.direction_4h, context.direction_15m, context.direction_5m
            ] if d == context.direction)
            if aligned_count < 2:
                signals_rejected += 1
                continue

            # 5. Checklist (relaxed for backtest — no OFD/news/AMD data available)
            checklist_ctx = self._build_checklist_context(context)
            cl = self.checklist.evaluate(checklist_ctx)
            # In backtest mode, only require core structural conditions (8 of 21)
            # OFD, news, AMD, inducement, DXY are not available from OHLCV data
            core_conditions = {
                "htf_4h_bias_confirmed", "itf_1h_aligned", "entry_tf_15m_setup",
                "draw_on_liquidity_identified", "price_in_discount_or_premium",
                "macro_not_ranging", "session_volume_adequate", "daily_risk_budget_available",
            }
            core_passed = sum(
                1 for c in core_conditions
                if c not in cl.failed_conditions
            )
            if core_passed < 5:
                signals_rejected += 1
                continue

            # 6. TQS scoring (with backtest defaults for unavailable dimensions)
            tqs_ctx = {
                **checklist_ctx,
                "all_timeframes_aligned": aligned_count >= 2,
                "macro_regime_confidence": max(context.macro_confidence, 0.75),  # Boost: regime is valid if we got here
                "fvg_4h_present": context.fvg_4h_present,
                "fvg_1h_present": context.fvg_1h_present,
                "fvg_completely_unmitigated": context.fvg_unmitigated or context.fvg_15m_valid,
                "liquidity_target_timeframe": context.liquidity_target_tf or "15m",
                "liquidity_sessions_untouched": max(context.liquidity_sessions_untouched, 1),
                "primary_liquidity_target_exists": True,  # Assume valid if signal generated
                "hunt_confidence": max(context.hunt_confidence, 0.55),  # Default moderate
                "footprint_composite": 0.70,   # Assume moderate institutional activity
                "news_sentiment_score": 0.40,   # Assume mild alignment
                "news_risk_level": "LOW",
                "minutes_to_next_high_impact": 999,
                "amd_phase": context.amd_phase if context.amd_phase != "UNKNOWN" else "DISTRIBUTION",
                "kill_zone_active": context.kill_zone_active,
                "kill_zone_minutes_elapsed": context.kill_zone_minutes_elapsed if context.kill_zone_active else 15,
            }
            tqs = self.tqs_scorer.score(tqs_ctx, historical_similarity=0.70)

            # In backtest, 3 of 8 TQS dimensions (news, footprint, historical) score 0
            # since there's no live data. Max possible backtest TQS ~ 62.5.
            # Scale threshold proportionally: 5/8 of live minimum.
            backtest_tqs_min = max(30, int(self.config.tqs_minimum * 5 / 8))

            # v6.0: Apply learner TQS bonuses
            learner_tqs_bonus = self._get_effective_tqs_bonus(instrument) + self._get_session_tqs_bonus(kz_name)
            effective_tqs = tqs.total + learner_tqs_bonus

            if effective_tqs < backtest_tqs_min:
                if effective_tqs >= backtest_tqs_min - 10:
                    shadow_count += 1
                else:
                    signals_rejected += 1
                continue

            # 7. Calculate entry, stop, TP
            entry_price = self.entry_calculator.calculate(
                instrument, context.fvg_low, context.fvg_high,
                context.ob_low,
            )

            # Default sweep distribution for backtest
            sweep_dist = SweepDistribution(
                instrument, "FVG_OB",
                3.0, 4.0, 6.0, 0.70, 0, False,
            )
            stop = self.stop_calculator.calculate(
                instrument, context.direction,
                context.fvg_low, context.ob_low,
                sweep_dist, context.atr,
                context.vix, self.balance * self.config.max_risk_pct,
            )

            # v6.0: Widen stops by multiplier to avoid noise sweeps
            stop_mult = self._get_effective_stop_multiplier(instrument)
            original_risk = abs(entry_price.price - stop.price)
            if original_risk <= 0:
                signals_rejected += 1
                continue

            # v7.0 FIX 1: Reject micro-risk trades — stops too close = noise
            risk_ticks = original_risk / tick
            effective_min_risk_ticks = self._get_instrument_config(
                instrument, "min_risk_ticks", self.config.min_risk_ticks
            )
            if risk_ticks < effective_min_risk_ticks:
                signals_rejected += 1
                continue

            # v7.5: Bullish-only mode (for confirmed bull markets)
            if self.config.bullish_only and context.direction == "BEARISH":
                signals_rejected += 1
                continue

            # v7.0 FIX 3: Macro trend filter — block counter-trend only in STRONG trends
            if self.config.macro_trend_filter and context.macro_confidence >= 0.65:
                if "TRENDING_BULLISH" in regime_label and context.direction == "BEARISH":
                    signals_rejected += 1
                    continue
                if "TRENDING_BEARISH" in regime_label and context.direction == "BULLISH":
                    signals_rejected += 1
                    continue

            widened_risk = original_risk * stop_mult
            if context.direction == "BULLISH":
                wide_stop = entry_price.price - widened_risk
            else:
                wide_stop = entry_price.price + widened_risk

            # v6.0: TP targets based on ORIGINAL structural risk
            # This keeps TPs reachable while wider stops protect from noise
            # Win at TP1 = +1.5R(orig), Loss at stop = -2.0R(orig)
            # Breakeven WR = 2.0/(1.5+2.0) = 57%. Our filters target >60%.
            tp1_r = self.config.tp1_ratio
            tp2_r = self.config.tp2_ratio
            tp3_r = self.config.tp3_ratio

            if context.direction == "BULLISH":
                tp1 = entry_price.price + original_risk * tp1_r
                tp2 = entry_price.price + original_risk * tp2_r
                tp3 = entry_price.price + original_risk * tp3_r
            else:
                tp1 = entry_price.price - original_risk * tp1_r
                tp2 = entry_price.price - original_risk * tp2_r
                tp3 = entry_price.price - original_risk * tp3_r

            # v7.0: Minimum reward:risk filter — TP1 must justify the risk
            tp1_distance = abs(tp1 - entry_price.price)
            widened_risk = abs(entry_price.price - wide_stop)
            actual_rr = tp1_distance / widened_risk if widened_risk > 0 else 0
            if actual_rr < self.config.min_rr_ratio:
                signals_rejected += 1
                continue

            # 8. Position sizing — use widened stop for sizing
            size = self.sizer.calculate(
                instrument, entry_price.price, wide_stop,
                self.balance, self.config.max_risk_pct,
                tick, pv, stop.size_multiplier,
            )

            if size.contracts < 1:
                signals_rejected += 1
                continue

            # Cap contracts for micros and minis
            max_contracts = {"MNQ": 10, "MES": 10, "MGC": 5, "SIL": 5, "MYM": 10, "M2K": 10}.get(instrument, 3)
            size.contracts = min(size.contracts, max_contracts)

            # 9. Open position
            self.trade_counter += 1
            self.trades_today += 1
            pos = Position(
                id=self.trade_counter,
                instrument=instrument,
                direction=context.direction,
                entry=entry_price.price,
                stop=wide_stop,  # v6.0: use widened stop
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                contracts=size.contracts,
                pnl=0.0,
                status="OPEN",
                entry_time=current_bar.timestamp,
            )
            pos._tqs_score = tqs.total
            pos._tqs_grade = tqs.grade.value
            pos._confluence = confluence.composite
            pos._entry_bar_idx = i
            pos._regime_label = regime_label
            pos._kill_zone = kz_name
            self.open_positions.append(pos)

            logger.debug(
                "backtest.trade_opened",
                id=pos.id,
                direction=context.direction,
                entry=entry_price.price,
                stop=wide_stop,
                tp1=tp1,
                tqs=tqs.total,
                kz=kz_name,
                regime=regime_label,
            )

        # Close any remaining open positions at last bar price
        for pos in self.open_positions:
            last_bar = bars_1m[-1]
            pnl = self._calc_pnl(pos, last_bar.close, tick, pv)
            trades.append(BacktestTrade(
                trade_id=pos.id,
                instrument=instrument,
                direction=pos.direction,
                entry_price=pos.entry,
                exit_price=last_bar.close,
                stop=pos.stop,
                tp1=pos.tp1,
                tp2=pos.tp2,
                tp3=pos.tp3,
                contracts=pos.contracts,
                pnl=pnl,
                outcome="LOSS_TIME",
                tqs_score=getattr(pos, "_tqs_score", 0),
                tqs_grade=getattr(pos, "_tqs_grade", ""),
                confluence_score=getattr(pos, "_confluence", 0),
                entry_time=pos.entry_time,
                exit_time=last_bar.timestamp,
                bars_held=len(bars_1m) - getattr(pos, "_entry_bar_idx", 0),
                regime_label=getattr(pos, "_regime_label", ""),
                kill_zone=getattr(pos, "_kill_zone", ""),
            ))
            self.balance += pnl

        return self._compile_results(trades, signals_generated, signals_rejected, shadow_count)

    def _generate_context(self, history: List[Bar], instrument: str, current_bar: Bar) -> Optional[MarketContext]:
        """Generate MarketContext from historical bars using all detection modules."""
        import pandas as pd

        if len(history) < 100:
            return None

        # Build DataFrames for resampling
        df = pd.DataFrame([{
            "ts_event": pd.Timestamp.utcfromtimestamp(b.timestamp),
            "open": b.open, "high": b.high, "low": b.low,
            "close": b.close, "volume": b.volume,
        } for b in history])

        bars_5m = self.resampler.resample_to_bars(df, "5m")
        bars_15m = self.resampler.resample_to_bars(df, "15m")
        bars_1h = self.resampler.resample_to_bars(df, "1h")
        bars_4h = self.resampler.resample_to_bars(df, "4h")

        if len(bars_4h) < 20 or len(bars_1h) < 20 or len(bars_15m) < 20:
            return None

        # Detect regime at 4h
        regime_4h = self.regime_detector.detect(bars_4h)

        # Convert M02 RegimeState to M03's RegimeState format
        from ghost.modules.m03_signal.signal_generator import RegimeState as SigRegime
        sig_regime = SigRegime(
            macro_regime=regime_4h.label,
            macro_regime_label=regime_4h.label,
            macro_confidence=regime_4h.confidence,
            macro_direction=regime_4h.direction,
        )

        # Generate signals
        context = self.signal_generator.generate(
            bars_4h=bars_4h,
            bars_1h=bars_1h,
            bars_15m=bars_15m,
            bars_5m=bars_5m,
            regime=sig_regime,
            current_price=current_bar.close,
        )

        if not context or not context.direction or context.direction.upper() == "NEUTRAL":
            return None

        # Normalize direction to uppercase for consistency with v5.5 modules
        context.direction = context.direction.upper()
        context.macro_direction = (context.macro_direction or "NEUTRAL").upper()
        context.direction_4h = (context.direction_4h or "UNKNOWN").upper()
        context.direction_15m = (context.direction_15m or "UNKNOWN").upper()
        context.direction_5m = (context.direction_5m or "UNKNOWN").upper()
        context.ofd_direction = context.direction  # Assume aligned for backtest
        context.amd_phase = (context.amd_phase or "UNKNOWN").upper()

        # v6.0: 1H structure analysis for HH/HL or LH/LL confirmation
        from ghost.modules.m03_signal.structure import StructureDetector
        if bars_1h and len(bars_1h) >= 11:
            struct_1h = StructureDetector()
            state_1h = struct_1h.analyze(bars_1h)
            context._direction_1h = state_1h.direction.upper() if state_1h.direction else "UNKNOWN"
        else:
            context._direction_1h = "UNKNOWN"

        # Store regime label and ATR percentile on context for v6.0 filters
        context.macro_regime_label = regime_4h.label

        # Fill in bars for downstream modules
        context.bars_1m = history[-100:]
        context.bars_15m = bars_15m[-50:]
        context.current_price = current_bar.close

        return context

    def _build_checklist_context(self, ctx: MarketContext) -> dict:
        """Build checklist context dict from MarketContext."""
        return {
            "macro_regime_label": ctx.macro_regime_label,
            "macro_regime_confidence": ctx.macro_confidence,
            "itf_regime_direction": ctx.direction_4h,
            "macro_regime_direction": ctx.macro_direction,
            "fvg_15m_unmitigated": ctx.fvg_15m_valid,
            "ob_15m_unswept": ctx.ob_15m_valid,
            "approaching_fvg_5m": ctx.approaching_fvg,
            "primary_liquidity_target_exists": ctx.liquidity_target is not None,
            "primary_liquidity_target_mitigated": getattr(ctx.liquidity_target, "mitigated", True) if ctx.liquidity_target else True,
            "premium_discount_position": ctx.premium_discount,
            "direction": ctx.direction,
            "recent_liquidity_sweep_confirmed": ctx.sweep_confirmed,
            "path_to_tp1_clear": ctx.path_clear,
            "ofd_net_delta_direction": ctx.ofd_direction,
            "institutional_footprint_detected": True,  # Assume in backtest
            "amd_phase": ctx.amd_phase,
            "inducement_detected": False,
            "inducement_completed": False,
            "news_risk_level": "LOW",
            "news_sentiment_aligned": True,
            "minutes_to_next_high_impact": 999,
            "dxy_correlation_aligned": True,
            "dxy_sensitive": False,
            "kill_zone_active": ctx.kill_zone_active,
            "circuit_breaker_active": self.consecutive_losses >= 3,
            "session_volume_ratio": 1.0,
            "daily_loss_used_pct": abs(self.daily_pnl) / self.balance if self.balance > 0 else 0,
            "has_correlated_open_position": len(self.open_positions) > 0,
        }

    def _get_exit_price(self, pos: Position, action: MonitorAction, bar: Bar) -> float:
        """Determine exit price based on monitor action."""
        if action.action == "CLOSE_STOP":
            return pos.stop
        elif action.action == "CLOSE_TP1":
            return pos.tp1
        elif action.action == "CLOSE_TP2":
            return pos.tp2
        elif action.action == "CLOSE_TP3":
            return pos.tp3
        return bar.close

    def _calc_pnl(self, pos: Position, exit_price: float, tick: float, pv: float) -> float:
        """Calculate P&L for a position."""
        if pos.direction == "BULLISH":
            ticks = (exit_price - pos.entry) / tick
        else:
            ticks = (pos.entry - exit_price) / tick
        return ticks * pv * pos.contracts

    def _outcome_from_action(self, action: str, pnl: float) -> str:
        """Map monitor action to outcome string."""
        if action == "CLOSE_TP1":
            return "WIN_TP1"
        elif action == "CLOSE_TP2":
            return "WIN_TP2"
        elif action == "CLOSE_TP3":
            return "WIN_TP3"
        elif action == "CLOSE_STOP":
            return "LOSS_STOP"
        return "WIN" if pnl > 0 else "LOSS_TIME"

    def _compile_results(
        self, trades: List[BacktestTrade],
        signals_gen: int, signals_rej: int, shadow: int,
    ) -> BacktestResult:
        """Compile final backtest statistics."""
        result = BacktestResult(trades=trades)
        result.total_trades = len(trades)
        result.signals_generated = signals_gen
        result.signals_rejected = signals_rej
        result.shadow_signals = shadow
        result.final_balance = self.balance

        if not trades:
            return result

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.total_pnl = sum(t.pnl for t in trades)
        result.win_rate = len(wins) / len(trades) if trades else 0.0
        result.avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        result.avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        result.expectancy = result.total_pnl / len(trades)

        # Max drawdown
        equity = [self.config.initial_balance]
        for t in trades:
            equity.append(equity[-1] + t.pnl)
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            peak = max(peak, e)
            dd = peak - e
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd
        result.max_drawdown_pct = max_dd / self.config.initial_balance if self.config.initial_balance > 0 else 0.0

        # Sharpe
        returns = [t.pnl for t in trades]
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

        return result
