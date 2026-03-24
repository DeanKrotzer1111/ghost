"""Ghost v5.5 Core Orchestrator — 10-gate signal processing pipeline."""
import structlog
from ghost.core.models import ProcessResult

from ghost.modules.m17_stop_calibration import (
    StopSweepAnalyzer, OptimalStopCalculator, DynamicStopMigrator, StopHuntPredictor,
)
from ghost.modules.m18_tp_calibration import TakeProfitCalibrationEngine, TPOvershootAnalyzer
from ghost.modules.m19_entry_calibration import OptimalEntryCalculator, EntryTimingEngine
from ghost.modules.m20_selectivity import PreSignalChecklist
from ghost.modules.m21_tqs import TradeQualityScorer
from ghost.modules.m22_footprint import SmartMoneyFootprintEngine
from ghost.modules.m23_mtf_pyramid import MTFConfluencePyramid
from ghost.modules.m24_weekly_profile import WeeklyProfileClassifier
from ghost.modules.m25_liquidity_void import LiquidityVoidEngine
from ghost.modules.m27_payout import PayoutOptimizationEngine
from ghost.modules.m28_ensemble import EnsemblePredictionEngine
from ghost.modules.m29_self_calibration import SelfCalibrationLoop

logger = structlog.get_logger()


class GhostOrchestrator:
    """Central orchestrator implementing the v5.5 10-gate signal processing pipeline.

    Gate order:
        1. MTF Pyramid alignment
        2. Weekly profile directional filter
        3. Smart money footprint detection
        4. Liquidity void awareness
        5. 21-condition checklist
        6. Trade Quality Score (TQS) with payout-phase scaling
        7. Optimal stop placement with hunt prediction
        8. Optimal entry calculation
        9. Multi-target TP calibration
        10. Ensemble LLM consensus voting
    """

    def __init__(self, db, settings, instrument_registry, minimax_client, qwen_client):
        self.db = db
        self.settings = settings
        self.instrument_registry = instrument_registry
        self.minimax_client = minimax_client
        self.qwen_client = qwen_client
        self.weekly_profile = None

    def setup(self):
        """Initialize all v5.5 module instances."""
        ts = self.instrument_registry.tick_sizes
        pv = self.instrument_registry.point_values

        self.sweep_analyzer = StopSweepAnalyzer(db=self.db)
        self.tp_overshoot = TPOvershootAnalyzer(db=self.db)
        self.stop_calculator = OptimalStopCalculator(ts, pv)
        self.stop_migrator = DynamicStopMigrator(ts)
        self.hunt_predictor = StopHuntPredictor(ts)
        self.tp_engine = TakeProfitCalibrationEngine(self.tp_overshoot, ts)
        self.entry_calculator = OptimalEntryCalculator(ts)
        self.entry_timing = EntryTimingEngine()
        self.checklist = PreSignalChecklist()
        self.tqs = TradeQualityScorer()
        self.footprint_engine = SmartMoneyFootprintEngine(ts)
        self.mtf_pyramid = MTFConfluencePyramid()
        self.weekly_classifier = WeeklyProfileClassifier()
        self.void_engine = LiquidityVoidEngine(ts)
        self.payout_engine = PayoutOptimizationEngine()
        self.ensemble_engine = EnsemblePredictionEngine(
            minimax_client=self.minimax_client,
            qwen_client=self.qwen_client,
        )
        self.calibration_loop = SelfCalibrationLoop(
            db=self.db,
            sweep_analyzer=self.sweep_analyzer,
            tp_analyzer=self.tp_overshoot,
            config=self.settings,
            telegram=getattr(self, "telegram", None),
        )

        logger.info("ghost.orchestrator.v55.ready", modules=12)

    async def process_signal_v55(self, raw_signal, account_state) -> ProcessResult:
        """Process a trading signal through all 10 gates."""
        inst = raw_signal.instrument
        d = raw_signal.direction

        # GATE 1: MTF PYRAMID
        pyramid = self.mtf_pyramid.analyze(
            raw_signal.macro_regime, raw_signal.macro_confidence,
            raw_signal.direction_4h, raw_signal.direction_4h_confidence,
            raw_signal.direction_15m, raw_signal.direction_15m_confidence,
            raw_signal.direction_5m, raw_signal.direction_5m_confidence,
        )
        if not pyramid.aligned:
            return ProcessResult(False, "MTF_PYRAMID_INCOMPLETE", pyramid=pyramid)

        # GATE 2: WEEKLY PROFILE
        if self.weekly_profile and self.settings.weekly_profile_enabled:
            if self.weekly_profile.bias != "NEUTRAL" and self.weekly_profile.bias != d:
                return ProcessResult(False, "WEEKLY_PROFILE_MISALIGNED")

        # GATE 3: FOOTPRINT
        fp = self.footprint_engine.analyze(
            inst, d,
            getattr(raw_signal, "tick_buffer", []) or [],
            getattr(raw_signal, "bars_1m", []) or [],
            getattr(raw_signal, "ofd_history", []) or [],
            getattr(raw_signal, "liquidity_pools", []) or [],
            raw_signal.current_price, raw_signal.fvg_low, raw_signal.fvg_high,
            getattr(raw_signal, "bid_size", 0.0) or 0.0,
            getattr(raw_signal, "ask_size", 0.0) or 0.0,
        )

        # GATE 4: VOID
        voids = self.void_engine.detect(
            inst, getattr(raw_signal, "bars_15m", []) or [], raw_signal.atr,
        )
        void_present = False
        if getattr(raw_signal, "tp1_estimate", None):
            void_present = self.void_engine.void_between_entry_and_tp(
                raw_signal.current_price, raw_signal.tp1_estimate, voids, d,
            ) is not None

        # GATE 5: CHECKLIST
        ctx = {
            "macro_regime_label": raw_signal.macro_regime,
            "macro_regime_confidence": raw_signal.macro_confidence,
            "itf_regime_direction": raw_signal.direction_4h,
            "macro_regime_direction": raw_signal.macro_direction,
            "fvg_15m_unmitigated": getattr(raw_signal, "fvg_15m_valid", False),
            "ob_15m_unswept": getattr(raw_signal, "ob_15m_valid", False),
            "approaching_fvg_5m": getattr(raw_signal, "approaching_fvg", False),
            "primary_liquidity_target_exists": getattr(raw_signal, "liquidity_target", None) is not None,
            "primary_liquidity_target_mitigated": getattr(
                getattr(raw_signal, "liquidity_target", None), "mitigated", True,
            ),
            "premium_discount_position": getattr(raw_signal, "premium_discount", 0.5),
            "direction": d,
            "recent_liquidity_sweep_confirmed": getattr(raw_signal, "sweep_confirmed", False),
            "path_to_tp1_clear": getattr(raw_signal, "path_clear", False),
            "ofd_net_delta_direction": getattr(raw_signal, "ofd_direction", ""),
            "institutional_footprint_detected": fp.signals_detected >= 1,
            "amd_phase": getattr(raw_signal, "amd_phase", "UNKNOWN"),
            "inducement_detected": getattr(raw_signal, "inducement_detected", False),
            "inducement_completed": getattr(raw_signal, "inducement_completed", False),
            "news_risk_level": getattr(raw_signal, "news_risk_level", "HIGH"),
            "news_sentiment_aligned": getattr(raw_signal, "news_aligned", False),
            "minutes_to_next_high_impact": getattr(raw_signal, "minutes_to_event", 0),
            "dxy_correlation_aligned": getattr(raw_signal, "dxy_aligned", True),
            "dxy_sensitive": getattr(raw_signal, "dxy_sensitive", False),
            "kill_zone_active": getattr(raw_signal, "kill_zone_active", False),
            "circuit_breaker_active": getattr(raw_signal, "circuit_breaker", False),
            "session_volume_ratio": getattr(raw_signal, "volume_ratio", 1.0),
            "daily_loss_used_pct": account_state.daily_loss_used_pct,
            "has_correlated_open_position": account_state.has_correlated_position,
        }
        cl = self.checklist.evaluate(ctx)
        if not cl.passed:
            return ProcessResult(
                False, f"CHECKLIST:{cl.failed_conditions[0]}",
                checklist=cl, pyramid=pyramid,
            )

        # GATE 6: TQS
        tqs_ctx = {
            **ctx,
            "all_timeframes_aligned": pyramid.aligned,
            "fvg_4h_present": getattr(raw_signal, "fvg_4h_present", False),
            "fvg_1h_present": getattr(raw_signal, "fvg_1h_present", False),
            "fvg_completely_unmitigated": getattr(raw_signal, "fvg_unmitigated", False),
            "liquidity_target_timeframe": getattr(raw_signal, "liquidity_target_tf", "15m"),
            "liquidity_sessions_untouched": getattr(raw_signal, "liquidity_sessions_untouched", 0),
            "hunt_confidence": getattr(raw_signal, "hunt_confidence", 0.0),
            "footprint_composite": fp.composite,
            "news_sentiment_score": getattr(raw_signal, "news_sentiment_score", 0.0),
            "kill_zone_minutes_elapsed": getattr(raw_signal, "kill_zone_minutes_elapsed", 999),
        }
        ps = self.payout_engine.get_state(
            account_state.current_pnl, self.settings.payout_target_dollars,
            account_state.avg_expectancy_20, account_state.avg_trades_per_day,
        )
        etm = max(self.settings.tqs_minimum_execute, ps.tqs_minimum)
        tq = self.tqs.score(tqs_ctx)
        if not tq.execute or tq.total < etm:
            return ProcessResult(
                False, f"TQS_{tq.total:.0f}_BELOW_{etm}",
                tqs=tq, checklist=cl, pyramid=pyramid,
            )

        # GATE 7: STOP
        sd = await self.sweep_analyzer.compute(inst)
        hp = self.hunt_predictor.predict(
            inst, raw_signal.current_price,
            getattr(raw_signal, "liquidity_pools", []) or [],
            getattr(raw_signal, "ofd_net_delta", 0.0),
            getattr(raw_signal, "kill_zone_active", False),
            getattr(raw_signal, "kill_zone_first_15min", False),
        )
        os = self.stop_calculator.calculate(
            inst, d, raw_signal.fvg_low,
            getattr(raw_signal, "ob_low", None), sd, raw_signal.atr,
            getattr(raw_signal, "vix", 18.0) or 18.0,
            account_state.max_risk_dollars,
        )
        os.hunt_predicted = hp.detected

        # GATE 8: ENTRY
        oe = self.entry_calculator.calculate(
            inst, raw_signal.fvg_low, raw_signal.fvg_high,
            getattr(raw_signal, "ob_low", None),
            getattr(raw_signal, "tick_buffer", None),
            getattr(raw_signal, "bid_prices", None),
        )

        # GATE 9: TP
        tp = await self.tp_engine.calculate(
            inst, d, oe.price, os.price,
            getattr(raw_signal, "liquidity_pools", []) or [],
            getattr(raw_signal, "htf_draws", []) or [],
            getattr(raw_signal, "hour_et", 9), void_present,
        )

        # GATE 10: ENSEMBLE
        ens = await self.ensemble_engine.run(
            self._build_ensemble_prompt(raw_signal, oe, os, tp, tq, fp),
            signal_id=getattr(raw_signal, "signal_id", ""),
        )
        if not ens.execute:
            return ProcessResult(
                False, f"ENSEMBLE_{ens.consensus.value}",
                ensemble=ens, tqs=tq, checklist=cl, pyramid=pyramid,
            )

        return ProcessResult(
            True,
            entry=oe.price, stop=os.price,
            tp1=tp.tp1, tp2=tp.tp2, tp3=tp.tp3,
            tqs=tq, checklist=cl, footprint=fp,
            pyramid=pyramid, ensemble=ens, payout_state=ps,
            void_present=void_present, hunt_predicted=hp.detected,
            size_multiplier=os.size_multiplier * ps.risk_multiplier,
        )

    def _build_ensemble_prompt(self, signal, entry, stop, tp, tqs, footprint) -> str:
        return (
            f"GHOST SIGNAL EVALUATION\n"
            f"Instrument: {signal.instrument} | Direction: {signal.direction}\n"
            f"Entry: {entry.price} ({', '.join(entry.sources)})\n"
            f"Stop: {stop.price} (structural: {stop.structural_level})\n"
            f"TP1: {tp.tp1} | TP2: {tp.tp2} | TP3: {tp.tp3}\n"
            f"TQS: {tqs.total}/100 ({tqs.grade.value}) | Weakest: {tqs.weakest_dimension}\n"
            f"Footprint: {footprint.composite:.2f} ({footprint.dominant_signal})\n"
            f"Regime: {signal.macro_regime} | ATR: {signal.atr}\n"
            f"Should this trade be executed? Respond with approved=true/false, "
            f"entry price, and stop_loss price."
        )
