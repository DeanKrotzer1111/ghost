"""Main signal generator — glue module that connects M03 to the orchestrator.

Takes bars at multiple timeframes + regime state, runs all ICT detectors,
and produces a populated MarketContext object for the v5.5 pipeline.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

from ghost.modules.m01_data.models import Bar, MarketContext, LiquidityPool
from ghost.modules.m03_signal.fvg import FVGDetector
from ghost.modules.m03_signal.order_block import OrderBlockDetector
from ghost.modules.m03_signal.liquidity import LiquidityDetector
from ghost.modules.m03_signal.structure import StructureDetector
from ghost.modules.m03_signal.amd import AMDDetector
from ghost.modules.m03_signal.killzone import KillZoneDetector

logger = structlog.get_logger(__name__)


@dataclass
class RegimeState:
    """External regime state passed in from M02 or similar."""
    macro_regime: str = "UNKNOWN"
    macro_regime_label: str = "UNKNOWN"
    macro_confidence: float = 0.5
    macro_direction: str = "UNKNOWN"
    vix: float = 18.0


class SignalGenerator:
    """Orchestrates all M03 ICT signal detectors and produces MarketContext.

    Usage:
        gen = SignalGenerator(instrument="ES")
        ctx = gen.generate(
            bars_4h=[...], bars_1h=[...], bars_15m=[...], bars_5m=[...],
            regime=RegimeState(...),
        )
    """

    def __init__(self, instrument: str = "", use_dst: bool = False):
        self.instrument = instrument

        # Per-timeframe detectors
        self._fvg_detectors: Dict[str, FVGDetector] = {
            "4h": FVGDetector(timeframe="4h"),
            "1h": FVGDetector(timeframe="1h"),
            "15m": FVGDetector(timeframe="15m"),
            "5m": FVGDetector(timeframe="5m"),
        }
        self._ob_detectors: Dict[str, OrderBlockDetector] = {
            "4h": OrderBlockDetector(timeframe="4h"),
            "1h": OrderBlockDetector(timeframe="1h"),
            "15m": OrderBlockDetector(timeframe="15m"),
            "5m": OrderBlockDetector(timeframe="5m"),
        }
        self._structure_detectors: Dict[str, StructureDetector] = {
            "4h": StructureDetector(),
            "1h": StructureDetector(),
            "15m": StructureDetector(),
            "5m": StructureDetector(),
        }
        self._liquidity = LiquidityDetector()
        self._amd = AMDDetector()
        self._kz = KillZoneDetector(use_dst=use_dst)
        self._log = logger.bind(component="SignalGenerator", instrument=instrument)

    def generate(
        self,
        bars_4h: Optional[List[Bar]] = None,
        bars_1h: Optional[List[Bar]] = None,
        bars_15m: Optional[List[Bar]] = None,
        bars_5m: Optional[List[Bar]] = None,
        regime: Optional[RegimeState] = None,
        current_price: Optional[float] = None,
    ) -> MarketContext:
        """Run all detectors and build a MarketContext.

        Args:
            bars_4h: 4-hour bars (optional).
            bars_1h: 1-hour bars (optional).
            bars_15m: 15-minute bars (optional).
            bars_5m: 5-minute bars (optional).
            regime: External regime state.
            current_price: Override for current price (else uses last bar close).

        Returns:
            Fully populated MarketContext.
        """
        bars_4h = bars_4h or []
        bars_1h = bars_1h or []
        bars_15m = bars_15m or []
        bars_5m = bars_5m or []
        regime = regime or RegimeState()

        # Determine current price from the finest-grain bars available
        if current_price is None:
            for bars in (bars_5m, bars_15m, bars_1h, bars_4h):
                if bars:
                    current_price = bars[-1].close
                    break
            else:
                current_price = 0.0

        # Determine timestamp from finest-grain bars
        timestamp = 0.0
        for bars in (bars_5m, bars_15m, bars_1h, bars_4h):
            if bars:
                timestamp = bars[-1].timestamp
                break

        # Compute ATR from 15m bars (primary), fallback to 5m
        atr = 0.0
        atr_bars = bars_15m or bars_5m or bars_1h
        if len(atr_bars) >= 2:
            atr = self._compute_atr(atr_bars)

        ctx = MarketContext(
            instrument=self.instrument,
            current_price=current_price,
            atr=atr,
            vix=regime.vix,
            macro_regime=regime.macro_regime,
            macro_regime_label=regime.macro_regime_label,
            macro_confidence=regime.macro_confidence,
            macro_direction=regime.macro_direction,
        )

        # --- FVG detection ---
        self._run_fvg(ctx, bars_4h, bars_1h, bars_15m, bars_5m, current_price)

        # --- Order Block detection ---
        self._run_ob(ctx, bars_15m, atr, current_price)

        # --- Liquidity detection ---
        self._run_liquidity(ctx, bars_15m, current_price)

        # --- Market Structure ---
        self._run_structure(ctx, bars_4h, bars_15m, bars_5m, current_price)

        # --- AMD phase ---
        self._run_amd(ctx, bars_15m or bars_5m, atr)

        # --- Kill zone ---
        self._run_killzone(ctx, timestamp)

        # --- Store bars on context for downstream ---
        ctx.bars_15m = bars_15m
        if bars_5m:
            ctx.bars_1m = bars_5m  # best granularity available

        self._log.debug("market_context_generated",
                        direction=ctx.direction,
                        amd_phase=ctx.amd_phase,
                        kz_active=ctx.kill_zone_active)
        return ctx

    # ------------------------------------------------------------------
    # Private runner methods
    # ------------------------------------------------------------------

    def _run_fvg(self, ctx: MarketContext, bars_4h: List[Bar],
                 bars_1h: List[Bar], bars_15m: List[Bar],
                 bars_5m: List[Bar], price: float) -> None:
        """Detect FVGs across timeframes and populate context."""
        tf_bars = {
            "4h": bars_4h,
            "1h": bars_1h,
            "15m": bars_15m,
            "5m": bars_5m,
        }

        for tf, bars in tf_bars.items():
            if bars and len(bars) >= 3:
                det = self._fvg_detectors[tf]
                det.detect(bars)

        # 4h FVG presence
        det_4h = self._fvg_detectors["4h"]
        ctx.fvg_4h_present = len(det_4h.active_fvgs) > 0

        # 1h FVG presence
        det_1h = self._fvg_detectors["1h"]
        ctx.fvg_1h_present = len(det_1h.active_fvgs) > 0

        # 15m FVG — use nearest to current price
        det_15m = self._fvg_detectors["15m"]
        nearest_15m = det_15m.get_nearest_fvg(price)
        if nearest_15m:
            ctx.fvg_15m_valid = True
            ctx.fvg_high = nearest_15m.high
            ctx.fvg_low = nearest_15m.low
            ctx.fvg_unmitigated = not nearest_15m.mitigated

            # Check if price is approaching the FVG
            distance = abs(nearest_15m.midpoint - price)
            if ctx.atr > 0 and distance < ctx.atr * 2:
                ctx.approaching_fvg = True

    def _run_ob(self, ctx: MarketContext, bars_15m: List[Bar],
                atr: float, price: float) -> None:
        """Detect order blocks on 15m and populate context."""
        if not bars_15m or len(bars_15m) < 5:
            return

        det = self._ob_detectors["15m"]
        det.detect(bars_15m, atr=atr if atr > 0 else None)

        nearest = det.get_nearest_ob(price)
        if nearest:
            ctx.ob_15m_valid = True
            ctx.ob_low = nearest.low

    def _run_liquidity(self, ctx: MarketContext, bars_15m: List[Bar],
                       price: float) -> None:
        """Detect liquidity pools and populate context."""
        if not bars_15m or len(bars_15m) < 3:
            return

        self._liquidity.detect(bars_15m)
        ctx.liquidity_pools = self._liquidity.active_pools

        # Find nearest target
        nearest = self._liquidity.get_nearest_pool(price)
        if nearest:
            ctx.liquidity_target = nearest
            ctx.liquidity_target_tf = "15m"

        # Check if any pool was just swept
        for pool in self._liquidity._swept_pools[-5:]:
            if pool.mitigated:
                ctx.sweep_confirmed = True
                break

    def _run_structure(self, ctx: MarketContext, bars_4h: List[Bar],
                       bars_15m: List[Bar], bars_5m: List[Bar],
                       price: float) -> None:
        """Run structure analysis and populate context."""
        # 4h structure for higher-timeframe direction
        if bars_4h and len(bars_4h) >= 11:
            state_4h = self._structure_detectors["4h"].analyze(bars_4h)
            ctx.direction_4h = state_4h.direction
            ctx.direction_4h_confidence = 0.7 if state_4h.direction != "UNKNOWN" else 0.3

        # 15m structure
        if bars_15m and len(bars_15m) >= 11:
            state_15m = self._structure_detectors["15m"].analyze(bars_15m)
            ctx.direction_15m = state_15m.direction
            ctx.direction_15m_confidence = 0.7 if state_15m.direction != "UNKNOWN" else 0.3
            ctx.premium_discount = state_15m.premium_discount

        # 5m structure
        if bars_5m and len(bars_5m) >= 11:
            state_5m = self._structure_detectors["5m"].analyze(bars_5m)
            ctx.direction_5m = state_5m.direction
            ctx.direction_5m_confidence = 0.7 if state_5m.direction != "UNKNOWN" else 0.3

        # Determine overall direction from MTF alignment
        directions = []
        for d in (ctx.direction_4h, ctx.direction_15m, ctx.direction_5m):
            if d in ("bullish", "bearish"):
                directions.append(d)

        if directions:
            bullish_count = sum(1 for d in directions if d == "bullish")
            bearish_count = sum(1 for d in directions if d == "bearish")
            if bullish_count > bearish_count:
                ctx.direction = "bullish"
            elif bearish_count > bullish_count:
                ctx.direction = "bearish"
            else:
                ctx.direction = directions[-1]  # tie-break with finest TF

        # Path clear: structure aligns with FVG direction
        if ctx.direction and ctx.fvg_15m_valid:
            ctx.path_clear = True

    def _run_amd(self, ctx: MarketContext, bars: List[Bar],
                 atr: float) -> None:
        """Classify AMD phase and populate context."""
        if not bars or len(bars) < 10:
            return
        state = self._amd.analyze(bars, atr=atr if atr > 0 else None)
        ctx.amd_phase = state.phase

    def _run_killzone(self, ctx: MarketContext, timestamp: float) -> None:
        """Classify kill zone and populate context."""
        if timestamp <= 0:
            return
        kz = self._kz.classify(timestamp)
        ctx.kill_zone_active = kz.active
        ctx.kill_zone_first_15min = kz.first_15min
        ctx.kill_zone_minutes_elapsed = kz.minutes_elapsed
        ctx.hour_et = kz.hour_et

    @staticmethod
    def _compute_atr(bars: List[Bar], period: int = 14) -> float:
        if len(bars) < 2:
            return 0.0
        trs: List[float] = []
        for i in range(1, len(bars)):
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i - 1].close),
                abs(bars[i].low - bars[i - 1].close),
            )
            trs.append(tr)
        n = min(period, len(trs))
        if n == 0:
            return 0.0
        return sum(trs[-n:]) / n

    def reset(self) -> None:
        """Reset all detector state."""
        for det in self._fvg_detectors.values():
            det.reset()
        for det in self._ob_detectors.values():
            det.reset()
        for det in self._structure_detectors.values():
            det.reset()
        self._liquidity.reset()
        self._amd.reset()
