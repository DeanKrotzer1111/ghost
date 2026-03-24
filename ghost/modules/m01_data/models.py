"""Data models for market data throughout the Ghost system."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Bar:
    """OHLCV bar for any timeframe."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str = ""

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0


@dataclass
class TickData:
    """Single tick-level trade."""
    timestamp: float
    price: float
    volume: int
    direction: int = 0  # 1=buy, -1=sell, 0=unknown
    sell_volume: int = 0


@dataclass
class OFDSnapshot:
    """Order Flow Delta snapshot."""
    timestamp: float
    cumulative_delta: float
    buy_volume: int = 0
    sell_volume: int = 0
    net_delta: float = 0.0


@dataclass
class LiquidityPool:
    """Liquidity concentration (BSL or SSL)."""
    level: float
    direction: str  # "BSL" or "SSL"
    strength: int = 1
    mitigated: bool = False
    formation_time: Optional[float] = None


@dataclass
class MarketContext:
    """Full market context passed between modules."""
    instrument: str
    direction: str = ""
    current_price: float = 0.0
    atr: float = 0.0
    vix: float = 18.0
    hour_et: int = 9

    # Regime
    macro_regime: str = "UNKNOWN"
    macro_regime_label: str = "UNKNOWN"
    macro_confidence: float = 0.5
    macro_direction: str = "UNKNOWN"

    # MTF directions
    direction_4h: str = "UNKNOWN"
    direction_4h_confidence: float = 0.5
    direction_15m: str = "UNKNOWN"
    direction_15m_confidence: float = 0.5
    direction_5m: str = "UNKNOWN"
    direction_5m_confidence: float = 0.5

    # FVG
    fvg_low: float = 0.0
    fvg_high: float = 0.0
    fvg_15m_valid: bool = False
    fvg_4h_present: bool = False
    fvg_1h_present: bool = False
    fvg_unmitigated: bool = False

    # Order blocks
    ob_low: Optional[float] = None
    ob_15m_valid: bool = False

    # Liquidity
    liquidity_pools: list = field(default_factory=list)
    liquidity_target: Optional[LiquidityPool] = None
    liquidity_target_tf: str = "15m"
    liquidity_sessions_untouched: int = 0
    sweep_confirmed: bool = False

    # Structure
    premium_discount: float = 0.5
    path_clear: bool = False
    approaching_fvg: bool = False

    # OFD
    ofd_direction: str = ""
    ofd_net_delta: float = 0.0
    ofd_history: list = field(default_factory=list)

    # AMD / Session
    amd_phase: str = "UNKNOWN"
    kill_zone_active: bool = False
    kill_zone_first_15min: bool = False
    kill_zone_minutes_elapsed: int = 999

    # News
    news_risk_level: str = "LOW"
    news_aligned: bool = True
    news_sentiment_score: float = 0.0
    minutes_to_event: int = 999

    # Misc
    dxy_aligned: bool = True
    dxy_sensitive: bool = False
    circuit_breaker: bool = False
    volume_ratio: float = 1.0
    hunt_confidence: float = 0.0

    # Bars
    tick_buffer: list = field(default_factory=list)
    bars_1m: list = field(default_factory=list)
    bars_15m: list = field(default_factory=list)
    bid_prices: list = field(default_factory=list)
    bid_size: float = 0.0
    ask_size: float = 0.0
    htf_draws: list = field(default_factory=list)
    tp1_estimate: Optional[float] = None
    signal_id: str = ""
