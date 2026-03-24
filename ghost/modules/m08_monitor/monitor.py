"""M08 Position Monitor — Tracks open positions and checks stop/TP levels."""
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from ghost.modules.m01_data.models import Bar

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open or closed position."""
    id: str = ""
    instrument: str = ""
    direction: str = ""          # LONG or SHORT
    entry: float = 0.0
    stop: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    contracts: int = 1
    pnl: float = 0.0
    status: str = "OPEN"         # OPEN / CLOSED
    entry_time: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]


@dataclass
class MonitorAction:
    """Action recommended by the monitor after evaluating a position."""
    action: str = "HOLD"         # HOLD / CLOSE_STOP / CLOSE_TP1 / CLOSE_TP2 / CLOSE_TP3 / MIGRATE_STOP
    new_stop: Optional[float] = None
    exit_price: float = 0.0
    pnl: float = 0.0


class PositionMonitor:
    """Monitors open positions against current price, checking stop loss and
    take-profit levels. For LONG positions, stop is below entry and TPs above.
    For SHORT positions, stop is above entry and TPs below."""

    def __init__(self, trail_after_tp1: bool = False):
        self.trail_after_tp1 = trail_after_tp1
        self._tp1_hit: dict[str, bool] = {}
        self._tp2_hit: dict[str, bool] = {}

    def update(self, position: Position, current_price: float, current_bar: Bar) -> MonitorAction:
        """Check if stop or any TP level has been hit on the current bar.
        Uses bar high/low to detect intra-bar hits."""
        if position.status != "OPEN":
            return MonitorAction(action="HOLD")

        bar_high = current_bar.high
        bar_low = current_bar.low

        if position.direction in ("LONG", "BULLISH"):
            return self._check_long(position, current_price, bar_high, bar_low)
        elif position.direction in ("SHORT", "BEARISH"):
            return self._check_short(position, current_price, bar_high, bar_low)

        return MonitorAction(action="HOLD")

    def _check_long(self, pos: Position, price: float, bar_high: float, bar_low: float) -> MonitorAction:
        """Check stop/TP for a LONG position."""
        # Stop loss hit (bar low touches or crosses stop)
        if bar_low <= pos.stop:
            pnl = (pos.stop - pos.entry) * pos.contracts
            logger.info("Position %s STOP HIT at %.2f, PnL=%.2f", pos.id, pos.stop, pnl)
            self._cleanup(pos.id)
            return MonitorAction(action="CLOSE_STOP", exit_price=pos.stop, pnl=pnl)

        # TP3 hit
        if pos.tp3 > 0 and bar_high >= pos.tp3:
            pnl = (pos.tp3 - pos.entry) * pos.contracts
            logger.info("Position %s TP3 HIT at %.2f, PnL=%.2f", pos.id, pos.tp3, pnl)
            self._cleanup(pos.id)
            return MonitorAction(action="CLOSE_TP3", exit_price=pos.tp3, pnl=pnl)

        # TP2 hit
        if pos.tp2 > 0 and bar_high >= pos.tp2 and not self._tp2_hit.get(pos.id, False):
            self._tp2_hit[pos.id] = True
            pnl = (pos.tp2 - pos.entry) * pos.contracts
            logger.info("Position %s TP2 HIT at %.2f, PnL=%.2f", pos.id, pos.tp2, pnl)
            if self.trail_after_tp1:
                new_stop = pos.tp1 if pos.tp1 > 0 else pos.entry
                return MonitorAction(action="CLOSE_TP2", exit_price=pos.tp2, pnl=pnl, new_stop=new_stop)
            return MonitorAction(action="CLOSE_TP2", exit_price=pos.tp2, pnl=pnl)

        # TP1 hit
        if pos.tp1 > 0 and bar_high >= pos.tp1 and not self._tp1_hit.get(pos.id, False):
            self._tp1_hit[pos.id] = True
            pnl = (pos.tp1 - pos.entry) * pos.contracts
            logger.info("Position %s TP1 HIT at %.2f, PnL=%.2f", pos.id, pos.tp1, pnl)
            if self.trail_after_tp1:
                new_stop = pos.entry  # Move stop to breakeven
                return MonitorAction(action="MIGRATE_STOP", exit_price=pos.tp1, pnl=pnl, new_stop=new_stop)
            return MonitorAction(action="CLOSE_TP1", exit_price=pos.tp1, pnl=pnl)

        # Unrealized PnL
        pos.pnl = (price - pos.entry) * pos.contracts
        return MonitorAction(action="HOLD")

    def _check_short(self, pos: Position, price: float, bar_high: float, bar_low: float) -> MonitorAction:
        """Check stop/TP for a SHORT position."""
        # Stop loss hit (bar high touches or crosses stop)
        if bar_high >= pos.stop:
            pnl = (pos.entry - pos.stop) * pos.contracts
            logger.info("Position %s STOP HIT at %.2f, PnL=%.2f", pos.id, pos.stop, pnl)
            self._cleanup(pos.id)
            return MonitorAction(action="CLOSE_STOP", exit_price=pos.stop, pnl=pnl)

        # TP3 hit
        if pos.tp3 > 0 and bar_low <= pos.tp3:
            pnl = (pos.entry - pos.tp3) * pos.contracts
            logger.info("Position %s TP3 HIT at %.2f, PnL=%.2f", pos.id, pos.tp3, pnl)
            self._cleanup(pos.id)
            return MonitorAction(action="CLOSE_TP3", exit_price=pos.tp3, pnl=pnl)

        # TP2 hit
        if pos.tp2 > 0 and bar_low <= pos.tp2 and not self._tp2_hit.get(pos.id, False):
            self._tp2_hit[pos.id] = True
            pnl = (pos.entry - pos.tp2) * pos.contracts
            logger.info("Position %s TP2 HIT at %.2f, PnL=%.2f", pos.id, pos.tp2, pnl)
            if self.trail_after_tp1:
                new_stop = pos.tp1 if pos.tp1 > 0 else pos.entry
                return MonitorAction(action="CLOSE_TP2", exit_price=pos.tp2, pnl=pnl, new_stop=new_stop)
            return MonitorAction(action="CLOSE_TP2", exit_price=pos.tp2, pnl=pnl)

        # TP1 hit
        if pos.tp1 > 0 and bar_low <= pos.tp1 and not self._tp1_hit.get(pos.id, False):
            self._tp1_hit[pos.id] = True
            pnl = (pos.entry - pos.tp1) * pos.contracts
            logger.info("Position %s TP1 HIT at %.2f, PnL=%.2f", pos.id, pos.tp1, pnl)
            if self.trail_after_tp1:
                new_stop = pos.entry  # Move stop to breakeven
                return MonitorAction(action="MIGRATE_STOP", exit_price=pos.tp1, pnl=pnl, new_stop=new_stop)
            return MonitorAction(action="CLOSE_TP1", exit_price=pos.tp1, pnl=pnl)

        # Unrealized PnL
        pos.pnl = (pos.entry - price) * pos.contracts
        return MonitorAction(action="HOLD")

    def _cleanup(self, position_id: str) -> None:
        """Remove tracking state for a closed position."""
        self._tp1_hit.pop(position_id, None)
        self._tp2_hit.pop(position_id, None)

    def reset(self) -> None:
        """Clear all internal state."""
        self._tp1_hit.clear()
        self._tp2_hit.clear()
