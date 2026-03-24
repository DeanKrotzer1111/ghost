"""M09 Trade Journal — Records trades and computes performance statistics."""
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class JournalEntry:
    """A single completed trade record."""
    trade_id: str = ""
    instrument: str = ""
    direction: str = ""
    entry: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    outcome: str = ""           # WIN / LOSS / BREAKEVEN
    tqs_grade: str = ""         # e.g., A+, A, B, C, D
    entry_time: float = 0.0
    exit_time: float = 0.0
    tags: list = field(default_factory=list)
    notes: str = ""


@dataclass
class JournalStats:
    """Aggregate performance statistics."""
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    trade_count: int = 0


class TradeJournal:
    """Records trade entries and computes performance statistics."""

    def __init__(self):
        self.entries: list[JournalEntry] = []

    def record(self, entry: JournalEntry) -> None:
        """Store a completed trade in the journal."""
        # Auto-classify outcome if not set
        if not entry.outcome:
            if entry.pnl > 0:
                entry.outcome = "WIN"
            elif entry.pnl < 0:
                entry.outcome = "LOSS"
            else:
                entry.outcome = "BREAKEVEN"

        self.entries.append(entry)
        logger.info(
            "Journal: %s %s %s PnL=%.2f (%s)",
            entry.trade_id, entry.instrument, entry.direction,
            entry.pnl, entry.outcome,
        )

    def get_stats(self) -> JournalStats:
        """Compute aggregate statistics from all recorded trades."""
        if not self.entries:
            return JournalStats()

        trade_count = len(self.entries)
        wins = [e for e in self.entries if e.pnl > 0]
        losses = [e for e in self.entries if e.pnl < 0]

        total_pnl = sum(e.pnl for e in self.entries)
        win_count = len(wins)
        win_rate = win_count / trade_count if trade_count > 0 else 0.0

        avg_win = sum(e.pnl for e in wins) / len(wins) if wins else 0.0
        avg_loss = sum(e.pnl for e in losses) / len(losses) if losses else 0.0

        gross_profit = sum(e.pnl for e in wins)
        gross_loss = abs(sum(e.pnl for e in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        # Expectancy: (win_rate * avg_win) + (loss_rate * avg_loss)
        # avg_loss is already negative
        loss_rate = 1.0 - win_rate
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        # Max drawdown: largest peak-to-trough decline in cumulative PnL
        max_drawdown = self._calculate_max_drawdown()

        return JournalStats(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            total_pnl=total_pnl,
            trade_count=trade_count,
        )

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from the sequence of trades."""
        if not self.entries:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for entry in self.entries:
            cumulative += entry.pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def get_entries_by_instrument(self, instrument: str) -> list[JournalEntry]:
        """Filter journal entries by instrument."""
        return [e for e in self.entries if e.instrument == instrument]

    def get_entries_by_tag(self, tag: str) -> list[JournalEntry]:
        """Filter journal entries by tag."""
        return [e for e in self.entries if tag in e.tags]

    def reset(self) -> None:
        """Clear all journal entries."""
        self.entries.clear()
