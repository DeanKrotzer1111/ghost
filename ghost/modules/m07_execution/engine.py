"""M07 Execution Engine — Simulates order execution for backtesting."""
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Represents a trade order."""
    id: str = ""
    instrument: str = ""
    direction: str = ""          # LONG or SHORT
    entry: float = 0.0
    stop: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    contracts: int = 1
    status: str = "PENDING"      # PENDING / FILLED / CANCELLED
    created_at: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class ExecutionResult:
    """Result of an order execution attempt."""
    filled: bool = False
    fill_price: float = 0.0
    slippage: float = 0.0
    fill_time: float = 0.0


class ExecutionEngine:
    """Handles order execution. In backtesting mode, simulates fills at entry price
    with configurable slippage."""

    def __init__(self, slippage_ticks: float = 0.0, tick_size: float = 0.25):
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.pending_orders: list[Order] = []
        self.filled_orders: list[Order] = []
        self.cancelled_orders: list[Order] = []

    def execute(self, order: Order) -> ExecutionResult:
        """Execute an order. For backtesting, simulates fill at entry price
        with optional slippage applied against the trader's favor."""
        slippage_amount = self.slippage_ticks * self.tick_size

        if order.direction == "LONG":
            fill_price = order.entry + slippage_amount
        elif order.direction == "SHORT":
            fill_price = order.entry - slippage_amount
        else:
            logger.warning("Order %s has unknown direction '%s', cancelling.", order.id, order.direction)
            order.status = "CANCELLED"
            self.cancelled_orders.append(order)
            return ExecutionResult(filled=False, fill_price=0.0, slippage=0.0, fill_time=0.0)

        order.status = "FILLED"
        fill_time = order.created_at
        self.filled_orders.append(order)

        logger.info(
            "Order %s FILLED: %s %s %d @ %.2f (slippage=%.4f)",
            order.id, order.direction, order.instrument, order.contracts,
            fill_price, slippage_amount,
        )

        return ExecutionResult(
            filled=True,
            fill_price=fill_price,
            slippage=slippage_amount,
            fill_time=fill_time,
        )

    def cancel(self, order: Order) -> None:
        """Cancel a pending order."""
        order.status = "CANCELLED"
        self.cancelled_orders.append(order)
        if order in self.pending_orders:
            self.pending_orders.remove(order)
        logger.info("Order %s CANCELLED.", order.id)

    def submit(self, order: Order) -> ExecutionResult:
        """Submit and immediately execute an order (backtest mode)."""
        self.pending_orders.append(order)
        result = self.execute(order)
        if result.filled and order in self.pending_orders:
            self.pending_orders.remove(order)
        return result

    def reset(self) -> None:
        """Clear all order lists."""
        self.pending_orders.clear()
        self.filled_orders.clear()
        self.cancelled_orders.clear()
