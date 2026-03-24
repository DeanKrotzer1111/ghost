"""Optimal Entry Calculator — weighted fusion of FVG midpoint, tick clusters, order book, historical PDF, and OB mitigation."""
import numpy as np
import structlog
from ghost.core.models import OptimalEntry

logger = structlog.get_logger()

W = {
    "fvg_midpoint": 1.0,
    "tick_cluster": 1.8,
    "order_book": 1.6,
    "historical_pdf": 1.4,
    "ob_mitigation": 1.5,
}


class OptimalEntryCalculator:
    """Fuses multiple price signals into an optimal entry using weighted averaging."""

    def __init__(self, tick_sizes: dict):
        self.tick_sizes = tick_sizes

    def calculate(
        self, instrument: str, fvg_low: float, fvg_high: float,
        ob_low: float = None, tick_buffer=None, bid_prices=None,
        historical_win_entries=None,
    ) -> OptimalEntry:
        tick = self.tick_sizes.get(instrument, 0.25)
        mid = (fvg_low + fvg_high) / 2.0
        sources = [("fvg_midpoint", mid, W["fvg_midpoint"])]
        cp = bp = op = None

        if tick_buffer:
            buys = [
                t.price for t in tick_buffer
                if fvg_low <= t.price <= fvg_high and getattr(t, "direction", 0) == 1
            ]
            if len(buys) >= 5:
                try:
                    from statistics import mode
                    cp = mode(buys)
                except Exception:
                    cp = float(np.median(buys))
                sources.append(("tick_cluster", cp, W["tick_cluster"]))

        if bid_prices:
            z = [p for p in bid_prices if fvg_low <= p <= fvg_high]
            if z:
                try:
                    from statistics import mode
                    bp = mode(z)
                    sources.append(("order_book", bp, W["order_book"]))
                except Exception:
                    pass

        if historical_win_entries and len(historical_win_entries) >= 10:
            ze = [e for e in historical_win_entries if fvg_low <= e <= fvg_high]
            if ze:
                sources.append(("historical_pdf", float(np.median(ze)), W["historical_pdf"]))

        if ob_low is not None and fvg_low <= ob_low <= fvg_high:
            op = ob_low + tick
            sources.append(("ob_mitigation", op, W["ob_mitigation"]))

        tw = sum(s[2] for s in sources)
        opt = sum(s[1] * s[2] for s in sources) / tw
        opt = round(opt / tick) * tick
        opt = max(fvg_low, min(fvg_high, opt))

        return OptimalEntry(
            opt, [s[0] for s in sources],
            min(1.0, 0.40 + len(sources) * 0.12), cp, bp, op,
        )
