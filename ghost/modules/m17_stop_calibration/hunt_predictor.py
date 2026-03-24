"""Stop Hunt Predictor — detects probable stop hunts using proximity, OFD, kill zones, and round numbers."""
import structlog
from dataclasses import dataclass
from typing import Optional

logger = structlog.get_logger()


@dataclass
class HuntPrediction:
    detected: bool
    target_level: Optional[float] = None
    probability: float = 0.0
    expected_candles: int = 0
    recommended_stop: Optional[float] = None
    recommendation: str = "NORMAL"


class StopHuntPredictor:
    """Predicts stop hunts by scoring proximity to liquidity pools, OFD divergence, and session timing."""

    THRESHOLD = 0.60

    def __init__(self, tick_sizes: dict):
        self.tick_sizes = tick_sizes

    def predict(
        self, instrument: str, current_price: float, liquidity_pools: list,
        ofd_net_delta: float, is_kill_zone: bool, is_kz_first_15min: bool,
        historical_hunt_rate: float = 0.50,
    ) -> HuntPrediction:
        tick = self.tick_sizes.get(instrument, 0.25)

        nearby = [
            p for p in liquidity_pools
            if getattr(p, "direction", "") == "SSL"
            and 0 < (current_price - p.level) < 50 * tick
        ]
        if not nearby:
            return HuntPrediction(detected=False)

        pool = min(nearby, key=lambda p: current_price - p.level)
        dist = (current_price - pool.level) / tick

        score = max(0.0, (50 - dist) / 50.0) * 0.30
        if is_kz_first_15min:
            score += 0.25
        elif is_kill_zone:
            score += 0.15

        if ofd_net_delta < -500:
            score += 0.20

        if any(abs(pool.level - round(pool.level / g) * g) < 3 * tick for g in [25, 50, 100, 250]):
            score += 0.10

        score += min(historical_hunt_rate, 0.15)

        if score >= self.THRESHOLD:
            return HuntPrediction(
                True, pool.level, score, max(1, int(dist / 5)),
                pool.level - 15 * tick, "WAIT_FOR_SWEEP_OR_STOP_BELOW_HUNT",
            )
        return HuntPrediction(detected=False, probability=score)
