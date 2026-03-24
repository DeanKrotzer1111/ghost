"""Liquidity pool detection for ICT signal engine."""
from typing import List, Optional

import structlog

from ghost.modules.m01_data.models import Bar, LiquidityPool

logger = structlog.get_logger(__name__)


class LiquidityDetector:
    """Detects Buy-Side Liquidity (BSL) and Sell-Side Liquidity (SSL).

    BSL = cluster of equal/near-equal highs (stop losses above).
    SSL = cluster of equal/near-equal lows (stop losses below).
    """

    def __init__(self, tolerance_pct: float = 0.001, min_touches: int = 2,
                 lookback: int = 20):
        """
        Args:
            tolerance_pct: How close highs/lows must be to count as "equal"
                           (fraction of price, e.g. 0.001 = 0.1%).
            min_touches: Minimum number of bars touching the level to form a pool.
            lookback: Number of bars to scan.
        """
        self.tolerance_pct = tolerance_pct
        self.min_touches = min_touches
        self.lookback = lookback
        self._active_pools: List[LiquidityPool] = []
        self._swept_pools: List[LiquidityPool] = []
        self._log = logger.bind(component="LiquidityDetector")

    @property
    def active_pools(self) -> List[LiquidityPool]:
        return list(self._active_pools)

    @property
    def bsl_pools(self) -> List[LiquidityPool]:
        return [p for p in self._active_pools if p.direction == "BSL"]

    @property
    def ssl_pools(self) -> List[LiquidityPool]:
        return [p for p in self._active_pools if p.direction == "SSL"]

    def detect(self, bars: List[Bar]) -> List[LiquidityPool]:
        """Scan bars for liquidity clusters. Returns newly detected pools.

        Uses the last `lookback` bars. Replaces the active pool list each call
        with a fresh scan (pools are structural, not incremental).
        """
        if len(bars) < self.min_touches:
            return []

        scan_bars = bars[-self.lookback:] if len(bars) > self.lookback else bars

        new_pools: List[LiquidityPool] = []

        # Detect BSL: clusters of near-equal highs
        bsl = self._find_clusters([b.high for b in scan_bars],
                                  [b.timestamp for b in scan_bars],
                                  direction="BSL")
        # Detect SSL: clusters of near-equal lows
        ssl = self._find_clusters([b.low for b in scan_bars],
                                  [b.timestamp for b in scan_bars],
                                  direction="SSL")

        all_new = bsl + ssl

        # Merge with existing: keep existing if same level, add new ones
        existing_levels = {(p.level, p.direction) for p in self._active_pools}
        for pool in all_new:
            key = (pool.level, pool.direction)
            if key not in existing_levels:
                new_pools.append(pool)
                self._active_pools.append(pool)
                self._log.debug("liquidity_pool_detected",
                                direction=pool.direction, level=pool.level,
                                strength=pool.strength)

        # Also rebuild active list — remove stale pools not found in latest scan
        new_levels = {(round(p.level, 6), p.direction) for p in all_new}
        refreshed: List[LiquidityPool] = []
        for p in self._active_pools:
            rounded_key = (round(p.level, 6), p.direction)
            if rounded_key in new_levels and not p.mitigated:
                refreshed.append(p)
        self._active_pools = refreshed

        # Update sweep status with latest bar
        if bars:
            self._update_sweeps(bars[-1])

        return new_pools

    def update(self, bar: Bar) -> None:
        """Check if the new bar sweeps any liquidity pools."""
        self._update_sweeps(bar)

    def _find_clusters(self, values: List[float], timestamps: List[float],
                       direction: str) -> List[LiquidityPool]:
        """Group near-equal values into clusters and return LiquidityPool objects."""
        if not values:
            return []

        # Sort values with their indices
        indexed = sorted(enumerate(values), key=lambda x: x[1])
        clusters: List[List[int]] = []
        current_cluster: List[int] = [indexed[0][0]]

        for k in range(1, len(indexed)):
            prev_val = indexed[k - 1][1]
            curr_val = indexed[k][1]
            ref_price = max(abs(prev_val), 1e-9)
            if abs(curr_val - prev_val) / ref_price <= self.tolerance_pct:
                current_cluster.append(indexed[k][0])
            else:
                if len(current_cluster) >= self.min_touches:
                    clusters.append(current_cluster)
                current_cluster = [indexed[k][0]]

        if len(current_cluster) >= self.min_touches:
            clusters.append(current_cluster)

        pools: List[LiquidityPool] = []
        for cluster_indices in clusters:
            cluster_values = [values[i] for i in cluster_indices]
            level = sum(cluster_values) / len(cluster_values)
            earliest_ts = min(timestamps[i] for i in cluster_indices)
            pool = LiquidityPool(
                level=round(level, 6),
                direction=direction,
                strength=len(cluster_indices),
                mitigated=False,
                formation_time=earliest_ts,
            )
            pools.append(pool)

        return pools

    def _update_sweeps(self, bar: Bar) -> None:
        """Mark pools as swept when price trades through the level."""
        still_active: List[LiquidityPool] = []
        for pool in self._active_pools:
            if pool.mitigated:
                self._swept_pools.append(pool)
                continue

            swept = False
            if pool.direction == "BSL":
                # Buy-side liquidity is swept when price goes above the level
                if bar.high > pool.level:
                    swept = True
            elif pool.direction == "SSL":
                # Sell-side liquidity is swept when price drops below the level
                if bar.low < pool.level:
                    swept = True

            if swept:
                pool.mitigated = True
                self._swept_pools.append(pool)
                self._log.debug("liquidity_swept", direction=pool.direction,
                                level=pool.level)
            else:
                still_active.append(pool)

        self._active_pools = still_active

    def get_nearest_pool(self, price: float, direction: str = "") -> Optional[LiquidityPool]:
        """Return the nearest unswept pool to the given price."""
        candidates = self._active_pools
        if direction:
            candidates = [p for p in candidates if p.direction == direction]
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p.level - price))

    def reset(self) -> None:
        """Clear all tracked pools."""
        self._active_pools.clear()
        self._swept_pools.clear()
