"""M04 — Confluence Scoring Engine for Ghost v5.5."""
from dataclasses import dataclass, field
from typing import Dict, Optional

from ghost.modules.m01_data.models import MarketContext


# ---------------------------------------------------------------------------
# Default factor weights (v5.5 baseline)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: Dict[str, float] = {
    "regime_alignment": 0.20,
    "fvg_quality": 0.15,
    "liquidity_target": 0.15,
    "structure_alignment": 0.15,
    "ofd_confirmation": 0.10,
    "amd_phase": 0.10,
    "kill_zone": 0.10,
    "news_alignment": 0.05,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ConfluenceResult:
    """Outcome of confluence scoring."""
    composite: float
    factor_scores: Dict[str, float]
    factors_above_threshold: int
    recommendation: str


# ---------------------------------------------------------------------------
# Individual factor scoring helpers
# ---------------------------------------------------------------------------

def _score_regime_alignment(ctx: MarketContext) -> float:
    """How well the macro regime aligns with the proposed trade direction."""
    if not ctx.direction or ctx.macro_direction == "UNKNOWN":
        return 0.0

    # Perfect alignment: macro direction matches trade direction
    if ctx.macro_direction == ctx.direction:
        score = 0.6 + 0.4 * ctx.macro_confidence
    # Neutral regime — partial credit scaled by confidence
    elif ctx.macro_regime in ("RANGING", "TRANSITION", "UNKNOWN"):
        score = 0.3 * ctx.macro_confidence
    else:
        # Opposing direction
        score = max(0.0, 0.15 - 0.15 * ctx.macro_confidence)

    return round(min(1.0, max(0.0, score)), 4)


def _score_fvg_quality(ctx: MarketContext) -> float:
    """Quality of the current FVG setup across timeframes."""
    score = 0.0

    # 15-min FVG validity is the primary gate
    if ctx.fvg_15m_valid:
        score += 0.40

    # Higher-timeframe FVG confluence
    if ctx.fvg_4h_present:
        score += 0.25
    if ctx.fvg_1h_present:
        score += 0.15

    # Unmitigated FVG adds edge
    if ctx.fvg_unmitigated:
        score += 0.20

    return round(min(1.0, max(0.0, score)), 4)


def _score_liquidity_target(ctx: MarketContext) -> float:
    """Evaluate quality of identified liquidity target."""
    if ctx.liquidity_target is None:
        return 0.0

    target = ctx.liquidity_target
    score = 0.0

    # Base credit for having a target
    score += 0.30

    # Strength of the pool (1-5 scale normalised)
    score += min(0.30, target.strength * 0.06)

    # Sweep confirmation is a strong signal
    if ctx.sweep_confirmed:
        score += 0.25

    # Untouched across sessions adds conviction
    sessions_bonus = min(0.15, ctx.liquidity_sessions_untouched * 0.05)
    score += sessions_bonus

    return round(min(1.0, max(0.0, score)), 4)


def _score_structure_alignment(ctx: MarketContext) -> float:
    """Market-structure alignment: premium/discount + path clarity."""
    score = 0.0

    direction = ctx.direction

    # Premium/discount zone (0.0 = deep discount, 1.0 = deep premium)
    if direction == "LONG":
        # Buying in discount is ideal
        discount_score = max(0.0, 1.0 - ctx.premium_discount)
        score += 0.40 * discount_score
    elif direction == "SHORT":
        # Selling in premium is ideal
        score += 0.40 * ctx.premium_discount
    else:
        score += 0.10  # unknown direction, small base

    # Clear path to target
    if ctx.path_clear:
        score += 0.35

    # Approaching an FVG (entry zone)
    if ctx.approaching_fvg:
        score += 0.25

    return round(min(1.0, max(0.0, score)), 4)


def _score_ofd_confirmation(ctx: MarketContext) -> float:
    """Order-flow delta confirmation."""
    if not ctx.ofd_direction:
        return 0.0

    score = 0.0

    # Direction alignment
    if ctx.direction and ctx.ofd_direction == ctx.direction:
        score += 0.50
    elif ctx.ofd_direction in ("NEUTRAL", "UNKNOWN", ""):
        score += 0.10
    else:
        # Opposing OFD — still give tiny base so it's not zero when delta is weak
        score += 0.0

    # Net delta magnitude — normalise loosely (assume +-500 is strong)
    delta_mag = min(1.0, abs(ctx.ofd_net_delta) / 500.0)
    direction_match = (ctx.direction == "LONG" and ctx.ofd_net_delta > 0) or \
                      (ctx.direction == "SHORT" and ctx.ofd_net_delta < 0)
    if direction_match:
        score += 0.50 * delta_mag

    return round(min(1.0, max(0.0, score)), 4)


def _score_amd_phase(ctx: MarketContext) -> float:
    """Score based on AMD (Accumulation-Manipulation-Distribution) phase."""
    phase = ctx.amd_phase.upper()
    phase_scores = {
        "DISTRIBUTION": 0.95,
        "MANIPULATION": 0.75,
        "ACCUMULATION": 0.50,
        "EXPANSION": 0.40,
        "UNKNOWN": 0.10,
    }
    return phase_scores.get(phase, 0.10)


def _score_kill_zone(ctx: MarketContext) -> float:
    """Score based on kill-zone timing."""
    if not ctx.kill_zone_active:
        return 0.05

    score = 0.60

    # First 15 minutes of a kill zone are the best entries
    if ctx.kill_zone_first_15min:
        score += 0.30
    else:
        # Decay after first 15 minutes
        decay = min(0.20, ctx.kill_zone_minutes_elapsed * 0.004)
        score += max(0.0, 0.20 - decay)

    return round(min(1.0, max(0.0, score)), 4)


def _score_news_alignment(ctx: MarketContext) -> float:
    """Score based on news/calendar risk."""
    # High risk news approaching — bad environment
    if ctx.news_risk_level == "HIGH" and ctx.minutes_to_event < 30:
        return 0.05
    if ctx.news_risk_level == "HIGH" and ctx.minutes_to_event < 60:
        return 0.20

    score = 0.50

    if ctx.news_aligned:
        score += 0.30

    # Sentiment boost (normalised -1..+1 to 0..0.20)
    if ctx.direction == "LONG":
        score += max(0.0, ctx.news_sentiment_score * 0.20)
    elif ctx.direction == "SHORT":
        score += max(0.0, -ctx.news_sentiment_score * 0.20)

    if ctx.news_risk_level == "LOW":
        score += 0.10

    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------
_FACTOR_FNS = {
    "regime_alignment": _score_regime_alignment,
    "fvg_quality": _score_fvg_quality,
    "liquidity_target": _score_liquidity_target,
    "structure_alignment": _score_structure_alignment,
    "ofd_confirmation": _score_ofd_confirmation,
    "amd_phase": _score_amd_phase,
    "kill_zone": _score_kill_zone,
    "news_alignment": _score_news_alignment,
}


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------
class ConfluenceScorer:
    """Compute a weighted confluence score from a MarketContext.

    Parameters
    ----------
    weight_overrides : dict, optional
        Override one or more default factor weights (v5.5 tuning).
        Keys must be valid factor names.  Missing keys fall back to
        ``DEFAULT_WEIGHTS``.
    threshold : float
        Per-factor threshold used to count ``factors_above_threshold``.
    """

    def __init__(
        self,
        weight_overrides: Optional[Dict[str, float]] = None,
        threshold: float = 0.50,
    ) -> None:
        self.weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)
        if weight_overrides:
            for key, val in weight_overrides.items():
                if key not in DEFAULT_WEIGHTS:
                    raise ValueError(
                        f"Unknown confluence factor: '{key}'. "
                        f"Valid factors: {list(DEFAULT_WEIGHTS.keys())}"
                    )
                self.weights[key] = val
        self.threshold = threshold

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def score(self, context: MarketContext) -> ConfluenceResult:
        """Evaluate all confluence factors and return a ``ConfluenceResult``."""
        factor_scores: Dict[str, float] = {}
        for name, fn in _FACTOR_FNS.items():
            factor_scores[name] = fn(context)

        # Weighted composite
        composite = sum(
            self.weights[name] * factor_scores[name] for name in factor_scores
        )
        composite = round(min(1.0, max(0.0, composite)), 4)

        factors_above = sum(
            1 for v in factor_scores.values() if v >= self.threshold
        )

        recommendation = _recommendation(composite)

        return ConfluenceResult(
            composite=composite,
            factor_scores=factor_scores,
            factors_above_threshold=factors_above,
            recommendation=recommendation,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recommendation(composite: float) -> str:
    if composite >= 0.75:
        return "STRONG_SETUP"
    if composite >= 0.55:
        return "MODERATE_SETUP"
    if composite >= 0.40:
        return "WEAK_SETUP"
    return "NO_SETUP"
