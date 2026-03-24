"""Trade Quality Score (TQS) Engine — 8-dimension scoring system (0-100) gating trade execution."""
import structlog
from ghost.core.models import TQSResult, TQSGrade

logger = structlog.get_logger()


class TradeQualityScorer:
    """Scores trades across 8 dimensions (12.5 pts each = 100 max).

    Grades:
        PREMIUM  >= 90  -> execute with full size
        STANDARD >= 85  -> execute with standard size
        SHADOW   >= 80  -> shadow log only (never silently discard)
        REJECT   <  80  -> reject with reason
    """

    def score(self, ctx: dict, historical_similarity: float = 0.5) -> TQSResult:
        s = {}

        # 1. Structure Clarity (12.5)
        mc = ctx.get("macro_regime_confidence", 0.5)
        atf = ctx.get("all_timeframes_aligned", False)
        s["structure_clarity"] = (
            12.5 if atf and mc >= 0.90 else
            10.0 if mc >= 0.80 else
            7.5 if mc >= 0.70 else
            5.0 if mc >= 0.60 else 0.0
        )

        # 2. FVG Quality (12.5)
        h4 = ctx.get("fvg_4h_present", False)
        h1 = ctx.get("fvg_1h_present", False)
        um = ctx.get("fvg_completely_unmitigated", False)
        f15 = ctx.get("fvg_15m_unmitigated", False)
        s["fvg_quality"] = (
            12.5 if h4 and h1 and um else
            10.0 if h4 and um else
            7.5 if h1 and um else
            5.0 if f15 else 0.0
        )

        # 3. Liquidity Target (12.5)
        ltf = ctx.get("liquidity_target_timeframe", "15m")
        lu = ctx.get("liquidity_sessions_untouched", 0)
        lte = ctx.get("primary_liquidity_target_exists", False)
        s["liquidity_target"] = (
            12.5 if ltf in ("4h", "1d") and lu >= 3 else
            10.0 if ltf == "1h" and lu >= 2 else
            7.5 if ltf == "15m" and lu >= 1 else
            5.0 if lte else 0.0
        )

        # 4. Sweep Quality (12.5)
        hc = ctx.get("hunt_confidence", 0.0)
        s["sweep_quality"] = (
            12.5 if hc >= 0.87 else
            10.0 if hc >= 0.79 else
            7.5 if hc >= 0.70 else
            5.0 if hc >= 0.50 else 0.0
        )

        # 5. Institutional Footprint (12.5)
        fp = ctx.get("footprint_composite", 0.0)
        s["institutional_footprint"] = (
            12.5 if fp >= 0.90 else
            10.0 if fp >= 0.80 else
            7.5 if fp >= 0.70 else
            5.0 if fp >= 0.50 else 0.0
        )

        # 6. News Alignment (12.5)
        ss = abs(ctx.get("news_sentiment_score", 0.0))
        nr = ctx.get("news_risk_level", "HIGH")
        me = ctx.get("minutes_to_next_high_impact", 0)
        s["news_alignment"] = (
            12.5 if ss >= 0.60 and nr == "LOW" and me >= 120 else
            10.0 if nr == "LOW" and me >= 60 else
            7.5 if nr == "NEUTRAL" and me >= 45 else
            5.0 if me >= 30 else 0.0
        )

        # 7. AMD / Session (12.5)
        amd = ctx.get("amd_phase", "UNKNOWN")
        kzm = ctx.get("kill_zone_minutes_elapsed", 999)
        kza = ctx.get("kill_zone_active", False)
        s["amd_session"] = (
            12.5 if amd == "DISTRIBUTION" and kzm <= 30 else
            10.0 if amd == "DISTRIBUTION" else
            7.5 if amd == "UNKNOWN" and kza else
            5.0 if kza else 0.0
        )

        # 8. Historical Similarity (12.5)
        s["historical_similarity"] = (
            12.5 if historical_similarity >= 0.90 else
            10.0 if historical_similarity >= 0.80 else
            7.5 if historical_similarity >= 0.70 else
            5.0 if historical_similarity >= 0.60 else 0.0
        )

        total = sum(s.values())
        grade = (
            TQSGrade.PREMIUM if total >= 90 else
            TQSGrade.STANDARD if total >= 85 else
            TQSGrade.SHADOW if total >= 80 else
            TQSGrade.REJECT
        )
        execute = grade in (TQSGrade.PREMIUM, TQSGrade.STANDARD)
        weakest = min(s, key=s.get)

        return TQSResult(
            total, grade, s, execute,
            grade == TQSGrade.SHADOW, weakest,
            weakest if not execute and grade != TQSGrade.SHADOW else None,
        )
