"""Supreme Selectivity — 21-condition pre-signal checklist enforcing institutional-grade trade quality."""
import structlog
from ghost.core.models import ChecklistResult

logger = structlog.get_logger()


class PreSignalChecklist:
    """Evaluates 21 mandatory conditions before any signal can proceed to scoring.

    Every condition must pass (AND logic). Failed conditions are tagged for
    journal analysis and TQS rejection reasons.
    """

    def evaluate(self, ctx: dict) -> ChecklistResult:
        d = ctx.get("direction", "BULLISH")

        conditions = [
            ("htf_4h_bias_confirmed",
             ctx.get("macro_regime_label") in ("TRENDING_BULLISH", "TRENDING_BEARISH")
             and ctx.get("macro_regime_confidence", 0) >= 0.75),

            ("itf_1h_aligned",
             ctx.get("itf_regime_direction", "") == ctx.get("macro_regime_direction", "X")),

            ("entry_tf_15m_setup",
             bool(ctx.get("fvg_15m_unmitigated")) or bool(ctx.get("ob_15m_unswept"))),

            ("trigger_tf_5m_ready",
             bool(ctx.get("approaching_fvg_5m"))),

            ("draw_on_liquidity_identified",
             bool(ctx.get("primary_liquidity_target_exists"))
             and not bool(ctx.get("primary_liquidity_target_mitigated"))),

            ("price_in_discount_or_premium",
             (ctx.get("premium_discount_position", 0.5) <= 0.50 if d == "BULLISH"
              else ctx.get("premium_discount_position", 0.5) >= 0.50)),

            ("ssl_or_bsl_swept_pre_entry",
             bool(ctx.get("recent_liquidity_sweep_confirmed"))),

            ("clear_path_to_tp1",
             bool(ctx.get("path_to_tp1_clear"))),

            ("ofd_confirmed_direction",
             ctx.get("ofd_net_delta_direction", "") == d),

            ("institutional_footprint_detected",
             bool(ctx.get("institutional_footprint_detected"))),

            ("amd_phase_ok",
             ctx.get("amd_phase", "UNKNOWN") in ("DISTRIBUTION", "UNKNOWN")),

            ("inducement_completed",
             not bool(ctx.get("inducement_detected"))
             or bool(ctx.get("inducement_completed"))),

            ("news_risk_ok",
             ctx.get("news_risk_level", "HIGH") in ("LOW", "NEUTRAL")
             and bool(ctx.get("news_sentiment_aligned"))),

            ("no_imminent_high_impact_event",
             ctx.get("minutes_to_next_high_impact", 999) > 30),

            ("dxy_alignment_ok",
             bool(ctx.get("dxy_correlation_aligned", True))
             or not bool(ctx.get("dxy_sensitive", False))),

            ("macro_not_ranging",
             ctx.get("macro_regime_label", "RANGING") not in ("RANGING", "INVALIDATED")),

            ("kill_zone_active",
             bool(ctx.get("kill_zone_active"))),

            ("no_circuit_breaker_active",
             not bool(ctx.get("circuit_breaker_active"))),

            ("session_volume_adequate",
             ctx.get("session_volume_ratio", 1.0) >= 0.70),

            ("daily_risk_budget_available",
             ctx.get("daily_loss_used_pct", 0) <= 0.40),

            ("no_correlated_position_open",
             not bool(ctx.get("has_correlated_open_position"))),
        ]

        passed = [n for n, r in conditions if r]
        failed = [n for n, r in conditions if not r]

        if failed:
            logger.info("checklist.failed", failed=failed)
        else:
            logger.info("checklist.all_21_passed")

        return ChecklistResult(
            len(failed) == 0, len(passed), len(conditions),
            failed, [f"CHECKLIST_{n.upper()}" for n in failed],
        )
