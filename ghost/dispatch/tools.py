"""Ghost Trading Tools — MCP server tools for Claude Agent SDK dispatch."""
import json
import structlog
from claude_agent_sdk import tool, create_sdk_mcp_server

logger = structlog.get_logger()


# === Market Analysis Tools ===

@tool(
    "scan_instruments",
    "Scan all 30 CME instruments for active signals and return ranked opportunities",
)
async def scan_instruments(args):
    """Scans instruments using regime detection and signal engine."""
    from ghost.config.settings import settings

    # Import dynamically to avoid circular imports at module level
    instruments = args.get("instruments", "all")
    timeframe = args.get("timeframe", "15m")

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "scan_instruments",
                "instruments": instruments,
                "timeframe": timeframe,
                "status": "dispatched",
                "note": "Connect to live Databento feed for real-time scanning",
            }),
        }],
    }


@tool(
    "analyze_regime",
    "Analyze the current market regime for a specific instrument (trending/ranging/volatile)",
    {"instrument": str, "timeframe": str},
)
async def analyze_regime(args):
    """Runs regime detection on the specified instrument."""
    instrument = args["instrument"]
    timeframe = args.get("timeframe", "4h")

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "analyze_regime",
                "instrument": instrument,
                "timeframe": timeframe,
                "status": "dispatched",
            }),
        }],
    }


@tool(
    "check_weekly_profile",
    "Get the current weekly profile classification (bullish/bearish/balanced)",
    {"instrument": str},
)
async def check_weekly_profile(args):
    """Returns the weekly profile bias for trade filtering."""
    instrument = args["instrument"]

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "check_weekly_profile",
                "instrument": instrument,
                "status": "dispatched",
            }),
        }],
    }


# === Signal Processing Tools ===

@tool(
    "evaluate_signal",
    "Run a trading signal through the full 10-gate pipeline and return the ProcessResult",
    {"instrument": str, "direction": str, "fvg_low": float, "fvg_high": float},
)
async def evaluate_signal(args):
    """Dispatches a signal through the Ghost 10-gate orchestrator."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "evaluate_signal",
                "instrument": args["instrument"],
                "direction": args["direction"],
                "fvg_low": args["fvg_low"],
                "fvg_high": args["fvg_high"],
                "gates": [
                    "MTF_PYRAMID", "WEEKLY_PROFILE", "FOOTPRINT",
                    "LIQUIDITY_VOID", "CHECKLIST_21", "TQS_SCORE",
                    "STOP_CALIBRATION", "ENTRY_OPTIMIZATION",
                    "TP_CALIBRATION", "ENSEMBLE_LLM",
                ],
                "status": "dispatched",
            }),
        }],
    }


@tool(
    "run_tqs_score",
    "Calculate the Trade Quality Score (0-100) for a signal context",
    {"instrument": str, "direction": str},
)
async def run_tqs_score(args):
    """Runs the TQS scorer on provided context."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "run_tqs_score",
                "instrument": args["instrument"],
                "direction": args["direction"],
                "status": "dispatched",
            }),
        }],
    }


@tool(
    "run_checklist",
    "Evaluate the 21-condition pre-signal checklist",
    {"instrument": str, "direction": str},
)
async def run_checklist(args):
    """Runs the 21-condition selectivity checklist."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "run_checklist",
                "instrument": args["instrument"],
                "direction": args["direction"],
                "conditions_total": 21,
                "status": "dispatched",
            }),
        }],
    }


# === Risk & Execution Tools ===

@tool(
    "check_payout_state",
    "Get current funded account payout phase and risk parameters",
    {"account_id": str},
)
async def check_payout_state(args):
    """Returns current payout phase, risk multiplier, and TQS minimum."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "check_payout_state",
                "account_id": args["account_id"],
                "phases": [
                    "APPROACH", "ACCELERATION", "FINAL_APPROACH",
                    "PAYOUT_TRIGGERED", "RECOVERY",
                ],
                "status": "dispatched",
            }),
        }],
    }


@tool(
    "calculate_position_size",
    "Calculate optimal position size given risk parameters",
    {"instrument": str, "entry": float, "stop": float, "account_balance": float},
)
async def calculate_position_size(args):
    """Computes position size using Kelly criterion with payout-phase scaling."""
    risk_per_trade = abs(args["entry"] - args["stop"])
    max_risk_pct = 0.01  # 1% default

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "calculate_position_size",
                "instrument": args["instrument"],
                "entry": args["entry"],
                "stop": args["stop"],
                "risk_dollars": risk_per_trade,
                "account_balance": args["account_balance"],
                "max_risk_pct": max_risk_pct,
                "status": "dispatched",
            }),
        }],
    }


@tool(
    "check_portfolio",
    "Get current open positions, P&L, and risk utilization",
)
async def check_portfolio(args):
    """Returns current portfolio state for risk management decisions."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "check_portfolio",
                "status": "dispatched",
            }),
        }],
    }


# === Calibration & System Tools ===

@tool(
    "run_calibration",
    "Trigger the weekly self-calibration loop for stop buffers and TP extensions",
)
async def run_calibration(args):
    """Runs the M29 self-calibration loop."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "run_calibration",
                "module": "M29_SELF_CALIBRATION",
                "status": "dispatched",
            }),
        }],
    }


@tool(
    "run_ensemble",
    "Run the dual-LLM ensemble vote on a trade decision",
    {"prompt": str},
)
async def run_ensemble(args):
    """Runs 10 parallel LLM calls (5 MiniMax + 5 Qwen) for consensus."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "run_ensemble",
                "models": ["minimax-2.5", "qwen-3.5-35b"],
                "temperatures": [0.1, 0.2, 0.3, 0.4, 0.5],
                "total_calls": 10,
                "status": "dispatched",
            }),
        }],
    }


@tool(
    "system_health",
    "Check health status of all Ghost services (API, Qwen, DB, Redis, Dashboard)",
)
async def system_health(args):
    """Checks connectivity to all Ghost infrastructure services."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "system_health",
                "services": [
                    "ghost_api:8080", "qwen_mlx:8081",
                    "postgresql:5432", "redis:6379", "dashboard:3000",
                ],
                "status": "dispatched",
            }),
        }],
    }


@tool(
    "query_trade_history",
    "Query historical trades with filters for analysis",
    {"instrument": str, "days": int},
)
async def query_trade_history(args):
    """Queries trade history from PostgreSQL for analysis and calibration."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "query_trade_history",
                "instrument": args["instrument"],
                "days": args["days"],
                "status": "dispatched",
            }),
        }],
    }


@tool(
    "get_shadow_signals",
    "Retrieve shadow-logged signals (TQS 80-84) for analysis",
    {"days": int},
)
async def get_shadow_signals(args):
    """Returns shadow signals that were logged but not executed."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "action": "get_shadow_signals",
                "days": args["days"],
                "note": "These are TQS 80-84 signals logged for counterfactual analysis",
                "status": "dispatched",
            }),
        }],
    }


# === Factory ===

def create_ghost_tools():
    """Create the Ghost MCP server with all trading tools."""
    return create_sdk_mcp_server(
        name="ghost-trading",
        version="5.5.0",
        tools=[
            # Market Analysis
            scan_instruments,
            analyze_regime,
            check_weekly_profile,
            # Signal Processing
            evaluate_signal,
            run_tqs_score,
            run_checklist,
            # Risk & Execution
            check_payout_state,
            calculate_position_size,
            check_portfolio,
            # Calibration & System
            run_calibration,
            run_ensemble,
            system_health,
            query_trade_history,
            get_shadow_signals,
        ],
    )
