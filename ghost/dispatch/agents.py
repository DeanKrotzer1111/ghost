"""Ghost Agent Dispatcher — orchestrates specialized Claude agents for trading operations."""
import asyncio
import structlog
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

from ghost.dispatch.tools import create_ghost_tools

logger = structlog.get_logger()

# === Specialized Agent Definitions ===

MARKET_ANALYST = AgentDefinition(
    description="Analyzes market conditions, regime, and identifies high-probability setups",
    prompt=(
        "You are Ghost's Market Analyst agent. Your job is to:\n"
        "1. Scan instruments for active trading signals\n"
        "2. Analyze market regime (trending/ranging/volatile)\n"
        "3. Check weekly profile bias alignment\n"
        "4. Identify the highest-probability setups\n"
        "5. Report findings with confidence levels\n\n"
        "Only flag setups where regime, weekly profile, and MTF pyramid align. "
        "Be extremely selective — Ghost targets 90%+ win rate."
    ),
    tools=[
        "mcp__ghost-trading__scan_instruments",
        "mcp__ghost-trading__analyze_regime",
        "mcp__ghost-trading__check_weekly_profile",
    ],
)

SIGNAL_PROCESSOR = AgentDefinition(
    description="Processes trading signals through the 10-gate validation pipeline",
    prompt=(
        "You are Ghost's Signal Processor agent. Your job is to:\n"
        "1. Run signals through the full 10-gate pipeline\n"
        "2. Evaluate the 21-condition checklist\n"
        "3. Calculate the Trade Quality Score (TQS)\n"
        "4. Run ensemble LLM consensus voting\n"
        "5. Only approve signals that pass ALL gates with TQS >= 85\n\n"
        "CRITICAL: Never approve a signal that fails any gate. "
        "Shadow-log TQS 80-84 signals but do NOT execute them."
    ),
    tools=[
        "mcp__ghost-trading__evaluate_signal",
        "mcp__ghost-trading__run_tqs_score",
        "mcp__ghost-trading__run_checklist",
        "mcp__ghost-trading__run_ensemble",
    ],
)

RISK_MANAGER = AgentDefinition(
    description="Manages portfolio risk, position sizing, and payout phase optimization",
    prompt=(
        "You are Ghost's Risk Manager agent. Your job is to:\n"
        "1. Check current portfolio positions and risk utilization\n"
        "2. Verify payout phase and adjust risk parameters accordingly\n"
        "3. Calculate optimal position sizes\n"
        "4. Enforce daily loss limits (max 3% daily, 1% per trade)\n"
        "5. Block trades that would violate funded account rules\n\n"
        "CRITICAL: In FINAL_APPROACH and PAYOUT_TRIGGERED phases, "
        "reduce risk aggressively. Zero evaluation violations is non-negotiable."
    ),
    tools=[
        "mcp__ghost-trading__check_portfolio",
        "mcp__ghost-trading__check_payout_state",
        "mcp__ghost-trading__calculate_position_size",
    ],
)

CALIBRATION_AGENT = AgentDefinition(
    description="Runs system calibration, monitors performance, and optimizes parameters",
    prompt=(
        "You are Ghost's Calibration agent. Your job is to:\n"
        "1. Run weekly self-calibration of stop buffers and TP extensions\n"
        "2. Analyze trade history for parameter drift\n"
        "3. Review shadow signals for missed opportunities\n"
        "4. Check system health across all services\n"
        "5. Report calibration changes and performance metrics\n\n"
        "Focus on statistical significance — only recommend changes "
        "backed by sufficient sample sizes (n >= 20)."
    ),
    tools=[
        "mcp__ghost-trading__run_calibration",
        "mcp__ghost-trading__query_trade_history",
        "mcp__ghost-trading__get_shadow_signals",
        "mcp__ghost-trading__system_health",
    ],
)


class GhostDispatcher:
    """Main dispatcher that coordinates Claude agents for Ghost trading operations.

    Usage:
        dispatcher = GhostDispatcher()

        # Run a full trading session
        async for msg in dispatcher.run_session("Scan MNQ and ES for setups"):
            print(msg)

        # Run specific agent tasks
        async for msg in dispatcher.analyze_market("Check NQ regime and weekly profile"):
            print(msg)

        async for msg in dispatcher.run_calibration("Run weekly calibration"):
            print(msg)
    """

    def __init__(self, system_prompt: str = None):
        self.ghost_tools = create_ghost_tools()
        self.session_id = None
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return (
            "You are Ghost, an autonomous AI futures trading system. "
            "You coordinate specialized agents to identify, validate, and execute trades "
            "across 30 CME instruments.\n\n"
            "RULES:\n"
            "1. Every trade must pass all 10 gates before execution\n"
            "2. Minimum TQS score of 85 to execute (shadow-log 80-84)\n"
            "3. Never violate funded account evaluation rules\n"
            "4. All 21 checklist conditions must pass (AND logic)\n"
            "5. Ensemble LLM must reach STRONG_APPROVE or UNANIMOUS_APPROVE\n"
            "6. Adjust risk based on payout phase proximity\n"
            "7. Ghost API runs on :8080, Qwen MLX on :8081 — NEVER swap these\n\n"
            "Win in silence."
        )

    async def run_session(self, prompt: str):
        """Run a full trading session with all agents coordinated."""
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                mcp_servers={"ghost-trading": self.ghost_tools},
                allowed_tools=[
                    "mcp__ghost-trading__scan_instruments",
                    "mcp__ghost-trading__analyze_regime",
                    "mcp__ghost-trading__check_weekly_profile",
                    "mcp__ghost-trading__evaluate_signal",
                    "mcp__ghost-trading__run_tqs_score",
                    "mcp__ghost-trading__run_checklist",
                    "mcp__ghost-trading__run_ensemble",
                    "mcp__ghost-trading__check_portfolio",
                    "mcp__ghost-trading__check_payout_state",
                    "mcp__ghost-trading__calculate_position_size",
                    "mcp__ghost-trading__system_health",
                ],
                agents={
                    "market-analyst": MARKET_ANALYST,
                    "signal-processor": SIGNAL_PROCESSOR,
                    "risk-manager": RISK_MANAGER,
                },
                system_prompt=self.system_prompt,
            ),
        ):
            if hasattr(message, "subtype") and message.subtype == "init":
                self.session_id = message.session_id
                logger.info("ghost.dispatch.session_started", session_id=self.session_id)
            if hasattr(message, "result"):
                yield message.result

    async def analyze_market(self, prompt: str):
        """Dispatch the market analyst agent."""
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                mcp_servers={"ghost-trading": self.ghost_tools},
                allowed_tools=[
                    "mcp__ghost-trading__scan_instruments",
                    "mcp__ghost-trading__analyze_regime",
                    "mcp__ghost-trading__check_weekly_profile",
                ],
                system_prompt=(
                    "You are Ghost's market analyst. Scan instruments, analyze regime, "
                    "and identify high-probability setups. Be extremely selective."
                ),
            ),
        ):
            if hasattr(message, "result"):
                yield message.result

    async def process_signal(self, prompt: str):
        """Dispatch the signal processor agent."""
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                mcp_servers={"ghost-trading": self.ghost_tools},
                allowed_tools=[
                    "mcp__ghost-trading__evaluate_signal",
                    "mcp__ghost-trading__run_tqs_score",
                    "mcp__ghost-trading__run_checklist",
                    "mcp__ghost-trading__run_ensemble",
                ],
                system_prompt=(
                    "You are Ghost's signal processor. Run signals through all 10 gates. "
                    "Only approve TQS >= 85 with ensemble STRONG_APPROVE or better."
                ),
            ),
        ):
            if hasattr(message, "result"):
                yield message.result

    async def manage_risk(self, prompt: str):
        """Dispatch the risk manager agent."""
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                mcp_servers={"ghost-trading": self.ghost_tools},
                allowed_tools=[
                    "mcp__ghost-trading__check_portfolio",
                    "mcp__ghost-trading__check_payout_state",
                    "mcp__ghost-trading__calculate_position_size",
                ],
                system_prompt=(
                    "You are Ghost's risk manager. Enforce position limits, "
                    "payout phase rules, and funded account compliance. Zero violations."
                ),
            ),
        ):
            if hasattr(message, "result"):
                yield message.result

    async def run_calibration(self, prompt: str = "Run weekly calibration"):
        """Dispatch the calibration agent."""
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                mcp_servers={"ghost-trading": self.ghost_tools},
                allowed_tools=[
                    "mcp__ghost-trading__run_calibration",
                    "mcp__ghost-trading__query_trade_history",
                    "mcp__ghost-trading__get_shadow_signals",
                    "mcp__ghost-trading__system_health",
                ],
                system_prompt=(
                    "You are Ghost's calibration agent. Recalibrate stop buffers and TP extensions. "
                    "Only recommend changes backed by n >= 20 samples."
                ),
            ),
        ):
            if hasattr(message, "result"):
                yield message.result

    async def resume_session(self, prompt: str):
        """Resume a previous dispatch session with full context."""
        if not self.session_id:
            raise ValueError("No active session to resume. Use run_session() first.")

        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(resume=self.session_id),
        ):
            if hasattr(message, "result"):
                yield message.result
