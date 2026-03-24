"""Ghost Dispatch CLI — command-line interface for dispatching Claude agents."""
import asyncio
import argparse
import sys
import structlog

from ghost.dispatch.agents import GhostDispatcher

logger = structlog.get_logger()


async def main():
    parser = argparse.ArgumentParser(
        description="Ghost Dispatch — AI-powered trading agent orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ghost-dispatch session "Scan MNQ and ES for setups during London session"
  ghost-dispatch analyze "Check NQ regime and weekly profile alignment"
  ghost-dispatch signal "Evaluate BULLISH MNQ signal at FVG 21450-21475"
  ghost-dispatch risk "Check portfolio risk and payout phase for account topstep_50k"
  ghost-dispatch calibrate "Run weekly calibration and report changes"
  ghost-dispatch health "Check all Ghost service health"
        """,
    )

    parser.add_argument(
        "command",
        choices=["session", "analyze", "signal", "risk", "calibrate", "health"],
        help="Dispatch command to run",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="Natural language prompt for the agent",
    )

    args = parser.parse_args()
    dispatcher = GhostDispatcher()

    dispatch_map = {
        "session": dispatcher.run_session,
        "analyze": dispatcher.analyze_market,
        "signal": dispatcher.process_signal,
        "risk": dispatcher.manage_risk,
        "calibrate": dispatcher.run_calibration,
        "health": lambda p: dispatcher.run_calibration(p or "Check system health"),
    }

    handler = dispatch_map[args.command]
    prompt = args.prompt or f"Run {args.command} operation"

    print(f"Ghost Dispatch > {args.command}: {prompt}")
    print("=" * 60)

    async for result in handler(prompt):
        print(result)


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
