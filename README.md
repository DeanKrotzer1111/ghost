# Ghost v5.5 — AI-Powered Autonomous Futures Trading System

> **Win In Silence** — A production-grade, multi-model AI trading system that autonomously identifies, validates, and executes futures trades across 30 CME instruments using institutional order flow analysis, ensemble LLM consensus, and adaptive self-calibration.

**Developer:** Dean Krotzer
**Stack:** Python 3.11+ | FastAPI | Claude Agent SDK | PostgreSQL | Redis | MiniMax 2.5 | Qwen 3.5-35B (MLX) | Claude Sonnet | Apple M4 Max 128GB

---

## Architecture Overview

Ghost is built as a **modular, event-driven trading pipeline** with a 10-gate signal processing architecture. Every trade candidate must pass through all 10 gates before execution — achieving a target **90%+ win rate** with zero funded account evaluation violations.

```
Signal → [MTF Pyramid] → [Weekly Profile] → [Footprint Detection] → [Liquidity Void]
       → [21-Point Checklist] → [TQS Scoring] → [Stop Calibration] → [Entry Optimization]
       → [TP Calibration] → [Ensemble LLM Vote] → Execute / Reject
```

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GHOST ORCHESTRATOR                          │
│                    (10-Gate Signal Pipeline)                        │
├─────────────┬─────────────┬─────────────┬──────────────────────────┤
│  DATA LAYER │  AI LAYER   │ RISK LAYER  │     EXECUTION LAYER      │
│             │             │             │                          │
│  Databento  │  MiniMax 2.5│  TQS Engine │  FastAPI :8080           │
│  PostgreSQL │  Qwen 35B   │  Payout Mgr │  Order Management        │
│  Redis      │  Claude     │  Checklist  │  Position Monitoring     │
│  OFD Engine │  Ensemble   │  Stop Calib │  Dynamic TP/SL           │
└─────────────┴─────────────┴─────────────┴──────────────────────────┘
```

---

## Claude Agent SDK Dispatch

Ghost integrates the **Claude Agent SDK** for autonomous agent orchestration. Specialized Claude agents handle different aspects of the trading pipeline — from market analysis to risk management — coordinated through a natural language dispatch interface.

### Dispatch Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    GHOST DISPATCHER                           │
│              (Claude Agent SDK Orchestration)                 │
├──────────────┬───────────────┬───────────────┬───────────────┤
│   MARKET     │    SIGNAL     │     RISK      │  CALIBRATION  │
│   ANALYST    │   PROCESSOR   │    MANAGER    │    AGENT      │
│              │               │               │               │
│  Scan instr  │  10-gate eval │  Portfolio    │  Weekly       │
│  Regime det  │  TQS scoring  │  Payout phase │  recalibrate  │
│  Weekly prof │  Ensemble LLM │  Position size│  Shadow review│
└──────────────┴───────────────┴───────────────┴───────────────┘
         ▼              ▼               ▼               ▼
    14 Custom MCP Tools (in-process, zero subprocess overhead)
```

### Quick Start — Dispatch CLI

```bash
# Full trading session with coordinated agents
ghost-dispatch session "Scan MNQ and ES for setups during London kill zone"

# Market analysis only
ghost-dispatch analyze "Check NQ regime and weekly profile alignment"

# Process a specific signal through all 10 gates
ghost-dispatch signal "Evaluate BULLISH MNQ signal at FVG 21450-21475"

# Risk check before execution
ghost-dispatch risk "Check portfolio risk and payout phase for topstep_50k"

# Weekly recalibration
ghost-dispatch calibrate "Run weekly calibration and report all parameter changes"

# System health check
ghost-dispatch health "Check all Ghost services"
```

### Python API

```python
import asyncio
from ghost.dispatch import GhostDispatcher

async def main():
    dispatcher = GhostDispatcher()

    # Run a full trading session with 3 coordinated agents
    async for result in dispatcher.run_session(
        "Scan all equity index futures for high-TQS setups"
    ):
        print(result)

    # Or dispatch individual specialist agents
    async for result in dispatcher.analyze_market("NQ regime check"):
        print(result)

    async for result in dispatcher.manage_risk("Check payout phase"):
        print(result)

asyncio.run(main())
```

### Custom MCP Tools (14 tools, in-process)

| Tool | Description |
|------|-------------|
| `scan_instruments` | Scan all 30 CME instruments for active signals |
| `analyze_regime` | Market regime detection (trending/ranging/volatile) |
| `check_weekly_profile` | Weekly bias classification |
| `evaluate_signal` | Full 10-gate signal pipeline evaluation |
| `run_tqs_score` | Trade Quality Score calculation (0-100) |
| `run_checklist` | 21-condition pre-signal checklist |
| `run_ensemble` | Dual-LLM multi-temperature consensus vote |
| `check_payout_state` | Funded account payout phase and risk params |
| `calculate_position_size` | Kelly criterion position sizing |
| `check_portfolio` | Current positions, P&L, risk utilization |
| `run_calibration` | Weekly stop/TP parameter recalibration |
| `query_trade_history` | Historical trade analysis |
| `get_shadow_signals` | TQS 80-84 shadow signal review |
| `system_health` | Service health check across all components |

---

## Key Technical Highlights

### Multi-Model AI Ensemble (M28)
- **Dual-LLM architecture**: MiniMax 2.5 (cloud) + Qwen 3.5-35B (local MLX inference on M4 Max)
- **Multi-temperature voting**: 10 parallel inference calls (5 per model) at temperatures 0.1–0.5
- **Consensus classification**: UNANIMOUS_APPROVE → STRONG_REJECT with confidence intervals
- Claude Sonnet serves as independent judge for adversarial validation

### Trade Quality Score Engine (M21)
- **8-dimensional scoring system** (0–100 scale, 12.5 points per dimension)
- Dimensions: Structure Clarity, FVG Quality, Liquidity Target, Sweep Quality, Institutional Footprint, News Alignment, AMD/Session Timing, Historical Similarity
- Grade tiers: PREMIUM (90+), STANDARD (85+), SHADOW (80+, logged only), REJECT (<80)
- Dynamic TQS thresholds that scale with payout phase proximity

### Smart Money Footprint Detection (M22)
- Real-time order flow analysis: absorption detection, iceberg orders, stop raids, delta divergence
- Bid/ask imbalance ratio tracking with weighted composite scoring
- Tick-level data processing for institutional activity identification

### Self-Calibrating System (M29)
- **Automated weekly recalibration** (Sunday 10pm ET) of stop buffers and TP extensions
- Per-instrument parameter optimization using rolling trade statistics
- Percentile-based sweep distribution analysis (p50/p75/p90)
- Session-bucketed TP overshoot learning (2-hour windows)

### Adaptive Stop Loss Calibration (M17)
- **Historical sweep distribution analysis** per instrument and setup type
- VIX-scaled stop placement (1.0x–1.6x based on volatility regime)
- Round-number avoidance algorithm preventing predictable stop placement
- Stop hunt prediction using liquidity pool proximity, OFD divergence, and kill zone timing
- Dynamic stop migration via R-multiple ratcheting and structural FVG shifts

### Multi-Timeframe Confluence Pyramid (M23)
- 4-layer directional alignment: Macro → Direction (4H) → Setup (15m) → Trigger (5m)
- Mandatory full alignment before any trade proceeds
- Weakest-layer identification for TQS penalty attribution

### Funded Account Payout Management (M27)
- 5-phase payout lifecycle: APPROACH → ACCELERATION → FINAL_APPROACH → PAYOUT_TRIGGERED → RECOVERY
- Dynamic risk scaling (0.5x–1.1x) and TQS floor adjustment per phase
- Trade frequency limits that tighten as payout target approaches
- Graduation criteria: 3 consecutive payouts, <40% max DD, >80% win rate, >$150 expectancy

---

## Module Architecture

| Module | Name | Description |
|--------|------|-------------|
| M00–M15 | Core v5.0–v5.3 | Data feeds, regime detection, ICT signals, risk management, backtesting, news/sentiment, LoRA training |
| **M17** | Stop Calibration | Sweep analysis, optimal stops, hunt prediction, dynamic migration |
| **M18** | TP Calibration | Overshoot learning, session-aware multi-target take profit |
| **M19** | Entry Calibration | Weighted entry fusion (FVG + tick clusters + order book + historical PDF) |
| **M20** | Supreme Selectivity | 21-condition AND-logic pre-signal gate |
| **M21** | Trade Quality Score | 8-dimension 0–100 scoring with grade-based execution gating |
| **M22** | Footprint Engine | Institutional order flow detection (absorption, icebergs, stop raids) |
| **M23** | MTF Pyramid | 4-layer multi-timeframe directional alignment |
| **M24** | Weekly Profile | Monday displacement classification for weekly bias |
| **M25** | Liquidity Void | Gap and displacement candle detection |
| **M27** | Payout Engine | Funded account phase management with adaptive risk/TQS |
| **M28** | Ensemble LLM | Dual-model multi-temperature consensus voting |
| **M29** | Self-Calibration | Automated weekly parameter recalibration |

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Runtime** | Python 3.11+ | Async-first with `asyncio` and `asyncpg` |
| **Agent Dispatch** | Claude Agent SDK | Autonomous agent orchestration with custom MCP tools |
| **API** | FastAPI | REST API with automatic OpenAPI docs |
| **Database** | PostgreSQL | Trade history, calibration data, shadow signals |
| **Cache** | Redis | Real-time state, rate limiting, session data |
| **LLM (Cloud)** | MiniMax 2.5 API | Primary ensemble inference |
| **LLM (Local)** | Qwen 3.5-35B via MLX | On-device inference at 80+ tok/s on M4 Max |
| **LLM (Judge)** | Claude Sonnet | Independent adversarial validation |
| **Data Feed** | Databento | Real-time and historical futures market data |
| **Compute** | Apple M4 Max 128GB | Local ML inference, backtesting, training |
| **Monitoring** | Grafana + Prometheus | Real-time system and P&L dashboards |
| **Notifications** | Telegram Bot | Trade alerts, calibration reports, health checks |

---

## Project Structure

```
ghost/
├── config/
│   ├── settings.py              # Pydantic-validated configuration
│   └── instruments.json         # 30 CME instrument specifications
├── core/
│   ├── models.py                # Domain types, enums, dataclasses
│   └── orchestrator.py          # 10-gate signal processing pipeline
├── modules/
│   ├── m00_backtest/            # Walk-forward backtesting engine
│   ├── m01_data/                # Databento feed management
│   ├── m02_regime/              # Market regime detection
│   ├── m03_signal/              # ICT signal generation (FVG, OB, liquidity)
│   ├── m04_confluence/          # Multi-factor confluence scoring
│   ├── m05_risk/                # Position sizing and risk management
│   ├── m06_sizing/              # Kelly criterion and Monte Carlo sizing
│   ├── m07_execution/           # Order execution and management
│   ├── m08_monitor/             # Position monitoring and alerts
│   ├── m09_journal/             # Trade journaling and analytics
│   ├── m10_dashboard/           # Real-time web dashboard
│   ├── m11_llm/                 # LLM integration layer
│   ├── m12_news/                # RSS/FinBERT news sentiment
│   ├── m13_training/            # LoRA fine-tuning with EWC
│   ├── m14_quant/               # Quantitative analytics
│   ├── m15_benchmark/           # Performance benchmarking
│   ├── m17_stop_calibration/    # [v5.5] Adaptive stop loss system
│   ├── m18_tp_calibration/      # [v5.5] Take profit calibration
│   ├── m19_entry_calibration/   # [v5.5] Optimal entry calculation
│   ├── m20_selectivity/         # [v5.5] 21-condition checklist
│   ├── m21_tqs/                 # [v5.5] Trade Quality Score engine
│   ├── m22_footprint/           # [v5.5] Smart money detection
│   ├── m23_mtf_pyramid/         # [v5.5] Multi-timeframe pyramid
│   ├── m24_weekly_profile/      # [v5.5] Weekly bias classifier
│   ├── m25_liquidity_void/      # [v5.5] Liquidity gap detection
│   ├── m27_payout/              # [v5.5] Funded account payout mgr
│   ├── m28_ensemble/            # [v5.5] Ensemble LLM voting
│   └── m29_self_calibration/    # [v5.5] Weekly auto-recalibration
├── dispatch/
│   ├── __init__.py              # Dispatch module exports
│   ├── tools.py                 # 14 custom MCP tools for Claude agents
│   ├── agents.py                # 4 specialized agent definitions + dispatcher
│   └── cli.py                   # ghost-dispatch CLI entry point
├── db/
│   └── migrations/
│       └── v55_additions.sql    # v5.5 schema additions
├── tests/                       # pytest test suite
├── debug/
│   └── health.py                # Service health checker
├── api.py                       # FastAPI application
└── __main__.py                  # Entry point
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Apple Silicon Mac (recommended for local Qwen inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/DeanKrotzer1111/ghost.git
cd ghost

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run database migrations
psql -d ghost -f ghost/db/migrations/v55_additions.sql

# Start local Qwen inference (Apple Silicon)
mlx_lm.server --model mlx-community/Qwen3.5-35B-A3B-8bit --port 8081 --host 0.0.0.0

# Start Ghost
python -m ghost
```

### Service Ports
| Service | Port |
|---------|------|
| Ghost FastAPI | `:8080` |
| Qwen MLX Local | `:8081` |
| Dashboard | `:3000` |
| PostgreSQL | `:5432` |
| Redis | `:6379` |

---

## Version History

| Version | Highlights |
|---------|-----------|
| **v5.5** | TQS engine, 21-condition checklist, ensemble LLM, self-calibration, payout optimization, footprint detection |
| v5.4 | Monte Carlo Kelly, adversarial LLM, 6-family LoRA, EWC, breaker blocks, behavioral anti-patterns, session guardrails |
| v5.3 | Native ICT signal engine (FVG, OB, liquidity), entry precision, stop hunt evasion, MiniMax+Qwen dual LLM |
| v5.2 | Paper trading engine, Topstep/Apex account simulation, evaluation rule enforcement |
| v5.1 | Walk-forward backtesting, 80% gate, LoRA pipeline, Brier score benchmarking |
| v5.0 | RSS news, FinBERT sentiment, browser-use, Qwen news-forced regime detection |

---

## Instruments Covered

30 CME futures across 6 asset classes:

- **Equity Indices:** MNQ, NQ, ES, MES, RTY, M2K, YM, MYM
- **Metals:** GC, MGC, SI, SIL, HG, PL
- **Energy:** CL, MCL, NG, RB, HO
- **Forex:** 6E, 6B, 6J, 6A, 6C, M6E
- **Bonds:** ZB, ZN, ZF, ZT
- **Agriculture:** ZC, ZS, ZW

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*Built by Dean Krotzer — AI Engineer specializing in autonomous trading systems, multi-model AI architectures, and real-time financial data processing.*
