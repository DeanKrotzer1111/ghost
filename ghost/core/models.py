"""Ghost v5.5 Core Data Models — All domain types and enums."""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict


# === Enums ===

class TQSGrade(Enum):
    PREMIUM = "PREMIUM"
    STANDARD = "STANDARD"
    SHADOW = "SHADOW"
    REJECT = "REJECT"


class PayoutPhase(Enum):
    APPROACH = "APPROACH"
    ACCELERATION = "ACCELERATION"
    FINAL_APPROACH = "FINAL_APPROACH"
    PAYOUT_TRIGGERED = "PAYOUT_TRIGGERED"
    RECOVERY = "RECOVERY"


class EnsembleConsensus(Enum):
    UNANIMOUS_APPROVE = "UNANIMOUS_APPROVE"
    STRONG_APPROVE = "STRONG_APPROVE"
    WEAK_APPROVE = "WEAK_APPROVE"
    UNCERTAIN = "UNCERTAIN"
    STRONG_REJECT = "STRONG_REJECT"


# === Data Classes ===

@dataclass
class TQSResult:
    total: float
    grade: TQSGrade
    dimension_scores: Dict[str, float]
    execute: bool
    shadow_log: bool
    weakest_dimension: str
    rejection_reason: Optional[str] = None


@dataclass
class ChecklistResult:
    passed: bool
    conditions_passed: int
    conditions_total: int
    failed_conditions: List[str]
    failure_tags: List[str]


@dataclass
class FootprintScore:
    composite: float
    signals_detected: int
    dominant_signal: str
    absorption_detected: bool
    iceberg_detected: bool
    stop_raid_detected: bool
    divergence_detected: bool
    bid_imbalance_ratio: float = 0.0


@dataclass
class PyramidResult:
    complete: bool
    strength: float
    macro_direction: str
    direction_layer: str
    setup_layer: str
    trigger_layer: str
    weakest_layer: str
    aligned: bool


@dataclass
class WeeklyProfile:
    profile_type: str
    bias: str
    weekly_draw: str
    confidence: float
    monday_displacement: Optional[str] = None


@dataclass
class LiquidityVoid:
    high: float
    low: float
    size_ticks: int
    direction: str
    filled: bool = False
    formation_time: Optional[float] = None


@dataclass
class OptimalEntry:
    price: float
    sources: List[str]
    confidence: float
    cluster_price: Optional[float] = None
    book_price: Optional[float] = None
    ob_mitigation_price: Optional[float] = None


@dataclass
class StopLevel:
    price: float
    structural_level: float
    sweep_buffer_ticks: int
    requires_size_reduction: bool
    size_multiplier: float = 1.0
    hunt_predicted: bool = False
    vix_multiplier: float = 1.0


@dataclass
class TPLevels:
    tp1: float
    tp2: float
    tp3: float
    confidence_tp1: float
    confidence_tp2: float
    confidence_tp3: float
    exit_pct_tp1: float = 0.50
    exit_pct_tp2: float = 0.30
    exit_pct_tp3: float = 0.20
    void_between_entry_tp1: bool = False
    session_modifier: float = 1.0


@dataclass
class EnsembleResult:
    consensus: EnsembleConsensus
    execute: bool
    confidence_lower: float
    confidence_upper: float
    point_confidence: float
    minimax_approve_rate: float
    qwen_approve_rate: float
    total_decisions: int = 10
    avg_entry: Optional[float] = None
    avg_stop: Optional[float] = None


@dataclass
class PayoutState:
    current_pnl: float
    payout_target: float
    phase: PayoutPhase
    days_to_payout_estimate: int
    risk_multiplier: float
    tqs_minimum: int
    max_trades_today: int
    pct_to_target: float = 0.0


@dataclass
class ProcessResult:
    accepted: bool
    rejection_reason: Optional[str] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    tqs: Optional[TQSResult] = None
    checklist: Optional[ChecklistResult] = None
    footprint: Optional[FootprintScore] = None
    pyramid: Optional[PyramidResult] = None
    ensemble: Optional[EnsembleResult] = None
    payout_state: Optional[PayoutState] = None
    void_present: bool = False
    hunt_predicted: bool = False
    size_multiplier: float = 1.0
