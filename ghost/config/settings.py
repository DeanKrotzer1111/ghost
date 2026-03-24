"""Ghost v5.5 Configuration — All settings with Pydantic validation."""
import os
from pydantic import Field
from pydantic_settings import BaseSettings


class GhostSettings(BaseSettings):
    """Central configuration for Ghost Trading System."""

    model_config = {"env_prefix": "GHOST_", "env_file": ".env"}

    # === Core Infrastructure ===
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)
    qwen_port: int = Field(default=8081)
    dashboard_port: int = Field(default=3000)

    # === Database ===
    database_url: str = Field(default="postgresql://ghost:ghost@localhost:5432/ghost")
    redis_url: str = Field(default="redis://localhost:6379/0")

    # === API Keys (loaded from environment) ===
    minimax_api_key: str = Field(default="")
    databento_api_key: str = Field(default="")
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")

    # === LLM Configuration ===
    minimax_model: str = Field(default="minimax-2.5")
    qwen_model: str = Field(default="mlx-community/Qwen3.5-35B-A3B-8bit")
    qwen_endpoint: str = Field(default="http://localhost:8081/v1")

    # === Risk Management ===
    max_daily_loss_pct: float = Field(default=0.03)
    max_position_risk_pct: float = Field(default=0.01)
    max_correlated_positions: int = Field(default=2)
    circuit_breaker_consecutive_losses: int = Field(default=3)

    # === v5.5 Additions (APPEND ONLY) ===
    tqs_minimum_execute: int = Field(default=85)
    tqs_minimum_shadow: int = Field(default=80)
    checklist_conditions_required: int = Field(default=21)
    sweep_buffer_fallback_ticks: int = Field(default=4)
    sweep_minimum_sample: int = Field(default=20)
    tp_overshoot_buffer_fallback_ticks: int = Field(default=6)
    tp_minimum_hit_rate_target: float = Field(default=0.70)
    self_calibration_day: str = Field(default="Sunday")
    self_calibration_hour_et: int = Field(default=22)
    weekly_profile_hour_et: int = Field(default=20)
    ensemble_temperatures: str = Field(default="0.1,0.2,0.3,0.4,0.5")
    ensemble_unanimous_only: bool = Field(default=False)
    ensemble_strong_approve_minimum: float = Field(default=0.80)
    payout_target_dollars: float = Field(default=3000.0)
    payout_approach_tqs: int = Field(default=85)
    payout_acceleration_tqs: int = Field(default=87)
    payout_final_approach_tqs: int = Field(default=90)
    payout_triggered_tqs: int = Field(default=92)
    payout_recovery_tqs: int = Field(default=88)
    weekly_profile_enabled: bool = Field(default=True)
    weekly_profile_bias_penalty_tqs: int = Field(default=15)
    void_detection_enabled: bool = Field(default=True)
    void_tqs_bonus: int = Field(default=8)
    shadow_log_enabled: bool = Field(default=True)


settings = GhostSettings()
