"""M13 Training Manager — Manages model training and evaluation cycles."""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    model_name: str = "ghost_default"
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2
    lookback_days: int = 90


@dataclass
class TrainingResult:
    """Result from a training or evaluation run."""
    model_name: str = ""
    metric_name: str = ""
    metric_value: float = 0.0
    timestamp: float = 0.0
    notes: str = ""


class TrainingManager:
    """Manages model training and evaluation. In backtesting mode, logs that
    training would occur without actually running any ML training loops."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.training_history: list[TrainingResult] = []
        self.is_backtesting: bool = True

    def train(
        self,
        training_data: Any = None,
        validation_data: Any = None,
        model_name: Optional[str] = None,
    ) -> TrainingResult:
        """Run a training cycle. In backtesting mode, logs the intent and
        records a placeholder result.

        Args:
            training_data: Training dataset (format depends on model).
            validation_data: Validation dataset.
            model_name: Override model name from config.

        Returns:
            TrainingResult with metadata about the training run.
        """
        name = model_name or self.config.model_name
        now = time.time()

        if self.is_backtesting:
            logger.info(
                "Training would occur for model '%s' with config: epochs=%d, lr=%.4f, batch=%d",
                name, self.config.epochs, self.config.learning_rate, self.config.batch_size,
            )
            data_size = len(training_data) if training_data is not None and hasattr(training_data, '__len__') else 0
            logger.info("Training data size: %d samples", data_size)

            result = TrainingResult(
                model_name=name,
                metric_name="backtest_placeholder",
                metric_value=0.0,
                timestamp=now,
                notes=f"Backtesting mode: training skipped. Data size={data_size}",
            )
        else:
            logger.info("Starting live training for model '%s'...", name)
            result = TrainingResult(
                model_name=name,
                metric_name="loss",
                metric_value=0.0,
                timestamp=now,
                notes="Live training not yet implemented.",
            )

        self.training_history.append(result)
        return result

    def evaluate(
        self,
        test_data: Any = None,
        model_name: Optional[str] = None,
    ) -> TrainingResult:
        """Evaluate a trained model. In backtesting mode, logs the intent.

        Args:
            test_data: Test dataset for evaluation.
            model_name: Which model to evaluate.

        Returns:
            TrainingResult with evaluation metrics.
        """
        name = model_name or self.config.model_name
        now = time.time()

        if self.is_backtesting:
            data_size = len(test_data) if test_data is not None and hasattr(test_data, '__len__') else 0
            logger.info(
                "Evaluation would occur for model '%s' on %d samples.",
                name, data_size,
            )
            result = TrainingResult(
                model_name=name,
                metric_name="backtest_eval_placeholder",
                metric_value=0.0,
                timestamp=now,
                notes=f"Backtesting mode: evaluation skipped. Data size={data_size}",
            )
        else:
            logger.info("Starting live evaluation for model '%s'...", name)
            result = TrainingResult(
                model_name=name,
                metric_name="accuracy",
                metric_value=0.0,
                timestamp=now,
                notes="Live evaluation not yet implemented.",
            )

        self.training_history.append(result)
        return result

    def get_history(self) -> list[TrainingResult]:
        """Return the full training/evaluation history."""
        return list(self.training_history)

    def reset(self) -> None:
        """Clear training history."""
        self.training_history.clear()
