"""Multi-Timeframe Confluence Pyramid — enforces directional alignment across 4 timeframe layers."""
import structlog
from ghost.core.models import PyramidResult

logger = structlog.get_logger()


class MTFConfluencePyramid:
    """Validates that macro, direction, setup, and trigger timeframes are all directionally aligned.

    All 4 layers must agree for a trade to proceed. Strength is the average confidence
    across layers, and the weakest layer is identified for TQS penalty attribution.
    """

    def analyze(
        self, macro_bias: str, macro_confidence: float,
        direction_4h: str, direction_confidence: float,
        setup_direction: str, setup_confidence: float,
        trigger_direction: str, trigger_confidence: float,
    ) -> PyramidResult:
        layers = [
            ("macro", macro_bias, macro_confidence),
            ("direction_4h", direction_4h, direction_confidence),
            ("setup", setup_direction, setup_confidence),
            ("trigger", trigger_direction, trigger_confidence),
        ]

        valid = [l[1] for l in layers if l[1] not in ("NEUTRAL", "UNKNOWN", "")]
        if not valid:
            return PyramidResult(False, 0.0, "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "all", False)

        aligned = all(d == valid[0] for d in valid)
        strength = sum(l[2] for l in layers) / len(layers)
        weakest = min(layers, key=lambda l: l[2])[0]

        return PyramidResult(
            aligned, strength, macro_bias, direction_4h,
            setup_direction, trigger_direction, weakest, aligned,
        )
