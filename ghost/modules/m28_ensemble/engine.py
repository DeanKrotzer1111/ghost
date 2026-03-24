"""Ensemble LLM Voting System — dual-model (MiniMax + Qwen) multi-temperature consensus engine."""
import asyncio
import structlog
from ghost.core.models import EnsembleResult, EnsembleConsensus

logger = structlog.get_logger()

TEMPS = [0.10, 0.20, 0.30, 0.40, 0.50]


class EnsemblePredictionEngine:
    """Runs 10 parallel LLM inference calls (5 MiniMax + 5 Qwen at varied temperatures)
    and derives consensus via approval rate thresholds.

    Consensus levels:
        UNANIMOUS_APPROVE  = 100% combined approval
        STRONG_APPROVE     >= 80%
        WEAK_APPROVE       >= 60%
        UNCERTAIN          = 20-60%
        STRONG_REJECT      <= 20%
    """

    def __init__(self, minimax_client, qwen_client):
        self.minimax = minimax_client
        self.qwen = qwen_client

    async def run(self, prompt: str, signal_id: str = "") -> EnsembleResult:
        mm = await asyncio.gather(
            *[self.minimax.decide(prompt, temperature=t, signal_id=signal_id) for t in TEMPS],
            return_exceptions=True,
        )
        q = await asyncio.gather(
            *[self.qwen.decide(prompt, temperature=t, signal_id=signal_id) for t in TEMPS],
            return_exceptions=True,
        )

        mok = [r for r in mm if not isinstance(r, Exception) and r.approved]
        qok = [r for r in q if not isinstance(r, Exception) and r.approved]

        mr = len(mok) / len(TEMPS)
        qr = len(qok) / len(TEMPS)
        comb = (mr + qr) / 2.0

        if comb >= 1.00:
            con, ci = EnsembleConsensus.UNANIMOUS_APPROVE, (0.88, 0.96)
        elif comb >= 0.80:
            con, ci = EnsembleConsensus.STRONG_APPROVE, (0.78, 0.88)
        elif comb >= 0.60:
            con, ci = EnsembleConsensus.WEAK_APPROVE, (0.62, 0.75)
        elif comb <= 0.20:
            con, ci = EnsembleConsensus.STRONG_REJECT, (0.05, 0.20)
        else:
            con, ci = EnsembleConsensus.UNCERTAIN, (0.40, 0.65)

        ex = con in (EnsembleConsensus.UNANIMOUS_APPROVE, EnsembleConsensus.STRONG_APPROVE)
        ao = mok + qok

        return EnsembleResult(
            con, ex, ci[0], ci[1], (ci[0] + ci[1]) / 2.0, mr, qr,
            len(TEMPS) * 2,
            sum(getattr(d, "entry", 0.0) for d in ao) / len(ao) if ao else None,
            sum(getattr(d, "stop_loss", 0.0) for d in ao) / len(ao) if ao else None,
        )
