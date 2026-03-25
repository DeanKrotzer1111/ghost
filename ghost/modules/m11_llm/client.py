"""M11 LLM Client — Interfaces with LLM APIs for trade decision support."""
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMDecision:
    """Decision returned by an LLM after evaluating a trade signal."""
    approved: bool = False
    confidence: float = 0.0
    entry: float = 0.0
    stop_loss: float = 0.0
    reasoning: str = ""


class LLMClient(ABC):
    """Base class for LLM-based trade decision clients."""

    @abstractmethod
    async def decide(
        self,
        prompt: str,
        temperature: float = 0.3,
        signal_id: str = "",
    ) -> LLMDecision:
        """Send a prompt to the LLM and get a trade decision back."""
        ...


SYSTEM_PROMPT = (
    "You are a trading assistant for futures markets. Analyze the trade setup and respond "
    "with valid JSON only, no other text: "
    '{"approved": bool, "confidence": float 0-1, "entry": float, "stop_loss": float, "reasoning": string}'
)


def _parse_llm_json(content: str, signal_id: str, source: str) -> LLMDecision:
    """Parse JSON from LLM response text. Shared by MinimaxClient and QwenClient."""
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
            return LLMDecision(
                approved=bool(parsed.get("approved", False)),
                confidence=float(parsed.get("confidence", 0.0)),
                entry=float(parsed.get("entry", 0.0)),
                stop_loss=float(parsed.get("stop_loss", 0.0)),
                reasoning=str(parsed.get("reasoning", "")),
            )
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse %s response for %s: %s", source, signal_id, e)

    return LLMDecision(approved=False, confidence=0.0, reasoning=f"Parse error: {content[:200]}")


class MinimaxClient(LLMClient):
    """Calls the MiniMax API for trade decisions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.minimaxi.chat/v1/text/chatcompletion_v2",
        model: str = "MiniMax-Text-01",
    ):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = base_url
        self.model = model

    async def decide(
        self,
        prompt: str,
        temperature: float = 0.3,
        signal_id: str = "",
    ) -> LLMDecision:
        """Call MiniMax API and parse the decision from the response."""
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.base_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return _parse_llm_json(content, signal_id, "MiniMax")

        except Exception as e:
            logger.error("MiniMax API call failed for signal %s: %s", signal_id, e)
            return LLMDecision(approved=False, confidence=0.0, reasoning=f"API error: {e}")


class QwenClient(LLMClient):
    """Calls a local Qwen MLX server (OpenAI-compatible) for trade decisions."""

    def __init__(
        self,
        base_url: str = "http://localhost:8081/v1/chat/completions",
        model: str = "qwen",
    ):
        self.base_url = base_url
        self.model = model

    async def decide(
        self,
        prompt: str,
        temperature: float = 0.3,
        signal_id: str = "",
    ) -> LLMDecision:
        """Call local Qwen MLX server and parse the decision."""
        import httpx

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(self.base_url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return _parse_llm_json(content, signal_id, "Qwen")

        except Exception as e:
            logger.error("Qwen API call failed for signal %s: %s", signal_id, e)
            return LLMDecision(approved=False, confidence=0.0, reasoning=f"API error: {e}")


class BacktestLLMClient(LLMClient):
    """LLM client for backtesting that scores trade context deterministically.

    Instead of calling an external API, it evaluates the trade setup based on
    regime alignment, FVG quality, and kill zone activity to produce a
    consistent, reproducible decision.
    """

    async def decide(
        self,
        prompt: str,
        temperature: float = 0.3,
        signal_id: str = "",
    ) -> LLMDecision:
        """Score trade context from the prompt and return a deterministic decision."""
        prompt_lower = prompt.lower()
        confidence = 0.40
        reasons = []

        # Check if regime is trending in trade direction
        trending_bullish = "trending_bullish" in prompt_lower or "trending bullish" in prompt_lower
        trending_bearish = "trending_bearish" in prompt_lower or "trending bearish" in prompt_lower
        direction_bullish = "direction: bullish" in prompt_lower or "direction:bullish" in prompt_lower or '"bullish"' in prompt_lower
        direction_bearish = "direction: bearish" in prompt_lower or "direction:bearish" in prompt_lower or '"bearish"' in prompt_lower

        regime_aligned = (
            (trending_bullish and direction_bullish) or
            (trending_bearish and direction_bearish)
        )
        if regime_aligned:
            confidence = 0.85
            reasons.append("regime trending in trade direction")

        # Check FVG quality
        fvg_high_quality = any(kw in prompt_lower for kw in [
            "fvg_unmitigated", "fvg unmitigated", "fvg_4h", "fvg 4h",
            "fvg_1h", "fvg 1h", "high quality fvg", "strong fvg",
        ])
        if fvg_high_quality:
            confidence += 0.05
            reasons.append("high quality FVG detected")

        # Check kill zone activity
        kz_active = any(kw in prompt_lower for kw in [
            "kill_zone_active", "kill zone active", "kill_zone: true",
            "ny session", "london session", "kz:ny", "kz:london",
        ])
        if kz_active:
            confidence += 0.05
            reasons.append("kill zone active")

        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        approved = confidence >= 0.60

        reasoning = f"BacktestLLM: {'approved' if approved else 'rejected'}"
        if reasons:
            reasoning += " — " + ", ".join(reasons)
        else:
            reasoning += " — no strong alignment signals found"

        logger.debug(
            "BacktestLLM decision for signal %s: approved=%s, confidence=%.2f",
            signal_id, approved, confidence,
        )

        return LLMDecision(
            approved=approved,
            confidence=round(confidence, 2),
            entry=0.0,
            stop_loss=0.0,
            reasoning=reasoning,
        )


class MockLLMClient(LLMClient):
    """Mock LLM client for backtesting. Always approves with fixed confidence."""

    def __init__(self, confidence: float = 0.85, approved: bool = True):
        self.confidence = confidence
        self.approved = approved

    async def decide(
        self,
        prompt: str,
        temperature: float = 0.3,
        signal_id: str = "",
    ) -> LLMDecision:
        """Return a fixed mock decision for backtesting."""
        logger.debug("MockLLM decision for signal %s: approved=%s, confidence=%.2f",
                      signal_id, self.approved, self.confidence)
        return LLMDecision(
            approved=self.approved,
            confidence=self.confidence,
            entry=0.0,       # Use signal's entry
            stop_loss=0.0,   # Use signal's stop
            reasoning="Mock LLM: auto-approved for backtesting",
        )
