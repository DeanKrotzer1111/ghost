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
                {
                    "role": "system",
                    "content": (
                        "You are a trading assistant. Analyze the trade setup and respond "
                        "with valid JSON: {\"approved\": bool, \"confidence\": float 0-1, "
                        "\"entry\": float, \"stop_loss\": float, \"reasoning\": string}"
                    ),
                },
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
            return self._parse_response(content, signal_id)

        except Exception as e:
            logger.error("MiniMax API call failed for signal %s: %s", signal_id, e)
            return LLMDecision(approved=False, confidence=0.0, reasoning=f"API error: {e}")

    def _parse_response(self, content: str, signal_id: str) -> LLMDecision:
        """Parse JSON from LLM response text."""
        try:
            # Try to extract JSON from the response
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
            logger.warning("Failed to parse MiniMax response for %s: %s", signal_id, e)

        return LLMDecision(approved=False, confidence=0.0, reasoning=f"Parse error: {content[:200]}")


class QwenClient(LLMClient):
    """Calls a local Qwen model at localhost:8081 for trade decisions."""

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
        """Call local Qwen and parse the decision."""
        import httpx

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a trading assistant. Analyze the trade setup and respond "
                        "with valid JSON: {\"approved\": bool, \"confidence\": float 0-1, "
                        "\"entry\": float, \"stop_loss\": float, \"reasoning\": string}"
                    ),
                },
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
            return self._parse_response(content, signal_id)

        except Exception as e:
            logger.error("Qwen API call failed for signal %s: %s", signal_id, e)
            return LLMDecision(approved=False, confidence=0.0, reasoning=f"API error: {e}")

    def _parse_response(self, content: str, signal_id: str) -> LLMDecision:
        """Parse JSON from LLM response text."""
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
            logger.warning("Failed to parse Qwen response for %s: %s", signal_id, e)

        return LLMDecision(approved=False, confidence=0.0, reasoning=f"Parse error: {content[:200]}")


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
