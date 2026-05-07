"""Explicit Anthropic client for shared text generation."""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from typing import Any


@contextmanager
def _hard_timeout(seconds: float):
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise_timeout(_signum, _frame) -> None:
        raise TimeoutError(f"Anthropic request timed out after {seconds:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _raise_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


class AnthropicTextGenerationClient:
    """Small Anthropic wrapper used by the shared text-generation layer."""

    def __init__(self, *, api_key: str, timeout_seconds: float = 300.0) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Missing optional dependency 'anthropic' required for provider 'anthropic'. "
                "Install the provider SDKs with: uv sync --extra llm"
            ) from exc

        self._timeout_seconds = float(timeout_seconds)
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)

    def generate_raw_response(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
        }
        # Newer Claude Opus variants reject the legacy temperature field entirely.
        if not str(model).startswith("claude-opus-4-7"):
            request_kwargs["temperature"] = temperature
        with _hard_timeout(self._timeout_seconds):
            return self._client.messages.create(
                **request_kwargs,
            )

    @staticmethod
    def extract_text(response: Any) -> str:
        chunks: list[str] = []
        for block in getattr(response, "content", []):
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "".join(chunks).strip()
