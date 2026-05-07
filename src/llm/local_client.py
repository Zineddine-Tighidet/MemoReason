"""Explicit adapter from shared text generation to local model execution."""

from __future__ import annotations

from typing import Any

from src.local_models.client import LocalLLMClient, LocalModelConfig


class LocalTextGenerationClient:
    """Thin adapter over ``LocalLLMClient`` for the shared LLM layer."""

    def __init__(self) -> None:
        self._client = LocalLLMClient()

    def generate_response_payload(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> dict[str, Any]:
        config = LocalModelConfig(
            name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        return self._client.generate(
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
