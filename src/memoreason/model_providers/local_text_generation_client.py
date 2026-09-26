"""Explicit adapter from shared text generation to local model execution."""

from __future__ import annotations

from typing import Any

from memoreason.model_providers.local_models.local_model_inference_client import LocalModelInferenceClient
from memoreason.model_providers.local_models.model_config import LocalModelConfig


class LocalTextGenerationClient:
    """Thin adapter over ``LocalModelInferenceClient`` for the shared LLM layer."""

    def __init__(self) -> None:
        self._client = LocalModelInferenceClient()

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
