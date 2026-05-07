"""Registry of the model configurations reported in the paper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.dataset_export.dataset_paths import DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class EvaluatedModelSpec:
    """One model configuration for benchmark evaluation."""

    registry_name: str
    provider: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 64
    seed: int | None = DEFAULT_RANDOM_SEED


DEFAULT_MODEL_REGISTRY: dict[str, EvaluatedModelSpec] = {
    "olmo-3-7b-think": EvaluatedModelSpec(
        registry_name="olmo-3-7b-think",
        provider="local",
        model_name="allenai/Olmo-3-7B-Think",
        max_tokens=1024,
    ),
    "olmo-3-7b-instruct": EvaluatedModelSpec(
        registry_name="olmo-3-7b-instruct",
        provider="local",
        model_name="allenai/Olmo-3-7B-Instruct",
        max_tokens=8192,
    ),
    "gpt-oss-20b-groq": EvaluatedModelSpec(
        registry_name="gpt-oss-20b-groq",
        provider="groq",
        model_name="openai/gpt-oss-20b",
        max_tokens=10000,
    ),
    "gemma-4-26b-a4b-it": EvaluatedModelSpec(
        registry_name="gemma-4-26b-a4b-it",
        provider="local",
        model_name="google/gemma-4-26B-A4B-it",
        max_tokens=8192,
    ),
    "gpt-oss-120b-groq": EvaluatedModelSpec(
        registry_name="gpt-oss-120b-groq",
        provider="groq",
        model_name="openai/gpt-oss-120b",
        max_tokens=8192,
    ),
    "qwen3.5-27b": EvaluatedModelSpec(
        registry_name="qwen3.5-27b",
        provider="local",
        model_name="Qwen/Qwen3.5-27B",
        max_tokens=8192,
    ),
    "qwen3.5-35b-a3b": EvaluatedModelSpec(
        registry_name="qwen3.5-35b-a3b",
        provider="local",
        model_name="Qwen/Qwen3.5-35B-A3B",
        max_tokens=8192,
    ),
    "claude-sonnet-4-6": EvaluatedModelSpec(
        registry_name="claude-sonnet-4-6",
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        max_tokens=512,
        seed=None,
    ),
}


def resolve_model_specs(model_names: Sequence[str] | None = None) -> list[EvaluatedModelSpec]:
    """Return the requested model specs from the registry."""
    if not model_names:
        return list(DEFAULT_MODEL_REGISTRY.values())

    resolved_specs: list[EvaluatedModelSpec] = []
    for model_name in model_names:
        if model_name not in DEFAULT_MODEL_REGISTRY:
            known = ", ".join(sorted(DEFAULT_MODEL_REGISTRY))
            raise KeyError(f"Unknown model {model_name!r}. Public registry entries: {known}.")
        resolved_specs.append(DEFAULT_MODEL_REGISTRY[model_name])
    return resolved_specs
