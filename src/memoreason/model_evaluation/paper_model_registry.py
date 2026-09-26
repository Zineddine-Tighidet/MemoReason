"""Registry of the eight models reported in the MemoReason paper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from memoreason.factual_to_fictional_dataset.dataset_paths import DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class PaperModelConfiguration:
    """One frozen model configuration used by benchmark evaluation."""

    model_id: str
    provider: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 64
    seed: int | None = DEFAULT_RANDOM_SEED


PAPER_MODEL_CONFIGURATIONS: dict[str, PaperModelConfiguration] = {
    "olmo-3-7b-think": PaperModelConfiguration(
        model_id="olmo-3-7b-think",
        provider="local",
        model_name="allenai/Olmo-3-7B-Think",
        max_tokens=8192,
    ),
    "olmo-3-7b-instruct": PaperModelConfiguration(
        model_id="olmo-3-7b-instruct",
        provider="local",
        model_name="allenai/Olmo-3-7B-Instruct",
        max_tokens=8192,
    ),
    "gpt-oss-20b-groq": PaperModelConfiguration(
        model_id="gpt-oss-20b-groq",
        provider="groq",
        model_name="openai/gpt-oss-20b",
        max_tokens=10000,
    ),
    "gemma-4-26b-a4b-it": PaperModelConfiguration(
        model_id="gemma-4-26b-a4b-it",
        provider="local",
        model_name="google/gemma-4-26B-A4B-it",
        max_tokens=8192,
    ),
    "gpt-oss-120b-groq": PaperModelConfiguration(
        model_id="gpt-oss-120b-groq",
        provider="groq",
        model_name="openai/gpt-oss-120b",
        max_tokens=8192,
    ),
    "qwen3.5-27b": PaperModelConfiguration(
        model_id="qwen3.5-27b",
        provider="local",
        model_name="Qwen/Qwen3.5-27B",
        max_tokens=8192,
    ),
    "qwen3.5-35b-a3b": PaperModelConfiguration(
        model_id="qwen3.5-35b-a3b",
        provider="local",
        model_name="Qwen/Qwen3.5-35B-A3B",
        max_tokens=8192,
    ),
    "claude-sonnet-4-6": PaperModelConfiguration(
        model_id="claude-sonnet-4-6",
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        max_tokens=512,
        seed=None,
    ),
}


def resolve_paper_model_configurations(
    model_names: Sequence[str] | None = None,
    *,
    temperature: float | None = None,
    seed: int | None = None,
) -> list[PaperModelConfiguration]:
    """Return publication defaults or an explicitly requested evaluation model.

    Reviewer-extension models stay out of the no-argument paper default, but an
    explicit model id may select one.  This keeps frozen paper tables stable
    while allowing the same audited evaluation pipeline to run supplemental
    models such as Llama.
    """
    if temperature is not None and temperature < 0:
        raise ValueError("Generation temperature must be non-negative.")

    selected = list(PAPER_MODEL_CONFIGURATIONS.values()) if not model_names else []
    explicit_configurations = dict(PAPER_MODEL_CONFIGURATIONS)
    if model_names:
        # Imported lazily because reviewer_model_registry reuses the frozen
        # PaperModelConfiguration data class defined in this module.
        from .reviewer_model_registry import REVIEWER_MODEL_CONFIGURATIONS

        explicit_configurations.update(REVIEWER_MODEL_CONFIGURATIONS)

    model_configurations: list[PaperModelConfiguration] = []
    for model_name in model_names or ():
        if model_name not in explicit_configurations:
            available = ", ".join(explicit_configurations)
            raise KeyError(f"Unknown paper model {model_name!r}. Expected one of: {available}.")
        model_configurations.append(explicit_configurations[model_name])
    if model_names:
        selected = model_configurations
    return [
        replace(
            configuration,
            temperature=configuration.temperature if temperature is None else temperature,
            seed=configuration.seed if seed is None else seed,
        )
        for configuration in selected
    ]


__all__ = [
    "PAPER_MODEL_CONFIGURATIONS",
    "PaperModelConfiguration",
    "resolve_paper_model_configurations",
]
