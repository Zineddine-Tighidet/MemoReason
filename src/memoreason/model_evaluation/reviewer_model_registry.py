"""Model configurations used by the rebuttal reviewer-model expansion.

These models are intentionally kept separate from the eight-model paper
registry: adding them to the paper default would silently change the scope of
the frozen tables and figures.
"""

from __future__ import annotations

from collections.abc import Sequence

from .paper_model_registry import PaperModelConfiguration


REVIEWER_MODEL_CONFIGURATIONS: dict[str, PaperModelConfiguration] = {
    "llama-3.1-8b-instruct": PaperModelConfiguration(
        model_id="llama-3.1-8b-instruct",
        provider="local",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=8192,
    ),
    "phi-4-mini-instruct": PaperModelConfiguration(
        model_id="phi-4-mini-instruct",
        provider="local",
        model_name="microsoft/Phi-4-mini-instruct",
        max_tokens=8192,
    ),
    "mistral-7b-instruct-v0.3": PaperModelConfiguration(
        model_id="mistral-7b-instruct-v0.3",
        provider="local",
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_tokens=64,
    ),
}


def resolve_reviewer_model_configurations(
    model_names: Sequence[str] | None = None,
) -> list[PaperModelConfiguration]:
    """Return the requested reviewer-model configurations in launch order."""
    selected = tuple(model_names) if model_names else tuple(REVIEWER_MODEL_CONFIGURATIONS)
    unknown = [name for name in selected if name not in REVIEWER_MODEL_CONFIGURATIONS]
    if unknown:
        available = ", ".join(REVIEWER_MODEL_CONFIGURATIONS)
        raise KeyError(f"Unknown reviewer model(s) {unknown!r}. Expected one of: {available}.")
    return [REVIEWER_MODEL_CONFIGURATIONS[name] for name in selected]


__all__ = [
    "REVIEWER_MODEL_CONFIGURATIONS",
    "resolve_reviewer_model_configurations",
]
