"""Structured YAML/JSON generation helpers for LLM-backed annotation steps."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable

import yaml

from .text_generation import TextGenerationRequest, TextGenerationResult, generate_text


_FENCED_BLOCK_PATTERN = re.compile(r"```(?:yaml|yml|json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class StructuredGenerationAttempt:
    """One structured-generation attempt and its validation outcome."""

    attempt_index: int
    text: str
    error: str | None = None


@dataclass(frozen=True)
class StructuredGenerationResult:
    """Parsed structured payload with traceable raw attempts."""

    payload: dict[str, Any]
    response: TextGenerationResult
    attempts: tuple[StructuredGenerationAttempt, ...]


def parse_yaml_or_json_mapping(text: str) -> dict[str, Any]:
    """Parse a mapping from raw LLM text, fenced YAML, or fenced JSON."""
    candidates = [match.group(1).strip() for match in _FENCED_BLOCK_PATTERN.finditer(text or "")]
    candidates.append(str(text or "").strip())
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        for loader_name, loader in (("json", json.loads), ("yaml", yaml.safe_load)):
            try:
                parsed = loader(candidate)
            except Exception as exc:  # noqa: BLE001 - error is surfaced to retry prompt.
                errors.append(f"{loader_name}: {exc}")
                continue
            if isinstance(parsed, dict):
                return parsed
            errors.append(f"{loader_name}: top-level payload was {type(parsed).__name__}, expected mapping")
    rendered_errors = "; ".join(errors[-4:]) if errors else "no parseable content"
    raise ValueError(f"Could not parse a YAML/JSON mapping from model output: {rendered_errors}")


def generate_structured_mapping(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    validator: Callable[[dict[str, Any]], None] | None = None,
    max_attempts: int = 3,
    max_tokens: int = 4096,
) -> StructuredGenerationResult:
    """Generate a structured mapping, retrying with validation feedback."""
    if int(max_attempts) < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}.")

    attempts: list[StructuredGenerationAttempt] = []
    prompt = user_prompt
    last_response: TextGenerationResult | None = None
    for attempt_index in range(1, int(max_attempts) + 1):
        response = generate_text(
            TextGenerationRequest(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=max_tokens,
                seed=None,
            )
        )
        last_response = response
        try:
            payload = parse_yaml_or_json_mapping(response.text)
            if validator is not None:
                validator(payload)
            attempts.append(StructuredGenerationAttempt(attempt_index=attempt_index, text=response.text))
            return StructuredGenerationResult(payload=payload, response=response, attempts=tuple(attempts))
        except Exception as exc:  # noqa: BLE001 - validation feedback is useful for LLM repair.
            error = str(exc)
            attempts.append(
                StructuredGenerationAttempt(
                    attempt_index=attempt_index,
                    text=response.text,
                    error=error,
                )
            )
            prompt = (
                f"{user_prompt}\n\n"
                "Your previous response could not be used.\n"
                f"Validation error:\n{error}\n\n"
                "Fix only the invalid fields. If an entity_ref was invalid because it is a number_* "
                "or temporal_* reference, remove it or replace it with a named entity from the source. "
                "Use an empty associated_entities list when no named entity is available.\n"
                "Return only the corrected YAML payload with the requested schema."
            )

    raise RuntimeError(
        "Structured generation failed after "
        f"{max_attempts} attempts: {attempts[-1].error if attempts else 'unknown error'}"
    ) from None


__all__ = [
    "StructuredGenerationAttempt",
    "StructuredGenerationResult",
    "generate_structured_mapping",
    "parse_yaml_or_json_mapping",
]
