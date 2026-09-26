"""Exact-match and judge-based scoring for parsed model outputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time

from memoreason.model_providers.text_generation import TextGenerationRequest, generate_text
from memoreason.factual_to_fictional_dataset.dataset_paths import DEFAULT_RANDOM_SEED
from .document_question_answering_prompt import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from .schema_aware_answer_matching import (
    score_canonical_prediction,
    score_prediction_with_schema,
)


SCORING_PROTOCOL_VERSION = "exact_match_or_judge_all_misses_v1"
JUDGE_MIN_MAX_TOKENS = 64
JUDGE_MAX_ATTEMPTS = 4
JUDGE_TOKEN_GROWTH_FACTOR = 2
JUDGE_RETRY_DELAY_SECONDS = 0.5


@dataclass(frozen=True)
class JudgeMatchConfiguration:
    """Configuration for the optional LLM-as-a-judge pass."""

    provider: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 8
    seed: int | None = DEFAULT_RANDOM_SEED


def exact_match_is_correct(predicted_answer: str, ground_truth: str) -> bool:
    """Backward-compatible two-string scorer for older call sites."""
    return score_canonical_prediction(predicted_answer, (ground_truth,))


def accepted_answer_match_is_correct(
    predicted_canonical: str,
    accepted_answers_canonical: tuple[str, ...],
    *,
    answer_schema: str | None = None,
    raw_prediction: str | None = None,
) -> bool:
    """Apply deterministic exact match against the accepted canonical answer set."""
    if answer_schema:
        return score_prediction_with_schema(
            predicted_canonical,
            accepted_answers_canonical,
            answer_schema=answer_schema,
            raw_prediction=raw_prediction,
        )
    return score_canonical_prediction(predicted_canonical, accepted_answers_canonical)


def judge_match_is_allowed(
    *,
    answer_schema: str | None,
    parsed_output_canonical: str | None,
    raw_prediction: str | None = None,
) -> bool:
    """Return whether a non-exact, non-empty prediction should be judged.

    Exact-match successes do not need LLM-as-a-judge. Non-empty misses do,
    including answers whose canonical parser output is empty but whose raw
    response still contains a meaningful prediction. A genuinely empty final
    response is a model failure and must remain deterministically incorrect.
    """
    del answer_schema
    return bool(str(parsed_output_canonical or "").strip() or str(raw_prediction or "").strip())


def parse_judge_verdict(text: str) -> bool | None:
    """Parse the protocol's CORRECT/INCORRECT verdict from provider output."""
    normalized = re.sub(r"\s+", " ", str(text or "").strip().upper())
    trimmed = normalized.lstrip(" `\"'([{:-")
    if trimmed.startswith("INCORRECT"):
        return False
    if trimmed.startswith("CORRECT"):
        return True
    match = re.search(r"\bVERDICT\s*[:=-]\s*(INCORRECT|CORRECT)\b", normalized)
    if not match:
        return None
    return match.group(1) == "CORRECT"


def judge_prediction(
    *,
    question_text: str,
    ground_truth: str,
    predicted_answer: str,
    judge_config: JudgeMatchConfiguration,
) -> tuple[bool, str]:
    """Run the judge model and return ``(is_correct, raw_judge_output)``."""
    last_raw_output = ""
    for attempt_index in range(JUDGE_MAX_ATTEMPTS):
        response = generate_text(
            TextGenerationRequest(
                provider=judge_config.provider,
                model=judge_config.model_name,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=build_judge_prompt(question_text, ground_truth, predicted_answer),
                temperature=judge_config.temperature,
                max_tokens=max(int(judge_config.max_tokens), JUDGE_MIN_MAX_TOKENS)
                * (JUDGE_TOKEN_GROWTH_FACTOR**attempt_index),
                seed=judge_config.seed,
            )
        )
        last_raw_output = response.text or response.reasoning_text or response.raw_response
        for candidate_text in (response.text, response.reasoning_text):
            verdict = parse_judge_verdict(candidate_text)
            if verdict is not None:
                return verdict, response.text or response.reasoning_text
        if attempt_index < JUDGE_MAX_ATTEMPTS - 1:
            time.sleep(JUDGE_RETRY_DELAY_SECONDS * (attempt_index + 1))
    raise ValueError(f"Judge response is not parseable as CORRECT/INCORRECT: {last_raw_output!r}")
