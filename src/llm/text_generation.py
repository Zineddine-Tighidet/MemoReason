"""Shared text-generation entrypoint for Anthropic, Gemini, Groq, and local models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from .anthropic_client import AnthropicTextGenerationClient
from .gemini_client import GeminiTextGenerationClient
from .groq_client import GroqTextGenerationClient
from .local_client import LocalTextGenerationClient

_LOCAL_TEXT_CLIENT: LocalTextGenerationClient | None = None
_DOTENV_LOADED = False


@dataclass(frozen=True)
class TextGenerationRequest:
    """Normalized request payload for text generation."""

    provider: str
    model: str
    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = None


@dataclass(frozen=True)
class TextGenerationResult:
    """Normalized response payload returned by ``generate_text``."""

    provider: str
    model: str
    text: str
    reasoning_text: str
    raw_response: str


def _load_repo_dotenv() -> None:
    """Load simple KEY=VALUE pairs from the repository ``.env`` file once."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        _DOTENV_LOADED = True
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

    _DOTENV_LOADED = True


def _require_api_key(env_names: list[str], provider: str) -> str:
    _load_repo_dotenv()
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    joined = ", ".join(env_names)
    raise RuntimeError(f"Missing API key for provider '{provider}'. Expected one of: {joined}.")


def _serialize_response(response: Any) -> str:
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2)
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump(), indent=2, default=str)
    if hasattr(response, "dict"):
        return json.dumps(response.dict(), indent=2, default=str)
    if isinstance(response, (dict, list)):
        return json.dumps(response, indent=2, default=str)
    return repr(response)


def _generate_local_text(request: TextGenerationRequest) -> TextGenerationResult:
    global _LOCAL_TEXT_CLIENT

    if _LOCAL_TEXT_CLIENT is None:
        _LOCAL_TEXT_CLIENT = LocalTextGenerationClient()

    response = _LOCAL_TEXT_CLIENT.generate_response_payload(
        model=request.model,
        system_prompt=request.system_prompt,
        user_prompt=request.user_prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        seed=request.seed,
    )
    raw_payload = response.get("raw_api_response_json") or response.get("raw_api_response_repr") or ""
    if response.get("error"):
        raise RuntimeError(f"Local generation failed for {request.model}: {response['error']}")
    return TextGenerationResult(
        provider=request.provider,
        model=request.model,
        text=str(response.get("content") or "").strip(),
        reasoning_text=str(response.get("reasoning_content") or "").strip(),
        raw_response=str(raw_payload),
    )


def generate_text(request: TextGenerationRequest) -> TextGenerationResult:
    """Generate text with the configured Anthropic, Gemini, Groq, or local backend."""
    provider = request.provider.strip().lower()

    if provider == "anthropic":
        api_key = _require_api_key(["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"], provider)
        request_timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "300"))
        client = AnthropicTextGenerationClient(api_key=api_key, timeout_seconds=request_timeout)
        response = client.generate_raw_response(
            model=request.model,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return TextGenerationResult(
            provider=provider,
            model=request.model,
            text=client.extract_text(response),
            reasoning_text="",
            raw_response=_serialize_response(response),
        )

    if provider == "local":
        return _generate_local_text(request)

    if provider == "gemini":
        api_key = _require_api_key(["GEMINI_API_KEY", "GOOGLE_API_KEY"], provider)
        request_timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "300"))
        client = GeminiTextGenerationClient(api_key=api_key, timeout_seconds=request_timeout)
        response = client.generate_raw_response(
            model=request.model,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            seed=request.seed,
        )
        return TextGenerationResult(
            provider=provider,
            model=request.model,
            text=client.extract_text(response),
            reasoning_text=client.extract_reasoning(response),
            raw_response=_serialize_response(response),
        )

    if provider == "groq":
        api_key = _require_api_key(["GROQ_API_KEY"], provider)
        request_timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "300"))
        client = GroqTextGenerationClient(api_key=api_key, timeout_seconds=request_timeout)
        response = client.generate_raw_response(
            model=request.model,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            seed=request.seed,
        )
        return TextGenerationResult(
            provider=provider,
            model=request.model,
            text=client.extract_text(response),
            reasoning_text=client.extract_reasoning(response),
            raw_response=_serialize_response(response),
        )

    raise ValueError(
        f"Unsupported provider: {request.provider!r}. Supported providers: anthropic, gemini, groq, local."
    )


__all__ = [
    "TextGenerationRequest",
    "TextGenerationResult",
    "generate_text",
]
