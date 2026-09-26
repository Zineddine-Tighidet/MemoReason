"""Shared text-generation entrypoint for Anthropic, Groq, and local models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any

from memoreason import PROJECT_ROOT_DIRECTORY

from .anthropic_client import AnthropicTextGenerationClient
from .groq_client import (
    GPT_OSS_MODEL_NAMES,
    GroqTextGenerationClient,
    gpt_oss_generation_config,
)
from .local_text_generation_client import LocalTextGenerationClient

_LOCAL_TEXT_CLIENT: LocalTextGenerationClient | None = None
_ANTHROPIC_CLIENT_STATE = threading.local()
_CACHE_LOCKS_GUARD = threading.Lock()
_CACHE_LOCKS: dict[str, threading.Lock] = {}
_DOTENV_LOADED = False
_CACHE_SCHEMA_VERSION = 2
_CACHE_CONTRACT = "memoreason_text_generation_cache_v2"


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

    env_path = PROJECT_ROOT_DIRECTORY / ".env"
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


def provider_generation_context(*, provider: str, model: str) -> dict[str, object]:
    """Return provider request parameters that affect reproducible generations."""
    _load_repo_dotenv()
    normalized_provider = str(provider).strip().lower()
    normalized_model = str(model).strip().lower()
    if normalized_provider == "groq" and normalized_model in GPT_OSS_MODEL_NAMES:
        return gpt_oss_generation_config()
    return {}


def _get_anthropic_client(
    *,
    api_key: str,
    timeout_seconds: float,
) -> AnthropicTextGenerationClient:
    """Reuse one HTTP client per worker thread and provider configuration."""
    configuration = (api_key, float(timeout_seconds))
    client = getattr(_ANTHROPIC_CLIENT_STATE, "client", None)
    if (
        client is not None
        and getattr(_ANTHROPIC_CLIENT_STATE, "configuration", None) == configuration
    ):
        return client

    replacement = AnthropicTextGenerationClient(
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    if client is not None:
        client.close()
    _ANTHROPIC_CLIENT_STATE.client = replacement
    _ANTHROPIC_CLIENT_STATE.configuration = configuration
    return replacement


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


def _request_cache_payload(request: TextGenerationRequest) -> dict[str, Any]:
    return {
        "provider": request.provider,
        "model": request.model,
        "system_prompt": request.system_prompt,
        "user_prompt": request.user_prompt,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "seed": request.seed,
        "provider_generation_context": provider_generation_context(
            provider=request.provider,
            model=request.model,
        ),
    }


def _payload_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _request_fingerprint(payload: dict[str, Any]) -> str:
    return _payload_sha256(payload)


def _result_cache_payload(result: TextGenerationResult) -> dict[str, str]:
    return {
        "provider": result.provider,
        "model": result.model,
        "text": result.text,
        "reasoning_text": result.reasoning_text,
        "raw_response": result.raw_response,
    }


def _generation_cache_configuration() -> tuple[Path | None, str]:
    raw_root = str(os.environ.get("MEMOREASON_TEXT_GENERATION_CACHE_DIR", "")).strip()
    default_mode = "read-write" if raw_root else "off"
    mode = str(os.environ.get("MEMOREASON_TEXT_GENERATION_CACHE_MODE", default_mode)).strip().lower()
    if mode not in {"off", "read-write", "replay-only"}:
        raise ValueError(
            "MEMOREASON_TEXT_GENERATION_CACHE_MODE must be one of: off, read-write, replay-only"
        )
    if mode != "off" and not raw_root:
        raise ValueError("MEMOREASON_TEXT_GENERATION_CACHE_DIR is required when cache mode is enabled")
    return (Path(raw_root).expanduser().resolve() if raw_root else None), mode


def _generation_cache_path(cache_root: Path, request: TextGenerationRequest, fingerprint: str) -> Path:
    provider = re.sub(r"[^A-Za-z0-9._-]+", "_", request.provider.strip().lower()) or "provider"
    model = re.sub(r"[^A-Za-z0-9._-]+", "_", request.model.strip()) or "model"
    return cache_root / provider / model / f"{fingerprint}.json"


def _result_from_cache(path: Path, request_payload: dict[str, Any], fingerprint: str) -> TextGenerationResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported text-generation cache schema: {path}")
    if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
        if payload.get("schema_version") == 1:
            raise RuntimeError(
                f"Legacy text-generation cache lacks a result checksum; regenerate it: {path}"
            )
        raise RuntimeError(f"Unsupported text-generation cache schema: {path}")
    if payload.get("contract") != _CACHE_CONTRACT:
        raise RuntimeError(f"Unsupported text-generation cache contract: {path}")
    if payload.get("request_sha256") != fingerprint or payload.get("request") != request_payload:
        raise RuntimeError(f"Text-generation cache request mismatch: {path}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"Text-generation cache result is invalid: {path}")
    result_sha256 = payload.get("result_sha256")
    if not isinstance(result_sha256, str) or not result_sha256:
        raise RuntimeError(f"Text-generation cache result checksum is missing: {path}")
    if result_sha256 != _payload_sha256(result):
        raise RuntimeError(f"Text-generation cache result checksum mismatch: {path}")
    return TextGenerationResult(
        provider=str(result.get("provider") or ""),
        model=str(result.get("model") or ""),
        text=str(result.get("text") or ""),
        reasoning_text=str(result.get("reasoning_text") or ""),
        raw_response=str(result.get("raw_response") or ""),
    )


def _write_generation_cache(
    path: Path,
    *,
    request_payload: dict[str, Any],
    fingerprint: str,
    result: TextGenerationResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result_payload = _result_cache_payload(result)
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "contract": _CACHE_CONTRACT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "request_sha256": fingerprint,
        "request": request_payload,
        "result_sha256": _payload_sha256(result_payload),
        "result": result_payload,
    }
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            cached = _result_from_cache(path, request_payload, fingerprint)
            if cached != result:
                raise RuntimeError(f"Conflicting text-generation cache result: {path}") from None
    finally:
        temporary_path.unlink(missing_ok=True)


def _generate_text_uncached(request: TextGenerationRequest) -> TextGenerationResult:
    """Generate one response without consulting the immutable response cache."""
    provider = request.provider.strip().lower()

    if provider == "anthropic":
        api_key = _require_api_key(["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"], provider)
        request_timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "300"))
        client = _get_anthropic_client(
            api_key=api_key,
            timeout_seconds=request_timeout,
        )
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

    raise ValueError(f"Unsupported provider: {request.provider!r}. Supported providers: anthropic, groq, local.")


def generate_text(request: TextGenerationRequest) -> TextGenerationResult:
    """Generate text, optionally replaying or recording an immutable response cassette."""
    cache_root, cache_mode = _generation_cache_configuration()
    if cache_mode == "off" or cache_root is None:
        return _generate_text_uncached(request)

    request_payload = _request_cache_payload(request)
    fingerprint = _request_fingerprint(request_payload)
    cache_path = _generation_cache_path(cache_root, request, fingerprint)
    lock_key = str(cache_path)
    with _CACHE_LOCKS_GUARD:
        cache_lock = _CACHE_LOCKS.setdefault(lock_key, threading.Lock())
    with cache_lock:
        if cache_path.is_file():
            return _result_from_cache(cache_path, request_payload, fingerprint)
        if cache_mode == "replay-only":
            raise FileNotFoundError(f"Text-generation replay cache miss: {cache_path}")

        result = _generate_text_uncached(request)
        _write_generation_cache(
            cache_path,
            request_payload=request_payload,
            fingerprint=fingerprint,
            result=result,
        )
        return result


__all__ = [
    "TextGenerationRequest",
    "TextGenerationResult",
    "generate_text",
]
