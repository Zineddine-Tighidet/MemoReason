"""Explicit Groq client for shared text generation."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import sys
from typing import Any

import requests


_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_VALID_GPT_OSS_REASONING_EFFORTS = {"low", "medium", "high"}
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
_RETRY_AFTER_SECONDS_RE = re.compile(r"Please try again in ([0-9]+(?:\.[0-9]+)?)(ms|s)", re.IGNORECASE)
_REQUEST_PACING_LOCK = threading.Lock()
_NEXT_REQUEST_NOT_BEFORE = 0.0
_THREAD_LOCAL = threading.local()


def _thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "groq_session", None)
    if isinstance(session, requests.Session):
        return session
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=256, pool_maxsize=256)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    setattr(_THREAD_LOCAL, "groq_session", session)
    return session


def _should_log_rate_limits() -> bool:
    value = str(os.environ.get("GROQ_DEBUG_RATE_LIMIT_HEADERS", "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _log_rate_limit_headers(*, status_code: int, headers: requests.structures.CaseInsensitiveDict[str], model: str) -> None:
    if not _should_log_rate_limits():
        return
    interesting = {
        key: value
        for key, value in headers.items()
        if key.lower().startswith("x-ratelimit") or key.lower() == "retry-after"
    }
    if not interesting:
        return
    compact = ", ".join(f"{key}={value}" for key, value in sorted(interesting.items(), key=lambda item: item[0].lower()))
    print(f"[groq-rate-limit] model={model} status={status_code} {compact}", file=sys.stderr, flush=True)


def _coerce_message_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    chunks.append(str(text).strip())
            elif item:
                chunks.append(str(item).strip())
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    if value is None:
        return ""
    return str(value).strip()


class GroqTextGenerationClient:
    """Small Groq wrapper used by the shared text-generation layer."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 300.0,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._api_key = str(api_key).strip()
        self._timeout_seconds = float(timeout_seconds)
        self._chat_completions_url = f"{base_url.rstrip('/')}/chat/completions"
        self._max_retries = self._read_max_retries()
        self._min_interval_seconds = self._read_min_interval_seconds()

    @staticmethod
    def _temperature_for_request(value: float) -> float:
        temperature = float(value)
        if temperature <= 0:
            return 1e-8
        return temperature

    @staticmethod
    def _gpt_oss_reasoning_effort() -> str:
        value = str(os.environ.get("GROQ_GPT_OSS_REASONING_EFFORT", "low")).strip().lower()
        return value if value in _VALID_GPT_OSS_REASONING_EFFORTS else "low"

    @staticmethod
    def _include_reasoning() -> bool:
        value = str(os.environ.get("GROQ_INCLUDE_REASONING", "")).strip().lower()
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_max_retries() -> int:
        raw_value = str(os.environ.get("GROQ_MAX_RETRIES", "12")).strip()
        try:
            return max(0, int(raw_value))
        except ValueError:
            return 12

    @staticmethod
    def _read_min_interval_seconds() -> float:
        raw_value = str(os.environ.get("GROQ_MIN_INTERVAL_SECONDS", "0")).strip()
        try:
            return max(0.0, float(raw_value))
        except ValueError:
            return 0.0

    @staticmethod
    def _retry_delay_seconds(*, attempt_index: int, error_body: str) -> float:
        match = _RETRY_AFTER_SECONDS_RE.search(error_body or "")
        if match:
            try:
                value = float(match.group(1))
                unit = match.group(2).lower()
                seconds = value / 1000.0 if unit == "ms" else value
                return max(0.05, seconds + 0.05)
            except ValueError:
                pass
        # Conservative capped exponential backoff.
        return min(30.0, 1.5 * (2 ** attempt_index))

    def _wait_for_request_slot(self) -> None:
        if self._min_interval_seconds <= 0:
            return
        global _NEXT_REQUEST_NOT_BEFORE
        while True:
            sleep_for = 0.0
            with _REQUEST_PACING_LOCK:
                now = time.monotonic()
                if now >= _NEXT_REQUEST_NOT_BEFORE:
                    _NEXT_REQUEST_NOT_BEFORE = now + self._min_interval_seconds
                    return
                sleep_for = _NEXT_REQUEST_NOT_BEFORE - now
            if sleep_for > 0:
                time.sleep(sleep_for)

    def generate_raw_response(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> Any:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self._temperature_for_request(temperature),
            "max_completion_tokens": int(max_tokens),
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = int(seed)

        if str(model).strip().lower() in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
            payload["reasoning_effort"] = self._gpt_oss_reasoning_effort()
            payload["include_reasoning"] = self._include_reasoning()

        session = _thread_session()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "parametric-shortcut-quick-eval/0.1",
        }
        body = ""
        for attempt_index in range(self._max_retries + 1):
            try:
                self._wait_for_request_slot()
                response = session.post(
                    self._chat_completions_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                _log_rate_limit_headers(status_code=response.status_code, headers=response.headers, model=model)
                if response.status_code >= 400:
                    error_body = response.text
                    if response.status_code in _RETRYABLE_HTTP_CODES and attempt_index < self._max_retries:
                        time.sleep(
                            self._retry_delay_seconds(
                                attempt_index=attempt_index,
                                error_body=error_body,
                            )
                        )
                        continue
                    raise RuntimeError(
                        f"Groq request failed with HTTP {response.status_code}: {error_body}"
                    )
                body = response.text
                break
            except requests.RequestException as exc:
                error_body = ""
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    error_body = exc.response.text
                if attempt_index < self._max_retries:
                    time.sleep(
                        self._retry_delay_seconds(attempt_index=attempt_index, error_body=error_body)
                    )
                    continue
                raise RuntimeError(f"Groq request failed: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Groq returned non-JSON response: {body[:500]}") from exc

    @staticmethod
    def extract_text(response: Any) -> str:
        if not isinstance(response, dict):
            return ""
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = (choices[0] or {}).get("message") or {}
        return _coerce_message_text(message.get("content"))

    @staticmethod
    def extract_reasoning(response: Any) -> str:
        if not isinstance(response, dict):
            return ""
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = (choices[0] or {}).get("message") or {}
        return _coerce_message_text(message.get("reasoning"))
