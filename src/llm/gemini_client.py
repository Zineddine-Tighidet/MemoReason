"""Explicit Gemini client for shared text generation."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

import requests


_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
_VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
_VALID_SAFETY_THRESHOLDS = {
    "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
    "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_NONE",
    "OFF",
}
_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
)
_REQUEST_PACING_LOCK = threading.Lock()
_NEXT_REQUEST_NOT_BEFORE = 0.0
_THREAD_LOCAL = threading.local()


def _thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "gemini_session", None)
    if isinstance(session, requests.Session):
        return session
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=256, pool_maxsize=256)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _THREAD_LOCAL.gemini_session = session
    return session


def _should_log_rate_limits() -> bool:
    value = str(os.environ.get("GEMINI_DEBUG_RATE_LIMIT_HEADERS", "")).strip().lower()
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
    print(f"[gemini-rate-limit] model={model} status={status_code} {compact}", file=sys.stderr, flush=True)


def _coerce_part_text(part: Any) -> str:
    if not isinstance(part, dict):
        return ""
    text = part.get("text")
    if text is None:
        return ""
    return str(text).strip()


class GeminiTextGenerationClient:
    """Small Gemini wrapper used by the shared text-generation layer."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 300.0,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._api_key = str(api_key).strip()
        self._timeout_seconds = float(timeout_seconds)
        self._base_url = base_url.rstrip("/")
        self._max_retries = self._read_max_retries()
        self._min_interval_seconds = self._read_min_interval_seconds()

    @staticmethod
    def _read_max_retries() -> int:
        raw_value = str(os.environ.get("GEMINI_MAX_RETRIES", "12")).strip()
        try:
            return max(0, int(raw_value))
        except ValueError:
            return 12

    @staticmethod
    def _read_min_interval_seconds() -> float:
        raw_value = str(os.environ.get("GEMINI_MIN_INTERVAL_SECONDS", "0")).strip()
        try:
            return max(0.0, float(raw_value))
        except ValueError:
            return 0.0

    @staticmethod
    def _retry_delay_seconds(*, attempt_index: int, headers: requests.structures.CaseInsensitiveDict[str]) -> float:
        retry_after = str(headers.get("retry-after") or "").strip()
        if retry_after:
            try:
                return max(0.05, float(retry_after) + 0.05)
            except ValueError:
                pass
        return min(30.0, 1.5 * (2 ** attempt_index))

    @staticmethod
    def _thinking_level_for_request(model: str) -> str:
        value = str(os.environ.get("GEMINI_THINKING_LEVEL", "low")).strip().lower()
        if value not in _VALID_THINKING_LEVELS:
            return "low"
        if value == "minimal" and "pro" in str(model).lower():
            return "low"
        return value

    @staticmethod
    def _include_thoughts() -> bool:
        value = str(os.environ.get("GEMINI_INCLUDE_THOUGHTS", "")).strip().lower()
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def _safety_settings() -> list[dict[str, str]]:
        threshold = str(os.environ.get("GEMINI_SAFETY_THRESHOLD", "")).strip().upper()
        if not threshold:
            return []
        if threshold not in _VALID_SAFETY_THRESHOLDS:
            return []
        return [{"category": category, "threshold": threshold} for category in _SAFETY_CATEGORIES]

    @classmethod
    def build_request_payload(
        cls,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
            "candidateCount": 1,
            "thinkingConfig": {
                "thinkingLevel": cls._thinking_level_for_request(model),
                "includeThoughts": cls._include_thoughts(),
            },
        }
        if seed is not None:
            generation_config["seed"] = int(seed)

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": generation_config,
        }
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        safety_settings = cls._safety_settings()
        if safety_settings:
            payload["safetySettings"] = safety_settings
        return payload

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
        payload = self.build_request_payload(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        url = f"{self._base_url}/models/{model}:generateContent"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "parametric-shortcut-quick-eval/0.1",
            "x-goog-api-key": self._api_key,
        }
        session = _thread_session()
        body = ""
        for attempt_index in range(self._max_retries + 1):
            try:
                self._wait_for_request_slot()
                response = session.post(
                    url,
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
                                headers=response.headers,
                            )
                        )
                        continue
                    raise RuntimeError(
                        f"Gemini request failed with HTTP {response.status_code}: {error_body}"
                    )
                body = response.text
                break
            except requests.RequestException as exc:
                if attempt_index < self._max_retries:
                    time.sleep(self._retry_delay_seconds(attempt_index=attempt_index, headers={}))
                    continue
                raise RuntimeError(f"Gemini request failed: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini returned non-JSON response: {body[:500]}") from exc

    @staticmethod
    def extract_text(response: Any) -> str:
        if not isinstance(response, dict):
            return ""
        candidates = response.get("candidates") or []
        if not candidates:
            return ""
        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
        chunks = [
            text
            for part in parts
            if not bool((part or {}).get("thought"))
            for text in [_coerce_part_text(part)]
            if text
        ]
        return "".join(chunks).strip()

    @staticmethod
    def extract_reasoning(response: Any) -> str:
        if not isinstance(response, dict):
            return ""
        candidates = response.get("candidates") or []
        if not candidates:
            return ""
        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
        chunks = [
            text
            for part in parts
            if bool((part or {}).get("thought"))
            for text in [_coerce_part_text(part)]
            if text
        ]
        return "\n".join(chunks).strip()
