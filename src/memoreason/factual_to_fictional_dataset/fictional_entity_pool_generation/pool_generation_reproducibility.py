"""Reproducibility controls for provider-backed entity-pool generation."""

from __future__ import annotations

from datetime import UTC, datetime
import os


_GENERATION_TIMESTAMP_ENV = "MEMOREASON_GENERATION_TIMESTAMP_UTC"


def generation_timestamp_utc() -> str:
    """Return a fixed UTC timestamp when configured, otherwise the current UTC time."""
    configured = str(os.environ.get(_GENERATION_TIMESTAMP_ENV, "")).strip()
    if not configured:
        return datetime.now(UTC).isoformat()
    normalized = f"{configured[:-1]}+00:00" if configured.endswith(("Z", "z")) else configured
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{_GENERATION_TIMESTAMP_ENV} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{_GENERATION_TIMESTAMP_ENV} must include an explicit UTC offset")
    return parsed.astimezone(UTC).isoformat()
