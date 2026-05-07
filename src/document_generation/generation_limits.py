"""Shared sampling bounds and generation limits for document generation."""

import math
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Named constants (formerly magic numbers scattered across functions)
# ---------------------------------------------------------------------------
_DEFAULT_NUMBER_MIN = 1
_DEFAULT_NUMBER_MAX = 200
_NUMBER_GENERATION_MAX_RETRIES = 40
# Most temporal generation stays capped at the paper's current-year horizon
# (2026), but future-dated source documents still need a wider safety envelope.
_MIN_YEAR = 1
_FUTURE_DATE_SAFETY_MAX_YEAR = 2100
_MIN_AGE = 6
_MAX_AGE = 100
_DEFAULT_UNCONSTRAINED_MIN_AGE = 18
_DEFAULT_UNCONSTRAINED_MAX_AGE = 90
_NUMBER_AND_AGE_RELATIVE_RANGE_RATIO = 0.20
_NUMBER_AND_AGE_RELATIVE_RANGE_MIN_DELTA = 1
_TEMPORAL_RELATIVE_RANGE_RATIO = 0.01
_TEMPORAL_RELATIVE_RANGE_MIN_DELTA = 1
_TEMPORAL_YEAR_RELATIVE_RANGE_HARD_CAP = 15
_CURRENT_YEAR_TEMPORAL_CAP = 2026
_CSP_EXACT_DOMAIN_LIMIT = 256
_CSP_MAX_SAMPLE_CANDIDATES = 384
MAX_MANUAL_ATTEMPTS = 40
MAX_MANUAL_BACKTRACK_STEPS = 20000
SLOW_SAMPLING_LOG_SECONDS = 10.0
SLOW_STAGE_LOG_SECONDS = 5.0
# Some reviewed templates carry dense but still solvable numeric systems in the
# low-to-mid 20s; allowing the exact CSP fallback there is slower but far more
# reliable than exhausting the retry sampler.
MAX_CSP_NUMBER_VARS = 32
MAX_NUMBER_CSP_STEPS = 200000
MAX_NUMBER_CSP_SECONDS = 2.0


def _relative_int_window(
    center: int,
    *,
    ratio: float = _NUMBER_AND_AGE_RELATIVE_RANGE_RATIO,
    min_delta: int = _NUMBER_AND_AGE_RELATIVE_RANGE_MIN_DELTA,
    small_value_threshold: Optional[int] = None,
    small_value_delta: Optional[int] = None,
    max_delta: Optional[int] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> Tuple[int, int]:
    """Return an integer window around ``center`` using ``center +/- ratio*|center|``.

    The window is inclusive and guaranteed to be non-empty after applying bounds.
    """
    delta = max(min_delta, int(math.ceil(abs(center) * ratio)))
    if small_value_threshold is not None and small_value_delta is not None and abs(center) < small_value_threshold:
        delta = max(delta, small_value_delta)
    if max_delta is not None:
        delta = min(delta, max_delta)
    low = center - delta
    high = center + delta

    if min_value is not None:
        low = max(low, min_value)
        high = max(high, min_value)
    if max_value is not None:
        low = min(low, max_value)
        high = min(high, max_value)

    if low > high:
        clamped = center
        if min_value is not None:
            clamped = max(clamped, min_value)
        if max_value is not None:
            clamped = min(clamped, max_value)
        low = high = clamped

    # Keep at least two candidates when possible to make "different from factual"
    # feasible without requiring full-generation retries.
    if low == high:
        if max_value is None or high < max_value:
            high += 1
        elif min_value is None or low > min_value:
            low -= 1

    return int(low), int(high)
