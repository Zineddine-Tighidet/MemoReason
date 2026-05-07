"""
Answer scoring utilities.

This module centralizes normalization and exact-match logic so evaluation code
does not carry scoring-specific legacy branches.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Any

from .annotation_runtime import find_entity_refs
from .entity_taxonomy import WORD_TO_NUMBER

# Named constants for scoring thresholds (formerly magic numbers).
_LENGTH_RATIO_THRESHOLD = 0.995
_MAX_EXTRA_SUFFIX_LENGTH = 25
_UNICODE_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2043": "-",
        "\ufe58": "-",
        "\ufe63": "-",
        "\uff0d": "-",
    }
)

# String version of WORD_TO_NUMBER for answer normalization, plus "thousand".
_WORD_TO_NUMBER_STR: dict[str, str] = {k: str(v) for k, v in WORD_TO_NUMBER.items()}
_WORD_TO_NUMBER_STR["thousand"] = "1000"


def _resolve_alias_pairs(answer: str) -> str:
    alias_pairs = [
        # Countries
        ("usa", "united states of america"),
        ("us", "united states"),
        ("u.s.", "united states"),
        ("u.s.a.", "united states of america"),
        ("uk", "united kingdom"),
        ("u.k.", "united kingdom"),
        ("uae", "united arab emirates"),
        ("drc", "democratic republic of congo"),
        ("prc", "peoples republic of china"),
        # Organizations
        ("mit", "massachusetts institute of technology"),
        ("m.i.t.", "massachusetts institute of technology"),
        ("nasa", "national aeronautics and space administration"),
        ("nato", "north atlantic treaty organization"),
        ("un", "united nations"),
        ("eu", "european union"),
        ("fbi", "federal bureau of investigation"),
        ("cia", "central intelligence agency"),
        ("nfl", "national football league"),
        ("nba", "national basketball association"),
        ("nyc", "new york city"),
        ("la", "los angeles"),
        ("sf", "san francisco"),
        ("dc", "district of columbia"),
        ("bbc", "british broadcasting corporation"),
        ("cnn", "cable news network"),
        ("nypd", "new york police department"),
        ("lapd", "los angeles police department"),
        ("irs", "internal revenue service"),
        ("doj", "department of justice"),
    ]
    alias_canonical: dict[str, str] = {}
    for short, long in alias_pairs:
        alias_canonical[short] = long
        alias_canonical[long] = long

    if answer in alias_canonical:
        return alias_canonical[answer]
    for short, long in alias_pairs:
        if answer.startswith(short + " "):
            return long + answer[len(short) :]
        if answer.endswith(" " + short):
            return answer[: -len(short)] + long
    return answer


def canonicalize_answer_value(value: Any) -> str:
    """Canonicalize answer values so boolean answers are always Yes/No."""
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        try:
            rounded = Decimal(str(value)).quantize(Decimal("0.01"))
        except InvalidOperation:
            return str(value).strip()
        return format(rounded.normalize(), "f").rstrip("0").rstrip(".")
    text = str(value).strip()
    low = text.lower()
    if low in {"true", "yes"}:
        return "Yes"
    if low in {"false", "no"}:
        return "No"
    return text


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    answer = str(answer).lower().strip()
    answer = answer.translate(_UNICODE_DASH_TRANSLATION)
    word_to_num = _WORD_TO_NUMBER_STR
    if answer in word_to_num:
        answer = word_to_num[answer]
    else:
        words = answer.split()
        if words:
            if words[0] in word_to_num:
                words[0] = word_to_num[words[0]]
            if len(words) > 1 and words[-1] in word_to_num:
                words[-1] = word_to_num[words[-1]]
            answer = " ".join(words)
        words = answer.split()
        if len(words) >= 2 and "hundred" in words:
            hundred_idx = words.index("hundred")
            if hundred_idx > 0 and words[hundred_idx - 1] in word_to_num:
                hundreds = int(word_to_num[words[hundred_idx - 1]])
                remainder = 0
                if hundred_idx + 1 < len(words):
                    remainder_words = words[hundred_idx + 1 :]
                    if remainder_words and remainder_words[0] in word_to_num:
                        remainder = int(word_to_num[remainder_words[0]])
                        if len(remainder_words) > 1 and remainder_words[1] in word_to_num:
                            remainder += int(word_to_num[remainder_words[1]])
                    elif len(remainder_words) == 1 and remainder_words[0] in word_to_num:
                        remainder = int(word_to_num[remainder_words[0]])
                total = hundreds * 100 + remainder
                if total > 0:
                    answer = str(total)
    answer = _resolve_alias_pairs(answer)

    street_abbrevs = {
        "st": "street",
        "ave": "avenue",
        "rd": "road",
        "blvd": "boulevard",
        "dr": "drive",
        "ln": "lane",
        "ct": "court",
        "pl": "place",
    }
    words = answer.split()
    for i, word in enumerate(words):
        word_clean = word.rstrip(".,;:")
        if word_clean in street_abbrevs:
            words[i] = word.replace(word_clean, street_abbrevs[word_clean])
    answer = " ".join(words)
    answer = re.sub(r"\b(a|an|the)\b", "", answer)
    answer = re.sub(r"[^\w\s.\-]", "", answer)
    answer = _resolve_alias_pairs(answer)
    answer = answer.rstrip(".,;:!?")
    answer = " ".join(answer.split())
    return answer


def try_parse_float(text: str) -> float | None:
    try:
        text = re.sub(
            r"\s*(people|persons|minutes?|hours?|days?|years?|times?|percent|%)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip()
        thousands_comma = re.search(r"\d{1,3}(,\d{3})+", text)
        thousands_space = re.search(r"\d{1,3}(\s\d{3})+", text)
        if thousands_comma:
            text = text.replace(",", "")
        elif thousands_space:
            text = text.replace(" ", "")
        number_match = re.search(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", text)
        if number_match:
            num_str = number_match.group()
            if num_str.startswith(".") or (num_str.startswith("-") and num_str[1:].startswith(".")):
                num_str = "-0" + num_str[1:] if num_str.startswith("-") else "0" + num_str
            return float(num_str)
        return None
    except (ValueError, AttributeError):
        return None


def is_float_close(pred: str, gt: str, rel_tol: float = 0.05, abs_tol: float = 0.5) -> bool:
    """Check if two numeric strings represent approximately equal values."""
    import math

    pred_float, gt_float = try_parse_float(pred), try_parse_float(gt)
    if pred_float is None or gt_float is None:
        return False
    if pred_float == int(pred_float) and gt_float == int(gt_float):
        return int(pred_float) == int(gt_float)
    return math.isclose(pred_float, gt_float, rel_tol=rel_tol, abs_tol=abs_tol)


def _contains_entity_refs(value: str) -> bool:
    """True when the value still contains unresolved entity refs."""
    return bool(find_entity_refs(str(value or "")))


def _is_numeric_token(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", value))


def exact_match(prediction: str, ground_truth: str) -> bool:
    pred_value = canonicalize_answer_value(prediction)
    gt_value = canonicalize_answer_value(ground_truth)

    if not pred_value or not gt_value:
        return False

    # Ground truth should be resolved to literal values before scoring.
    if _contains_entity_refs(gt_value):
        return False

    pred_norm = normalize_answer(pred_value)
    gt_norm = normalize_answer(gt_value)
    if pred_norm == gt_norm:
        return True

    try:
        if int(pred_value.strip()) == int(gt_value.strip()):
            return True
    except (ValueError, AttributeError):
        pass

    if is_float_close(pred_value, gt_value):
        return True

    pred_clean = re.sub(r"[^a-z0-9]", "", pred_norm)
    gt_clean = re.sub(r"[^a-z0-9]", "", gt_norm)

    if pred_clean in gt_clean or gt_clean in pred_clean:
        shorter = min(len(pred_clean), len(gt_clean))
        longer = max(len(pred_clean), len(gt_clean))
        ratio = shorter / longer if longer > 0 else 0
        shorter_str = pred_clean if len(pred_clean) < len(gt_clean) else gt_clean
        longer_str = gt_clean if len(pred_clean) < len(gt_clean) else pred_clean
        if _is_numeric_token(shorter_str):
            number_pattern = r"(?:^|\D)" + re.escape(shorter_str) + r"(?:\D|$)"
            if re.search(number_pattern, longer_str) and ratio >= _LENGTH_RATIO_THRESHOLD:
                return True
            return False
        if (
            shorter_str.lower() in ["yes", "no", "before", "after"]
            and longer_str.lower().startswith(shorter_str.lower())
            and ratio >= _LENGTH_RATIO_THRESHOLD
        ):
            return True
        if len(gt_norm) >= 5 and pred_norm.startswith(gt_norm):
            extra = pred_norm[len(gt_norm) :].strip()
            if len(extra) <= _MAX_EXTRA_SUFFIX_LENGTH and any(
                word in extra for word in ["police", "officer", "judge", "agent", "soldier", "worker", "employee"]
            ):
                return True
        return False
    return False


def exact_match_strict(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(canonicalize_answer_value(prediction)) == normalize_answer(
        canonicalize_answer_value(ground_truth)
    )


__all__ = [
    "canonicalize_answer_value",
    "exact_match",
    "exact_match_strict",
    "is_float_close",
    "normalize_answer",
    "try_parse_float",
]
