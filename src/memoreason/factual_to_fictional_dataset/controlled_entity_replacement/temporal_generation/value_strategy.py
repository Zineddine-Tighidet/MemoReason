"""Temporal surface parsing and sampling-bound helpers."""

from __future__ import annotations

import random
import re
from datetime import date
from typing import Any, ClassVar

from .date_rule_application import TemporalDateRuleApplicationMixin
from .temporal_date_and_year_values import TemporalDateValueMixin, TemporalYearDomainMixin

_MIN_DISTINCT_YEARS_PER_SERIES = 20


class TemporalValueStrategyMixin(TemporalYearDomainMixin, TemporalDateValueMixin, TemporalDateRuleApplicationMixin):
    """Shared temporal value parsing, surface rendering, and domain helpers."""

    _MONTHS: ClassVar[list[str]] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    _FICTIONAL_TIMEZONE_SUFFIXES = (
        "local standard time",
        "regional standard time",
        "civil time",
        "local time",
    )

    @staticmethod
    def _fictional_timestamp_suffix(suffix: str) -> str:
        rendered = str(suffix or "").strip()
        if not rendered:
            return ""
        # Short timezone abbreviations describe the clock convention rather
        # than a named factual entity and are needed to preserve the surface
        # format.  Geographic long forms are fictionalized below.
        if re.fullmatch(r"[A-Z]{2,5}", rendered):
            return rendered
        return random.choice(TemporalValueStrategyMixin._FICTIONAL_TIMEZONE_SUFFIXES)

    @staticmethod
    def _add_years_safe(value: date, years: int) -> date:
        target_year = int(value.year) + int(years)
        try:
            return value.replace(year=target_year)
        except ValueError:
            # Leap-day fallback: clamp to Feb 28 when the target year is not leap.
            return value.replace(year=target_year, day=28)

    @staticmethod
    def _parse_timestamp_minutes_and_style(value: Any) -> tuple[int | None, dict[str, Any] | None]:
        raw = str(value or "").strip()
        if not raw:
            return None, None

        meridiem_match = re.fullmatch(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
            r"\s*(?P<meridiem>a\.m\.|p\.m\.|am|pm|a\.m|p\.m)"
            r"(?:\s+(?P<suffix>.+))?",
            raw,
            flags=re.IGNORECASE,
        )
        if meridiem_match:
            hour = int(meridiem_match.group("hour"))
            minute = int(meridiem_match.group("minute"))
            second = meridiem_match.group("second")
            meridiem_token = meridiem_match.group("meridiem")
            suffix = meridiem_match.group("suffix") or ""
            meridiem = "am" if meridiem_token.casefold().startswith("a") else "pm"
            if meridiem == "am":
                hour24 = 0 if hour == 12 else hour
            else:
                hour24 = 12 if hour == 12 else hour + 12
            return (
                (hour24 * 60) + minute,
                {
                    "kind": "12h",
                    "second": second,
                    "meridiem_token": meridiem_token,
                    "suffix": suffix,
                },
            )

        clock_match = re.fullmatch(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
            r"(?:\s+(?P<suffix>.+))?",
            raw,
        )
        if clock_match:
            hour = int(clock_match.group("hour")) % 24
            minute = int(clock_match.group("minute"))
            return (
                (hour * 60) + minute,
                {
                    "kind": "24h",
                    "second": clock_match.group("second"),
                    "suffix": clock_match.group("suffix") or "",
                },
            )

        return None, None

    @staticmethod
    def _format_timestamp_like(template: str | None, total_minutes: int) -> str | None:
        minutes, style = TemporalValueStrategyMixin._parse_timestamp_minutes_and_style(template)
        if style is None:
            return template
        del minutes
        total_minutes %= 24 * 60
        hour24, minute = divmod(total_minutes, 60)
        second = style.get("second")
        suffix = TemporalValueStrategyMixin._fictional_timestamp_suffix(style.get("suffix") or "")

        if style.get("kind") == "12h":
            meridiem = "a.m." if hour24 < 12 else "p.m."
            original_token = str(style.get("meridiem_token") or "").casefold()
            if "." not in original_token:
                meridiem = "am" if hour24 < 12 else "pm"
            elif original_token.endswith(".m") and not original_token.endswith(".m."):
                meridiem = "a.m" if hour24 < 12 else "p.m"
            hour12 = hour24 % 12
            if hour12 == 0:
                hour12 = 12
            rendered = f"{hour12}:{minute:02d}"
            if second is not None:
                rendered += f":{int(second):02d}"
            rendered = f"{rendered} {meridiem}"
            if suffix:
                rendered = f"{rendered} {suffix}"
            return rendered

        rendered = f"{hour24:02d}:{minute:02d}"
        if second is not None:
            rendered += f":{int(second):02d}"
        if suffix:
            rendered = f"{rendered} {suffix}"
        return rendered

    def _factual_temporal_year(self, temporal_id: str) -> int | None:
        if not self.factual_entities or not self.factual_entities.temporals:
            return None
        factual_temporal = self.factual_entities.temporals.get(temporal_id)
        if factual_temporal is None:
            return None
        return self._temporal_year_from_entity(factual_temporal)

    def _factual_temporal_timestamp(self, temporal_id: str) -> str | None:
        if not self.factual_entities or not self.factual_entities.temporals:
            return None
        factual_temporal = self.factual_entities.temporals.get(temporal_id)
        if factual_temporal is None:
            return None
        timestamp = (
            getattr(factual_temporal, "timestamp", None)
            if not isinstance(factual_temporal, dict)
            else factual_temporal.get("timestamp")
        )
        return str(timestamp) if timestamp else None

    def _fictionalize_timestamp_surface(self, timestamp: str) -> str:
        original = str(timestamp or "").strip()
        if not original:
            return original

        meridiem_match = re.fullmatch(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
            r"\s*(?P<meridiem>a\.m\.|p\.m\.|am|pm|a\.m|p\.m)"
            r"(?:\s+(?P<suffix>.+))?",
            original,
            flags=re.IGNORECASE,
        )
        if meridiem_match:
            hour = int(meridiem_match.group("hour"))
            minute = int(meridiem_match.group("minute"))
            second = meridiem_match.group("second")
            meridiem = meridiem_match.group("meridiem")
            suffix = self._fictional_timestamp_suffix(meridiem_match.group("suffix") or "")
            shifted_hour = ((hour - 1 + random.randint(1, 5)) % 12) + 1
            shifted_minute = (minute + random.choice((7, 11, 13, 17, 23))) % 60
            pieces = [f"{shifted_hour}:{shifted_minute:02d}"]
            if second is not None:
                shifted_second = (int(second) + random.choice((5, 9, 13, 17))) % 60
                pieces[0] += f":{shifted_second:02d}"
            pieces.append(meridiem)
            if suffix:
                pieces.append(suffix)
            return " ".join(pieces)

        clock_match = re.fullmatch(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
            r"(?:\s+(?P<suffix>.+))?",
            original,
        )
        if clock_match:
            hour = int(clock_match.group("hour"))
            minute = int(clock_match.group("minute"))
            second = clock_match.group("second")
            suffix = self._fictional_timestamp_suffix(clock_match.group("suffix") or "")
            shifted_hour = (hour + random.randint(1, 11)) % 24
            shifted_minute = (minute + random.choice((7, 11, 13, 17, 23))) % 60
            pieces = [f"{shifted_hour:02d}:{shifted_minute:02d}"]
            if second is not None:
                shifted_second = (int(second) + random.choice((5, 9, 13, 17))) % 60
                pieces[0] += f":{shifted_second:02d}"
            if suffix:
                pieces.append(suffix)
            return " ".join(pieces)

        return f"{original} local time"


__all__ = ["TemporalValueStrategyMixin"]
