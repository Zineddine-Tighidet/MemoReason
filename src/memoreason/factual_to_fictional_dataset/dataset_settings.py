"""Dataset setting identifiers for factual and fictional exports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import re

from memoreason.benchmark_definition.entity_taxonomy import (
    REPLACE_MODE_ALL,
    REPLACE_MODE_NON_NUMERICAL,
    REPLACE_MODE_NUMERICAL_TEMPORAL,
)


_FICTIONAL_PERCENT_PATTERN = re.compile(r"^fictional_(?P<percent>[0-9]+(?:_[0-9]+)?)pct$")


@dataclass(frozen=True)
class FactualToFictionalDatasetSetting:
    """One export setting for the factual or fictional dataset."""

    setting_id: str
    setting_family: str
    replacement_proportion: float
    display_label: str
    short_label: str
    is_factual: bool
    compare_to_factual: bool
    replace_mode: str = REPLACE_MODE_ALL

    def to_payload(self) -> dict[str, object]:
        """Return a YAML-safe dict representation."""
        return asdict(self)


def _normalize_proportion(proportion: float) -> float:
    normalized = float(proportion)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"Replacement proportion must be in [0.0, 1.0], got {proportion!r}.")
    return round(normalized, 6)


def _format_percent_slug(proportion: float) -> str:
    percentage = _normalize_proportion(proportion) * 100.0
    text = f"{percentage:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "_")


def _format_percent_label(proportion: float) -> str:
    return f"{_format_percent_slug(proportion).replace('_', '.')}%"


def factual_setting() -> FactualToFictionalDatasetSetting:
    """Return the factual dataset setting."""
    return FactualToFictionalDatasetSetting(
        setting_id="factual",
        setting_family="factual",
        replacement_proportion=0.0,
        display_label="Factual (0%)",
        short_label="0%",
        is_factual=True,
        compare_to_factual=False,
        replace_mode=REPLACE_MODE_ALL,
    )


def fictional_setting(
    replacement_proportion: float = 1.0,
    *,
    setting_id: str | None = None,
    display_label_prefix: str = "Fictional",
    setting_family: str = "fictional",
    replace_mode: str = REPLACE_MODE_ALL,
) -> FactualToFictionalDatasetSetting:
    """Return a fictional dataset setting for one replacement proportion."""
    proportion = _normalize_proportion(replacement_proportion)
    if proportion == 1.0:
        return FactualToFictionalDatasetSetting(
            setting_id=setting_id or "fictional",
            setting_family=setting_family,
            replacement_proportion=1.0,
            display_label=f"{display_label_prefix} (100%)",
            short_label="100%",
            is_factual=False,
            compare_to_factual=True,
            replace_mode=replace_mode,
        )

    percent_label = _format_percent_label(proportion)
    return FactualToFictionalDatasetSetting(
        setting_id=setting_id or f"fictional_{_format_percent_slug(proportion)}pct",
        setting_family=setting_family,
        replacement_proportion=proportion,
        display_label=f"{display_label_prefix} ({percent_label})",
        short_label=percent_label,
        is_factual=False,
        compare_to_factual=True,
        replace_mode=replace_mode,
    )


def parse_dataset_setting(setting_id: str) -> FactualToFictionalDatasetSetting:
    """Parse one stored setting id into a structured dataset setting."""
    normalized = str(setting_id).strip().lower()
    if normalized == "factual":
        return factual_setting()
    if normalized in {"fictional", "fictional_100pct"}:
        return fictional_setting(1.0)
    if normalized in {"fictional_numtemp", "fictional_numerical_temporal", "fictional_numerical_temporal_100pct"}:
        return fictional_setting(
            1.0,
            setting_id="fictional_numtemp",
            display_label_prefix="Fictional Num+Temp",
            setting_family="fictional_numtemp",
            replace_mode=REPLACE_MODE_NUMERICAL_TEMPORAL,
        )
    if normalized in {"fictional_named", "fictional_non_numerical", "fictional_named_100pct"}:
        return fictional_setting(
            1.0,
            setting_id="fictional_named",
            display_label_prefix="Fictional Named",
            setting_family="fictional_named",
            replace_mode=REPLACE_MODE_NON_NUMERICAL,
        )

    match = _FICTIONAL_PERCENT_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(f"Unknown dataset setting: {setting_id!r}")
    percent_text = match.group("percent").replace("_", ".")
    return fictional_setting(float(percent_text) / 100.0)


def dataset_setting_sort_key(setting: str | FactualToFictionalDatasetSetting) -> tuple[int, float, str]:
    """Return a stable sort key for dataset setting ids."""
    spec = setting if isinstance(setting, FactualToFictionalDatasetSetting) else parse_dataset_setting(setting)
    return (0 if spec.is_factual else 1, spec.replacement_proportion, spec.setting_id)


def order_dataset_settings(
    settings: Sequence[str | FactualToFictionalDatasetSetting],
) -> list[FactualToFictionalDatasetSetting]:
    """Deduplicate and order dataset settings."""
    deduped: dict[str, FactualToFictionalDatasetSetting] = {}
    for setting in settings:
        spec = setting if isinstance(setting, FactualToFictionalDatasetSetting) else parse_dataset_setting(setting)
        deduped[spec.setting_id] = spec
    return sorted(deduped.values(), key=dataset_setting_sort_key)


def resolve_dataset_settings(
    *,
    explicit_settings: Sequence[str] | None = None,
    include_factual: bool = True,
    fictional_proportions: Sequence[float] | None = None,
) -> list[FactualToFictionalDatasetSetting]:
    """Resolve the dataset settings requested by a script or workflow."""
    if explicit_settings and fictional_proportions:
        raise ValueError("Use either explicit setting ids or fictional proportions, not both.")

    if explicit_settings:
        return order_dataset_settings(explicit_settings)

    settings: list[FactualToFictionalDatasetSetting] = []
    if include_factual:
        settings.append(factual_setting())

    for proportion in fictional_proportions or (1.0,):
        settings.append(fictional_setting(proportion))
    return order_dataset_settings(settings)
