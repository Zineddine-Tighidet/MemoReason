"""General expression evaluation and entity lookup for MemoReason rules."""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from .century_expressions import century_end, century_of, century_start
from .document_schema import EntityCollection
from .entity_taxonomy import parse_entity_id
from .organization_types import (
    ORG_ENTITY_TYPES,
    canonicalize_organization_type,
    get_organization_name,
    organization_attribute_value,
)
from .annotation_references import (
    ENTITY_REF_PATTERN,
    ENTITY_REF_VALIDATION_PATTERN,
    PERSON_RELATIONSHIP_REF_PATTERN,
    find_entity_refs,
    normalize_entity_ref,
)
from .annotation_values import (
    _APPROX_EQUAL_ABS_TOL,
    _APPROX_EQUAL_REL_TOL,
    _YEAR_ONLY_PATTERN,
    _coerce_numeric_surface,
    _parse_date_surface,
)

logger = logging.getLogger(__name__)

# Bound to the composed class by ``rule_engine`` after module import.
RuleEngine: Any = None


class RuleEvaluationMixin:
    def evaluate_expression(expression: str, entities: EntityCollection) -> Any:
        if not expression or not isinstance(expression, str):
            return expression
        expression = expression.strip()
        cond = re.match(r"^(.+?)\s+if\s+(.+?)\s+else\s+(.+)$", expression)
        if cond:
            cond_result = RuleEngine._evaluate_condition(cond.group(2).strip(), entities)
            return RuleEngine._resolve_value((cond.group(1) if cond_result else cond.group(3)).strip(), entities)
        if expression.lower() in ("true", "false"):
            return expression.lower() == "true"
        normalized = RuleEngine._normalize_comparison_expression(expression)
        if RuleEngine._contains_comparison_operator(normalized):
            return RuleEngine._evaluate_condition(normalized, entities)
        return RuleEngine._evaluate_arithmetic(normalized, entities)

    @staticmethod
    def _evaluate_condition(condition: str, entities: EntityCollection) -> bool:
        normalized = RuleEngine._normalize_comparison_expression(condition)
        for op in (">=", "<=", "==", "!=", ">", "<"):
            if op in normalized:
                parts = normalized.split(op, 1)
                if len(parts) == 2:
                    left = RuleEngine._evaluate_arithmetic(parts[0].strip(), entities)
                    right = RuleEngine._evaluate_arithmetic(parts[1].strip(), entities)
                    return RuleEngine._compare(op, left, right)
        return bool(RuleEngine._evaluate_arithmetic(normalized, entities))

    @staticmethod
    def _compare(op: str, left: Any, right: Any) -> bool:
        """Compare left and right with op, coercing to numbers when possible to avoid str vs int TypeError."""
        left_date = _parse_date_surface(left)
        right_date = _parse_date_surface(right)
        left_is_year_like = bool(
            isinstance(left, (int, float))
            or (isinstance(left, str) and _YEAR_ONLY_PATTERN.fullmatch(left.strip() or ""))
        )
        right_is_year_like = bool(
            isinstance(right, (int, float))
            or (isinstance(right, str) and _YEAR_ONLY_PATTERN.fullmatch(right.strip() or ""))
        )
        if left_date is not None and right_date is not None and left_is_year_like == right_is_year_like:
            if op == ">":
                return left_date > right_date
            if op == "<":
                return left_date < right_date
            if op == ">=":
                return left_date >= right_date
            if op == "<=":
                return left_date <= right_date
            if op == "==":
                return left_date == right_date
            if op == "!=":
                return left_date != right_date

        # Allow year-vs-date comparisons such as `temporal_2.year == temporal_4.date`.
        left_year_only = None
        right_year_only = None
        if isinstance(left, (int, float)) and float(left).is_integer() and 1000 <= int(float(left)) <= 9999:
            left_year_only = int(float(left))
        elif isinstance(left, str) and _YEAR_ONLY_PATTERN.fullmatch(left.strip() or ""):
            left_year_only = int(left.strip())
        if isinstance(right, (int, float)) and float(right).is_integer() and 1000 <= int(float(right)) <= 9999:
            right_year_only = int(float(right))
        elif isinstance(right, str) and _YEAR_ONLY_PATTERN.fullmatch(right.strip() or ""):
            right_year_only = int(right.strip())

        if left_year_only is not None and right_date is not None:
            right_year = int(right_date.year)
            left_year = int(left_year_only)
            if op == ">":
                return left_year > right_year
            if op == "<":
                return left_year < right_year
            if op == ">=":
                return left_year >= right_year
            if op == "<=":
                return left_year <= right_year
            if op == "==":
                return left_year == right_year
            if op == "!=":
                return left_year != right_year
        if right_year_only is not None and left_date is not None:
            left_year = int(left_date.year)
            right_year = int(right_year_only)
            if op == ">":
                return left_year > right_year
            if op == "<":
                return left_year < right_year
            if op == ">=":
                return left_year >= right_year
            if op == "<=":
                return left_year <= right_year
            if op == "==":
                return left_year == right_year
            if op == "!=":
                return left_year != right_year

        try:
            if op == ">":
                return left > right
            if op == "<":
                return left < right
            if op == ">=":
                return left >= right
            if op == "<=":
                return left <= right
        except TypeError:
            pass

        left_n = _coerce_numeric_surface(left)
        right_n = _coerce_numeric_surface(right)
        if left_n is not None and right_n is not None:
            left_f = float(left_n)
            right_f = float(right_n)
            if op == ">":
                return left_f > right_f
            if op == "<":
                return left_f < right_f
            if op == ">=":
                return left_f >= right_f
            if op == "<=":
                return left_f <= right_f
            both_integral = left_f.is_integer() and right_f.is_integer()
            approx_equal = math.isclose(
                left_f,
                right_f,
                rel_tol=_APPROX_EQUAL_REL_TOL,
                abs_tol=_APPROX_EQUAL_ABS_TOL,
            )
            if op == "==":
                return (left_f == right_f) if both_integral else approx_equal
            if op == "!=":
                return (left_f != right_f) if both_integral else (not approx_equal)

        if op in ("==", "!="):
            left_s = str(left or "").strip().lower()
            right_s = str(right or "").strip().lower()
            return left_s == right_s if op == "==" else left_s != right_s
        return False

    @staticmethod
    def _evaluate_arithmetic(expression: str, entities: EntityCollection) -> Any:
        expr = str(expression or "").strip()
        weekday_difference = RuleEngine._evaluate_weekday_difference(expr, entities)
        if weekday_difference is not None:
            return weekday_difference
        shifted_day = RuleEngine._evaluate_weekday_shift(expr, entities)
        if shifted_day is not None:
            return shifted_day
        year_result = RuleEngine._evaluate_year_function_expression(expr, entities)
        if year_result is not None:
            return year_result
        if PERSON_RELATIONSHIP_REF_PATTERN.fullmatch(expr):
            return RuleEngine._get_entity_value(entities, expr)

        temporal_offset = RuleEngine._evaluate_temporal_offset_expression(expr, entities)
        if temporal_offset is not None:
            return temporal_offset

        date_difference = RuleEngine._evaluate_date_difference_expression(expr, entities)
        if date_difference is not None:
            return date_difference

        timestamp_difference = RuleEngine._evaluate_timestamp_difference_expression(expr, entities)
        if timestamp_difference is not None:
            return timestamp_difference

        expr = re.sub(
            r"\b(temporal_\d+)\.date\.(year|month|day)\b",
            lambda m: (
                "None"
                if (value := RuleEngine._resolve_temporal_date_component(m.group(1), m.group(2), entities)) is None
                else (repr(value) if isinstance(value, str) else str(value))
            ),
            expr,
            flags=re.IGNORECASE,
        )

        m = ENTITY_REF_PATTERN.match(expr)
        if m and m.group(0) == expr:
            return RuleEngine._get_entity_value(entities, m.group(0))
        refs = find_entity_refs(expression)
        for ref in refs:
            if RuleEngine._get_entity_value(entities, ref) is None:
                return None
        # Plain space-separated refs only: return values joined by space.
        if refs and not ENTITY_REF_PATTERN.sub("", expr).strip():
            values = [RuleEngine._get_entity_value(entities, ref) for ref in refs]
            if all(v is not None for v in values):
                return " ".join(str(v) for v in values)

        def replacement_for_ref(entity_ref: str) -> str:
            v = RuleEngine._get_entity_value(entities, entity_ref)
            if v is None:
                return "None"
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, str):
                parsed_numeric = _coerce_numeric_surface(v)
                if parsed_numeric is not None:
                    return str(parsed_numeric)
                return json.dumps(v)
            return str(v)

        for ref in sorted(refs, key=len, reverse=True):
            expr = expr.replace(ref, replacement_for_ref(ref))
        try:
            return eval(
                expr,
                {
                    "__builtins__": {},
                    "century_of": century_of,
                    "century_start": century_start,
                    "century_end": century_end,
                },
                {},
            )
        except Exception as e:
            logger.debug("Arithmetic eval failed for %r: %s", expression, e)
            return expression

    @staticmethod
    def _get_entity_value(entities: EntityCollection, entity_ref: str) -> Any | None:
        if not entity_ref:
            return None
        entity_ref = normalize_entity_ref(entity_ref)
        entity_id, attribute = (
            (entity_ref.split(".", 1)[0], entity_ref.split(".", 1)[1]) if "." in entity_ref else (entity_ref, None)
        )
        entity_type, _ = parse_entity_id(entity_id)
        if not entity_type:
            return None
        canonical_entity_type = canonicalize_organization_type(entity_type) or entity_type
        coll = {
            "number": entities.numbers,
            "person": entities.persons,
            "place": entities.places,
            "temporal": entities.temporals,
            "event": entities.events,
            "award": entities.awards,
            "legal": entities.legals,
            "product": entities.products,
            "organization": entities.organizations,
        }
        for organization_type in ORG_ENTITY_TYPES:
            coll[organization_type] = entities.organizations
        target = coll.get(canonical_entity_type, {}) if canonical_entity_type in coll else {}
        entity = target.get(entity_id)
        if entity is None:
            parsed_type, parsed_index = parse_entity_id(entity_id)
            if parsed_type is not None and parsed_index is not None:
                canonical_id = f"{parsed_type}_{parsed_index}"
                if canonical_id != entity_id:
                    entity = target.get(canonical_id)
        if entity is None and entity_id.startswith(("organization_", "organisation_")):
            alternate_ids = []
            if entity_id.startswith("organization_"):
                alternate_ids.append(entity_id.replace("organization_", "organisation_", 1))
            if entity_id.startswith("organisation_"):
                alternate_ids.append(entity_id.replace("organisation_", "organization_", 1))
            for alternate_id in alternate_ids:
                entity = target.get(alternate_id)
                if entity is not None:
                    break
        if entity is None and canonical_entity_type in ORG_ENTITY_TYPES:
            _, entity_index = parse_entity_id(entity_id)
            if entity_index is not None:
                entity = entities.organizations.get(f"organization_{entity_index}")
        if entity is None:
            return None
        if attribute:
            if canonical_entity_type == "person" and attribute.startswith("relationship."):
                other_person_id = attribute.split(".", 1)[1].strip()
                relationship_map = getattr(entity, "relationships", None) or {}
                if isinstance(relationship_map, dict):
                    value = relationship_map.get(other_person_id)
                    if value is not None:
                        return value
                return getattr(entity, "relationship", None)
            if canonical_entity_type in ORG_ENTITY_TYPES:
                val = organization_attribute_value(entity, attribute)
            else:
                val = getattr(entity, attribute, None) if hasattr(entity, attribute) else None
            if canonical_entity_type == "temporal" and val is None:
                parsed_date = _parse_date_surface(getattr(entity, "date", None))
                if parsed_date is not None:
                    if attribute == "year":
                        val = int(parsed_date.year)
                    elif attribute == "month":
                        # Keep month textual to align with annotation surfaces.
                        val = parsed_date.strftime("%B")
                    elif attribute == "day_of_month":
                        val = int(parsed_date.day)
            if attribute == "int" and val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
            return val
        if entity_type == "number":
            for attr in ("int", "str", "float", "percent", "proportion", "fraction"):
                value = getattr(entity, attr, None)
                if value is not None:
                    return value
            return None
        # Person-specific cascade: try name first, then age (so bare `person_X`
        # resolves to age when the entity has no name, enabling rules like `person_2 < 18`).
        if canonical_entity_type == "person":
            for attr in ("full_name", "first_name", "last_name", "age"):
                v = getattr(entity, attr, None)
                if v is not None:
                    return v
            return None
        if canonical_entity_type == "place":
            for attr in ("city", "country", "state", "region", "natural_site", "street", "continent", "demonym"):
                value = getattr(entity, attr, None)
                if value is not None:
                    return value
            return None
        if canonical_entity_type == "legal":
            return getattr(entity, "name", None) or getattr(entity, "reference_code", None)
        if canonical_entity_type in {"award", "product"}:
            return getattr(entity, "name", None)
        if canonical_entity_type in ORG_ENTITY_TYPES:
            return get_organization_name(entity)
        for attr in ("full_name", "city", "date", "name"):
            v = getattr(entity, attr, None)
            if v is not None:
                return v
        return None

    @staticmethod
    def _resolve_value(value: str, entities: EntityCollection) -> Any:
        value = value.strip()
        if ENTITY_REF_VALIDATION_PATTERN.match(value):
            return RuleEngine._get_entity_value(entities, value)
        try:
            return RuleEngine._evaluate_arithmetic(value, entities)
        except Exception as e:
            logger.debug("Resolve value failed for %r: %s", value, e)
            return value
