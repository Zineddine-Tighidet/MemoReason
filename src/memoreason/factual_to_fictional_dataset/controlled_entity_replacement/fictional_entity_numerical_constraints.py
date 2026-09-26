"""Numerical and temporal generation stage for fictional sampling."""

from __future__ import annotations

from fractions import Fraction
import math
import re
from typing import Any

from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs
from memoreason.benchmark_definition.document_schema import EntityCollection

from .number_temporal_generator import NumberTemporalGenerator
from .number_uniqueness import number_entity_uniqueness_value
from .sampling_checks import (
    _coerce_number_value,
    _number_order_bucket,
    _values_equal,
)


class NumericalRuleConstraintsMixin:
    """Numerical ordering domains and rule constraints."""

    _TEMPORAL_PRODUCT_RULE_PATTERN = re.compile(
        r"""
        ^\s*
        (?:
            \((?P<left_expr>[^()]+)\)\s*\*\s*(?P<left_number>number_\d+)\.(?:int|str)
            |
            (?P<right_number>number_\d+)\.(?:int|str)\s*\*\s*\((?P<right_expr>[^()]+)\)
        )
        \s*(?:==|=)\s*(?P<target_number>number_\d+)\.(?:int|str)\s*$
        """,
        re.VERBOSE,
    )
    _TEMPORAL_PRODUCT_CONSTANT_RULE_PATTERN = re.compile(
        r"""
        ^\s*
        (?:
            \((?P<left_expr>[^()]+)\)\s*\*\s*(?P<left_const>[-+]?\d+(?:\.\d+)?)
            |
            (?P<right_const>[-+]?\d+(?:\.\d+)?)\s*\*\s*\((?P<right_expr>[^()]+)\)
        )
        \s*(?:==|=)\s*(?P<target_const>[-+]?\d+(?:\.\d+)?)\s*$
        """,
        re.VERBOSE,
    )
    _SINGLE_NUMBER_RULE_SIDE_RE = re.compile(r"^\s*(number_\d+)\.(int|str)\s*$")
    _FIXED_NUMBER_EQUALITY_RE = re.compile(
        r"""
        ^\s*
        (?:
            (?P<left_num>number_\d+)\.(?:int|str|float|percent|proportion)\s*(?:==|=)\s*(?P<right_const>[-+]?\d+(?:\.\d+)?)
            |
            (?P<left_const>[-+]?\d+(?:\.\d+)?)\s*(?:==|=)\s*(?P<right_num>number_\d+)\.(?:int|str|float|percent|proportion)
        )
        \s*$
        """,
        re.VERBOSE,
    )
    _FIXED_REF_EQUALITY_RE = re.compile(
        r"""
        ^\s*
        (?:
            (?P<left_ref>(?:number_\d+)\.(?:int|str|float|percent|proportion)|(?:temporal_\d+)\.year)
            \s*(?:==|=)\s*
            (?P<right_const>[-+]?\d+(?:\.\d+)?)
            |
            (?P<left_const>[-+]?\d+(?:\.\d+)?)
            \s*(?:==|=)\s*
            (?P<right_ref>(?:number_\d+)\.(?:int|str|float|percent|proportion)|(?:temporal_\d+)\.year)
        )
        \s*$
        """,
        re.VERBOSE,
    )
    _TEMPORAL_DIFF_PLUS_CONST_RE = re.compile(
        r"""
        ^\s*
        (?P<left>temporal_\d+)\.year
        \s*-\s*
        (?P<right>temporal_\d+)\.year
        (?:\s*\+\s*(?P<const>\d+))?
        \s*$
        """,
        re.VERBOSE,
    )

    def _number_order_attr(self, number: Any) -> str | None:
        getter = number.get if isinstance(number, dict) else getattr
        for attr in ("float", "percent", "proportion", "int"):
            raw_value = getter(attr, None) if isinstance(number, dict) else getter(number, attr, None)
            if raw_value is not None:
                return attr
        return None

    def _order_preserving_temporal_year_candidates(
        self,
        *,
        generator,
        collection: EntityCollection,
        temporal_id: str,
        excluded_years: set[int],
        decade_year_temporal_ids: set[str],
        ignore_temporal_ids: set[str] | None = None,
    ) -> list[int]:
        base_low, base_high = generator._temporal_year_base_range(temporal_id)
        base_domain = list(
            generator._temporal_year_domain(
                temporal_id,
                base_low,
                base_high,
                excluded_years,
                decade_year_temporal_ids,
            )
        )
        if not base_domain or not self.factual_entities:
            return base_domain

        factual_temporal = self.factual_entities.temporals.get(temporal_id)
        factual_year = generator._temporal_year_from_entity(factual_temporal)
        if factual_year is None:
            return base_domain

        ignored_ids = set(ignore_temporal_ids or set())
        lower_bound: int | None = None
        upper_bound: int | None = None
        for other_id, other_factual in self.factual_entities.temporals.items():
            if other_id == temporal_id or other_id in ignored_ids:
                continue
            other_factual_year = generator._temporal_year_from_entity(other_factual)
            if other_factual_year is None:
                continue
            other_current = collection.temporals.get(other_id)
            other_current_year = generator._temporal_year_from_entity(other_current)
            if other_current_year is None:
                continue
            if other_factual_year < factual_year:
                lower_bound = max(
                    lower_bound if lower_bound is not None else int(other_current_year) + 1,
                    int(other_current_year) + 1,
                )
            elif other_factual_year > factual_year:
                upper_bound = min(
                    upper_bound if upper_bound is not None else int(other_current_year) - 1,
                    int(other_current_year) - 1,
                )

        filtered_domain = [
            candidate
            for candidate in base_domain
            if (lower_bound is None or candidate >= lower_bound)
            if (upper_bound is None or candidate <= upper_bound)
        ]
        if lower_bound is None and upper_bound is None:
            return filtered_domain or base_domain
        return filtered_domain

    def _rules_referencing_entity_ids(
        self,
        rules: list[str],
        entity_ids: set[str],
    ) -> list[str]:
        if not entity_ids:
            return list(rules)
        relevant_rules: list[str] = []
        for raw_rule in rules:
            rule_text = str(raw_rule)
            if any(f"{entity_id}." in rule_text for entity_id in entity_ids):
                relevant_rules.append(raw_rule)
        return relevant_rules

    def _build_number_ordering_rules(
        self,
        required_number_specs: list[tuple[str, list[str]]] | None = None,
        *,
        excluded_number_ids: set[str] | None = None,
    ) -> list[str]:
        if not self.factual_entities or not self.factual_entities.numbers or not required_number_specs:
            return []

        bucketed_rows: dict[str, list[tuple[float, str, str]]] = {}
        ordering_excluded_number_ids = set(getattr(self, "ordering_excluded_number_ids", set()) or set())
        ordering_excluded_number_ids.update(excluded_number_ids or set())
        for number_id, _attrs in required_number_specs:
            if number_id in ordering_excluded_number_ids:
                continue
            factual_number = self.factual_entities.numbers.get(number_id)
            if factual_number is None:
                continue
            bucket = _number_order_bucket(factual_number)
            factual_value = _coerce_number_value(factual_number)
            attr = self._number_order_attr(factual_number)
            if bucket is None or factual_value is None or attr is None:
                continue
            attr_name = "int" if bucket == "int_like" else attr
            bucketed_rows.setdefault(bucket, []).append((float(factual_value), number_id, attr_name))

        ordering_rules: list[str] = []
        for rows in bucketed_rows.values():
            rows.sort(key=lambda item: (item[0], item[1]))
            for left_index in range(len(rows) - 1):
                left_value, left_id, left_attr = rows[left_index]
                right_value, right_id, right_attr = rows[left_index + 1]
                if math.isclose(left_value, right_value, abs_tol=1e-9):
                    continue
                ordering_rules.append(f"{left_id}.{left_attr} < {right_id}.{right_attr}")
        return ordering_rules

    def _fixed_number_ids_from_rules(self, rules: list[str]) -> set[str]:
        fixed_ids: set[str] = set()
        for raw_rule in rules or []:
            cleaned = str(raw_rule or "").split("#", 1)[0].strip()
            if not cleaned:
                continue
            split = NumberTemporalGenerator._split_rule(cleaned)
            if split is None:
                continue
            lhs, op, rhs = split
            if op not in {"=", "=="}:
                continue
            left_refs = find_entity_refs(lhs)
            right_refs = find_entity_refs(rhs)
            if len(left_refs) == 1 and not right_refs:
                ref = left_refs[0]
                constant = rhs
            elif len(right_refs) == 1 and not left_refs:
                ref = right_refs[0]
                constant = lhs
            else:
                match = self._FIXED_NUMBER_EQUALITY_RE.fullmatch(cleaned)
                if match is None:
                    continue
                ref = match.group("left_num") or match.group("right_num")
                constant = match.group("right_const") or match.group("left_const") or ""
            if not ref.startswith("number_") or self.factual_entities is None:
                continue
            factual_value = RuleEngine._get_entity_value(self.factual_entities, ref)
            cleaned_constant = str(constant).strip().strip('"').strip("'")
            if factual_value is not None and _values_equal(factual_value, cleaned_constant):
                fixed_ids.add(ref.split(".", 1)[0])
        return fixed_ids

    def _linear_number_rules_for_solver(
        self,
        *,
        generator: NumberTemporalGenerator,
        rules: list[str],
        required_number_ids: set[str],
        existing_entities: EntityCollection,
    ) -> list[str]:
        linear_rules: list[str] = []
        for raw_rule in rules:
            if (
                generator._collect_linear_constraints(
                    [raw_rule],
                    required_number_ids,
                    existing_entities,
                )
                is not None
            ):
                linear_rules.append(raw_rule)
        return linear_rules

    def _repair_orbital_product_number_rules(
        self,
        *,
        generator: NumberTemporalGenerator,
        collection: EntityCollection,
        numeric_rules: list[str],
        required_attr_map: dict[str, set[str]],
        avoid_numbers: dict[str, set[int | float]],
    ) -> None:
        """Repair the Sputnik-style nonlinear orbit duration/distance rules."""
        cleaned_rules = {str(rule or "").split("#", 1)[0].strip() for rule in numeric_rules}
        required_patterns = {
            "number_14.int * number_9.float / 60 / 24 >= number_13.str * 30",
            "number_14.int * number_9.float / 60 / 24 < number_13.str * 30 + 30",
            "number_7.int * number_9.float * 60 * number_14.int <= number_15.int * 1.2",
            "number_7.int * number_9.float * 60 * number_14.int >= number_15.int *0.8",
        }
        if not required_patterns.issubset(cleaned_rules):
            return

        def number_entity(number_id: str, value: int):
            return generator._build_number_entity(
                number_id,
                int(value),
                required_attrs=required_attr_map.get(number_id),
            )

        def is_forbidden(number_id: str, entity) -> bool:
            signature = number_entity_uniqueness_value(entity)
            return signature is not None and signature in set(avoid_numbers.get(number_id) or set())

        def int_domain_bounds(number_id: str) -> tuple[int, int] | None:
            low, high = generator._number_base_range(number_id)
            if low > high:
                return None
            forbidden = generator._coerce_forbidden_number_values(avoid_numbers.get(number_id))
            low, high = generator._expand_int_domain_to_escape_forbidden(
                int(low),
                int(high),
                avoid=forbidden,
                min_value=1,
            )
            return int(low), int(high)

        def int_domain(number_id: str) -> list[int]:
            bounds = int_domain_bounds(number_id)
            if bounds is None:
                return []
            low, high = bounds
            return [
                value
                for value in range(int(low), int(high) + 1)
                if not is_forbidden(number_id, number_entity(number_id, value))
            ]

        def assign(number_id: str, value: int) -> None:
            collection.numbers[number_id] = number_entity(number_id, value)

        def closest_candidates(low: int, high: int, preferred: int, count: int):
            """Yield the closest distinct integers without materializing a wide interval."""
            if low > high or count <= 0:
                return
            preferred = min(max(int(preferred), int(low)), int(high))
            yielded = 0
            offset = 0
            while yielded < min(int(count), int(high) - int(low) + 1):
                candidates = (preferred,) if offset == 0 else (preferred - offset, preferred + offset)
                for candidate in candidates:
                    if candidate < low or candidate > high:
                        continue
                    yield candidate
                    yielded += 1
                    if yielded >= min(int(count), int(high) - int(low) + 1):
                        return
                offset += 1

        def ceil_fraction(value: Fraction) -> int:
            return -(-value.numerator // value.denominator)

        def repair() -> bool:
            n13_values = [value for value in int_domain("number_13") if value > 1]
            n9_values = [value for value in int_domain("number_9") if value > 0]
            n7_values = [value for value in int_domain("number_7") if value > 0]
            n14_base = int_domain_bounds("number_14")
            if not (n13_values and n9_values and n7_values and n14_base):
                return False

            factual_n13 = generator._factual_number_int("number_13")
            preferred_n13_values = sorted(n13_values, key=lambda value: (value == factual_n13, abs(value - 2), value))
            for n13 in preferred_n13_values:
                for n9_seed in sorted(n9_values, key=lambda value: (abs(value - 80), value)):
                    n9_entity = number_entity("number_9", n9_seed)
                    n9 = float(getattr(n9_entity, "float", None) or getattr(n9_entity, "int", n9_seed))
                    if n9 <= 0 or not n9.is_integer():
                        continue
                    n9_fraction = Fraction(int(n9), 1)
                    n14_low = ceil_fraction(Fraction(n13 * 30 * 60 * 24, 1) / n9_fraction)
                    n14_high = ceil_fraction(Fraction((n13 * 30 + 30) * 60 * 24, 1) / n9_fraction) - 1
                    if n14_low > n14_high:
                        continue
                    base_low, base_high = n14_base
                    preferred_n14 = min(max(1152, base_low), base_high)
                    # For fixed positive n7/n9, n15 is strictly increasing in
                    # n14.  Therefore each forbidden n14 or n15 signature can
                    # reject at most one candidate; B+1 nearest candidates are
                    # a complete exhaustion proof, not an arbitrary cutoff.
                    candidate_count = (
                        len(set(avoid_numbers.get("number_14") or set()))
                        + len(set(avoid_numbers.get("number_15") or set()))
                        + 1
                    )
                    for n7 in sorted(n7_values, key=lambda value: (abs(value - 11), value)):
                        for n14 in closest_candidates(n14_low, n14_high, preferred_n14, candidate_count):
                            n14_entity = number_entity("number_14", n14)
                            if is_forbidden("number_14", n14_entity):
                                continue
                            distance = int(n7 * int(n9) * 60 * n14)
                            n15_entity = number_entity("number_15", distance)
                            if is_forbidden("number_15", n15_entity):
                                continue
                            assign("number_13", n13)
                            collection.numbers["number_9"] = n9_entity
                            collection.numbers["number_14"] = n14_entity
                            assign("number_7", n7)
                            collection.numbers["number_15"] = n15_entity
                            if all(
                                is_valid for _, is_valid in RuleEngine.validate_all_rules(numeric_rules, collection)
                            ):
                                return True
            return False

        if repair():
            return


__all__ = ["NumericalRuleConstraintsMixin"]
