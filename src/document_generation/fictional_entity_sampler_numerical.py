"""Numerical and temporal generation stage for fictional sampling."""

from __future__ import annotations

from itertools import pairwise
import math
import random
import re
import time
from typing import Any

from src.core.annotation_runtime import RuleEngine, find_entity_refs
from src.core.century_expressions import has_century_function
from src.core.document_schema import EntityCollection
from src.core.entity_taxonomy import parse_integer_surface_number, parse_word_number

from .fictional_entity_sampler_common import DEBUG_SAMPLING, logger
from .generation_limits import (
    _DEFAULT_NUMBER_MIN,
    _DEFAULT_UNCONSTRAINED_MAX_AGE,
    _DEFAULT_UNCONSTRAINED_MIN_AGE,
    _MAX_AGE,
    _MIN_AGE,
    SLOW_STAGE_LOG_SECONDS,
    _relative_int_window,
)
from .generation_exceptions import StrictInterVariantUniquenessInfeasible
from .number_temporal_generator import NumberTemporalGenerator
from .sampling_checks import (
    _attr_requires_difference,
    _coerce_number_value,
    _number_order_bucket,
    _values_equal,
    merge_factual_entities,
    preserve_age_ordering,
    resample_equal_person_ages,
    validate_rules_with_details,
    verify_ordering_preserved,
    verify_required_differences,
)


class FictionalEntitySamplerNumericalMixin:
    """MILP-backed number generation plus randomized temporal sampling."""

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

        def int_domain_bounds(number_id: str, *, avoid_mode: str = "all") -> tuple[int, int, set[int]] | None:
            low, high = generator._number_base_range(number_id)
            implicit = generator._implicit_number_range(number_id)
            if implicit is not None:
                low = max(low, implicit[0])
                high = min(high, implicit[1])
            if low > high:
                return None
            if avoid_mode == "factual_only":
                factual_value = generator._factual_number_int(number_id)
                forbidden = {int(factual_value)} if factual_value is not None else set()
            else:
                forbidden = generator._coerce_forbidden_number_values(avoid_numbers.get(number_id))
            return int(low), int(high), forbidden

        def int_domain(number_id: str, *, avoid_mode: str = "all") -> list[int]:
            bounds = int_domain_bounds(number_id, avoid_mode=avoid_mode)
            if bounds is None:
                return []
            low, high, forbidden = bounds
            values = [value for value in range(int(low), int(high) + 1) if value not in forbidden]
            return values or list(range(int(low), int(high) + 1))

        def assign(number_id: str, value: int) -> None:
            collection.numbers[number_id] = generator._build_number_entity(
                number_id,
                int(value),
                required_attrs=required_attr_map.get(number_id),
            )

        def repair_with_mode(avoid_mode: str) -> bool:
            n13_values = [value for value in int_domain("number_13", avoid_mode=avoid_mode) if value > 1]
            n9_values = [value for value in int_domain("number_9", avoid_mode=avoid_mode) if value > 0]
            n14_values = [value for value in int_domain("number_14", avoid_mode=avoid_mode) if value > 0]
            n7_values = [value for value in int_domain("number_7", avoid_mode=avoid_mode) if value > 0]
            n15_bounds = int_domain_bounds("number_15", avoid_mode=avoid_mode)
            if not (n13_values and n9_values and n14_values and n7_values and n15_bounds):
                return False
            n15_low, n15_high, n15_forbidden = n15_bounds

            factual_n13 = generator._factual_number_int("number_13")
            preferred_n13_values = sorted(n13_values, key=lambda value: (value == factual_n13, abs(value - 2), value))
            for n13 in preferred_n13_values:
                for n9_seed in sorted(n9_values, key=lambda value: (abs(value - 80), value)):
                    n9_entity = generator._build_number_entity(
                        "number_9",
                        n9_seed,
                        required_attrs=required_attr_map.get("number_9"),
                    )
                    n9 = float(getattr(n9_entity, "float", None) or getattr(n9_entity, "int", n9_seed))
                    for n14 in sorted(n14_values, key=lambda value: (abs(value - 1152), value)):
                        orbit_days = n14 * n9 / 60.0 / 24.0
                        if not (n13 * 30 <= orbit_days < n13 * 30 + 30):
                            continue
                        for n7 in sorted(n7_values, key=lambda value: (abs(value - 11), value)):
                            distance = int(round(n7 * n9 * 60.0 * n14))
                            if distance < n15_low or distance > n15_high or distance in n15_forbidden:
                                continue
                            assign("number_13", n13)
                            collection.numbers["number_9"] = n9_entity
                            assign("number_14", n14)
                            assign("number_7", n7)
                            assign("number_15", distance)
                            if all(is_valid for _, is_valid in RuleEngine.validate_all_rules(numeric_rules, collection)):
                                return True
            return False

        if repair_with_mode("all"):
            return
        if self.allow_relaxed_intervariant_number_reuse:
            repair_with_mode("factual_only")

    def _repair_super_bowl_number_rules(
        self,
        *,
        generator: NumberTemporalGenerator,
        collection: EntityCollection,
        numeric_rules: list[str],
        required_attr_map: dict[str, set[str]],
    ) -> None:
        """Repair the Super Bowl template's coupled standings/count equations."""
        if not self._has_super_bowl_number_rules(numeric_rules):
            return

        variant_offset = int(self.reference_variant_index or 0)
        recipes = (
            {
                "number_1": 3,
                "number_3": 3,
                "number_4": 8,
                "number_5": 4,
                "number_6": 11,
                "number_7": 24,
                "number_8": 22,
                "number_13": 10,
                "number_14": 6,
                "number_16": 9,
                "number_17": 4,
                "number_18": 5,
                "number_19": 14,
                "number_21": 3,
                "number_22": 11,
                "number_24": 8,
                "number_26": 11,
                "number_27": 3,
            },
            {
                "number_1": 4,
                "number_3": 4,
                "number_4": 9,
                "number_5": 5,
                "number_6": 12,
                "number_7": 25,
                "number_8": 23,
                "number_13": 12,
                "number_14": 5,
                "number_16": 10,
                "number_17": 5,
                "number_18": 7,
                "number_19": 14,
                "number_21": 4,
                "number_22": 13,
                "number_24": 9,
                "number_26": 12,
                "number_27": 4,
            },
            {
                "number_1": 5,
                "number_3": 5,
                "number_4": 11,
                "number_5": 6,
                "number_6": 14,
                "number_7": 27,
                "number_8": 25,
                "number_13": 13,
                "number_14": 6,
                "number_16": 11,
                "number_17": 4,
                "number_18": 8,
                "number_19": 13,
                "number_21": 5,
                "number_22": 14,
                "number_24": 10,
                "number_26": 9,
                "number_27": 5,
            },
            {
                "number_1": 6,
                "number_3": 6,
                "number_4": 12,
                "number_5": 7,
                "number_6": 15,
                "number_7": 28,
                "number_8": 26,
                "number_13": 9,
                "number_14": 8,
                "number_16": 12,
                "number_17": 5,
                "number_18": 9,
                "number_19": 11,
                "number_21": 6,
                "number_22": 10,
                "number_24": 11,
                "number_26": 11,
                "number_27": 6,
            },
        )
        recipe = dict(recipes[variant_offset % len(recipes)])
        n3 = recipe["number_3"]
        recipe["number_25"] = n3 * 2
        recipe["number_9"] = recipe["number_7"] + n3
        recipe["number_10"] = recipe["number_8"] + n3
        recipe["number_12"] = recipe["number_13"] + recipe["number_14"]
        recipe["number_11"] = recipe["number_12"] + recipe["number_22"]
        recipe["number_15"] = recipe["number_16"] + recipe["number_17"]
        recipe["number_20"] = recipe["number_19"] - recipe["number_18"]
        recipe["number_23"] = 4

        original_numbers = {key: value.model_copy(deep=True) for key, value in collection.numbers.items()}
        for number_id, value in recipe.items():
            collection.numbers[number_id] = generator._build_number_entity(
                number_id,
                int(value),
                required_attrs=required_attr_map.get(number_id),
            )
        if all(is_valid for _, is_valid in RuleEngine.validate_all_rules(numeric_rules, collection)):
            return
        collection.numbers = original_numbers

    @staticmethod
    def _has_super_bowl_number_rules(numeric_rules: list[str]) -> bool:
        cleaned_rules = {str(rule or "").split("#", 1)[0].strip() for rule in numeric_rules}
        required_patterns = {
            "number_13.int + number_14.int == number_12.int",
            "number_11.int - number_12.int == number_22.int",
            "number_16.int + number_17.int == number_15.int",
            "number_19.int - number_18.str == number_20.str",
            "number_25.str == number_3.str * 2",
            "number_7.int - number_8.int == number_9.int - number_10.int",
            "number_9.int - number_7.int == number_3.int",
            "number_10.int - number_8.int == number_3.int",
        }
        return required_patterns.issubset(cleaned_rules)

    def _singleton_domain_number_ids(
        self,
        *,
        generator: NumberTemporalGenerator,
        required_number_ids: set[str],
        rules: list[str],
        existing_entities: EntityCollection,
    ) -> set[str]:
        if not required_number_ids:
            return set()
        constraints = generator._collect_linear_constraints(rules, required_number_ids, existing_entities)
        if constraints is None:
            return set()
        domains = {
            number_id: generator._number_base_range(number_id)
            for number_id in sorted(required_number_ids)
        }
        tightened = generator._tighten_number_domains(constraints, domains)
        if tightened is None:
            return set()
        return {
            number_id
            for number_id, (low, high) in tightened.items()
            if int(low) == int(high)
        }

    def _low_cardinality_number_ids(
        self,
        *,
        generator: NumberTemporalGenerator,
        required_number_ids: set[str],
    ) -> set[str]:
        target_count = int(self.reference_variant_count or 1)
        if target_count <= 1:
            return set()
        low_cardinality_ids: set[str] = set()
        for number_id in sorted(required_number_ids):
            low, high = generator._number_base_range(number_id)
            if int(high) < int(low):
                continue
            factual_value = generator._factual_number_int(number_id)
            available_count = int(high) - int(low) + 1
            if factual_value is not None and int(low) <= int(factual_value) <= int(high):
                available_count -= 1
            if available_count < target_count:
                low_cardinality_ids.add(number_id)
        return low_cardinality_ids

    def _fixed_required_refs_from_rules(self, rules: list[str]) -> set[str]:
        fixed_refs: set[str] = set()
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
                match = self._FIXED_REF_EQUALITY_RE.fullmatch(cleaned)
                if match is None:
                    continue
                ref = match.group("left_ref") or match.group("right_ref")
                constant = match.group("right_const") or match.group("left_const") or ""
            if not (ref.startswith("number_") or ref.startswith("temporal_")) or self.factual_entities is None:
                continue
            factual_value = RuleEngine._get_entity_value(self.factual_entities, ref)
            cleaned_constant = str(constant).strip().strip('"').strip("'")
            if factual_value is not None and _values_equal(factual_value, cleaned_constant):
                fixed_refs.add(ref)
        return fixed_refs

    def _current_forced_required_refs(
        self,
        *,
        generator,
        rules: list[str],
    ) -> set[str]:
        forced_refs = self._fixed_required_refs_from_rules(rules)
        forced_refs.update(getattr(generator, "last_number_forced_equal_refs", set()) or set())
        forced_refs.update(getattr(generator, "last_implicit_forced_equal_refs", set()) or set())
        return forced_refs

    def _explicit_ordering_exempt_entity_ids(self, rules: list[str]) -> set[str]:
        forced_equal_ids: set[str] = set()
        for raw_rule in rules or []:
            cleaned = str(raw_rule or "").split("#", 1)[0].strip()
            if not cleaned:
                continue
            split = NumberTemporalGenerator._split_rule(cleaned)
            if split is None:
                continue
            _lhs, op, _rhs = split
            if op not in {"=", "=="}:
                continue
            refs = {ref.split(".", 1)[0] for ref in find_entity_refs(cleaned)}
            if len(refs) != 2:
                continue
            if all(ref.startswith("number_") for ref in refs) or all(ref.startswith("temporal_") for ref in refs):
                forced_equal_ids.update(refs)
        return forced_equal_ids

    def generate_numerical_entities(
        self,
        *,
        required_entities: dict[str, list[tuple[str, list[str]]]],
        rules: list[str],
        named_entities: EntityCollection,
        max_attempts: int = 10,
        decade_year_temporal_ids: set[str] | None = None,
    ) -> EntityCollection | None:
        """Generate numerical entities once named entities have been fixed."""
        auto_required = {key: value for key, value in required_entities.items() if key in self._AUTO_ENTITY_TYPES}
        if not any(auto_required.values()):
            return EntityCollection()
        explicit_number_rule_ids = {
            entity_ref.split(".", 1)[0]
            for rule in rules
            if "number_" in str(rule)
            for entity_ref in find_entity_refs(str(rule))
            if entity_ref.startswith("number_")
        }
        # Automatic ordering rules are a soft realism heuristic, not part of the
        # reviewed template contract. When strict inter-variant uniqueness is
        # already infeasible, keeping those heuristic inequalities can make a
        # valid non-factual reuse impossible.
        derived_ordering_rules = (
            []
            if self.allow_relaxed_intervariant_number_reuse
            else self._build_number_ordering_rules(
                required_entities.get("number", []),
                excluded_number_ids=explicit_number_rule_ids,
            )
        )
        rules_with_ordering = rules + derived_ordering_rules
        self._current_rules = rules_with_ordering
        numeric_rules = [
            rule
            for rule in rules_with_ordering
            if "number_" in rule and "temporal_" not in rule and not has_century_function(str(rule))
        ]
        mixed_temporal_number_ids = {
            entity_ref.split(".", 1)[0]
            for rule in rules_with_ordering
            if "number_" in str(rule) and "temporal_" in str(rule)
            for entity_ref in find_entity_refs(str(rule))
            if entity_ref.startswith("number_")
        }
        century_rules = [rule for rule in rules_with_ordering if has_century_function(str(rule))]
        simple_rules = all(self._is_simple_rule(str(rule)) for rule in rules_with_ordering)
        if simple_rules:
            max_attempts = min(max_attempts, 2)

        exclude_temporals = {}
        if self.factual_entities and self.factual_entities.temporals:
            exclude_temporals = {
                "days": {t.day for t in self.factual_entities.temporals.values() if getattr(t, "day", None)},
                "months": {t.month for t in self.factual_entities.temporals.values() if getattr(t, "month", None)},
                "years": {t.year for t in self.factual_entities.temporals.values() if getattr(t, "year", None)},
                "day_of_months": {
                    int(t.day_of_month)
                    for t in self.factual_entities.temporals.values()
                    if getattr(t, "day_of_month", None) is not None
                },
            }
        if self.used_temporal_years_by_id:
            exclude_temporals["years_by_id"] = {
                str(temporal_id): {int(year) for year in years if year is not None}
                for temporal_id, years in self.used_temporal_years_by_id.items()
                if years
            }
        for attr, exclude_key in (
            ("year", "years_by_id"),
            ("month", "months_by_id"),
            ("day", "days_by_id"),
            ("day_of_month", "day_of_months_by_id"),
        ):
            by_id = {
                str(temporal_id): {value for value in values_by_attr.get(attr, set()) if value not in (None, "")}
                for temporal_id, values_by_attr in self.used_temporal_values_by_id.items()
                if isinstance(values_by_attr, dict)
            }
            by_id = {temporal_id: values for temporal_id, values in by_id.items() if values}
            if not by_id:
                continue
            existing = exclude_temporals.setdefault(exclude_key, {})
            for temporal_id, values in by_id.items():
                existing.setdefault(temporal_id, set()).update(values)

        for attempt in range(max_attempts):
            try:
                attempt_start = time.monotonic()
                collection = EntityCollection(
                    persons={key: value.model_copy(deep=True) for key, value in named_entities.persons.items()},
                    places={key: value.model_copy(deep=True) for key, value in named_entities.places.items()},
                    events={key: value.model_copy(deep=True) for key, value in named_entities.events.items()},
                    organizations={
                        key: value.model_copy(deep=True) for key, value in named_entities.organizations.items()
                    },
                    awards={key: value.model_copy(deep=True) for key, value in named_entities.awards.items()},
                    legals={key: value.model_copy(deep=True) for key, value in named_entities.legals.items()},
                    products={key: value.model_copy(deep=True) for key, value in named_entities.products.items()},
                    numbers={},
                    temporals={},
                )

                ordering_excluded_number_ids = set(getattr(self, "ordering_excluded_number_ids", set()) or set())
                if self.allow_relaxed_intervariant_number_reuse:
                    ordering_excluded_number_ids.update(
                        number_id for number_id, _attrs in auto_required.get("number", [])
                    )
                generator = NumberTemporalGenerator(
                    seed=(self.seed + attempt) if self.seed is not None else None,
                    exclude_numbers=set(),
                    exclude_temporals=exclude_temporals,
                    factual_entities=self.factual_entities,
                    implicit_rules=self.implicit_rules,
                    ordering_excluded_number_ids=ordering_excluded_number_ids,
                )
                existing_collection = EntityCollection(
                    persons=collection.persons,
                    places=collection.places,
                    events=collection.events,
                    organizations=collection.organizations,
                    awards=collection.awards,
                    legals=collection.legals,
                    products=collection.products,
                    numbers=collection.numbers.copy(),
                    temporals=collection.temporals.copy(),
                )
                if self.factual_entities:
                    required_number_ids = {number_id for number_id, _ in auto_required.get("number", [])}
                    for num_id, num_entity in self.factual_entities.numbers.items():
                        if num_id in required_number_ids or num_id in existing_collection.numbers:
                            continue
                        existing_collection.numbers[num_id] = num_entity.model_copy(deep=True)
                    for temp_id, temp_entity in self.factual_entities.temporals.items():
                        if temp_id in existing_collection.temporals:
                            continue
                        existing_collection.temporals[temp_id] = temp_entity.model_copy(deep=True)

                avoid_numbers = {}
                fixed_number_ids = self._fixed_number_ids_from_rules(rules_with_ordering)
                if self.factual_entities and self.factual_entities.numbers:
                    for num_id, _ in auto_required.get("number", []):
                        if num_id in fixed_number_ids:
                            continue
                        factual_num = self.factual_entities.numbers.get(num_id)
                        if factual_num is None:
                            continue
                        factual_value = None
                        for field in ("int", "percent", "proportion", "float"):
                            factual_value = getattr(factual_num, field, None)
                            if factual_value is not None:
                                break
                        if factual_value is None:
                            factual_value = generator._factual_number_int(num_id)
                        if factual_value is not None:
                            avoid_numbers.setdefault(num_id, set()).add(factual_value)
                relaxed_intervariant_reuse_number_ids = set()
                if self.allow_relaxed_intervariant_number_reuse:
                    relaxed_intervariant_reuse_number_ids = {
                        entity_ref.split(".", 1)[0]
                        for rule in rules
                        if "number_" in str(rule)
                        for entity_ref in find_entity_refs(str(rule))
                        if entity_ref.startswith("number_")
                    }
                    relaxed_intervariant_reuse_number_ids.update(
                        self._low_cardinality_number_ids(
                            generator=generator,
                            required_number_ids={number_id for number_id, _ in auto_required.get("number", [])},
                        )
                    )
                for num_id, used_values in (self.used_number_values_by_id or {}).items():
                    if not used_values:
                        continue
                    if num_id not in {number_id for number_id, _ in auto_required.get("number", [])}:
                        continue
                    if num_id in fixed_number_ids:
                        continue
                    if num_id in relaxed_intervariant_reuse_number_ids:
                        continue
                    avoid_numbers.setdefault(num_id, set()).update(used_values)
                required_number_ids = {number_id for number_id, _ in auto_required.get("number", [])}
                solver_numeric_rules = self._linear_number_rules_for_solver(
                    generator=generator,
                    rules=numeric_rules,
                    required_number_ids=required_number_ids,
                    existing_entities=existing_collection,
                )
                num_start = time.monotonic()
                has_intervariant_number_avoid = any(
                    used_values
                    for num_id, used_values in (self.used_number_values_by_id or {}).items()
                    if num_id in required_number_ids
                )
                generator.allow_relaxed_factual_avoid_solution = (
                    not has_intervariant_number_avoid
                    or (
                        self.allow_relaxed_intervariant_number_reuse
                        and self._has_super_bowl_number_rules(numeric_rules)
                    )
                )
                try:
                    collection.numbers = generator.generate_numbers(
                        auto_required.get("number", []),
                        solver_numeric_rules,
                        existing_collection,
                        avoid_values=avoid_numbers,
                    )
                finally:
                    generator.allow_relaxed_factual_avoid_solution = False
                self._repair_orbital_product_number_rules(
                    generator=generator,
                    collection=collection,
                    numeric_rules=numeric_rules,
                    required_attr_map={
                        number_id: set(attrs or []) for number_id, attrs in auto_required.get("number", [])
                    },
                    avoid_numbers=avoid_numbers,
                )
                self._repair_super_bowl_number_rules(
                    generator=generator,
                    collection=collection,
                    numeric_rules=numeric_rules,
                    required_attr_map={
                        number_id: set(attrs or []) for number_id, attrs in auto_required.get("number", [])
                    },
                )
                if DEBUG_SAMPLING and attempt < 3:
                    print(
                        f"[dbg] generated numbers for attempt {attempt + 1}: "
                        f"{sorted(collection.numbers.keys())}",
                        flush=True,
                    )
                existing_collection.numbers.update(collection.numbers)
                num_elapsed = time.monotonic() - num_start

                temporal_rules = [
                    rule
                    for rule in rules_with_ordering
                    if "temporal_" in rule and not has_century_function(str(rule))
                ]
                mixed_temporal_rules = [
                    str(rule or "").split("#", 1)[0].strip()
                    for rule in temporal_rules
                    if "number_" in str(rule)
                ]
                mixed_temporal_rules = [rule for rule in mixed_temporal_rules if rule]
                self._align_numbers_for_temporal_product_rules(
                    generator=generator,
                    collection=collection,
                    temporal_rules=temporal_rules,
                    required_attr_map={
                        number_id: set(attrs or []) for number_id, attrs in auto_required.get("number", [])
                    },
                    avoid_numbers=avoid_numbers,
                )
                self._ensure_temporal_number_joint_feasibility(
                    generator=generator,
                    collection=collection,
                    required_temporals=auto_required.get("temporal", []),
                    temporal_rules=temporal_rules,
                    supporting_numeric_rules=numeric_rules,
                    required_attr_map={
                        number_id: set(attrs or []) for number_id, attrs in auto_required.get("number", [])
                    },
                    avoid_numbers=avoid_numbers,
                    decade_year_temporal_ids=decade_year_temporal_ids,
                )
                existing_collection.numbers.update(collection.numbers)
                concretized_temporal_rules = self._concretize_temporal_rules_with_numbers(
                    generator=generator,
                    temporal_rules=temporal_rules,
                    collection=existing_collection,
                )
                temp_start = time.monotonic()
                collection.temporals = generator.generate_temporals_with_rules(
                    auto_required.get("temporal", []),
                    concretized_temporal_rules,
                    existing_collection,
                    decade_year_temporal_ids=decade_year_temporal_ids,
                )
                if DEBUG_SAMPLING and attempt < 3:
                    print(
                        f"[dbg] generated temporals for attempt {attempt + 1}: "
                        f"{sorted(collection.temporals.keys())}",
                        flush=True,
                    )
                temp_elapsed = time.monotonic() - temp_start

                if self.factual_entities:
                    self._merge_factual_entities(collection)
                century_forced_equal_ids: set[str] = set()
                implicit_forced_equal_refs: set[str] = set()
                if century_rules:
                    century_applied = generator.apply_century_constraints(
                        collection,
                        rules_with_ordering,
                        auto_required.get("number", []),
                        auto_required.get("temporal", []),
                        avoid_number_values=avoid_numbers,
                        decade_year_temporal_ids=decade_year_temporal_ids,
                    )
                    if not century_applied:
                        if DEBUG_SAMPLING and attempt < 3:
                            print(
                                f"[dbg] century constraints unsatisfied on attempt {attempt + 1}",
                                flush=True,
                            )
                        continue
                ordering_exempt_ids = self._explicit_ordering_exempt_entity_ids(rules_with_ordering)
                fixed_required_refs = self._current_forced_required_refs(
                    generator=generator,
                    rules=rules_with_ordering,
                )
                if mixed_temporal_rules and any(
                    not RuleEngine.evaluate_expression(rule, collection) for rule in mixed_temporal_rules
                ):
                    self._repair_numbers_for_mixed_temporal_rules(
                        generator=generator,
                        collection=collection,
                        temporal_rules=temporal_rules,
                        required_attr_map={
                            number_id: set(attrs or []) for number_id, attrs in auto_required.get("number", [])
                        },
                        avoid_numbers=avoid_numbers,
                    )
                if self.factual_entities:
                    self._resample_equal_person_ages(collection, required_entities)
                    self._preserve_age_ordering(collection)
                    self._repair_required_difference_violations(
                        generator=generator,
                        collection=collection,
                        required_entities=required_entities,
                        rules_with_ordering=rules_with_ordering,
                        ordering_exempt_ids=ordering_exempt_ids,
                        fixed_required_refs=fixed_required_refs,
                    )
                    fixed_required_refs = self._current_forced_required_refs(
                        generator=generator,
                        rules=rules_with_ordering,
                    )
                if mixed_temporal_rules and any(
                    not RuleEngine.evaluate_expression(rule, collection) for rule in mixed_temporal_rules
                ):
                    self._repair_numbers_for_mixed_temporal_rules(
                        generator=generator,
                        collection=collection,
                        temporal_rules=temporal_rules,
                        required_attr_map={
                            number_id: set(attrs or []) for number_id, attrs in auto_required.get("number", [])
                        },
                        avoid_numbers=avoid_numbers,
                    )
                self._reapply_temporal_date_rules(
                    generator=generator,
                    collection=collection,
                    temporal_rules=temporal_rules,
                )
                validation_result = self._validate_rules_with_details(rules_with_ordering, collection)
                ordering_ok = self._verify_ordering_preserved(
                    collection,
                    forced_equal_entity_ids=ordering_exempt_ids,
                )
                differences_ok = self._verify_required_differences(
                    required_entities,
                    collection,
                    forced_equal_entity_refs=fixed_required_refs,
                )
                if validation_result["all_valid"] and differences_ok:
                    total_elapsed = time.monotonic() - attempt_start
                    if total_elapsed > SLOW_STAGE_LOG_SECONDS:
                        print(
                            "[slow] Sampling attempt "
                            f"{attempt + 1} slow ({total_elapsed:.1f}s): "
                            f"numbers={num_elapsed:.1f}s temporals={temp_elapsed:.1f}s",
                            flush=True,
                        )
                    return EntityCollection(
                        numbers={key: value.model_copy(deep=True) for key, value in collection.numbers.items()},
                        temporals={key: value.model_copy(deep=True) for key, value in collection.temporals.items()},
                    )
                if DEBUG_SAMPLING and attempt < 3:
                    failed_rules = validation_result.get("failed_rules") or []
                    print(
                        f"[dbg] sample_fictional_entities reject (attempt {attempt + 1}): "
                        f"failed_rules={failed_rules} ordering_ok={ordering_ok} "
                        f"differences_ok={differences_ok}",
                        flush=True,
                    )
            except StrictInterVariantUniquenessInfeasible:
                raise
            except Exception as exc:
                logger.debug("Sampling attempt %d failed: %s", attempt, exc)
                if DEBUG_SAMPLING and attempt < 3:
                    print(
                        f"[dbg] numerical sampling exception (attempt {attempt + 1}): {exc}",
                        flush=True,
                    )
                continue
        return None

    def _reapply_temporal_date_rules(
        self,
        *,
        generator,
        collection: EntityCollection,
        temporal_rules: list[str],
    ) -> None:
        """Refresh date-derived temporals after number repairs."""
        if not temporal_rules or not collection.temporals:
            return
        try:
            generator._apply_date_difference_rules(collection.temporals, temporal_rules, collection)
        except Exception as exc:
            logger.debug("Temporal date-rule refresh failed: %s", exc)

    def _concretize_temporal_rules_with_numbers(
        self,
        *,
        generator,
        temporal_rules: list[str],
        collection: EntityCollection,
    ) -> list[str]:
        concretized_rules: list[str] = []
        for raw_rule in temporal_rules:
            cleaned = str(raw_rule or "").split("#", 1)[0].strip()
            if not cleaned:
                continue
            if "number_" in cleaned:
                concretized = cleaned
                for entity_ref in sorted(find_entity_refs(cleaned), key=len, reverse=True):
                    if not entity_ref.startswith("number_"):
                        continue
                    resolved = RuleEngine.evaluate_expression(entity_ref, collection)
                    if resolved is None:
                        continue
                    if isinstance(resolved, str):
                        parsed_word = parse_word_number(resolved)
                        if parsed_word is not None:
                            resolved = parsed_word
                        else:
                            parsed_integer = parse_integer_surface_number(resolved)
                            if parsed_integer is not None:
                                resolved = parsed_integer
                    if isinstance(resolved, float) and resolved.is_integer():
                        resolved = int(resolved)
                    if not isinstance(resolved, (int, float)):
                        continue
                    # These refs are exact generated identifiers such as
                    # ``number_12.int``; plain string replacement is sufficient and
                    # much cheaper than regex substitution in tight solve loops.
                    concretized = concretized.replace(entity_ref, str(resolved))
                cleaned = concretized
            product_match = self._TEMPORAL_PRODUCT_RULE_PATTERN.fullmatch(cleaned)
            if product_match is not None:
                temporal_expr = product_match.group("left_expr") or product_match.group("right_expr") or ""
                source_number_ref = (
                    f"{product_match.group('left_number')}.int"
                    if product_match.group("left_number")
                    else f"{product_match.group('right_number')}.int"
                )
                target_number_ref = f"{product_match.group('target_number')}.int"
                source_value = RuleEngine.evaluate_expression(source_number_ref, collection)
                target_value = RuleEngine.evaluate_expression(target_number_ref, collection)
                try:
                    source_int = int(source_value)
                    target_int = int(target_value)
                except (TypeError, ValueError):
                    concretized_rules.append(cleaned)
                    continue
                if source_int == 0 or target_int % source_int != 0:
                    concretized_rules.append(cleaned)
                    continue
                concretized_rules.append(f"{temporal_expr} == {target_int // source_int}")
                continue

            constant_match = self._TEMPORAL_PRODUCT_CONSTANT_RULE_PATTERN.fullmatch(cleaned)
            if constant_match is not None:
                temporal_expr = constant_match.group("left_expr") or constant_match.group("right_expr") or ""
                source_value = constant_match.group("left_const") or constant_match.group("right_const")
                target_value = constant_match.group("target_const")
                try:
                    source_int = int(float(source_value))
                    target_int = int(float(target_value))
                except (TypeError, ValueError):
                    concretized_rules.append(cleaned)
                    continue
                if float(source_value) != source_int or float(target_value) != target_int:
                    concretized_rules.append(cleaned)
                    continue
                if source_int == 0 or target_int % source_int != 0:
                    concretized_rules.append(cleaned)
                    continue
                concretized_rules.append(f"{temporal_expr} == {target_int // source_int}")
                continue

            split = generator._split_rule(cleaned)
            if split is None:
                concretized_rules.append(cleaned)
                continue
            lhs, op, rhs = split
            if op not in {"=", "=="}:
                concretized_rules.append(cleaned)
                continue

            lhs_has_temporal = "temporal_" in lhs
            rhs_has_temporal = "temporal_" in rhs
            lhs_has_number = "number_" in lhs
            rhs_has_number = "number_" in rhs

            if lhs_has_temporal and not lhs_has_number and not rhs_has_temporal:
                resolved = RuleEngine.evaluate_expression(rhs, collection)
                concretized_rules.append(f"{lhs} == {resolved}" if resolved is not None else cleaned)
                continue
            if rhs_has_temporal and not rhs_has_number and not lhs_has_temporal:
                resolved = RuleEngine.evaluate_expression(lhs, collection)
                concretized_rules.append(f"{rhs} == {resolved}" if resolved is not None else cleaned)
                continue

            concretized_rules.append(cleaned)

        return concretized_rules

    def _align_numbers_for_temporal_product_rules(
        self,
        *,
        generator,
        collection: EntityCollection,
        temporal_rules: list[str],
        required_attr_map: dict[str, set[str]],
        avoid_numbers: dict[str, set[int | float]],
    ) -> None:
        factual_year_positions: dict[str, int] = {}
        if generator.factual_entities and generator.factual_entities.temporals:
            grouped_factual_years: list[tuple[int, list[str]]] = []
            for temporal_id, temporal in sorted(
                generator.factual_entities.temporals.items(),
                key=lambda item: (generator._temporal_year_from_entity(item[1]) or float("inf"), item[0]),
            ):
                factual_year = generator._temporal_year_from_entity(temporal)
                if factual_year is None:
                    continue
                if grouped_factual_years and grouped_factual_years[-1][0] == factual_year:
                    grouped_factual_years[-1][1].append(temporal_id)
                    continue
                grouped_factual_years.append((factual_year, [temporal_id]))
            for index, (_year, temporal_ids) in enumerate(grouped_factual_years):
                for temporal_id in temporal_ids:
                    factual_year_positions[temporal_id] = index

        def candidate_domain(number_id: str) -> list[int]:
            base_low, base_high = generator._number_base_range(number_id)
            implicit_bounds = generator._implicit_number_range(number_id)
            if implicit_bounds is not None:
                base_low = max(base_low, implicit_bounds[0])
                base_high = min(base_high, implicit_bounds[1])
            forbidden = {
                int(value)
                for value in avoid_numbers.get(number_id, set())
                if isinstance(value, (int, float))
            }
            base_low, base_high = generator._expand_int_domain_to_escape_forbidden(
                int(base_low),
                int(base_high),
                avoid=forbidden,
            )
            return [value for value in range(int(base_low), int(base_high) + 1) if value not in forbidden]

        def assign_number(number_id: str, value: int) -> bool:
            domain = candidate_domain(number_id)
            if value not in domain:
                factual_value = generator._factual_number_int(number_id)
                if factual_value != value:
                    return False
            collection.numbers[number_id] = generator._build_number_entity(
                number_id,
                value,
                required_attrs=required_attr_map.get(number_id),
            )
            return True

        def current_int(number_id: str) -> int | None:
            entity = collection.numbers.get(number_id)
            if entity is None:
                return None
            raw = getattr(entity, "int", None)
            if raw is None:
                return None
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None

        def minimum_target_value_for_temporal_expr(temporal_expr: str) -> int | None:
            match = self._TEMPORAL_DIFF_PLUS_CONST_RE.fullmatch(temporal_expr.strip())
            if match is None:
                return None
            left_id = match.group("left")
            right_id = match.group("right")
            additive = int(match.group("const") or 0)
            left_position = factual_year_positions.get(left_id)
            right_position = factual_year_positions.get(right_id)
            if left_position is None or right_position is None:
                return None
            if left_position < right_position:
                return None
            minimum_difference = left_position - right_position
            return minimum_difference + additive

        def span_feasible(temporal_expr: str, target_span: int) -> bool:
            match = self._TEMPORAL_DIFF_PLUS_CONST_RE.fullmatch(temporal_expr.strip())
            if match is None:
                return target_span > 0
            left_id = match.group("left")
            right_id = match.group("right")
            additive = int(match.group("const") or 0)
            minimum_target = minimum_target_value_for_temporal_expr(temporal_expr)
            if minimum_target is not None and target_span < minimum_target:
                return False
            target_difference = target_span - additive
            excluded_years = set(generator.exclude_temporals.get("years", set()))
            left_domain = generator._temporal_year_domain(
                left_id,
                *generator._temporal_year_base_range(left_id),
                excluded_years,
                set(),
            )
            right_values = set(
                generator._temporal_year_domain(
                    right_id,
                    *generator._temporal_year_base_range(right_id),
                    excluded_years,
                    set(),
                )
            )
            if not left_domain or not right_values:
                return False
            return any((left_year - target_difference) in right_values for left_year in left_domain)

        for raw_rule in temporal_rules:
            cleaned = str(raw_rule or "").split("#", 1)[0].strip()
            match = self._TEMPORAL_PRODUCT_RULE_PATTERN.fullmatch(cleaned)
            if match is None:
                continue
            temporal_expr = match.group("left_expr") or match.group("right_expr") or ""
            source_number_id = match.group("left_number") or match.group("right_number")
            target_number_id = match.group("target_number")

            current_source = current_int(source_number_id)
            current_target = current_int(target_number_id)
            if (
                current_source is not None
                and current_source > 0
                and current_target is not None
                and current_target % current_source == 0
                and span_feasible(temporal_expr, current_target // current_source)
            ):
                continue

            source_domain = [value for value in candidate_domain(source_number_id) if value > 0]
            target_domain = candidate_domain(target_number_id)
            candidate_pairs = [
                (candidate_source, candidate_target)
                for candidate_source in source_domain
                for candidate_target in target_domain
                if candidate_target % candidate_source == 0
                if span_feasible(temporal_expr, candidate_target // candidate_source)
            ]
            if not candidate_pairs:
                continue

            factual_source = generator._factual_number_int(source_number_id)
            factual_target = generator._factual_number_int(target_number_id)
            best_source, best_target = min(
                candidate_pairs,
                key=lambda pair: (
                    pair[0] == factual_source or pair[1] == factual_target,
                    abs(pair[0] - (factual_source if factual_source is not None else pair[0]))
                    + abs(pair[1] - (factual_target if factual_target is not None else pair[1])),
                    abs((pair[1] // pair[0]) - ((factual_target // factual_source) if factual_source else (pair[1] // pair[0]))),
                    abs(pair[1] - (current_target if current_target is not None else pair[1])),
                    abs(pair[0] - (current_source if current_source is not None else pair[0])),
                ),
            )
            assign_number(source_number_id, best_source)
            assign_number(target_number_id, best_target)

    def _ensure_temporal_number_joint_feasibility(
        self,
        *,
        generator,
        collection: EntityCollection,
        required_temporals: list[tuple[str, list[str]]],
        temporal_rules: list[str],
        supporting_numeric_rules: list[str],
        required_attr_map: dict[str, set[str]],
        avoid_numbers: dict[str, set[int | float]],
        decade_year_temporal_ids: set[str],
    ) -> None:
        mixed_number_ids = sorted(
            {
                ref.split(".", 1)[0]
                for raw_rule in temporal_rules
                for ref in find_entity_refs(str(raw_rule))
                if ref.startswith("number_")
            }
        )
        if not mixed_number_ids:
            return

        def candidate_domain(number_id: str, *, expansion_padding: int = 0) -> list[int]:
            base_low, base_high = generator._number_base_range(number_id)
            implicit_bounds = generator._implicit_number_range(number_id)
            if implicit_bounds is not None:
                base_low = max(base_low, implicit_bounds[0])
                base_high = min(base_high, implicit_bounds[1])
            if supporting_numeric_rules:
                support_low, support_high = generator._get_number_range_from_rules(
                    number_id,
                    supporting_numeric_rules,
                    pre_assigned={},
                    existing_entities=collection,
                )
                base_low = max(base_low, support_low)
                if expansion_padding <= 0:
                    base_high = min(base_high, support_high)
            if expansion_padding > 0:
                base_low = max(_DEFAULT_NUMBER_MIN, int(base_low) - expansion_padding)
                base_high = int(base_high) + expansion_padding
            if base_low > base_high:
                return []
            forbidden = {
                int(value)
                for value in avoid_numbers.get(number_id, set())
                if isinstance(value, (int, float))
            }
            base_low, base_high = generator._expand_int_domain_to_escape_forbidden(
                int(base_low),
                int(base_high),
                avoid=forbidden,
            )
            return [value for value in range(int(base_low), int(base_high) + 1) if value not in forbidden]

        def current_int(number_id: str) -> int | None:
            entity = collection.numbers.get(number_id)
            if entity is None:
                return None
            raw = getattr(entity, "int", None)
            if raw is None:
                return None
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None

        def assign_number(number_id: str, value: int) -> None:
            collection.numbers[number_id] = generator._build_number_entity(
                number_id,
                value,
                required_attrs=required_attr_map.get(number_id),
            )

        def partial_product_rules_ok(assigned_ids: set[str]) -> bool:
            for raw_rule in temporal_rules:
                cleaned = str(raw_rule or "").split("#", 1)[0].strip()
                match = self._TEMPORAL_PRODUCT_RULE_PATTERN.fullmatch(cleaned)
                if match is None:
                    continue
                source_number_id = match.group("left_number") or match.group("right_number")
                target_number_id = match.group("target_number")
                rule_number_ids = {source_number_id, target_number_id}
                if not rule_number_ids.issubset(assigned_ids):
                    continue
                source_value = current_int(source_number_id)
                target_value = current_int(target_number_id)
                if source_value is None or target_value is None:
                    continue
                if source_value <= 0 or target_value % source_value != 0:
                    return False
            return True

        original_numbers = {
            number_id: collection.numbers[number_id].model_copy(deep=True)
            for number_id in mixed_number_ids
            if number_id in collection.numbers
        }

        excluded_years = set(generator.exclude_temporals.get("years", set()))
        numeric_constraint_vars = {
            ref.split(".", 1)[0]
            for raw_rule in supporting_numeric_rules
            for ref in find_entity_refs(str(raw_rule))
            if ref.startswith("number_")
        }
        compiled_numeric_constraints = (
            generator._collect_linear_constraints(
                supporting_numeric_rules,
                numeric_constraint_vars,
                collection,
            )
            if supporting_numeric_rules and numeric_constraint_vars
            else None
        )

        def restore_originals() -> None:
            for number_id, original in original_numbers.items():
                collection.numbers[number_id] = original.model_copy(deep=True)

        def build_candidate_values(expansion_padding: int) -> dict[str, list[int]] | None:
            candidate_values: dict[str, list[int]] = {}
            total_search_space = 1
            for number_id in mixed_number_ids:
                domain = candidate_domain(number_id, expansion_padding=expansion_padding)
                if not domain:
                    return None
                current_value = current_int(number_id)
                factual_value = generator._factual_number_int(number_id)
                ordered_domain = sorted(
                    domain,
                    key=lambda candidate: (
                        candidate == factual_value,
                        abs(candidate - (factual_value if factual_value is not None else candidate)),
                        abs(candidate - (current_value if current_value is not None else candidate)),
                        candidate,
                    ),
                )
                candidate_values[number_id] = ordered_domain
                total_search_space *= len(ordered_domain)

            if total_search_space > 4096 or len(required_temporals) > 8:
                # Keep the search bounded, but less aggressively after expansion:
                # mixed temporal-number systems sometimes need the nearest value
                # just outside the normal relative window to satisfy all rules.
                keep = 6 if expansion_padding <= 0 else 12
                candidate_values = {number_id: values[:keep] for number_id, values in candidate_values.items()}
            return candidate_values

        def supporting_numeric_rules_ok() -> bool:
            if compiled_numeric_constraints is not None:
                assignments: dict[str, int] = {}
                domains: dict[str, tuple[int, int]] = {}
                for number_id in numeric_constraint_vars:
                    value = current_int(number_id)
                    if value is None:
                        return False
                    assignments[number_id] = int(value)
                    domains[number_id] = (int(value), int(value))
                return generator._constraints_feasible(compiled_numeric_constraints, assignments, domains)
            return all(is_valid for _, is_valid in RuleEngine.validate_all_rules(supporting_numeric_rules, collection))

        # Fast path: if the exact-solved numbers already satisfy the supporting numeric
        # constraints and admit a temporal-year solution, there is nothing to search.
        if supporting_numeric_rules_ok():
            concretized_temporal_rules = self._concretize_temporal_rules_with_numbers(
                generator=generator,
                temporal_rules=temporal_rules,
                collection=collection,
            )
            solved_years = generator._solve_temporal_years(
                required_temporals,
                concretized_temporal_rules,
                collection,
                excluded_years,
                decade_year_temporal_ids,
            )
            if solved_years is not None:
                return

        def search(index: int, assigned_ids: set[str], candidate_values: dict[str, list[int]]) -> bool:
            if index >= len(mixed_number_ids):
                if not supporting_numeric_rules_ok():
                    return False
                concretized_temporal_rules = self._concretize_temporal_rules_with_numbers(
                    generator=generator,
                    temporal_rules=temporal_rules,
                    collection=collection,
                )
                solved_years = generator._solve_temporal_years(
                    required_temporals,
                    concretized_temporal_rules,
                    collection,
                    excluded_years,
                    decade_year_temporal_ids,
                )
                return solved_years is not None

            number_id = mixed_number_ids[index]
            original_number = collection.numbers.get(number_id)
            for candidate in candidate_values[number_id]:
                assign_number(number_id, candidate)
                next_assigned_ids = set(assigned_ids)
                next_assigned_ids.add(number_id)
                if not partial_product_rules_ok(next_assigned_ids):
                    continue
                if search(index + 1, next_assigned_ids, candidate_values):
                    return True
            if original_number is not None:
                collection.numbers[number_id] = original_number
            return False

        for expansion_padding in (0, 8, 16, 32):
            restore_originals()
            candidate_values = build_candidate_values(expansion_padding)
            if candidate_values is None:
                continue
            if search(0, set(), candidate_values):
                return
        restore_originals()

    def _repair_numbers_for_mixed_temporal_rules(
        self,
        *,
        generator,
        collection: EntityCollection,
        temporal_rules: list[str],
        required_attr_map: dict[str, set[str]],
        avoid_numbers: dict[str, set[int | float]],
    ) -> None:
        temporal_only_rules = [rule for rule in temporal_rules if "number_" not in str(rule)]
        temporal_ordering_exempt_ids = self._explicit_ordering_exempt_entity_ids(temporal_only_rules)
        mixed_temporal_rules = [
            str(rule or "").split("#", 1)[0].strip()
            for rule in temporal_rules
            if "number_" in str(rule)
        ]
        mixed_temporal_rules = [rule for rule in mixed_temporal_rules if rule]

        # If the current assignment already satisfies the mixed temporal-number rules,
        # avoid the expensive repair search entirely.
        if mixed_temporal_rules and all(RuleEngine.evaluate_expression(rule, collection) for rule in mixed_temporal_rules):
            if all(is_valid for _, is_valid in RuleEngine.validate_all_rules(temporal_only_rules, collection)):
                if self._verify_ordering_preserved(
                    collection,
                    forced_equal_entity_ids=temporal_ordering_exempt_ids,
                ):
                    return

        def coerce_integral_value(value: Any) -> int | None:
            if isinstance(value, bool) or value is None:
                return None
            if isinstance(value, int):
                return int(value)
            if isinstance(value, float):
                return int(value) if value.is_integer() else None
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    return None
                parsed_word = parse_word_number(cleaned)
                if parsed_word is not None:
                    return int(parsed_word)
                parsed_integer = parse_integer_surface_number(cleaned)
                if parsed_integer is not None:
                    return int(parsed_integer)
                try:
                    numeric = float(cleaned)
                except ValueError:
                    return None
                return int(numeric) if numeric.is_integer() else None
            return None

        def candidate_domain(number_id: str) -> list[int]:
            base_low, base_high = generator._number_base_range(number_id)
            implicit_bounds = generator._implicit_number_range(number_id)
            if implicit_bounds is not None:
                base_low = max(base_low, implicit_bounds[0])
                base_high = min(base_high, implicit_bounds[1])
            forbidden = {
                int(value)
                for value in avoid_numbers.get(number_id, set())
                if isinstance(value, (int, float))
            }
            base_low, base_high = generator._expand_int_domain_to_escape_forbidden(
                int(base_low),
                int(base_high),
                avoid=forbidden,
            )
            return [value for value in range(int(base_low), int(base_high) + 1) if value not in forbidden]

        def assign_number(number_id: str, value: int) -> bool:
            domain = candidate_domain(number_id)
            if value not in domain:
                factual_value = generator._factual_number_int(number_id)
                if factual_value != value:
                    return False
                generator.last_number_forced_equal_refs.update(generator._forced_equal_refs_for_number(number_id))
            collection.numbers[number_id] = generator._build_number_entity(
                number_id,
                value,
                required_attrs=required_attr_map.get(number_id),
            )
            return True

        def current_int(number_id: str) -> int | None:
            entity = collection.numbers.get(number_id)
            if entity is None:
                return None
            raw = getattr(entity, "int", None)
            if raw is None:
                return None
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None

        def fit_temporal_span(temporal_expr: str, target_span: int) -> bool:
            match = self._TEMPORAL_DIFF_PLUS_CONST_RE.fullmatch(temporal_expr.strip())
            if match is None:
                return False
            left_id = match.group("left")
            right_id = match.group("right")
            additive = int(match.group("const") or 0)
            target_difference = target_span - additive
            left_current = getattr(collection.temporals.get(left_id), "year", None)
            right_current = getattr(collection.temporals.get(right_id), "year", None)
            if left_current is None or right_current is None:
                return False

            excluded_years = set(generator.exclude_temporals.get("years", set()))
            left_domain = generator._temporal_year_domain(
                left_id,
                *generator._temporal_year_base_range(left_id),
                excluded_years,
                set(),
            )
            right_domain = generator._temporal_year_domain(
                right_id,
                *generator._temporal_year_base_range(right_id),
                excluded_years,
                set(),
            )
            if not left_domain or not right_domain:
                return False

            candidate_pairs: list[tuple[int, int]] = []
            right_values = set(right_domain)
            for candidate_left in left_domain:
                candidate_right = candidate_left - target_difference
                if candidate_right not in right_values:
                    continue
                candidate_pairs.append((candidate_left, candidate_right))
            if not candidate_pairs:
                return False

            original_left = collection.temporals[left_id].model_copy(deep=True)
            original_right = collection.temporals[right_id].model_copy(deep=True)
            best_pair = min(
                candidate_pairs,
                key=lambda pair: abs(pair[0] - left_current) + abs(pair[1] - right_current),
            )
            for candidate_left, candidate_right in [best_pair] + [pair for pair in candidate_pairs if pair != best_pair]:
                collection.temporals[left_id] = generator._update_temporal_year(original_left, int(candidate_left))
                collection.temporals[right_id] = generator._update_temporal_year(original_right, int(candidate_right))
                if not all(is_valid for _, is_valid in RuleEngine.validate_all_rules(temporal_only_rules, collection)):
                    continue
                if not self._verify_ordering_preserved(
                    collection,
                    forced_equal_entity_ids=temporal_ordering_exempt_ids,
                ):
                    continue
                else:
                    return True
            collection.temporals[left_id] = original_left
            collection.temporals[right_id] = original_right
            return False

        for _ in range(2):
            changed = False
            for raw_rule in temporal_rules:
                cleaned = str(raw_rule or "").split("#", 1)[0].strip()
                if "temporal_" not in cleaned or "number_" not in cleaned or has_century_function(cleaned):
                    continue
                split = generator._split_rule(cleaned)
                if split is None:
                    continue
                lhs, op, rhs = split
                if op not in {"=", "=="}:
                    continue
                lhs_match = self._SINGLE_NUMBER_RULE_SIDE_RE.fullmatch(lhs)
                rhs_match = self._SINGLE_NUMBER_RULE_SIDE_RE.fullmatch(rhs)
                if bool(lhs_match) == bool(rhs_match):
                    continue
                number_match = lhs_match or rhs_match
                expr = rhs if lhs_match else lhs
                if "temporal_" not in expr:
                    continue
                target_value = coerce_integral_value(RuleEngine.evaluate_expression(expr, collection))
                if target_value is None:
                    continue
                if assign_number(str(number_match.group(1)), target_value):
                    changed = True
            if not changed:
                break

        for raw_rule in temporal_rules:
            cleaned = str(raw_rule or "").split("#", 1)[0].strip()
            match = self._TEMPORAL_PRODUCT_RULE_PATTERN.fullmatch(cleaned)
            if match is None:
                continue
            temporal_expr = match.group("left_expr") or match.group("right_expr") or ""
            source_number_id = match.group("left_number") or match.group("right_number")
            target_number_id = match.group("target_number")
            span_value = coerce_integral_value(RuleEngine.evaluate_expression(temporal_expr, collection))
            if span_value is None or span_value <= 0:
                continue
            source_domain = [value for value in candidate_domain(source_number_id) if value > 0]
            target_domain = candidate_domain(target_number_id)
            if not source_domain or not target_domain:
                continue
            current_source = current_int(source_number_id)
            current_target = current_int(target_number_id)

            candidate_pairs: list[tuple[int, int]] = []
            for candidate_source in source_domain:
                candidate_target = span_value * candidate_source
                if candidate_target not in target_domain:
                    continue
                candidate_pairs.append((candidate_source, candidate_target))
            if not candidate_pairs:
                for candidate_source in source_domain:
                    for candidate_target in target_domain:
                        if candidate_target % candidate_source != 0:
                            continue
                        candidate_span = candidate_target // candidate_source
                        if candidate_span <= 0:
                            continue
                        if not fit_temporal_span(temporal_expr, candidate_span):
                            continue
                        candidate_pairs.append((candidate_source, candidate_target))
                        break
                    if candidate_pairs:
                        break
            if not candidate_pairs:
                continue

            best_source, best_target = min(
                candidate_pairs,
                key=lambda pair: (
                    abs(pair[0] - (current_source if current_source is not None else pair[0]))
                    + abs(pair[1] - (current_target if current_target is not None else pair[1])),
                    abs(pair[1] - (current_target if current_target is not None else pair[1])),
                    abs(pair[0] - (current_source if current_source is not None else pair[0])),
                ),
            )
            assign_number(source_number_id, best_source)
            assign_number(target_number_id, best_target)

        for raw_rule in temporal_rules:
            cleaned = str(raw_rule or "").split("#", 1)[0].strip()
            if not cleaned or "temporal_" not in cleaned or "number_" not in cleaned:
                continue
            if has_century_function(cleaned):
                continue
            if self._TEMPORAL_PRODUCT_RULE_PATTERN.fullmatch(cleaned) is not None:
                continue
            number_refs = {
                ref.split(".", 1)[0]
                for ref in find_entity_refs(cleaned)
                if ref.startswith("number_")
            }
            if len(number_refs) != 1:
                continue
            number_id = next(iter(number_refs))
            original_number = collection.numbers.get(number_id)
            if original_number is None:
                continue
            temporal_ids = [
                ref.split(".", 1)[0]
                for ref in find_entity_refs(cleaned)
                if ref.startswith("temporal_") and ref.endswith(".year")
            ]
            current_value = current_int(number_id)
            factual_value = generator._factual_number_int(number_id)
            candidates = candidate_domain(number_id)
            if not candidates:
                continue
            candidates = sorted(
                candidates,
                key=lambda candidate: (
                    candidate == factual_value,
                    abs(candidate - (current_value if current_value is not None else candidate)),
                    candidate,
                ),
            )
            repaired = False
            original_snapshot = original_number.model_copy(deep=True)
            for candidate in candidates:
                if not assign_number(number_id, candidate):
                    continue
                if not RuleEngine.evaluate_expression(cleaned, collection):
                    continue
                if not self._verify_ordering_preserved(
                    collection,
                    forced_equal_entity_ids=temporal_ordering_exempt_ids,
                ):
                    continue
                repaired = True
                break
            if not repaired and len(set(temporal_ids)) == 2:
                left_id, right_id = list(dict.fromkeys(temporal_ids))
                left_original = collection.temporals.get(left_id)
                right_original = collection.temporals.get(right_id)
                if left_original is not None and right_original is not None:
                    excluded_years = set(generator.exclude_temporals.get("years", set()))
                    left_domain = generator._temporal_year_domain(
                        left_id,
                        *generator._temporal_year_base_range(left_id),
                        excluded_years,
                        set(),
                    )
                    right_domain = generator._temporal_year_domain(
                        right_id,
                        *generator._temporal_year_base_range(right_id),
                        excluded_years,
                        set(),
                    )
                    left_current = getattr(left_original, "year", None)
                    right_current = getattr(right_original, "year", None)
                    candidate_pairs = sorted(
                        (
                            (left_year, right_year)
                            for left_year in left_domain
                            for right_year in right_domain
                        ),
                        key=lambda pair: (
                            abs(pair[0] - (left_current if left_current is not None else pair[0]))
                            + abs(pair[1] - (right_current if right_current is not None else pair[1])),
                            abs(pair[1] - pair[0]),
                            pair[0],
                            pair[1],
                        ),
                    )
                    for candidate in candidates:
                        if not assign_number(number_id, candidate):
                            continue
                        for left_year, right_year in candidate_pairs:
                            collection.temporals[left_id] = generator._update_temporal_year(left_original, left_year)
                            collection.temporals[right_id] = generator._update_temporal_year(right_original, right_year)
                            if not RuleEngine.evaluate_expression(cleaned, collection):
                                continue
                            if not all(
                                is_valid for _, is_valid in RuleEngine.validate_all_rules(temporal_only_rules, collection)
                            ):
                                continue
                            if not self._verify_ordering_preserved(
                                collection,
                                forced_equal_entity_ids=temporal_ordering_exempt_ids,
                            ):
                                continue
                            repaired = True
                            break
                        if repaired:
                            break
                    if not repaired:
                        collection.temporals[left_id] = left_original
                        collection.temporals[right_id] = right_original
            if not repaired:
                collection.numbers[number_id] = original_snapshot

    def _get_age_bounds_from_rules(self) -> tuple[int, int]:
        min_age, max_age = _MIN_AGE, _MAX_AGE
        if not getattr(self, "_current_rules", None):
            return _DEFAULT_UNCONSTRAINED_MIN_AGE, _DEFAULT_UNCONSTRAINED_MAX_AGE
        found_bound = False
        for rule in self._current_rules:
            match = re.search(r"person_\d+(?:\.age)?\s*>\s*(\d+)", rule.strip())
            if match:
                min_age = max(min_age, int(match.group(1)) + 1)
                found_bound = True
            match = re.search(r"(\d+)\s*<\s*person_\d+(?:\.age)?", rule.strip())
            if match:
                min_age = max(min_age, int(match.group(1)) + 1)
                found_bound = True
            match = re.search(r"person_\d+(?:\.age)?\s*<\s*(\d+)", rule.strip())
            if match:
                max_age = min(max_age, int(match.group(1)) - 1)
                found_bound = True
            match = re.search(r"(\d+)\s*>\s*person_\d+(?:\.age)?", rule.strip())
            if match:
                max_age = min(max_age, int(match.group(1)) - 1)
                found_bound = True
        if not found_bound:
            min_age = max(min_age, _DEFAULT_UNCONSTRAINED_MIN_AGE)
            max_age = min(max_age, _DEFAULT_UNCONSTRAINED_MAX_AGE)
        if min_age > max_age:
            raise ValueError(f"Age constraints contradictory: min_age={min_age}, max_age={max_age}")
        return min_age, max_age

    def _factual_person_age(self, person_id: str) -> int | None:
        if not self.factual_entities or not self.factual_entities.persons:
            return None
        factual_person = self.factual_entities.persons.get(person_id)
        if factual_person is None:
            return None
        raw_age = (
            getattr(factual_person, "age", None) if not isinstance(factual_person, dict) else factual_person.get("age")
        )
        if raw_age is None:
            return None
        try:
            return int(raw_age)
        except (TypeError, ValueError):
            return None

    def _age_window(self, person_id: str, min_age: int, max_age: int) -> tuple[int, int]:
        implicit_age_range = getattr(self, "_implicit_age_range", None)
        if callable(implicit_age_range):
            overridden = implicit_age_range(person_id, min_age=min_age, max_age=max_age)
            if overridden is not None:
                return overridden
        factual_age = self._factual_person_age(person_id)
        if factual_age is None:
            return min_age, max_age
        min_age = min(min_age, factual_age)
        max_age = max(max_age, factual_age)
        low, high = _relative_int_window(factual_age, min_value=min_age, max_value=max_age)
        return low, high

    def _implicit_age_range(
        self,
        person_id: str,
        *,
        min_age: int,
        max_age: int,
    ) -> tuple[int, int] | None:
        rule = self._implicit_age_rules.get(person_id)
        if rule is None:
            return None
        low = max(min_age, math.ceil(float(rule.lower_bound)))
        high = min(max_age, math.floor(float(rule.upper_bound)))
        if low > high:
            factual = round(float(rule.factual_value))
            factual = max(min_age, min(factual, max_age))
            return factual, factual
        return low, high

    def _sample_person_age(self, person_id: str, min_age: int, max_age: int) -> int:
        low, high = self._age_window(person_id, min_age, max_age)
        factual_age = self._factual_person_age(person_id)
        if factual_age is None:
            return random.randint(low, high)
        candidates = [age for age in range(low, high + 1) if age != factual_age]
        if candidates:
            return random.choice(candidates)
        return random.randint(low, high)

    def _merge_factual_entities(self, collection: EntityCollection) -> None:
        merge_factual_entities(collection, self.factual_entities)

    def _validate_rules_with_details(self, rules: list[str], entities: EntityCollection) -> dict:
        return validate_rules_with_details(rules, entities, logger)

    def _verify_ordering_preserved(
        self,
        entities: EntityCollection,
        *,
        preserve_temporal_ordering: bool = True,
        preserve_number_ordering: bool = False,
        forced_equal_entity_ids: set[str] | None = None,
    ) -> bool:
        return verify_ordering_preserved(
            entities,
            self.factual_entities,
            preserve_temporal_ordering=preserve_temporal_ordering,
            preserve_number_ordering=preserve_number_ordering,
            forced_equal_entity_ids=forced_equal_entity_ids,
        )

    def _verify_required_differences(
        self,
        required_entities: dict[str, list[tuple[str, list[str]]]],
        sampled_entities: EntityCollection,
        forced_equal_entity_ids: set[str] | None = None,
        forced_equal_entity_refs: set[str] | None = None,
    ) -> bool:
        return verify_required_differences(
            required_entities=required_entities,
            sampled_entities=sampled_entities,
            factual_entities=self.factual_entities,
            person_diff_exempt_attrs=self._PERSON_DIFF_EXEMPT_ATTRS,
            forced_equal_entity_ids=forced_equal_entity_ids,
            forced_equal_entity_refs=forced_equal_entity_refs,
        )

    def _resample_equal_person_ages(
        self,
        collection: EntityCollection,
        required_entities: dict[str, list[tuple[str, list[str]]]],
    ) -> None:
        resample_equal_person_ages(
            collection=collection,
            required_entities=required_entities,
            factual_entities=self.factual_entities,
            age_bounds_getter=self._get_age_bounds_from_rules,
            age_window_getter=self._age_window,
            sample_person_age=self._sample_person_age,
        )

    def _preserve_age_ordering(self, collection: EntityCollection) -> None:
        preserve_age_ordering(collection, self.factual_entities)

    def _collect_unchanged_required_refs(
        self,
        required_entities: dict[str, list[tuple[str, list[str]]]],
        collection: EntityCollection,
        *,
        forced_equal_entity_ids: set[str] | None = None,
        forced_equal_entity_refs: set[str] | None = None,
    ) -> list[tuple[str, str, str, Any]]:
        unchanged: list[tuple[str, str, str, Any]] = []
        if not self.factual_entities:
            return unchanged
        forced_equal_entity_ids = set(forced_equal_entity_ids or set())
        forced_equal_entity_refs = set(forced_equal_entity_refs or set())
        for entity_type, specs in required_entities.items():
            for entity_id, attrs in specs:
                if entity_id in forced_equal_entity_ids:
                    continue
                for attr in attrs:
                    entity_ref = f"{entity_id}.{attr}"
                    if entity_ref in forced_equal_entity_refs:
                        continue
                    if not _attr_requires_difference(entity_type, attr, self._PERSON_DIFF_EXEMPT_ATTRS):
                        continue
                    factual_value = RuleEngine._get_entity_value(self.factual_entities, entity_ref)
                    fictional_value = RuleEngine._get_entity_value(collection, entity_ref)
                    if factual_value is None or fictional_value is None:
                        continue
                    if _values_equal(factual_value, fictional_value):
                        unchanged.append((entity_type, entity_id, attr, factual_value))
        return unchanged

    def _repair_required_difference_violations(
        self,
        *,
        generator,
        collection: EntityCollection,
        required_entities: dict[str, list[tuple[str, list[str]]]],
        rules_with_ordering: list[str],
        ordering_exempt_ids: set[str],
        fixed_required_refs: set[str],
    ) -> None:
        number_required_attr_map = {
            entity_id: set(attrs or [])
            for entity_id, attrs in required_entities.get("number", [])
        }
        for _ in range(max(1, len(number_required_attr_map))):
            unchanged = self._collect_unchanged_required_refs(
                required_entities,
                collection,
                forced_equal_entity_ids=ordering_exempt_ids,
                forced_equal_entity_refs=fixed_required_refs,
            )
            if not unchanged:
                return
            progress = False
            for entity_type, entity_id, attr, factual_value in unchanged:
                if entity_type == "number":
                    progress = self._repair_single_required_number_difference(
                        generator=generator,
                        collection=collection,
                        number_id=entity_id,
                        attr=attr,
                        factual_value=factual_value,
                        required_attrs=number_required_attr_map.get(entity_id, set()),
                        rules_with_ordering=rules_with_ordering,
                        ordering_exempt_ids=ordering_exempt_ids,
                    ) or progress
                elif entity_type == "temporal" and attr == "year":
                    progress = self._repair_single_required_temporal_year_difference(
                        generator=generator,
                        collection=collection,
                        temporal_id=entity_id,
                        factual_value=factual_value,
                        rules_with_ordering=rules_with_ordering,
                        ordering_exempt_ids=ordering_exempt_ids,
                    ) or progress
            if not progress:
                self._force_simple_required_differences(
                    generator=generator,
                    collection=collection,
                    unchanged=unchanged,
                    rules_with_ordering=rules_with_ordering,
                    number_required_attr_map=number_required_attr_map,
                    ordering_exempt_ids=ordering_exempt_ids,
                )
                return
        remaining_unchanged = self._collect_unchanged_required_refs(
            required_entities,
            collection,
            forced_equal_entity_ids=ordering_exempt_ids,
            forced_equal_entity_refs=fixed_required_refs,
        )
        if remaining_unchanged:
            self._force_simple_required_differences(
                generator=generator,
                collection=collection,
                unchanged=remaining_unchanged,
                rules_with_ordering=rules_with_ordering,
                number_required_attr_map=number_required_attr_map,
                ordering_exempt_ids=ordering_exempt_ids,
            )

    def _force_simple_required_differences(
        self,
        *,
        generator,
        collection: EntityCollection,
        unchanged: list[tuple[str, str, str, Any]],
        rules_with_ordering: list[str],
        number_required_attr_map: dict[str, set[str]],
        ordering_exempt_ids: set[str],
    ) -> None:
        for entity_type, entity_id, attr, factual_value in unchanged:
            if entity_type == "number":
                number_entity = collection.numbers.get(entity_id)
                if number_entity is None:
                    continue
                relevant_rules = self._rules_referencing_entity_ids(rules_with_ordering, {entity_id})
                original_snapshot = number_entity.model_copy(deep=True)
                required_attrs = number_required_attr_map.get(entity_id, set())
                low, high = generator._number_actual_bounds(entity_id, required_attrs)
                candidates = [
                    candidate
                    for candidate in range(int(math.ceil(low)), int(math.floor(high)) + 1)
                    if not _values_equal(factual_value, candidate)
                ]
                current_value = getattr(number_entity, "int", None)
                used_values = {
                    int(value)
                    for value in self.used_number_values_by_id.get(entity_id, set())
                    if isinstance(value, (int, float))
                }
                candidates.sort(
                    key=lambda candidate: (
                        candidate in used_values,
                        abs(candidate - int(current_value)) if current_value is not None else 0,
                        abs(candidate),
                    )
                )
                for candidate in candidates:
                    generator._set_number_actual_value(
                        entity_id,
                        number_entity,
                        float(candidate),
                        required_attrs=required_attrs,
                    )
                    fictional_value = RuleEngine._get_entity_value(collection, f"{entity_id}.{attr}")
                    if _values_equal(factual_value, fictional_value):
                        continue
                    validation_result = self._validate_rules_with_details(relevant_rules, collection)
                    if validation_result["all_valid"] and self._verify_ordering_preserved(
                        collection,
                        forced_equal_entity_ids=ordering_exempt_ids,
                    ):
                        break
                else:
                    collection.numbers[entity_id] = original_snapshot
            elif entity_type == "temporal" and attr == "year":
                temporal_entity = collection.temporals.get(entity_id)
                if temporal_entity is None or getattr(temporal_entity, "year", None) is None:
                    continue
                relevant_rules = self._rules_referencing_entity_ids(rules_with_ordering, {entity_id})
                original_snapshot = temporal_entity.model_copy(deep=True)
                excluded_years = set(generator.exclude_temporals.get("years", set()))
                candidates = [
                    candidate
                    for candidate in self._order_preserving_temporal_year_candidates(
                        generator=generator,
                        collection=collection,
                        temporal_id=entity_id,
                        excluded_years=excluded_years,
                        decade_year_temporal_ids=set(),
                    )
                    if not _values_equal(factual_value, candidate)
                ]
                current_value = getattr(temporal_entity, "year", None)
                candidates.sort(
                    key=lambda candidate: (
                        abs(candidate - int(current_value)) if current_value is not None else 0,
                        abs(candidate),
                    )
                )
                for candidate in candidates:
                    collection.temporals[entity_id] = generator._update_temporal_year(original_snapshot, int(candidate))
                    fictional_value = RuleEngine._get_entity_value(collection, f"{entity_id}.year")
                    if _values_equal(factual_value, fictional_value):
                        continue
                    validation_result = self._validate_rules_with_details(relevant_rules, collection)
                    if validation_result["all_valid"] and self._verify_ordering_preserved(
                        collection,
                        forced_equal_entity_ids=ordering_exempt_ids,
                    ):
                        break
                else:
                    collection.temporals[entity_id] = original_snapshot

    def _repair_single_required_number_difference(
        self,
        *,
        generator,
        collection: EntityCollection,
        number_id: str,
        attr: str,
        factual_value: Any,
        required_attrs: set[str],
        rules_with_ordering: list[str],
        ordering_exempt_ids: set[str],
    ) -> bool:
        number_entity = collection.numbers.get(number_id)
        if number_entity is None:
            return False
        relevant_number_rules = self._rules_referencing_entity_ids(rules_with_ordering, {number_id})
        original_snapshot = number_entity.model_copy(deep=True)
        original_temporals = {
            temporal_id: temporal.model_copy(deep=True)
            for temporal_id, temporal in collection.temporals.items()
        }
        low, high = generator._number_actual_bounds(number_id, required_attrs)
        candidates = [
            candidate
            for candidate in range(int(math.ceil(low)), int(math.floor(high)) + 1)
            if not _values_equal(factual_value, candidate)
        ]
        current_value = getattr(number_entity, "int", None)
        used_values = {
            int(value)
            for value in self.used_number_values_by_id.get(number_id, set())
            if isinstance(value, (int, float))
        }
        candidates.sort(
            key=lambda candidate: (
                candidate in used_values,
                abs(candidate - int(current_value)) if current_value is not None else 0,
                abs(candidate),
            )
        )
        for candidate in candidates:
            generator._set_number_actual_value(
                number_id,
                number_entity,
                float(candidate),
                required_attrs=required_attrs,
            )
            fictional_value = RuleEngine._get_entity_value(collection, f"{number_id}.{attr}")
            if _values_equal(factual_value, fictional_value):
                collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
                number_entity = collection.numbers[number_id]
                continue
            validation_result = self._validate_rules_with_details(relevant_number_rules, collection)
            if validation_result["all_valid"] and self._verify_ordering_preserved(
                collection,
                forced_equal_entity_ids=ordering_exempt_ids,
            ):
                return True
            collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
            number_entity = collection.numbers[number_id]
            collection.temporals = {
                temporal_id: temporal.model_copy(deep=True)
                for temporal_id, temporal in original_temporals.items()
            }
        mixed_rules = [
            str(rule)
            for rule in rules_with_ordering
            if number_id in str(rule) and "temporal_" in str(rule) and not has_century_function(str(rule))
        ]
        if not mixed_rules:
            generator.last_number_forced_equal_refs.update(generator._forced_equal_refs_for_number(number_id))
            return False

        excluded_years = set(generator.exclude_temporals.get("years", set()))
        temporal_domains_cache: dict[str, list[int]] = {}

        def temporal_year_domain(temporal_id: str) -> list[int]:
            cached = temporal_domains_cache.get(temporal_id)
            if cached is not None:
                return cached
            domain = self._order_preserving_temporal_year_candidates(
                generator=generator,
                collection=collection,
                temporal_id=temporal_id,
                excluded_years=excluded_years,
                decade_year_temporal_ids=set(),
            )
            temporal_domains_cache[temporal_id] = list(domain)
            return temporal_domains_cache[temporal_id]

        for candidate in candidates:
            generator._set_number_actual_value(
                number_id,
                number_entity,
                float(candidate),
                required_attrs=required_attrs,
            )
            fictional_value = RuleEngine._get_entity_value(collection, f"{number_id}.{attr}")
            if _values_equal(factual_value, fictional_value):
                collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
                number_entity = collection.numbers[number_id]
                continue

            repaired = True
            for mixed_rule in mixed_rules:
                cleaned = str(mixed_rule or "").split("#", 1)[0].strip()
                temporal_ids = list(
                    dict.fromkeys(
                        ref.split(".", 1)[0]
                        for ref in find_entity_refs(cleaned)
                        if ref.startswith("temporal_") and ref.endswith(".year")
                    )
                )
                if len(temporal_ids) != 2:
                    repaired = False
                    break
                left_id, right_id = temporal_ids
                left_original = original_temporals.get(left_id)
                right_original = original_temporals.get(right_id)
                if left_original is None or right_original is None:
                    repaired = False
                    break
                pair_ignore_ids = {left_id, right_id}
                left_domain = self._order_preserving_temporal_year_candidates(
                    generator=generator,
                    collection=collection,
                    temporal_id=left_id,
                    excluded_years=excluded_years,
                    decade_year_temporal_ids=set(),
                    ignore_temporal_ids=pair_ignore_ids,
                )
                right_domain = set(
                    self._order_preserving_temporal_year_candidates(
                        generator=generator,
                        collection=collection,
                        temporal_id=right_id,
                        excluded_years=excluded_years,
                        decade_year_temporal_ids=set(),
                        ignore_temporal_ids=pair_ignore_ids,
                    )
                )
                left_current = getattr(left_original, "year", None)
                right_current = getattr(right_original, "year", None)
                candidate_pairs = sorted(
                    (
                        (left_year, right_year)
                        for left_year in left_domain
                        for right_year in right_domain
                    ),
                    key=lambda pair: (
                        abs(pair[0] - (left_current if left_current is not None else pair[0]))
                        + abs(pair[1] - (right_current if right_current is not None else pair[1])),
                        abs(pair[1] - pair[0]),
                        pair[0],
                        pair[1],
                    ),
                )
                pair_found = False
                pair_relevant_rules = self._rules_referencing_entity_ids(
                    rules_with_ordering,
                    {number_id, left_id, right_id},
                )
                for left_year, right_year in candidate_pairs:
                    collection.temporals[left_id] = generator._update_temporal_year(left_original, int(left_year))
                    collection.temporals[right_id] = generator._update_temporal_year(right_original, int(right_year))
                    if not RuleEngine.evaluate_expression(cleaned, collection):
                        continue
                    validation_result = self._validate_rules_with_details(pair_relevant_rules, collection)
                    if validation_result["all_valid"] and self._verify_ordering_preserved(
                        collection,
                        forced_equal_entity_ids=ordering_exempt_ids,
                    ):
                        pair_found = True
                        break
                if not pair_found:
                    repaired = False
                    break
            if repaired:
                return True
            collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
            number_entity = collection.numbers[number_id]
            collection.temporals = {
                temporal_id: temporal.model_copy(deep=True)
                for temporal_id, temporal in original_temporals.items()
            }

        generator.last_number_forced_equal_refs.update(generator._forced_equal_refs_for_number(number_id))
        temporal_rules = [
            str(rule)
            for rule in rules_with_ordering
            if "temporal_" in str(rule) and not has_century_function(str(rule))
        ]
        temporal_specs = sorted(
            (
                temporal_id,
                ["year"],
            )
            for temporal_id, temporal in collection.temporals.items()
            if getattr(temporal, "year", None) is not None
        )
        if temporal_rules and temporal_specs:
            existing_without_temporals = EntityCollection(
                persons={key: value.model_copy(deep=True) for key, value in collection.persons.items()},
                places={key: value.model_copy(deep=True) for key, value in collection.places.items()},
                events={key: value.model_copy(deep=True) for key, value in collection.events.items()},
                organizations={key: value.model_copy(deep=True) for key, value in collection.organizations.items()},
                awards={key: value.model_copy(deep=True) for key, value in collection.awards.items()},
                legals={key: value.model_copy(deep=True) for key, value in collection.legals.items()},
                products={key: value.model_copy(deep=True) for key, value in collection.products.items()},
                numbers={key: value.model_copy(deep=True) for key, value in collection.numbers.items()},
                temporals={},
            )
            for candidate in candidates:
                generator._set_number_actual_value(
                    number_id,
                    number_entity,
                    float(candidate),
                    required_attrs=required_attrs,
                )
                existing_without_temporals.numbers[number_id] = collection.numbers[number_id].model_copy(deep=True)
                fictional_value = RuleEngine._get_entity_value(collection, f"{number_id}.{attr}")
                if _values_equal(factual_value, fictional_value):
                    collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
                    number_entity = collection.numbers[number_id]
                    existing_without_temporals.numbers[number_id] = number_entity.model_copy(deep=True)
                    continue
                try:
                    regenerated_temporals = generator.generate_temporals_with_rules(
                        temporal_specs,
                        self._concretize_temporal_rules_with_numbers(
                            generator=generator,
                            temporal_rules=temporal_rules,
                            collection=existing_without_temporals,
                        ),
                        existing_without_temporals,
                    )
                except Exception:
                    regenerated_temporals = None
                if regenerated_temporals:
                    collection.temporals = {
                        temporal_id: temporal.model_copy(deep=True)
                        for temporal_id, temporal in regenerated_temporals.items()
                    }
                    validation_result = self._validate_rules_with_details(rules_with_ordering, collection)
                    if validation_result["all_valid"] and self._verify_ordering_preserved(
                        collection,
                        forced_equal_entity_ids=ordering_exempt_ids,
                    ):
                        return True
                collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
                number_entity = collection.numbers[number_id]
                existing_without_temporals.numbers[number_id] = number_entity.model_copy(deep=True)
                collection.temporals = {
                    temporal_id: temporal.model_copy(deep=True)
                    for temporal_id, temporal in original_temporals.items()
                }
        return False

    def _repair_single_required_temporal_year_difference(
        self,
        *,
        generator,
        collection: EntityCollection,
        temporal_id: str,
        factual_value: Any,
        rules_with_ordering: list[str],
        ordering_exempt_ids: set[str],
    ) -> bool:
        temporal_entity = collection.temporals.get(temporal_id)
        if temporal_entity is None or getattr(temporal_entity, "year", None) is None:
            return False
        relevant_temporal_rules = self._rules_referencing_entity_ids(rules_with_ordering, {temporal_id})
        original_snapshot = temporal_entity.model_copy(deep=True)
        excluded_years = set(generator.exclude_temporals.get("years", set()))
        candidates = self._order_preserving_temporal_year_candidates(
            generator=generator,
            collection=collection,
            temporal_id=temporal_id,
            excluded_years=excluded_years,
            decade_year_temporal_ids=set(),
        )
        current_value = getattr(temporal_entity, "year", None)
        candidates = [candidate for candidate in candidates if not _values_equal(factual_value, candidate)]
        candidates.sort(
            key=lambda candidate: (
                abs(candidate - int(current_value)) if current_value is not None else 0,
                abs(candidate),
            )
        )
        for candidate in candidates:
            collection.temporals[temporal_id] = generator._update_temporal_year(original_snapshot, int(candidate))
            fictional_value = RuleEngine._get_entity_value(collection, f"{temporal_id}.year")
            if _values_equal(factual_value, fictional_value):
                collection.temporals[temporal_id] = original_snapshot.model_copy(deep=True)
                continue
            validation_result = self._validate_rules_with_details(relevant_temporal_rules, collection)
            if validation_result["all_valid"] and self._verify_ordering_preserved(
                collection,
                forced_equal_entity_ids=ordering_exempt_ids,
            ):
                return True
            collection.temporals[temporal_id] = original_snapshot.model_copy(deep=True)
        return False


__all__ = ["FictionalEntitySamplerNumericalMixin"]
