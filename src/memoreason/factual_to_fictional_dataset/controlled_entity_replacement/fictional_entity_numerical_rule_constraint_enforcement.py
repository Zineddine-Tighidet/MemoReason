"""Numerical and temporal generation stage for fictional sampling."""

from __future__ import annotations


from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs
from memoreason.benchmark_definition.document_schema import EntityCollection

from .linear_constraint_certification import transformed_constraints_use_safe_integer_lattice
from .number_temporal_generator import NumberTemporalGenerator
from .sampling_checks import (
    _values_equal,
)


class NumericalRuleConstraintEnforcementMixin:
    """Enforce document-specific coupled numerical constraints."""

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
        if constraints is None or not transformed_constraints_use_safe_integer_lattice(constraints):
            return set()
        domains = {number_id: generator._number_base_range(number_id) for number_id in sorted(required_number_ids)}
        tightened = generator._tighten_number_domains(constraints, domains)
        if tightened is None:
            return set()
        return {number_id for number_id, (low, high) in tightened.items() if int(low) == int(high)}

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
        collection: EntityCollection,
    ) -> set[str]:
        candidate_refs = self._fixed_required_refs_from_rules(rules)
        candidate_refs.update(getattr(generator, "last_number_forced_equal_refs", set()) or set())
        candidate_refs.update(getattr(generator, "last_implicit_forced_equal_refs", set()) or set())
        if self.factual_entities is None:
            return set()
        forced_refs: set[str] = set()
        for entity_ref in candidate_refs:
            factual_value = RuleEngine._get_entity_value(self.factual_entities, entity_ref)
            current_value = RuleEngine._get_entity_value(collection, entity_ref)
            if factual_value is None or current_value is None:
                continue
            if _values_equal(factual_value, current_value):
                forced_refs.add(entity_ref)
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


__all__ = ["NumericalRuleConstraintEnforcementMixin"]
