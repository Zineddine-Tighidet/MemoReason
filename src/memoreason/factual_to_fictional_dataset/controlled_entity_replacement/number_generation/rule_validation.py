"""Rule parsing and solving utilities for number generation."""

import re


from memoreason.benchmark_definition.century_expressions import has_century_function
from memoreason.benchmark_definition.document_schema import EntityCollection, NumberEntity
from memoreason.benchmark_definition.annotation_runtime import RuleEngine

LinearExpr = tuple[dict[str, float], float]
LinearConstraintTriple = tuple[LinearExpr, str, LinearExpr]


class NumberRuleValidationMixin:
    """Derive rule ranges and validate generated number assignments."""

    def _get_number_range_from_rules(
        self,
        var: str,
        rules: list[str],
        pre_assigned: dict[str, int],
        existing_entities: EntityCollection | None = None,
    ) -> tuple[int, int]:
        min_val, max_val = self._number_base_range(var)

        def get_value(var_name):
            if var_name in pre_assigned:
                return pre_assigned[var_name]
            if existing_entities:
                if var_name in existing_entities.numbers:
                    return self._number_entity_int_value(existing_entities.numbers[var_name])
                v = RuleEngine._get_entity_value(existing_entities, var_name)
                if v is not None:
                    try:
                        return int(v)
                    except (ValueError, TypeError):
                        return None
            return None

        for rule in rules:
            rule = self._normalize_number_rule(self._strip_rule_comment(str(rule)))
            if has_century_function(rule):
                continue
            split = self._split_rule(rule)
            if not split:
                continue
            lhs, op, rhs = split
            lhs_token = self._number_token(lhs)
            rhs_token = self._number_token(rhs)
            lhs_value = self._int_literal(lhs)
            rhs_value = self._int_literal(rhs)
            if lhs_value is None and lhs_token is not None and lhs_token != var:
                lhs_value = get_value(lhs_token)
            if rhs_value is None and rhs_token is not None and rhs_token != var:
                rhs_value = get_value(rhs_token)

            if lhs_token == var and rhs_value is not None:
                bound = int(rhs_value)
                if op == "<":
                    max_val = min(max_val, bound - 1)
                elif op == "<=":
                    max_val = min(max_val, bound)
                elif op == ">":
                    min_val = max(min_val, bound + 1)
                elif op == ">=":
                    min_val = max(min_val, bound)
                elif op in ("=", "=="):
                    min_val = max(min_val, bound)
                    max_val = min(max_val, bound)
            if rhs_token == var and lhs_value is not None:
                bound = int(lhs_value)
                if op == "<":
                    min_val = max(min_val, bound + 1)
                elif op == "<=":
                    min_val = max(min_val, bound)
                elif op == ">":
                    max_val = min(max_val, bound - 1)
                elif op == ">=":
                    max_val = min(max_val, bound)
                elif op in ("=", "=="):
                    min_val = max(min_val, bound)
                    max_val = min(max_val, bound)
        return min_val, max_val

    def _parse_equality_constraints(
        self,
        rules: list[str],
        number_ids: list[str],
    ) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
        """Parse equality and reverse-equality constraints from rules."""
        equality_constraints: dict[str, str] = {}
        reverse_equality_constraints: dict[str, tuple[str, str]] = {}
        for rule in rules:
            rule = self._normalize_number_rule(self._strip_rule_comment(str(rule)))
            if has_century_function(rule):
                continue
            split = self._split_rule(rule)
            if not split:
                continue
            lhs, op, rhs = split
            if op not in ("=", "=="):
                continue
            lhs_refs = {ref for ref in re.findall(r"\bnumber_\d+\b", lhs) if ref in number_ids}
            rhs_refs = {ref for ref in re.findall(r"\bnumber_\d+\b", rhs) if ref in number_ids}
            rhs_token = self._number_token(rhs)
            lhs_token = self._number_token(lhs)
            if (
                lhs_token in number_ids
                and rhs_token in number_ids
                and lhs.strip() == lhs_token
                and rhs.strip() == rhs_token
            ):
                equality_constraints[rhs_token] = lhs_token
                continue
            if rhs_token in number_ids:
                equality_constraints[rhs_token] = lhs
                for num_id in lhs_refs - {rhs_token}:
                    reverse_equality_constraints.setdefault(num_id, (lhs, rhs_token))
            if lhs_token in number_ids:
                equality_constraints[lhs_token] = rhs
                for num_id in rhs_refs - {lhs_token}:
                    reverse_equality_constraints.setdefault(num_id, (rhs, lhs_token))
        return equality_constraints, reverse_equality_constraints

    @staticmethod
    def _validate_numbers_against_rules(
        numbers: dict[str, NumberEntity],
        rules: list[str],
        existing_entities: EntityCollection | None = None,
    ) -> bool:
        test_collection = EntityCollection()
        if existing_entities:
            test_collection.persons = existing_entities.persons.copy()
            test_collection.places = existing_entities.places.copy()
            test_collection.events = existing_entities.events.copy()
            test_collection.organizations = existing_entities.organizations.copy()
            test_collection.temporals = existing_entities.temporals.copy()
            test_collection.numbers = existing_entities.numbers.copy()
            test_collection.numbers.update(numbers)
        else:
            test_collection.numbers = numbers
        validation = RuleEngine.validate_all_rules(rules, test_collection)
        return all(is_valid for _, is_valid in validation)


__all__ = ["NumberRuleValidationMixin"]
