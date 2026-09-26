"""Numerical and temporal generation stage for fictional sampling."""

from __future__ import annotations

from typing import Any

from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs
from memoreason.benchmark_definition.century_expressions import has_century_function
from memoreason.benchmark_definition.document_schema import EntityCollection
from memoreason.benchmark_definition.entity_taxonomy import parse_integer_surface_number, parse_word_number


def _affine_two_year_candidate_pairs(
    *,
    generator,
    rule: str,
    collection: EntityCollection,
    left_id: str,
    right_id: str,
    left_domain: list[int],
    right_domain: list[int],
) -> list[tuple[int, int]] | None:
    """Solve a two-year affine equality without scanning a Cartesian product.

    ``None`` means the rule is outside the supported affine-equality contract;
    an empty list means it is supported but infeasible in the supplied domains.
    Number references are resolved from ``collection`` by the temporal parser,
    so the result is specific to the currently assigned numerical candidate.
    """
    split = generator._split_rule(rule)
    if split is None:
        return None
    lhs, operator, rhs = split
    if operator not in {"=", "=="}:
        return None

    required_ids = {left_id, right_id}
    left = generator._parse_temporal_linear_expr(lhs, required_ids, collection)
    right = generator._parse_temporal_linear_expr(rhs, required_ids, collection)
    if left is None or right is None:
        return None

    coefficients: dict[str, int] = {}
    for temporal_id in required_ids:
        raw_coefficient = left[0].get(temporal_id, 0) - right[0].get(temporal_id, 0)
        try:
            coefficient = int(raw_coefficient)
        except (TypeError, ValueError, OverflowError):
            return None
        if coefficient != raw_coefficient:
            return None
        coefficients[temporal_id] = coefficient
    if coefficients[left_id] == 0 or coefficients[right_id] == 0:
        return None

    raw_constant = left[1] - right[1]
    try:
        constant = int(raw_constant)
    except (TypeError, ValueError, OverflowError):
        return None
    if constant != raw_constant:
        return None

    left_coefficient = coefficients[left_id]
    right_coefficient = coefficients[right_id]
    right_values = set(right_domain)
    candidate_pairs: list[tuple[int, int]] = []
    for left_year in left_domain:
        right_numerator = -constant - (left_coefficient * left_year)
        if right_numerator % right_coefficient != 0:
            continue
        right_year = right_numerator // right_coefficient
        if right_year in right_values:
            candidate_pairs.append((left_year, right_year))
    return candidate_pairs


class MixedTemporalRuleConstraintEnforcementMixin:
    """Enforce rules that couple temporal and numerical entities."""

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
            str(rule or "").split("#", 1)[0].strip() for rule in temporal_rules if "number_" in str(rule)
        ]
        mixed_temporal_rules = [rule for rule in mixed_temporal_rules if rule]

        # If the current assignment already satisfies the mixed temporal-number rules,
        # avoid the expensive repair search entirely.
        if mixed_temporal_rules and all(
            RuleEngine.evaluate_expression(rule, collection) for rule in mixed_temporal_rules
        ):
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
            forbidden = {int(value) for value in avoid_numbers.get(number_id, set()) if isinstance(value, (int, float))}
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
            for candidate_left, candidate_right in [best_pair] + [
                pair for pair in candidate_pairs if pair != best_pair
            ]:
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
            number_refs = {ref.split(".", 1)[0] for ref in find_entity_refs(cleaned) if ref.startswith("number_")}
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
                    for candidate in candidates:
                        if not assign_number(number_id, candidate):
                            continue
                        candidate_pairs = _affine_two_year_candidate_pairs(
                            generator=generator,
                            rule=cleaned,
                            collection=collection,
                            left_id=left_id,
                            right_id=right_id,
                            left_domain=left_domain,
                            right_domain=right_domain,
                        )
                        if candidate_pairs is None:
                            # This repair stage is deliberately fail-closed for
                            # non-affine rules.  The outer sampler can retry;
                            # it must never perform an unbounded year product.
                            break
                        candidate_pairs.sort(
                            key=lambda pair: (
                                abs(pair[0] - (left_current if left_current is not None else pair[0]))
                                + abs(pair[1] - (right_current if right_current is not None else pair[1])),
                                abs(pair[1] - pair[0]),
                                pair[0],
                                pair[1],
                            )
                        )
                        for left_year, right_year in candidate_pairs:
                            collection.temporals[left_id] = generator._update_temporal_year(left_original, left_year)
                            collection.temporals[right_id] = generator._update_temporal_year(right_original, right_year)
                            if not RuleEngine.evaluate_expression(cleaned, collection):
                                continue
                            if not all(
                                is_valid
                                for _, is_valid in RuleEngine.validate_all_rules(temporal_only_rules, collection)
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


__all__ = ["MixedTemporalRuleConstraintEnforcementMixin"]
