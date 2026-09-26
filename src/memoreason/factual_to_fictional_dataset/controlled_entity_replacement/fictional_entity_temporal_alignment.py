"""Numerical and temporal generation stage for fictional sampling."""

from __future__ import annotations


from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs
from memoreason.benchmark_definition.document_schema import EntityCollection
from memoreason.benchmark_definition.entity_taxonomy import parse_integer_surface_number, parse_word_number

from .fictional_entity_sampler_common import logger


class TemporalRuleAlignmentMixin:
    """Concretize and align temporal rules with numerical entities."""

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
                    abs(
                        (pair[1] // pair[0])
                        - ((factual_target // factual_source) if factual_source else (pair[1] // pair[0]))
                    ),
                    abs(pair[1] - (current_target if current_target is not None else pair[1])),
                    abs(pair[0] - (current_source if current_source is not None else pair[0])),
                ),
            )
            assign_number(source_number_id, best_source)
            assign_number(target_number_id, best_target)


__all__ = ["TemporalRuleAlignmentMixin"]
