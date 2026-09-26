"""Numerical and temporal generation stage for fictional sampling."""

from __future__ import annotations

import time

from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs
from memoreason.benchmark_definition.century_expressions import has_century_function
from memoreason.benchmark_definition.document_schema import EntityCollection

from .fictional_entity_sampler_common import DEBUG_SAMPLING, logger
from .generation_limits import SLOW_STAGE_LOG_SECONDS
from .generation_exceptions import StrictInterVariantUniquenessInfeasible
from .number_temporal_generator import NumberTemporalGenerator
from .intervariant_numtemp_uniqueness import IntervariantNumtempUniquenessMixin
from .temporal_exclusion_planning import build_temporal_exclusions


class NumericalGenerationMixin(IntervariantNumtempUniquenessMixin):
    """Main numerical and temporal entity generation stage."""

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
        derived_ordering_rules = self._build_number_ordering_rules(
            required_entities.get("number", []),
            excluded_number_ids=explicit_number_rule_ids,
        )
        rules_with_ordering = rules + derived_ordering_rules
        self._current_rules = rules_with_ordering
        numeric_rules = [
            rule
            for rule in rules_with_ordering
            if "number_" in rule and "temporal_" not in rule and not has_century_function(str(rule))
        ]
        fixed_number_ids = self._fixed_number_ids_from_rules(rules_with_ordering)
        if not self.allow_factual_numtemp_values:
            numeric_rules += self._required_number_factual_difference_rules(
                auto_required.get("number", []), fixed_number_ids
            )
        mixed_temporal_number_ids = self._mixed_temporal_number_component(rules)
        century_rules = [rule for rule in rules_with_ordering if has_century_function(str(rule))]
        simple_rules = all(self._is_simple_rule(str(rule)) for rule in rules_with_ordering)
        if simple_rules:
            max_attempts = min(max_attempts, 2)

        self.last_relaxed_numtemp_reuse_audit: list[dict[str, object]] = []
        best_relaxed_candidate: tuple[tuple[object, ...], EntityCollection, list[dict[str, object]]] | None = None
        exclude_temporals = build_temporal_exclusions(
            factual_entities=self.factual_entities,
            used_years_by_id=self.used_temporal_years_by_id,
            used_values_by_id=self.used_temporal_values_by_id,
        )

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
                generator = NumberTemporalGenerator(
                    seed=(self.seed + attempt) if self.seed is not None else None,
                    exclude_numbers=set(),
                    exclude_temporals=exclude_temporals,
                    factual_entities=self.factual_entities,
                    implicit_rules=self.implicit_rules,
                    ordering_excluded_number_ids=ordering_excluded_number_ids,
                )
                generator.reference_variant_index = self.reference_variant_index
                generator.reference_variant_count = self.reference_variant_count
                allowed_temporal_reuse_refs: set[tuple[str, str]] = set()
                if self.allow_factual_numtemp_values:
                    for temporal_id, attrs in auto_required.get("temporal", []):
                        for attr in attrs:
                            relaxed_attrs = ("year", "month", "day", "day_of_month") if attr == "date" else (attr,)
                            for normalized_attr in relaxed_attrs:
                                if normalized_attr not in {"year", "month", "day", "day_of_month", "timestamp"}:
                                    continue
                                allowed_temporal_reuse_refs.add((str(temporal_id), normalized_attr))
                                exclude_key = {
                                    "year": "years_by_id",
                                    "month": "months_by_id",
                                    "day": "days_by_id",
                                    "day_of_month": "day_of_months_by_id",
                                    "timestamp": "timestamps_by_id",
                                }[normalized_attr]
                                (generator.exclude_temporals.get(exclude_key) or {}).pop(str(temporal_id), None)
                elif self.allow_relaxed_intervariant_number_reuse:
                    allowed_temporal_reuse_refs = self._saturated_temporal_reuse_refs(
                        generator=generator,
                        required_temporals=auto_required.get("temporal", []),
                        decade_year_temporal_ids=set(decade_year_temporal_ids or set()),
                        has_temporal_rules=any("temporal_" in str(rule) for rule in rules_with_ordering),
                    )
                    years_by_id = generator.exclude_temporals.get("years_by_id") or {}
                    for temporal_id, attr in allowed_temporal_reuse_refs:
                        if attr == "year":
                            years_by_id.pop(temporal_id, None)
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
                if not self.allow_factual_numtemp_values and self.factual_entities and self.factual_entities.numbers:
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
                required_number_ids = {number_id for number_id, _ in auto_required.get("number", [])}
                solver_numeric_rules = self._linear_number_rules_for_solver(
                    generator=generator,
                    rules=numeric_rules,
                    required_number_ids=required_number_ids,
                    existing_entities=existing_collection,
                )
                relaxed_intervariant_reuse_number_ids: set[str] = set()
                pre_solver_relaxed_intervariant_reuse_number_ids: set[str] = set()
                if self.allow_relaxed_intervariant_number_reuse:
                    relaxed_required_number_ids = (
                        set(required_number_ids)
                        if self.allow_factual_numtemp_values
                        else set(mixed_temporal_number_ids)
                    )
                    relaxed_required_number_ids -= fixed_number_ids
                    relaxed_intervariant_reuse_number_ids.update(relaxed_required_number_ids)
                    pre_solver_relaxed_intervariant_reuse_number_ids.update(relaxed_required_number_ids)
                    hard_bounded_saturated_ids = self._saturated_hard_bounded_linear_number_ids(
                        generator=generator,
                        required_number_ids=required_number_ids,
                        rules=solver_numeric_rules,
                        existing_entities=existing_collection,
                    )
                    relaxed_intervariant_reuse_number_ids.update(hard_bounded_saturated_ids)
                    pre_solver_relaxed_intervariant_reuse_number_ids.update(hard_bounded_saturated_ids)
                    if len(solver_numeric_rules) < len(numeric_rules):
                        nonlinear_saturated_ids = self._saturated_low_cardinality_number_ids(
                            generator=generator,
                            required_number_ids=required_number_ids,
                            rules=numeric_rules,
                            existing_entities=existing_collection,
                        )
                        relaxed_intervariant_reuse_number_ids.update(nonlinear_saturated_ids)
                        pre_solver_relaxed_intervariant_reuse_number_ids.update(nonlinear_saturated_ids)
                for num_id, used_values in (self.used_number_values_by_id or {}).items():
                    if not used_values:
                        continue
                    if num_id not in {number_id for number_id, _ in auto_required.get("number", [])}:
                        continue
                    if num_id in fixed_number_ids:
                        continue
                    if num_id in pre_solver_relaxed_intervariant_reuse_number_ids:
                        continue
                    avoid_numbers.setdefault(num_id, set()).update(used_values)
                num_start = time.monotonic()
                has_intervariant_number_avoid = any(
                    used_values
                    for num_id, used_values in (self.used_number_values_by_id or {}).items()
                    if num_id in required_number_ids
                )
                generator.allow_relaxed_factual_avoid_solution = (
                    not has_intervariant_number_avoid or self.allow_relaxed_intervariant_number_reuse
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
                if self.allow_relaxed_intervariant_number_reuse and generator.last_number_solution_used_relaxed_avoid:
                    relaxed_intervariant_reuse_number_ids.update(
                        number_id
                        for number_id, entity in collection.numbers.items()
                        if self._number_intervariant_value(entity)
                        in set(self.used_number_values_by_id.get(number_id) or set())
                    )
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
                        f"[dbg] generated numbers for attempt {attempt + 1}: {sorted(collection.numbers.keys())}",
                        flush=True,
                    )
                existing_collection.numbers.update(collection.numbers)
                num_elapsed = time.monotonic() - num_start

                temporal_rules = [
                    rule for rule in rules_with_ordering if "temporal_" in rule and not has_century_function(str(rule))
                ]
                mixed_temporal_rules = [
                    str(rule or "").split("#", 1)[0].strip() for rule in temporal_rules if "number_" in str(rule)
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
                        f"[dbg] generated temporals for attempt {attempt + 1}: {sorted(collection.temporals.keys())}",
                        flush=True,
                    )
                temp_elapsed = time.monotonic() - temp_start

                if self.factual_entities:
                    self._merge_factual_entities(collection)
                century_forced_equal_ids: set[str] = set()  # noqa: F841
                implicit_forced_equal_refs: set[str] = set()  # noqa: F841
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
                    collection=collection,
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
                    if not self.allow_factual_numtemp_values:
                        self._repair_required_difference_violations(
                            generator=generator,
                            collection=collection,
                            required_entities=required_entities,
                            rules_with_ordering=rules_with_ordering,
                            ordering_exempt_ids=ordering_exempt_ids,
                            fixed_required_refs=fixed_required_refs,
                            allowed_number_reuse_ids=relaxed_intervariant_reuse_number_ids,
                        )
                    fixed_required_refs = self._current_forced_required_refs(
                        generator=generator,
                        rules=rules_with_ordering,
                        collection=collection,
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
                differences_ok = self.allow_factual_numtemp_values or self._verify_required_differences(
                    required_entities,
                    collection,
                    forced_equal_entity_refs=fixed_required_refs,
                )
                runtime_reuse_audit = (
                    self._runtime_relaxed_number_reuse_audit(
                        collection=collection,
                        required_numbers=auto_required.get("number", []),
                        allowed_number_reuse_ids=relaxed_intervariant_reuse_number_ids,
                        fixed_required_refs=fixed_required_refs,
                    )
                    + self._runtime_relaxed_temporal_reuse_audit(
                        collection=collection,
                        required_temporals=auto_required.get("temporal", []),
                        statically_allowed_refs=allowed_temporal_reuse_refs,
                        fixed_required_refs=fixed_required_refs,
                    )
                    if self.allow_relaxed_intervariant_number_reuse
                    else []
                )
                runtime_allowed_temporal_refs = {
                    (str(item["entity_ref"]), str(item["entity_attr"]))
                    for item in runtime_reuse_audit
                    if item.get("entity_bucket") == "temporals"
                }
                intervariant_values_ok = self._intervariant_numtemp_values_are_valid(
                    collection=collection,
                    required_numbers=auto_required.get("number", []),
                    required_temporals=auto_required.get("temporal", []),
                    allowed_number_reuse_ids=relaxed_intervariant_reuse_number_ids,
                    allowed_temporal_reuse_refs=allowed_temporal_reuse_refs | runtime_allowed_temporal_refs,
                    fixed_required_refs=fixed_required_refs,
                )
                if validation_result["all_valid"] and ordering_ok and differences_ok and intervariant_values_ok:
                    total_elapsed = time.monotonic() - attempt_start
                    if total_elapsed > SLOW_STAGE_LOG_SECONDS:
                        print(
                            "[slow] Sampling attempt "
                            f"{attempt + 1} slow ({total_elapsed:.1f}s): "
                            f"numbers={num_elapsed:.1f}s temporals={temp_elapsed:.1f}s",
                            flush=True,
                        )
                    candidate = EntityCollection(
                        numbers={key: value.model_copy(deep=True) for key, value in collection.numbers.items()},
                        temporals={key: value.model_copy(deep=True) for key, value in collection.temporals.items()},
                    )
                    if runtime_reuse_audit:
                        # Minimize reuse across the deterministic relaxed attempts.
                        score = self._runtime_reuse_candidate_score(runtime_reuse_audit)
                        if best_relaxed_candidate is None or score < best_relaxed_candidate[0]:
                            best_relaxed_candidate = (score, candidate, runtime_reuse_audit)
                        continue
                    self.last_relaxed_numtemp_reuse_audit = []
                    return candidate
                if DEBUG_SAMPLING and attempt < 3:
                    failed_rules = validation_result.get("failed_rules") or []
                    print(
                        f"[dbg] sample_fictional_entities reject (attempt {attempt + 1}): "
                        f"failed_rules={failed_rules} ordering_ok={ordering_ok} "
                        f"differences_ok={differences_ok} intervariant_values_ok={intervariant_values_ok}",
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
        if best_relaxed_candidate is not None:
            _score, candidate, audit = best_relaxed_candidate
            self.last_relaxed_numtemp_reuse_audit = [dict(item) for item in audit]
            return candidate
        return None


__all__ = ["NumericalGenerationMixin"]
