"""Scoped proofs for numeric and temporal inter-variant reuse."""

from __future__ import annotations

import json

from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs
from memoreason.benchmark_definition.document_schema import EntityCollection

from .linear_constraint_certification import transformed_constraints_use_safe_integer_lattice
from .number_temporal_generator import NumberTemporalGenerator
from .number_uniqueness import number_entity_uniqueness_payload, number_entity_uniqueness_value

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class IntervariantNumtempUniquenessMixin:
    """Validate that relaxed reuse is limited to exhausted local domains."""

    @staticmethod
    def _mixed_temporal_number_component(rules: list[str]) -> set[str]:
        """Return numbers transitively coupled to a temporal-number rule."""
        number_sets: list[set[str]] = []
        connected: set[str] = set()
        for raw_rule in rules:
            cleaned = str(raw_rule or "").split("#", 1)[0].strip()
            number_ids = {ref.split(".", 1)[0] for ref in find_entity_refs(cleaned) if ref.startswith("number_")}
            if not number_ids:
                continue
            number_sets.append(number_ids)
            if "temporal_" in cleaned:
                connected.update(number_ids)
        changed = True
        while changed:
            changed = False
            for number_ids in number_sets:
                if connected.intersection(number_ids) and not number_ids.issubset(connected):
                    connected.update(number_ids)
                    changed = True
        return connected

    def _required_number_factual_difference_rules(
        self,
        required_numbers: list[tuple[str, list[str]]],
        fixed_number_ids: set[str],
    ) -> list[str]:
        """Express the final no-own-factual contract as hard solver rules."""
        if self.factual_entities is None:
            return []
        rules: list[str] = []
        numeric_attrs = {"int", "str", "float", "percent", "proportion"}
        for number_id, attrs in required_numbers:
            entity = self.factual_entities.numbers.get(number_id)
            value = self._number_intervariant_value(entity) if entity is not None else None
            attr = next((item for item in attrs if item in numeric_attrs), None)
            if number_id not in fixed_number_ids and value is not None and attr is not None:
                rules.append(f"{number_id}.{attr} != {value!r}")
        return rules

    @staticmethod
    def _number_intervariant_value(entity) -> int | float | None:
        return number_entity_uniqueness_value(entity)

    def _prior_runtime_reuse_occurrence_count(self, item: dict[str, object]) -> int:
        """Count prior occurrences of an already-used value, including audits."""
        signature = json.dumps(item.get("value") or {}, sort_keys=True, ensure_ascii=True)
        extra_reuses = sum(
            1
            for prior in getattr(self, "prior_relaxed_intervariant_reuse_audit", [])
            if str(prior.get("entity_bucket") or "") == str(item.get("entity_bucket") or "")
            and str(prior.get("entity_ref") or "") == str(item.get("entity_ref") or "")
            and str(prior.get("entity_attr") or "") == str(item.get("entity_attr") or "")
            and json.dumps(prior.get("value") or {}, sort_keys=True, ensure_ascii=True) == signature
        )
        return 1 + extra_reuses

    def _runtime_reuse_candidate_score(self, audit: list[dict[str, object]]) -> tuple[object, ...]:
        """Prefer fewer reused fields, then least-used values, then canonical values."""
        return (
            len(audit),
            sum(self._prior_runtime_reuse_occurrence_count(item) for item in audit),
            tuple((str(item["entity_ref"]), str(item["entity_attr"]), repr(item["value"])) for item in audit),
        )

    @staticmethod
    def _effective_temporal_attrs(attrs: list[str] | set[str]) -> set[str]:
        effective = set(attrs or [])
        if "date" in effective:
            effective.update({"year", "month", "day_of_month"})
        return effective & {"year", "month", "day", "day_of_month", "timestamp"}

    def _saturated_temporal_reuse_refs(
        self,
        *,
        generator: NumberTemporalGenerator,
        required_temporals: list[tuple[str, list[str]]],
        decade_year_temporal_ids: set[str],
        has_temporal_rules: bool,
    ) -> set[tuple[str, str]]:
        """Return only temporal attributes whose local unused domain is exhausted."""
        saturated: set[tuple[str, str]] = set()
        global_candidates = {
            "day": set(_WEEKDAYS),
            "month": set(generator._MONTHS),
            "day_of_month": set(range(1, 29)),
        }

        for temporal_id, attrs in required_temporals:
            effective_attrs = self._effective_temporal_attrs(attrs)
            used_by_attr = self.used_temporal_values_by_id.get(temporal_id, {}) or {}
            for attr in effective_attrs & {"day", "month", "day_of_month"}:
                # Global factual values are only a first-pass preference in
                # temporal generation.  Capacity is therefore the complete
                # surface domain minus this ref's own hard factual exclusion.
                candidates = set(global_candidates[attr])
                factual_temporal = (
                    self.factual_entities.temporals.get(temporal_id) if self.factual_entities is not None else None
                )
                factual_value = getattr(factual_temporal, attr, None) if factual_temporal is not None else None
                if factual_value not in (None, ""):
                    candidates.discard(factual_value)
                used_values = set(used_by_attr.get(attr) or set())
                if candidates and candidates.issubset(used_values):
                    saturated.add((temporal_id, attr))

            if "year" not in effective_attrs:
                continue
            if has_temporal_rules:
                # Both factual-relative and annotated implicit year ranges are
                # soft windows for the exact temporal solver.  With a temporal
                # rule, the v09 solver can expand them to preserve joint
                # feasibility without reusing sibling years, so local
                # exhaustion is not a valid capacity certificate.  Without a
                # temporal rule, ordered-year generation uses this local
                # domain exactly and exhaustion can be certified (space_12).
                continue
            low, high = generator._temporal_year_base_range(temporal_id)
            factual_year = generator._factual_temporal_year(temporal_id)
            globally_excluded_years = {
                int(value) for value in (generator.exclude_temporals.get("years") or set()) if value is not None
            }
            candidates = {
                year
                for year in range(int(low), int(high) + 1)
                if (factual_year is not None and year != int(factual_year))
                or (factual_year is None and year not in globally_excluded_years)
            }
            if temporal_id in decade_year_temporal_ids:
                candidates = {year for year in candidates if year % 10 == 0}
            used_years = {
                int(value)
                for value in (
                    set(self.used_temporal_years_by_id.get(temporal_id) or set())
                    | set(used_by_attr.get("year") or set())
                )
                if value is not None
            }
            if candidates and candidates.issubset(used_years):
                saturated.add((temporal_id, "year"))
        return saturated

    def _intervariant_numtemp_values_are_valid(
        self,
        *,
        collection: EntityCollection,
        required_numbers: list[tuple[str, list[str]]],
        required_temporals: list[tuple[str, list[str]]],
        allowed_number_reuse_ids: set[str],
        allowed_temporal_reuse_refs: set[tuple[str, str]],
        fixed_required_refs: set[str],
    ) -> bool:
        for number_id, _attrs in required_numbers:
            entity = collection.numbers.get(number_id)
            value = self._number_intervariant_value(entity) if entity is not None else None
            if value is None or value not in set(self.used_number_values_by_id.get(number_id) or set()):
                continue
            if number_id in allowed_number_reuse_ids:
                continue
            if any(ref.startswith(f"{number_id}.") for ref in fixed_required_refs):
                continue
            return False

        for temporal_id, attrs in required_temporals:
            entity = collection.temporals.get(temporal_id)
            if entity is None:
                continue
            used_by_attr = self.used_temporal_values_by_id.get(temporal_id, {}) or {}
            for attr in self._effective_temporal_attrs(attrs):
                value = getattr(entity, attr, None)
                if value in (None, ""):
                    continue
                used_values = set(used_by_attr.get(attr) or set())
                if attr == "year":
                    used_values.update(self.used_temporal_years_by_id.get(temporal_id) or set())
                if value not in used_values:
                    continue
                if (temporal_id, attr) in allowed_temporal_reuse_refs:
                    continue
                if f"{temporal_id}.{attr}" in fixed_required_refs:
                    continue
                return False
        return True

    def _runtime_relaxed_temporal_reuse_audit(
        self,
        *,
        collection: EntityCollection,
        required_temporals: list[tuple[str, list[str]]],
        statically_allowed_refs: set[tuple[str, str]],
        fixed_required_refs: set[str],
    ) -> list[dict[str, object]]:
        """Describe exact prior temporal values used by this relaxed candidate.

        The caller reaches this helper only after the complete strict sampler
        call failed.  Capacity-certified and explicitly fixed attributes are
        handled by their existing proof paths; this audit is limited to the
        additional per-component reuse needed by the relaxed candidate.
        """
        audit: list[dict[str, object]] = []
        for temporal_id, attrs in required_temporals:
            entity = collection.temporals.get(temporal_id)
            if entity is None:
                continue
            factual = self.factual_entities.temporals.get(temporal_id) if self.factual_entities is not None else None
            used_by_attr = self.used_temporal_values_by_id.get(temporal_id, {}) or {}
            for attr in sorted(self._effective_temporal_attrs(attrs)):
                value = getattr(entity, attr, None)
                if value in (None, ""):
                    continue
                used_values = set(used_by_attr.get(attr) or set())
                if attr == "year":
                    used_values.update(self.used_temporal_years_by_id.get(temporal_id) or set())
                if value not in used_values:
                    continue
                ref = (str(temporal_id), attr)
                if ref in statically_allowed_refs or f"{temporal_id}.{attr}" in fixed_required_refs:
                    continue
                own_factual_value = getattr(factual, attr, None) if factual is not None else None
                # Never turn the scientific "different from its own factual
                # value" contract into a runtime reuse exemption.
                if own_factual_value not in (None, "") and value == own_factual_value:
                    continue
                audit.append(
                    {
                        "entity_bucket": "temporals",
                        "entity_ref": str(temporal_id),
                        "entity_attr": attr,
                        "value": {attr: value},
                        "own_factual_value": (
                            {attr: own_factual_value} if own_factual_value not in (None, "") else None
                        ),
                        "reason": "strict_sampler_budget_exhausted_then_minimum_reuse_relaxed_attempt",
                    }
                )
        return audit

    def _runtime_relaxed_number_reuse_audit(
        self,
        *,
        collection: EntityCollection,
        required_numbers: list[tuple[str, list[str]]],
        allowed_number_reuse_ids: set[str],
        fixed_required_refs: set[str],
    ) -> list[dict[str, object]]:
        """Describe the exact scoped number reuse in one relaxed candidate."""
        audit: list[dict[str, object]] = []
        for number_id, _attrs in required_numbers:
            if number_id not in allowed_number_reuse_ids:
                continue
            if any(ref.startswith(f"{number_id}.") for ref in fixed_required_refs):
                continue
            entity = collection.numbers.get(number_id)
            value = self._number_intervariant_value(entity) if entity is not None else None
            if value is None or value not in set(self.used_number_values_by_id.get(number_id) or set()):
                continue
            factual = self.factual_entities.numbers.get(number_id) if self.factual_entities is not None else None
            own_factual_value = self._number_intervariant_value(factual) if factual is not None else None
            if own_factual_value is not None and value == own_factual_value:
                continue
            normalized_value = number_entity_uniqueness_payload(entity)
            if normalized_value is None:
                continue
            audit.append(
                {
                    "entity_bucket": "numbers",
                    "entity_ref": str(number_id),
                    "entity_attr": "",
                    "value": normalized_value,
                    "own_factual_value": (number_entity_uniqueness_payload(factual) if factual is not None else None),
                    "reason": "strict_sampler_budget_exhausted_then_minimum_reuse_relaxed_attempt",
                }
            )
        return audit

    def _saturated_low_cardinality_number_ids(
        self,
        *,
        generator: NumberTemporalGenerator,
        required_number_ids: set[str],
        rules: list[str],
        existing_entities: EntityCollection,
    ) -> set[str]:
        """Certify exhaustion for exactly enumerable unary nonlinear rules."""
        saturated: set[str] = set()
        low_cardinality_ids = self._low_cardinality_number_ids(
            generator=generator,
            required_number_ids=required_number_ids,
        )
        for number_id in low_cardinality_ids:
            if generator._factual_number_kind(number_id) != "int":
                continue
            low, high = generator._number_base_range(number_id)
            factual_value = generator._factual_number_int(number_id)
            candidates = {
                value
                for value in range(int(low), int(high) + 1)
                if factual_value is None or value != int(factual_value)
            }
            used_values = generator._coerce_forbidden_number_values(self.used_number_values_by_id.get(number_id))
            if not candidates or not candidates.issubset(used_values):
                continue

            relevant_rules: list[str] = []
            unary_rules_are_evaluable = True
            for raw_rule in rules:
                cleaned = str(raw_rule or "").split("#", 1)[0].strip()
                refs = find_entity_refs(cleaned)
                number_refs = {ref.split(".", 1)[0] for ref in refs if ref.startswith("number_")}
                if number_id not in number_refs:
                    continue
                if number_refs != {number_id} or not generator._number_rule_is_evaluable(
                    cleaned,
                    {number_id},
                    existing_entities,
                ):
                    unary_rules_are_evaluable = False
                    break
                relevant_rules.append(cleaned)
            if not unary_rules_are_evaluable or not relevant_rules:
                continue

            # Number ranges, including annotated implicit ranges, deliberately
            # expand to avoid prior variants.  Reproduce the widest exact
            # domain reachable by that workflow and enumerate every novel
            # candidate against the unary rules before certifying exhaustion.
            forbidden_values = set(used_values)
            if factual_value is not None:
                forbidden_values.add(int(factual_value))
            min_value = 0 if factual_value == 0 else 1
            expanded_low, expanded_high = generator._expand_int_domain_to_escape_forbidden(
                int(low),
                int(high),
                avoid=forbidden_values,
                min_value=min_value,
            )
            max_padding = max(getattr(generator, "_EXACT_AVOID_EXPANSION_STEPS", (0,)))
            reachable_low = max(min_value, int(expanded_low) - int(max_padding))
            reachable_high = int(expanded_high) + int(max_padding)
            candidate_collection = existing_entities.model_copy(deep=True)
            novel_feasible_value_exists = False
            for candidate in range(reachable_low, reachable_high + 1):
                if candidate in forbidden_values:
                    continue
                candidate_collection.numbers[number_id] = generator._build_number_entity(number_id, candidate)
                try:
                    if all(RuleEngine.evaluate_expression(rule, candidate_collection) for rule in relevant_rules):
                        novel_feasible_value_exists = True
                        break
                except Exception:
                    unary_rules_are_evaluable = False
                    break
            if unary_rules_are_evaluable and not novel_feasible_value_exists:
                saturated.add(number_id)
        return saturated

    def _saturated_hard_bounded_linear_number_ids(
        self,
        *,
        generator: NumberTemporalGenerator,
        required_number_ids: set[str],
        rules: list[str],
        existing_entities: EntityCollection,
    ) -> set[str]:
        """Certify exhaustion only when explicit linear rules impose a finite upper bound."""
        if not required_number_ids or not rules:
            return set()
        constraints = generator._collect_linear_constraints(rules, required_number_ids, existing_entities)
        if constraints is None or not constraints:
            return set()
        if not transformed_constraints_use_safe_integer_lattice(constraints):
            # Domain tightening uses a one-unit gap for strict comparisons.
            # Without an integer-lattice proof, a fractional bound such as
            # ``number_1 < 2.5`` must not turn the still-valid value 2 into a
            # false capacity-exhaustion certificate.
            return set()

        # Implicit annotation ranges are intentionally expandable to preserve
        # inter-variant uniqueness. Probe twice from much wider domains so a
        # bound inherited only from the artificial probe ceiling is not
        # mistaken for a hard rule bound.
        probe_high = 10**12
        probe_domains = {
            number_id: (0 if generator._factual_number_int(number_id) == 0 else 1, probe_high)
            for number_id in sorted(required_number_ids)
        }
        first = generator._tighten_number_domains(constraints, probe_domains)
        second = generator._tighten_number_domains(
            constraints,
            {number_id: (low, probe_high * 2) for number_id, (low, _high) in probe_domains.items()},
        )
        if first is None or second is None:
            return set()

        saturated: set[str] = set()
        for number_id in sorted(required_number_ids):
            low = int(probe_domains[number_id][0])
            first_high = int(first[number_id][1])
            second_high = int(second[number_id][1])
            if first_high != second_high or first_high >= probe_high:
                continue
            factual_value = generator._factual_number_int(number_id)
            used_values = generator._coerce_forbidden_number_values(self.used_number_values_by_id.get(number_id))
            available_count = first_high - low + 1
            if factual_value is not None and low <= int(factual_value) <= first_high:
                available_count -= 1
            if available_count <= 0 or available_count > len(used_values):
                continue
            candidates = {
                value for value in range(low, first_high + 1) if factual_value is None or value != int(factual_value)
            }
            if candidates and candidates.issubset(used_values):
                saturated.add(number_id)
        return saturated


__all__ = ["IntervariantNumtempUniquenessMixin"]
