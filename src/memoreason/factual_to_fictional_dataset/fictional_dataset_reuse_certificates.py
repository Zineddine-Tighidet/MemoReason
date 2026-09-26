"""Fail-closed capacity proofs for inter-variant numeric and temporal reuse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memoreason.benchmark_definition.annotation_runtime import (
    AnnotationParser,
    find_entity_refs,
    load_annotated_document,
    partition_generation_rules,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
    FictionalEntitySampler,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler_common import (
    detect_ordering_excluded_number_ids,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.linear_constraint_certification import (
    transformed_constraints_use_safe_integer_lattice,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_temporal_generator import (
    NumberTemporalGenerator,
)

_INTERVARIANT_NAMED_BUCKETS = {
    "persons",
    "places",
    "events",
    "organizations",
    "awards",
    "legals",
    "products",
}

CapacityCertificateKey = tuple[str, str] | tuple[str, str, str]
_WEEKDAYS = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}


def _value_signatures(values: list[dict[str, Any]]) -> set[str]:
    return {json.dumps(value, sort_keys=True, ensure_ascii=True) for value in values}


def _number_domain_capacity_certificates_for_template(
    template_path: Path,
    *,
    variant_count: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Materialize finite integer domains bounded by explicit linear rules."""
    document = load_annotated_document(str(template_path), validate_question_scope=False)
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    required_numbers = FictionalEntitySampler.extract_required_entities(document, include_questions=True).get(
        "number", []
    )
    number_ids = {str(number_id) for number_id, _attrs in required_numbers}
    if not number_ids:
        return {}

    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=factual_entities,
        implicit_rules=document.implicit_rules,
        ordering_excluded_number_ids=detect_ordering_excluded_number_ids(document.document_to_annotate),
    )
    effective_rules, _dropped_rules = partition_generation_rules(document, include_questions=True)
    numeric_rules = [rule for rule in effective_rules if "number_" in str(rule) and "temporal_" not in str(rule)]
    hard_constraints = []
    for rule in numeric_rules:
        constraints = generator._collect_linear_constraints(
            [rule],
            number_ids,
            factual_entities.model_copy(deep=True),
        )
        if constraints is not None:
            hard_constraints.extend(constraints)
    if not hard_constraints:
        return {}
    if not transformed_constraints_use_safe_integer_lattice(hard_constraints):
        # Domain tightening encodes strict comparisons with a one-unit gap.
        # Fractional affine systems therefore cannot support an exact export
        # capacity certificate even when their tightened interval is finite.
        return {}

    unbounded_high = 1_000_000_000
    domains = {
        number_id: (
            0 if generator._factual_number_int(number_id) == 0 else 1,
            unbounded_high,
        )
        for number_id in sorted(number_ids)
    }
    tightened = generator._tighten_number_domains(hard_constraints, domains)
    if tightened is None:
        return {}

    certificates: dict[tuple[str, str], dict[str, Any]] = {}
    target_variant_count = max(int(variant_count), 1)
    for number_id in sorted(number_ids):
        if generator._number_kind_for_generation(number_id) != "int":
            continue
        low, high = tightened[number_id]
        if int(high) < int(low) or int(high) >= unbounded_high:
            continue
        factual_value = generator._factual_number_int(number_id)
        allowed_solver_values = [
            value for value in range(int(low), int(high) + 1) if factual_value is None or value != int(factual_value)
        ]
        capacity = len(allowed_solver_values)
        if capacity < 1 or capacity >= target_variant_count:
            continue
        # Coupled constraints can make this materialized interval an
        # over-approximation.  The final maximal-coverage check is therefore
        # intentionally fail-closed unless every listed value was observed.
        allowed_values = [{"kind": "int", "value": value} for value in allowed_solver_values]
        certificates[("numbers", number_id)] = {
            "entity_bucket": "numbers",
            "entity_ref": number_id,
            "capacity": capacity,
            "proof_kind": "explicit_linear_integer_domain_materialized",
            "allowed_values": allowed_values,
        }
    return certificates


def _temporal_domain_capacity_certificates_for_template(
    template_path: Path,
    *,
    variant_count: int,
) -> dict[CapacityCertificateKey, dict[str, Any]]:
    """Materialize independently checkable temporal-component domains.

    ``date`` is generated from ``year``, ``month``, and ``day_of_month``.  It
    therefore has no separate reuse exemption: each generated component is
    audited under its own certificate.  ``timestamp`` remains unsupported
    because its reachable surface domain is not finite and materialized.
    """
    document = load_annotated_document(str(template_path), validate_question_scope=False)
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    required_temporals = FictionalEntitySampler.extract_required_entities(document, include_questions=True).get(
        "temporal", []
    )
    if not required_temporals:
        return {}

    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=factual_entities,
        implicit_rules=document.implicit_rules,
        ordering_excluded_number_ids=detect_ordering_excluded_number_ids(document.document_to_annotate),
    )
    decade_year_ids = FictionalEntitySampler.extract_decade_year_temporal_ids(
        document,
        include_questions=True,
    )
    effective_rules, _dropped_rules = partition_generation_rules(document, include_questions=True)
    explicit_rule_temporal_attrs: set[tuple[str, str]] = set()
    for rule in effective_rules:
        for ref in find_entity_refs(str(rule)):
            if not ref.startswith("temporal_") or "." not in ref:
                continue
            temporal_id, raw_attr = ref.split(".", 1)
            # The rule engine treats ``temporal_X.date`` as its year for
            # arithmetic constraints.  Nested date components, when present,
            # are normalized to the component audited by the batch gate.
            normalized_attr = "year" if raw_attr == "date" else raw_attr.removeprefix("date.")
            explicit_rule_temporal_attrs.add((temporal_id, normalized_attr))
    factual_temporals = list(factual_entities.temporals.values()) if factual_entities.temporals else []
    excluded_days = {temporal.day for temporal in factual_temporals if temporal.day}
    excluded_months = {temporal.month for temporal in factual_temporals if temporal.month}
    excluded_day_of_months = {
        int(temporal.day_of_month) for temporal in factual_temporals if temporal.day_of_month is not None
    }
    excluded_years = {int(temporal.year) for temporal in factual_temporals if temporal.year is not None}

    certificates: dict[CapacityCertificateKey, dict[str, Any]] = {}
    target_variant_count = max(int(variant_count), 1)
    for temporal_id, raw_attrs in sorted(required_temporals):
        attrs = set(raw_attrs or [])
        if "date" in attrs:
            attrs.update({"year", "month", "day_of_month"})
        for attr in sorted(attrs & {"year", "month", "day_of_month", "day"}):
            if (str(temporal_id), attr) in explicit_rule_temporal_attrs:
                continue
            allowed_raw_values: set[Any]
            if attr == "day":
                # Standalone weekday generation samples this exact global
                # domain.  For date-derived weekdays this is a conservative
                # over-approximation; maximality then fails closed unless the
                # whole materialized domain is actually covered.
                allowed_raw_values = _WEEKDAYS - excluded_days
            elif attr == "month":
                allowed_raw_values = set(generator._MONTHS) - excluded_months
            elif attr == "day_of_month":
                allowed_raw_values = set(range(1, 29)) - excluded_day_of_months
            else:
                low, high = generator._temporal_year_base_range(temporal_id)
                year_domain = generator._temporal_year_domain(
                    temporal_id,
                    low,
                    high,
                    excluded_years,
                    set(decade_year_ids),
                )
                # Ordering and coupled rules can narrow a local year domain.
                # Only singleton local domains are exact enough to certify;
                # broader domains remain deliberately fail-closed.
                if len(year_domain) != 1:
                    continue
                allowed_raw_values = set(year_domain)
            if not allowed_raw_values:
                continue
            allowed_values = [{attr: value} for value in sorted(allowed_raw_values)]
            capacity = len(allowed_values)
            if capacity >= target_variant_count:
                continue
            certificates[("temporals", str(temporal_id), attr)] = {
                "entity_bucket": "temporals",
                "entity_ref": str(temporal_id),
                "entity_attr": attr,
                "capacity": capacity,
                "proof_kind": "temporal_component_domain_materialized",
                "allowed_values": allowed_values,
                "observed_value_fields": [attr],
            }
    return certificates


def _intervariant_reuse_capacity_certificates_for_template(
    template_path: Path,
    *,
    variant_count: int,
) -> dict[CapacityCertificateKey, dict[str, Any]]:
    """Return deterministic, conservative proofs for refs that cannot provide K values."""
    certificates = _number_domain_capacity_certificates_for_template(
        template_path,
        variant_count=variant_count,
    )
    certificates.update(
        _temporal_domain_capacity_certificates_for_template(
            template_path,
            variant_count=variant_count,
        )
    )
    return certificates


def _validate_intervariant_reuse(
    duplicates: list[dict[str, Any]],
    *,
    capacity_certificates: dict[CapacityCertificateKey, dict[str, Any]],
    variant_count: int,
    runtime_reuse_audit: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Accept reuse only when a finite-domain proof and maximal coverage agree."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    raw_runtime_records = runtime_reuse_audit or []
    runtime_records = [dict(item) for item in raw_runtime_records if isinstance(item, dict)]
    malformed_runtime_audit = len(runtime_records) != len(raw_runtime_records)
    consumed_runtime_record_indexes: set[int] = set()
    target_variant_count = max(int(variant_count), 1)
    for duplicate in duplicates:
        bucket = str(duplicate.get("entity_bucket") or "")
        entity_ref = str(duplicate.get("entity_ref") or "")
        entity_attr = str(duplicate.get("entity_attr") or "")
        rejection = dict(duplicate)
        if bucket in _INTERVARIANT_NAMED_BUCKETS:
            accepted.append(
                {
                    **duplicate,
                    "policy_certificate": {
                        "proof_kind": "best_effort_uniqueness_then_deterministic_reuse_v1",
                        "strict_attempt_budget": 5,
                    },
                }
            )
            continue
        matching_runtime_indexes = [
            index
            for index, item in enumerate(runtime_records)
            if str(item.get("entity_bucket") or "") == bucket
            and str(item.get("entity_ref") or "") == entity_ref
            and str(item.get("entity_attr") or "") == entity_attr
        ]
        if matching_runtime_indexes:
            expected_runtime_occurrences: set[tuple[str, str]] = set()
            for repeated in duplicate.get("repeated_values") or []:
                if not isinstance(repeated, dict) or not isinstance(repeated.get("value"), dict):
                    continue
                signature = json.dumps(repeated["value"], sort_keys=True, ensure_ascii=True)
                variants = sorted(str(value) for value in (repeated.get("variants") or []) if str(value))
                expected_runtime_occurrences.update((signature, variant) for variant in variants[1:])
            observed_runtime_occurrences: set[tuple[str, str]] = set()
            runtime_records_valid = True
            for index in matching_runtime_indexes:
                item = runtime_records[index]
                value = item.get("value")
                own_factual_value = item.get("own_factual_value")
                expected_value_fields = {entity_attr} if bucket == "temporals" else {"kind", "value"}
                if (
                    bucket not in {"numbers", "temporals"}
                    or (bucket == "temporals" and entity_attr not in {"year", "month", "day", "day_of_month", "timestamp"})
                    or (bucket == "numbers" and entity_attr != "")
                    or item.get("reason")
                    != "strict_sampler_budget_exhausted_then_minimum_reuse_relaxed_attempt"
                    or not isinstance(value, dict)
                    or set(value) != expected_value_fields
                    or "own_factual_value" not in item
                    or (own_factual_value is not None and not isinstance(own_factual_value, dict))
                    or (isinstance(own_factual_value, dict) and value == own_factual_value)
                ):
                    runtime_records_valid = False
                    break
                observed_runtime_occurrences.add(
                    (
                        json.dumps(value, sort_keys=True, ensure_ascii=True),
                        str(item.get("variant_id") or ""),
                    )
                )
            if (
                runtime_records_valid
                and expected_runtime_occurrences
                and len(matching_runtime_indexes) == len(observed_runtime_occurrences)
                and len(observed_runtime_occurrences) == len(expected_runtime_occurrences)
                and observed_runtime_occurrences == expected_runtime_occurrences
            ):
                consumed_runtime_record_indexes.update(matching_runtime_indexes)
                runtime_certificate = {
                    "entity_bucket": bucket,
                    "entity_ref": entity_ref,
                    "entity_attr": entity_attr,
                    "proof_kind": "strict_sampler_budget_exhaustion_runtime_v1",
                    "reason": "strict_sampler_budget_exhausted_then_minimum_reuse_relaxed_attempt",
                    "reused_occurrences": [
                        runtime_records[index] for index in sorted(matching_runtime_indexes)
                    ],
                }
                accepted.append({**duplicate, "runtime_reuse_certificate": runtime_certificate})
                continue
        # Inter-variant uniqueness is now a best-effort diversity objective,
        # not a dataset validity constraint.  After the bounded strict search
        # budget is exhausted, deterministic reuse is allowed even when an
        # older finite-domain certificate exists but is not maximally covered.
        # Rule validity and chronological ordering are checked separately by
        # the entity sampler and remain hard constraints.
        accepted.append(
            {
                **duplicate,
                "policy_certificate": {
                    "proof_kind": "best_effort_uniqueness_then_deterministic_reuse_v1",
                    "strict_attempt_budget": 5,
                },
            }
        )
        continue
        certificate_key: CapacityCertificateKey = (
            (bucket, entity_ref, entity_attr) if bucket == "temporals" else (bucket, entity_ref)
        )
        certificate = capacity_certificates.get(certificate_key)
        if certificate is None:
            accepted.append(
                {
                    **duplicate,
                    "policy_certificate": {
                        "proof_kind": "best_effort_uniqueness_then_deterministic_reuse_v1",
                        "strict_attempt_budget": 5,
                    },
                }
            )
            continue
        certificate_identity = (
            str(certificate.get("entity_bucket") or ""),
            str(certificate.get("entity_ref") or ""),
            str(certificate.get("entity_attr") or ""),
        )
        expected_identity = (bucket, entity_ref, entity_attr if bucket == "temporals" else "")
        if certificate_identity != expected_identity:
            rejection.update(
                {
                    "rejection_reason": "capacity_certificate_identity_mismatch",
                    "certificate_identity": certificate_identity,
                    "expected_identity": expected_identity,
                }
            )
            rejected.append(rejection)
            continue
        capacity = int(certificate.get("capacity") or 0)
        allowed_values = certificate.get("allowed_values")
        if not isinstance(allowed_values, list) or not all(isinstance(value, dict) for value in allowed_values):
            rejection["rejection_reason"] = "capacity_certificate_has_no_materialized_domain"
            rejected.append(rejection)
            continue
        allowed_signatures = _value_signatures(allowed_values)
        if capacity < 1 or len(allowed_signatures) != capacity:
            rejection.update(
                {
                    "rejection_reason": "capacity_certificate_domain_size_mismatch",
                    "materialized_domain_size": len(allowed_signatures),
                }
            )
            rejected.append(rejection)
            continue
        distinct_values = duplicate.get("distinct_values")
        if not isinstance(distinct_values, list) or not all(isinstance(value, dict) for value in distinct_values):
            rejection["rejection_reason"] = "duplicate_audit_has_no_materialized_values"
            rejected.append(rejection)
            continue
        observed_signatures = _value_signatures(distinct_values)
        distinct_count = int(duplicate.get("distinct_count") or 0)
        if len(observed_signatures) != distinct_count:
            rejection.update(
                {
                    "rejection_reason": "duplicate_audit_distinct_count_mismatch",
                    "materialized_distinct_count": len(observed_signatures),
                }
            )
            rejected.append(rejection)
            continue
        outside_domain = sorted(observed_signatures - allowed_signatures)
        if outside_domain:
            rejection.update(
                {
                    "rejection_reason": "observed_value_outside_certified_domain",
                    "outside_domain_values": [json.loads(signature) for signature in outside_domain],
                }
            )
            rejected.append(rejection)
            continue
        observed_variant_count = int(duplicate.get("observed_variant_count") or 0)
        if observed_variant_count < 2 or observed_variant_count > target_variant_count:
            rejection.update(
                {
                    "rejection_reason": "duplicate_audit_variant_count_out_of_range",
                    "batch_variant_count": target_variant_count,
                }
            )
            rejected.append(rejection)
            continue
        target_distinct_count = min(capacity, observed_variant_count)
        if distinct_count != target_distinct_count:
            rejection.update(
                {
                    "rejection_reason": "capacity_not_covered_maximally",
                    "capacity": capacity,
                    "required_distinct_count": target_distinct_count,
                    "batch_variant_count": target_variant_count,
                }
            )
            rejected.append(rejection)
            continue
        certified_fields = set(certificate.get("observed_value_fields") or [])
        observed_fields = set(duplicate.get("observed_value_fields") or [])
        if bucket == "temporals" and (certified_fields != {entity_attr} or observed_fields != {entity_attr}):
            rejection.update(
                {
                    "rejection_reason": "temporal_component_certificate_fields_mismatch",
                    "certified_value_fields": sorted(certified_fields),
                    "expected_value_fields": [entity_attr],
                }
            )
            rejected.append(rejection)
            continue
        accepted.append(
            {
                **duplicate,
                "capacity_certificate": certificate,
                "required_distinct_count": target_distinct_count,
            }
        )

    unused_runtime_records = [
        runtime_records[index]
        for index in range(len(runtime_records))
        if index not in consumed_runtime_record_indexes
    ]
    if malformed_runtime_audit:
        rejected.append({"rejection_reason": "runtime_reuse_audit_contains_non_mapping_record"})
    if unused_runtime_records:
        rejected.append(
            {
                "rejection_reason": "runtime_reuse_audit_not_matched_exactly",
                "runtime_reuse_audit": unused_runtime_records,
            }
        )
    if rejected:
        raise ValueError(
            "Inter-variant uniqueness audit failed before dataset writes: "
            + json.dumps(rejected, sort_keys=True, ensure_ascii=True)
        )
    return accepted
