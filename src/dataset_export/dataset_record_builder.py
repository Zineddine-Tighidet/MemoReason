"""Shared record-building and validation helpers for dataset export."""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import yaml
from src.document_generation.fictional_document_renderer import FictionalDocumentRenderer
from src.document_generation.fictional_entity_sampler import FictionalEntitySampler
from src.document_generation.fictional_entity_sampler_common import detect_ordering_excluded_number_ids
from src.document_generation.number_temporal_generator import NumberTemporalGenerator
from src.document_generation.generated_variant_yaml import build_generated_question_payloads
from src.document_generation.sampling_checks import verify_ordering_preserved
from src.core.century_expressions import has_century_function
from src.core.answer_evaluation import AnswerEvaluator
from src.core.document_schema import EntityCollection
from src.core.entity_taxonomy import (
    REPLACE_MODE_ALL,
    REPLACE_MODE_NON_NUMERICAL,
    ENTITY_TAXONOMY,
    FULL_REPLACE_ENTITY_TYPES,
    LEGACY_ENTITY_ATTRIBUTES,
    PARTIAL_REPLACE_ATTRIBUTES,
    parse_integer_surface_number,
    parse_entity_id,
    parse_word_number,
)
from src.core.annotation_runtime import (
    AnnotationParser,
    RuleEngine,
    find_entity_refs,
    find_rule_sanity_errors,
    load_annotated_document,
    partition_generation_rules,
)
from .dataset_settings import DatasetSettingSpec, factual_setting
from .dataset_paths import (
    PROJECT_ROOT,
    document_variant_path,
    format_document_variant_id,
    resolve_template_identity,
)

_INLINE_ANNOTATION_PATTERN = re.compile(r"\[[^\]]+?;\s*[^\]]+?\]")
_MONTH_NAME_PATTERN = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_SHORT_YEAR_DATE_PATTERN = re.compile(
    rf"\b(?:in|on|at|by|from)\s+\d{{1,2}}\s+{_MONTH_NAME_PATTERN}\s+\d{{2}}\b",
    re.IGNORECASE,
)
_DOUBLE_YEAR_DATE_PATTERN = re.compile(
    r"\b(?:in|on|at|by|from)\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{4}\b",
    re.IGNORECASE,
)
_BIRTH_YEAR_PATTERN = re.compile(r"\bborn\b[^.]{0,80}?(\d{4})", re.IGNORECASE)
_AGE_YEAR_PATTERN = re.compile(r"\bat age\s+(\d{1,3})\s+in\s+(?:[A-Za-z]+\s+)?(\d{4})\b", re.IGNORECASE)
_ALIAS_TAUTOLOGY_PATTERN = re.compile(r"\b(?P<name>[A-Z][A-Za-z0-9'’\" -]{2,}?)\s*,\s*also known as\s+(?P=name)\b")
_PAREN_TAUTOLOGY_PATTERN = re.compile(r"\b(?P<name>[A-Z][A-Za-z0-9'’\" -]{2,}?)\s*\(\s*(?P=name)\s*\)")
_THOUSANDS_COMMA_PATTERN = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+(?:\.\d+)?)(?!\d)")
_BLANKED_NUMTEMP_RENDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bshared the\s{2,}", re.IGNORECASE),
    re.compile(r"\bwon the\s{2,}", re.IGNORECASE),
    re.compile(r"\bin\s+,", re.IGNORECASE),
    re.compile(r"\baged\s+,", re.IGNORECASE),
    re.compile(r"\bdied in\s+,", re.IGNORECASE),
    re.compile(r"\bin\s{2,}different scientific fields", re.IGNORECASE),
    re.compile(r"\blegacy of\s{2,}", re.IGNORECASE),
    re.compile(r"\bfounded .* in\s+,", re.IGNORECASE),
    re.compile(r"\bdeclared\s+ the Year", re.IGNORECASE),
)
_FACTUAL_LEAK_ATTRS = frozenset(
    {
        "full_name",
        "first_name",
        "last_name",
        "name",
        "nationality",
        "demonym",
        "country",
        "state",
        "region",
        "continent",
        "city",
    }
)
_GENERIC_SINGLE_TOKEN_NAME_LITERALS = frozenset({"allied", "united"})
_QUESTION_TYPE_ALIASES = {
    "arthmetic": "arithmetic",
    "arith": "arithmetic",
    "temporal_reasoning": "temporal",
    "temporal-reasoning": "temporal",
    "temporal reasoning": "temporal",
}


def _semantically_equal(lhs: Any, rhs: Any) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    if isinstance(lhs, bool) or isinstance(rhs, bool):
        return lhs is rhs
    normalized_candidates = []
    for value in (lhs, rhs):
        normalized = None
        for parser in (parse_word_number, parse_integer_surface_number):
            try:
                normalized = parser(str(value).strip())
            except Exception:
                normalized = None
            if normalized is not None:
                break
        normalized_candidates.append(normalized)
    if normalized_candidates[0] is not None and normalized_candidates[1] is not None:
        return normalized_candidates[0] == normalized_candidates[1]
    for parser in (parse_word_number, parse_integer_surface_number):
        try:
            lhs_parsed = parser(str(lhs).strip())
            rhs_parsed = parser(str(rhs).strip())
        except Exception:
            lhs_parsed = None
            rhs_parsed = None
        if lhs_parsed is not None and rhs_parsed is not None:
            return lhs_parsed == rhs_parsed
    try:
        return abs(float(lhs) - float(rhs)) <= 1e-9
    except (TypeError, ValueError):
        return str(lhs).strip().casefold() == str(rhs).strip().casefold()


def normalize_question_type(question_type: str | None) -> str:
    """Normalize question-type labels from legacy templates."""
    if not question_type:
        return "unknown"
    cleaned = str(question_type).strip().lower()
    return _QUESTION_TYPE_ALIASES.get(cleaned, cleaned)


def answer_behavior_label(
    answer_type: str | bool | None = None,
    is_answer_invariant: bool | None = None,
) -> str:
    """Return normalized answer behavior label (variant, invariant, refusal)."""
    if isinstance(answer_type, bool) and is_answer_invariant is None:
        is_answer_invariant = answer_type
        answer_type = None

    cleaned = str(answer_type or "").strip().lower()
    if cleaned in {"variant", "invariant", "refusal"}:
        return cleaned
    return "invariant" if bool(is_answer_invariant) else "variant"


def strip_inline_annotations(text: str) -> str:
    """Remove inline annotation metadata while keeping the visible surface text."""
    stripped = re.sub(r"\[([^\]]+?);\s*[^\]]+?\]", r"\1", text or "")
    return _normalize_numeric_surface_commas(stripped)


def _normalize_numeric_surface_commas(text: str) -> str:
    """Normalize thousands-separated numerals to plain digits for factual export."""
    return _THOUSANDS_COMMA_PATTERN.sub(lambda match: match.group(1).replace(",", ""), text or "")


def _relative_to_project(path: Path) -> str:
    candidate = Path(path)
    candidate = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    try:
        return str(candidate.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


def _serialize_entities(entities: EntityCollection) -> dict[str, Any]:
    return {
        "persons": {key: value.model_dump(warnings=False) for key, value in entities.persons.items()},
        "places": {key: value.model_dump(warnings=False) for key, value in entities.places.items()},
        "events": {key: value.model_dump(warnings=False) for key, value in entities.events.items()},
        "organizations": {key: value.model_dump(warnings=False) for key, value in entities.organizations.items()},
        "awards": {key: value.model_dump(warnings=False) for key, value in entities.awards.items()},
        "legals": {key: value.model_dump(warnings=False) for key, value in entities.legals.items()},
        "products": {key: value.model_dump(warnings=False) for key, value in entities.products.items()},
        "temporals": {key: value.model_dump(warnings=False) for key, value in entities.temporals.items()},
        "numbers": {key: value.model_dump(warnings=False) for key, value in entities.numbers.items()},
    }


def _deserialize_entities(payload: dict[str, Any] | None) -> EntityCollection:
    return EntityCollection.model_validate(payload or {})


def _question_entries_from_template(document) -> list[dict[str, Any]]:
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    questions_payload = [
        {
            "question_id": question.question_id,
            "question": strip_inline_annotations(question.question),
            "answer": question.answer,
            "question_type": normalize_question_type(question.question_type),
            "answer_type": answer_behavior_label(question.answer_type),
            "reasoning_chain": list(question.reasoning_chain or []),
        }
        for question in document.questions
    ]
    evaluated_answers = AnswerEvaluator.evaluate_all_answers(questions_payload, factual_entities)
    question_entries = AnswerEvaluator.build_question_entries_with_answers(
        questions_payload,
        evaluated_answers,
        factual_entities,
    )
    question_entries_for_export: list[dict[str, Any]] = []
    for question_entry, source_question in zip(question_entries, document.questions, strict=True):
        question_entries_for_export.append(
            {
                "question_id": question_entry["question_id"],
                "question_type": normalize_question_type(question_entry.get("question_type")),
                "answer_behavior": answer_behavior_label(
                    getattr(source_question, "answer_type", None),
                ),
                "answer_type": answer_behavior_label(
                    getattr(source_question, "answer_type", None),
                ),
                "question_text": question_entry["question"],
                "reasoning_chain": [
                    _normalize_numeric_surface_commas(step) for step in (source_question.reasoning_chain or [])
                ],
                "answer_expression": question_entry.get("answer_expression", ""),
                "answer_entities": question_entry.get("answer_entities"),
                "evaluated_answer": str(question_entry.get("evaluated_answer", "")).strip(),
                "accepted_answer_overrides": [
                    _normalize_numeric_surface_commas(str(value))
                    for value in (getattr(source_question, "accepted_answer_overrides", []) or [])
                ],
            }
        )
    return question_entries_for_export


def build_factual_dataset_record(template_path: Path, *, seed: int) -> dict[str, Any]:
    """Build the factual dataset record for one template."""
    document = load_annotated_document(str(template_path), validate_question_scope=False)
    theme, document_id = resolve_template_identity(template_path)
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    setting_spec = factual_setting()
    return {
        "document_id": document_id,
        "document_theme": theme,
        "document_setting": setting_spec.setting_id,
        "document_setting_family": setting_spec.setting_family,
        "document_variant_id": format_document_variant_id(1),
        "document_variant_index": 1,
        "replacement_proportion": setting_spec.replacement_proportion,
        "generation_seed": seed,
        "source_template_path": _relative_to_project(template_path),
        "document_text": strip_inline_annotations(document.document_to_annotate),
        "num_entities_replaced": 0,
        "replaced_factual_entities": {},
        "questions": _question_entries_from_template(document),
        "entities_used": _serialize_entities(factual_entities),
    }


def _write_yaml(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=10000),
        encoding="utf-8",
    )
    return output_path


def remove_stale_fictional_dataset_exports(
    *,
    template_path: Path,
    setting_spec: DatasetSettingSpec,
) -> None:
    theme, document_id = resolve_template_identity(template_path)
    output_dir = document_variant_path(theme, document_id, setting_spec.setting_id).parent
    if not output_dir.exists():
        return
    for stale_path in sorted(output_dir.glob(f"{document_id}.yaml")):
        stale_path.unlink()
    for stale_path in sorted(output_dir.glob(f"{document_id}_*.yaml")):
        stale_path.unlink()


def _drop_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {key: _drop_nulls(item) for key, item in value.items() if item is not None}
        return {key: item for key, item in cleaned.items() if item not in ({}, [], None)}
    if isinstance(value, list):
        cleaned_list = [_drop_nulls(item) for item in value]
        return [item for item in cleaned_list if item not in ({}, [], None)]
    return value


def _question_entries_from_generated(
    document,
    generated_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    by_question_id = {question.question_id: question for question in document.questions}
    question_entries_for_export: list[dict[str, Any]] = []
    for question_entry in generated_payload.get("questions", []) or []:
        question_id = str(question_entry.get("question_id") or "")
        source_question = by_question_id.get(question_id)
        question_entries_for_export.append(
            {
                "question_id": question_id,
                "question_type": normalize_question_type(
                    question_entry.get("question_type")
                    or (source_question.question_type if source_question is not None else None)
                ),
                "answer_behavior": answer_behavior_label(
                    getattr(source_question, "answer_type", None) if source_question is not None else None,
                ),
                "answer_type": answer_behavior_label(
                    getattr(source_question, "answer_type", None) if source_question is not None else None,
                ),
                "question_text": question_entry.get("question", ""),
                "answer_expression": question_entry.get("answer_expression", ""),
                "answer_entities": question_entry.get("answer_entities"),
                "evaluated_answer": str(question_entry.get("evaluated_answer", "")).strip(),
                "accepted_answer_overrides": list(
                    getattr(source_question, "accepted_answer_overrides", []) if source_question is not None else []
                ),
            }
        )
    return question_entries_for_export


def _contains_inline_annotations(text: str) -> bool:
    return bool(_INLINE_ANNOTATION_PATTERN.search(text or ""))


def _required_refs_for_difference_check(document) -> dict[str, list[tuple[str, list[str]]]]:
    return FictionalEntitySampler.extract_required_entities(document, include_questions=True)


def _explicitly_forced_equal_refs(document, factual_entities: EntityCollection) -> set[str]:
    fixed_refs: set[str] = set()
    equality_neighbors: dict[str, set[str]] = {}
    for raw_rule in (document.rules or []):
        cleaned = str(raw_rule or "").split("#", 1)[0].strip()
        if not cleaned:
            continue
        split = NumberTemporalGenerator._split_rule(cleaned)
        if split is None:
            continue
        left, op, right = split
        if op not in {"=", "=="}:
            continue
        left_refs = find_entity_refs(left)
        right_refs = find_entity_refs(right)
        if len(left_refs) == 1 and len(right_refs) == 1:
            left_ref = left_refs[0]
            right_ref = right_refs[0]
            if (left_ref.startswith("number_") or left_ref.startswith("temporal_")) and (
                right_ref.startswith("number_") or right_ref.startswith("temporal_")
            ):
                equality_neighbors.setdefault(left_ref, set()).add(right_ref)
                equality_neighbors.setdefault(right_ref, set()).add(left_ref)
            continue
        if len(left_refs) == 1 and not right_refs:
            ref = left_refs[0]
            constant = right
        elif len(right_refs) == 1 and not left_refs:
            ref = right_refs[0]
            constant = left
        else:
            continue
        if not (ref.startswith("number_") or ref.startswith("temporal_")):
            continue
        factual_value = RuleEngine._get_entity_value(factual_entities, ref)
        if factual_value is None:
            continue
        cleaned_constant = constant.strip().strip('"').strip("'")
        if _semantically_equal(factual_value, cleaned_constant):
            fixed_refs.add(ref)
    queue = list(fixed_refs)
    while queue:
        current_ref = queue.pop()
        current_value = RuleEngine._get_entity_value(factual_entities, current_ref)
        for neighbor_ref in equality_neighbors.get(current_ref, ()):
            if neighbor_ref in fixed_refs:
                continue
            neighbor_value = RuleEngine._get_entity_value(factual_entities, neighbor_ref)
            if current_value is None or neighbor_value is None:
                continue
            if _semantically_equal(current_value, neighbor_value):
                fixed_refs.add(neighbor_ref)
                queue.append(neighbor_ref)
    required_number_specs = _required_refs_for_difference_check(document).get("number", [])
    required_number_ids = {number_id for number_id, _attrs in required_number_specs}
    if required_number_ids:
        effective_rules, _dropped_rules = partition_generation_rules(document, include_questions=True)
        numeric_rules = [
            rule
            for rule in effective_rules
            if "number_" in str(rule) and "temporal_" not in str(rule) and not has_century_function(str(rule))
        ]
        ordering_excluded_number_ids = detect_ordering_excluded_number_ids(getattr(document, "document_to_annotate", ""))
        generator = NumberTemporalGenerator(
            factual_entities=factual_entities,
            implicit_rules=getattr(document, "implicit_rules", None),
            ordering_excluded_number_ids=ordering_excluded_number_ids,
        )
        active_numeric_rules = generator._number_evaluable_rules(
            numeric_rules,
            required_number_ids,
            factual_entities,
        )
        ordering_rules = generator._ordering_number_rules(list(required_number_ids), factual_entities)
        if ordering_rules:
            active_numeric_rules = [*active_numeric_rules, *ordering_rules]
        constraints = generator._collect_linear_constraints(active_numeric_rules, required_number_ids, factual_entities)
        if constraints is not None:
            base_domains = {number_id: generator._number_base_range(number_id) for number_id in required_number_ids}
            tightened_domains = generator._tighten_number_domains(constraints, base_domains)
            if tightened_domains is not None:
                for number_id, (low, high) in tightened_domains.items():
                    if low != high:
                        continue
                    factual_number = factual_entities.numbers.get(number_id)
                    factual_value = generator._number_entity_int_value(factual_number) if factual_number is not None else None
                    if factual_value is None or int(low) != int(factual_value):
                        continue
                    fixed_refs.update(generator._forced_equal_refs_for_number(number_id))
    return fixed_refs


def _verify_full_fictional_manual_differences(
    document,
    *,
    factual_entities: EntityCollection,
    fictional_entities: EntityCollection,
    replace_mode: str,
) -> None:
    exempt_person_attrs = FictionalEntitySampler._PERSON_DIFF_EXEMPT_ATTRS
    explicitly_fixed_refs = _explicitly_forced_equal_refs(document, factual_entities)
    unchanged_refs: list[str] = []
    fully_replaced_types = FULL_REPLACE_ENTITY_TYPES.get(replace_mode, frozenset())
    partially_replaced_attrs = PARTIAL_REPLACE_ATTRIBUTES.get(replace_mode, {})
    for entity_type, specs in _required_refs_for_difference_check(document).items():
        for entity_id, attrs in specs:
            for attr in attrs:
                if entity_type == "person" and attr in exempt_person_attrs:
                    continue
                if entity_type in fully_replaced_types:
                    should_change = True
                else:
                    should_change = attr in partially_replaced_attrs.get(entity_type, frozenset())
                if not should_change:
                    continue
                ref = f"{entity_id}.{attr}"
                if ref in explicitly_fixed_refs:
                    continue
                factual_value = RuleEngine._get_entity_value(factual_entities, ref)
                fictional_value = RuleEngine._get_entity_value(fictional_entities, ref)
                if factual_value is None or fictional_value is None:
                    continue
                if _semantically_equal(factual_value, fictional_value):
                    unchanged_refs.append(ref)
    if unchanged_refs:
        rendered = ", ".join(sorted(set(unchanged_refs))[:12])
        raise ValueError(f"Full-fictional export kept factual values for required refs: {rendered}.")


def _verify_declared_replacements_differ(
    *,
    factual_entities: EntityCollection,
    fictional_entities: EntityCollection,
    replaced_factual_entities: dict[str, Any] | None,
    exempt_refs: set[str] | None = None,
) -> None:
    unchanged_entities: list[str] = []
    exempt_refs = set(exempt_refs or set())
    for entity_entries in (replaced_factual_entities or {}).values():
        if not isinstance(entity_entries, dict):
            continue
        for entity_id, attr_map in entity_entries.items():
            if not isinstance(attr_map, dict):
                continue
            entity_type, _ = parse_entity_id(str(entity_id))
            valid_attrs = ENTITY_TAXONOMY.get(entity_type or "", frozenset()) | LEGACY_ENTITY_ATTRIBUTES.get(
                entity_type or "",
                frozenset(),
            )
            any_changed = False
            for attr in attr_map:
                if attr not in valid_attrs:
                    continue
                ref = f"{entity_id}.{attr}"
                if ref in exempt_refs:
                    any_changed = True
                    break
                factual_value = RuleEngine._get_entity_value(factual_entities, ref)
                fictional_value = RuleEngine._get_entity_value(fictional_entities, ref)
                if factual_value is None or fictional_value is None:
                    continue
                if not _semantically_equal(factual_value, fictional_value):
                    any_changed = True
                    break
            if not any_changed:
                unchanged_entities.append(str(entity_id))
    if unchanged_entities:
        rendered = ", ".join(sorted(set(unchanged_entities))[:12])
        raise ValueError(f"Declared replaced entities kept their factual values: {rendered}.")


def _factual_literal_from_attr_value(attr: str, value: Any) -> str | None:
    if attr not in _FACTUAL_LEAK_ATTRS or value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) < 4 or not re.search(r"[A-Za-z]", text):
        return None
    if attr == "name" and " " not in text and len(text) < 6:
        return None
    if attr == "name" and " " not in text and text == text.lower():
        return None
    if attr == "name" and " " not in text and text.casefold() in _GENERIC_SINGLE_TOKEN_NAME_LITERALS:
        return None
    return text


def _literal_exemption_keys(text: str) -> set[str]:
    keys = {text.casefold()}
    tokens = [token.strip("'’-") for token in re.findall(r"[A-Za-z][A-Za-z'’-]*", text)]
    tokens = [token for token in tokens if token]
    for token in tokens:
        if len(token) >= 4:
            keys.add(token.casefold())
    for start in range(len(tokens)):
        for end in range(start + 2, min(len(tokens), start + 4) + 1):
            phrase = " ".join(tokens[start:end])
            if len(phrase) >= 4:
                keys.add(phrase.casefold())
    return keys


def _iter_replaced_factual_literals(
    replaced_factual_entities: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    literals: list[tuple[str, str]] = []
    for entity_entries in (replaced_factual_entities or {}).values():
        if not isinstance(entity_entries, dict):
            continue
        for attr_map in entity_entries.values():
            if not isinstance(attr_map, dict):
                continue
            for attr, value in attr_map.items():
                text = _factual_literal_from_attr_value(str(attr), value)
                if text is not None:
                    literals.append((str(attr), text))
    return literals


def _kept_factual_literal_exemptions(
    original_document: Any,
    replaced_factual_entities: dict[str, Any] | None,
) -> set[str]:
    factual_entities = AnnotationParser.extract_factual_entities(original_document, include_questions=True)
    factual_payload = _serialize_entities(factual_entities)
    replaced_attr_refs: set[tuple[str, str, str]] = set()
    for bucket, entity_entries in (replaced_factual_entities or {}).items():
        if not isinstance(entity_entries, dict):
            continue
        for entity_id, attr_map in entity_entries.items():
            if not isinstance(attr_map, dict):
                continue
            for attr in attr_map:
                replaced_attr_refs.add((str(bucket), str(entity_id), str(attr)))

    exemptions: set[str] = set()
    for bucket, entity_entries in factual_payload.items():
        if not isinstance(entity_entries, dict):
            continue
        for entity_id, attr_map in entity_entries.items():
            if not isinstance(attr_map, dict):
                continue
            for attr, value in attr_map.items():
                if (str(bucket), str(entity_id), str(attr)) in replaced_attr_refs:
                    continue
                text = _factual_literal_from_attr_value(str(attr), value)
                if text is not None:
                    exemptions.update(_literal_exemption_keys(text))
    return exemptions


def _semantic_payload_issues(
    generated_payload: dict[str, Any],
    *,
    original_document: Any | None = None,
) -> list[str]:
    document_text = str(generated_payload.get("generated_document") or "")
    entities = _deserialize_entities(generated_payload.get("entities_used") or {})
    replaced_entities = generated_payload.get("replaced_factual_entities") or {}
    question_texts = [
        str(question_entry.get("question") or "") for question_entry in generated_payload.get("questions") or []
    ]
    combined_text = "\n".join([document_text, *question_texts])
    issues: list[str] = []

    if match := _DOUBLE_YEAR_DATE_PATTERN.search(document_text):
        issues.append(f"malformed date rendering: {match.group(0)!r}")
    if match := _SHORT_YEAR_DATE_PATTERN.search(document_text):
        issues.append(f"short-year date rendering: {match.group(0)!r}")
    if match := _ALIAS_TAUTOLOGY_PATTERN.search(document_text):
        issues.append(f"tautological aliasing: {match.group(0)!r}")
    if match := _PAREN_TAUTOLOGY_PATTERN.search(document_text):
        issues.append(f"parenthetical tautology: {match.group(0)!r}")

    birth_match = _BIRTH_YEAR_PATTERN.search(document_text)
    if birth_match:
        birth_year = int(birth_match.group(1))
        for age_text, year_text in _AGE_YEAR_PATTERN.findall(document_text):
            age = int(age_text)
            event_year = int(year_text)
            if abs((event_year - birth_year) - age) > 1:
                issues.append(f"birth-year chronology contradiction: birth={birth_year}, age={age}, year={event_year}")
                break

    kept_literal_exemptions = (
        _kept_factual_literal_exemptions(original_document, replaced_entities)
        if original_document is not None
        else set()
    )
    leaked_literals: list[str] = []
    for _attr, literal in _iter_replaced_factual_literals(generated_payload.get("replaced_factual_entities")):
        if literal.casefold() in kept_literal_exemptions:
            continue
        pattern = rf"(?<![A-Za-z0-9]){re.escape(literal)}(?![A-Za-z0-9])"
        if re.search(pattern, combined_text, flags=re.IGNORECASE):
            leaked_literals.append(literal)
    if leaked_literals:
        issues.append(
            "replaced factual literals still appear in output: " + ", ".join(sorted(set(leaked_literals))[:10])
        )

    if (replaced_entities.get("numbers") or replaced_entities.get("temporals")) and any(
        pattern.search(document_text) for pattern in _BLANKED_NUMTEMP_RENDER_PATTERNS
    ):
        issues.append("blank numeric/temporal render detected in generated document text")

    primary_org = entities.organizations.get("entreprise_org_1")
    primary_org_name = str(getattr(primary_org, "name", "") or "").strip()
    if primary_org_name:
        normalized_org_name = re.sub(r"[^a-z0-9]+", " ", primary_org_name.lower()).strip()
        org_tokens = {token for token in normalized_org_name.split() if token}
        for place in entities.places.values():
            for attr in ("country", "city", "region", "state", "natural_site"):
                value = str(getattr(place, attr, "") or "").strip()
                normalized_place = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
                if (
                    value
                    and normalized_place
                    and (
                        normalized_org_name == normalized_place
                        or normalized_place in org_tokens
                    )
                    and re.search(rf"\b{re.escape(value)}\b", document_text)
                ):
                    if (
                        primary_org_name.casefold() in kept_literal_exemptions
                        and value.casefold() in kept_literal_exemptions
                    ):
                        continue
                    issues.append(
                        f"place/self-reference collision: organization {primary_org_name!r} overlaps with place {value!r}"
                    )
                    break
            if issues and issues[-1].startswith("place/self-reference collision:"):
                break

    return issues


def _verify_generated_payload(
    *,
    original_document,
    render_source_document,
    generated_payload: dict[str, Any],
    setting_spec: DatasetSettingSpec,
    source_mode: str,
) -> None:
    entities = _deserialize_entities(generated_payload.get("entities_used") or {})
    rendered_document = FictionalDocumentRenderer.render_document(render_source_document, entities)
    expected_questions = build_generated_question_payloads(rendered_document, entities)
    actual_document_text = str(generated_payload.get("generated_document") or "")
    if not actual_document_text:
        raise ValueError("Generated payload has empty document text.")
    if actual_document_text != rendered_document.generated_document:
        raise ValueError("Generated payload document_text drifted from a fresh render of the source template.")
    if _contains_inline_annotations(actual_document_text):
        raise ValueError("Generated payload still contains inline annotation syntax in document_text.")

    question_entries = generated_payload.get("questions") or []
    if len(question_entries) != len(expected_questions):
        raise ValueError(
            f"Generated payload has {len(question_entries)} questions but {len(expected_questions)} were expected."
        )

    actual_by_id = {str(entry.get("question_id") or ""): entry for entry in question_entries}
    expected_by_id = {str(entry["question_id"]): entry for entry in expected_questions}
    if set(actual_by_id) != set(expected_by_id):
        raise ValueError("Generated payload question IDs do not match the template question IDs.")

    for question_id, expected_entry in expected_by_id.items():
        actual_entry = actual_by_id[question_id]
        actual_question_text = str(actual_entry.get("question") or "")
        if not actual_question_text:
            raise ValueError(f"{question_id}: generated question text is empty.")
        if actual_question_text != str(expected_entry["question"]):
            raise ValueError(f"{question_id}: question text drifted from rendered output.")
        if _contains_inline_annotations(actual_question_text):
            raise ValueError(f"{question_id}: generated question text still contains annotation syntax.")

        actual_expr = str(actual_entry.get("answer_expression") or "").strip()
        expected_expr = str(expected_entry["answer_expression"] or "").strip()
        if actual_expr != expected_expr:
            raise ValueError(f"{question_id}: answer_expression drifted from the rendered question payload.")

        actual_answer_entities = actual_entry.get("answer_entities")
        expected_answer_entities = expected_entry["answer_entities"]
        if actual_answer_entities != expected_answer_entities:
            raise ValueError(f"{question_id}: answer_entities drifted from the evaluated entity refs.")

        actual_evaluated = str(actual_entry.get("evaluated_answer") or "").strip()
        expected_evaluated = str(expected_entry["evaluated_answer"] or "").strip()
        if actual_evaluated != expected_evaluated:
            raise ValueError(f"{question_id}: evaluated_answer drifted from recomputation.")

        referenced_answer_refs = find_entity_refs(expected_expr)
        if referenced_answer_refs and not actual_answer_entities:
            raise ValueError(f"{question_id}: answer_expression references entities but answer_entities is empty.")
        if referenced_answer_refs:
            missing_refs = [ref for ref in referenced_answer_refs if ref not in (actual_answer_entities or {})]
            if missing_refs:
                raise ValueError(f"{question_id}: answer_entities is missing refs: {missing_refs}.")
        if expected_expr and not actual_evaluated:
            raise ValueError(f"{question_id}: evaluated_answer resolved to an empty string.")

    all_rules = [str(rule) for rule in (original_document.rules or []) if str(rule).split("#", 1)[0].strip()]
    effective_rules, dropped_rules = partition_generation_rules(original_document, include_questions=True)
    sanity_errors = find_rule_sanity_errors(all_rules)
    if sanity_errors:
        raise ValueError("\n".join(sanity_errors))
    rule_results = RuleEngine.validate_all_rules(effective_rules, entities)
    failed_rules = [rule for rule, is_valid in rule_results if not is_valid]
    if failed_rules:
        rendered_failed = "\n".join(failed_rules[:10])
        raise ValueError(f"Generated payload violates document rules:\n{rendered_failed}")

    factual_entities = AnnotationParser.extract_factual_entities(original_document, include_questions=True)
    explicitly_fixed_refs = _explicitly_forced_equal_refs(original_document, factual_entities)
    _verify_declared_replacements_differ(
        factual_entities=factual_entities,
        fictional_entities=entities,
        replaced_factual_entities=generated_payload.get("replaced_factual_entities"),
        exempt_refs=explicitly_fixed_refs,
    )

    if setting_spec.replacement_proportion == 1.0:
        _verify_full_fictional_manual_differences(
            original_document,
            factual_entities=factual_entities,
            fictional_entities=entities,
            replace_mode=setting_spec.replace_mode,
        )

    semantic_issues = _semantic_payload_issues(generated_payload, original_document=original_document)
    if semantic_issues:
        rendered = "\n".join(semantic_issues[:10])
        raise ValueError(f"Generated payload failed semantic linting:\n{rendered}")

    if source_mode not in {
        "pool",
        "derived_from_full_fictional",
    }:
        raise ValueError(f"Unknown fictional generation source mode: {source_mode!r}")
