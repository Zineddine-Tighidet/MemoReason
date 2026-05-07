"""Document-level text replacement helpers."""

import re
from decimal import Decimal
from typing import Any, Optional

from src.core.document_schema import AnnotatedDocument, EntityCollection, FictionalDocument
from src.core.annotation_runtime import AnnotationParser, RuleEngine, get_appropriate_relationship
from src.core.entity_taxonomy import (
    infer_int_surface_format,
    infer_str_surface_format,
    render_integer_surface_number,
    render_word_surface_number,
)


class FictionalDocumentRenderer:
    """Render a fictional document variant from an annotated factual template."""

    _GENDER_NOUN_CHILD_SINGULAR = frozenset({"boy", "girl"})
    _GENDER_NOUN_CHILD_PLURAL = frozenset({"boys", "girls"})
    _GENDER_NOUN_ADULT_SINGULAR = frozenset({"man", "woman", "guy", "lady"})
    _GENDER_NOUN_ADULT_PLURAL = frozenset({"men", "women", "guys", "ladies"})
    _GENDER_ADJECTIVES_SINGULAR = frozenset({"male", "female"})
    _GENDER_ADJECTIVES_PLURAL = frozenset({"males", "females"})
    _ARTICLELESS_NUMBER_WORDS = frozenset(
        {
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        }
    )
    _GLOBAL_LITERAL_REWRITE_ATTRS = frozenset(
        {
            "first_name",
            "last_name",
            "full_name",
            "name",
            "nationality",
            "demonym",
            "country",
            "state",
            "region",
            "continent",
            "city",
            "natural_site",
        }
    )
    _DUPLICATE_PAREN_PATTERN = re.compile(r"\b(?P<name>[A-Z][A-Za-z0-9'’\" -]{2,}?)\s*\(\s*(?P=name)\s*\)")
    _BIRTH_YEAR_PATTERN = re.compile(r"\bborn\b[^.]{0,80}?(\d{4})", re.IGNORECASE)
    _AGE_IN_YEAR_PATTERN = re.compile(r"\bat age\s+(\d{1,3})\s+in\s+((?:[A-Za-z]+\s+)?)(\d{4})\b", re.IGNORECASE)
    _DUPLICATE_SUFFIXES: tuple[str, ...] = (
        "region",
        "city",
        "state",
        "country",
        "province",
        "prefecture",
        "county",
        "ocean",
        "sea",
        "gulf",
        "bay",
        "river",
        "peninsula",
        "island",
        "islands",
        "mountain",
        "mountains",
        "lake",
    )
    _DEFINITE_ARTICLE_KEEP_TOKENS = frozenset(
        {
            "academy",
            "agency",
            "authority",
            "bank",
            "bay",
            "bureau",
            "channel",
            "city",
            "coast",
            "command",
            "commission",
            "commonwealth",
            "conference",
            "council",
            "county",
            "department",
            "desert",
            "dominions",
            "emirates",
            "federation",
            "framework",
            "gulf",
            "hospital",
            "institute",
            "island",
            "islands",
            "isle",
            "isles",
            "kingdom",
            "lake",
            "ministry",
            "mountain",
            "mountains",
            "ocean",
            "office",
            "peninsula",
            "plant",
            "plan",
            "power",
            "process",
            "program",
            "project",
            "protocol",
            "prefecture",
            "province",
            "region",
            "republic",
            "river",
            "sea",
            "state",
            "states",
            "station",
            "strait",
            "union",
            "university",
        }
    )
    _DEFINITE_ARTICLE_SINGLETONS = frozenset({"bahamas", "gambia", "netherlands", "philippines"})

    @staticmethod
    def render_document(annotated_doc: AnnotatedDocument, fictional_entities: EntityCollection) -> FictionalDocument:
        """Replace annotated factual mentions in the document and questions."""
        preserve_original_gender_ids = FictionalDocumentRenderer._find_ambiguous_gender_entity_ids(annotated_doc)
        age_anchor_map = FictionalDocumentRenderer._build_age_anchor_map(annotated_doc)
        generated_text = FictionalDocumentRenderer._replace_annotations(
            annotated_doc.document_to_annotate,
            fictional_entities,
            preserve_original_gender_ids=preserve_original_gender_ids,
            age_anchor_map=age_anchor_map,
        )
        generated_questions = []
        for q in annotated_doc.questions:
            gen_question = FictionalDocumentRenderer._replace_annotations(
                q.question,
                fictional_entities,
                preserve_original_gender_ids=preserve_original_gender_ids,
                age_anchor_map=age_anchor_map,
            )
            generated_questions.append(
                {
                    "question_id": q.question_id,
                    "question": gen_question,
                    "answer": q.answer,
                    "question_type": q.question_type,
                    "answer_type": q.answer_type,
                    "reasoning_chain": list(q.reasoning_chain or []),
                }
            )
        generated_text, generated_questions = FictionalDocumentRenderer._rewrite_leftover_factual_literals(
            annotated_doc,
            fictional_entities,
            generated_text,
            generated_questions,
        )
        return FictionalDocument(
            document_id=annotated_doc.document_id,
            document_theme=annotated_doc.document_theme,
            generated_document=generated_text,
            entities_used=fictional_entities,
            questions=generated_questions,
            evaluated_answers=[],
        )

    @staticmethod
    def _find_ambiguous_gender_entity_ids(annotated_doc: AnnotatedDocument) -> set[str]:
        gender_surfaces: dict[str, set[str]] = {}
        for text in [annotated_doc.document_to_annotate, *(q.question for q in annotated_doc.questions)]:
            for ann in AnnotationParser.parse_annotations(text):
                if ann.attribute != "gender":
                    continue
                original = str(ann.original_text or "").strip().lower()
                if not original:
                    continue
                gender_surfaces.setdefault(ann.entity_id, set()).add(original)
        return {entity_id for entity_id, surfaces in gender_surfaces.items() if len(surfaces) > 1}

    @staticmethod
    def _build_age_anchor_map(annotated_doc: AnnotatedDocument) -> dict[str, int]:
        anchors: dict[str, int] = {}
        for text in [annotated_doc.document_to_annotate, *(q.question for q in annotated_doc.questions)]:
            for ann in AnnotationParser.parse_annotations(text):
                if ann.attribute != "age":
                    continue
                try:
                    age_value = int(str(ann.original_text or "").strip())
                except (TypeError, ValueError):
                    continue
                anchors[ann.entity_id] = min(age_value, anchors.get(ann.entity_id, age_value))
        return anchors

    @staticmethod
    def _replace_annotations(
        text: str,
        entities: EntityCollection,
        *,
        preserve_original_gender_ids: set[str] | None = None,
        age_anchor_map: dict[str, int] | None = None,
    ) -> str:
        """Replace all annotations in text with fictional values."""
        annotations = AnnotationParser.parse_annotations(text)
        annotations.sort(key=lambda x: x.start_pos, reverse=True)
        result = text
        preserve_original_gender_ids = preserve_original_gender_ids or set()
        for ann in annotations:
            fictional_value = FictionalDocumentRenderer._get_fictional_value(
                entities,
                ann.entity_id,
                ann.attribute,
                ann.original_text,
                preserve_original_gender=ann.attribute == "gender" and ann.entity_id in preserve_original_gender_ids,
                source_text=text,
                start_pos=ann.start_pos,
                end_pos=ann.end_pos,
                age_anchor_map=age_anchor_map,
            )
            if FictionalDocumentRenderer._is_sentence_start_annotation(text, ann.start_pos):
                fictional_value = FictionalDocumentRenderer._capitalize_first_alpha(fictional_value)
            result = result[: ann.start_pos] + fictional_value + result[ann.end_pos :]
        return result

    @staticmethod
    def _rewrite_leftover_factual_literals(
        annotated_doc: AnnotatedDocument,
        fictional_entities: EntityCollection,
        generated_text: str,
        generated_questions: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        replacements = FictionalDocumentRenderer._literal_rewrite_map(annotated_doc, fictional_entities)
        rewritten_text = FictionalDocumentRenderer._apply_literal_rewrites(generated_text, replacements)
        normalized_text = FictionalDocumentRenderer._normalize_replaced_text(rewritten_text)
        normalized_questions: list[dict[str, Any]] = []
        for question_entry in generated_questions:
            updated_entry = dict(question_entry)
            rewritten_question = FictionalDocumentRenderer._apply_literal_rewrites(
                str(question_entry.get("question") or ""),
                replacements,
            )
            updated_entry["question"] = FictionalDocumentRenderer._normalize_replaced_text(
                rewritten_question
            )
            normalized_questions.append(updated_entry)
        return normalized_text, normalized_questions

    @staticmethod
    def _normalize_replaced_text(text: str) -> str:
        normalized = FictionalDocumentRenderer._collapse_duplicate_surface_suffixes(text)
        normalized = FictionalDocumentRenderer._fix_indefinite_articles(normalized)
        normalized = FictionalDocumentRenderer._fix_definite_articles(normalized)
        normalized = FictionalDocumentRenderer._collapse_duplicate_definite_articles(normalized)
        normalized = FictionalDocumentRenderer._collapse_duplicate_parentheticals(normalized)
        normalized = FictionalDocumentRenderer._repair_birth_age_chronology(normalized)
        normalized = FictionalDocumentRenderer._capitalize_sentence_starts(normalized)
        return normalized

    @staticmethod
    def _literal_rewrite_map(
        annotated_doc: AnnotatedDocument,
        fictional_entities: EntityCollection,
    ) -> dict[str, str]:
        replacements: dict[str, str] = {}
        for ann in AnnotationParser.parse_annotations(annotated_doc.document_to_annotate):
            if not ann.attribute or ann.attribute not in FictionalDocumentRenderer._GLOBAL_LITERAL_REWRITE_ATTRS:
                continue
            original_text = " ".join(str(ann.original_text or "").split())
            if len(original_text) < 4 or not re.search(r"[A-Za-z]", original_text):
                continue
            replacement = " ".join(
                FictionalDocumentRenderer._get_fictional_value(
                    fictional_entities,
                    ann.entity_id,
                    ann.attribute,
                    ann.original_text,
                ).split()
            )
            if not replacement or replacement == original_text:
                continue
            replacements.setdefault(original_text, replacement)
            if original_text.lower() != original_text:
                replacements.setdefault(original_text.lower(), replacement.lower())
            if " " in original_text and not original_text.endswith("s") and not replacement.endswith("s"):
                replacements.setdefault(f"{original_text}s", f"{replacement}s")
                replacements.setdefault(f"{original_text.lower()}s", f"{replacement.lower()}s")
            if ann.attribute == "full_name":
                original_parts = original_text.split()
                replacement_parts = replacement.split()
                if len(original_parts) >= 2 and len(replacement_parts) >= 2:
                    replacements.setdefault(original_parts[-1], replacement_parts[-1])
                    replacements.setdefault(original_parts[0], replacement_parts[0])
        return replacements

    @staticmethod
    def _apply_literal_rewrites(text: str, replacements: dict[str, str]) -> str:
        if not replacements:
            return text
        ordered_literals = sorted(replacements, key=lambda value: (-len(value), value))
        pattern = re.compile("|".join(rf"(?<!\w){re.escape(literal)}(?!\w)" for literal in ordered_literals))

        def _replace(match: re.Match[str]) -> str:
            original = match.group(0)
            replacement = replacements.get(original, original)
            if replacement == original:
                return replacement
            if FictionalDocumentRenderer._is_sentence_start_annotation(text, match.start()):
                replacement = FictionalDocumentRenderer._capitalize_first_alpha(replacement)
            return replacement

        return pattern.sub(_replace, text)

    @staticmethod
    def _fix_indefinite_articles(text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            article = match.group(1)
            next_word = match.group(2)
            if FictionalDocumentRenderer._should_drop_indefinite_article(next_word):
                return next_word
            needs_an = FictionalDocumentRenderer._needs_an_for_token(next_word)
            desired = "an" if needs_an else "a"
            if article[0].isupper():
                desired = desired.capitalize()
            return f"{desired} {next_word}"

        return re.sub(r"\b([Aa]n?)\s+([A-Za-z0-9][A-Za-z0-9'’.\-]*)\b", _replace, text)

    @staticmethod
    def _should_drop_indefinite_article(token: str) -> bool:
        lowered = str(token or "").strip().lower()
        return lowered in FictionalDocumentRenderer._ARTICLELESS_NUMBER_WORDS

    @staticmethod
    def _needs_an_for_token(token: str) -> bool:
        stripped = str(token or "").strip()
        if not stripped:
            return False
        lowered = stripped.lower()
        if lowered[:1] in {"a", "e", "i", "o", "u"}:
            return True
        if stripped[:1].isdigit():
            digits = re.sub(r"\D", "", stripped)
            return digits.startswith(("8", "11", "18"))
        return False

    @staticmethod
    def _collapse_duplicate_surface_suffixes(text: str) -> str:
        normalized = text
        for suffix in FictionalDocumentRenderer._DUPLICATE_SUFFIXES:
            pattern = re.compile(
                rf"\b(?P<phrase>[A-Z][A-Za-z0-9'’\-]*(?:\s+[A-Z][A-Za-z0-9'’\-]*)*\s+{suffix})\s+{suffix}\b",
                flags=re.IGNORECASE,
            )
            normalized = pattern.sub(lambda match: match.group("phrase"), normalized)
        return normalized

    @staticmethod
    def _should_keep_definite_article(phrase: str) -> bool:
        tokens = re.findall(r"[A-Za-z][A-Za-z'’-]*", phrase)
        if not tokens:
            return True
        lowered = [token.lower() for token in tokens]
        if len(lowered) == 1 and lowered[0] in FictionalDocumentRenderer._DEFINITE_ARTICLE_SINGLETONS:
            return True
        if len(tokens) == 1:
            token = tokens[0]
            if len(token) >= 2 and token.isupper():
                return True
            return False
        if any(token in FictionalDocumentRenderer._DEFINITE_ARTICLE_KEEP_TOKENS for token in lowered):
            return True
        return True

    @staticmethod
    def _fix_definite_articles(text: str) -> str:
        pattern = re.compile(r"\b([Tt]he)\s+([A-Z][A-Za-z'’.\-]*(?:\s+(?:of|the|and|[A-Z][A-Za-z'’.\-]*)){0,8})\b")

        def _replace(match: re.Match[str]) -> str:
            article = match.group(1)
            phrase = match.group(2)
            trailing_context = text[match.end() : match.end() + 40]
            next_token_match = re.match(r"\s+([A-Za-z][A-Za-z'’-]*)", trailing_context)
            if next_token_match and next_token_match.group(1)[:1].islower():
                return match.group(0)
            if (
                next_token_match
                and next_token_match.group(1).lower() in FictionalDocumentRenderer._DEFINITE_ARTICLE_KEEP_TOKENS
            ):
                return match.group(0)
            if FictionalDocumentRenderer._should_keep_definite_article(phrase):
                return match.group(0)
            replacement = phrase
            if article[:1].isupper() or FictionalDocumentRenderer._is_sentence_start_annotation(text, match.start()):
                replacement = FictionalDocumentRenderer._capitalize_first_alpha(replacement)
            return replacement

        return pattern.sub(_replace, text)

    @staticmethod
    def _collapse_duplicate_definite_articles(text: str) -> str:
        return re.sub(r"\b([Tt]he)\s+[Tt]he\s+", lambda match: f"{match.group(1)} ", text)

    @staticmethod
    def _collapse_duplicate_parentheticals(text: str) -> str:
        return FictionalDocumentRenderer._DUPLICATE_PAREN_PATTERN.sub(lambda match: match.group("name"), text)

    @staticmethod
    def _repair_birth_age_chronology(text: str) -> str:
        birth_match = FictionalDocumentRenderer._BIRTH_YEAR_PATTERN.search(text)
        if not birth_match:
            return text
        birth_year = int(birth_match.group(1))

        def _replace(match: re.Match[str]) -> str:
            rendered_age = int(match.group(1))
            middle = match.group(2)
            event_year = int(match.group(3))
            implied_age = max(0, event_year - birth_year)
            if abs(implied_age - rendered_age) <= 1:
                return match.group(0)
            return f"at age {implied_age} in {middle}{event_year}"

        return FictionalDocumentRenderer._AGE_IN_YEAR_PATTERN.sub(_replace, text)

    @staticmethod
    def _capitalize_first_alpha(value: str) -> str:
        if not value:
            return value
        chars = list(value)
        for idx, ch in enumerate(chars):
            if ch.isalpha():
                chars[idx] = ch.upper()
                break
        return "".join(chars)

    @staticmethod
    def _capitalize_sentence_starts(text: str) -> str:
        if not text:
            return text
        chars = list(text)
        capitalize_next = True
        idx = 0
        while idx < len(chars):
            char = chars[idx]
            if capitalize_next and char.isalpha():
                chars[idx] = char.upper()
                capitalize_next = False
            elif char in ".!?":
                capitalize_next = True
            elif char == "\n":
                next_index = idx + 1
                while next_index < len(chars) and chars[next_index] == "\n":
                    capitalize_next = True
                    next_index += 1
            elif not char.isspace() and char not in "\"')]}":
                capitalize_next = False
            idx += 1
        return "".join(chars)

    @staticmethod
    def _is_sentence_start_annotation(source_text: str, start_pos: int) -> bool:
        """Return True when an annotation starts at a sentence boundary."""
        if start_pos <= 0:
            return True

        i = start_pos - 1
        while i >= 0 and source_text[i].isspace():
            i -= 1
        if i < 0:
            return True

        # Allow closing punctuation/quotes before a sentence boundary, e.g. ." [X; ...]
        while i >= 0 and source_text[i] in "\"')]}":
            i -= 1
            while i >= 0 and source_text[i].isspace():
                i -= 1
        if i < 0:
            return True

        if source_text[i] in ".!?":
            return True

        gap_text = source_text[i + 1 : start_pos]
        return "\n\n" in gap_text

    @staticmethod
    def _apply_original_casing(template: str, value: str) -> str:
        if template.isupper():
            return value.upper()
        if template.islower():
            return value.lower()
        if template.istitle():
            return value.title()
        return value

    @staticmethod
    def _coerce_age(age_value: Any) -> Optional[int]:
        try:
            return int(age_value) if age_value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _render_gender_surface_form(
        target_gender: Optional[str],
        original_value: Optional[str],
        age_value: Any,
    ) -> Optional[str]:
        if not original_value:
            return None
        target = (target_gender or "").strip().lower()
        if target not in {"male", "female"}:
            return None

        original = str(original_value).strip()
        if not original:
            return None
        original_lower = original.lower()

        if original_lower in FictionalDocumentRenderer._GENDER_ADJECTIVES_SINGULAR:
            return FictionalDocumentRenderer._apply_original_casing(original, target)
        if original_lower in FictionalDocumentRenderer._GENDER_ADJECTIVES_PLURAL:
            return FictionalDocumentRenderer._apply_original_casing(original, f"{target}s")

        all_noun_terms = (
            FictionalDocumentRenderer._GENDER_NOUN_CHILD_SINGULAR
            | FictionalDocumentRenderer._GENDER_NOUN_CHILD_PLURAL
            | FictionalDocumentRenderer._GENDER_NOUN_ADULT_SINGULAR
            | FictionalDocumentRenderer._GENDER_NOUN_ADULT_PLURAL
        )
        if original_lower not in all_noun_terms:
            return None

        age = FictionalDocumentRenderer._coerce_age(age_value)
        if age is not None:
            is_adult = age >= 18
        else:
            is_adult = (
                original_lower in FictionalDocumentRenderer._GENDER_NOUN_ADULT_SINGULAR
                or original_lower in FictionalDocumentRenderer._GENDER_NOUN_ADULT_PLURAL
            )

        is_plural = (
            original_lower in FictionalDocumentRenderer._GENDER_NOUN_CHILD_PLURAL
            or original_lower in FictionalDocumentRenderer._GENDER_NOUN_ADULT_PLURAL
        )
        if target == "male":
            replacement = ("men" if is_plural else "man") if is_adult else ("boys" if is_plural else "boy")
        else:
            replacement = ("women" if is_plural else "woman") if is_adult else ("girls" if is_plural else "girl")

        return FictionalDocumentRenderer._apply_original_casing(original, replacement)

    @staticmethod
    def _format_numeric_surface(value: int | float) -> str:
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            quantized = Decimal(str(value)).quantize(Decimal("0.01"))
            return format(quantized.normalize(), "f").rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _temporal_parts_from_entity(temporal_entity: Any) -> tuple[Optional[int], Optional[str], Optional[int]]:
        year = getattr(temporal_entity, "year", None)
        month = getattr(temporal_entity, "month", None)
        day_of_month = getattr(temporal_entity, "day_of_month", None)
        date_value = getattr(temporal_entity, "date", None)
        if (month is None or day_of_month is None) and isinstance(date_value, str):
            month_day_year = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", date_value.strip())
            if month_day_year:
                if day_of_month is None:
                    day_of_month = int(month_day_year.group(1))
                if month is None:
                    month = month_day_year.group(2)
                if year is None:
                    year = int(month_day_year.group(3))
            month_year = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", date_value.strip())
            if month_year:
                if month is None:
                    month = month_year.group(1)
                if year is None:
                    year = int(month_year.group(2))
        return year, month, day_of_month

    @staticmethod
    def _render_temporal_date_surface(temporal_entity: Any, original_value: Optional[str]) -> Optional[str]:
        if temporal_entity is None:
            return None
        date_value = getattr(temporal_entity, "date", None)
        year, month, day_of_month = FictionalDocumentRenderer._temporal_parts_from_entity(temporal_entity)
        original = str(original_value or "").strip()
        if not original:
            return str(date_value) if date_value is not None else None
        year_range = re.fullmatch(r"(\d{4})\s*[-\u2013\u2014]\s*(\d{2}|\d{4})", original)
        if year_range and year is not None:
            end_text = year_range.group(2)
            span = (int(end_text) - int(year_range.group(1))) if len(end_text) == 4 else int(end_text)
            end_year = year + span
            if len(end_text) == 2:
                return f"{year}\u2013{end_year % 100:02d}"
            return f"{year}\u2013{end_year}"
        if re.fullmatch(r"[A-Za-z]+", original):
            return month or (
                str(date_value).split()[1]
                if isinstance(date_value, str) and len(str(date_value).split()) >= 2
                else str(date_value)
            )
        if re.fullmatch(r"[A-Za-z]+\s+\d{4}", original) and month and year is not None:
            return f"{month} {year}"
        if re.fullmatch(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", original) and month and year is not None:
            day = day_of_month if day_of_month is not None else 1
            return f"{day} {month} {year}"
        if re.fullmatch(r"[A-Za-z]+\s+\d{1,2},\s+\d{4}", original) and month and year is not None:
            day = day_of_month if day_of_month is not None else 1
            return f"{month} {day}, {year}"
        if re.fullmatch(r"[A-Za-z]+\s+\d{1,2}", original) and month:
            day = day_of_month if day_of_month is not None else 1
            return f"{month} {day}"
        if re.fullmatch(r"\d{1,2}\s+[A-Za-z]+", original) and month:
            day = day_of_month if day_of_month is not None else 1
            return f"{day} {month}"
        return str(date_value) if date_value is not None else None

    @staticmethod
    def _render_name_variant(original_value: Optional[str], base_value: str) -> str:
        original = str(original_value or "").strip()
        value = str(base_value or "").strip()
        if not original or not value:
            return value
        original_lower = original.lower()
        value_lower = value.lower()
        if original_lower.startswith("project "):
            if value_lower.startswith("project "):
                return value
            if value_lower.endswith(" program"):
                core = value[:-8].strip()
                return f"Project {core}"
            return f"Project {value}"
        if original_lower.endswith(" program"):
            if value_lower.endswith(" program"):
                return value
            if value_lower.startswith("project "):
                core = value[8:].strip()
                return f"{core} program"
            if "project" in value_lower:
                return value
            return f"{value} program"
        if len(original) > 4 and re.fullmatch(r"[A-Za-z0-9]+", original) and " " in value:
            if value_lower.startswith("project "):
                return value[8:].strip()
            return value.split()[0]
        return value

    @staticmethod
    def _needs_parenthetical_long_form(source_text: Optional[str], end_pos: Optional[int]) -> bool:
        if source_text is None or end_pos is None:
            return False
        return source_text[end_pos:].lstrip().startswith("(")

    @staticmethod
    def _expand_single_token_name(entity_id: str, value: str) -> str:
        if " " in value:
            return value
        entity_type = entity_id.split("_", 1)[0]
        if entity_type in {
            "organization",
            "entreprise",
            "government",
            "educational",
            "media",
            "military",
            "ngo",
        }:
            return f"{value} Group"
        if entity_type == "product":
            return f"{value} Suite"
        return value

    @staticmethod
    def _get_fictional_value(
        entities: EntityCollection,
        entity_id: str,
        attribute: Optional[str],
        original_value: str = None,
        *,
        preserve_original_gender: bool = False,
        source_text: str | None = None,
        start_pos: int | None = None,
        end_pos: int | None = None,
        age_anchor_map: dict[str, int] | None = None,
    ) -> str:
        """Get fictional value for an entity.attribute."""
        del start_pos
        if attribute == "date":
            temporal_entity = entities.temporals.get(entity_id)
            rendered = FictionalDocumentRenderer._render_temporal_date_surface(temporal_entity, original_value)
            if rendered is not None:
                return rendered
        if attribute in {"int", "str"}:
            number_entity = entities.numbers.get(entity_id)
            if number_entity is not None:
                if attribute == "int" and number_entity.int is not None:
                    int_surface_format = (
                        infer_int_surface_format(original_value or "") or number_entity.int_surface_format
                    )
                    return render_integer_surface_number(int(number_entity.int), int_surface_format)
                if attribute == "str" and number_entity.int is not None:
                    str_surface_format = (
                        infer_str_surface_format(original_value or "") or number_entity.str_surface_format
                    )
                    if str_surface_format is not None:
                        return render_word_surface_number(int(number_entity.int), str_surface_format)
                if attribute == "str" and number_entity.str is not None:
                    return str(number_entity.str)
        if attribute == "gender":
            if preserve_original_gender and original_value is not None:
                return str(original_value)
            person_entity = entities.persons.get(entity_id)
            if person_entity is not None:
                rendered = FictionalDocumentRenderer._render_gender_surface_form(
                    target_gender=person_entity.gender,
                    original_value=original_value,
                    age_value=person_entity.age,
                )
                if rendered is not None:
                    return rendered
        if attribute == "age":
            person_entity = entities.persons.get(entity_id)
            rendered_age = FictionalDocumentRenderer._coerce_age(
                getattr(person_entity, "age", None) if person_entity else None
            )
            original_age = FictionalDocumentRenderer._coerce_age(original_value)
            anchor_age = (age_anchor_map or {}).get(entity_id)
            if rendered_age is not None:
                if original_age is not None and anchor_age is not None:
                    return str(rendered_age + (original_age - anchor_age))
                return str(rendered_age)
        if attribute and "." in attribute:
            parts = attribute.split(".")
            if parts[0] == "relationship":
                person_entity = entities.persons.get(entity_id)
                if person_entity and getattr(person_entity, "relationships", None):
                    stored = person_entity.relationships.get(parts[1])
                    if stored is not None:
                        return str(stored)
                if original_value and person_entity:
                    return get_appropriate_relationship(original_value, person_entity)
                entity_ref = f"{entity_id}.relationship.{parts[1]}"
                val = RuleEngine._get_entity_value(entities, entity_ref)
                if val is None:
                    return str(original_value) if original_value else ""
                return (
                    FictionalDocumentRenderer._format_numeric_surface(val)
                    if isinstance(val, (int, float))
                    else str(val)
                )
        entity_ref = f"{entity_id}.{attribute}" if attribute else entity_id
        value = RuleEngine._get_entity_value(entities, entity_ref)
        if value is None:
            return ""
        if attribute == "name":
            rendered_name = FictionalDocumentRenderer._render_name_variant(original_value, str(value))
            if FictionalDocumentRenderer._needs_parenthetical_long_form(source_text, end_pos):
                return FictionalDocumentRenderer._expand_single_token_name(entity_id, rendered_name)
            return rendered_name
        return (
            FictionalDocumentRenderer._format_numeric_surface(value) if isinstance(value, (int, float)) else str(value)
        )


# ---------------------------------------------------------------------------
# Number and temporal entity generation (constraint-aware)
