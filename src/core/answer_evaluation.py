"""Evaluate template answer expressions against document entities."""

from __future__ import annotations

import ast
import re
from typing import Any

from src.core.annotation_runtime import (
    RuleEngine,
    _coerce_numeric_surface,
    find_entity_refs,
    is_valid_entity_ref,
)
from src.core.answer_matching import canonicalize_answer_value
from src.core.document_schema import EntityCollection


class AnswerEvaluator:
    """Evaluate answer expressions for benchmark questions."""

    @staticmethod
    def evaluate_answer(answer_expr: str, entities: EntityCollection) -> Any:
        if not answer_expr:
            return ""
        if isinstance(answer_expr, str) and answer_expr.strip().startswith("[") and answer_expr.strip().endswith("]"):
            try:
                parsed = ast.literal_eval(answer_expr.strip())
                if isinstance(parsed, list) and len(parsed) > 0:
                    answer_expr = str(parsed[0])
            except (ValueError, SyntaxError):
                answer_expr = answer_expr.strip()[1:-1].strip()
                if (answer_expr.startswith("'") and answer_expr.endswith("'")) or (
                    answer_expr.startswith('"') and answer_expr.endswith('"')
                ):
                    answer_expr = answer_expr[1:-1]
        answer_expr = AnswerEvaluator._clean_semicolon_syntax(answer_expr)
        conditional_result = AnswerEvaluator._evaluate_conditional_expression(answer_expr, entities)
        if conditional_result is not None:
            return canonicalize_answer_value(conditional_result)
        textual_result = AnswerEvaluator._evaluate_textual_expression(answer_expr, entities)
        if textual_result is not None:
            return canonicalize_answer_value(textual_result)
        result = RuleEngine.evaluate_expression(answer_expr, entities)
        if str(result or "").strip() == str(answer_expr or "").strip():
            textual_result = AnswerEvaluator._render_symbol_affixed_entity_ref(answer_expr, entities)
            if textual_result is not None:
                return canonicalize_answer_value(textual_result)
        return canonicalize_answer_value(result)

    @staticmethod
    def _clean_semicolon_syntax(expr: str) -> str:
        """Normalize legacy answer-expression semicolon syntax."""
        if not expr or not isinstance(expr, str):
            return expr
        expr = expr.strip()
        if "[" in expr and ";" in expr:
            cleaned = re.sub(r"\[([^\]]+);\s*([^\]]+)\]", r"\2", expr)
            if cleaned != expr:
                return cleaned.strip()
        if ";" in expr:
            parts = expr.split(";", 1)
            if len(parts) == 2:
                literal, refs_part = parts[0].strip(), parts[1].strip()
                if is_valid_entity_ref(refs_part):
                    return refs_part
                stripped = refs_part.strip('"').strip("'").strip()
                if stripped and find_entity_refs(stripped):
                    return stripped
                return literal
        return expr

    @staticmethod
    def _evaluate_conditional_expression(answer_expr: str, entities: EntityCollection) -> Any | None:
        expr = str(answer_expr or "").strip()
        if not re.match(r"^.+?\s+if\s+.+?\s+else\s+.+$", expr):
            return None
        return RuleEngine.evaluate_expression(expr, entities)

    @staticmethod
    def _evaluate_textual_expression(answer_expr: str, entities: EntityCollection) -> str | None:
        concatenated = AnswerEvaluator._evaluate_string_concatenation(answer_expr, entities)
        if concatenated is not None:
            return concatenated
        return AnswerEvaluator._render_symbol_affixed_entity_ref(answer_expr, entities)

    @staticmethod
    def _evaluate_string_concatenation(answer_expr: str, entities: EntityCollection) -> str | None:
        parts = AnswerEvaluator._split_concat_terms(answer_expr)
        if len(parts) <= 1:
            return None

        resolved_parts: list[Any] = []
        has_string_literal = False
        has_non_numeric_part = False
        for part in parts:
            if not part:
                return None
            if AnswerEvaluator._is_string_literal(part):
                has_string_literal = True
            resolved = AnswerEvaluator._resolve_concat_term(part, entities)
            if resolved is None:
                return None
            resolved_parts.append(resolved)
            if not AnswerEvaluator._is_numericish_value(resolved):
                has_non_numeric_part = True

        if not has_string_literal and not has_non_numeric_part:
            return None
        return "".join(AnswerEvaluator._stringify_concat_part(part) for part in resolved_parts).strip()

    @staticmethod
    def _split_concat_terms(expression: str) -> list[str]:
        expr = str(expression or "").strip()
        if not expr or "+" not in expr:
            return [expr] if expr else []

        terms: list[str] = []
        buffer: list[str] = []
        quote_char: str | None = None
        escaping = False
        for char in expr:
            if escaping:
                buffer.append(char)
                escaping = False
                continue
            if quote_char is not None:
                buffer.append(char)
                if char == "\\":
                    escaping = True
                elif char == quote_char:
                    quote_char = None
                continue
            if char in {"'", '"'}:
                quote_char = char
                buffer.append(char)
                continue
            if char == "+":
                terms.append("".join(buffer).strip())
                buffer = []
                continue
            buffer.append(char)
        terms.append("".join(buffer).strip())
        return terms

    @staticmethod
    def _is_string_literal(token: str) -> bool:
        stripped = str(token or "").strip()
        if len(stripped) < 2 or stripped[0] not in {"'", '"'} or stripped[-1] != stripped[0]:
            return False
        try:
            return isinstance(ast.literal_eval(stripped), str)
        except (ValueError, SyntaxError):
            return False

    @staticmethod
    def _resolve_concat_term(token: str, entities: EntityCollection) -> Any | None:
        stripped = str(token or "").strip()
        if not stripped:
            return None
        if AnswerEvaluator._is_string_literal(stripped):
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return None
        result = RuleEngine.evaluate_expression(stripped, entities)
        if result is None:
            return None
        if str(result).strip() == stripped and find_entity_refs(stripped):
            return AnswerEvaluator._render_symbol_affixed_entity_ref(stripped, entities)
        return result

    @staticmethod
    def _render_symbol_affixed_entity_ref(answer_expr: str, entities: EntityCollection) -> str | None:
        expr = str(answer_expr or "").strip()
        refs = find_entity_refs(expr)
        if len(refs) != 1:
            return None
        ref = refs[0]
        resolved = RuleEngine._get_entity_value(entities, ref)
        if resolved is None:
            return None

        literal_shell = expr.replace(ref, " ").strip()
        if not literal_shell or re.search(r"[A-Za-z0-9_]", literal_shell):
            return None

        rendered = expr.replace(ref, canonicalize_answer_value(resolved))
        rendered = re.sub(r"\s+", " ", rendered).strip()
        rendered = re.sub(r"([$€£¥])\s+(?=\d)", r"\1", rendered)
        rendered = re.sub(r"(?<=\d)\s+(%)", r"\1", rendered)
        return rendered or None

    @staticmethod
    def _is_numericish_value(value: Any) -> bool:
        return _coerce_numeric_surface(value) is not None

    @staticmethod
    def _stringify_concat_part(value: Any) -> str:
        if value is True:
            return "Yes"
        if value is False:
            return "No"
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def evaluate_all_answers(questions: list[Any], entities: EntityCollection) -> list[dict[str, Any]]:
        results = []
        for question in questions:
            if isinstance(question, dict):
                question_id, question_text, answer = (
                    question.get("question_id"),
                    question.get("question"),
                    question.get("answer", ""),
                )
            else:
                question_id, question_text, answer = question.question_id, question.question, question.answer
            answer_expr = AnswerEvaluator._clean_semicolon_syntax(answer)
            evaluated = AnswerEvaluator.evaluate_answer(answer_expr, entities)
            results.append(
                {
                    "question_id": question_id,
                    "question": question_text,
                    "answer_expression": answer_expr,
                    "evaluated_answer": canonicalize_answer_value(evaluated),
                }
            )
        return results

    @staticmethod
    def build_question_entries_with_answers(
        questions: list[dict[str, Any]],
        evaluated_answers: list[dict[str, Any]],
        entities: EntityCollection,
    ) -> list[dict[str, Any]]:
        eval_answers_map = {ea["question_id"]: ea["evaluated_answer"] for ea in evaluated_answers}
        entries = []
        for question in questions:
            answer_expr = question.get("answer", "")
            cleaned_expr = AnswerEvaluator._clean_semicolon_syntax(answer_expr)
            entity_refs = find_entity_refs(str(answer_expr))
            answer_entities = {
                ref: RuleEngine._get_entity_value(entities, ref)
                for ref in entity_refs
                if RuleEngine._get_entity_value(entities, ref) is not None
            }
            entries.append(
                {
                    "question_id": question["question_id"],
                    "question_type": question.get("question_type"),
                    "question": question["question"],
                    "answer_expression": cleaned_expr,
                    "answer_entities": answer_entities if answer_entities else None,
                    "evaluated_answer": eval_answers_map.get(question["question_id"], ""),
                }
            )
        return entries


__all__ = ["AnswerEvaluator"]
