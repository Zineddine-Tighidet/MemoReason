"""Static proofs that a reviewed variant answer can actually change."""

from __future__ import annotations

import ast
from fractions import Fraction
from itertools import pairwise
import re
from typing import Any

from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs
from memoreason.benchmark_definition.answer_expression_evaluation import AnswerEvaluator
from memoreason.benchmark_definition.document_schema import AnnotatedDocument, EntityCollection
from memoreason.benchmark_definition.entity_taxonomy import parse_integer_surface_number, parse_word_number

AffineExpression = tuple[dict[str, Fraction], Fraction]


def _entity_ref_from_ast(node: ast.AST) -> str | None:
    parts: list[str] = []
    cursor = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        return None
    parts.append(cursor.id)
    entity_ref = ".".join(reversed(parts))
    if not re.fullmatch(r"[a-z]+_\d+(?:\.[A-Za-z_]\w*)*", entity_ref):
        return None
    return entity_ref


def _fraction_from_literal(value: Any) -> Fraction | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return Fraction(str(value))
    cleaned = str(value).strip()
    parsed_integer = parse_integer_surface_number(cleaned)
    if parsed_integer is None:
        parsed_integer = parse_word_number(cleaned)
    if parsed_integer is not None:
        return Fraction(parsed_integer)
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def _combine_affine(
    lhs: AffineExpression,
    rhs: AffineExpression,
    *,
    rhs_scale: Fraction = Fraction(1),
) -> AffineExpression:
    coefficients = dict(lhs[0])
    for entity_ref, coefficient in rhs[0].items():
        combined = coefficients.get(entity_ref, Fraction(0)) + rhs_scale * coefficient
        if combined:
            coefficients[entity_ref] = combined
        else:
            coefficients.pop(entity_ref, None)
    return coefficients, lhs[1] + rhs_scale * rhs[1]


def _scale_affine(expression: AffineExpression, factor: Fraction) -> AffineExpression:
    return (
        {entity_ref: coefficient * factor for entity_ref, coefficient in expression[0].items() if coefficient},
        expression[1] * factor,
    )


def _affine_expression(node: ast.AST) -> AffineExpression | None:
    if isinstance(node, ast.Constant):
        literal = _fraction_from_literal(node.value)
        return ({}, literal) if literal is not None else None
    if isinstance(node, ast.Name):
        literal = _fraction_from_literal(node.id)
        if literal is not None:
            return {}, literal
    entity_ref = _entity_ref_from_ast(node)
    if entity_ref is not None:
        return {entity_ref: Fraction(1)}, Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _affine_expression(node.operand)
        if operand is None:
            return None
        return operand if isinstance(node.op, ast.UAdd) else _scale_affine(operand, Fraction(-1))
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        lhs = _affine_expression(node.left)
        rhs = _affine_expression(node.right)
        if lhs is None or rhs is None:
            return None
        return _combine_affine(lhs, rhs, rhs_scale=Fraction(-1) if isinstance(node.op, ast.Sub) else Fraction(1))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        lhs = _affine_expression(node.left)
        rhs = _affine_expression(node.right)
        if lhs is None or rhs is None or (lhs[0] and rhs[0]):
            return None
        return _scale_affine(lhs, rhs[1]) if lhs[0] else _scale_affine(rhs, lhs[1])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        numerator = _affine_expression(node.left)
        denominator = _affine_expression(node.right)
        if numerator is None or denominator is None or denominator[0] or not denominator[1]:
            return None
        return _scale_affine(numerator, Fraction(1, 1) / denominator[1])
    return None


def _guaranteed_rule_nodes(rules: list[str]) -> list[ast.AST]:
    guaranteed: list[ast.AST] = []

    def collect(node: ast.AST) -> None:
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            for value in node.values:
                collect(value)
            return
        guaranteed.append(node)

    for raw_rule in rules:
        cleaned = str(raw_rule or "").split("#", 1)[0].strip()
        if not cleaned:
            continue
        normalized = RuleEngine._normalize_comparison_expression(cleaned)
        try:
            collect(ast.parse(normalized, mode="eval").body)
        except SyntaxError:
            continue
    return guaranteed


def _equality_rows(guaranteed_rule_nodes: list[ast.AST]) -> list[AffineExpression]:
    rows: list[AffineExpression] = []
    for node in guaranteed_rule_nodes:
        if not isinstance(node, ast.Compare) or not node.ops or not all(isinstance(op, ast.Eq) for op in node.ops):
            continue
        operands = [node.left, *node.comparators]
        for lhs_node, rhs_node in pairwise(operands):
            lhs = _affine_expression(lhs_node)
            rhs = _affine_expression(rhs_node)
            if lhs is None or rhs is None:
                continue
            difference = _combine_affine(lhs, rhs, rhs_scale=Fraction(-1))
            if difference[0]:
                rows.append(difference)
    return rows


def _fixed_affine_value(
    expression: AffineExpression,
    equality_rows: list[AffineExpression],
) -> Fraction | None:
    if not expression[0]:
        return expression[1]
    variables = sorted({*expression[0], *(entity_ref for row in equality_rows for entity_ref in row[0])})
    rows = [[*(row[0].get(variable, Fraction(0)) for variable in variables), -row[1]] for row in equality_rows]
    pivot_rows: list[tuple[int, list[Fraction]]] = []
    for column in range(len(variables)):
        pivot_index = next((index for index, row in enumerate(rows) if row[column]), None)
        if pivot_index is None:
            continue
        pivot = rows.pop(pivot_index)
        pivot_value = pivot[column]
        pivot = [value / pivot_value for value in pivot]
        for row_index, row in enumerate(rows):
            factor = row[column]
            if factor:
                rows[row_index] = [value - factor * pivot_value for value, pivot_value in zip(row, pivot, strict=True)]
        for pivot_index, (pivot_column, prior) in enumerate(pivot_rows):
            factor = prior[column]
            if factor:
                pivot_rows[pivot_index] = (
                    pivot_column,
                    [value - factor * pivot_value for value, pivot_value in zip(prior, pivot, strict=True)],
                )
        pivot_rows.append((column, pivot))

    residual = [expression[0].get(variable, Fraction(0)) for variable in variables]
    fixed_value = expression[1]
    for column, row in pivot_rows:
        factor = residual[column]
        if not factor:
            continue
        residual = [value - factor * row_value for value, row_value in zip(residual, row[:-1], strict=True)]
        fixed_value += factor * row[-1]
    return fixed_value if not any(residual) else None


def _expression_is_fixed_by_explicit_rules(
    *,
    parsed_expression: ast.AST,
    expression_refs: set[str],
    rules: list[str],
) -> bool:
    guaranteed_nodes = _guaranteed_rule_nodes(rules)
    if isinstance(parsed_expression, ast.IfExp):
        guaranteed_predicates = {ast.dump(node, include_attributes=False) for node in guaranteed_nodes}
        if ast.dump(parsed_expression.test, include_attributes=False) in guaranteed_predicates:
            selected_branch = parsed_expression.body
            selected_refs = set(find_entity_refs(ast.unparse(selected_branch)))
            if not selected_refs:
                return True
            return _expression_is_fixed_by_explicit_rules(
                parsed_expression=selected_branch,
                expression_refs=selected_refs,
                rules=rules,
            )

    affine = _affine_expression(parsed_expression)
    equality_rows = _equality_rows(guaranteed_nodes)
    if affine is not None and equality_rows and _fixed_affine_value(affine, equality_rows) is not None:
        return True

    literal_bound_refs: set[str] = set()
    for node in guaranteed_nodes:
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
            continue
        lhs, rhs = node.left, node.comparators[0]
        for ref_node, literal_node in ((lhs, rhs), (rhs, lhs)):
            entity_ref = _entity_ref_from_ast(ref_node)
            if entity_ref is not None and isinstance(literal_node, ast.Constant):
                literal_bound_refs.add(entity_ref)
    return bool(expression_refs) and expression_refs <= literal_bound_refs


def _expression_is_algebraically_constant(*, parsed_expression: ast.AST, expression_refs: set[str]) -> bool:
    if not expression_refs:
        return False
    if isinstance(parsed_expression, ast.IfExp) and ast.dump(
        parsed_expression.body,
        include_attributes=False,
    ) == ast.dump(parsed_expression.orelse, include_attributes=False):
        return True
    affine = _affine_expression(parsed_expression)
    return affine is not None and not affine[0]


def static_variant_answer_contract_errors(
    *,
    generation_document: AnnotatedDocument,
    factual_entities: EntityCollection,
) -> list[str]:
    """Return variant questions whose declared answer cannot vary by construction."""
    errors: list[str] = []
    for question in generation_document.questions:
        if str(getattr(question, "answer_type", "") or "").strip().lower() != "variant":
            continue
        question_id = str(getattr(question, "question_id", "") or "<unknown>")
        expression = AnswerEvaluator._clean_semicolon_syntax(str(getattr(question, "answer", "") or "")).strip()
        refs = find_entity_refs(expression)
        if not refs:
            errors.append(f"{question_id}: variant answer expression has no entity reference: {expression!r}")
            continue
        factual_answer = AnswerEvaluator.evaluate_answer(expression, factual_entities)
        if str(factual_answer or "").strip() == expression:
            errors.append(
                f"{question_id}: variant answer expression does not resolve its entity reference: {expression!r}"
            )
            continue
        try:
            parsed = ast.parse(expression, mode="eval").body
        except SyntaxError:
            parsed = None
        if (
            isinstance(parsed, ast.BinOp)
            and isinstance(parsed.op, (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod))
            and ast.dump(parsed.left, include_attributes=False) == ast.dump(parsed.right, include_attributes=False)
        ):
            errors.append(f"{question_id}: variant answer expression is self-cancelling: {expression!r}")
            continue
        expression_refs = set(refs)
        if parsed is not None and _expression_is_algebraically_constant(
            parsed_expression=parsed,
            expression_refs=expression_refs,
        ):
            errors.append(f"{question_id}: variant answer expression is algebraically constant: {expression!r}")
            continue
        if parsed is not None and _expression_is_fixed_by_explicit_rules(
            parsed_expression=parsed,
            expression_refs=expression_refs,
            rules=generation_document.rules,
        ):
            errors.append(f"{question_id}: variant answer expression is fixed by explicit rules: {expression!r}")
    return errors


def _fraction_rule_literal(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"({value.numerator} / {value.denominator})"


def _solver_supported_answer_ref(entity_ref: str) -> bool:
    if "." not in entity_ref:
        return False
    entity_id, attr = entity_ref.split(".", 1)
    if entity_id.startswith("number_"):
        return attr in {"int", "str", "float", "percent", "proportion"}
    if entity_id.startswith("temporal_"):
        return attr in {"year", "date", "date.year"}
    return False


def _inverted_comparison_operator(operator: ast.cmpop) -> ast.cmpop | None:
    inverted_types: dict[type[ast.cmpop], type[ast.cmpop]] = {
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.Lt: ast.GtE,
        ast.LtE: ast.Gt,
        ast.Gt: ast.LtE,
        ast.GtE: ast.Lt,
    }
    inverted_type = inverted_types.get(type(operator))
    return inverted_type() if inverted_type is not None else None


def _conditional_answer_difference_rule(
    *,
    expression: ast.AST,
    factual_entities: EntityCollection,
    active_numerical_entity_ids: set[str],
) -> str | None:
    """Return the opposite branch condition for a simple constant conditional.

    This turns expressions such as ``Yes if number_1.int > 20 else No``
    into a solver-visible constraint.  Only a single comparison with constant
    branch answers is accepted; richer conditionals remain protected by the
    final semantic collision gate and deterministic retries.
    """
    if not isinstance(expression, ast.IfExp) or not isinstance(expression.test, ast.Compare):
        return None
    if len(expression.test.ops) != 1 or len(expression.test.comparators) != 1:
        return None
    body_expression = ast.unparse(expression.body)
    else_expression = ast.unparse(expression.orelse)
    if find_entity_refs(body_expression) or find_entity_refs(else_expression):
        return None
    body_answer = AnswerEvaluator.evaluate_answer(body_expression, factual_entities)
    else_answer = AnswerEvaluator.evaluate_answer(else_expression, factual_entities)
    if str(body_answer).strip() == str(else_answer).strip():
        return None
    full_answer = AnswerEvaluator.evaluate_answer(ast.unparse(expression), factual_entities)
    full_answer_text = str(full_answer).strip()
    body_selected = full_answer_text == str(body_answer).strip()
    else_selected = full_answer_text == str(else_answer).strip()
    if body_selected == else_selected:
        return None
    condition_refs = find_entity_refs(ast.unparse(expression.test))
    if not condition_refs or not all(_solver_supported_answer_ref(ref) for ref in condition_refs):
        return None
    condition_ref_ids = {ref.split(".", 1)[0] for ref in condition_refs}
    if not condition_ref_ids.intersection(active_numerical_entity_ids):
        return None
    comparison = expression.test
    operator = comparison.ops[0]
    if body_selected:
        operator = _inverted_comparison_operator(operator)
        if operator is None:
            return None
    opposite_condition = ast.Compare(
        left=comparison.left,
        ops=[operator],
        comparators=comparison.comparators,
    )
    return ast.unparse(ast.fix_missing_locations(opposite_condition))


def variant_answer_difference_rules(
    *,
    generation_document: AnnotatedDocument,
    factual_entities: EntityCollection,
    active_numerical_entity_ids: set[str],
) -> list[str]:
    """Build solver constraints for feasible affine variant answers.

    The constraints are scoped to a concrete replacement layout.  This means
    numeric answers are constrained for full and num+temp generation, but not
    for named-only ablations where those entities intentionally stay factual.
    Simple constant-branch conditionals are converted to the opposite branch
    constraint.  Other non-affine/string answers remain covered by the final
    semantic collision gate and deterministic retries.
    """
    rules: list[str] = []
    seen: set[str] = set()
    for question in generation_document.questions:
        if str(getattr(question, "answer_type", "") or "").strip().lower() != "variant":
            continue
        expression = AnswerEvaluator._clean_semicolon_syntax(str(getattr(question, "answer", "") or "")).strip()
        refs = find_entity_refs(expression)
        if not refs or not all(_solver_supported_answer_ref(ref) for ref in refs):
            continue
        ref_ids = {ref.split(".", 1)[0] for ref in refs}
        if not ref_ids.intersection(active_numerical_entity_ids):
            continue
        try:
            parsed_expression = ast.parse(expression, mode="eval").body
        except SyntaxError:
            continue
        conditional_rule = _conditional_answer_difference_rule(
            expression=parsed_expression,
            factual_entities=factual_entities,
            active_numerical_entity_ids=active_numerical_entity_ids,
        )
        if conditional_rule is not None:
            if conditional_rule not in seen:
                seen.add(conditional_rule)
                rules.append(conditional_rule)
            continue
        affine = _affine_expression(parsed_expression)
        if affine is None or not affine[0]:
            continue
        factual_answer = AnswerEvaluator.evaluate_answer(expression, factual_entities)
        factual_value = _fraction_from_literal(factual_answer)
        if factual_value is None:
            continue
        rule = f"({expression}) != {_fraction_rule_literal(factual_value)}"
        if rule not in seen:
            seen.add(rule)
            rules.append(rule)
    return rules


__all__ = [
    "static_variant_answer_contract_errors",
    "variant_answer_difference_rules",
]
