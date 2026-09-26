"""Public API for MemoReason reviewed documents, annotations, and rules.

The implementation remains split by scientific responsibility. This module is
the deliberate import boundary consumed by dataset construction, model
evaluation, paper reporting, and the annotation interface.
"""

from .annotated_document_io import create_run_dir, fictional_document_to_yaml_dict, load_annotated_document
from .annotation_parsing import Annotation, AnnotationParser
from .annotation_references import (
    find_entity_refs,
    get_appropriate_relationship,
    is_valid_entity_ref,
    map_relationship_for_gender,
    normalize_entity_ref,
    normalize_text_entity_refs,
)
from .annotation_rules import (
    compose_rule_text,
    find_rule_sanity_errors,
    normalize_document_taxonomy,
    normalize_rule_expressions,
    normalize_rules_for_storage,
    partition_generation_rules,
    split_rule_text_and_comment,
)
from .annotation_validation import (
    AnnotationValidationError,
    validate_annotations,
    validate_question_and_answer_entity_scope,
)
from .entity_taxonomy import (
    ENTITY_TAXONOMY,
    FULL_REPLACE_ENTITY_TYPES,
    PARTIAL_REPLACE_ATTRIBUTES,
    REPLACE_MODE_ALL,
    REPLACE_MODE_NON_NUMERICAL,
    REPLACE_MODE_NUMERICAL,
    REPLACE_MODE_TEMPORAL,
    VALID_ENTITY_TYPES,
    VALID_REPLACE_MODES,
    WORD_TO_NUMBER,
    parse_entity_id,
    parse_word_number,
    replace_mode_label,
)
from .implicit_numeric_rules import normalize_implicit_rules_for_storage
from .entity_pool_io import load_entity_pool
from .organization_types import ORG_ENTITY_TYPES
from .rule_engine import RuleEngine

__all__ = [
    "ENTITY_TAXONOMY",
    "FULL_REPLACE_ENTITY_TYPES",
    "ORG_ENTITY_TYPES",
    "PARTIAL_REPLACE_ATTRIBUTES",
    "REPLACE_MODE_ALL",
    "REPLACE_MODE_NON_NUMERICAL",
    "REPLACE_MODE_NUMERICAL",
    "REPLACE_MODE_TEMPORAL",
    "VALID_ENTITY_TYPES",
    "VALID_REPLACE_MODES",
    "WORD_TO_NUMBER",
    "Annotation",
    "AnnotationParser",
    "AnnotationValidationError",
    "RuleEngine",
    "compose_rule_text",
    "create_run_dir",
    "fictional_document_to_yaml_dict",
    "find_entity_refs",
    "find_rule_sanity_errors",
    "get_appropriate_relationship",
    "is_valid_entity_ref",
    "load_annotated_document",
    "load_entity_pool",
    "map_relationship_for_gender",
    "normalize_document_taxonomy",
    "normalize_entity_ref",
    "normalize_text_entity_refs",
    "normalize_implicit_rules_for_storage",
    "normalize_rule_expressions",
    "normalize_rules_for_storage",
    "parse_entity_id",
    "parse_word_number",
    "partition_generation_rules",
    "replace_mode_label",
    "split_rule_text_and_comment",
    "validate_annotations",
    "validate_question_and_answer_entity_scope",
]
