"""Compatibility facade for rendering fictional document variants."""

from .fictional_document_rendering import (
    _build_age_anchor_map,
    _find_ambiguous_gender_entity_ids,
    _replace_annotations,
    render_document,
)
from .fictional_document_rendering_constants import (
    _GENDER_NOUN_CHILD_SINGULAR,
    _GENDER_NOUN_CHILD_PLURAL,
    _GENDER_NOUN_ADULT_SINGULAR,
    _GENDER_NOUN_ADULT_PLURAL,
    _GENDER_ADJECTIVES_SINGULAR,
    _GENDER_ADJECTIVES_PLURAL,
    _ARTICLELESS_NUMBER_WORDS,
    _GLOBAL_LITERAL_REWRITE_ATTRS,
    _DUPLICATE_PAREN_PATTERN,
    _BIRTH_YEAR_PATTERN,
    _AGE_IN_YEAR_PATTERN,
    _DUPLICATE_SUFFIXES,
    _DEFINITE_ARTICLE_KEEP_TOKENS,
    _DEFINITE_ARTICLE_SINGLETONS,
)
from .fictional_document_surface_rendering import (
    _capitalize_first_alpha,
    _capitalize_sentence_starts,
    _is_sentence_start_annotation,
    _apply_original_casing,
    _coerce_age,
    _render_gender_surface_form,
    _format_numeric_surface,
    _temporal_parts_from_entity,
    _render_temporal_date_surface,
    _render_name_variant,
    _needs_parenthetical_long_form,
    _expand_single_token_name,
    _get_fictional_value,
)
from .fictional_document_text_normalization import (
    _rewrite_leftover_factual_literals,
    _normalize_replaced_text,
    _literal_rewrite_map,
    _apply_literal_rewrites,
    _fix_indefinite_articles,
    _should_drop_indefinite_article,
    _needs_an_for_token,
    _collapse_duplicate_surface_suffixes,
    _should_keep_definite_article,
    _fix_definite_articles,
    _collapse_duplicate_definite_articles,
    _collapse_duplicate_parentheticals,
    _repair_birth_age_chronology,
)


class FictionalDocumentRenderer:
    """Render a fictional document variant from an annotated factual template."""

    _GENDER_NOUN_CHILD_SINGULAR = _GENDER_NOUN_CHILD_SINGULAR
    _GENDER_NOUN_CHILD_PLURAL = _GENDER_NOUN_CHILD_PLURAL
    _GENDER_NOUN_ADULT_SINGULAR = _GENDER_NOUN_ADULT_SINGULAR
    _GENDER_NOUN_ADULT_PLURAL = _GENDER_NOUN_ADULT_PLURAL
    _GENDER_ADJECTIVES_SINGULAR = _GENDER_ADJECTIVES_SINGULAR
    _GENDER_ADJECTIVES_PLURAL = _GENDER_ADJECTIVES_PLURAL
    _ARTICLELESS_NUMBER_WORDS = _ARTICLELESS_NUMBER_WORDS
    _GLOBAL_LITERAL_REWRITE_ATTRS = _GLOBAL_LITERAL_REWRITE_ATTRS
    _DUPLICATE_PAREN_PATTERN = _DUPLICATE_PAREN_PATTERN
    _BIRTH_YEAR_PATTERN = _BIRTH_YEAR_PATTERN
    _AGE_IN_YEAR_PATTERN = _AGE_IN_YEAR_PATTERN
    _DUPLICATE_SUFFIXES = _DUPLICATE_SUFFIXES
    _DEFINITE_ARTICLE_KEEP_TOKENS = _DEFINITE_ARTICLE_KEEP_TOKENS
    _DEFINITE_ARTICLE_SINGLETONS = _DEFINITE_ARTICLE_SINGLETONS

    render_document = staticmethod(render_document)
    _find_ambiguous_gender_entity_ids = staticmethod(_find_ambiguous_gender_entity_ids)
    _build_age_anchor_map = staticmethod(_build_age_anchor_map)
    _replace_annotations = staticmethod(_replace_annotations)
    _rewrite_leftover_factual_literals = staticmethod(_rewrite_leftover_factual_literals)
    _normalize_replaced_text = staticmethod(_normalize_replaced_text)
    _literal_rewrite_map = staticmethod(_literal_rewrite_map)
    _apply_literal_rewrites = staticmethod(_apply_literal_rewrites)
    _fix_indefinite_articles = staticmethod(_fix_indefinite_articles)
    _should_drop_indefinite_article = staticmethod(_should_drop_indefinite_article)
    _needs_an_for_token = staticmethod(_needs_an_for_token)
    _collapse_duplicate_surface_suffixes = staticmethod(_collapse_duplicate_surface_suffixes)
    _should_keep_definite_article = staticmethod(_should_keep_definite_article)
    _fix_definite_articles = staticmethod(_fix_definite_articles)
    _collapse_duplicate_definite_articles = staticmethod(_collapse_duplicate_definite_articles)
    _collapse_duplicate_parentheticals = staticmethod(_collapse_duplicate_parentheticals)
    _repair_birth_age_chronology = staticmethod(_repair_birth_age_chronology)
    _capitalize_first_alpha = staticmethod(_capitalize_first_alpha)
    _capitalize_sentence_starts = staticmethod(_capitalize_sentence_starts)
    _is_sentence_start_annotation = staticmethod(_is_sentence_start_annotation)
    _apply_original_casing = staticmethod(_apply_original_casing)
    _coerce_age = staticmethod(_coerce_age)
    _render_gender_surface_form = staticmethod(_render_gender_surface_form)
    _format_numeric_surface = staticmethod(_format_numeric_surface)
    _temporal_parts_from_entity = staticmethod(_temporal_parts_from_entity)
    _render_temporal_date_surface = staticmethod(_render_temporal_date_surface)
    _render_name_variant = staticmethod(_render_name_variant)
    _needs_parenthetical_long_form = staticmethod(_needs_parenthetical_long_form)
    _expand_single_token_name = staticmethod(_expand_single_token_name)
    _get_fictional_value = staticmethod(_get_fictional_value)


__all__ = ["FictionalDocumentRenderer"]
