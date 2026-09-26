"""Load annotated MemoReason documents and serialize fictional documents."""

from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import yaml

from .document_schema import AnnotatedDocument, FictionalDocument, ImplicitRule, Question
from .implicit_numeric_rules import normalize_implicit_rule_exclusions, normalize_implicit_rules_for_storage
from .annotation_rules import normalize_document_taxonomy, normalize_rule_expressions
from .annotation_validation import validate_annotations, validate_question_and_answer_entity_scope

UTC = UTC


def _sanitize(s: str) -> str:
    """Replace characters unsafe for directory names."""
    return s.replace("/", "-").replace(" ", "_")[:64]


def create_run_dir(
    provider: str,
    base_dir: Path | None = None,
    *,
    seed: int | None = None,
    models: list[str] | None = None,
    documents: list[str] | None = None,
    proportions: list[float] | None = None,
    num_versions: int | None = None,
    skip_generation: bool = False,
    skip_evaluation: bool = False,
    **kwargs: Any,
) -> Path:
    """Create a traceable run directory and write run_params.json.

    Directory name format: {timestamp}_{provider}_seed_{seed}_models_{...}_docs_{...}
    This ensures each run is uniquely identifiable by when it ran, which
    provider served the models, and all relevant hyperparameters.
    """
    if base_dir is None:
        base_dir = Path("results") / _sanitize(provider)
    base_dir = Path(base_dir)
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    parts = [timestamp, _sanitize(provider)]
    if seed is not None:
        parts.append(f"seed_{seed}")
    if documents:
        parts.append("docs_" + "_".join(_sanitize(d) for d in documents[:5]))
    if models:
        parts.append("models_" + "_".join(_sanitize(m) for m in models[:3]))
    dir_name = "_".join(parts)
    run_dir = base_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    params = {"provider": provider, "datetime_iso": now.isoformat(), "timestamp": timestamp}
    if seed is not None:
        params["seed"] = seed
    if models is not None:
        params["models"] = models
    if documents is not None:
        params["documents"] = documents
    if proportions is not None:
        params["proportions"] = proportions
    if num_versions is not None:
        params["num_versions"] = num_versions
    if skip_generation:
        params["skip_generation"] = skip_generation
    if skip_evaluation:
        params["skip_evaluation"] = skip_evaluation
    params.update(kwargs)
    with open(run_dir / "run_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return run_dir


def load_annotated_document(yaml_path: str, *, validate_question_scope: bool = True) -> AnnotatedDocument:
    """
    Load an annotated document from a YAML file.

    Expected YAML structure:
    document:
      document_id: ...
      document_theme: ...
      document_to_annotate: ...
      rules: [...]
      questions: [...]
      evaluated_answers: [...]  # Optional
    """
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "document" not in data:
        raise ValueError(f"YAML file {yaml_path} does not contain 'document' key")

    doc_data = data["document"]
    if not isinstance(doc_data, dict):
        raise ValueError(f"YAML file {yaml_path} does not contain a document mapping")

    # Validate raw annotations before any normalization so legacy/invalid refs fail loudly.
    doc_id = doc_data.get("document_id", yaml_path)
    validate_annotations(
        doc_data.get("document_to_annotate", ""),
        source_label=f"{doc_id}/document_to_annotate",
    )
    validate_annotations(
        doc_data.get("fictionalized_annotated_template_document", ""),
        source_label=f"{doc_id}/fictionalized_annotated_template_document",
    )
    for q_data in doc_data.get("questions", []):
        qid = q_data.get("question_id", "?")
        q_label = f"{doc_id}/question/{qid}"
        validate_annotations(
            q_data.get("question", ""),
            source_label=f"{q_label}/question",
        )
        for step_index, reasoning_step in enumerate(q_data.get("reasoning_chain", []) or [], start=1):
            validate_annotations(
                str(reasoning_step),
                source_label=f"{q_label}/reasoning_chain[{step_index}]",
            )
        raw_answer = q_data.get("answer", "")
        if isinstance(raw_answer, list) and len(raw_answer) == 1:
            answer_text = str(raw_answer[0])
        elif isinstance(raw_answer, bool):
            answer_text = ""  # Yes/No answers have no entity refs
        else:
            answer_text = str(raw_answer)
        validate_annotations(
            answer_text,
            source_label=f"{q_label}/answer",
        )

    if validate_question_scope:
        validate_question_and_answer_entity_scope(
            doc_data.get("document_to_annotate", ""),
            doc_data.get("questions", []),
            source_label=str(doc_id),
        )

    doc_data = normalize_document_taxonomy(doc_data)

    # Parse questions
    questions = []
    for q_data in doc_data.get("questions", []):
        raw_answer = q_data.get("answer", "")
        if raw_answer is True:
            answer_str = "Yes"
        elif raw_answer is False:
            answer_str = "No"
        elif isinstance(raw_answer, list) and len(raw_answer) == 1 and isinstance(raw_answer[0], str):
            # Preserve inner string for list-wrapped composite answers (e.g. ['Boston Marathon; "place_1.city event_1.type"'])
            answer_str = raw_answer[0]
        else:
            answer_str = str(raw_answer)
        raw_answer_type = q_data.get("answer_type")
        if raw_answer_type is None:
            invariant_flag = q_data.get("is_answer_invariant")
            if invariant_flag is True:
                raw_answer_type = "invariant"
            elif invariant_flag is False:
                raw_answer_type = "variant"
        raw_answer_overrides = q_data.get("accepted_answer_overrides") or []
        if isinstance(raw_answer_overrides, str):
            accepted_answer_overrides = [raw_answer_overrides.strip()] if raw_answer_overrides.strip() else []
        else:
            accepted_answer_overrides = [str(item).strip() for item in raw_answer_overrides if str(item).strip()]
        question = Question(
            question_id=q_data["question_id"],
            question=q_data["question"],
            answer=answer_str,
            question_type=q_data.get("question_type"),
            answer_type=raw_answer_type,
            reasoning_chain=[str(step).strip() for step in (q_data.get("reasoning_chain") or []) if str(step).strip()],
            accepted_answer_overrides=accepted_answer_overrides,
        )
        questions.append(question)

    # Parse evaluated_answers if present
    evaluated_answers = None
    if "evaluated_answers" in doc_data:
        evaluated_answers = doc_data["evaluated_answers"]
    runtime_rule_expressions = normalize_rule_expressions(doc_data.get("rules", [])) or []
    implicit_rules_data = normalize_implicit_rules_for_storage(doc_data.get("implicit_rules", [])) or []

    # Create document
    doc = AnnotatedDocument(
        document_id=doc_data["document_id"],
        document_theme=doc_data.get("document_theme", ""),
        original_document=doc_data.get("original_document", ""),
        document_to_annotate=doc_data.get("document_to_annotate", ""),
        fictionalized_annotated_template_document=doc_data.get("fictionalized_annotated_template_document", ""),
        questions=questions,
        rules=runtime_rule_expressions,
        implicit_rules=[ImplicitRule(**rule_data) for rule_data in implicit_rules_data],
        implicit_rule_exclusions=normalize_implicit_rule_exclusions(doc_data.get("implicit_rule_exclusions")),
        evaluated_answers=evaluated_answers,
    )

    return doc


def fictional_document_to_yaml_dict(
    generated_doc: FictionalDocument,
    question_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the dict used for YAML serialization of a generated fictional document.
    question_entries should come from AnswerEvaluator.build_question_entries_with_answers.
    """
    return {
        "document_id": generated_doc.document_id,
        "document_theme": generated_doc.document_theme,
        "generated_document": generated_doc.generated_document,
        "questions": question_entries,
        "entities_used": {
            "persons": {k: v.model_dump() for k, v in generated_doc.entities_used.persons.items()},
            "places": {k: v.model_dump() for k, v in generated_doc.entities_used.places.items()},
            "events": {k: v.model_dump() for k, v in generated_doc.entities_used.events.items()},
            "organizations": {k: v.model_dump() for k, v in generated_doc.entities_used.organizations.items()},
            "awards": {k: v.model_dump() for k, v in generated_doc.entities_used.awards.items()},
            "legals": {k: v.model_dump() for k, v in generated_doc.entities_used.legals.items()},
            "products": {k: v.model_dump() for k, v in generated_doc.entities_used.products.items()},
            "temporals": {k: v.model_dump() for k, v in generated_doc.entities_used.temporals.items()},
            "numbers": {k: v.model_dump() for k, v in generated_doc.entities_used.numbers.items()},
        },
    }
