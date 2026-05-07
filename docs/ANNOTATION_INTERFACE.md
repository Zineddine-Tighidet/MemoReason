# Annotation Interface

The local web UI is included so others can inspect or extend the MemoReason annotation workflow.

## Run Locally

```bash
uv sync --extra web
uv run uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The app stores local working state in `web/data/`, which is ignored by git and should not be published.

## Data Source

By default, the UI reads the anonymized templates in:

```text
data/HUMAN_ANNOTATED_TEMPLATES
```

To annotate a fresh YAML source folder:

```bash
ANNOTATION_SOURCE_DIR=/path/to/source_yaml \
uv run uvicorn web.app:app --host 127.0.0.1 --port 8000
```

The expected source format is the same YAML structure used by the released templates: document metadata, annotated text, entities, rules, and questions.

## Typical Workflow

1. Select a theme and document from the dashboard.
2. Annotate entities in the document text.
3. Add or review rule constraints.
4. Draft questions and answers.
5. Save locally, then export templates if you want to regenerate benchmark documents.