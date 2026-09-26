from pathlib import Path

import yaml

from memoreason.benchmark_definition.annotation_runtime import load_entity_pool
from memoreason.benchmark_definition.document_schema import AnnotatedDocument
from memoreason.factual_to_fictional_dataset.fictional_entity_pool_generation.pool_normalization import (
    _extract_mapping_payload,
)
from web.services import entity_pool_service


def _write_pool(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_load_pool_refreshes_existing_runtime_copy_from_gcs(tmp_path, monkeypatch) -> None:
    runtime_root = tmp_path / "runtime_pools"
    runtime_path = runtime_root / "award_winners" / "awards_01_entity_pool.yaml"

    stale_pool = {
        "persons": [{"full_name": "Local Stale Winner"}],
        "media_orgs": [{"name": "Old Broadcaster"}],
    }
    fresh_pool = {
        "persons": [{"full_name": "Remote Fresh Winner"}],
        "media_orgs": [{"name": "New Broadcaster"}],
    }
    _write_pool(runtime_path, stale_pool)

    def _fake_restore(path: Path) -> bool:
        if path != runtime_path:
            return False
        _write_pool(path, fresh_pool)
        return True

    monkeypatch.setattr(entity_pool_service, "RUNTIME_POOL_ROOT", runtime_root)
    monkeypatch.setattr(entity_pool_service, "restore_work_file_from_gcs", _fake_restore)

    document = AnnotatedDocument(
        document_id="awards_01",
        document_theme="award_winners",
        original_document="",
        document_to_annotate="",
        rules=[],
        questions=[],
    )

    loaded_pool, source, loaded_path = entity_pool_service.get_or_generate_pool(
        document,
        required_entities={},
        seed=23,
        theme_id="award_winners",
    )

    assert source == "document_pool"
    assert loaded_path == runtime_path
    assert loaded_pool["persons"] == [{"full_name": "Remote Fresh Winner"}]
    assert loaded_pool["media_orgs"] == [{"name": "New Broadcaster"}]


def test_get_or_generate_pool_uses_explicit_theme_id_when_document_theme_is_blank(monkeypatch) -> None:
    monkeypatch.setenv("ANNOTATION_ALLOW_POOL_GENERATION", "true")
    calls: dict[str, object] = {}

    def _fake_generate(document, required_entities, *, theme, config):
        calls["document_id"] = document.document_id
        calls["required_entities"] = required_entities
        calls["theme"] = theme
        calls["config"] = config
        return {"persons": []}

    monkeypatch.setattr(entity_pool_service, "_load_pool_for_document", lambda doc, theme_id=None: (None, None))
    monkeypatch.setattr(entity_pool_service, "generate_fictional_entity_replacement_pool", _fake_generate)

    document = AnnotatedDocument(
        document_id="bio_01",
        document_theme="",
        original_document="",
        document_to_annotate="",
        rules=[],
        questions=[],
    )

    pool, source, loaded_path = entity_pool_service.get_or_generate_pool(
        document,
        required_entities={"person": [("person_1", ["full_name"])]},
        seed=23,
        theme_id="biographies_of_famous_personalities",
    )

    assert pool == {"persons": []}
    assert source == "auto_generated"
    assert loaded_path is None
    assert calls["document_id"] == "bio_01"
    assert calls["required_entities"] == {"person": [("person_1", ["full_name"])]}
    assert calls["theme"] == "biographies_of_famous_personalities"


def test_load_entity_pool_preserves_type_only_event_reference_variants(tmp_path: Path) -> None:
    pool_path = tmp_path / "type_only_events.yaml"
    _write_pool(
        pool_path,
        {
            "events": {
                "event_9": {
                    "required_attributes": ["type"],
                    "count": 2,
                    "variants": [
                        {"type": "quadreniad"},
                        {"type": "vexiliad"},
                    ],
                }
            }
        },
    )

    loaded = load_entity_pool(str(pool_path))

    assert loaded["_reference_pools"]["events"]["event_9"]["count"] == 2
    assert loaded["_reference_pools"]["events"]["event_9"]["variants"] == [
        {"type": "quadreniad"},
        {"type": "vexiliad"},
    ]


def test_extract_mapping_payload_skips_leading_chatter_before_bucket_block() -> None:
    raw = """
Wait, these look like random letters. Let me rewrite them cleanly.

ngos:
  ngo_1:
    required_attributes:
    - name
    count: 1
    variants:
    - name: Solmere Aid Collective
""".strip()

    parsed = _extract_mapping_payload(raw)

    assert parsed["ngos"]["ngo_1"]["variants"][0]["name"] == "Solmere Aid Collective"
