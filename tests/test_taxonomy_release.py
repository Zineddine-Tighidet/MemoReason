"""Check taxonomy editing against isolated state and the immutable release."""

import importlib
import sys

import pytest
import yaml


@pytest.fixture
def taxonomy_ui(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    state = tmp_path / "state"
    monkeypatch.setenv("ANNOTATION_STATE_DIR", str(state))
    monkeypatch.setenv("DEFAULT_ADMIN_USERNAME", "taxonomy-admin")
    monkeypatch.setenv("DEFAULT_ADMIN_PASSWORD", "taxonomy-test-password")
    monkeypatch.setenv("ANNOTATION_MAINTENANCE_MODE", "off")
    for variable in ("ANNOTATION_SOURCE_DIR", "ANNOTATION_ENABLE_REMOTE_STATE",
                     "APP_CANONICAL_HOST", "SESSION_SECRET_KEY"):
        monkeypatch.delenv(variable, raising=False)
    old_modules = {
        name: module for name, module in sys.modules.items()
        if name == "web" or name.startswith("web.")
    }
    for name in old_modules:
        sys.modules.pop(name, None)
    app_module = importlib.import_module("web.app")
    taxonomy = importlib.import_module("web.api.taxonomy")
    persistence = importlib.import_module("web.services.persistence")

    def no_cloud_client():
        pytest.fail("Local taxonomy editing attempted cloud access")

    monkeypatch.setattr(persistence, "_storage_client", no_cloud_client)
    try:
        with TestClient(app_module.app) as client:
            response = client.post("/api/v1/auth/login", json={
                "username": "taxonomy-admin", "password": "taxonomy-test-password",
            })
            assert response.status_code == 200, response.text
            yield client, taxonomy, state
    finally:
        importlib.import_module("web.services.db").close_db()
        for name in tuple(sys.modules):
            if name == "web" or name.startswith("web."):
                sys.modules.pop(name, None)
        sys.modules.update(old_modules)


def test_taxonomy_page_initializes_from_frozen_root(taxonomy_ui):
    client, taxonomy, state = taxonomy_ui
    baseline = taxonomy.TAXONOMY_SOURCE_FILE.read_bytes()
    assert client.get("/taxonomy").status_code == 200
    response = client.get("/api/v1/taxonomy/extended")
    assert response.status_code == 200, response.text
    assert len(response.json()["entities"]) == 14
    assert "full_name" in response.json()["entities"]["person"]["attributes"]
    assert (state / "TAXONOMY.md").read_bytes() == baseline
    assert yaml.safe_load((state / "entity_taxonomy_extended.yaml").read_text()) == response.json()
    assert taxonomy.TAXONOMY_SOURCE_FILE.read_bytes() == baseline


def test_taxonomy_edits_and_markdown_update_stay_in_local_state(taxonomy_ui):
    client, taxonomy, state = taxonomy_ui
    baseline = taxonomy.TAXONOMY_SOURCE_FILE.read_bytes()
    data = client.get("/api/v1/taxonomy/extended").json()
    data["entities"]["person"]["attributes"]["local_note"] = {
        "description": "A local annotation field", "examples": ["note"],
    }
    response = client.put("/api/v1/taxonomy/extended", json=data)
    assert response.status_code == 200, response.text
    assert client.get("/api/v1/taxonomy/extended").json() == data
    assert "local_note" in client.get("/api/v1/taxonomy").json()["person"]
    response = client.post("/api/v1/taxonomy/update-doc")
    assert response.status_code == 200, response.text
    assert "local_note" in (state / "TAXONOMY.md").read_text()
    assert client.get("/api/v1/taxonomy/extended").json() == data
    assert taxonomy.TAXONOMY_SOURCE_FILE.read_bytes() == baseline


def test_missing_working_markdown_preserves_existing_yaml(taxonomy_ui):
    client, taxonomy, state = taxonomy_ui
    baseline = taxonomy.TAXONOMY_SOURCE_FILE.read_bytes()
    existing = {"entities": {"local_entity": {
        "description": "Previously saved local taxonomy", "attributes": {},
    }}}
    taxonomy.save_extended_taxonomy(existing)
    assert not (state / "TAXONOMY.md").exists()
    response = client.get("/api/v1/taxonomy/extended")
    assert response.status_code == 200, response.text
    assert response.json() == existing
    assert client.post("/api/v1/taxonomy/update-doc").status_code == 200
    assert "local_entity" in (state / "TAXONOMY.md").read_text()
    assert taxonomy.TAXONOMY_SOURCE_FILE.read_bytes() == baseline


def test_legacy_readme_aliases_still_use_explicit_paths(taxonomy_ui, tmp_path, monkeypatch):
    client, taxonomy, state = taxonomy_ui
    baseline = taxonomy.TAXONOMY_SOURCE_FILE.read_bytes()
    legacy_doc = tmp_path / "legacy.md"
    legacy_doc.write_bytes(baseline)
    monkeypatch.setattr(taxonomy, "README_FILE", legacy_doc)
    monkeypatch.setattr(taxonomy, "TAXONOMY_FILE", tmp_path / "legacy.yaml")
    data = taxonomy.load_extended_taxonomy(sync_from_readme=True)
    data["entities"]["person"]["description"] = "Legacy local description"
    taxonomy.save_extended_taxonomy(data)
    assert taxonomy.load_extended_taxonomy(sync_from_readme=False) == data
    response = client.post("/api/v1/taxonomy/update-readme")
    assert response.status_code == 200, response.text
    assert "Legacy local description" in legacy_doc.read_text()
    assert not (state / "TAXONOMY.md").exists()
    assert taxonomy.TAXONOMY_SOURCE_FILE.read_bytes() == baseline


def test_state_cannot_alias_the_frozen_taxonomy(taxonomy_ui, monkeypatch):
    _client, taxonomy, _state = taxonomy_ui
    baseline = taxonomy.TAXONOMY_SOURCE_FILE.read_bytes()
    monkeypatch.setattr(taxonomy, "README_FILE", taxonomy.TAXONOMY_SOURCE_FILE)
    with pytest.raises(ValueError, match="frozen release taxonomy"):
        taxonomy.load_extended_taxonomy()
    assert taxonomy.TAXONOMY_SOURCE_FILE.read_bytes() == baseline
