"""Exercise the local annotation UI without cloud state or provider requests."""

import hashlib
import importlib
import sys
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def local_ui(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    source = tmp_path / "templates" / "companies_and_organizations"
    source.mkdir(parents=True)
    path = source / "company_01.yaml"
    payload = {
        "document": {
            "document_id": "company_01",
            "document_theme": "companies_and_organizations",
            "original_document": "The company opened in 2001.",
            "document_to_annotate": "The company opened in [2001; temporal_1.year].",
            "questions": [],
            "rules": [],
        }
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setenv("ANNOTATION_SOURCE_DIR", str(source.parent))
    monkeypatch.setenv("ANNOTATION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("DEFAULT_ADMIN_USERNAME", "local-admin")
    monkeypatch.setenv("DEFAULT_ADMIN_PASSWORD", "local-test-password-123")
    monkeypatch.setenv("ANNOTATION_FEEDBACK_RESOLVERS", "local-admin, second-admin")
    monkeypatch.setenv("ANNOTATION_REFERENCE_USERNAME", "reference")
    # A stray configured bucket must not opt a local reviewer into cloud sync.
    monkeypatch.setenv("ANNOTATION_STATE_GCS_BUCKET", "unused-example-bucket")
    monkeypatch.delenv("ANNOTATION_ENABLE_REMOTE_STATE", raising=False)
    monkeypatch.delenv("APP_CANONICAL_HOST", raising=False)
    monkeypatch.setenv("ANNOTATION_MAINTENANCE_MODE", "off")
    monkeypatch.delenv("ANNOTATION_ALLOW_POOL_GENERATION", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)

    # Modules resolve configuration on import. Keep each test's storage isolated.
    old_modules = {
        name: module for name, module in sys.modules.items()
        if name == "web" or name.startswith("web.")
    }
    for name in old_modules:
        sys.modules.pop(name, None)
    app_module = importlib.import_module("web.app")
    persistence = importlib.import_module("web.services.persistence")

    def no_cloud_client():
        pytest.fail("A local interface action attempted remote cloud access")

    monkeypatch.setattr(persistence, "_storage_client", no_cloud_client)
    try:
        with TestClient(app_module.app) as client:
            yield client, path, tmp_path / "state"
    finally:
        importlib.import_module("web.services.db").close_db()
        for name in tuple(sys.modules):
            if name == "web" or name.startswith("web."):
                sys.modules.pop(name, None)
        sys.modules.update(old_modules)


def _login(client):
    response = client.post("/api/v1/auth/login", json={
        "username": "local-admin", "password": "local-test-password-123",
    })
    assert response.status_code == 200, response.text
    assert response.json()["user"]["role"] == "power_user"


def test_local_login_dashboard_and_source_inventory(local_ui):
    client, source_path, state_dir = local_ui
    response = client.get("/login")
    assert response.status_code == 200
    assert "fonts.googleapis.com" not in response.text
    assert client.get("/api/v1/themes").status_code == 401
    _login(client)
    assert client.get("/").status_code == 200
    themes = client.get("/api/v1/themes").json()
    assert sum(theme["total_docs"] for theme in themes) == 1
    assert client.get("/editor/companies_and_organizations/company_01").status_code == 200
    assert (state_dir / "annotation.db").is_file()


def test_edit_creates_working_copy_and_preserves_release_source(local_ui):
    client, source_path, state_dir = local_ui
    before_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    _login(client)
    url = "/api/v1/documents/companies_and_organizations/company_01"
    response = client.get(url)
    assert response.status_code == 200, response.text
    document = response.json()
    document["document_to_annotate"] = "The company opened in [2001; temporal_1.year]. It operates locally."
    saved = client.put(url, json=document)
    assert saved.status_code == 200, saved.text
    assert saved.json()["status"] == "saved"
    assert "It operates locally." in client.get(url).json()["document_to_annotate"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before_hash
    working_copy = state_dir / "annotation_workspace/local-admin/companies_and_organizations/company_01.yaml"
    assert working_copy.is_file()


def test_reference_uses_curated_source_without_named_account(local_ui):
    client, source_path, state_dir = local_ui
    _login(client)
    response = client.get(
        "/api/v1/documents/companies_and_organizations/company_01/reference-bootstrap"
    )
    assert response.status_code == 200, response.text
    assert response.json()["reference_mode"] is True
    assert response.json()["document"]["document_id"] == "company_01"
    assert not (state_dir / "annotation_workspace/reference").exists()
    from web.services import workflow_service
    assert workflow_service.resolver_requires_reviewer_acceptance("LOCAL-ADMIN")
    assert not workflow_service.resolver_requires_reviewer_acceptance("unknown-user")


def test_missing_entity_pool_does_not_start_provider_generation(local_ui, monkeypatch):
    _client, _source_path, _state_dir = local_ui
    from web.services import entity_pool_service

    monkeypatch.setattr(entity_pool_service, "_load_pool_for_document", lambda *_args, **_kwargs: (None, None))

    def no_provider(*_args, **_kwargs):
        pytest.fail("A missing pool triggered a provider request without opt-in")

    monkeypatch.setattr(entity_pool_service, "generate_fictional_entity_replacement_pool", no_provider)
    with pytest.raises(ValueError, match="ANNOTATION_ALLOW_POOL_GENERATION"):
        entity_pool_service.get_or_generate_pool(None, {}, 23)


def test_shipped_template_inventory_is_accessible_and_unchanged(local_ui, monkeypatch):
    """Check every shipped template through the actual authenticated document API."""
    source_root = Path(__file__).resolve().parents[1] / "data/HUMAN_ANNOTATED_TEMPLATES"
    paths = sorted(source_root.glob("*/*.yaml"))
    if not paths:
        pytest.skip("The optional release-data bundle is not installed")
    assert len(paths) == 100
    from web.services import yaml_service

    monkeypatch.setattr(yaml_service, "SOURCE_DIR", source_root)
    monkeypatch.setattr(yaml_service, "THEMES", sorted({path.parent.name for path in paths}))
    client, _source_path, _state_dir = local_ui
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    _login(client)
    themes = client.get("/api/v1/themes").json()
    assert sum(theme["total_docs"] for theme in themes) == len(paths)
    for path in paths:
        response = client.get(f"/api/v1/documents/{path.parent.name}/{path.stem}")
        assert response.status_code == 200, (path.name, response.text)
        assert response.json()["document_id"] == path.stem
        assert len(response.json()["questions"]) == 12
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path]


def test_review_export_preserves_source_templates(local_ui, monkeypatch):
    _client, source_path, state_dir = local_ui
    from web.services import review_campaign_service

    monkeypatch.setattr(review_campaign_service, "HUMAN_ANNOTATED_TEMPLATES_DIR", source_path.parent.parent)
    snapshot_path = state_dir / "review-snapshot.yaml"
    snapshot_path.write_text(yaml.safe_dump({"document": {"rules": ["2001 < 2002"]}}), encoding="utf-8")
    monkeypatch.setattr(review_campaign_service, "list_completed_review_artifacts", lambda *_args, **_kwargs: [{
        "theme": source_path.parent.name,
        "doc_id": source_path.stem,
        "snapshot_path": snapshot_path,
    }])
    source_bytes = source_path.read_bytes()
    assert review_campaign_service.export_reviewed_template_fields("rules")["updated"] == 1
    exported = state_dir / "annotation_workspace/_exports" / source_path.parent.name / source_path.name
    assert yaml.safe_load(exported.read_text())["document"]["rules"] == ["2001 < 2002"]
    assert source_path.read_bytes() == source_bytes
    with pytest.raises(ValueError, match="preserve the frozen"):
        review_campaign_service.export_reviewed_template_fields("rules", output_dir=source_path.parent.parent)
