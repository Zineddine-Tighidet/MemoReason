"""Local-first settings shared by the annotation interface services."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.getenv("ANNOTATION_STATE_DIR", str(PROJECT_ROOT / "web" / "data"))).expanduser().resolve()
DB_PATH = STATE_DIR / "annotation.db"
WORK_DIR = STATE_DIR / "annotation_workspace"
REFERENCE_USERNAME = os.getenv("ANNOTATION_REFERENCE_USERNAME", "reference").strip() or "reference"


def remote_state_enabled() -> bool:
    """Remote persistence requires an explicit opt-in and a configured bucket."""
    return (
        os.getenv("ANNOTATION_ENABLE_REMOTE_STATE", "").strip().lower() in {"1", "true", "yes"}
        and bool(os.getenv("ANNOTATION_STATE_GCS_BUCKET", "").strip())
    )
