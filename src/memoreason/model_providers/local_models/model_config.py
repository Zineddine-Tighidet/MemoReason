"""Resolve local model paths, devices, and reproducibility settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .model_registry import MODEL_PATH_MAPPING
from .runtime_dependencies import TRANSFORMERS_AVAILABLE
from . import runtime_dependencies as _dependencies

# ``torch`` is intentionally resolved through the optional-dependency module so
# importing configuration still works on hosts without transformers.
if TRANSFORMERS_AVAILABLE:
    torch = _dependencies.torch


def _default_local_files_only(model_path: str) -> bool:
    """Prefer local-only loading for concrete paths and allow downloads for HF ids."""
    if Path(model_path).exists():
        return True
    return False


def _cached_hf_snapshot_path(model_id: str) -> str | None:
    """Resolve a downloaded Hugging Face snapshot path for a model id when available."""
    if not model_id or "/" not in model_id:
        return None
    if Path(model_id).exists():
        return str(Path(model_id))

    repo_dir = f"models--{model_id.replace('/', '--')}"
    candidate_roots: list[Path] = []
    for env_name in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        value = os.environ.get(env_name)
        if not value:
            continue
        root = Path(value)
        candidate_roots.append(root)
        candidate_roots.append(root / "hub")
    home_cache = Path.home() / ".cache" / "huggingface"
    candidate_roots.append(home_cache)
    candidate_roots.append(home_cache / "hub")

    seen: set[Path] = set()
    for root in candidate_roots:
        if root in seen:
            continue
        seen.add(root)
        snapshot_root = root / repo_dir / "snapshots"
        if not snapshot_root.exists():
            continue
        snapshots = sorted(
            (path for path in snapshot_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0])
    return None


def _snapshot_has_weights(snapshot_path: Path) -> bool:
    """Return whether a snapshot contains at least one model weight artifact."""
    explicit_files = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "pytorch_model.bin",
    )
    if any((snapshot_path / name).exists() for name in explicit_files):
        return True
    return any(snapshot_path.glob(pattern) for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"))


def _snapshot_has_tokenizer(snapshot_path: Path) -> bool:
    """Return whether a snapshot contains at least one tokenizer artifact."""
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "tokenizer.model.v3",
        "tokenizer.mm.model.v3",
        "tekken.json",
        "merges.txt",
        "vocab.json",
    )
    return any((snapshot_path / name).exists() for name in tokenizer_files)


def _snapshot_is_usable(model_id: str, snapshot_path: str) -> bool:
    """
    Guard against partial HF snapshots.

    Some shared caches contain only README/config metadata. Only redirect to
    a cached snapshot when it has both weights and tokenizer assets.
    """
    path = Path(snapshot_path)
    if not path.exists() or not path.is_dir():
        return False
    return _snapshot_has_weights(path) and _snapshot_has_tokenizer(path)


def _best_auto_device() -> str:
    """Pick a sensible default device for the current host."""
    if not TRANSFORMERS_AVAILABLE:
        return "auto"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LocalModelConfig:
    """Configuration for a local LLM."""

    name: str
    model_path: str | None = (
        None  # HuggingFace model ID or path to GGUF file (auto-filled from MODEL_PATH_MAPPING if None)
    )
    backend: str = "transformers"  # "transformers" or "llama-cpp"
    device: str = "auto"  # "cuda", "cpu", or "auto"
    context_window: int = 4096
    temperature: float = 0.0
    max_tokens: int = 5000
    seed: int | None = None  # Random seed for reproducibility
    local_files_only: bool | None = None

    def __post_init__(self):
        """Auto-fill model_path from MODEL_PATH_MAPPING if not provided."""
        model_path_override = os.environ.get("MEMOREASON_LOCAL_MODEL_PATH")
        override_model_name = os.environ.get("MEMOREASON_LOCAL_MODEL_NAME")
        if model_path_override and override_model_name == self.name:
            override_path = Path(model_path_override).expanduser().resolve()
            if not override_path.exists():
                raise FileNotFoundError(f"MEMOREASON_LOCAL_MODEL_PATH does not exist for {self.name}: {override_path}")
            self.model_path = str(override_path)
            self.local_files_only = True
        if self.model_path is None:
            if self.name not in MODEL_PATH_MAPPING:
                # Fall back to treating the model name as a Hugging Face model id.
                self.model_path = self.name
            else:
                mapping = MODEL_PATH_MAPPING[self.name]
                preferred_path = mapping.get("path")
                if preferred_path and Path(preferred_path).exists():
                    self.model_path = preferred_path
                    # Override backend/device/seed if the concrete local path exists.
                    if self.backend == "transformers" and "backend" in mapping:
                        self.backend = mapping["backend"]
                    if self.device == "auto" and "device" in mapping:
                        self.device = mapping["device"]
                    if self.seed is None and "seed" in mapping:
                        self.seed = mapping["seed"]
                    if self.local_files_only is None and "local_files_only" in mapping:
                        self.local_files_only = bool(mapping["local_files_only"])
                else:
                    # Missing shared path: fall back to the HF id and local host defaults.
                    self.model_path = self.name
        cached_snapshot = _cached_hf_snapshot_path(str(self.model_path))
        if cached_snapshot is not None and _snapshot_is_usable(str(self.model_path), cached_snapshot):
            self.model_path = cached_snapshot
            self.local_files_only = True
        if self.local_files_only is None:
            self.local_files_only = _default_local_files_only(str(self.model_path))
        if self.device == "auto":
            self.device = _best_auto_device()
