"""Optional shared-cache paths for local models.

Remote GPT-OSS and Claude models are configured in the scientific model
registry and never appear here. If a shared path is unavailable, the local
client falls back to the Hugging Face model id or an explicit
``MEMOREASON_LOCAL_MODEL_PATH`` override.
"""

from __future__ import annotations

import os
from typing import Any


_MODEL_HUB = os.environ.get("MEMOREASON_MODEL_HUB", "models").rstrip("/")

MODEL_PATH_MAPPING: dict[str, dict[str, Any]] = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "path": f"{_MODEL_HUB}/meta-llama/Llama-3.1-8B-Instruct",
        "backend": "transformers",
        "device": "cuda",
        "seed": 24,
        "local_files_only": True,
    },
    "microsoft/Phi-4-mini-instruct": {
        "path": f"{_MODEL_HUB}/microsoft/Phi-4-mini-instruct",
        "backend": "transformers",
        "device": "cuda",
        "seed": 24,
        "local_files_only": True,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "path": f"{_MODEL_HUB}/mistralai/Mistral-7B-Instruct-v0.3",
        "backend": "transformers",
        "device": "cuda",
        "seed": 24,
        "local_files_only": True,
    },
    "allenai/Olmo-3-7B-Think": {
        "path": f"{_MODEL_HUB}/allenai/Olmo-3-7B-Think",
        "backend": "transformers",
        "device": "cuda",
        "seed": 24,
        "local_files_only": True,
    },
    "allenai/Olmo-3-7B-Instruct": {
        "path": f"{_MODEL_HUB}/allenai/Olmo-3-7B-Instruct",
        "backend": "transformers",
        "device": "cuda",
        "seed": 24,
        "local_files_only": True,
    },
    "google/gemma-4-26B-A4B-it": {
        "path": f"{_MODEL_HUB}/google/gemma-4-26B-A4B-it",
        "backend": "transformers",
        "device": "cuda",
        "seed": 24,
        "local_files_only": True,
    },
    "Qwen/Qwen3.5-27B": {
        "path": f"{_MODEL_HUB}/Qwen/Qwen3.5-27B",
        "backend": "transformers",
        "device": "cuda",
        "seed": 24,
        "local_files_only": True,
    },
    "Qwen/Qwen3.5-35B-A3B": {
        "path": f"{_MODEL_HUB}/Qwen/Qwen3.5-35B-A3B",
        "backend": "transformers",
        "device": "cuda",
        "seed": 24,
        "local_files_only": True,
    },
}


__all__ = ["MODEL_PATH_MAPPING"]
