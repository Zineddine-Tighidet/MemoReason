"""Local model runtime used to reproduce the paper's open-weight model answers."""

from typing import Any

from .local_model_generation import LocalModelGenerationMixin
from .local_model_loading import LocalModelLoadingMixin
from .local_model_prompting import LocalModelPromptingMixin


class LocalModelInferenceClient(LocalModelPromptingMixin, LocalModelLoadingMixin, LocalModelGenerationMixin):
    """Run local language models with the frozen MemoReason generation settings."""

    def __init__(self):
        self.loaded_models: dict[str, Any] = {}
        self.loaded_tokenizers: dict[str, Any] = {}
        self.loaded_pipelines: dict[str, Any] = {}
