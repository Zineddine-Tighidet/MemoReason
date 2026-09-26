"""Local-model configuration and execution for the paper's open-weight models."""

from .local_model_inference_client import LocalModelInferenceClient
from .model_config import LocalModelConfig
from .model_registry import MODEL_PATH_MAPPING

__all__ = ["LocalModelInferenceClient", "LocalModelConfig", "MODEL_PATH_MAPPING"]
