"""Shared LLM entrypoints with explicit backend modules."""

from .anthropic_client import AnthropicTextGenerationClient
from .groq_client import GroqTextGenerationClient
from .local_text_generation_client import LocalTextGenerationClient
from .text_generation import TextGenerationRequest, TextGenerationResult, generate_text

__all__ = [
    "AnthropicTextGenerationClient",
    "GroqTextGenerationClient",
    "LocalTextGenerationClient",
    "TextGenerationRequest",
    "TextGenerationResult",
    "generate_text",
]
