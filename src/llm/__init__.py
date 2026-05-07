"""Shared LLM entrypoints with explicit backend modules."""

from .anthropic_client import AnthropicTextGenerationClient
from .gemini_client import GeminiTextGenerationClient
from .groq_client import GroqTextGenerationClient
from .local_client import LocalTextGenerationClient
from .text_generation import TextGenerationRequest, TextGenerationResult, generate_text

__all__ = [
    "AnthropicTextGenerationClient",
    "GeminiTextGenerationClient",
    "GroqTextGenerationClient",
    "LocalTextGenerationClient",
    "TextGenerationRequest",
    "TextGenerationResult",
    "generate_text",
]
