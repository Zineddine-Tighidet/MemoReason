"""Optional inference backends used by local MemoReason model runs."""

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline

    try:
        from transformers import AutoProcessor
    except ImportError:
        AutoProcessor = None
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoProcessor = None
    AutoModelForImageTextToText = None

try:
    import mistral_common  # noqa: F401

    MISTRAL_COMMON_AVAILABLE = True
except ImportError:
    MISTRAL_COMMON_AVAILABLE = False

try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Keep imports of the refactored client available on hosts without inference
# dependencies; the availability flags still guard every execution path.
if not TRANSFORMERS_AVAILABLE:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    hf_pipeline = None
    torch = None
if not LLAMA_CPP_AVAILABLE:
    Llama = None
