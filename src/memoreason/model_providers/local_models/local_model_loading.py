"""Load transformers and llama.cpp models for local MemoReason evaluation."""

from __future__ import annotations

from typing import Any

from .model_config import LocalModelConfig
from .runtime_dependencies import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    LLAMA_CPP_AVAILABLE,
    Llama,
    MISTRAL_COMMON_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
    hf_pipeline,
    torch,
)


class LocalModelLoadingMixin:
    def _load_transformers_model(self, config: LocalModelConfig):
        """Load a model using transformers."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install the LLM dependencies with: uv sync --extra llm")
        if self._is_magistral_model(config.name) and not MISTRAL_COMMON_AVAILABLE:
            raise ImportError(
                "Magistral models require mistral-common for correct tokenization/chat templating. "
                "Install the LLM dependencies with: uv sync --extra llm"
            )
        if self._is_gpt_oss_120b_model(config.name) and config.device == "cuda" and not self._has_kernels_package():
            raise ImportError(
                "gpt-oss-120b requires kernels>=0.12.0 on CUDA hosts so Transformers keeps the "
                "published MXFP4 weights. Without kernels it falls back to bf16, which exceeds the "
                "memory budget of a single H100. Install with: uv sync --extra llm or "
                "pip install -U 'kernels>=0.12.0'"
            )

        print(f"   Loading {config.name} from {config.model_path}...")

        local_files_only = bool(config.local_files_only)
        trust_remote_code = self._trust_remote_code(config.name)
        tokenizer_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if self._is_gemma4_model(config.name):
            if AutoProcessor is None or AutoModelForImageTextToText is None:
                raise ImportError(
                    "Gemma 4 models require AutoProcessor and AutoModelForImageTextToText support in Transformers."
                )
            tokenizer = AutoProcessor.from_pretrained(
                config.model_path,
                **tokenizer_kwargs,
            )
        else:
            if self._is_magistral_model(config.name):
                tokenizer_kwargs["tokenizer_type"] = "mistral"
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                **tokenizer_kwargs,
            )
        torch_dtype: Any = torch.float32
        device_map: str | None = "auto" if config.device == "cuda" else None
        if config.device == "cuda":
            if self._is_gpt_oss_model(config.name):
                torch_dtype = "auto"
            else:
                torch_dtype = torch.bfloat16
            device_map = "auto"
        elif config.device == "mps":
            torch_dtype = torch.float16
        elif config.device == "cpu":
            torch_dtype = torch.float32

        model_loader_cls = AutoModelForCausalLM
        if self._is_gemma4_model(config.name) or self._is_magistral_model(config.name):
            if AutoModelForImageTextToText is None:
                raise ImportError(
                    "This model requires transformers.AutoModelForImageTextToText, but the active "
                    "Transformers build does not provide it."
                )
            model_loader_cls = AutoModelForImageTextToText

        model = model_loader_cls.from_pretrained(
            config.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if config.device == "mps":
            model = model.to("mps")
        elif config.device == "cpu":
            model = model.to("cpu")

        # Set pad token if not present
        if self._is_gemma4_model(config.name):
            base_tokenizer = getattr(tokenizer, "tokenizer", None)
            if base_tokenizer is not None and getattr(base_tokenizer, "pad_token", None) is None:
                base_tokenizer.pad_token = base_tokenizer.eos_token
        else:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        self.loaded_models[config.name] = model
        self.loaded_tokenizers[config.name] = tokenizer
        if self._is_gemma3_model(config.name):
            self.loaded_pipelines[config.name] = hf_pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

        print(f"   Loaded {config.name}")

    def _load_llama_cpp_model(self, config: LocalModelConfig):
        """Load a model using llama.cpp."""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not installed. Install the LLM dependencies with: uv sync --extra llm")

        print(f"   Loading {config.name} from {config.model_path}...")

        model = Llama(
            model_path=config.model_path,
            n_ctx=config.context_window,
            n_threads=None,  # Auto-detect
            verbose=False,
        )

        self.loaded_models[config.name] = model
        print(f"   Loaded {config.name}")

    def load_model(self, config: LocalModelConfig):
        """Load a local LLM model."""
        if config.name in self.loaded_models:
            return  # Already loaded

        if config.backend == "transformers":
            self._load_transformers_model(config)
        elif config.backend == "llama-cpp":
            self._load_llama_cpp_model(config)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")
