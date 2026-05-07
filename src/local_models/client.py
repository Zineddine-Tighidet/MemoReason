"""Minimal local Transformers backend for open-weight model evaluations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


MODEL_PATH_MAPPING: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class LocalModelConfig:
    """Configuration for one local/Hugging Face model run."""

    name: str
    model_path: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True

    @property
    def model_id(self) -> str:
        return str(self.model_path or self.name).strip()


class LocalLLMClient:
    """Run chat-style generation through ``transformers``.

    The public release intentionally does not encode machine-specific model
    paths. Pass a Hugging Face model id as ``LocalModelConfig.name`` or provide
    ``model_path`` explicitly for a local checkpoint.
    """

    def __init__(self, default_model: str | None = None) -> None:
        self.default_model = default_model
        self._loaded_model_id: str | None = None
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self, config: LocalModelConfig) -> tuple[Any, Any]:
        model_id = config.model_id
        if self._loaded_model_id == model_id and self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Local model evaluation requires: uv sync --extra llm") from exc

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=config.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=config.device_map,
            torch_dtype=getattr(torch, config.torch_dtype, config.torch_dtype)
            if config.torch_dtype != "auto"
            else "auto",
            trust_remote_code=config.trust_remote_code,
        )
        model.eval()

        self._loaded_model_id = model_id
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    @staticmethod
    def _format_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"{system_prompt.strip()}\n\n{user_prompt.strip()}\n"

    def generate(
        self,
        prompt: str | None = None,
        *,
        config: LocalModelConfig | None = None,
        system_prompt: str = "",
        user_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | str:
        """Generate a response.

        The keyword form returns the normalized payload used by the benchmark.
        The positional ``prompt`` form returns text for older helper code.
        """
        legacy_text_mode = config is None and user_prompt is None
        if config is None:
            model_name = self.default_model
            if not model_name:
                raise ValueError("Local generation requires a model name or LocalModelConfig.")
            config = LocalModelConfig(
                name=model_name,
                max_tokens=max_tokens or 512,
                temperature=0.0 if temperature is None else float(temperature),
            )
            user_prompt = prompt or ""

        tokenizer, model = self._load(config)
        if config.seed is not None:
            try:
                import torch

                torch.manual_seed(int(config.seed))
            except Exception:
                pass

        rendered_prompt = self._format_prompt(tokenizer, system_prompt, user_prompt or "")
        encoded = tokenizer(rendered_prompt, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generation_kwargs = {
            "max_new_tokens": int(max_tokens or config.max_tokens),
            "do_sample": float(temperature if temperature is not None else config.temperature) > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        effective_temperature = float(temperature if temperature is not None else config.temperature)
        if effective_temperature > 0:
            generation_kwargs["temperature"] = effective_temperature

        output_ids = model.generate(**encoded, **generation_kwargs)
        generated_ids = output_ids[0][encoded["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if legacy_text_mode:
            return text
        return {
            "content": text,
            "reasoning_content": "",
            "raw_api_response_repr": json.dumps(
                {
                    "model": config.model_id,
                    "max_new_tokens": generation_kwargs["max_new_tokens"],
                    "temperature": effective_temperature,
                },
                sort_keys=True,
            ),
        }


__all__ = ["LocalLLMClient", "LocalModelConfig", "MODEL_PATH_MAPPING"]
