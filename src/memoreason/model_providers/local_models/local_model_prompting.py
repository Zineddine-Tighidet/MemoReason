"""Build model-specific chat messages and serialized prompts."""

from __future__ import annotations

import importlib.util
from typing import Any


class LocalModelPromptingMixin:
    @staticmethod
    def _is_qwen35_model(model_name: str) -> bool:
        return "qwen3.5" in str(model_name).strip().lower()

    @staticmethod
    def _is_gpt_oss_model(model_name: str) -> bool:
        return str(model_name).strip().lower().startswith("gpt-oss-")

    @staticmethod
    def _is_gpt_oss_120b_model(model_name: str) -> bool:
        return str(model_name).strip().lower() == "gpt-oss-120b"

    @staticmethod
    def _is_gemma3_model(model_name: str) -> bool:
        return str(model_name).strip().lower().startswith("gemma-3-")

    @staticmethod
    def _is_gemma4_model(model_name: str) -> bool:
        return str(model_name).strip().lower().startswith("google/gemma-4-")

    @staticmethod
    def _is_magistral_model(model_name: str) -> bool:
        return "magistral" in str(model_name).strip().lower()

    @staticmethod
    def _is_openreasoning_nemotron_model(model_name: str) -> bool:
        lowered = str(model_name).strip().lower()
        return lowered.startswith("nvidia/openreasoning-nemotron-") or lowered.startswith("openreasoning-nemotron-")

    @staticmethod
    def _is_phi4_flash_reasoning_model(model_name: str) -> bool:
        return str(model_name).strip().lower() == "microsoft/phi-4-mini-flash-reasoning"

    @staticmethod
    def _trust_remote_code(model_name: str) -> bool:
        """Use the Transformers-native implementation for Phi-4 Mini Instruct."""
        return str(model_name).strip().lower() != "microsoft/phi-4-mini-instruct"

    @staticmethod
    def _qwen35_enable_thinking(model_name: str) -> bool:
        """Disable Qwen3.5 thinking mode for direct-answer benchmark evaluation."""
        return False

    @staticmethod
    def _gpt_oss_reasoning_effort(model_name: str) -> str:
        """Keep GPT-OSS on the lightest supported reasoning setting for direct-answer QA."""
        return "low"

    @staticmethod
    def _has_kernels_package() -> bool:
        return importlib.util.find_spec("kernels") is not None

    def _build_chat_messages(self, system_prompt: str, user_prompt: str, model_name: str) -> list[dict[str, Any]]:
        """Build provider-appropriate chat messages before applying the tokenizer template."""
        if self._is_gemma3_model(model_name):
            messages = []
            if system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }
            )
            return messages

        if self._is_phi4_flash_reasoning_model(model_name):
            merged_prompt = user_prompt or ""
            if system_prompt:
                merged_prompt = f"{system_prompt}\n\n{merged_prompt}" if merged_prompt else system_prompt
            return [{"role": "user", "content": merged_prompt}]

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _format_chat_prompt(self, system_prompt: str, user_prompt: str, model_name: str, tokenizer=None) -> str:
        """
        Formatte le prompt en respectant l'alternance stricte user/assistant.
        """
        if tokenizer is not None:
            # Injection forcée du template si manquant pour Mistral
            if "mistral" in model_name.lower() and (
                not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None
            ):
                tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}"

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                try:
                    messages = self._build_chat_messages(system_prompt, user_prompt, model_name)

                    # Keep the tokenizer-provided Qwen3.5 chat template verbatim.
                    if self._is_qwen35_model(model_name):
                        template_kwargs = {
                            "tokenize": False,
                            "add_generation_prompt": True,
                        }
                        if not self._qwen35_enable_thinking(model_name):
                            template_kwargs["enable_thinking"] = False
                        return tokenizer.apply_chat_template(messages, **template_kwargs)

                    if self._is_gemma4_model(model_name):
                        return tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )

                    if self._is_gpt_oss_model(model_name):
                        return tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            reasoning_effort=self._gpt_oss_reasoning_effort(model_name),
                        )

                    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception as e:
                    print(f"   Chat template failed for {model_name}, using fallback: {e}")

        # Fallback manuel si apply_chat_template échoue
        if "mistral" in model_name.lower():
            full_text = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            return f"<s>[INST] {full_text} [/INST]"

        return f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
