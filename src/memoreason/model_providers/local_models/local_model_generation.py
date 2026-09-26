"""Run local model decoding with the frozen MemoReason generation settings."""

from __future__ import annotations

# Preserve historical optional annotations and the broad fallback around
# tokenizer chat-template formatting.
# ruff: noqa: E722, RUF013

import json
from typing import Any

from .model_config import LocalModelConfig
from .reasoning_response import extract_gpt_oss_harmony_content, extract_reasoning_from_content
from .runtime_dependencies import hf_pipeline, torch


class LocalModelGenerationMixin:
    def generate_transformers(
        self,
        config: LocalModelConfig,
        prompt_text: str = None,
        system_prompt: str = None,
        user_prompt: str = None,
    ) -> str:
        model = self.loaded_models[config.name]
        tokenizer = self.loaded_tokenizers[config.name]

        if self._is_gemma4_model(config.name):
            messages = self._build_chat_messages(system_prompt, user_prompt, config.name)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            generation_kwargs = {
                "max_new_tokens": config.max_tokens,
                "do_sample": config.temperature > 0,
            }
            if config.temperature > 0:
                generation_kwargs["temperature"] = config.temperature
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False).strip()

        if self._is_gemma3_model(config.name):
            messages = self._build_chat_messages(system_prompt, user_prompt, config.name)

            pipe = self.loaded_pipelines.get(config.name)
            if pipe is None:
                pipe = hf_pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
                self.loaded_pipelines[config.name] = pipe

            generation_kwargs = {
                "max_new_tokens": config.max_tokens,
                "return_full_text": False,
                "do_sample": config.temperature > 0,
            }
            if config.temperature > 0:
                generation_kwargs["temperature"] = config.temperature

            try:
                outputs = pipe(messages, **generation_kwargs)
            except Exception:
                # Google docs also show a batched chat form ([[...messages...]]).
                outputs = pipe([messages], **generation_kwargs)

            generated = outputs[0]["generated_text"] if outputs else ""
            if isinstance(generated, list) and generated:
                last_message = generated[-1]
                if isinstance(last_message, dict):
                    content = last_message.get("content", "")
                    if isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(str(item.get("text", "")).strip())
                        return "\n".join(part for part in text_parts if part).strip()
                    return str(content).strip()
                return str(last_message).strip()
            return str(generated).strip()

        if self._is_openreasoning_nemotron_model(config.name):
            messages = self._build_chat_messages(system_prompt, user_prompt, config.name)

            pipe = self.loaded_pipelines.get(config.name)
            if pipe is None:
                pipe = hf_pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
                self.loaded_pipelines[config.name] = pipe

            generation_kwargs = {
                "max_new_tokens": config.max_tokens,
                "do_sample": config.temperature > 0,
            }
            if config.temperature > 0:
                generation_kwargs["temperature"] = config.temperature

            outputs = pipe(messages, **generation_kwargs)
            generated = outputs[0]["generated_text"] if outputs else ""
            if isinstance(generated, list) and generated:
                last_message = generated[-1]
                if isinstance(last_message, dict):
                    return str(last_message.get("content", "")).strip()
                return str(last_message).strip()
            return str(generated).strip()

        if self._is_phi4_flash_reasoning_model(config.name):
            messages = self._build_chat_messages(system_prompt, user_prompt, config.name)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Phi-4-mini-flash-reasoning is math-oriented and tends to start a
            # long chain-of-thought even for our direct-answer QA benchmark.
            # Priming the assistant prefix with "ANSWER:" keeps it in the short
            # answer regime and avoids multi-minute reasoning traces.
            prompt_text = f"{prompt_text}ANSWER:"
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            generation_kwargs = {
                "max_new_tokens": min(config.max_tokens, 64),
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if tokenizer.eos_token_id is not None:
                generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)

            generated_text = tokenizer.decode(
                outputs[0][input_len:],
                skip_special_tokens=False,
            ).strip()
            if not generated_text:
                return "ANSWER:"
            return f"ANSWER: {generated_text.lstrip()}"

        # Gestion des graines (seed)
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

        # Utilisation de la nouvelle logique de formatage
        if prompt_text is None:
            if system_prompt is not None and user_prompt is not None:
                prompt_text = self._format_chat_prompt(system_prompt, user_prompt, config.name, tokenizer)
            else:
                raise ValueError("Either prompt_text or (system_prompt, user_prompt) must be provided")

        # Encodage et génération
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=config.context_window)
        if config.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif config.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        # Paramètres de génération (Greedy si temp=0)
        do_sample = config.temperature > 0
        gen_config = {
            "max_new_tokens": config.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        eos_token_ids: list[int] = []
        if self._is_gpt_oss_model(config.name):
            for special_token in ("<|return|>", "<|call|>"):
                try:
                    token_id = tokenizer.convert_tokens_to_ids(special_token)
                except Exception:
                    token_id = None
                if token_id is not None and token_id != tokenizer.unk_token_id and token_id not in eos_token_ids:
                    eos_token_ids.append(int(token_id))
        elif tokenizer.eos_token_id is not None:
            eos_token_ids.append(int(tokenizer.eos_token_id))
        gen_config["eos_token_id"] = eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0]
        if do_sample:
            gen_config["temperature"] = config.temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_config)

        # Extraction propre du texte généré (sans le prompt)
        input_length = inputs["input_ids"].shape[1]
        skip_special_tokens = not self._is_gpt_oss_model(config.name)
        generated_text = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=skip_special_tokens,
        )

        return generated_text.strip()

    def generate_llama_cpp(
        self,
        config: LocalModelConfig,
        prompt_text: str,
    ) -> str:
        """Generate using llama.cpp backend."""
        model = self.loaded_models[config.name]

        # Prepare kwargs for generation
        generation_kwargs = {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "echo": False,  # Don't echo the prompt
            "stop": ["</s>", "\n\n\n"],  # Common stop sequences
        }

        # Add seed if provided (llama.cpp supports seed parameter)
        if config.seed is not None:
            generation_kwargs["seed"] = config.seed

        response = model(prompt_text, **generation_kwargs)

        return response["choices"][0]["text"].strip()

    def generate(
        self,
        config: LocalModelConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """
        Generate a response from a local LLM.

        Returns a dict compatible with LLMaaS evaluation pipeline:
        {
            "content": str,
            "reasoning_content": str,  # Empty for local models (unless model supports it)
            "raw_api_response_json": dict,
            "raw_api_response_repr": str,
            "error": Optional[str]
        }
        """
        try:
            # Ensure model is loaded
            if config.name not in self.loaded_models:
                self.load_model(config)

            # Get the formatted prompt that will be sent to the model
            formatted_prompt = None
            if config.backend == "transformers":
                tokenizer = self.loaded_tokenizers.get(config.name)
                # Get the formatted prompt before generation
                if system_prompt is not None and user_prompt is not None:
                    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                        try:
                            if self._is_qwen35_model(config.name):
                                messages = self._build_chat_messages(system_prompt, user_prompt, config.name)
                                template_kwargs = {
                                    "tokenize": False,
                                    "add_generation_prompt": True,
                                }
                                if not self._qwen35_enable_thinking(config.name):
                                    template_kwargs["enable_thinking"] = False
                                formatted_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
                            elif self._is_gemma4_model(config.name):
                                messages = self._build_chat_messages(system_prompt, user_prompt, config.name)
                                formatted_prompt = tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False,
                                )
                            elif self._is_gpt_oss_model(config.name):
                                messages = self._build_chat_messages(system_prompt, user_prompt, config.name)
                                formatted_prompt = tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    reasoning_effort=self._gpt_oss_reasoning_effort(config.name),
                                )
                            else:
                                messages = self._build_chat_messages(system_prompt, user_prompt, config.name)
                                formatted_prompt = tokenizer.apply_chat_template(
                                    messages, tokenize=False, add_generation_prompt=True
                                )
                        except:
                            formatted_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                    else:
                        formatted_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                raw_content = self.generate_transformers(config, None, system_prompt, user_prompt)
            elif config.backend == "llama-cpp":
                # Format prompt for llama-cpp (simpler format usually works)
                formatted_prompt = self._format_chat_prompt(system_prompt, user_prompt, config.name)
                raw_content = self.generate_llama_cpp(config, formatted_prompt)
            else:
                raise ValueError(f"Unknown backend: {config.backend}")

            # Extract reasoning if present in the response
            if self._is_gpt_oss_model(config.name):
                content, reasoning_content = extract_gpt_oss_harmony_content(raw_content)
            else:
                content, reasoning_content = extract_reasoning_from_content(raw_content)

            # Debug: log if reasoning was extracted (remove in production if desired)
            if reasoning_content:
                print(f"   Extracted reasoning ({len(reasoning_content)} chars)")
            elif "<think>" in raw_content.lower() or "<reasoning>" in raw_content.lower():
                print(f"   Found reasoning tags but extraction failed. Raw content preview: {raw_content[:200]}")

            # Format response to match LLMaaS interface
            raw_response = {
                "model": config.name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "formatted_prompt": formatted_prompt,  # The actual prompt sent to the model
                "content": content,
                "raw_content": raw_content,  # Keep original before extraction
                "reasoning_content": reasoning_content,
                "backend": config.backend,
            }

            return {
                "content": content,
                "reasoning_content": reasoning_content,
                "raw_api_response_json": json.dumps(raw_response),
                "raw_api_response_repr": repr(raw_response),
                "error": None,
            }

        except Exception as e:
            import traceback

            traceback.print_exc()
            return {
                "content": "",
                "reasoning_content": "",
                "raw_api_response_json": json.dumps({"error": str(e)}),
                "raw_api_response_repr": repr(e),
                "error": str(e),
            }
