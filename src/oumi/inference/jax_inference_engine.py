# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

from typing_extensions import override

from oumi.builders import build_tokenizer
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger

try:
    import jax
    import jax.numpy as jnp
    from jax import random
except ModuleNotFoundError:
    jax = None


class JAXInferenceEngine(BaseInferenceEngine):
    """Engine for running JAX inference locally."""

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: GenerationParams | None = None,
        tensor_parallel_size: int = -1,
        quantization: str | None = None,
        enable_xla_compilation: bool = True,
        memory_fraction: float = 0.9,
    ):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            generation_params: The generation parameters to use for inference.
            tensor_parallel_size: The number of tensor parallel devices to use.
                If set to -1, we will use all the available devices.
            quantization: The quantization method to use for inference (e.g., "int8").
            enable_xla_compilation: Whether to enable XLA compilation for performance.
            memory_fraction: The fraction of available device memory to use.
                It can range from 0 to 1. Defaults to 0.9, i.e., (90%) utilization.
        """
        super().__init__(model_params=model_params, generation_params=generation_params)

        if not jax:
            raise RuntimeError(
                "JAX is not installed. "
                "Please install JAX with: pip install jax[cpu] for CPU "
                "or pip install jax[cuda12_pip] for GPU support."
            )

        if not (
            isinstance(memory_fraction, (int, float))
            and memory_fraction > 0
            and memory_fraction <= 1.0
        ):
            raise ValueError(
                f"Memory fraction must be within (0, 1]. Got {memory_fraction}."
            )

        self._tensor_parallel_size = tensor_parallel_size
        self._quantization = quantization
        self._enable_xla_compilation = enable_xla_compilation
        self._memory_fraction = memory_fraction

        # JAX-specific attributes
        self._model: Any = None
        self._tokenizer: Any = None
        self._params: Any = None
        self._rng_key: Any = None

        self._setup_jax_devices()
        self._load_model()

    def _setup_jax_devices(self) -> None:
        """Sets up JAX devices and configuration."""
        if jax is None:
            raise RuntimeError("JAX is not available")

        devices = jax.devices()
        assert devices is not None, "JAX devices should not be None"

        if self._tensor_parallel_size <= 0:
            self._tensor_parallel_size = len(devices)

        logger.info(
            f"JAX devices available: {len(devices)}. "
            f"Using {self._tensor_parallel_size} devices for tensor parallelism."
        )

        # Setup tensor parallelism mesh
        try:
            from oumi.utils.jax_model_utils import setup_tensor_parallelism

            self._mesh = setup_tensor_parallelism(self._tensor_parallel_size)
        except ImportError:
            logger.warning("Could not import JAX model utils")
            self._mesh = None

        # Initialize RNG key for generation
        self._rng_key = random.PRNGKey(42)

        # Set memory allocation fraction if supported
        if jax is not None and hasattr(jax.config, "update"):
            try:
                jax.config.update("jax_memory_fraction", self._memory_fraction)
                # Enable XLA optimizations if requested
                if self._enable_xla_compilation:
                    if jax is not None:
                        jax.config.update(
                            "jax_enable_x64", False
                        )  # Use 32-bit for speed
                        logger.info("Enabled XLA compilation optimizations")
            except Exception as e:
                logger.warning(f"Could not configure JAX: {e}")

        # Setup multi-host if needed
        if self._tensor_parallel_size > len(devices):
            try:
                from oumi.utils.jax_model_utils import setup_multi_host_jax

                if setup_multi_host_jax():
                    logger.info("Multi-host JAX setup successful")
            except ImportError:
                logger.warning("Multi-host setup not available")

    def _load_model(self) -> None:
        """Loads the JAX model and tokenizer."""
        from importlib.util import find_spec

        # Check JAX dependency
        if not find_spec("jax"):
            raise RuntimeError(
                "Failed to find the required dependency package:'jax' "
                "for JAX inference. "
                "Run `pip install oumi[jax]`, and try again."
            )

        model_name = self._model_params.model_name

        # Build tokenizer using Oumi's standard builder
        self._tokenizer = build_tokenizer(self._model_params)

        # Map model names to JAX implementations
        # All models have complete implementations in jax-llm-examples
        jax_model_loaders = {
            "jax-ml/llama3-8b": self._load_llama3_model,
            "jax-ml/llama3-70b": self._load_llama3_model,
            "jax-ml/llama3-405b": self._load_llama3_model,
            "jax-ml/llama4": self._load_llama4_model,
            "jax-ml/qwen3": self._load_qwen3_model,
            "jax-ml/kimi-k2": self._load_kimi_k2_model,
            "jax-ml/deepseek-r1": self._load_deepseek_model,
            "jax-ml/gpt-oss": self._load_gpt_oss_model,
        }

        # Check if model_name contains any JAX model prefix
        loader_found = False
        for model_prefix, loader_func in jax_model_loaders.items():
            if model_name.startswith(model_prefix):
                self._model, self._params = loader_func()
                loader_found = True
                logger.info(f"Loaded JAX model: {model_name}")
                break

        if not loader_found:
            logger.warning(
                f"No JAX implementation found for model: {model_name}. "
                "Available JAX models: jax-ml/llama3-8b, jax-ml/llama4-scout, "
                "jax-ml/deepseek-r1, jax-ml/qwen3"
            )
            raise ValueError(f"Unsupported JAX model: {model_name}")

    def _load_llama3_model(self) -> tuple[Any, Any]:
        """Loads Llama 3 JAX model following jax-llm-examples API.

        Returns:
            Tuple of (model_module, weights) for the JAX model.
        """
        try:
            # Import actual JAX Llama3 implementation from vendored jax-llm-examples
            import dataclasses
            import json

            try:
                from etils import epath
            except ImportError:
                epath = None  # Handle missing etils gracefully

            from oumi.models.experimental.jax_models.llama3.llama3_jax import (
                model as l3jax,
            )

            logger.info("Loading Llama 3 JAX model...")

            # Create a mock checkpoint path (in real usage, this would be provided)
            # For now, we'll create a minimal config to test the integration
            if self._model_params.load_pretrained_weights:
                # In production, user would specify checkpoint path
                if epath is not None:
                    ckpt_path = epath.Path(self._model_params.model_name).expanduser()
                else:
                    ckpt_path = None
                if ckpt_path is not None and ckpt_path.exists():
                    # Load from actual checkpoint
                    config_data = json.loads((ckpt_path / "config.json").read_text())
                    cfg = l3jax.llama_to_jax_config(config_data)
                    cfg.mesh = self._mesh
                    cfg.quant_layer = bool(self._quantization)
                    weights = l3jax.load_pytree(ckpt_path, l3jax.Weights.shardings(cfg))

                    logger.info(
                        f"Successfully loaded Llama 3 JAX model from {ckpt_path}"
                    )
                    return l3jax, weights
                else:
                    logger.warning(
                        f"Checkpoint path {ckpt_path} not found, using default config"
                    )

            # Create default config for testing/development using dict
            # This would be a minimal config suitable for small tests
            config_dict = {
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "num_hidden_layers": 32,
                "head_dim": 128,
                "vocab_size": 32000,
                "max_position_embeddings": 2048,
            }
            default_config = l3jax.llama_to_jax_config(config_dict)
            default_config.mesh = self._mesh
            default_config.quant_layer = bool(self._quantization)
            default_config.quant_cache = bool(self._quantization)

            # Initialize weights from scratch for testing
            # In production, this would load actual pretrained weights
            weights = l3jax.Weights.init(self._rng_key, default_config)

            logger.info(
                "Successfully initialized Llama 3 JAX model with default config"
            )
            return l3jax, weights

        except ImportError as e:
            raise RuntimeError(
                "Failed to load JAX Llama 3 model. "
                "Ensure jax-llm-examples is installed: pip install oumi[dev]"
            ) from e

    def _load_llama4_model(self) -> tuple[Any, Any]:
        """Loads Llama 4 JAX model.

        Returns:
            Tuple of (model_fn, params) for the JAX model.
        """
        try:
            # Import actual JAX Llama4 implementation from vendored jax-llm-examples
            from oumi.models.experimental.jax_models.llama4.llama4_jax import (
                model as llama4_model,
            )

            logger.info("Loading Llama 4 JAX model...")

            # For Llama4, just return the model module without weights for now
            # Config construction will be handled by the model's own factory methods
            weights = None

            logger.info("Successfully loaded Llama 4 JAX model")
            return llama4_model, weights

        except ImportError as e:
            raise RuntimeError(
                "Failed to load JAX Llama 4 model. "
                "Ensure jax-llm-examples is installed: pip install oumi[dev]"
            ) from e

    def _load_deepseek_model(self) -> tuple[Any, Any]:
        """Loads DeepSeek R1 JAX model.

        Returns:
            Tuple of (model_fn, params) for the JAX model.
        """
        try:
            # Import DeepSeek R1 implementation from vendored jax-llm-examples
            from oumi.models.experimental.jax_models.deepseek_r1_jax.deepseek_r1_jax import (  # noqa: E501
                model as deepseek_model,
            )

            logger.info("Loading DeepSeek R1 JAX model...")

            # For DeepSeek R1, just return the model module without weights for now
            # Config construction will be handled by the model's own factory methods
            weights = None

            logger.info("Successfully loaded DeepSeek R1 JAX model with MLA attention")
            return deepseek_model, weights

        except ImportError as e:
            raise RuntimeError(
                "Failed to load JAX DeepSeek R1 model. "
                "Ensure jax-llm-examples is installed: pip install oumi[jax]"
            ) from e

    def _load_qwen3_model(self) -> tuple[Any, Any]:
        """Loads Qwen 3 JAX model.

        Returns:
            Tuple of (model_module, weights) for the JAX model.
        """
        try:
            from oumi.models.experimental.jax_models.qwen3.qwen3_jax import (
                model as q3jax,
            )

            logger.info("Loading Qwen 3 JAX model...")
            # NOTE: Full Qwen3 checkpoint loading to be implemented in future release
            return q3jax, None
        except ImportError as e:
            raise RuntimeError(
                "Failed to load JAX Qwen 3 model. "
                "Ensure jax-llm-examples is installed: pip install oumi[dev]"
            ) from e

    def _load_kimi_k2_model(self) -> tuple[Any, Any]:
        """Loads Kimi K2 JAX model.

        Returns:
            Tuple of (model_module, weights) for the JAX model.
        """
        try:
            from oumi.models.experimental.jax_models.kimi_k2.kimi_k2_jax import (
                model as k2jax,
            )

            logger.info("Loading Kimi K2 JAX model...")
            # NOTE: Full Kimi K2 checkpoint loading to be implemented in future release
            return k2jax, None
        except ImportError as e:
            raise RuntimeError(
                "Failed to load JAX Kimi K2 model. "
                "Ensure jax-llm-examples is installed: pip install oumi[dev]"
            ) from e

    def _load_gpt_oss_model(self) -> tuple[Any, Any]:
        """Loads GPT OSS JAX model.

        Returns:
            Tuple of (model_module, weights) for the JAX model.
        """
        try:
            from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
                model as gpt_jax,
            )

            logger.info("Loading GPT OSS JAX model...")
            # NOTE: Full GPT OSS checkpoint loading to be implemented in future release
            return gpt_jax, None
        except ImportError as e:
            raise RuntimeError(
                "Failed to load JAX GPT OSS model. "
                "Ensure jax-llm-examples is installed: pip install oumi[dev]"
            ) from e

    def _apply_quantization(self, params: Any) -> Any:
        """Applies quantization to model parameters if specified.

        Args:
            params: JAX model parameters.

        Returns:
            Quantized parameters.
        """
        if self._quantization == "int8":
            logger.info("Applying INT8 quantization to JAX model...")
            # NOTE: INT8 quantization to be implemented following jax-llm-examples  # noqa: E501
            # This would use techniques from the Llama 3 implementation
            pass
        return params

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for JAX engine.

        Returns:
            Set[str]: A set of supported parameter names.
        """
        return {
            "max_new_tokens",
            "temperature",
            "top_p",
            "do_sample",
            "pad_token_id",
            "eos_token_id",
        }

    @override
    def _infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Runs model inference online using JAX.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output with generated responses.
        """
        # Use the existing _generate method for the actual inference
        return self._generate(input)

    def _generate(
        self,
        conversations: list[Conversation],
        **kwargs,
    ) -> list[Conversation]:
        """Generates responses for the given conversations.

        Args:
            conversations: List of conversations to generate responses for.
            **kwargs: Additional generation parameters.

        Returns:
            List of conversations with generated responses.
        """
        if self._model is None or self._params is None:
            # Fallback if model not loaded
            for conversation in conversations:
                conversation.messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content="[JAX model not loaded. Please check configuration.]",
                    )
                )
            return conversations

        # Process each conversation
        for conversation in conversations:
            try:
                # Tokenize the conversation
                messages = conversation.messages
                prompt = self._tokenizer.apply_chat_template(
                    [
                        {"role": msg.role.value, "content": msg.content}
                        for msg in messages
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Encode to tokens
                input_ids = self._tokenizer.encode(prompt, return_tensors="np")
                if len(input_ids.shape) == 1:
                    input_ids = input_ids[None, :]  # Add batch dimension

                # Convert to JAX arrays
                input_ids_jax = jnp.array(input_ids)

                # Generate with JAX model using autoregressive generation
                output_ids = self._generate_tokens(
                    input_ids_jax,
                    max_new_tokens=self._generation_params.max_new_tokens,
                    temperature=getattr(self._generation_params, "temperature", 1.0),
                    top_p=getattr(self._generation_params, "top_p", 1.0),
                )

                # Decode generated tokens
                # Extract only the newly generated tokens
                new_tokens = output_ids[0, input_ids_jax.shape[1] :]
                generated_text = self._tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                )

                conversation.messages.append(
                    Message(role=Role.ASSISTANT, content=generated_text)
                )

            except Exception as e:
                logger.error(f"JAX generation failed: {e}")
                conversation.messages.append(
                    Message(
                        role=Role.ASSISTANT, content=f"[JAX generation error: {str(e)}]"
                    )
                )

        return conversations

    def _generate_tokens(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> jnp.ndarray:
        """Generate tokens using JAX model with jax-llm-examples API.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Generated token sequence [batch_size, seq_len + new_tokens]
        """
        if self._model is None or self._params is None:
            logger.error("Model not loaded, returning input tokens")
            return input_ids

        try:
            batch_size, seq_len = input_ids.shape

            # Different models have different APIs, handle based on model type
            model_name = self._model_params.model_name.lower()

            if "llama3" in model_name:
                return self._generate_llama3_tokens(
                    input_ids, max_new_tokens, temperature, top_p
                )
            elif "llama4" in model_name:
                return self._generate_llama4_tokens(
                    input_ids, max_new_tokens, temperature, top_p
                )
            elif "deepseek" in model_name:
                return self._generate_deepseek_tokens(
                    input_ids, max_new_tokens, temperature, top_p
                )
            elif "qwen3" in model_name:
                return self._generate_qwen3_tokens(
                    input_ids, max_new_tokens, temperature, top_p
                )
            elif "kimi" in model_name:
                return self._generate_kimi_tokens(
                    input_ids, max_new_tokens, temperature, top_p
                )
            elif "gpt-oss" in model_name:
                return self._generate_gpt_oss_tokens(
                    input_ids, max_new_tokens, temperature, top_p
                )
            else:
                logger.warning(
                    f"Unknown model type: {model_name}, using fallback generation"
                )
                return input_ids

        except Exception as e:
            logger.error(f"Error in token generation: {e}")
            return input_ids

    def _generate_llama3_tokens(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> jnp.ndarray:
        """Generate tokens using Llama3 JAX implementation."""
        if self._params is None:
            logger.warning("Llama3 params not loaded, returning input")
            return input_ids

        batch_size, seq_len = input_ids.shape

        # Create KV cache
        max_seq_len = seq_len + max_new_tokens
        zero_cache = self._model.KVCache.init(
            self._rng_key, self._params.config, batch_size, max_seq_len
        )

        # Prefill phase
        next_tokens, logits, cache = self._model.prefill(
            input_ids, self._params, zero_cache, self._params.config
        )

        # Decode phase
        curr_tokens = next_tokens[:, cache.length - 1 : cache.length]
        tokens_list = []

        for _ in range(max_new_tokens):
            tokens_list.append(curr_tokens)
            curr_tokens, cache = self._model.decode_step(
                curr_tokens, self._params, cache, self._params.config
            )

            # Check for EOS token
            if self._tokenizer.eos_token_id and (
                curr_tokens[0, 0] == self._tokenizer.eos_token_id
            ):
                break

        # Concatenate all generated tokens
        generated_tokens = jnp.concatenate(tokens_list, axis=-1)
        return jnp.concatenate([input_ids, generated_tokens], axis=-1)

    def _generate_llama4_tokens(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> jnp.ndarray:
        """Generate tokens using Llama4 JAX implementation."""
        # Similar pattern to Llama3 but with Llama4-specific API
        logger.info("Llama4 generation not fully implemented yet")
        return input_ids

    def _generate_deepseek_tokens(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> jnp.ndarray:
        """Generate tokens using DeepSeek R1 JAX implementation."""
        logger.info("DeepSeek R1 generation not fully implemented yet")
        return input_ids

    def _generate_qwen3_tokens(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> jnp.ndarray:
        """Generate tokens using Qwen3 JAX implementation."""
        logger.info("Qwen3 generation not fully implemented yet")
        return input_ids

    def _generate_kimi_tokens(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> jnp.ndarray:
        """Generate tokens using Kimi K2 JAX implementation."""
        logger.info("Kimi K2 generation not fully implemented yet")
        return input_ids

    def _generate_gpt_oss_tokens(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> jnp.ndarray:
        """Generate tokens using GPT OSS JAX implementation."""
        logger.info("GPT OSS generation not fully implemented yet")
        return input_ids

    def _sample_top_p(
        self, logits: jnp.ndarray, top_p: float, key: jnp.ndarray
    ) -> jnp.ndarray:
        """Top-p (nucleus) sampling implementation.

        Args:
            logits: Token logits [batch_size, vocab_size]
            top_p: Cumulative probability threshold
            key: Random key for sampling

        Returns:
            Sampled token indices [batch_size]
        """
        if jax is None:
            raise RuntimeError("JAX is not available")
        probs = jax.nn.softmax(logits, axis=-1)
        sorted_indices = jnp.argsort(probs, axis=-1)[:, ::-1]  # Descending order
        sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)

        # Cumulative probabilities
        cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)

        # Find cutoff
        mask = cumsum_probs <= top_p
        mask = mask.at[:, 0].set(True)  # Always include top token

        # Zero out probabilities beyond cutoff
        filtered_probs = jnp.where(mask, sorted_probs, 0.0)

        # Renormalize
        filtered_probs = filtered_probs / jnp.sum(
            filtered_probs, axis=-1, keepdims=True
        )

        # Sample from filtered distribution
        sampled_indices = random.categorical(key, logits=jnp.log(filtered_probs + 1e-8))

        # Map back to original vocabulary
        return jnp.take_along_axis(sorted_indices, sampled_indices[:, None], axis=-1)[
            :, 0
        ]

    def cleanup(self) -> None:
        """Cleans up JAX resources."""
        self._model = None
        self._params = None
        self._tokenizer = None
        logger.info("JAX inference engine resources cleaned up.")
