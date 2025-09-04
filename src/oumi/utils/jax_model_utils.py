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

"""JAX model utilities adapted from jax-llm-examples for advanced features."""

import warnings
from typing import Any, Optional

from oumi.utils.logging import logger

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import shard_map
    from jax.sharding import Mesh
except ImportError:
    jax = None
    jnp = None
    shard_map = None


def setup_tensor_parallelism(num_devices: int) -> Optional[Mesh]:
    """Setup JAX mesh for tensor parallelism.

    Args:
        num_devices: Number of devices for parallelism.

    Returns:
        JAX Mesh object for distributed computation.
    """
    if jax is None:
        logger.warning("JAX not available for tensor parallelism")
        return None

    try:
        devices = jax.devices()[:num_devices]
        mesh = Mesh(devices, axis_names=("tp",))
        logger.info(f"Created JAX mesh with {num_devices} devices: {mesh}")
        return mesh
    except Exception as e:
        logger.error(f"Failed to setup tensor parallelism: {e}")
        return None


def apply_int8_quantization(params: dict[str, Any]) -> dict[str, Any]:
    """Apply int8 quantization to model parameters.

    Args:
        params: Model parameters dictionary.

    Returns:
        Quantized parameters.
    """
    if jnp is None:
        logger.warning("JAX not available for quantization")
        return params

    quantized_params = {}

    try:
        for key, value in params.items():
            if isinstance(value, dict):
                # Recursively quantize nested parameters
                quantized_params[key] = apply_int8_quantization(value)
            elif isinstance(value, jnp.ndarray) and value.dtype in (
                jnp.float32,
                jnp.bfloat16,
            ):
                # Quantize weight matrices
                if len(value.shape) >= 2:  # Only quantize 2D+ tensors
                    # Simple int8 quantization
                    scale = jnp.max(jnp.abs(value)) / 127.0
                    quantized = jnp.round(value / scale).astype(jnp.int8)
                    quantized_params[key] = {
                        "quantized": quantized,
                        "scale": scale,
                        "original_dtype": value.dtype,
                    }
                else:
                    quantized_params[key] = value  # Keep 1D tensors unquantized
            else:
                quantized_params[key] = value

        logger.info("Applied int8 quantization to model parameters")
        return quantized_params

    except Exception as e:
        logger.error(f"Failed to apply quantization: {e}")
        return params


def dequantize_int8(quantized_param: dict[str, Any]) -> Optional[Any]:
    """Dequantize int8 parameters back to float.

    Args:
        quantized_param: Dictionary with quantized tensor, scale, and dtype.

    Returns:
        Dequantized parameter.
    """
    if not isinstance(quantized_param, dict) or "quantized" not in quantized_param:
        return quantized_param

    try:
        quantized = quantized_param["quantized"]
        scale = quantized_param["scale"]
        original_dtype = quantized_param.get("original_dtype", jnp.float32)

        return quantized.astype(original_dtype) * scale
    except Exception as e:
        logger.error(f"Failed to dequantize parameter: {e}")
        return quantized_param


def create_attention_mask(
    seq_len: int, attention_type: str = "causal"
) -> Optional[Any]:
    """Create attention mask for different attention patterns.

    Args:
        seq_len: Sequence length.
        attention_type: Type of attention ("causal", "bidirectional", "ragged").

    Returns:
        Attention mask array.
    """
    if jnp is None:
        return None

    if attention_type == "causal":
        # Lower triangular mask for causal attention
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    elif attention_type == "bidirectional":
        # Full attention
        mask = jnp.ones((seq_len, seq_len))
    elif attention_type == "ragged":
        # NOTE: Ragged attention pattern implementation pending based on jax-llm-examples  # noqa: E501
        warnings.warn("Ragged attention not fully implemented")
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")

    return mask


def load_checkpoint_jax(checkpoint_path: str) -> Optional[dict[str, Any]]:
    """Load JAX model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Loaded parameters or None if failed.
    """
    try:
        # This would integrate with jax-llm-examples checkpoint loading
        # For now, placeholder implementation
        logger.info(f"Loading JAX checkpoint from {checkpoint_path}")

        # NOTE: Checkpoint loading implementation to follow jax-llm-examples patterns
        # This would use their chkpt_utils.py functionality
        warnings.warn("JAX checkpoint loading not fully implemented")
        return None

    except Exception as e:
        logger.error(f"Failed to load JAX checkpoint: {e}")
        return None


def save_checkpoint_jax(params: dict[str, Any], checkpoint_path: str) -> bool:
    """Save JAX model checkpoint.

    Args:
        params: Model parameters to save.
        checkpoint_path: Path to save checkpoint.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # This would integrate with jax-llm-examples checkpoint saving
        logger.info(f"Saving JAX checkpoint to {checkpoint_path}")

        # NOTE: Checkpoint saving implementation to follow jax-llm-examples patterns
        warnings.warn("JAX checkpoint saving not fully implemented")
        return False

    except Exception as e:
        logger.error(f"Failed to save JAX checkpoint: {e}")
        return False


def convert_pytorch_to_jax_weights(
    pytorch_state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Convert PyTorch state dict to JAX parameters.

    Args:
        pytorch_state_dict: PyTorch model state dictionary.

    Returns:
        JAX-compatible parameters.
    """
    if jnp is None:
        logger.warning("JAX not available for weight conversion")
        return {}

    try:
        jax_params = {}

        for key, tensor in pytorch_state_dict.items():
            # Convert PyTorch tensor to numpy, then to JAX
            if hasattr(tensor, "detach"):
                numpy_array = tensor.detach().cpu().numpy()
            else:
                numpy_array = tensor

            jax_params[key] = jnp.array(numpy_array)

        logger.info(f"Converted {len(jax_params)} PyTorch weights to JAX")
        return jax_params

    except Exception as e:
        logger.error(f"Failed to convert PyTorch weights to JAX: {e}")
        return {}


def get_model_flops(model_config: dict[str, Any], seq_len: int) -> int:
    """Estimate FLOPs for model inference.

    Args:
        model_config: Model configuration dictionary.
        seq_len: Input sequence length.

    Returns:
        Estimated FLOPs for forward pass.
    """
    try:
        # Basic FLOP estimation for transformer models
        # This follows patterns from the JAX scaling book

        hidden_size = model_config.get("hidden_size", model_config.get("d_model", 4096))
        num_layers = model_config.get("num_layers", model_config.get("n_layer", 32))
        vocab_size = model_config.get("vocab_size", 32000)

        # Attention FLOPs: 4 * batch * seq_len^2 * hidden_size * num_layers
        attention_flops = 4 * seq_len * seq_len * hidden_size * num_layers

        # MLP FLOPs: 8 * batch * seq_len * hidden_size^2 * num_layers (4x expansion)  # noqa: E501
        mlp_flops = 8 * seq_len * (hidden_size**2) * num_layers

        # Embedding FLOPs: batch * seq_len * hidden_size * vocab_size
        embedding_flops = seq_len * hidden_size * vocab_size

        total_flops = attention_flops + mlp_flops + embedding_flops

        logger.info(f"Estimated {total_flops:,} FLOPs for model inference")
        return total_flops

    except Exception as e:
        logger.error(f"Failed to estimate model FLOPs: {e}")
        return 0


def setup_multi_host_jax() -> bool:
    """Setup JAX for multi-host distributed execution.

    Returns:
        True if setup successful, False otherwise.
    """
    if jax is None:
        logger.warning("JAX not available for multi-host setup")
        return False

    try:
        # Initialize JAX distributed
        jax.distributed.initialize()

        # Get process info
        process_count = jax.process_count()
        process_index = jax.process_index()

        logger.info(f"JAX multi-host setup: process {process_index}/{process_count}")
        return True

    except Exception as e:
        logger.error(f"Failed to setup JAX multi-host: {e}")
        return False
