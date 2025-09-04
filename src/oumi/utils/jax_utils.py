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

"""JAX utilities for performance optimization and tensor conversion."""

import time
from typing import Any

import numpy as np
import torch

from oumi.utils.logging import logger

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None


def torch_to_jax(tensor: torch.Tensor) -> Any:
    """Convert PyTorch tensor to JAX array.

    Args:
        tensor: PyTorch tensor to convert.

    Returns:
        JAX array or None if JAX not available.
    """
    if jnp is None:
        logger.warning("JAX not available, cannot convert tensor")
        return None

    # Convert to numpy first, then to JAX
    numpy_array = tensor.detach().cpu().numpy()
    return jnp.array(numpy_array)


def jax_to_torch(array: Any, device: str = "cpu") -> torch.Tensor | None:
    """Convert JAX array to PyTorch tensor.

    Args:
        array: JAX array to convert.
        device: Target PyTorch device.

    Returns:
        PyTorch tensor or None if conversion fails.
    """
    if jax is None:
        logger.warning("JAX not available, cannot convert array")
        return None

    try:
        # Convert JAX array to numpy, then to PyTorch
        numpy_array = np.array(array)
        return torch.from_numpy(numpy_array).to(device)
    except Exception as e:
        logger.error(f"Failed to convert JAX array to PyTorch: {e}")
        return None


def benchmark_inference_engines(
    models: dict[str, Any],
    test_prompts: list[str],
    num_runs: int = 5,
) -> dict[str, dict[str, float]]:
    """Benchmark different inference engines for performance comparison.

    Args:
        models: Dictionary of model_name -> inference_engine
        test_prompts: List of test prompts to run
        num_runs: Number of runs for averaging

    Returns:
        Performance metrics for each model
    """
    results = {}

    for model_name, engine in models.items():
        logger.info(f"Benchmarking {model_name}...")

        times = []
        tokens_per_second = []

        for run in range(num_runs):
            start_time = time.time()

            # Run inference on all test prompts
            for prompt in test_prompts:
                # This would need to be adapted based on actual engine interface
                _ = engine.generate(prompt)

            end_time = time.time()
            run_time = end_time - start_time
            times.append(run_time)

            # Calculate tokens/second (rough estimate)
            total_tokens = len(test_prompts) * 50  # Assume 50 tokens per response
            tps = total_tokens / run_time
            tokens_per_second.append(tps)

        results[model_name] = {
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "avg_tokens_per_second": np.mean(tokens_per_second),
            "std_tokens_per_second": np.std(tokens_per_second),
        }

    return results


def check_jax_devices() -> dict[str, Any]:
    """Check available JAX devices and their properties.

    Returns:
        Device information dictionary.
    """
    if jax is None:
        return {"error": "JAX not available"}

    devices = jax.devices()
    device_info = {
        "num_devices": len(devices),
        "device_types": [str(device.device_kind) for device in devices],
        "devices": [str(device) for device in devices],
    }

    # Check for TPUs
    tpu_devices = [d for d in devices if "tpu" in str(d).lower()]
    device_info["has_tpus"] = len(tpu_devices) > 0
    device_info["num_tpus"] = len(tpu_devices)

    # Check for GPUs
    gpu_devices = [d for d in devices if "gpu" in str(d).lower()]
    device_info["has_gpus"] = len(gpu_devices) > 0
    device_info["num_gpus"] = len(gpu_devices)

    return device_info


def setup_jax_for_performance() -> None:
    """Configure JAX for optimal performance."""
    if jax is None:
        logger.warning("JAX not available, cannot configure")
        return

    try:
        # Enable XLA optimizations
        jax.config.update("jax_enable_x64", False)  # Use 32-bit for speed
        jax.config.update("jax_default_matmul_precision", "high")

        logger.info("JAX configured for performance")
    except Exception as e:
        logger.warning(f"Could not configure JAX: {e}")


def memory_usage_mb() -> float:
    """Get current JAX device memory usage in MB.

    Returns:
        Memory usage in megabytes.
    """
    if jax is None:
        return 0.0

    try:
        # This is a simplified version - actual implementation would
        # depend on specific JAX device memory tracking
        devices = jax.devices()
        if devices:
            # Placeholder - would need actual memory tracking
            return 0.0
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")

    return 0.0
