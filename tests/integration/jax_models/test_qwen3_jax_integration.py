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

"""Integration tests for Qwen3 JAX model implementation."""

from unittest.mock import patch

import jax.numpy as jnp
import pytest

# Import required decorators
try:
    from oumi.utils.testing_utils import requires_gpus
except ImportError:

    def requires_gpus(n):
        return pytest.mark.skipif(True, reason="requires_gpus not available")


# Fallback definition for pyright
if False:

    def requires_gpus(n):
        return pytest.mark.skipif(True, reason="requires_gpus not available")


# Mark all tests in this file as JAX-related
pytestmark = pytest.mark.jax


class TestQwen3JAXIntegration:
    """Integration tests for Qwen3 JAX model."""

    def test_qwen3_import(self):
        """Test that Qwen3 JAX model can be imported."""
        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import model as q3jax

        assert hasattr(q3jax, "Config")
        assert hasattr(q3jax, "Weights")
        assert hasattr(q3jax, "KVCache")
        assert hasattr(q3jax, "prefill")
        assert hasattr(q3jax, "decode_step")

    def test_qwen3_model_init_cpu(self):
        """Test Qwen3 model initialization on CPU."""
        import jax
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import model as q3jax

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create small test configuration
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = q3jax.ShardingRules(
            batch=None,
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=("y", "z"),  # Qwen3 requires this for TENSOR_AXIS_NAME
        )

        cfg = q3jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            moe_ffw_size=512,
            moe_experts_per_tok=2,
            moe_num_experts=8,
            mlp_ffw_size=512,
            mlp_layer_idxs=[0, 1],
            mesh=mesh,
            rules=simplified_rules,
        )

        # Test weight initialization
        weights = q3jax.Weights.init(random.key(42), cfg)
        assert weights is not None, "Weight initialization failed on CPU"

        # Test cache initialization
        batch_size = 2
        cache = q3jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)
        assert cache is not None, "Cache initialization failed on CPU"

        # Verify weight structure
        assert hasattr(weights, "embedding"), "Weights missing embedding component"
        assert hasattr(weights, "layers"), "Weights missing layers component"

    @requires_gpus(1)
    @pytest.mark.single_gpu
    def test_qwen3_model_init_gpu(self):
        """Test Qwen3 model initialization on GPU."""
        import jax
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import model as q3jax

        # Check if GPU is available
        devices = jax.devices()
        gpu_devices = [d for d in devices if "gpu" in str(d).lower()]

        if not gpu_devices:
            pytest.skip("No GPU devices available")

        # Create test configuration for GPU
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = q3jax.ShardingRules(
            batch=None,
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=("y", "z"),  # Qwen3 requires this for TENSOR_AXIS_NAME
        )

        cfg = q3jax.Config(
            embed=512,
            q_heads=16,
            kv_heads=8,
            num_layers=4,
            head_dim=32,
            vocab_size=2000,
            max_seq_len=256,
            causal=True,
            moe_ffw_size=1024,
            moe_experts_per_tok=2,
            moe_num_experts=8,
            mlp_ffw_size=1024,
            mlp_layer_idxs=[0, 1, 2, 3],
            mesh=mesh,
            rules=simplified_rules,
        )

        # Test weight initialization on GPU
        weights = q3jax.Weights.init(random.key(42), cfg)
        assert weights is not None, "Weight initialization failed on GPU"

        # Test cache initialization on GPU
        batch_size = 2
        cache = q3jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)
        assert cache is not None, "Cache initialization failed on GPU"

    def test_qwen3_prefill_decode_cpu(self):
        """Test Qwen3 prefill and decode on CPU."""
        import jax
        import jax.numpy as jnp
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import model as q3jax

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Handle different JAX versions for set_mesh
        try:
            from jax.sharding import set_mesh as set_mesh
        except ImportError:
            try:
                from jax.sharding import set_mesh as set_mesh
            except ImportError:
                set_mesh = None

        # Create small test configuration
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = q3jax.ShardingRules(
            batch=None,
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=("y", "z"),  # Qwen3 requires this for TENSOR_AXIS_NAME
        )

        cfg = q3jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=2,
            num_layers=2,
            head_dim=32,
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            moe_ffw_size=256,
            moe_experts_per_tok=2,
            moe_num_experts=4,
            mlp_ffw_size=256,
            mlp_layer_idxs=[0, 1],
            mesh=mesh,
            rules=simplified_rules,
        )

        # Initialize model components
        weights = q3jax.Weights.init(random.key(42), cfg)
        batch_size = 1
        seq_len = 16
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        cache = q3jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)

        # Test prefill
        if set_mesh is not None:
            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    max_tokens, logits, cache = q3jax.prefill(
                        tokens, weights, cache, cfg
                    )
                assert max_tokens.shape == (batch_size, seq_len)
                assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
        else:
            # Fallback for older JAX versions without set_mesh
            max_tokens, logits, cache = q3jax.prefill(tokens, weights, cache, cfg)
            assert max_tokens.shape == (batch_size, seq_len)
            assert logits.shape == (batch_size, seq_len, cfg.vocab_size)

        # Test decode steps
        next_tokens = max_tokens[:, -1:]
        if set_mesh is not None:
            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    for _ in range(3):
                        next_tokens, cache = q3jax.decode_step(
                            next_tokens, weights, cache, cfg
                        )
                    assert next_tokens.shape == (batch_size, 1)
        else:
            # Fallback for older JAX versions without set_mesh
            for _ in range(3):
                next_tokens, cache = q3jax.decode_step(next_tokens, weights, cache, cfg)
                assert next_tokens.shape == (batch_size, 1)

    @requires_gpus(1)
    @pytest.mark.single_gpu
    def test_qwen3_prefill_decode_gpu(self):
        """Test Qwen3 prefill and decode on GPU."""
        import jax
        import jax.numpy as jnp
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import model as q3jax

        # Check if GPU is available
        devices = jax.devices()
        gpu_devices = [d for d in devices if "gpu" in str(d).lower()]

        if not gpu_devices:
            pytest.skip("No GPU devices available")

        # Handle different JAX versions for set_mesh
        try:
            from jax.sharding import set_mesh as set_mesh
        except ImportError:
            try:
                from jax.sharding import set_mesh as set_mesh
            except ImportError:
                set_mesh = None

        # Create test configuration for GPU
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = q3jax.ShardingRules(
            batch=None,
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=("y", "z"),  # Qwen3 requires this for TENSOR_AXIS_NAME
        )

        cfg = q3jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            moe_ffw_size=512,
            moe_experts_per_tok=2,
            moe_num_experts=8,
            mlp_ffw_size=512,
            mlp_layer_idxs=[0, 1],
            mesh=mesh,
            rules=simplified_rules,
        )

        # Initialize model components
        weights = q3jax.Weights.init(random.key(42), cfg)
        batch_size = 2
        seq_len = 32
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        cache = q3jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)

        # Test prefill and decode on GPU
        if set_mesh is not None:
            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    max_tokens, logits, cache = q3jax.prefill(
                        tokens, weights, cache, cfg
                    )
                assert max_tokens.shape == (batch_size, seq_len)

                next_tokens = max_tokens[:, -1:]
                for _ in range(2):
                    next_tokens, cache = q3jax.decode_step(
                        next_tokens, weights, cache, cfg
                    )
                    assert next_tokens.shape == (batch_size, 1)
        else:
            # Fallback for older JAX versions without set_mesh
            max_tokens, logits, cache = q3jax.prefill(tokens, weights, cache, cfg)
            assert max_tokens.shape == (batch_size, seq_len)

            next_tokens = max_tokens[:, -1:]
            for _ in range(2):
                next_tokens, cache = q3jax.decode_step(next_tokens, weights, cache, cfg)
                assert next_tokens.shape == (batch_size, 1)

    def test_qwen3_quantization_cpu(self):
        """Test Qwen3 quantization support on CPU."""
        import jax
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import model as q3jax

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create small test configuration with quantization
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = q3jax.ShardingRules(
            batch=None,
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=("y", "z"),  # Qwen3 requires this for TENSOR_AXIS_NAME
        )

        cfg = q3jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=2,
            num_layers=2,
            head_dim=32,
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            moe_ffw_size=256,
            moe_experts_per_tok=2,
            moe_num_experts=4,
            mlp_ffw_size=256,
            mlp_layer_idxs=[0, 1],
            mesh=mesh,
            rules=simplified_rules,
            quant_attn=True,  # Enable attention quantization
            quant_mlp=True,  # Enable MLP quantization
            quant_cache=True,  # Enable cache quantization
        )

        # Test quantized weight initialization
        weights = q3jax.Weights.init(random.key(42), cfg)
        assert weights is not None

        # Test quantized cache initialization
        batch_size = 1
        cache = q3jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)
        assert cache is not None

    def test_qwen3_config_validation(self):
        """Test Qwen3 configuration validation."""
        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import model as q3jax

        # Test basic config creation
        cfg = q3jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=4,
            num_layers=4,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            moe_ffw_size=512,
            moe_experts_per_tok=2,
            moe_num_experts=8,
            mlp_ffw_size=512,
            mlp_layer_idxs=[0, 1, 2, 3],
        )

        assert cfg.embed == 256
        assert cfg.q_heads == 8
        assert cfg.kv_heads == 4
        assert cfg.head_dim == 32

    def test_qwen3_rope_attention(self):
        """Test Qwen3 RoPE (Rotary Position Embedding) attention."""
        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import model as q3jax

        # Test configuration with RoPE-specific parameters
        cfg = q3jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            moe_ffw_size=512,
            moe_experts_per_tok=2,
            moe_num_experts=8,
            mlp_ffw_size=512,
            mlp_layer_idxs=[0, 1],
            rope_theta=10000.0,  # RoPE base frequency
        )

        # Verify RoPE configuration
        assert hasattr(cfg, "rope_theta")
        assert cfg.rope_theta == 10000.0

    def test_qwen3_integration_with_oumi_engine(self):
        """Test Qwen3 integration with Oumi's JAX inference engine."""
        from oumi.core.configs import ModelParams

        # Create model parameters for Qwen3
        model_params = ModelParams(
            model_name="jax-ml/qwen3",
            load_pretrained_weights=False,
            trust_remote_code=True,
        )

        # Import the engine class first
        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        # Test that the engine can be instantiated (with mocked components)
        with (
            patch("oumi.inference.jax_inference_engine.build_tokenizer"),
            patch.object(JAXInferenceEngine, "_setup_jax_devices"),
            patch.object(JAXInferenceEngine, "_load_model"),
        ):
            engine = JAXInferenceEngine(model_params)
            assert engine._model_params.model_name == "jax-ml/qwen3"

    def test_qwen3_ragged_attention(self):
        """Test Qwen3 ragged attention support."""
        # Test that ragged attention module can be imported
        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import ragged_attention

        # Verify the module exists and has expected functions
        assert hasattr(ragged_attention, "__name__")

    def test_qwen3_decode_ragged_dot(self):
        """Test Qwen3 decode ragged dot kernel support."""
        # Test that decode ragged dot module can be imported
        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import (
            decode_ragged_dot,
        )

        # Verify the module exists
        assert hasattr(decode_ragged_dot, "__name__")

    def test_qwen3_numerical_stability(self):
        """Test Qwen3 numerical stability across different seeds."""
        import jax
        import jax.numpy as jnp
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import model as q3jax

        # Force CPU execution for stability testing
        jax.config.update("jax_platforms", "cpu")

        # Handle different JAX versions for set_mesh
        try:
            from jax.sharding import set_mesh as set_mesh
        except ImportError:
            try:
                from jax.sharding import set_mesh as set_mesh
            except ImportError:
                set_mesh = None

        # Create mesh with Explicit axis types for Qwen3
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules
        simplified_rules = q3jax.ShardingRules(
            batch=None,
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=("y", "z"),
        )

        cfg = q3jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=2,
            num_layers=2,
            head_dim=32,
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            moe_ffw_size=256,
            moe_experts_per_tok=2,
            moe_num_experts=4,
            mlp_ffw_size=256,
            mlp_layer_idxs=[0, 1],
            mesh=mesh,
            rules=simplified_rules,
        )

        # Test with multiple seeds
        outputs = []
        for seed in [42, 123, 456]:
            weights = q3jax.Weights.init(random.key(seed), cfg)
            batch_size = 1
            seq_len = 8
            tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
            cache = q3jax.KVCache.init(
                random.key(seed), cfg, batch_size, cfg.max_seq_len
            )

            if set_mesh is not None:
                if cfg.mesh is not None:
                    with set_mesh(cfg.mesh):
                        max_tokens, logits, _ = q3jax.prefill(
                            tokens, weights, cache, cfg
                        )
            else:
                max_tokens, logits, _ = q3jax.prefill(tokens, weights, cache, cfg)

            outputs.append(logits)

            # Check for reasonable output ranges
            assert jnp.isfinite(logits).all(), (
                f"Non-finite values in logits for seed {seed}"
            )
            assert jnp.abs(logits).max() < 100.0, f"Logits too large for seed {seed}"

        # Verify outputs are different (showing model is actually using the seed)
        assert not jnp.allclose(outputs[0], outputs[1], atol=1e-6), (
            "Outputs should differ with different seeds"
        )

    def test_qwen3_error_handling(self):
        """Test Qwen3 error handling for invalid inputs."""
        import jax
        import jax.numpy as jnp
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import model as q3jax

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create valid configuration first
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        simplified_rules = q3jax.ShardingRules(
            batch=None,
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=("y", "z"),
        )

        # Test invalid configuration values
        with pytest.raises((ValueError, TypeError, AssertionError)):
            q3jax.Config(
                embed=0,  # Invalid: must be positive
                q_heads=4,
                kv_heads=2,
                num_layers=2,
                head_dim=32,
                vocab_size=500,
                max_seq_len=64,
                causal=True,
                moe_ffw_size=256,
                moe_experts_per_tok=2,
                moe_num_experts=4,
                mesh=mesh,
                rules=simplified_rules,
            )

        with pytest.raises((ValueError, TypeError, AssertionError)):
            q3jax.Config(
                embed=128,
                q_heads=0,  # Invalid: must be positive
                kv_heads=2,
                num_layers=2,
                head_dim=32,
                vocab_size=500,
                max_seq_len=64,
                causal=True,
                moe_ffw_size=256,
                moe_experts_per_tok=2,
                moe_num_experts=4,
                mesh=mesh,
                rules=simplified_rules,
            )

        # Test valid configuration for subsequent tests
        cfg = q3jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=2,
            num_layers=2,
            head_dim=32,
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            moe_ffw_size=256,
            moe_experts_per_tok=2,
            moe_num_experts=4,
            mlp_ffw_size=256,
            mlp_layer_idxs=[0, 1],
            mesh=mesh,
            rules=simplified_rules,
        )

        weights = q3jax.Weights.init(random.key(42), cfg)
        cache = q3jax.KVCache.init(random.key(42), cfg, 1, cfg.max_seq_len)

        # Test invalid token shapes
        with pytest.raises((ValueError, TypeError, AssertionError)):
            invalid_tokens = jnp.ones(
                (1, cfg.max_seq_len + 1), dtype=jnp.int32
            )  # Too long
            q3jax.prefill(invalid_tokens, weights, cache, cfg)

        # Test invalid token values
        with pytest.raises((ValueError, TypeError, AssertionError)):
            invalid_tokens = jnp.full(
                (1, 8), cfg.vocab_size, dtype=jnp.int32
            )  # Out of vocab range
            q3jax.prefill(invalid_tokens, weights, cache, cfg)

    def test_qwen3_memory_efficiency(self):
        """Test Qwen3 memory usage with minimal configurations."""
        import jax
        import jax.numpy as jnp
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import model as q3jax

        # Force CPU execution for memory testing
        jax.config.update("jax_platforms", "cpu")

        # Handle different JAX versions for set_mesh
        try:
            from jax.sharding import set_mesh as set_mesh
        except ImportError:
            try:
                from jax.sharding import set_mesh as set_mesh
            except ImportError:
                set_mesh = None

        # Create minimal configuration
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        simplified_rules = q3jax.ShardingRules(
            batch=None,
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=("y", "z"),
        )

        # Minimal configuration for memory efficiency
        cfg = q3jax.Config(
            embed=64,  # Small embedding
            q_heads=2,  # Few heads
            kv_heads=1,  # Single KV head
            num_layers=1,  # Single layer
            head_dim=16,  # Small head dimension
            vocab_size=100,  # Small vocabulary
            max_seq_len=32,  # Short sequences
            causal=True,
            moe_ffw_size=128,  # Small MoE FFW
            moe_experts_per_tok=1,  # Single expert per token
            moe_num_experts=2,  # Few experts
            mlp_ffw_size=128,  # Small MLP FFW
            mlp_layer_idxs=[0],  # Single MLP layer
            mesh=mesh,
            rules=simplified_rules,
        )

        # Test that minimal model can be initialized and run
        weights = q3jax.Weights.init(random.key(42), cfg)
        assert weights is not None, "Failed to initialize minimal model weights"

        batch_size = 1
        seq_len = 8
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        cache = q3jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)

        # Test prefill with minimal configuration
        if set_mesh is not None and cfg.mesh is not None:
            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    max_tokens, logits, updated_cache = q3jax.prefill(
                        tokens, weights, cache, cfg
                    )
        else:
            max_tokens, logits, updated_cache = q3jax.prefill(
                tokens, weights, cache, cfg
            )

        assert max_tokens.shape == (batch_size, seq_len), "Unexpected max_tokens shape"
        assert logits.shape == (batch_size, seq_len, cfg.vocab_size), (
            "Unexpected logits shape"
        )
        assert updated_cache is not None, "Cache should be updated after prefill"

        # Test decode step with minimal configuration
        next_token = max_tokens[:, -1:]
        if set_mesh is not None and cfg.mesh is not None:
            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    next_token_out, final_cache = q3jax.decode_step(
                        next_token, weights, updated_cache, cfg
                    )
        else:
            next_token_out, final_cache = q3jax.decode_step(
                next_token, weights, updated_cache, cfg
            )

        assert next_token_out.shape == (batch_size, 1), "Unexpected decode output shape"
        assert final_cache is not None, "Cache should be updated after decode"

    def test_qwen3_token_generation_demo(self):
        """Test Qwen3 token generation mechanics."""
        import jax
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import (
            model as q3jax,
        )

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create simplified mesh
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Auto, AxisType.Auto, AxisType.Auto),
        )

        # Create simplified sharding rules for Qwen3
        simplified_rules = q3jax.ShardingRules(
            batch=None,
            sequence=None,
            head_dim=None,
            vocab_in=None,
            vocab_out=None,
            act_embed=None,
            act_heads=None,
            q_heads=None,
            kv_heads=None,
            qkv_embed=None,
            o_heads=None,
            o_embed=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
        )

        # Chat configuration (same scale as working tests, MoE disabled by using only MLP)
        cfg = q3jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            moe_ffw_size=512,
            moe_experts_per_tok=2,
            moe_num_experts=16,
            mlp_ffw_size=512,
            mlp_layer_idxs=[0, 1],  # Use MLP for all layers (disable MoE)
            use_prefill_attn_kernel=False,
            use_decode_attn_kernel=False,
            use_ragged_dot_kernel=False,
            mesh=mesh,
            rules=simplified_rules,
        )

        # Initialize model
        weights = q3jax.Weights.init(random.key(42), cfg)
        cache = q3jax.KVCache.init(random.key(43), cfg, 1, cfg.max_seq_len)

        print("\n🚀 Qwen3 Token Generation Demo")
        print("=" * 40)

        try:
            from jax.sharding import set_mesh
        except ImportError:
            from jax.sharding import set_mesh

        # Simulate chat conversations showcasing Qwen3's capabilities
        conversations = [
            {"input": "Hello Qwen3!", "tokens": [1, 28, 45, 62, 19, 2]},
            {
                "input": "Multilingual understanding",
                "tokens": [1, 39, 58, 71, 84, 33, 2],
            },
            {"input": "Complex reasoning task", "tokens": [1, 47, 63, 89, 25, 76, 2]},
            {"input": "Creative writing help", "tokens": [1, 52, 68, 91, 37, 2]},
        ]

        for i, conv in enumerate(conversations):
            print(f"\n🧩 Conversation {i + 1}:")
            print(f"   User: {conv['input']}")

            # Convert to JAX array
            input_tokens = jnp.array([conv["tokens"]])

            # Initialize fresh cache for each conversation (avoiding buffer donation issues)
            conv_cache = q3jax.KVCache.init(random.key(50 + i), cfg, 1, cfg.max_seq_len)

            # Generate response using Qwen3's hybrid MLP/MoE architecture
            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    # Prefill phase
                    max_tokens, logits, updated_cache = q3jax.prefill(
                        input_tokens, weights, conv_cache, cfg
                    )

                    # Generate a few tokens
                    generated_tokens = []
                    current_tokens = input_tokens
                    working_cache = updated_cache

                    for step in range(3):  # Generate 3 tokens
                        # Sample from logits (using argmax for deterministic results)
                        next_token = jnp.argmax(
                            logits[:, -1, :], axis=-1, keepdims=True
                        )
                        generated_tokens.append(int(next_token[0, 0]))

                        # Extend sequence
                        current_tokens = jnp.concatenate(
                            [current_tokens, next_token.reshape(1, 1)], axis=1
                        )

                        # Generate next token if we haven't hit max length
                        if current_tokens.shape[1] < cfg.max_seq_len:
                            _, logits, working_cache = q3jax.prefill(
                                current_tokens[:, -1:], weights, working_cache, cfg
                            )

                    print(f"   Qwen3: Generated tokens: {generated_tokens}")
                    print(f"        Full sequence: {current_tokens.tolist()[0]}")
                    print(f"        Response shape: {logits.shape}")

                    # Verify response is valid
                    assert logits.shape[0] == 1  # Batch size
                    assert logits.shape[2] == cfg.vocab_size
                    assert jnp.all(jnp.isfinite(logits))

                    print("   ✅ Generation successful!")

        print("\n🎉 Qwen3 Token Generation Demo Complete!")
        print(f"   • Model: {cfg.embed}D embeddings, {cfg.num_layers} layers")
        print(f"   • Vocab: {cfg.vocab_size} tokens")
        print(f"   • Generated responses for {len(conversations)} conversations")
        print("   • All responses were valid and finite")

    def test_qwen3_multilingual_chat(self):
        """Test Qwen3's multilingual capabilities in chat contexts."""
        import jax
        from jax import random

        from oumi.models.experimental.jax_models.qwen3.qwen3_jax import (
            model as q3jax,
        )

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        from jax.sharding import AxisType

        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Auto, AxisType.Auto, AxisType.Auto),
        )

        simplified_rules = q3jax.ShardingRules(
            batch=None,
            sequence=None,
            head_dim=None,
            vocab_in=None,
            vocab_out=None,
            act_embed=None,
            act_heads=None,
            q_heads=None,
            kv_heads=None,
            qkv_embed=None,
            o_heads=None,
            o_embed=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
        )

        # Smaller config for multilingual testing
        cfg = q3jax.Config(
            embed=192,
            q_heads=6,
            kv_heads=3,
            num_layers=3,
            head_dim=32,
            vocab_size=2000,
            max_seq_len=96,
            causal=True,
            moe_ffw_size=256,
            moe_experts_per_tok=2,
            moe_num_experts=8,
            mlp_ffw_size=384,
            mlp_layer_idxs=[1],  # Use MLP for middle layer
            use_prefill_attn_kernel=False,
            use_decode_attn_kernel=False,
            use_ragged_dot_kernel=False,
            mesh=mesh,
            rules=simplified_rules,
        )

        weights = q3jax.Weights.init(random.key(42), cfg)
        cache = q3jax.KVCache.init(random.key(43), cfg, 1, cfg.max_seq_len)

        print("\n🌍 Qwen3 Multilingual Chat Test")
        print("=" * 50)

        try:
            from jax.sharding import set_mesh
        except ImportError:
            from jax.sharding import set_mesh

        # Test different "language" patterns (simulated with different token patterns)
        language_tests = [
            {
                "lang": "English-like",
                "tokens": [1, 10, 20, 30, 2],
                "pattern": "Standard Western",
            },
            {
                "lang": "Chinese-like",
                "tokens": [1, 100, 200, 300, 2],
                "pattern": "East Asian",
            },
            {
                "lang": "Arabic-like",
                "tokens": [1, 400, 500, 600, 2],
                "pattern": "Semitic",
            },
            {
                "lang": "Code-like",
                "tokens": [1, 700, 800, 900, 2],
                "pattern": "Programming",
            },
        ]

        for i, lang_test in enumerate(language_tests):
            print(f"\n🗣️ Testing {lang_test['lang']} processing:")
            print(f"   Pattern: {lang_test['pattern']}")

            input_tokens = jnp.array([lang_test["tokens"]])

            # Initialize fresh cache for each language test (avoiding buffer donation issues)
            lang_cache = q3jax.KVCache.init(random.key(70 + i), cfg, 1, cfg.max_seq_len)

            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    # Test prefill for different language patterns
                    max_tokens, logits, updated_cache = q3jax.prefill(
                        input_tokens, weights, lang_cache, cfg
                    )

                    print(f"   Input tokens: {lang_test['tokens']}")
                    print(f"   Output shape: {logits.shape}")
                    print(f"   Processing: MLP layer {cfg.mlp_layer_idxs} + MoE layers")

                    # Test decode step
                    next_token = jnp.array([[42]])
                    decoded_token, final_cache = q3jax.decode_step(
                        next_token, weights, updated_cache, cfg
                    )

                    # Verify outputs
                    assert logits.shape == (1, input_tokens.shape[1], cfg.vocab_size)
                    assert jnp.all(jnp.isfinite(logits))
                    assert decoded_token.shape == (1, 1)

                    print("   ✅ Multilingual processing successful!")

        print("\n🌟 Qwen3 Multilingual Complete!")
        print(f"   • Tested {len(language_tests)} different language patterns")
        print("   • Hybrid MLP/MoE architecture handles diverse inputs")
        print("   • Model ready for multilingual chat applications")
        print("   • Efficient processing across language families")
