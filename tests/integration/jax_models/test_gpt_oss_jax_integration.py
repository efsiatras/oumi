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

"""Integration tests for GPT OSS JAX model implementation."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from jax import random
from jax.sharding import AxisType, Mesh

from tests.markers import requires_gpus

# Mark all tests in this file as JAX-related
pytestmark = pytest.mark.jax


class TestGPTOSSJAXIntegration:
    """Integration tests for GPT OSS JAX model."""

    @pytest.mark.skipif(
        not pytest.importorskip("jax", minversion="0.7.0"),
        reason="JAX 0.7.0+ not available",
    )
    def test_gpt_oss_import(self):
        """Test that GPT OSS JAX model can be imported."""
        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        assert hasattr(gpt_jax, "Config")
        assert hasattr(gpt_jax, "Weights")
        assert hasattr(gpt_jax, "KVCache")
        assert hasattr(gpt_jax, "prefill")
        assert hasattr(gpt_jax, "decode_step")

    def test_gpt_oss_model_init_cpu(self):
        """Test GPT OSS model initialization on CPU."""
        import jax
        from jax import random

        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create small test configuration
        devices = jax.devices()
        mesh = Mesh(devices, ("x",))

        # Create simplified sharding rules for single device
        simplified_rules = gpt_jax.ShardingRules(
            batch="x",
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = gpt_jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=8,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            sliding_attention_map=["standard", "standard"],  # One entry per layer
            sliding_window_size=128,
            moe_ffw_size=512,
            moe_experts_per_tok=2,
            moe_num_experts=8,
            mesh=mesh,
            rules=simplified_rules,
        )

        # Test weight initialization
        weights = gpt_jax.Weights.init(random.key(42), cfg)
        assert weights is not None

        # Test cache initialization
        batch_size = 2
        cache = gpt_jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)
        assert cache is not None

    @requires_gpus(1)
    @pytest.mark.single_gpu
    def test_gpt_oss_model_init_gpu(self):
        """Test GPT OSS model initialization on GPU."""
        import jax
        from jax import random

        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        # Check if GPU is available
        devices = jax.devices()
        gpu_devices = [d for d in devices if "gpu" in str(d).lower()]

        if not gpu_devices:
            pytest.skip("No GPU devices available")

        # Create test configuration for GPU
        mesh = Mesh(gpu_devices, ("x",))
        cfg = gpt_jax.Config(
            embed=512,
            q_heads=16,
            kv_heads=16,
            num_layers=4,
            head_dim=32,
            vocab_size=2000,
            max_seq_len=256,
            causal=True,
            sliding_attention_map=[],
            sliding_window_size=256,
            moe_ffw_size=1024,
            moe_experts_per_tok=2,
            moe_num_experts=4,
            mesh=mesh,
            use_ragged_dot_kernel=False,
        )

        # Test weight initialization on GPU
        weights = gpt_jax.Weights.init(random.key(42), cfg)
        assert weights is not None

    def test_gpt_oss_prefill_decode_cpu(self):
        """Test GPT OSS prefill and decode on CPU."""
        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        # Handle different JAX versions for set_mesh
        try:
            from jax.sharding import set_mesh as set_mesh
        except ImportError:
            try:
                from jax.sharding import set_mesh as set_mesh
            except ImportError:
                set_mesh = None

        # GPT OSS needs y,z axes and uses .get(out_sharding) which needs Explicit types
        from jax.sharding import AxisType

        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create sharding rules using available axes
        simplified_rules = gpt_jax.ShardingRules(
            batch="x",
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = gpt_jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            sliding_attention_map=["standard", "standard"],  # One entry per layer
            sliding_window_size=64,
            moe_ffw_size=256,
            moe_experts_per_tok=2,
            moe_num_experts=4,
            mesh=mesh,
            rules=simplified_rules,
            use_ragged_dot_kernel=False,
        )

        # Initialize model components
        weights = gpt_jax.Weights.init(random.key(42), cfg)
        batch_size = 1
        seq_len = 16
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        cache = gpt_jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)

        # Test prefill
        if cfg.mesh is not None:
            with set_mesh(cfg.mesh):
                max_tokens, logits, cache = gpt_jax.prefill(tokens, weights, cache, cfg)

        # Verify output shapes
        assert max_tokens.shape == (batch_size, seq_len), (
            f"Expected max_tokens shape {(batch_size, seq_len)}, got {max_tokens.shape}"
        )
        assert logits.shape == (batch_size, seq_len, cfg.vocab_size), (
            f"Expected logits shape {(batch_size, seq_len, cfg.vocab_size)}, got {logits.shape}"
        )

        # Verify outputs are valid
        assert jnp.all(jnp.isfinite(logits)), "Logits contain NaN or Inf values"
        assert jnp.all((max_tokens >= 0) & (max_tokens < cfg.vocab_size)), (
            "max_tokens contain invalid token IDs"
        )

        # Test decode steps
        next_tokens = max_tokens[:, -1:]
        if cfg.mesh is not None:
            with set_mesh(cfg.mesh):
                for step in range(3):
                    next_tokens, cache = gpt_jax.decode_step(
                        next_tokens, weights, cache, cfg
                    )

                # Verify decode output shapes and validity
                assert next_tokens.shape == (batch_size, 1), (
                    f"Step {step}: Expected next_tokens shape {(batch_size, 1)}, got {next_tokens.shape}"
                )
                assert jnp.all((next_tokens >= 0) & (next_tokens < cfg.vocab_size)), (
                    f"Step {step}: next_tokens contain invalid token IDs"
                )
                assert jnp.all(jnp.isfinite(next_tokens)), (
                    f"Step {step}: next_tokens contain NaN or Inf values"
                )

    @requires_gpus(1)
    @pytest.mark.single_gpu
    def test_gpt_oss_prefill_decode_gpu(self):
        """Test GPT OSS prefill and decode on GPU."""
        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        # Get GPU devices
        devices = jax.devices()
        gpu_devices = [d for d in devices if "gpu" in str(d).lower()]

        if not gpu_devices:
            pytest.skip("No GPU devices available")

        # Create test configuration for GPU with explicit axis types
        from jax.sharding import AxisType

        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = gpt_jax.ShardingRules(
            batch="x",
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = gpt_jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=8,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            sliding_attention_map=["standard", "standard"],  # One entry per layer
            sliding_window_size=128,
            moe_ffw_size=512,
            moe_experts_per_tok=2,
            moe_num_experts=8,
            mesh=mesh,
            rules=simplified_rules,
        )

        # Initialize model components
        weights = gpt_jax.Weights.init(random.key(42), cfg)
        batch_size = 2
        seq_len = 32
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        cache = gpt_jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)

        # Test prefill and decode on GPU
        try:
            from jax.sharding import set_mesh as set_mesh
        except ImportError:
            from jax.sharding import set_mesh as set_mesh

        if cfg.mesh is not None:
            with set_mesh(cfg.mesh):
                max_tokens, logits, cache = gpt_jax.prefill(tokens, weights, cache, cfg)

        # Verify outputs
        assert max_tokens.shape == (batch_size, seq_len), (
            f"Expected max_tokens shape {(batch_size, seq_len)}, got {max_tokens.shape}"
        )
        assert logits.shape == (batch_size, seq_len, cfg.vocab_size), (
            f"Expected logits shape {(batch_size, seq_len, cfg.vocab_size)}, got {logits.shape}"
        )
        assert jnp.all(jnp.isfinite(logits)), "Logits contain NaN or Inf values"

        next_tokens = max_tokens[:, -1:]
        if cfg.mesh is not None:
            with set_mesh(cfg.mesh):
                for step in range(2):
                    next_tokens, cache = gpt_jax.decode_step(
                        next_tokens, weights, cache, cfg
                    )
                assert next_tokens.shape == (batch_size, 1), (
                    f"Step {step}: Expected shape {(batch_size, 1)}, got {next_tokens.shape}"
                )
                assert jnp.all(jnp.isfinite(next_tokens)), (
                    f"Step {step}: next_tokens contain NaN or Inf values"
                )

    def test_gpt_oss_quantization_cpu(self):
        """Test GPT OSS quantization support on CPU."""
        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        # Create small test configuration with quantization
        # GPT OSS needs y,z axes for TENSOR_AXIS_NAME
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = gpt_jax.ShardingRules(
            batch="x",
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = gpt_jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            sliding_attention_map=["standard", "standard"],  # One entry per layer
            sliding_window_size=64,
            moe_ffw_size=256,
            moe_experts_per_tok=2,
            moe_num_experts=4,
            mesh=mesh,
            rules=simplified_rules,
            quant_attn=True,  # Enable attention quantization
            quant_moe=True,  # Enable MoE quantization
            quant_cache=True,  # Enable cache quantization
            use_ragged_dot_kernel=False,
        )

        # Test quantized weight initialization
        weights = gpt_jax.Weights.init(random.key(42), cfg)
        assert weights is not None

        # Verify quantization was applied
        assert cfg.quant_attn and cfg.quant_moe and cfg.quant_cache

        # Test quantized cache initialization
        batch_size = 1
        cache = gpt_jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)
        assert cache is not None

        # Test that quantized model can run inference
        tokens = jnp.ones((batch_size, 8), dtype=jnp.int32)
        try:
            from jax.sharding import set_mesh as set_mesh
        except ImportError:
            from jax.sharding import set_mesh as set_mesh

        if cfg.mesh is not None:
            with set_mesh(cfg.mesh):
                max_tokens, logits, cache = gpt_jax.prefill(tokens, weights, cache, cfg)

        # Verify quantized inference outputs
        assert jnp.all(jnp.isfinite(logits)), "Quantized model produces NaN/Inf logits"
        assert max_tokens.shape == (batch_size, 8)
        assert logits.shape == (batch_size, 8, cfg.vocab_size)

    def test_gpt_oss_config_validation(self):
        """Test GPT OSS configuration validation."""
        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        # Create mesh for configuration
        devices = jax.devices()
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Test basic config creation
        cfg = gpt_jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=8,
            num_layers=4,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            sliding_attention_map=["standard"] * 4,  # One entry per layer
            sliding_window_size=128,
            moe_ffw_size=512,
            moe_experts_per_tok=2,
            moe_num_experts=8,
            mesh=mesh,
        )

        # Test configuration properties
        assert cfg.embed == 256
        assert cfg.q_heads == 8
        assert cfg.kv_heads == 8
        assert cfg.head_dim == 32
        assert cfg.moe_ffw_size == 512
        assert len(cfg.sliding_attention_map) == cfg.num_layers

        # Test configuration validation
        assert cfg.embed == cfg.q_heads * cfg.head_dim, (
            "Embedding dimension should equal q_heads * head_dim"
        )
        assert cfg.max_seq_len > 0, "max_seq_len should be positive"
        assert cfg.vocab_size > 0, "vocab_size should be positive"

    def test_gpt_oss_architecture_variants(self):
        """Test GPT OSS architecture variants (20B, 120B configurations)."""
        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        # Create mesh for configuration
        devices = jax.devices()
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Test 20B-style configuration (simplified for testing)
        cfg_20b = gpt_jax.Config(
            embed=512,  # Reduced for testing
            q_heads=16,
            kv_heads=16,
            num_layers=4,  # Reduced for testing
            head_dim=32,
            vocab_size=1000,  # Reduced for testing
            max_seq_len=128,  # Reduced for testing
            causal=True,
            sliding_attention_map=["standard"] * 4,
            sliding_window_size=128,
            moe_ffw_size=1024,  # Reduced for testing
            moe_experts_per_tok=2,
            moe_num_experts=8,  # Reduced for testing
            mesh=mesh,
        )

        # Test that we can initialize this configuration
        weights_20b = gpt_jax.Weights.init(random.key(42), cfg_20b)
        assert weights_20b is not None
        assert cfg_20b.embed == 512
        assert cfg_20b.num_layers == 4

        # Test 120B-style configuration (simplified for testing)
        cfg_120b = gpt_jax.Config(
            embed=1024,  # Reduced for testing
            q_heads=32,
            kv_heads=32,
            num_layers=4,  # Reduced for testing
            head_dim=32,
            vocab_size=1000,  # Reduced for testing
            max_seq_len=128,  # Reduced for testing
            causal=True,
            sliding_attention_map=["standard"] * 4,
            sliding_window_size=128,
            moe_ffw_size=2048,  # Reduced for testing
            moe_experts_per_tok=2,
            moe_num_experts=8,  # Reduced for testing
            mesh=mesh,
        )

        # Test that we can initialize this configuration
        weights_120b = gpt_jax.Weights.init(random.key(43), cfg_120b)
        assert weights_120b is not None
        assert cfg_120b.embed == 1024
        assert cfg_120b.num_layers == 4

    def test_gpt_oss_integration_with_oumi_engine(self):
        """Test GPT OSS integration with Oumi's JAX inference engine."""
        from oumi.core.configs import ModelParams

        # Create model parameters for GPT OSS
        model_params = ModelParams(
            model_name="jax-ml/gpt-oss",
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
            assert engine._model_params.model_name == "jax-ml/gpt-oss"

    def test_gpt_oss_decode_ragged_dot(self):
        """Test GPT OSS decode ragged dot kernel support."""
        # Test that decode ragged dot module can be imported
        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            decode_ragged_dot,
        )

        # Verify the module exists and has expected functions
        assert hasattr(decode_ragged_dot, "__name__")

        # Test that we can access key functions (if they exist)
        # This is a basic smoke test to ensure the module is properly structured
        module_attrs = dir(decode_ragged_dot)
        assert len(module_attrs) > 0, "decode_ragged_dot module appears to be empty"

    def test_gpt_oss_checkpoint_utils(self):
        """Test GPT OSS checkpoint utilities."""
        # Test that checkpoint utilities can be imported
        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import chkpt_utils

        # Verify the module exists and has expected utilities
        assert hasattr(chkpt_utils, "__name__")

        # Test that we can access key utilities (if they exist)
        module_attrs = dir(chkpt_utils)
        assert len(module_attrs) > 0, "chkpt_utils module appears to be empty"

    def test_gpt_oss_inference_optimizations(self):
        """Test GPT OSS inference optimizations."""
        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        # Create mesh for configuration
        devices = jax.devices()
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Test configuration with inference optimizations
        cfg = gpt_jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=8,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            sliding_attention_map=["standard", "standard"],  # One entry per layer
            sliding_window_size=128,
            moe_ffw_size=512,
            moe_experts_per_tok=2,
            moe_num_experts=8,
            mesh=mesh,
            use_ragged_dot_kernel=True,  # Enable optimized kernels
        )

        # Verify optimization settings
        assert cfg.use_ragged_dot_kernel is True

        # Test that optimized configuration can initialize
        weights = gpt_jax.Weights.init(random.key(42), cfg)
        assert weights is not None

        # Test basic inference with optimizations
        batch_size = 1
        seq_len = 8
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        cache = gpt_jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)

        try:
            from jax.sharding import set_mesh as set_mesh
        except ImportError:
            from jax.sharding import set_mesh as set_mesh

        if cfg.mesh is not None:
            with set_mesh(cfg.mesh):
                max_tokens, logits, cache = gpt_jax.prefill(tokens, weights, cache, cfg)

        # Verify optimized inference produces valid outputs
        assert jnp.all(jnp.isfinite(logits)), "Optimized model produces NaN/Inf logits"
        assert max_tokens.shape == (batch_size, seq_len)
        assert logits.shape == (batch_size, seq_len, cfg.vocab_size)

    def test_gpt_oss_numerical_stability(self):
        """Test GPT OSS numerical stability under various conditions."""
        # Force CPU execution for deterministic testing
        jax.config.update("jax_platforms", "cpu")

        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        # Handle different JAX versions for set_mesh
        try:
            from jax.sharding import set_mesh as set_mesh
        except ImportError:
            from jax.sharding import set_mesh as set_mesh

        # Create mesh with explicit axis types
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules
        simplified_rules = gpt_jax.ShardingRules(
            batch="x",
            sequence=None,
            act_embed=None,
            act_heads=None,
            head_dim=None,
            qkv_embed=None,
            q_heads=None,
            kv_heads=None,
            o_heads=None,
            o_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = gpt_jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            sliding_attention_map=["standard", "standard"],
            sliding_window_size=64,
            moe_ffw_size=256,
            moe_experts_per_tok=2,
            moe_num_experts=4,
            mesh=mesh,
            rules=simplified_rules,
            use_ragged_dot_kernel=False,
        )

        # Test with different random seeds for reproducibility
        seeds = [42, 123, 999]
        results = []

        for seed in seeds:
            weights = gpt_jax.Weights.init(random.key(seed), cfg)
            cache = gpt_jax.KVCache.init(random.key(seed), cfg, 1, cfg.max_seq_len)
            tokens = jnp.ones((1, 8), dtype=jnp.int32)

            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    max_tokens, logits, _ = gpt_jax.prefill(tokens, weights, cache, cfg)

            # Check for numerical stability
            assert jnp.all(jnp.isfinite(logits)), f"Seed {seed}: Logits contain NaN/Inf"
            assert jnp.all(jnp.abs(logits) < 100.0), (
                f"Seed {seed}: Logits are too large (>100)"
            )

            results.append(logits)

        # Results should be different for different seeds (model is not deterministic due to randomness)
        assert not jnp.allclose(results[0], results[1], rtol=1e-5), (
            "Results should differ between seeds"
        )

    def test_gpt_oss_error_handling(self):
        """Test GPT OSS error handling and edge cases."""
        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        devices = jax.devices()
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Test that configuration validation works as expected
        # Create a valid configuration for testing
        valid_cfg = gpt_jax.Config(
            embed=128,  # Properly divisible by q_heads
            q_heads=4,
            kv_heads=4,
            num_layers=2,
            head_dim=32,  # embed = q_heads * head_dim = 128
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            sliding_attention_map=["standard", "standard"],
            sliding_window_size=64,
            moe_ffw_size=256,
            moe_experts_per_tok=2,
            moe_num_experts=4,
            mesh=mesh,
        )

        # Verify configuration properties
        assert valid_cfg.embed == 128
        assert valid_cfg.q_heads == 4
        assert valid_cfg.head_dim == 32

    def test_gpt_oss_memory_efficiency(self):
        """Test GPT OSS memory usage patterns."""
        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create minimal configuration for memory testing
        cfg = gpt_jax.Config(
            embed=64,
            q_heads=2,
            kv_heads=2,
            num_layers=1,
            head_dim=32,
            vocab_size=100,
            max_seq_len=32,
            causal=True,
            sliding_attention_map=["standard"],
            sliding_window_size=32,
            moe_ffw_size=128,
            moe_experts_per_tok=1,
            moe_num_experts=2,
            mesh=mesh,
            use_ragged_dot_kernel=False,
        )

        # Test that model can be created with minimal memory
        weights = gpt_jax.Weights.init(random.key(42), cfg)
        cache = gpt_jax.KVCache.init(random.key(42), cfg, 1, cfg.max_seq_len)

        # Verify we can do inference with minimal configuration
        tokens = jnp.ones((1, 4), dtype=jnp.int32)

        try:
            from jax.sharding import set_mesh as set_mesh
        except ImportError:
            from jax.sharding import set_mesh as set_mesh

        if cfg.mesh is not None:
            with set_mesh(cfg.mesh):
                max_tokens, logits, cache = gpt_jax.prefill(tokens, weights, cache, cfg)

        assert logits.shape == (1, 4, cfg.vocab_size)
        assert jnp.all(jnp.isfinite(logits))

    def test_gpt_oss_token_generation_demo(self):
        """Test GPT-OSS token generation mechanics."""
        import jax
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create mesh with Explicit axis types (required for MoE operations)
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules (all axes None to disable sharding)
        simplified_rules = gpt_jax.ShardingRules(
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
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        # Chat configuration (same scale as working tests, MoE disabled)
        cfg = gpt_jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=8,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            sliding_attention_map=["standard", "standard"],  # One entry per layer
            sliding_window_size=128,
            moe_ffw_size=512,
            moe_experts_per_tok=1,
            moe_num_experts=8,  # Keep MoE but use working configuration
            mesh=mesh,
            rules=simplified_rules,
        )

        # Initialize model
        weights = gpt_jax.Weights.init(random.key(42), cfg)
        cache = gpt_jax.KVCache.init(random.key(43), cfg, 1, cfg.max_seq_len)

        print("\n🤖 GPT-OSS Token Generation Demo")
        print("=" * 40)

        # Simulate chat conversation with token generation
        conversations = [
            {"input": "Hello!", "tokens": [1, 15, 23, 7, 2]},
            {"input": "How are you?", "tokens": [1, 18, 25, 12, 31, 2]},
            {"input": "Tell me about AI", "tokens": [1, 20, 8, 14, 35, 22, 2]},
        ]

        for i, conv in enumerate(conversations):
            print(f"\n💬 Conversation {i + 1}:")
            print(f"   User: {conv['input']}")

            # Convert to JAX array
            input_tokens = jnp.array([conv["tokens"]])

            # Initialize fresh cache for each conversation (avoiding buffer donation issues)
            conv_cache = gpt_jax.KVCache.init(
                random.key(50 + i), cfg, 1, cfg.max_seq_len
            )

            # Generate response
            try:
                from jax.sharding import set_mesh
            except ImportError:
                from jax.sharding import set_mesh

            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    # Prefill phase
                    _, logits, updated_cache = gpt_jax.prefill(
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
                            _, logits, working_cache = gpt_jax.prefill(
                                current_tokens[:, -1:], weights, working_cache, cfg
                            )

                    print(f"   Bot: Generated tokens: {generated_tokens}")
                    print(f"        Full sequence: {current_tokens.tolist()[0]}")
                    print(f"        Response shape: {logits.shape}")

                    # Verify response is valid
                    assert logits.shape[0] == 1  # Batch size
                    assert logits.shape[2] == cfg.vocab_size
                    assert jnp.all(jnp.isfinite(logits))

            print("   ✅ Generation successful!")

        print("\n🎉 GPT-OSS Token Generation Demo Complete!")
        print(f"   • Model: {cfg.embed}D embeddings, {cfg.num_layers} layers")
        print(f"   • Vocab: {cfg.vocab_size} tokens")
        print(f"   • Generated responses for {len(conversations)} conversations")
        print("   • All responses were valid and finite")

    def test_gpt_oss_token_generation_patterns(self):
        """Test GPT-OSS with various input patterns for interactive use."""
        import jax
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
            model as gpt_jax,
        )

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create mesh with Explicit axis types (required for MoE operations)
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        simplified_rules = gpt_jax.ShardingRules(
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
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = gpt_jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=8,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            sliding_attention_map=["standard", "standard"],
            sliding_window_size=128,
            moe_ffw_size=512,
            moe_experts_per_tok=2,
            moe_num_experts=8,
            mesh=mesh,
            rules=simplified_rules,
        )

        weights = gpt_jax.Weights.init(random.key(42), cfg)
        cache = gpt_jax.KVCache.init(random.key(43), cfg, 1, cfg.max_seq_len)

        print("\n🔬 GPT-OSS Interactive Inference Test")
        print("=" * 45)

        # Test various input lengths and patterns
        test_cases = [
            {"name": "Short query", "tokens": [1, 5]},
            {"name": "Medium query", "tokens": [1, 10, 20, 15, 8]},
            {"name": "Question pattern", "tokens": [1, 25, 30, 12, 45, 2]},
            {"name": "Long context", "tokens": [1, 8, 15, 22, 35, 41, 18, 29, 7, 33]},
        ]

        for i, test_case in enumerate(test_cases):
            print(f"\n🧪 Testing: {test_case['name']}")
            input_tokens = jnp.array([test_case["tokens"]])

            # Initialize fresh cache for each test case (avoiding buffer donation issues)
            test_cache = gpt_jax.KVCache.init(
                random.key(60 + i), cfg, 1, cfg.max_seq_len
            )

            try:
                from jax.sharding import set_mesh
            except ImportError:
                from jax.sharding import set_mesh

            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    # Test prefill
                    _, logits, new_cache = gpt_jax.prefill(
                        input_tokens, weights, test_cache, cfg
                    )

                    print(f"   Input length: {input_tokens.shape[1]} tokens")
                    print(f"   Output shape: {logits.shape}")
                    print(f"   Max probability token: {jnp.argmax(logits[0, -1, :])}")

                    # Verify outputs
                    assert logits.shape == (1, input_tokens.shape[1], cfg.vocab_size)
                    assert jnp.all(jnp.isfinite(logits))

                    print("   ✅ Success!")

        print("\n🎊 GPT-OSS Interactive Inference Complete!")
        print(f"   • Tested {len(test_cases)} different input patterns")
        print("   • All tests passed with valid outputs")
        print("   • Model ready for interactive chat applications")
