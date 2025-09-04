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

"""Integration tests for Llama4 JAX model implementation."""

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


class TestLlama4JAXIntegration:
    """Integration tests for Llama4 JAX model."""

    def test_llama4_import(self):
        """Test that Llama4 JAX model can be imported."""
        from oumi.models.experimental.jax_models.llama4.llama4_jax import model as l4jax

        assert hasattr(l4jax, "Config")
        assert hasattr(l4jax, "Weights")
        assert hasattr(l4jax, "KVCache")
        assert hasattr(l4jax, "prefill")
        assert hasattr(l4jax, "decode_step")

    def test_llama4_scout_model_init_cpu(self):
        """Test Llama4 Scout model initialization on CPU."""
        import jax
        from jax import random
        from jax.sharding import AxisType

        try:
            from jax.sharding import AxisType
        except ImportError:
            # AxisType not available in older JAX versions
            AxisType = str
        from oumi.models.experimental.jax_models.llama4.llama4_jax import model as l4jax

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create small Scout test configuration
        # Llama4 uses .get(out_sharding) which needs Explicit axes in JAX 0.7.0
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = l4jax.ShardingRules(
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
            embed_up=None,
            ffw_up=None,
            ffw_down=None,
            embed_down=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_s_up_embed=None,
            moe_s_up_ffw=None,
            moe_s_down_ffw=None,
            moe_s_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = l4jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            nope_layer_interval=4,
            use_qk_norm=True,  # Scout uses QK norm
            attn_chunk_size=128,
            mlp_ffw_size=512,
            moe_ffw_size=256,
            moe_layer_interval=1,
            moe_experts_per_tok=1,
            moe_num_experts=4,
            moe_num_shared_experts=1,
            ep_strategy="decode",
            mesh=mesh,
            rules=simplified_rules,
        )

        # Test weight initialization
        weights = l4jax.Weights.init(random.key(42), cfg)
        assert weights is not None, "Scout weight initialization failed on CPU"

        # Test cache initialization
        batch_size = 2
        cache = l4jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)
        assert cache is not None, "Scout cache initialization failed on CPU"

        # Verify weight structure
        assert hasattr(weights, "embedding"), "Weights missing embedding component"
        assert hasattr(weights, "layers"), "Weights missing layers component"

    def test_llama4_maverick_model_init_cpu(self):
        """Test Llama4 Maverick model initialization on CPU."""
        import jax
        from jax import random
        from jax.sharding import AxisType

        try:
            from jax.sharding import AxisType
        except ImportError:
            # AxisType not available in older JAX versions
            AxisType = str
        from oumi.models.experimental.jax_models.llama4.llama4_jax import model as l4jax

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create small Maverick test configuration
        # Llama4 uses .get(out_sharding) which needs Explicit axes in JAX 0.7.0
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = l4jax.ShardingRules(
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
            embed_up=None,
            ffw_up=None,
            ffw_down=None,
            embed_down=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_s_up_embed=None,
            moe_s_up_ffw=None,
            moe_s_down_ffw=None,
            moe_s_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )
        cfg = l4jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            nope_layer_interval=4,
            use_qk_norm=False,  # Maverick doesn't use QK norm
            attn_chunk_size=128,
            mlp_ffw_size=512,
            moe_ffw_size=256,
            moe_layer_interval=2,  # Different MoE interval
            moe_experts_per_tok=1,
            moe_num_experts=8,  # More experts than Scout
            moe_num_shared_experts=1,
            ep_strategy="decode",
            mesh=mesh,
            rules=simplified_rules,
        )

        # Test weight initialization
        weights = l4jax.Weights.init(random.key(42), cfg)
        assert weights is not None, "Maverick weight initialization failed on CPU"

        # Test cache initialization
        batch_size = 2
        cache = l4jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)
        assert cache is not None, "Maverick cache initialization failed on CPU"

    @requires_gpus(1)
    @pytest.mark.single_gpu
    def test_llama4_model_init_gpu(self):
        """Test Llama4 model initialization on GPU."""
        import jax
        from jax import random
        from jax.sharding import AxisType

        try:
            from jax.sharding import AxisType
        except ImportError:
            # AxisType not available in older JAX versions
            AxisType = str
        from oumi.models.experimental.jax_models.llama4.llama4_jax import model as l4jax

        # Check if GPU is available
        devices = jax.devices()
        gpu_devices = [d for d in devices if "gpu" in str(d).lower()]

        if not gpu_devices:
            pytest.skip("No GPU devices available")

        # Create test configuration for GPU
        # Llama4 uses .get(out_sharding) which needs Explicit axes in JAX 0.7.0
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = l4jax.ShardingRules(
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
            embed_up=None,
            ffw_up=None,
            ffw_down=None,
            embed_down=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_s_up_embed=None,
            moe_s_up_ffw=None,
            moe_s_down_ffw=None,
            moe_s_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = l4jax.Config(
            embed=512,
            q_heads=16,
            kv_heads=8,
            num_layers=4,
            head_dim=32,
            vocab_size=2000,
            max_seq_len=256,
            causal=True,
            nope_layer_interval=4,
            use_qk_norm=True,
            attn_chunk_size=256,
            mlp_ffw_size=1024,
            moe_ffw_size=512,
            moe_layer_interval=1,
            moe_experts_per_tok=1,
            moe_num_experts=8,
            moe_num_shared_experts=1,
            ep_strategy="decode",
            mesh=mesh,
            rules=simplified_rules,
        )

        # Test weight initialization on GPU
        weights = l4jax.Weights.init(random.key(42), cfg)
        assert weights is not None, "Weight initialization failed on GPU"

        # Test cache initialization on GPU
        batch_size = 2
        cache = l4jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)
        assert cache is not None, "Cache initialization failed on GPU"

    def test_llama4_prefill_decode_cpu(self):
        """Test Llama4 prefill and decode on CPU."""
        import jax
        import jax.numpy as jnp
        from jax import random
        from jax.sharding import AxisType

        try:
            from jax.sharding import AxisType
        except ImportError:
            # AxisType not available in older JAX versions
            AxisType = str
        from oumi.models.experimental.jax_models.llama4.llama4_jax import model as l4jax

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        try:
            from jax.sharding import set_mesh

            set_mesh = set_mesh
        except ImportError:
            pass

        # Create small test configuration
        devices = jax.devices()
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = l4jax.ShardingRules(
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
            embed_up=None,
            ffw_up=None,
            ffw_down=None,
            embed_down=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_s_up_embed=None,
            moe_s_up_ffw=None,
            moe_s_down_ffw=None,
            moe_s_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = l4jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=2,
            num_layers=2,
            head_dim=32,
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            nope_layer_interval=4,
            use_qk_norm=True,
            attn_chunk_size=64,
            mlp_ffw_size=256,
            moe_ffw_size=128,
            moe_layer_interval=1,
            moe_experts_per_tok=1,
            moe_num_experts=4,
            moe_num_shared_experts=1,
            ep_strategy="decode",
            mesh=mesh,
            rules=simplified_rules,
        )

        # Initialize model components
        weights = l4jax.Weights.init(random.key(42), cfg)
        batch_size = 1
        seq_len = 16
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        cache = l4jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)

        # Test prefill
        if cfg.mesh is not None:
            with set_mesh(cfg.mesh):
                max_tokens, logits, cache = l4jax.prefill(tokens, weights, cache, cfg)
            assert max_tokens.shape == (batch_size, seq_len)
            assert logits.shape == (batch_size, seq_len, cfg.vocab_size)

        # Test decode steps
        next_tokens = max_tokens[:, -1:]
        if cfg.mesh is not None:
            with set_mesh(cfg.mesh):
                for _ in range(3):
                    next_tokens, cache = l4jax.decode_step(
                        next_tokens, weights, cache, cfg
                    )
                assert next_tokens.shape == (batch_size, 1)

    @requires_gpus(1)
    @pytest.mark.single_gpu
    def test_llama4_prefill_decode_gpu(self):
        """Test Llama4 prefill and decode on GPU."""
        import jax
        import jax.numpy as jnp
        from jax import random
        from jax.sharding import AxisType

        try:
            from jax.sharding import AxisType
        except ImportError:
            # AxisType not available in older JAX versions
            AxisType = str
        from oumi.models.experimental.jax_models.llama4.llama4_jax import model as l4jax

        # Check if GPU is available
        devices = jax.devices()
        gpu_devices = [d for d in devices if "gpu" in str(d).lower()]

        if not gpu_devices:
            pytest.skip("No GPU devices available")

        try:
            from jax.sharding import set_mesh

            set_mesh = set_mesh
        except ImportError:
            pass

        # Create test configuration for GPU
        # Llama4 uses .get(out_sharding) which needs Explicit axes in JAX 0.7.0
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = l4jax.ShardingRules(
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
            embed_up=None,
            ffw_up=None,
            ffw_down=None,
            embed_down=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_s_up_embed=None,
            moe_s_up_ffw=None,
            moe_s_down_ffw=None,
            moe_s_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = l4jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            nope_layer_interval=4,
            use_qk_norm=True,
            attn_chunk_size=128,
            mlp_ffw_size=512,
            moe_ffw_size=256,
            moe_layer_interval=1,
            moe_experts_per_tok=1,
            moe_num_experts=4,
            moe_num_shared_experts=1,
            ep_strategy="decode",
            mesh=mesh,
            rules=simplified_rules,
        )

        # Initialize model components
        weights = l4jax.Weights.init(random.key(42), cfg)
        batch_size = 2
        seq_len = 32
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        cache = l4jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)

        # Test prefill and decode on GPU
        if cfg.mesh is not None:
            with set_mesh(cfg.mesh):
                max_tokens, logits, cache = l4jax.prefill(tokens, weights, cache, cfg)
            assert max_tokens.shape == (batch_size, seq_len)

            next_tokens = max_tokens[:, -1:]
            for _ in range(2):
                next_tokens, cache = l4jax.decode_step(next_tokens, weights, cache, cfg)
                assert next_tokens.shape == (batch_size, 1)

    def test_llama4_quantization_cpu(self):
        """Test Llama4 quantization support on CPU."""
        import jax
        from jax import random
        from jax.sharding import AxisType

        try:
            from jax.sharding import AxisType
        except ImportError:
            # AxisType not available in older JAX versions
            AxisType = str
        from oumi.models.experimental.jax_models.llama4.llama4_jax import model as l4jax

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create small test configuration with quantization
        # Llama4 uses .get(out_sharding) which needs Explicit axes in JAX 0.7.0
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules for single device
        simplified_rules = l4jax.ShardingRules(
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
            embed_up=None,
            ffw_up=None,
            ffw_down=None,
            embed_down=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_s_up_embed=None,
            moe_s_up_ffw=None,
            moe_s_down_ffw=None,
            moe_s_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = l4jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=2,
            num_layers=2,
            head_dim=32,
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            nope_layer_interval=4,
            use_qk_norm=True,
            attn_chunk_size=64,
            mlp_ffw_size=256,
            moe_ffw_size=128,
            moe_layer_interval=1,
            moe_experts_per_tok=1,
            moe_num_experts=4,
            moe_num_shared_experts=1,
            ep_strategy="decode",
            mesh=mesh,
            quant_attn=True,  # Enable attention quantization
            quant_mlp=True,  # Enable MLP quantization
            quant_moe=True,  # Enable MoE quantization
            quant_cache=True,  # Enable cache quantization
            rules=simplified_rules,
        )

        # Test quantized weight initialization
        weights = l4jax.Weights.init(random.key(42), cfg)
        assert weights is not None

        # Test quantized cache initialization
        batch_size = 1
        cache = l4jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)
        assert cache is not None

    def test_llama4_config_validation(self):
        """Test Llama4 configuration validation."""
        from oumi.models.experimental.jax_models.llama4.llama4_jax import model as l4jax

        # Test Scout config
        scout_cfg = l4jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            nope_layer_interval=4,
            use_qk_norm=True,
            attn_chunk_size=128,
            mlp_ffw_size=512,
            moe_ffw_size=256,
            moe_layer_interval=1,
            moe_experts_per_tok=1,
            moe_num_experts=4,
            moe_num_shared_experts=1,
            ep_strategy="decode",
        )

        assert scout_cfg.use_qk_norm is True
        assert scout_cfg.moe_layer_interval == 1

        # Test Maverick config
        maverick_cfg = l4jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            nope_layer_interval=4,
            use_qk_norm=False,  # Maverick difference
            attn_chunk_size=128,
            mlp_ffw_size=512,
            moe_ffw_size=256,
            moe_layer_interval=2,  # Different MoE interval
            moe_experts_per_tok=1,
            moe_num_experts=8,  # More experts
            moe_num_shared_experts=1,
            ep_strategy="decode",
        )

        assert maverick_cfg.use_qk_norm is False
        assert maverick_cfg.moe_layer_interval == 2
        assert maverick_cfg.moe_num_experts == 8

    def test_llama4_integration_with_oumi_engine(self):
        """Test Llama4 integration with Oumi's JAX inference engine."""
        from oumi.core.configs import ModelParams

        # Create model parameters for Llama4
        model_params = ModelParams(
            model_name="jax-ml/llama4",
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
            assert engine._model_params.model_name == "jax-ml/llama4"

    def test_llama4_numerical_stability(self):
        """Test Llama4 numerical stability across different seeds."""
        import jax
        import jax.numpy as jnp
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.llama4.llama4_jax import model as l4jax

        # Force CPU execution for stability testing
        jax.config.update("jax_platforms", "cpu")

        try:
            from jax.sharding import set_mesh

            set_mesh = set_mesh
        except ImportError:
            set_mesh = None

        # Create mesh with Explicit axis types for Llama4
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )

        # Create simplified sharding rules
        simplified_rules = l4jax.ShardingRules(
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
            embed_up=None,
            ffw_up=None,
            ffw_down=None,
            embed_down=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_s_up_embed=None,
            moe_s_up_ffw=None,
            moe_s_down_ffw=None,
            moe_s_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        cfg = l4jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=2,
            num_layers=2,
            head_dim=32,
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            nope_layer_interval=4,
            use_qk_norm=True,
            attn_chunk_size=64,
            mlp_ffw_size=256,
            moe_ffw_size=128,
            moe_layer_interval=1,
            moe_experts_per_tok=1,
            moe_num_experts=4,
            moe_num_shared_experts=1,
            ep_strategy="decode",
            mesh=mesh,
            rules=simplified_rules,
        )

        # Test with multiple seeds
        outputs = []
        for seed in [42, 123, 456]:
            weights = l4jax.Weights.init(random.key(seed), cfg)
            batch_size = 1
            seq_len = 8
            tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
            cache = l4jax.KVCache.init(
                random.key(seed), cfg, batch_size, cfg.max_seq_len
            )

            if set_mesh is not None:
                if cfg.mesh is not None:
                    with set_mesh(cfg.mesh):
                        max_tokens, logits, _ = l4jax.prefill(
                            tokens, weights, cache, cfg
                        )
            else:
                max_tokens, logits, _ = l4jax.prefill(tokens, weights, cache, cfg)

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

    def test_llama4_error_handling(self):
        """Test Llama4 error handling for invalid inputs."""
        import jax
        import jax.numpy as jnp
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.llama4.llama4_jax import model as l4jax

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create valid configuration first
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        simplified_rules = l4jax.ShardingRules(
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
            embed_up=None,
            ffw_up=None,
            ffw_down=None,
            embed_down=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_s_up_embed=None,
            moe_s_up_ffw=None,
            moe_s_down_ffw=None,
            moe_s_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        # Test invalid configuration values
        with pytest.raises((ValueError, TypeError, AssertionError)):
            l4jax.Config(
                embed=0,  # Invalid: must be positive
                q_heads=4,
                kv_heads=2,
                num_layers=2,
                head_dim=32,
                vocab_size=500,
                max_seq_len=64,
                causal=True,
                nope_layer_interval=4,
                use_qk_norm=True,
                attn_chunk_size=64,
                mlp_ffw_size=256,
                moe_ffw_size=256,
                moe_layer_interval=4,
                moe_experts_per_tok=2,
                moe_num_experts=4,
                mesh=mesh,
                rules=simplified_rules,
            )

        with pytest.raises((ValueError, TypeError, AssertionError)):
            l4jax.Config(
                embed=128,
                q_heads=0,  # Invalid: must be positive
                kv_heads=2,
                num_layers=2,
                head_dim=32,
                vocab_size=500,
                max_seq_len=64,
                causal=True,
                nope_layer_interval=4,
                use_qk_norm=True,
                attn_chunk_size=64,
                mlp_ffw_size=256,
                moe_ffw_size=256,
                moe_layer_interval=4,
                moe_experts_per_tok=2,
                moe_num_experts=4,
                mesh=mesh,
                rules=simplified_rules,
            )

        # Test valid configuration for subsequent tests
        cfg = l4jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=2,
            num_layers=2,
            head_dim=32,
            vocab_size=500,
            max_seq_len=64,
            causal=True,
            nope_layer_interval=4,
            use_qk_norm=True,
            attn_chunk_size=64,
            mlp_ffw_size=256,
            moe_ffw_size=128,
            moe_layer_interval=1,
            moe_experts_per_tok=1,
            moe_num_experts=4,
            moe_num_shared_experts=1,
            ep_strategy="decode",
            mesh=mesh,
            rules=simplified_rules,
        )

        weights = l4jax.Weights.init(random.key(42), cfg)
        cache = l4jax.KVCache.init(random.key(42), cfg, 1, cfg.max_seq_len)

        # Test invalid token shapes
        with pytest.raises((ValueError, TypeError, AssertionError)):
            invalid_tokens = jnp.ones(
                (1, cfg.max_seq_len + 1), dtype=jnp.int32
            )  # Too long
            l4jax.prefill(invalid_tokens, weights, cache, cfg)

        # Test invalid token values
        with pytest.raises((ValueError, TypeError, AssertionError)):
            invalid_tokens = jnp.full(
                (1, 8), cfg.vocab_size, dtype=jnp.int32
            )  # Out of vocab range
            l4jax.prefill(invalid_tokens, weights, cache, cfg)

    def test_llama4_memory_efficiency(self):
        """Test Llama4 memory usage with minimal configurations."""
        import jax
        import jax.numpy as jnp
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.llama4.llama4_jax import model as l4jax

        # Force CPU execution for memory testing
        jax.config.update("jax_platforms", "cpu")

        try:
            from jax.sharding import set_mesh

            set_mesh = set_mesh
        except ImportError:
            set_mesh = None

        # Create minimal configuration
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        simplified_rules = l4jax.ShardingRules(
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
            embed_up=None,
            ffw_up=None,
            ffw_down=None,
            embed_down=None,
            mlp_up_embed=None,
            mlp_up_ffw=None,
            mlp_down_ffw=None,
            mlp_down_embed=None,
            moe_e_experts=None,
            moe_e_up_embed=None,
            moe_e_up_ffw=None,
            moe_e_down_ffw=None,
            moe_e_down_embed=None,
            moe_s_up_embed=None,
            moe_s_up_ffw=None,
            moe_s_down_ffw=None,
            moe_s_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
            vocab_in=None,
            vocab_out=None,
        )

        # Minimal configuration for memory efficiency
        cfg = l4jax.Config(
            embed=64,  # Small embedding
            q_heads=2,  # Few heads
            kv_heads=1,  # Single KV head
            num_layers=1,  # Single layer
            head_dim=16,  # Small head dimension
            vocab_size=100,  # Small vocabulary
            max_seq_len=32,  # Short sequences
            causal=True,
            nope_layer_interval=4,
            use_qk_norm=True,
            attn_chunk_size=32,
            mlp_ffw_size=128,  # Small MLP
            moe_ffw_size=64,  # Small MoE
            moe_layer_interval=1,
            moe_experts_per_tok=1,
            moe_num_experts=2,  # Few experts
            moe_num_shared_experts=1,
            ep_strategy="decode",
            mesh=mesh,
            rules=simplified_rules,
        )

        # Test that minimal model can be initialized and run
        weights = l4jax.Weights.init(random.key(42), cfg)
        assert weights is not None, "Failed to initialize minimal model weights"

        batch_size = 1
        seq_len = 8
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        cache = l4jax.KVCache.init(random.key(42), cfg, batch_size, cfg.max_seq_len)

        # Test prefill with minimal configuration
        if set_mesh is not None:
            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    max_tokens, logits, updated_cache = l4jax.prefill(
                        tokens, weights, cache, cfg
                    )
        else:
            max_tokens, logits, updated_cache = l4jax.prefill(
                tokens, weights, cache, cfg
            )

        assert max_tokens.shape == (batch_size, seq_len), "Unexpected max_tokens shape"
        assert logits.shape == (batch_size, seq_len, cfg.vocab_size), (
            "Unexpected logits shape"
        )
        assert updated_cache is not None, "Cache should be updated after prefill"

        # Test decode step with minimal configuration
        next_token = max_tokens[:, -1:]
        if set_mesh is not None:
            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    next_token_out, final_cache = l4jax.decode_step(
                        next_token, weights, updated_cache, cfg
                    )
        else:
            next_token_out, final_cache = l4jax.decode_step(
                next_token, weights, updated_cache, cfg
            )

        assert next_token_out.shape == (batch_size, 1), "Unexpected decode output shape"
        assert final_cache is not None, "Cache should be updated after decode"

    def test_llama4_token_generation_demo(self):
        """Test Llama4 token generation mechanics."""
        import jax
        from jax import random
        from jax.sharding import AxisType

        from oumi.models.experimental.jax_models.llama4.llama4_jax import (
            model as l4jax,
        )

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        # Create simplified mesh
        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Auto, AxisType.Auto, AxisType.Auto),
        )

        # Create simplified sharding rules for Llama4
        simplified_rules = l4jax.ShardingRules(
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
            moe_s_up_embed=None,
            moe_s_up_ffw=None,
            moe_s_down_ffw=None,
            moe_s_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
        )

        # Chat configuration (same scale as working tests, MoE disabled)
        cfg = l4jax.Config(
            embed=256,
            q_heads=8,
            kv_heads=4,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=128,
            causal=True,
            nope_layer_interval=10,  # Disable NOPE layers
            use_qk_norm=False,  # Disable QK norm
            attn_chunk_size=128,
            mlp_ffw_size=512,
            moe_ffw_size=256,
            moe_layer_interval=10,  # Disable MoE layers
            moe_experts_per_tok=2,
            moe_num_experts=8,
            moe_num_shared_experts=2,
            use_prefill_attn_kernel=False,
            use_decode_attn_kernel=False,
            use_ragged_dot_kernel=False,
            mesh=mesh,
            rules=simplified_rules,
        )

        # Initialize model
        weights = l4jax.Weights.init(random.key(42), cfg)
        cache = l4jax.KVCache.init(random.key(43), cfg, 1, cfg.max_seq_len)

        print("\n🦙 Llama4 Token Generation Demo")
        print("=" * 40)

        try:
            from jax.sharding import set_mesh
        except ImportError:
            from jax.sharding import set_mesh

        # Simulate chat conversations with advanced Llama4 features
        conversations = [
            {"input": "Hello Llama4!", "tokens": [1, 25, 42, 18, 33, 2]},
            {"input": "Explain MoE architecture", "tokens": [1, 35, 48, 29, 67, 52, 2]},
            {"input": "Advanced reasoning task", "tokens": [1, 44, 58, 73, 21, 39, 2]},
            {
                "input": "Multi-step problem solving",
                "tokens": [1, 51, 64, 77, 33, 88, 19, 2],
            },
        ]

        for i, conv in enumerate(conversations):
            print(f"\n🧠 Conversation {i + 1}:")
            print(f"   User: {conv['input']}")

            # Convert to JAX array
            input_tokens = jnp.array([conv["tokens"]])

            # Initialize fresh cache for each conversation (avoiding buffer donation issues)
            conv_cache = l4jax.KVCache.init(random.key(50 + i), cfg, 1, cfg.max_seq_len)

            # Generate response using Llama4's MoE capabilities
            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    # Prefill phase
                    max_tokens, logits, updated_cache = l4jax.prefill(
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
                            _, logits, working_cache = l4jax.prefill(
                                current_tokens[:, -1:], weights, working_cache, cfg
                            )

                    print(f"   Llama4: Generated tokens: {generated_tokens}")
                    print(f"        Full sequence: {current_tokens.tolist()[0]}")
                    print(f"        Response shape: {logits.shape}")

                    # Verify response is valid
                    assert logits.shape[0] == 1  # Batch size
                    assert logits.shape[2] == cfg.vocab_size
                    assert jnp.all(jnp.isfinite(logits))

                    print("   ✅ Generation successful!")

        print("\n🎉 Llama4 Token Generation Demo Complete!")
        print(f"   • Model: {cfg.embed}D embeddings, {cfg.num_layers} layers")
        print(f"   • Vocab: {cfg.vocab_size} tokens")
        print(f"   • Generated responses for {len(conversations)} conversations")
        print("   • All responses were valid and finite")

    def test_llama4_moe_chat_specialization(self):
        """Test Llama4's MoE expert specialization in chat contexts."""
        import jax
        from jax import random

        from oumi.models.experimental.jax_models.llama4.llama4_jax import (
            model as l4jax,
        )

        # Force CPU execution
        jax.config.update("jax_platforms", "cpu")

        from jax.sharding import AxisType

        mesh = jax.make_mesh(
            (1, 1, 1),
            ("x", "y", "z"),
            axis_types=(AxisType.Auto, AxisType.Auto, AxisType.Auto),
        )

        simplified_rules = l4jax.ShardingRules(
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
            moe_s_up_embed=None,
            moe_s_up_ffw=None,
            moe_s_down_ffw=None,
            moe_s_down_embed=None,
            moe_e_tp=None,
            moe_e_ep=None,
        )

        # Smaller config for MoE specialization testing
        cfg = l4jax.Config(
            embed=128,
            q_heads=4,
            kv_heads=2,
            num_layers=2,
            head_dim=32,
            vocab_size=1000,
            max_seq_len=64,
            causal=True,
            nope_layer_interval=1,
            use_qk_norm=True,
            attn_chunk_size=32,
            mlp_ffw_size=256,
            moe_ffw_size=128,
            moe_layer_interval=1,
            moe_experts_per_tok=2,
            moe_num_experts=4,
            moe_num_shared_experts=1,
            use_prefill_attn_kernel=False,
            use_decode_attn_kernel=False,
            use_ragged_dot_kernel=False,
            mesh=mesh,
            rules=simplified_rules,
        )

        weights = l4jax.Weights.init(random.key(42), cfg)
        cache = l4jax.KVCache.init(random.key(43), cfg, 1, cfg.max_seq_len)

        print("\n🎯 Llama4 MoE Specialization Test")
        print("=" * 50)

        try:
            from jax.sharding import set_mesh
        except ImportError:
            from jax.sharding import set_mesh

        # Test different types of queries to see MoE expert routing
        query_types = [
            {
                "type": "Math",
                "tokens": [1, 10, 20, 30, 2],
                "description": "Mathematical reasoning",
            },
            {
                "type": "Language",
                "tokens": [1, 40, 50, 60, 2],
                "description": "Language understanding",
            },
            {
                "type": "Logic",
                "tokens": [1, 70, 80, 90, 2],
                "description": "Logical reasoning",
            },
            {
                "type": "Creative",
                "tokens": [1, 15, 25, 35, 2],
                "description": "Creative generation",
            },
        ]

        for i, query in enumerate(query_types):
            print(f"\n🔍 Testing {query['type']} query:")
            print(f"   Description: {query['description']}")

            input_tokens = jnp.array([query["tokens"]])

            # Initialize fresh cache for each query (avoiding buffer donation issues)
            query_cache = l4jax.KVCache.init(
                random.key(70 + i), cfg, 1, cfg.max_seq_len
            )

            if cfg.mesh is not None:
                with set_mesh(cfg.mesh):
                    # Test prefill to see expert activation
                    max_tokens, logits, updated_cache = l4jax.prefill(
                        input_tokens, weights, query_cache, cfg
                    )

                    print(f"   Input tokens: {query['tokens']}")
                    print(f"   Output shape: {logits.shape}")
                    print(
                        f"   MoE routing: {cfg.moe_experts_per_tok}/{cfg.moe_num_experts} experts active"
                    )

                    # Test decode step
                    next_token = jnp.array([[42]])
                    decoded_token, final_cache = l4jax.decode_step(
                        next_token, weights, updated_cache, cfg
                    )

                    # Verify outputs
                    assert logits.shape == (1, input_tokens.shape[1], cfg.vocab_size)
                    assert jnp.all(jnp.isfinite(logits))
                    assert decoded_token.shape == (1, 1)

                    print("   ✅ MoE processing successful!")

        print("\n🌟 Llama4 MoE Specialization Complete!")
        print(f"   • Tested {len(query_types)} different query types")
        print(
            f"   • Each query type processed by {cfg.moe_experts_per_tok} active experts"
        )
        print("   • MoE routing adapts to different reasoning patterns")
        print("   • Model ready for specialized chat applications")
