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

"""Integration tests for JAX inference engine."""

from unittest.mock import patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role

# Mark all tests in this file as JAX-related
pytestmark = pytest.mark.jax


@pytest.fixture
def jax_model_params():
    """Create JAX-specific model parameters."""
    return ModelParams(
        model_name="jax-ml/llama3-8b",
        load_pretrained_weights=False,  # Don't load real weights in tests
        trust_remote_code=True,
        torch_dtype_str="bfloat16",
    )


@pytest.fixture
def jax_generation_params():
    """Create JAX-specific generation parameters."""
    return GenerationParams(
        max_new_tokens=20,
        temperature=0.8,
        top_p=0.9,
    )


class TestJAXIntegration:
    """Integration tests for JAX inference engine."""

    @pytest.mark.skipif(
        not pytest.importorskip("jax", minversion="0.4.31"), reason="JAX not available"
    )
    def test_jax_engine_initialization(self, jax_model_params):
        """Test that JAX engine can be initialized properly."""
        with (
            patch("oumi.inference.jax_inference_engine.build_tokenizer"),
            patch.object(
                "oumi.inference.jax_inference_engine.JAXInferenceEngine", "_load_model"
            ),
        ):
            from oumi.inference.jax_inference_engine import JAXInferenceEngine

            engine = JAXInferenceEngine(jax_model_params)
            assert engine is not None
            assert engine._model_params.model_name == "jax-ml/llama3-8b"

    @pytest.mark.skipif(
        not pytest.importorskip("jax", minversion="0.4.31"), reason="JAX not available"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("jax_llm_examples"),
        reason="jax-llm-examples not available",
    )
    def test_jax_model_loading(self, jax_model_params):
        """Test JAX model loading with actual jax-llm-examples."""
        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        # This test will only run if jax-llm-examples is actually installed
        with patch("oumi.inference.jax_inference_engine.build_tokenizer"):
            engine = JAXInferenceEngine(jax_model_params)
            # If we get here without exceptions, model loading worked
            assert True

    @pytest.mark.single_gpu
    @pytest.mark.skipif(
        not pytest.importorskip("jax", minversion="0.4.31"), reason="JAX not available"
    )
    def test_jax_gpu_inference(self, jax_model_params, jax_generation_params):
        """Test JAX inference on GPU."""
        with (
            patch("oumi.inference.jax_inference_engine.build_tokenizer"),
            patch.object(
                "oumi.inference.jax_inference_engine.JAXInferenceEngine", "_load_model"
            ),
        ):
            from oumi.inference.jax_inference_engine import JAXInferenceEngine

            engine = JAXInferenceEngine(
                jax_model_params, generation_params=jax_generation_params
            )

            # Test basic inference
            conversation = Conversation(
                messages=[Message(role=Role.USER, content="Hello, how are you?")]
            )

            result = engine._generate([conversation])
            assert len(result) == 1
            assert len(result[0].messages) == 2

    @pytest.mark.tpu
    @pytest.mark.skipif(
        not pytest.importorskip("jax", minversion="0.4.31"), reason="JAX not available"
    )
    def test_jax_tpu_inference(self, jax_model_params):
        """Test JAX inference on TPU."""
        # This test requires TPU environment setup
        import jax

        devices = jax.devices()
        tpu_devices = [d for d in devices if "tpu" in str(d).lower()]

        if not tpu_devices:
            pytest.skip("No TPU devices available")

        with (
            patch("oumi.inference.jax_inference_engine.build_tokenizer"),
            patch.object(
                "oumi.inference.jax_inference_engine.JAXInferenceEngine", "_load_model"
            ),
        ):
            from oumi.inference.jax_inference_engine import JAXInferenceEngine

            engine = JAXInferenceEngine(
                jax_model_params, tensor_parallel_size=len(tpu_devices)
            )

            assert engine._tensor_parallel_size == len(tpu_devices)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(
        not pytest.importorskip("jax", minversion="0.4.31"), reason="JAX not available"
    )
    def test_jax_multi_device_inference(self, jax_model_params):
        """Test JAX inference across multiple devices."""
        import jax

        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("Multiple devices not available for testing")

        with (
            patch("oumi.inference.jax_inference_engine.build_tokenizer"),
            patch.object(
                "oumi.inference.jax_inference_engine.JAXInferenceEngine", "_load_model"
            ),
        ):
            from oumi.inference.jax_inference_engine import JAXInferenceEngine

            engine = JAXInferenceEngine(
                jax_model_params, tensor_parallel_size=len(devices)
            )

            assert engine._tensor_parallel_size == len(devices)

    def test_jax_utils_import(self):
        """Test that JAX utilities can be imported."""
        from oumi.utils.jax_model_utils import setup_tensor_parallelism
        from oumi.utils.jax_utils import jax_to_torch, torch_to_jax

        # Test functions exist
        assert callable(torch_to_jax)
        assert callable(jax_to_torch)
        assert callable(setup_tensor_parallelism)

    def test_jax_builder_integration(self, jax_model_params):
        """Test JAX engine integration with Oumi's builder system."""
        from oumi.builders.inference_engines import build_inference_engine
        from oumi.core.configs import InferenceEngineType

        with (
            patch("oumi.inference.jax_inference_engine.build_tokenizer"),
            patch.object(
                "oumi.inference.jax_inference_engine.JAXInferenceEngine", "_load_model"
            ),
        ):
            engine = build_inference_engine(
                engine_type=InferenceEngineType.JAX, model_params=jax_model_params
            )

            assert engine is not None
            assert hasattr(engine, "_generate")
