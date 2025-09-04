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

from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role

# Mark all tests in this file as JAX-related
pytestmark = pytest.mark.jax


@pytest.fixture
def model_params():
    """Create test model parameters."""
    return ModelParams(
        model_name="jax-ml/llama3-8b",
        load_pretrained_weights=False,  # Avoid loading real weights in tests
        trust_remote_code=True,
    )


@pytest.fixture
def generation_params():
    """Create test generation parameters."""
    return GenerationParams(
        max_new_tokens=10,
        temperature=0.8,
        top_p=0.9,
    )


@pytest.fixture
def sample_conversation():
    """Create a sample conversation for testing."""
    return Conversation(
        messages=[Message(role=Role.USER, content="Hello, how are you?")]
    )


class TestJAXInferenceEngine:
    """Test suite for JAX Inference Engine."""

    def test_jax_import_error(self, model_params):
        """Test that proper error is raised when JAX is not available."""
        with patch.dict("sys.modules", {"jax": None}):
            with pytest.raises(RuntimeError, match="JAX is not installed"):
                from oumi.inference.jax_inference_engine import JAXInferenceEngine

                JAXInferenceEngine(model_params)

    @patch("oumi.inference.jax_inference_engine.jax")
    @patch("oumi.inference.jax_inference_engine.build_tokenizer")
    def test_initialization(self, mock_build_tokenizer, mock_jax, model_params):
        """Test JAX engine initialization."""
        # Mock JAX components
        mock_jax.devices.return_value = ["device0", "device1"]
        mock_jax.random.PRNGKey.return_value = MagicMock()
        mock_build_tokenizer.return_value = MagicMock()

        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        with patch.object(JAXInferenceEngine, "_load_model"):
            engine = JAXInferenceEngine(model_params)

            assert engine._tensor_parallel_size == 2  # Two devices
            assert engine._model_params.model_name == "jax-ml/llama3-8b"

    @patch("oumi.inference.jax_inference_engine.jax")
    @patch("oumi.inference.jax_inference_engine.build_tokenizer")
    def test_unsupported_model(self, mock_build_tokenizer, mock_jax, model_params):
        """Test error handling for unsupported models."""
        model_params.model_name = "unsupported-model"

        mock_jax.devices.return_value = ["device0"]
        mock_jax.random.PRNGKey.return_value = MagicMock()
        mock_build_tokenizer.return_value = MagicMock()

        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        with (
            patch.object(JAXInferenceEngine, "_setup_jax_devices"),
            patch("importlib.util.find_spec", return_value=True),
        ):
            with pytest.raises(ValueError, match="Unsupported JAX model"):
                JAXInferenceEngine(model_params)

    @patch("oumi.inference.jax_inference_engine.jax")
    @patch("oumi.inference.jax_inference_engine.build_tokenizer")
    def test_memory_fraction_validation(
        self, mock_build_tokenizer, mock_jax, model_params
    ):
        """Test memory fraction parameter validation."""
        mock_jax.devices.return_value = ["device0"]
        mock_jax.random.PRNGKey.return_value = MagicMock()

        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        with patch.object(JAXInferenceEngine, "_load_model"):
            # Test invalid memory fraction
            with pytest.raises(ValueError, match="Memory fraction must be within"):
                JAXInferenceEngine(model_params, memory_fraction=1.5)

    @patch("oumi.inference.jax_inference_engine.jax")
    @patch("oumi.inference.jax_inference_engine.build_tokenizer")
    def test_generation_fallback(
        self, mock_build_tokenizer, mock_jax, model_params, sample_conversation
    ):
        """Test generation fallback when model is not loaded."""
        mock_jax.devices.return_value = ["device0"]
        mock_jax.random.PRNGKey.return_value = MagicMock()
        mock_build_tokenizer.return_value = MagicMock()

        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        with patch.object(JAXInferenceEngine, "_load_model") as mock_load:
            # Simulate failed model loading
            mock_load.side_effect = lambda: setattr(self, "_model", None) or setattr(
                self, "_params", None
            )
            engine = JAXInferenceEngine(model_params)
            engine._model = None
            engine._params = None

            # Test generation with unloaded model
            result = engine._generate([sample_conversation])

            assert len(result) == 1
            assert len(result[0].messages) == 2  # Original + generated
            assert "JAX model not loaded" in result[0].messages[1].content

    def test_dependency_check(self, model_params):
        """Test dependency checking functionality."""
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(
                RuntimeError, match="Failed to find the required dependency"
            ):
                from oumi.inference.jax_inference_engine import JAXInferenceEngine

                engine = JAXInferenceEngine(model_params)
                engine._load_model()

    @patch("oumi.inference.jax_inference_engine.jnp")
    @patch("oumi.inference.jax_inference_engine.jax")
    def test_top_p_sampling(self, mock_jax, mock_jnp):
        """Test top-p sampling implementation."""
        # This would test the _sample_top_p method
        # Implementation depends on actual JAX arrays
        pass  # Placeholder for more detailed JAX-specific tests
