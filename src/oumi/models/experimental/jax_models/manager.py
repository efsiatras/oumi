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

"""JAX Model Manager - Unified interface for downloading, converting, and loading JAX models
Following the patterns from jax-llm-examples
"""

import dataclasses
import importlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from .registry import (
    get_implementation_module,
    get_model_info,
    get_recommended_model,
    get_supported_models,
    validate_model_name,
)


class JAXModelManager:
    """Unified manager for JAX models following jax-llm-examples patterns"""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the JAX model manager

        Args:
            cache_dir: Directory to cache models (default: ~/.cache/oumi_jax_models)
        """
        self.cache_dir = cache_dir or Path.home() / ".cache/oumi_jax_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_available_models(self) -> dict[str, Any]:
        """List all available models with their info"""
        models = get_supported_models()
        return {
            name: {
                "model_id": info.model_id,
                "architecture": info.architecture,
                "description": info.description,
                "size_gb": info.size_gb,
                "requires_auth": info.requires_auth,
                "recommended_hardware": info.recommended_hardware,
                "notes": info.notes,
            }
            for name, info in models.items()
        }

    def recommend_model(
        self, max_size_gb: Optional[float] = None, requires_no_auth: bool = True
    ) -> Optional[str]:
        """Recommend a model based on constraints"""
        model_info = get_recommended_model(max_size_gb, requires_no_auth)
        if model_info:
            # Find the model name from the registry
            for name, info in get_supported_models().items():
                if info.model_id == model_info.model_id:
                    return name
        return None

    def download_model(self, model_name: str, force_download: bool = False) -> Path:
        """Download a model from HuggingFace

        Args:
            model_name: Name of the model from the registry
            force_download: Force re-download even if model exists

        Returns:
            Path to the downloaded model directory
        """
        if not validate_model_name(model_name):
            available = list(get_supported_models().keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}"
            )

        model_info = get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model info not found for '{model_name}'")

        # Create model directory
        model_dir = self.cache_dir / model_name / "hf_original"

        if model_dir.exists() and not force_download:
            print(f"✅ Model already downloaded: {model_dir}")
            return model_dir

        print(f"📥 Downloading {model_info.model_id}...")

        if model_info.requires_auth:
            print("⚠️  This model requires HuggingFace authentication!")
            print("   Run: huggingface-cli login")

        try:
            from huggingface_hub import snapshot_download

            # Set up environment for faster downloads
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

            model_dir.parent.mkdir(parents=True, exist_ok=True)

            snapshot_download(
                repo_id=model_info.model_id,
                local_dir=model_dir,
                ignore_patterns=["*.bin"],  # Only download safetensors
            )

            print(f"✅ Downloaded to: {model_dir}")
            return model_dir

        except Exception as e:
            if model_info.requires_auth and "401" in str(e):
                raise RuntimeError(
                    f"Authentication failed for {model_info.model_id}. "
                    f"Please run: huggingface-cli login"
                )
            raise RuntimeError(f"Download failed: {e}")

    def convert_model(self, model_name: str, force_convert: bool = False) -> Path:
        """Convert a downloaded model to JAX format

        Args:
            model_name: Name of the model from the registry
            force_convert: Force re-conversion even if JAX model exists

        Returns:
            Path to the converted JAX model directory
        """
        if not validate_model_name(model_name):
            available = list(get_supported_models().keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}"
            )

        model_info = get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model info not found for '{model_name}'")

        # Paths
        hf_model_dir = self.cache_dir / model_name / "hf_original"
        jax_model_dir = self.cache_dir / model_name / "jax_converted"

        if not hf_model_dir.exists():
            raise RuntimeError(
                f"Original model not found. Download first: {hf_model_dir}"
            )

        if (
            jax_model_dir.exists()
            and (jax_model_dir / "config.json").exists()
            and not force_convert
        ):
            print(f"✅ JAX model already converted: {jax_model_dir}")
            return jax_model_dir

        # Clean up if force converting
        if force_convert and jax_model_dir.exists():
            print("🧹 Cleaning up existing JAX model for force conversion...")
            import shutil

            shutil.rmtree(jax_model_dir)

        print(f"🔄 Converting {model_name} to JAX format...")

        try:
            # Import the correct implementation
            impl_module_path = get_implementation_module(model_info.architecture)
            if not impl_module_path:
                raise ValueError(
                    f"No implementation found for architecture: {model_info.architecture}"
                )

            # Dynamic import
            model_module = importlib.import_module(f"{impl_module_path}.model")
            utils_module = importlib.import_module(f"{impl_module_path}.chkpt_utils")

            # Load HF config and convert
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(str(hf_model_dir))

            # Convert to JAX config using the appropriate function
            if hasattr(model_module, "llama_to_jax_config"):
                jax_config = model_module.llama_to_jax_config(hf_config.to_dict())
            elif hasattr(model_module, "hf_to_jax_config"):
                jax_config = model_module.hf_to_jax_config(hf_config.to_dict())
            else:
                raise RuntimeError(
                    f"No config conversion function found in {model_module}"
                )

            # Create mesh for conversion
            import jax
            from jax.sharding import AxisType

            mesh = jax.make_mesh(
                (1, 1, jax.device_count()),
                ("x", "y", "z"),
                devices=jax.devices(),
                axis_types=(AxisType.Explicit,) * 3,
            )
            jax_config = dataclasses.replace(jax_config, mesh=mesh)

            print(
                f"📋 Config: {jax_config.num_layers} layers, {jax_config.vocab_size} vocab"
            )

            # Load model tensors
            safetensors_files = list(hf_model_dir.glob("*.safetensors"))
            if not safetensors_files:
                raise RuntimeError(f"No safetensors files found in {hf_model_dir}")

            model_tensors = {}
            for file_path in tqdm(safetensors_files, desc="Loading tensors"):
                from safetensors import safe_open

                with safe_open(file_path, framework="torch") as f:
                    for key in f.keys():
                        model_tensors[key] = f.get_tensor(key)

            print(
                f"📂 Loaded {len(model_tensors)} tensors from {len(safetensors_files)} files"
            )

            # Convert using the appropriate method
            if hasattr(utils_module, "convert_model_or_layer"):
                # Llama3/Qwen3 style conversion - handle different quantization params
                try:
                    # Try Llama3 style (quant_layer)
                    unquantized_config = dataclasses.replace(
                        jax_config, quant_layer=False
                    )
                except TypeError:
                    # Try Qwen3 style (quant_moe, quant_mlp, quant_attn)
                    try:
                        unquantized_config = dataclasses.replace(
                            jax_config,
                            quant_moe=False,
                            quant_mlp=False,
                            quant_attn=False,
                            quant_cache=False,
                        )
                    except TypeError:
                        # Fallback: use config as-is
                        unquantized_config = jax_config

                weights = model_module.Weights.abstract(unquantized_config)
                converted_weights = utils_module.convert_model_or_layer(
                    weights, model_tensors, jax_config, sequential=False
                )
            elif hasattr(utils_module, "convert_hf_checkpoint"):
                # DeepSeek R1 style conversion - need params_map format
                params_map = {}
                for file_path in safetensors_files:
                    from safetensors import safe_open

                    with safe_open(file_path, framework="torch") as f:
                        for key in f.keys():
                            tensor = f.get_tensor(key)
                            params_map[key] = {
                                "file": file_path.name,
                                "shape": list(tensor.shape),
                                "dtype": str(tensor.dtype).replace("torch.", ""),
                            }

                # Use convert_hf_checkpoint
                utils_module.convert_hf_checkpoint(
                    params_map, hf_model_dir, jax_model_dir, jax_config
                )

                # Copy additional files
                import shutil

                for file in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
                    src = hf_model_dir / file
                    if src.exists():
                        shutil.copy(src, jax_model_dir / file)

                print(f"✅ Model converted to: {jax_model_dir}")
                return jax_model_dir
            else:
                raise RuntimeError(f"No conversion function found in {utils_module}")

            # Save converted weights (for convert_model_or_layer path)
            jax_model_dir.mkdir(parents=True, exist_ok=True)
            model_module.save_pytree(converted_weights, jax_model_dir)

            # Copy tokenizer and config files
            import shutil

            for file in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
                src = hf_model_dir / file
                if src.exists():
                    shutil.copy(src, jax_model_dir / file)

            print(f"✅ Model converted to: {jax_model_dir}")
            return jax_model_dir

        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def load_model(
        self, model_name: str, auto_download: bool = True, auto_convert: bool = True
    ) -> tuple[Any, Any, Any]:
        """Load a JAX model for inference

        Args:
            model_name: Name of the model from the registry
            auto_download: Automatically download if not present
            auto_convert: Automatically convert if not present

        Returns:
            Tuple of (weights, config, tokenizer)
        """
        if not validate_model_name(model_name):
            available = list(get_supported_models().keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}"
            )

        model_info = get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model info not found for '{model_name}'")

        # Check paths
        hf_model_dir = self.cache_dir / model_name / "hf_original"
        jax_model_dir = self.cache_dir / model_name / "jax_converted"

        # Auto-download if needed
        if not hf_model_dir.exists() and auto_download:
            print(f"📥 Auto-downloading {model_name}...")
            self.download_model(model_name)

        # Auto-convert if needed
        if not jax_model_dir.exists() and auto_convert:
            print(f"🔄 Auto-converting {model_name}...")
            self.convert_model(model_name)

        if not jax_model_dir.exists():
            raise RuntimeError(f"JAX model not found: {jax_model_dir}")

        print(f"⏳ Loading JAX model: {model_name}")

        try:
            # Import the correct implementation
            impl_module_path = get_implementation_module(model_info.architecture)
            model_module = importlib.import_module(f"{impl_module_path}.model")

            # Load config
            with open(jax_model_dir / "config.json") as f:
                hf_config = json.load(f)

            if hasattr(model_module, "llama_to_jax_config"):
                config = model_module.llama_to_jax_config(hf_config)
            elif hasattr(model_module, "hf_to_jax_config"):
                config = model_module.hf_to_jax_config(hf_config)
            else:
                raise RuntimeError("No config conversion function found")

            # Create mesh
            import jax
            from jax.sharding import AxisType

            mesh = jax.make_mesh(
                (1, 1, jax.device_count()),
                ("x", "y", "z"),
                devices=jax.devices(),
                axis_types=(AxisType.Explicit,) * 3,
            )
            # Set up config with mesh and disable quantization
            try:
                # Try Llama3 style
                config = dataclasses.replace(
                    config, mesh=mesh, quant_layer=False, quant_cache=False
                )
            except TypeError:
                # Try Qwen3 style
                try:
                    config = dataclasses.replace(
                        config,
                        mesh=mesh,
                        quant_moe=False,
                        quant_mlp=False,
                        quant_attn=False,
                        quant_cache=False,
                    )
                except TypeError:
                    # Fallback: just set mesh
                    config = dataclasses.replace(config, mesh=mesh)

            # Load weights
            try:
                weights = model_module.load_pytree(
                    jax_model_dir, model_module.Weights.shardings(config)
                )
                print("✅ Loaded weights with proper sharding")
            except:
                print("⚠️  Sharding load failed, trying structure conversion...")
                raw_weights = model_module.load_pytree(jax_model_dir, None)

                # Convert to structured format
                if (
                    isinstance(raw_weights.get("layers", []), list)
                    and raw_weights["layers"]
                ):
                    structured_layers = []
                    for layer_dict in raw_weights["layers"]:
                        if isinstance(layer_dict, dict):
                            layer = model_module.Layer(**layer_dict)
                            structured_layers.append(layer)
                        else:
                            structured_layers.append(layer_dict)

                    weights = model_module.Weights(
                        layers=structured_layers,
                        embedding=raw_weights["embedding"],
                        gamma_final=raw_weights["gamma_final"],
                        lm_head=raw_weights["lm_head"],
                    )
                else:
                    weights = raw_weights

                print("✅ Loaded weights with structure conversion")

            # Load tokenizer
            if hasattr(model_module, "load_tokenizer"):
                tokenizer = model_module.load_tokenizer(
                    jax_model_dir / "tokenizer.json",
                    jax_model_dir / "tokenizer_config.json",
                )
            else:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(str(hf_model_dir))

            print(f"✅ Successfully loaded {model_name}")
            return weights, config, tokenizer

        except Exception as e:
            print(f"❌ Loading failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def get_model_path(
        self, model_name: str, model_type: str = "jax_converted"
    ) -> Path:
        """Get the path to a model directory"""
        if model_type not in ["hf_original", "jax_converted"]:
            raise ValueError("model_type must be 'hf_original' or 'jax_converted'")

        return self.cache_dir / model_name / model_type
