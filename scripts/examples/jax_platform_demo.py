#!/usr/bin/env python3
"""JAX Platform Demonstration
Shows the complete integration of JAX models following jax-llm-examples patterns
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from oumi.models.experimental.jax_models import (
    JAXModelManager,
    get_recommended_model,
    get_supported_models,
)


def main():
    """Demonstrate the JAX platform integration"""
    print("🚀 JAX Models Platform Demonstration")
    print("=" * 60)

    # Initialize manager
    manager = JAXModelManager()
    print(f"📂 Cache directory: {manager.cache_dir}")

    # List available models
    print("\n📋 Available Models:")
    models = get_supported_models()
    for name, info in list(models.items())[:3]:  # Show first 3
        print(f"   • {name}: {info.description} ({info.size_gb}GB)")
    print(f"   ... and {len(models) - 3} more models")

    # Get recommendation
    print("\n✨ Recommended Model (small, public):")
    recommended = get_recommended_model(max_size_gb=5.0, requires_no_auth=True)
    if recommended:
        print(f"   {recommended.model_id}")
        print(f"   Architecture: {recommended.architecture}")
        print(f"   Size: {recommended.size_gb}GB")
        print(f"   Hardware: {recommended.recommended_hardware}")

    print("\n🎯 Platform Features:")
    print("   ✅ Unified model registry with 11+ supported models")
    print("   ✅ Automatic download from HuggingFace")
    print("   ✅ Automatic conversion to JAX format")
    print("   ✅ Architecture-specific implementations")
    print("   ✅ CLI interface for easy usage")
    print("   ✅ Following jax-llm-examples patterns exactly")

    print("\n📖 Usage Examples:")
    print("   # List all models")
    print("   python -m oumi.models.experimental.jax_models list")
    print()
    print("   # Get recommendations")
    print(
        "   python -m oumi.models.experimental.jax_models recommend --max-size-gb 10 --requires-no-auth"
    )
    print()
    print("   # Download and run inference")
    print("   python -m oumi.models.experimental.jax_models run qwen2.5-0.5b-instruct")
    print()
    print("   # Or use the Python API")
    print("   from oumi.models.experimental.jax_models import JAXModelManager")
    print("   manager = JAXModelManager()")
    print("   weights, config, tokenizer = manager.load_model('qwen2.5-0.5b-instruct')")

    print("\n🏗️  Architecture Support:")
    architectures = {info.architecture for info in models.values()}
    for arch in sorted(architectures):
        arch_models = [
            name for name, info in models.items() if info.architecture == arch
        ]
        print(f"   • {arch}: {len(arch_models)} models")

    print("\n✨ Platform successfully integrated following jax-llm-examples patterns!")
    print(f"   Total: {len(models)} models across {len(architectures)} architectures")


if __name__ == "__main__":
    main()
