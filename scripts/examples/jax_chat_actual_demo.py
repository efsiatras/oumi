#!/usr/bin/env python3
"""JAX Chat Demo - Using existing converted model
Shows actual text generation with JAX models
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def main():
    print("🚀 JAX Chat Demo - Real Text Generation")
    print("=" * 55)

    # Use existing converted DeepSeek R1 8B JAX model
    model_path = Path("/Users/siatras/.cache/jax_demo/deepseek_r1_8b_jax")

    if not model_path.exists():
        print("❌ No converted JAX model found")
        return

    print(f"📂 Using converted JAX model: {model_path}")

    try:
        # Import JAX and model
        import json

        import jax
        import numpy as np
        from jax import numpy as jnp
        from jax import random
        from jax.sharding import PartitionSpec as P
        from jax.sharding import set_mesh

        # Import DeepSeek R1 JAX implementation
        from oumi.models.experimental.jax_models.deepseek_r1_jax.deepseek_r1_jax import (
            model as r1jax,
        )

        print("✅ JAX and model imports successful")

        # Load tokenizer
        tokenizer = r1jax.load_tokenizer(
            model_path / "tokenizer.json", model_path / "tokenizer_config.json"
        )
        print("✅ Loaded tokenizer")

        # Load config
        with open(model_path / "config.json") as f:
            hf_config = json.load(f)

        config = r1jax.hf_to_jax_config(hf_config)
        print(f"✅ Config: {config.num_layers} layers, {config.vocab_size} vocab")

        # Create mesh for single device
        mesh = jax.make_mesh((1,), ("x",))
        import dataclasses

        config = dataclasses.replace(config, mesh=mesh)

        print(f"✅ Created mesh with {jax.device_count()} device(s)")

        # Load JAX weights
        weights = r1jax.load_pytree(model_path, r1jax.Weights.shardings(config))
        print("✅ Loaded JAX weights successfully")

        # Prepare test prompts
        prompts = [
            "Hello! My name is",
            "The weather today is",
            "In the future, AI will",
        ]

        print(f"\n💭 Testing with {len(prompts)} prompts")

        # Encode inputs
        def encode_simple(texts, max_len=50):
            """Simple encoding for demo"""
            inputs = []
            for text in texts:
                # Simple tokenization
                tokens = tokenizer.encode(text)
                # Pad/truncate to max_len
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                else:
                    tokens = tokens + [tokenizer.pad_token_id or 0] * (
                        max_len - len(tokens)
                    )
                inputs.append(tokens)
            return np.array(inputs)

        input_tokens = encode_simple(prompts)
        print(f"✅ Encoded inputs: {input_tokens.shape}")

        # Run JAX inference
        print("\n🚀 Running JAX inference...")

        with set_mesh(config.mesh):
            # Initialize KV cache
            batch_size, seq_len = input_tokens.shape
            zero_cache = r1jax.KVCache.init(
                random.key(42), config, batch_size, seq_len + 20
            )

            # Run prefill
            next_tokens, logits, cache = r1jax.prefill(
                input_tokens, weights, zero_cache, config
            )

            # Get initial tokens
            curr_tokens = next_tokens.at[:, cache.iter - 1 : cache.iter].get(
                out_sharding=P(None, None)
            )

            print("✅ Prefill successful")

            # Generate tokens
            generated_tokens = []
            for step in range(10):  # Generate 10 tokens
                generated_tokens.append(curr_tokens)
                curr_tokens, cache = r1jax.decode_step(
                    curr_tokens, weights, cache, config
                )

                if step % 3 == 2:
                    print(f"   Generated {step + 1}/10 tokens...")

            # Concatenate generated tokens
            all_generated = np.array(jnp.concatenate(generated_tokens, axis=-1))

        print("✅ Generation complete!")

        # Decode and display results
        print("\n🎉 Generated Text:")
        print("=" * 55)

        for i, (prompt, generated) in enumerate(zip(prompts, all_generated)):
            try:
                generated_text = tokenizer.decode(generated)
                print(f"\n💬 Prompt {i + 1}: {prompt}")
                print(f"🤖 Generated: {generated_text}")
                print("-" * 40)
            except Exception as e:
                print(f"❌ Decode error for prompt {i + 1}: {e}")

        print("\n✨ SUCCESS! JAX model generated actual text!")
        print("   Using DeepSeek R1 8B with JAX compilation")
        print("   Real inference pipeline working end-to-end")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
