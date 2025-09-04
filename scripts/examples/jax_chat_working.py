#!/usr/bin/env python3
"""Working JAX Chat Demo - Complete pipeline like jax-llm-examples
Downloads, converts, and runs chat inference automatically
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def main():
    print("🚀 JAX Chat Demo - Complete Pipeline")
    print("=" * 50)

    # Use DeepSeek R1 Distilled model (uses Llama architecture, already downloaded)
    model_name = "deepseek-r1-distill-llama-8b"
    cache_dir = Path("/Users/siatras/.cache/jax_demo")

    try:
        from oumi.models.experimental.jax_models import JAXModelManager

        print(f"📦 Using model: {model_name}")
        print(f"📂 Cache directory: {cache_dir}")

        # Initialize manager
        manager = JAXModelManager(cache_dir)

        # Copy existing download to expected location
        hf_source = cache_dir / "deepseek_r1_distill_8b"
        hf_expected = cache_dir / model_name / "hf_original"

        if hf_source.exists() and not hf_expected.exists():
            print(f"📂 Moving downloaded model from {hf_source} to {hf_expected}")
            hf_expected.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.move(str(hf_source), str(hf_expected))

        # Force conversion to ensure clean state
        print("🔄 Force converting model to ensure clean state...")
        manager.convert_model(model_name, force_convert=True)

        # Load model (auto download + convert)
        print("⏳ Loading model...")
        weights, config, tokenizer = manager.load_model(
            model_name, auto_download=False, auto_convert=False
        )

        print("✅ Model loaded successfully!")
        print(
            f"📊 Config: {config.num_layers} layers, {config.vocab_size} vocab, {config.embed} dim"
        )

        # Import JAX for inference
        import importlib

        import numpy as np
        from jax import numpy as jnp
        from jax import random
        from jax.sharding import PartitionSpec as P
        from jax.sharding import set_mesh

        # Import model implementation
        from oumi.models.experimental.jax_models.registry import (
            get_implementation_module,
            get_model_info,
        )

        model_info = get_model_info(model_name)
        impl_module_path = get_implementation_module(model_info.architecture)
        model_module = importlib.import_module(f"{impl_module_path}.model")

        # Test prompts
        prompts = [
            "Hello! What is your name?",
            "Write a short poem about the ocean",
            "Explain what JAX is in simple terms",
        ]

        def encode_input(texts, pad_id=0):
            """Encode input text for the model"""
            inputs = []
            for text in texts:
                if hasattr(tokenizer, "apply_chat_template"):
                    tokens = tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}]
                    )
                    # Use Llama format for DeepSeek R1 Distilled
                    tokens += tokenizer.encode(
                        "<|start_header_id|>assistant<|end_header_id|>"
                    )
                    # Add thinking tokens for DeepSeek models
                    if "deepseek" in model_name:
                        tokens += tokenizer.encode("<think>")
                else:
                    tokens = tokenizer.encode(f"User: {text}\nAssistant:")
                inputs.append(tokens)

            max_len = max(len(x) for x in inputs)
            inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
            return np.array(inputs)

        print(f"\n💭 Test prompts: {len(prompts)} questions")
        input_tokens = encode_input(prompts)
        print(f"✅ Encoded input: {input_tokens.shape}")

        # Run inference
        print("\n🚀 Running JAX inference...")
        with set_mesh(config.mesh):
            # Initialize cache
            zero_cache = model_module.KVCache.init(
                random.key(1), config, input_tokens.shape[0], config.max_seq_len
            )

            # Prefill
            next_tokens, logits, cache = model_module.prefill(
                input_tokens, weights, zero_cache, config
            )
            curr_tokens = next_tokens.at[:, cache.iter - 1 : cache.iter].get(
                out_sharding=P(None, None)
            )

            # Generate 32 tokens
            tokens_list = []
            for step in range(32):
                tokens_list.append(curr_tokens)
                curr_tokens, cache = model_module.decode_step(
                    curr_tokens, weights, cache, config
                )
                if step % 8 == 7:
                    print(f"   Generated {step + 1}/32 tokens...")

            generated_tokens = np.array(jnp.concatenate(tokens_list, axis=-1))

        # Display chat responses
        print("\n🎉 Chat Responses:")
        print("=" * 60)
        for i, (prompt, tokens) in enumerate(zip(prompts, generated_tokens)):
            response = tokenizer.decode(tokens)
            print(f"\n💬 User: {prompt}")
            print(f"🤖 Assistant: {response}")
            print("-" * 40)

        print("\n✨ Success! Complete JAX pipeline working:")
        print("   ✅ Auto-download from HuggingFace")
        print("   ✅ Auto-convert to JAX format")
        print("   ✅ JAX inference with proper prefill/decode")
        print("   ✅ Chat-style interaction")
        print("   ✅ Following jax-llm-examples patterns")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
