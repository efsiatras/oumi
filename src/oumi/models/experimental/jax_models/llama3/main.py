#!/usr/bin/env python3
"""Minimal Llama 3 inference using JAX
Based on jax-llm-examples/llama3/main.py
"""

import dataclasses
import json
from pathlib import Path
from pprint import pprint

import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from jax.sharding import AxisType, set_mesh
from jax.sharding import PartitionSpec as P
from llama3_jax import model as l3jax


def encode_input(tokenizer, texts: list[str], model_name: str, pad_id: int = 0):
    """Encode input text for the model"""
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}])
        + tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>")
        + ([] if "deepseek" not in model_name.lower() else tokenizer.encode("<think>"))
        for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
    return np.array(inputs)


if __name__ == "__main__":
    # Configuration
    quant = False  # Set to True for quantized inference

    # Model path - update this to your converted JAX model directory
    ckpt_path = Path(
        "~/models/jax/deepseek-ai--DeepSeek-R1-Distill-Llama-8B"
    ).expanduser()

    if not ckpt_path.exists():
        print("❌ Model not found. Please:")
        print("1. Download model:")
        print(
            "   python scripts/download_model.py --model-id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dest-root-path ~/models/hf/"
        )
        print("2. Convert to JAX:")
        print(
            "   python scripts/convert_weights.py --source-path ~/models/hf/deepseek-ai--DeepSeek-R1-Distill-Llama-8B --dest-path ~/models/jax/deepseek-ai--DeepSeek-R1-Distill-Llama-8B"
        )
        exit(1)

    if quant:
        ckpt_path = ckpt_path.parent / f"{ckpt_path.name}-quant"

    print(f"🚀 Loading model from: {ckpt_path}")

    # Load tokenizer
    tokenizer = l3jax.load_tokenizer(
        ckpt_path / "tokenizer.json", ckpt_path / "tokenizer_config.json"
    )
    print("✅ Loaded tokenizer")

    # Create mesh
    mesh = jax.make_mesh(
        (1, 1, jax.device_count()),
        ("x", "y", "z"),
        devices=jax.devices(),
        axis_types=(AxisType.Explicit,) * 3,
    )
    print(f"✅ Created mesh with {jax.device_count()} devices")

    # Load config and create JAX config
    cfg = l3jax.llama_to_jax_config(json.loads((ckpt_path / "config.json").read_text()))
    cfg = dataclasses.replace(cfg, mesh=mesh, quant_layer=quant, quant_cache=quant)
    print(f"✅ Config: {cfg.num_layers} layers, {cfg.vocab_size} vocab")

    # Load weights
    weights = l3jax.load_pytree(ckpt_path, l3jax.Weights.shardings(cfg))
    print("✅ Loaded model weights")

    # Prepare input
    input = encode_input(
        tokenizer,
        [
            "Tell me your name",
            "What is the weather like expressed in long prose in Old English",
            "Do you like ice cream, be extremely precise",
        ],
        model_name=ckpt_path.name,
    )
    print(f"✅ Encoded input: {input.shape}")

    # Run inference
    print("🚀 Running JAX inference...")
    with set_mesh(cfg.mesh):
        zero_cache = l3jax.KVCache.init(
            random.key(1), cfg, input.shape[0], cfg.max_seq_len
        )
        next_tokens, logits, cache = l3jax.prefill(input, weights, zero_cache, cfg)
        curr_tokens = next_tokens.at[:, cache.iter - 1 : cache.iter].get(
            out_sharding=P(None, None)
        )

        tokens_list = []
        for _ in range(16):
            tokens_list.append(curr_tokens)
            curr_tokens, cache = l3jax.decode_step(curr_tokens, weights, cache, cfg)
        tokens = np.array(jnp.concatenate(tokens_list, axis=-1))

    # Decode responses
    responses = [tokenizer.decode(row) for row in tokens]
    print("🎉 Responses:")
    pprint(responses)
