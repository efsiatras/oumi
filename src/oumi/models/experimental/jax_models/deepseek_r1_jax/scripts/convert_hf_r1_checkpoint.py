#!/usr/bin/env python3
# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert HuggingFace DeepSeek R1 model to JAX format
Based on jax-llm-examples/deepseek_r1_jax/scripts/convert_hf_r1_checkpoint.py
"""

import dataclasses
import gzip
import json
from argparse import ArgumentParser
from pathlib import Path

import jax
from jax.sharding import PartitionSpec as P


def main():
    """Convert DeepSeek R1 checkpoint to JAX format"""
    parser = ArgumentParser()
    parser.add_argument(
        "--root-dir", required=True, help="Directory with *.safetensors files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Name of the output directory"
    )
    args = parser.parse_args()

    try:
        from deepseek_r1_jax import chkpt_utils as utils
        from deepseek_r1_jax.model import Config, ShardingRules
    except ImportError:
        # Try relative import for our structure
        import sys

        sys.path.append(str(Path(__file__).parent.parent.absolute()))
        from deepseek_r1_jax import chkpt_utils as utils
        from deepseek_r1_jax.model import Config, ShardingRules

    root_path, dest_path = Path(args.root_dir), Path(args.output_dir)

    # Create default config
    cfg = Config()
    cfg.quantize_mlp = False
    cfg.quantize_attn = True
    cfg.quantize_moe = True

    # Create mesh with fully replicated sharding
    rules = ShardingRules(*(None for _ in dataclasses.fields(ShardingRules)))
    cfg = dataclasses.replace(cfg, mesh=jax.make_mesh((1,), P("x")), rules=rules)

    # Load params map
    params_map_path = Path(__file__).parent / "r1_hf_ckpt_params_map.json.gz"
    if not params_map_path.exists():
        raise FileNotFoundError(f"Params map not found: {params_map_path}")

    params_map = json.loads(gzip.decompress(params_map_path.read_bytes()))

    print(f"📂 Using params map with {len(params_map)} entries")
    print("🔄 Converting DeepSeek R1 checkpoint...")

    utils.convert_hf_checkpoint(params_map, root_path, dest_path, cfg)

    print(f"✅ Conversion complete! JAX model saved to: {dest_path}")


if __name__ == "__main__":
    main()
