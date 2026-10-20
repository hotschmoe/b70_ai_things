#!/usr/bin/env python3
"""Conservative Flash-Next 12+36 grouped-pool capacity (no torch/GPU).

Assumes the observed three GDN groups of 12 layers, align mode (two state
blocks/group/live stream), and no MTP/prefill checkpoints. A null block is
held back. Verify the actual KV specs when changing backend or quant format.
"""
import argparse
import json
import math


def capacity(budget, dtype="bf16", streams=1, block_size=None):
    token_bytes = {"bf16": 2048, "fp8": 1024, "fp8_head": 1040, "qsa": 1184}[dtype]
    state_bytes = 48*128*128*4 + 10240*3*2
    block_size = block_size or math.ceil(state_bytes/(64*token_bytes))*64
    pool_page = 12 * block_size * token_bytes
    blocks = budget // pool_page
    attention_blocks = max(0, (blocks-1-6*streams)//streams)
    tokens = attention_blocks * block_size
    return dict(budget_bytes=budget, block_tokens=block_size,
                pool_bytes_per_block=pool_page, pool_blocks=blocks,
                streams=streams, tokens_per_stream=tokens,
                total_tokens=tokens*streams,
                target_200k_per_stream_required_bytes=(1+streams*(math.ceil(200000/block_size)+6))*pool_page)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("budget", type=int)
    p.add_argument("--dtype", choices=("bf16", "fp8", "fp8_head", "qsa"), default="bf16")
    p.add_argument("--streams", type=int, default=1)
    p.add_argument("--block-size", type=int)
    args = p.parse_args()
    print(json.dumps(capacity(args.budget, args.dtype, args.streams, args.block_size), indent=2))


if __name__ == "__main__":
    main()
