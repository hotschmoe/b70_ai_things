"""Compare folded-BF16 and native-E4M3 NVFP4 scale paths on one XPU.

The candidate op keeps checkpoint-native E4M3 block scales and supplies the
FP32 global weight scale as a separate oneDNN source scale. The incumbent op
folds both into a BF16 block-scale tensor at load time.

Run inside the vLLM XPU image with an experimental _xpu_C.abi3.so mounted.
The default tensor is a real Qwen3.6-27B gate projection.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from safetensors import safe_open

import vllm_xpu_kernels._xpu_C  # noqa: F401


DEFAULT_KEY = "model.language_model.layers.0.mlp.gate_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--m", default="1,2,4,6,8,16,64,512,2048")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    return parser.parse_args()


def load_triplet(path: str, key: str) -> tuple[torch.Tensor, ...]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(f"{key}.weight")
        block_scale = handle.get_tensor(f"{key}.weight_scale")
        global_scale = handle.get_tensor(f"{key}.weight_scale_2")
    return weight, block_scale, global_scale


def bench(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def median_rounds(fn, iters: int, warmup: int, rounds: int) -> float:
    samples = [bench(fn, iters, warmup) for _ in range(rounds)]
    return statistics.median(samples)


def main() -> None:
    args = parse_args()
    if not hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16_f8scale"):
        raise RuntimeError("experimental f8-scale op is not present in _xpu_C")

    weight_cpu, scale_cpu, global_cpu = load_triplet(args.file, args.key)
    weight = weight_cpu.to("xpu")
    scale = scale_cpu.to("xpu")
    global_scale = global_cpu.max().to(device="xpu", dtype=torch.float32)

    # Both oneDNN paths consume NT views. Contiguous transposes give scales the
    # exact [K/16,N] grouped-quant layout used by the production shim.
    weight_nt = weight.t()
    scale_f8_nt = scale.t().contiguous()
    scale_bf16_nt = (
        scale.to(torch.float32).mul(global_scale).to(torch.bfloat16).t().contiguous()
    )

    n, packed_k = weight.shape
    k = packed_k * 2
    print(
        f"device={torch.xpu.get_device_name(0)} key={args.key} "
        f"N={n} K={k} scale_dtype={scale.dtype} global={float(global_cpu.max()):.9g}"
    )
    print(
        f"bytes: packed={weight.numel()} f8_scale={scale_f8_nt.numel()} "
        f"bf16_scale={scale_bf16_nt.numel() * 2}"
    )
    print(
        f"{'M':>6} {'bf16_ms':>10} {'f8_ms':>10} {'speedup':>9} "
        f"{'max_abs':>11} {'rel_l2':>11}"
    )

    for m in [int(item) for item in args.m.split(",")]:
        x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16) * 0.1

        def folded():
            return torch.ops._xpu_C.nvfp4_gemm_w4a16(
                x, weight_nt, None, scale_bf16_nt, 16
            )

        def native_f8():
            return torch.ops._xpu_C.nvfp4_gemm_w4a16_f8scale(
                x, weight_nt, None, scale_f8_nt, global_scale, 16
            )

        old_y = folded()
        new_y = native_f8()
        torch.xpu.synchronize()
        delta = old_y.float() - new_y.float()
        max_abs = delta.abs().max().item()
        rel_l2 = delta.norm().item() / max(new_y.float().norm().item(), 1e-12)

        # A-B-A order makes thermal drift visible without favoring the new op.
        old_a = median_rounds(folded, args.iters, args.warmup, args.rounds)
        new_t = median_rounds(native_f8, args.iters, args.warmup, args.rounds)
        old_b = median_rounds(folded, args.iters, args.warmup, args.rounds)
        old_t = (old_a + old_b) / 2.0
        print(
            f"{m:6d} {old_t:10.4f} {new_t:10.4f} {old_t / new_t:9.4f} "
            f"{max_abs:11.4g} {rel_l2:11.4g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
