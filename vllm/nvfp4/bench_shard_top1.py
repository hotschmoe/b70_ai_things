"""Correctness and latency probe for the fused XPU shard top-1 operator."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,2,4,8")
    parser.add_argument("--width", type=int, default=124160)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=7)
    return parser.parse_args()


def elapsed_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def median_ms(fn, warmup: int, iters: int, rounds: int) -> float:
    return statistics.median(
        elapsed_ms(fn, warmup, iters) for _ in range(rounds)
    )


def main() -> None:
    args = parse_args()
    if not hasattr(torch.ops._xpu_C, "xpu_shard_top1"):
        raise RuntimeError("xpu_shard_top1 is not present in _xpu_C")

    print(
        f"device={torch.xpu.get_device_name(0)} width={args.width} "
        f"dtype=bf16 iters={args.iters} rounds={args.rounds}"
    )
    print(
        f"{'rows':>5} {'fused_ms':>10} {'arg_gather_ms':>14} "
        f"{'max_ms':>10} {'speedup':>9} {'value_bad':>10} {'index_bad':>10}"
    )

    for rows in [int(value) for value in args.rows.split(",")]:
        torch.manual_seed(1701 + rows)
        logits = torch.randn(
            rows, args.width, device="xpu", dtype=torch.bfloat16
        )
        # Force deterministic unique maxima, including both ends of the shard.
        expected_indices = torch.arange(rows, device="xpu") * 7919
        expected_indices %= args.width
        logits[torch.arange(rows, device="xpu"), expected_indices] = 64

        def fused():
            return torch.ops._xpu_C.xpu_shard_top1(logits)

        def arg_gather():
            indices = logits.argmax(dim=-1)
            values = logits.gather(-1, indices.unsqueeze(-1)).squeeze(-1).float()
            return values, indices

        def stock_max():
            return logits.max(dim=-1)

        fused_values, fused_indices = fused()
        reference_values, reference_indices = arg_gather()
        torch.xpu.synchronize()
        value_bad = int((fused_values != reference_values).sum().item())
        index_bad = int((fused_indices != reference_indices).sum().item())
        expected_bad = int((fused_indices != expected_indices).sum().item())
        if value_bad or index_bad or expected_bad:
            raise RuntimeError(
                "correctness failure: "
                f"value_bad={value_bad} index_bad={index_bad} "
                f"expected_bad={expected_bad}"
            )

        fused_ms = median_ms(fused, args.warmup, args.iters, args.rounds)
        arg_ms = median_ms(arg_gather, args.warmup, args.iters, args.rounds)
        max_ms = median_ms(stock_max, args.warmup, args.iters, args.rounds)
        print(
            f"{rows:5d} {fused_ms:10.5f} {arg_ms:14.5f} "
            f"{max_ms:10.5f} {arg_ms / fused_ms:9.3f} "
            f"{value_bad:10d} {index_bad:10d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
