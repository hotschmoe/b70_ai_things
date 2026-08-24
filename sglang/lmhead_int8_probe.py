#!/usr/bin/env python3
"""Gate an INT8 W8A8 LM head at the exact TP=2 shard shape."""

import ctypes
import os
import time

import torch
from safetensors import safe_open


SO = os.environ.get("B70_XPU_C_SO", "/work/kernel/_xpu_C.abi3.so")
CKPT = os.environ.get("CKPT", "/models/qwen3.6-27b/w8a8-sqgptq/model.safetensors")
RANK = int(os.environ.get("RANK", "0"))
TP = int(os.environ.get("TP", "2"))
CHUNK_ROWS = int(os.environ.get("CHUNK_ROWS", "8192"))


def sync():
    torch.xpu.synchronize()


def bench(fn, warm=12, iterations=40):
    for _ in range(warm):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    sync()
    return 1000.0 * (time.perf_counter() - start) / iterations


def quantize_per_channel(weight):
    rows, cols = weight.shape
    quant = torch.empty((rows, cols), dtype=torch.int8, device=weight.device)
    scale = torch.empty(rows, dtype=torch.float16, device=weight.device)
    error_sq = 0.0
    reference_sq = 0.0
    for row0 in range(0, rows, CHUNK_ROWS):
        row1 = min(row0 + CHUNK_ROWS, rows)
        values = weight[row0:row1].to(torch.float32)
        scales = values.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8) / 127.0
        q = torch.round(values / scales).clamp_(-127, 127).to(torch.int8)
        served_scales = scales.to(torch.float16)
        quant[row0:row1] = q
        scale[row0:row1] = served_scales.reshape(-1)
        dequant = q.to(torch.float32) * served_scales.to(torch.float32)
        error_sq += float(torch.sum((dequant - values) ** 2).item())
        reference_sq += float(torch.sum(values**2).item())
    return quant, scale, (error_sq / reference_sq) ** 0.5


def w8a16(x, weight_nt, weight_scale):
    return torch.ops._xpu_C.int8_gemm_w8a16(
        x.to(torch.float16).contiguous(), weight_nt, weight_scale, None
    )


def w8a8(x, weight_nt, weight_scale):
    xf = x.to(torch.float16).contiguous()
    q, scale, _ = torch.ops._xpu_C.dynamic_per_token_int8_quant(xf, True, 8)
    return torch.ops._xpu_C.int8_gemm_w8a8(
        q, scale, None, weight_nt, weight_scale, None, None, torch.float16
    )


def main():
    if TP <= 0 or not 0 <= RANK < TP:
        raise SystemExit(f"invalid TP/RANK: TP={TP} RANK={RANK}")
    ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)
    required = (
        "dynamic_per_token_int8_quant",
        "int8_gemm_w8a16",
        "int8_gemm_w8a8",
    )
    missing = [name for name in required if not hasattr(torch.ops._xpu_C, name)]
    if missing:
        raise SystemExit(f"missing operators: {missing}")

    with safe_open(CKPT, framework="pt", device="cpu") as handle:
        full = handle.get_slice("lm_head.weight")
        vocab, hidden = full.get_shape()
        if vocab % TP:
            raise SystemExit(f"vocab={vocab} is not divisible by TP={TP}")
        shard_rows = vocab // TP
        row0 = RANK * shard_rows
        row1 = row0 + shard_rows
        host_weight = full[row0:row1, :]

    weight = host_weight.to("xpu")
    print(
        f"CONFIG\tfull={vocab}x{hidden}\trank={RANK}/{TP}\t"
        f"shard={tuple(weight.shape)}\tdtype={weight.dtype}",
        flush=True,
    )
    quant_nk, weight_scale, weight_rel_l2 = quantize_per_channel(weight)
    weight_nt = quant_nk.t()
    if weight_nt.stride(0) != 1:
        raise SystemExit(f"weight NT stride mismatch: {weight_nt.stride()}")
    print(
        f"QUANT\tweight_rel_l2={weight_rel_l2:.8f}\t"
        f"bf16_gib={weight.numel() * 2 / 2**30:.3f}\t"
        f"int8_gib={(quant_nk.numel() + 2 * weight_scale.numel()) / 2**30:.3f}",
        flush=True,
    )

    torch.manual_seed(20260824)
    for rows in (1, 11):
        x = torch.randn((rows, hidden), dtype=torch.bfloat16, device="xpu") * 0.05
        reference = x @ weight.t()
        reference_f32 = reference.to(torch.float32)
        baseline_ms = bench(lambda: x @ weight.t())
        candidates = {
            "w8a16": w8a16(x, weight_nt, weight_scale),
            "w8a8": w8a8(x, weight_nt, weight_scale),
        }
        for route, candidate in candidates.items():
            candidate_f32 = candidate.to(torch.float32)
            rel_l2 = float(torch.linalg.vector_norm(candidate_f32 - reference_f32).item())
            rel_l2 /= float(torch.linalg.vector_norm(reference_f32).item())
            top1 = float(
                (reference.argmax(-1) == candidate.argmax(-1)).float().mean().item()
            )
            candidate_ms = bench(
                (lambda: w8a16(x, weight_nt, weight_scale))
                if route == "w8a16"
                else (lambda: w8a8(x, weight_nt, weight_scale))
            )
            print(
                f"RESULT\tM={rows}\troute={route}\tbf16_ms={baseline_ms:.4f}\t"
                f"int8_ms={candidate_ms:.4f}\tspeedup={baseline_ms / candidate_ms:.3f}\t"
                f"rel_l2={rel_l2:.8f}\ttop1_agreement={top1:.6f}\t"
                f"finite={bool(torch.isfinite(candidate).all())}",
                flush=True,
            )
        cross_top1 = float(
            (candidates["w8a16"].argmax(-1) == candidates["w8a8"].argmax(-1))
            .float()
            .mean()
            .item()
        )
        cross_rel_l2 = float(
            torch.linalg.vector_norm(
                candidates["w8a16"].to(torch.float32)
                - candidates["w8a8"].to(torch.float32)
            ).item()
        ) / float(torch.linalg.vector_norm(candidates["w8a16"].to(torch.float32)).item())
        print(
            f"CROSS\tM={rows}\tw8a16_vs_w8a8_rel_l2={cross_rel_l2:.8f}\t"
            f"top1_agreement={cross_top1:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
