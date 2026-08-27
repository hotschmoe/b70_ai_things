#!/usr/bin/env python3
"""Deterministic Qwen3.8 NVFP4 operator oracle for the vLLM 0.28 port."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

import vllm_xpu_kernels._xpu_C  # noqa: F401


E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
DEFAULT_KEY = "model.language_model.layers.0.mlp.gate_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--m", default="1,8")
    return parser.parse_args()


def load_tensor(model: Path, key: str) -> torch.Tensor:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = model / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def dequantize(
    packed: torch.Tensor, block_scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    n, packed_k = packed.shape
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    nibble = torch.stack((lo, hi), dim=-1).reshape(n, packed_k * 2).long()
    values = E2M1[nibble]
    scales = block_scale.float().repeat_interleave(16, dim=1)
    return (values * scales * global_scale.float()).to(torch.bfloat16)


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    delta = actual_f - expected_f
    rel_l2 = float(delta.norm() / expected_f.norm().clamp_min(1e-12))
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_f.reshape(-1), expected_f.reshape(-1), dim=0
        )
    )
    max_abs = float(delta.abs().max())
    print(
        f"{name}: cosine={cosine:.8f} rel_l2={rel_l2:.8g} "
        f"max_abs={max_abs:.8g}"
    )
    if not torch.isfinite(actual_f).all():
        raise RuntimeError(f"{name}: non-finite output")
    if cosine < 0.999 or rel_l2 > 0.01:
        raise RuntimeError(f"{name}: failed reference tolerance")


def main() -> None:
    args = parse_args()
    model = Path(args.model)
    packed = load_tensor(model, f"{args.key}.weight")
    block_scale = load_tensor(model, f"{args.key}.weight_scale")
    global_scale = load_tensor(model, f"{args.key}.weight_scale_2").reshape(1)
    weight_ref = dequantize(packed, block_scale, global_scale).to("xpu")

    packed_nt = packed.to("xpu").t()
    folded_scale_nt = (
        block_scale.float()
        .mul(global_scale.float())
        .to(torch.bfloat16)
        .t()
        .contiguous()
        .to("xpu")
    )
    native_scale_nt = block_scale.t().contiguous().to("xpu")
    global_scale_xpu = global_scale.to(device="xpu", dtype=torch.float32)

    print(
        f"torch={torch.__version__} device={torch.xpu.get_device_name(0)} "
        f"key={args.key} N={packed.shape[0]} K={packed.shape[1] * 2}"
    )
    print(
        "ops="
        f"folded:{hasattr(torch.ops._xpu_C, 'nvfp4_gemm_w4a16')},"
        "native_f8:"
        f"{hasattr(torch.ops._xpu_C, 'nvfp4_gemm_w4a16_f8scale')}"
    )

    torch.manual_seed(20260827)
    for m in (int(value) for value in args.m.split(",")):
        x = torch.randn(m, packed.shape[1] * 2, dtype=torch.bfloat16).to("xpu")
        reference = torch.nn.functional.linear(x, weight_ref)
        folded = torch.ops._xpu_C.nvfp4_gemm_w4a16(
            x, packed_nt, None, folded_scale_nt, 16
        )
        native = torch.ops._xpu_C.nvfp4_gemm_w4a16_f8scale(
            x, packed_nt, None, native_scale_nt, global_scale_xpu, 16
        )
        folded_repeat = torch.ops._xpu_C.nvfp4_gemm_w4a16(
            x, packed_nt, None, folded_scale_nt, 16
        )
        torch.xpu.synchronize()

        if not torch.equal(folded, folded_repeat):
            raise RuntimeError(f"M={m}: folded operator is not deterministic")
        compare(f"M={m} folded", folded, reference)
        compare(f"M={m} native_f8", native, reference)

    print("VERDICT: PASS")


if __name__ == "__main__":
    main()
