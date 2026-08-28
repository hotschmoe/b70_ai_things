#!/usr/bin/env python3
"""Validate current torch-2.13 XPU static-FP8 GEMM on a real model tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_KEY = "model.language_model.layers.0.linear_attn.in_proj_qkv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--m", default="1,8,128")
    return parser.parse_args()


def load_tensor(model: Path, key: str) -> torch.Tensor:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = model / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    delta = actual.float() - expected.float()
    rel_l2 = float(delta.norm() / expected.float().norm().clamp_min(1e-12))
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual.float().reshape(-1), expected.float().reshape(-1), dim=0
        )
    )
    max_abs = float(delta.abs().max())
    print(
        f"{name}: cosine={cosine:.8f} rel_l2={rel_l2:.8g} "
        f"max_abs={max_abs:.8g}"
    )
    if not torch.isfinite(actual).all():
        raise RuntimeError(f"{name}: non-finite output")
    if cosine < 0.999 or rel_l2 > 0.01:
        raise RuntimeError(f"{name}: failed reference tolerance")


def main() -> None:
    args = parse_args()
    weight = load_tensor(args.model, f"{args.key}.weight").to("xpu")
    weight_scale = load_tensor(
        args.model, f"{args.key}.weight_scale"
    ).max().to(device="xpu", dtype=torch.float32).reshape(1)
    input_scale = load_tensor(
        args.model, f"{args.key}.input_scale"
    ).max().to(device="xpu", dtype=torch.float32).reshape(1)
    weight_t = weight.t()

    print(
        f"torch={torch.__version__} device={torch.xpu.get_device_name(0)} "
        f"key={args.key} N={weight.shape[0]} K={weight.shape[1]}"
    )

    torch.manual_seed(20260828)
    fp8_limit = torch.finfo(torch.float8_e4m3fn).max
    for m in (int(value) for value in args.m.split(",")):
        x = torch.randn(m, weight.shape[1], dtype=torch.bfloat16, device="xpu")
        qinput = (
            x.float()
            .div(input_scale)
            .clamp(min=-fp8_limit, max=fp8_limit)
            .to(torch.float8_e4m3fn)
        )

        actual = torch._scaled_mm(
            qinput,
            weight_t,
            scale_a=input_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )
        if isinstance(actual, tuple):
            actual = actual[0]
        repeat = torch._scaled_mm(
            qinput,
            weight_t,
            scale_a=input_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )
        if isinstance(repeat, tuple):
            repeat = repeat[0]

        dequant_input = qinput.to(torch.bfloat16).mul(input_scale)
        dequant_weight = weight.to(torch.bfloat16).mul(weight_scale)
        reference = torch.nn.functional.linear(dequant_input, dequant_weight)
        torch.xpu.synchronize()

        if not torch.equal(actual, repeat):
            raise RuntimeError(f"M={m}: XPU scaled_mm is not deterministic")
        compare(f"M={m} scaled_mm", actual, reference)

    print("VERDICT: PASS")


if __name__ == "__main__":
    main()
