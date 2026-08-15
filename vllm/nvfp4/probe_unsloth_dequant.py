#!/usr/bin/env python3
"""CPU dequant probe: Unsloth CT NVFP4 vs Inferact ModelOpt vs official BF16.

Compares one MLP linear (default layer0.gate_proj) under several unpack /
scale conventions. No GPU. Prints cosine / RMSE / max-abs vs BF16.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def st_header(path: Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    return n, meta


def load_raw(path: Path, key: str) -> tuple[np.ndarray, str, list]:
    """Load a safetensors tensor as raw uint8 / decoded numpy, no torch."""
    hdr_n, meta = st_header(path)
    info = meta[key]
    dtype = info["dtype"]
    shape = info["shape"]
    start, end = info["data_offsets"]
    with path.open("rb") as f:
        f.seek(8 + hdr_n + start)
        buf = f.read(end - start)
    if dtype == "U8":
        return np.frombuffer(buf, dtype=np.uint8).reshape(shape), dtype, shape
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32).reshape(shape), dtype, shape
    if dtype == "F16":
        return np.frombuffer(buf, dtype=np.float16).reshape(shape).astype(np.float32), dtype, shape
    if dtype == "BF16":
        u16 = np.frombuffer(buf, dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape), dtype, shape
    if dtype in ("F8_E4M3", "F8_E4M3FN"):
        return np.frombuffer(buf, dtype=np.uint8).reshape(shape), dtype, shape
    raise ValueError(f"unhandled dtype {dtype} for {key}")

E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

ROOT = Path("/models") if Path("/models/qwen3.8-27b").is_dir() else Path(
    "/mnt/vm_8tb/github/b70_ai_things/models/files"
)
UNSLOTH = ROOT / "qwen3.8-27b/nvfp4-unsloth/model.safetensors"
MODELOPT = ROOT / "qwen3.8-27b/nvfp4-modelopt"
BF16_DIR = ROOT / "qwen3.8-27b/bf16"


def f8e4m3_to_f32(u8: np.ndarray) -> np.ndarray:
    """IEEE float8 e4m3fn (no inf; 0x7F/0xFF are NaN)."""
    u8 = u8.astype(np.uint8)
    sign = np.where((u8 & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exp = ((u8 >> 3) & 0x0F).astype(np.int32)
    mant = (u8 & 0x07).astype(np.int32)
    out = np.empty(u8.shape, dtype=np.float32)
    sub = exp == 0
    out[sub] = sign[sub] * (mant[sub].astype(np.float32) / 8.0) * (2.0 ** -6)
    nan = (exp == 15) & (mant == 7)
    out[nan] = np.nan
    norm = ~sub & ~nan
    out[norm] = sign[norm] * (1.0 + mant[norm].astype(np.float32) / 8.0) * (
        2.0 ** (exp[norm] - 7).astype(np.float32)
    )
    return out


def unpack_e2m1(packed: np.ndarray, high_first: bool) -> np.ndarray:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    if high_first:
        nib = np.stack([hi, lo], axis=-1)
    else:
        nib = np.stack([lo, hi], axis=-1)
    return E2M1[nib.reshape(packed.shape[0], -1)]


def dequant_nvfp4(packed, scale_f8, global_scale, high_first=False, invert_global=False):
    w = unpack_e2m1(packed, high_first=high_first)
    s = f8e4m3_to_f32(scale_f8)
    g = float(np.asarray(global_scale).reshape(-1)[0])
    if invert_global:
        g = 1.0 / g
    # scale [N, K/16] -> [N, K]
    s_exp = np.repeat(s, 16, axis=1)
    return (w * s_exp * g).astype(np.float32)


def stats(name, a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    cos = float(np.dot(a, b) / (an * bn + 1e-12))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mx = float(np.max(np.abs(a - b)))
    print(f"  {name:28s}  cos={cos:7.4f}  rmse={rmse:10.5f}  maxabs={mx:10.5f}  "
          f"|a|={an:.4g} |b|={bn:.4g}")
    return cos


def load_bf16(key: str) -> np.ndarray:
    idx = json.loads((BF16_DIR / "model.safetensors.index.json").read_text())
    shard = BF16_DIR / idx["weight_map"][key]
    arr, _, _ = load_raw(shard, key)
    return arr


def load_modelopt(key_prefix: str):
    idx = json.loads((MODELOPT / "model.safetensors.index.json").read_text())

    def grab(suffix):
        k = key_prefix + suffix
        shard = MODELOPT / idx["weight_map"][k]
        arr, _, _ = load_raw(shard, k)
        return arr

    return grab("weight"), grab("weight_scale"), grab("weight_scale_2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="model.language_model.layers.0.mlp.gate_proj")
    args = ap.parse_args()
    prefix = args.layer
    print("layer", prefix)

    packed, _, pshape = load_raw(UNSLOTH, prefix + ".weight_packed")
    scale, sc_dt, sshape = load_raw(UNSLOTH, prefix + ".weight_scale")
    wgs, _, _ = load_raw(UNSLOTH, prefix + ".weight_global_scale")
    igs, _, _ = load_raw(UNSLOTH, prefix + ".input_global_scale")
    print("unsloth packed", pshape, packed.dtype, "scale", sshape, sc_dt)
    print("unsloth wgs", float(wgs.reshape(-1)[0]), "igs", float(igs.reshape(-1)[0]))
    print("packed nibble hist (first 1M bytes):")
    sample = packed.reshape(-1)[:1_000_000]
    lo = sample & 0x0F
    hi = (sample >> 4) & 0x0F
    for name, nib in (("lo", lo), ("hi", hi)):
        hist = np.bincount(nib, minlength=16)
        print(f"  {name}", hist.tolist())

    bf = load_bf16(prefix + ".weight")
    print("bf16", bf.shape, "meanabs", float(np.mean(np.abs(bf))),
          "maxabs", float(np.max(np.abs(bf))))

    print("UNSLOTH vs BF16:")
    for invert in (True, False):
        for high in (False, True):
            w = dequant_nvfp4(packed, scale, wgs, high_first=high, invert_global=invert)
            stats(f"inv={invert} high_first={high}", w, bf)

    try:
        mw, ms, ms2 = load_modelopt(prefix + ".")
        print("modelopt weight", mw.shape, mw.dtype, "scale", ms.shape, "s2",
              float(np.asarray(ms2).reshape(-1)[0]))
        wmo = dequant_nvfp4(mw, ms, ms2, high_first=False, invert_global=False)
        print("MODELOPT vs BF16:")
        stats("modelopt native", wmo, bf)
        print("UNSLOTH vs MODELOPT:")
        w_u = dequant_nvfp4(packed, scale, wgs, high_first=False, invert_global=True)
        stats("unsloth CT-inv vs modelopt", w_u, wmo)
    except Exception as e:
        print("modelopt skip:", type(e).__name__, e)

    # packed-byte overlap vs modelopt (same layer, different quant)
    try:
        same = float((packed == mw).mean()) if mw.shape == packed.shape else -1.0
        print(f"packed byte equality vs modelopt: {same:.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
