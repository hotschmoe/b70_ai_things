# NVFP4 format crack: decode nvidia/Qwen3-8B-NVFP4 modelopt tensors in pure numpy.
# NVFP4 = E2M1 4-bit weights, packed 2/byte, + FP8-E4M3 block scale per 16 elems
# along K, + fp32 per-tensor global scale (weight_scale_2).
# dequant(w) = e2m1_lut[nibble] * e4m3(block_scale) * weight_scale_2
#
# Usage: python3 vllm/nvfp4/01_crack_format.py
import json
import struct

import numpy as np
from safetensors import safe_open


def read_raw_u8(path: str, tensor_name: str) -> np.ndarray:
    """Read any tensor's raw bytes as uint8 (numpy has no fp8 dtype)."""
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(hlen))
        meta = hdr[tensor_name]
        start, end = meta["data_offsets"]
        fh.seek(8 + hlen + start)
        raw = np.frombuffer(fh.read(end - start), dtype=np.uint8)
    itemsize = (end - start) // int(np.prod(meta["shape"]))
    assert itemsize == 1, f"expected 1-byte dtype, got {itemsize}"
    return raw.reshape(meta["shape"])

MODEL = "models/files/qwen3-8b/nvfp4-modelopt/model-00001-of-00002.safetensors"

# E2M1 magnitude LUT for the 3 magnitude bits (sign handled separately)
E2M1_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def decode_e4m3(u8: np.ndarray) -> np.ndarray:
    """Decode uint8-viewed float8_e4m3fn to float32 (numpy has no fp8)."""
    u = u8.astype(np.uint32)
    sign = np.where(u & 0x80, -1.0, 1.0).astype(np.float32)
    exp = (u >> 3) & 0xF
    man = (u & 0x7).astype(np.float32)
    # e4m3fn: bias 7, no inf, exp=0 is subnormal (val = man/8 * 2^-6),
    # exp>0 normal (val = (1+man/8) * 2^(exp-7)); 0x7F/0xFF = NaN
    val = np.where(
        exp == 0,
        (man / 8.0) * 2.0**-6,
        (1.0 + man / 8.0) * (2.0 ** (exp.astype(np.int32) - 7)),
    ).astype(np.float32)
    nan_mask = (u & 0x7F) == 0x7F
    out = sign * val
    out[nan_mask] = np.nan
    return out


def unpack_fp4(packed: np.ndarray) -> np.ndarray:
    """[N, K/2] uint8 -> [N, K] float32 e2m1 values. low nibble first."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    both = np.stack([lo, hi], axis=-1).reshape(packed.shape[0], -1)
    sign = np.where(both & 0x8, -1.0, 1.0).astype(np.float32)
    return sign * E2M1_LUT[both & 0x7]


def dequant(name: str, f) -> np.ndarray:
    w_packed = f.get_tensor(name + ".weight")               # uint8 [N, K/2]
    ws = read_raw_u8(MODEL, name + ".weight_scale")         # e4m3 raw bytes [N, K/16]
    ws2 = f.get_tensor(name + ".weight_scale_2")            # fp32 scalar
    w = unpack_fp4(w_packed)                                 # [N, K]
    scales = decode_e4m3(ws) * ws2                           # [N, K/16] fp32
    N, K = w.shape
    w = w.reshape(N, K // 16, 16) * scales[:, :, None]
    return w.reshape(N, K)


with safe_open(MODEL, "numpy") as f:
    names = sorted(f.keys())
    print(f"{len(names)} tensors in shard 1")
    # Schema of layer 0
    print("\n--- layer 0 schema ---")
    for n in names:
        if ".layers.0." in n:
            s = f.get_slice(n)
            print(f"{n:70s} {s.get_dtype():8s} {s.get_shape()}")

    print("\n--- dequant sanity: layer 0 q_proj ---")
    tgt = "model.layers.0.self_attn.q_proj"
    w = dequant(tgt, f)
    print("shape", w.shape, "dtype", w.dtype)
    print(f"std {w.std():.5f}  mean {w.mean():.6f}  absmax {np.abs(w).max():.4f}")
    print("NaN:", np.isnan(w).sum(), " Inf:", np.isinf(w).sum())
    # distribution of raw fp4 codes (should use full range if quant is healthy)
    wp = f.get_tensor(tgt + ".weight")
    codes = np.concatenate([wp & 0xF, wp >> 4]).ravel()
    hist = np.bincount(codes, minlength=16)
    print("fp4 code histogram (0-15):", hist.tolist())
    # block scale stats
    ws = decode_e4m3(read_raw_u8(MODEL, tgt + ".weight_scale"))
    ws2 = f.get_tensor(tgt + ".weight_scale_2")
    print(f"block scales: min {ws.min():.4f} max {ws.max():.4f} NaN {np.isnan(ws).sum()}")
    print(f"weight_scale_2: {ws2}")
    # extra scales present?
    for n in names:
        if ".layers.0.self_attn." in n and "scale" in n and "weight_scale" != n.rsplit(".",1)[-1]:
            t = f.get_tensor(n)
            print(f"{n:70s} -> {np.array2string(np.asarray(t).ravel()[:4], precision=6)}")
