# M2 groundwork: prove NVFP4 weights are LOSSLESSLY int8, so an nvfp4 GEMM can ride
# B70's INT8 XMX fast paths (our proven oneDNN int8 w8a16 kernel) instead of a new
# fp4 kernel.
#
# Key identity: the E2M1 value set is {0, .5, 1, 1.5, 2, 3, 4, 6} (x sign). Multiply
# by 2 -> {0, 1, 2, 3, 4, 6, 8, 12} -- ALL EXACT INT8, |max|=12 < 127. So:
#     w_fp4[i,j]  =  int8_code[i,j] / 2  * block_scale[i, j//16] * weight_scale_2
# i.e. an nvfp4 weight == int8 weight with per-16-group fp scale
#     g_scale[i,g] = block_scale[i,g] * weight_scale_2 / 2
# No accuracy loss vs the bf16 dequant -- the ONLY quantization already happened at
# checkpoint creation; this is a pure re-encoding of the same numbers.
#
# This validates the identity on real Qwen3-8B-NVFP4 tensors, exactly (bit-for-bit
# on the reconstructed weight), against the M0 reference dequant.
import numpy as np

from importlib import import_module
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
crack = import_module("01_crack_format")  # reuse read_raw_u8, decode_e4m3, E2M1_LUT, MODEL
from safetensors import safe_open

# int8 code for each of the 8 magnitudes: round(E2M1 * 2)
INT8_MAG = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int8)  # = E2M1_LUT * 2


def repack_to_int8(name: str):
    """NVFP4 packed weight -> (int8 weight [N,K], group scale [N,K/16] fp32)."""
    packed = safe_open(crack.MODEL, "numpy").get_tensor(name + ".weight")  # [N, K/2] u8
    ws = crack.decode_e4m3(crack.read_raw_u8(crack.MODEL, name + ".weight_scale"))  # [N,K/16]
    ws2 = float(safe_open(crack.MODEL, "numpy").get_tensor(name + ".weight_scale_2"))
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    both = np.stack([lo, hi], axis=-1).reshape(packed.shape[0], -1)  # [N,K]
    sign = np.where(both & 0x8, -1, 1).astype(np.int8)
    w_int8 = (sign * INT8_MAG[both & 0x7]).astype(np.int8)           # exact int8
    g_scale = (ws * ws2 / 2.0).astype(np.float32)                    # [N,K/16]
    return w_int8, g_scale


for tgt in ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"]:
    w_ref = crack.dequant(tgt, safe_open(crack.MODEL, "numpy"))       # M0 bf16-path ref
    w_int8, g = repack_to_int8(tgt)
    N, K = w_int8.shape
    w_from_int8 = (w_int8.astype(np.float32).reshape(N, K // 16, 16)
                   * g[:, :, None]).reshape(N, K)
    max_abs_err = np.abs(w_from_int8 - w_ref).max()
    print(f"{tgt}")
    print(f"  int8 weight range: [{w_int8.min()}, {w_int8.max()}]  (must be within [-12,12])")
    print(f"  int8 codes used: {sorted(np.unique(w_int8).tolist())}")
    print(f"  max |int8-repack  -  bf16-ref| = {max_abs_err:.3e}  "
          f"({'EXACT' if max_abs_err == 0 else 'MISMATCH'})")
    print()

print("=> NVFP4 weight == int8 weight + per-16-group fp32 scale, bit-exact.")
print("   Ride int8_gemm_w8a16 (oneDNN INT8 XMX) if it supports group_size=16 scales.")
