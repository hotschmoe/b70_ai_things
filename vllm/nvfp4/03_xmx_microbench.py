# M2 on-GPU: prove NVFP4-as-int8 rides B70 INT8 XMX via the EXISTING oneDNN
# int8_gemm_w8a16 op (from w8a8_kernel_v0240/_xpu_C.abi3.so), and measure the
# ceiling vs the bf16 F.linear the dequant serve uses today.
#
# Runs INSIDE the int8g-v0240 container with the built .so mounted over the pkg.
# Two probes:
#   (A) per-channel int8 scale  -> should JUST WORK on the existing op (proves XMX
#       int8 GEMM harness + measures speed). Numerics: approximate (collapses the
#       16-wide K groups to one per-channel scale) -- speed signal only.
#   (B) block-quant K-group=16  -> the NVFP4-faithful grouping. Probes whether the
#       existing square-{g,g} op accepts a [K/16, N] weight-scale (K-only group).
#       Expected to FAIL/mismatch until the kernel is changed to group {16,1}.
import time

import numpy as np
import torch

# the int8 ops are registered by importing the kernels package inside the container
import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C.*)

DEV = "xpu"
MODEL = "/models/qwen3-8b/nvfp4-modelopt/model-00001-of-00002.safetensors"

E2M1_INT8 = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int8)


def _read_raw_u8(name):
    import json, struct
    with open(MODEL, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(hlen))
        meta = hdr[name]
        s, e = meta["data_offsets"]
        fh.seek(8 + hlen + s)
        return np.frombuffer(fh.read(e - s), np.uint8).reshape(meta["shape"])


def _decode_e4m3(u8):
    u = u8.astype(np.uint32)
    sign = np.where(u & 0x80, -1.0, 1.0).astype(np.float32)
    exp = (u >> 3) & 0xF
    man = (u & 0x7).astype(np.float32)
    val = np.where(exp == 0, (man / 8) * 2.0**-6,
                   (1 + man / 8) * (2.0 ** (exp.astype(np.int32) - 7))).astype(np.float32)
    return sign * val


def load_int8(name):
    from safetensors import safe_open
    with safe_open(MODEL, "numpy") as f:
        packed = f.get_tensor(name + ".weight")            # [N, K/2] u8
        ws2 = float(f.get_tensor(name + ".weight_scale_2"))
    ws = _decode_e4m3(_read_raw_u8(name + ".weight_scale"))  # [N, K/16]
    lo, hi = packed & 0xF, (packed >> 4) & 0xF
    both = np.stack([lo, hi], -1).reshape(packed.shape[0], -1)  # [N,K]
    sign = np.where(both & 0x8, -1, 1).astype(np.int8)
    w_int8 = (sign * E2M1_INT8[both & 0x7]).astype(np.int8)     # [N,K] exact
    g_scale = (ws * ws2 / 2.0).astype(np.float32)              # [N,K/16]
    return w_int8, g_scale


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.time() - t) / iters * 1e3  # ms


TGT = "model.layers.0.self_attn.q_proj"
w_int8_np, g_np = load_int8(TGT)                 # [N,K], [N,K/16]
N, K = w_int8_np.shape
print(f"{TGT}: N={N} K={K}  int8 range [{w_int8_np.min()},{w_int8_np.max()}]")

# bf16 reference weight (exact dequant) + reference matmul
w_bf16 = torch.from_numpy(
    (w_int8_np.astype(np.float32).reshape(N, K // 16, 16) * g_np[:, :, None]).reshape(N, K)
).to(DEV, torch.bfloat16)

w_int8 = torch.from_numpy(w_int8_np).to(DEV)                # [N,K] s8
g = torch.from_numpy(g_np).to(DEV)                          # [N,K/16] f32

for M in (1, 8, 64):
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1
    ref = torch.nn.functional.linear(x, w_bf16)             # [M,N]
    t_bf16 = bench(lambda: torch.nn.functional.linear(x, w_bf16))

    # weight for oneDNN w8a16 is [K, N] (NT: mat2 transposed). is_nt True -> pass [N,K]?
    # dnnl_matmul_w8a16_int8 takes mat2 [k,n] s8; is_nt transposes. We give w [N,K] and set is_nt.
    # Probe A: per-channel scale = mean of the 16-groups (speed signal, approx numerics)
    sc_pc = g.mean(dim=1, keepdim=True).to(torch.bfloat16)   # [N,1] -> per-channel (mask 1<<1 wants [1,N]?)
    print(f"\n--- M={M} ---")
    print(f"bf16 F.linear: {t_bf16:.3f} ms")

    # signature: int8_gemm_w8a16(A[b,m,k] f16/bf16, B[k,n] s8 NT, B_scale?, bias?)
    wt = w_int8.t().contiguous()                            # [K,N] s8
    try:
        # Probe A: per-channel scale [N] (mean of 16-groups). speed signal; approx numerics.
        scA = sc_pc.reshape(N).to(torch.bfloat16)
        yA = torch.ops._xpu_C.int8_gemm_w8a16(x, wt, scA, None)
        errA = (yA.float() - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-9)
        tA = bench(lambda: torch.ops._xpu_C.int8_gemm_w8a16(x, wt, scA, None))
        print(f"int8 XMX (per-channel): {tA:.3f} ms  speedup {t_bf16/tA:.2f}x  rel-err {errA:.3f} (approx)")
    except Exception as e:
        print(f"int8 XMX (per-channel) FAILED: {type(e).__name__}: {str(e)[:160]}")

    try:
        # Probe B: NVFP4-faithful K-group=16. wrapper doc says block scale layout is [k/g, n].
        sc_kg = g.t().contiguous().to(torch.bfloat16)        # [K/16, N]
        yB = torch.ops._xpu_C.int8_gemm_w8a16(x, wt, sc_kg, None)
        errB = (yB.float() - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-9)
        tB = bench(lambda: torch.ops._xpu_C.int8_gemm_w8a16(x, wt, sc_kg, None))
        tag = "EXACT-ish" if errB < 1e-2 else "MISMATCH (square-group bug)"
        print(f"int8 XMX (K-group16):   {tB:.3f} ms  speedup {t_bf16/tB:.2f}x  rel-err {errB:.5f}  ({tag})")
    except Exception as e:
        print(f"int8 XMX (K-group16) FAILED: {type(e).__name__}: {str(e)[:160]}")

print("\nlegend: A proves the XMX int8 harness+speed; B tells us if the existing")
print("square-group op already does NVFP4 K-only grouping or needs the {16,1} change.")
