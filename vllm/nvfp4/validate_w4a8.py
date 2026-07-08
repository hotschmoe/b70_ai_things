# Validate + bench the NEW nvfp4_gemm_w4a8 block-scaled INT8 op on a REAL 27B MLP weight.
#
# Loads real gate_proj (N=17408 K=5120) NVFP4 tensors, repacks E2M1 -> s8 + per-16-K-group
# scale (block*global/2), and checks:
#   (1) KERNEL correctness: y vs a bf16 matmul of the SAME int8-dequantized activation
#       (isolates the block-scaled-int8 dequant; should be ~exact modulo s32->bf16 rounding).
#   (2) END-TO-END W4A8 quality: y vs F.linear(x_bf16, w_bf16_exact) = the current W4A16
#       nvfp4 numerics (error here = the int8-ACTIVATION quant cost of going W4A16 -> W4A8).
#   (3) SPEED vs nvfp4_gemm_w4a16 (current path) and bf16 F.linear at prefill M.
#
# Run in int8g-v0240 with the nvfp4pref_kernel .so mounted (has nvfp4_gemm_w4a8 + w4a16).
import os, time, json, struct
import numpy as np
import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401

DEV = "xpu"
MODELDIR = "/models/qwen3.6-27b/nvfp4-modelopt"
TGT = os.environ.get("TGT", "model.language_model.layers.3.mlp.gate_proj")
MS = [int(x) for x in (os.environ.get("MS") or "512,2048,8192").split(",")]

E2M1_INT8 = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int8)  # E2M1 magnitude * 2 (exact s8)


def _find_shard(name):
    import glob
    for f in sorted(glob.glob(f"{MODELDIR}/*.safetensors")):
        with open(f, "rb") as fh:
            hl = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(hl))
        if name + ".weight" in hdr:
            return f, hdr
    raise KeyError(name)


def _raw(f, hdr, name):
    meta = hdr[name]; s, e = meta["data_offsets"]
    with open(f, "rb") as fh:
        base = 8 + struct.unpack("<Q", open(f, "rb").read(8))[0]
        fh.seek(base + s); buf = fh.read(e - s)
    dt = {"U8": np.uint8, "F8_E4M3": np.uint8, "F32": np.float32, "BF16": np.uint16}[meta["dtype"]]
    return np.frombuffer(buf, dt).reshape(meta["shape"]), meta["dtype"]


def _e4m3(u8):
    u = u8.astype(np.uint32)
    sign = np.where(u & 0x80, -1.0, 1.0).astype(np.float32)
    exp = (u >> 3) & 0xF; man = (u & 0x7).astype(np.float32)
    val = np.where(exp == 0, (man / 8) * 2.0**-6,
                   (1 + man / 8) * (2.0 ** (exp.astype(np.int32) - 7))).astype(np.float32)
    return sign * val


f, hdr = _find_shard(TGT)
packed, _ = _raw(f, hdr, TGT + ".weight")            # [N, K/2] u8
ws_u8, _ = _raw(f, hdr, TGT + ".weight_scale")       # [N, K/16] f8e4m3 bytes
ws2, _ = _raw(f, hdr, TGT + ".weight_scale_2")       # scalar f32
ws2 = float(ws2.reshape(-1)[0])
N, Kh = packed.shape; K = Kh * 2
lo, hi = packed & 0xF, (packed >> 4) & 0xF
both = np.stack([lo, hi], -1).reshape(N, K)          # low nibble first
sign = np.where(both & 0x8, -1, 1).astype(np.int8)
w_s8 = (sign * E2M1_INT8[both & 0x7]).astype(np.int8)          # [N,K] exact s8
g = (_e4m3(ws_u8) * ws2 / 2.0).astype(np.float32)             # [N,K/16] group scale
print(f"{TGT}: N={N} K={K}  s8 range [{w_s8.min()},{w_s8.max()}]  ws2={ws2:.4g}")

# exact bf16 reference weight = s8 * group_scale (== the current W4A16 nvfp4 numerics)
w_bf16 = torch.from_numpy(
    (w_s8.astype(np.float32).reshape(N, K // 16, 16) * g[:, :, None]).reshape(N, K)
).to(DEV, torch.bfloat16)

w_s8_t = torch.from_numpy(w_s8).to(DEV)               # [N,K] s8
g_nt = torch.from_numpy(g.T.copy()).to(DEV, torch.bfloat16)   # [K/16, N] bf16 (NT for op)
# nvfp4 w4a16 needs the packed [N,K/2] weight (.t view) + [K/16,N] scale
packed_t = torch.from_numpy(packed).to(DEV)           # [N, K/2] u8
_LUT_S8 = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12], dtype=torch.int8, device=DEV)


def repack_f4_to_s8(packed_dev):
    # transient in-torch decode: [N,K/2] u8 -> [N,K] s8 (what a single-card serve would do
    # per prefill forward since a resident s8 copy is 31GB and does not fit one card).
    lo = packed_dev & 0x0F
    hi = (packed_dev >> 4) & 0x0F
    both = torch.stack([lo, hi], dim=-1).reshape(packed_dev.shape[0], -1)  # [N,K] low-first
    sgn = torch.where((both & 0x8) != 0, -1, 1).to(torch.int8)
    return sgn * _LUT_S8[(both & 0x7).long()]


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup): fn()
    torch.xpu.synchronize(); t = time.time()
    for _ in range(iters): fn()
    torch.xpu.synchronize()
    return (time.time() - t) / iters * 1e3


print(f"\n{'M':>6} | {'kern relerr':>11} | {'w4a8-vs-w4a16 relerr':>20} | "
      f"{'bf16 ms':>8} {'w4a16 ms':>9} {'w4a8 ms':>8} | {'w4a8/w4a16':>10} {'w4a8/bf16':>10}")
for M in MS:
    torch.manual_seed(M)
    x = (torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1)
    # per-token symmetric int8 activation quant
    amax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)   # [M,1]
    asc = (amax / 127.0).to(torch.float32)                      # [M,1] scale
    xs8 = torch.clamp(torch.round(x / asc), -127, 127).to(torch.int8)
    x_deq = (xs8.to(torch.float32) * asc).to(torch.bfloat16)    # int8-dequantized act

    ref_w4a16 = torch.nn.functional.linear(x, w_bf16).float()          # W4A16 (bf16 act) reference
    ref_kern = torch.nn.functional.linear(x_deq, w_bf16).float()       # same int8 act, bf16 matmul

    y = torch.ops._xpu_C.nvfp4_gemm_w4a8(xs8, asc, w_s8_t.t(), g_nt, 16, torch.bfloat16).float()
    kern_relerr = (y - ref_kern).abs().max().item() / (ref_kern.abs().max().item() + 1e-9)
    e2e_relerr = (y - ref_w4a16).abs().max().item() / (ref_w4a16.abs().max().item() + 1e-9)

    t_bf16 = bench(lambda: torch.nn.functional.linear(x, w_bf16))
    t_a = bench(lambda: torch.ops._xpu_C.nvfp4_gemm_w4a16(x, packed_t.t(), None, g_nt, 16))
    t_w = bench(lambda: torch.ops._xpu_C.nvfp4_gemm_w4a8(xs8, asc, w_s8_t.t(), g_nt, 16, torch.bfloat16))
    # realistic single-card serve cost: transient f4->s8 repack + block-scaled int8 gemm
    t_rep = bench(lambda: repack_f4_to_s8(packed_t))
    t_serve = t_rep + t_w
    print(f"{M:>6} | {kern_relerr:>11.2e} | {e2e_relerr:>20.2e} | "
          f"{t_bf16:>8.3f} {t_a:>9.3f} {t_w:>8.3f} | {t_a/t_w:>10.2f} {t_bf16/t_w:>10.2f} | "
          f"rep {t_rep:>6.3f} serve {t_serve:>7.3f} a/serve {t_a/t_serve:>5.2f}")

print("\nkern relerr: block-scaled-int8 kernel vs bf16 matmul of the SAME int8 act (want ~1e-2, dequant-exact).")
print("w4a8-vs-w4a16 relerr: end-to-end int8-activation cost vs current W4A16 (quality signal).")
print("w4a8/w4a16 col: SPEED ratio nvfp4_gemm_w4a16_ms / nvfp4_gemm_w4a8_ms (>1 = INT8 prefill win).")
