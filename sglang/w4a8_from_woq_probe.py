#!/usr/bin/env python3
# w4a8_from_woq_probe.py -- THE GATE for serving the Lorbus int4-AutoRound (auto_gptq-packed)
# checkpoint through the oneDNN int4_gemm ops (int4_gemm_w4a16 decode / int4_gemm_w4a8 prefill).
#
# The int4_gemm ops were VERIFIED on COMPRESSED-TENSORS packing (weight [N,K/8] i32 + weight_scale
# [N,K/g], sym zp=8, pass weight.t()). The Lorbus ckpt is auto_gptq-packed:
#   qweight [K/8, N] i32  (8 nibbles per int32, nibble i at bit 4i = K-index k8*8+i)
#   qzeros  [K/g, N/8] i32 (IGNORED for sym=True in auto_round_kernel)
#   scales  [K/g, N] (group_size=128)
#
# CLAIM (from reading auto_round_kernel.qlinear.unpack_to_8bit_signed + post_init, sym path):
#   ARK dequant = (nibble - 8) * scale, nibble i at bit 4i for K-index k8*8+i, zeros ignored.
#   The int4_gemm op dequant = (nibble - B_zp) * B_scale with the SAME nibble->K-index mapping.
#   => conversion is PURELY a memory relayout: the op needs B=[K/8,N] with stride[0]==1, but
#      auto_gptq qweight=[K/8,N] is contiguous (stride[0]==N). Fix: B = qweight.t().contiguous().t()
#      (a [K/8,N] view over a [N,K/8] contiguous buffer -> stride[0]==1, same logical values).
#      B_scale = scales.to(fp16).contiguous() (already [K/g,N]); B_zp = tensor([8]) (sym).
#
# GATE: relerr(int4_gemm_w4a16(converted Lorbus weights), woqgemm(SAME Lorbus layer)) < 1e-2 on
#       down_proj AND a fused/GDN layer (in_proj_qkv). The reference is auto_round_kernel woqgemm
#       (the proven 23.5 t/s daily-driver path), built EXACTLY as sglang/patches/woq_shim.py does.
#
# Run (card 0, sglang-xpu:woq, oneAPI on LD_LIBRARY_PATH): see the docker wrapper in the runner.
# ASCII only.
import os
import sys
import time
import ctypes
import json

import torch  # MUST import torch before dlopen-ing the kernel .so

DEV = "xpu"
MODEL = os.environ.get("MODEL_DIR", "/models/Lorbus_Qwen3.6-27B-int4-AutoRound")
SO = os.environ.get("B70_XPU_C_SO", "/work/w4a8_kernel/_xpu_C.abi3.so")
GROUP = 128

# (layer-suffix, label) pairs to validate. down_proj = plain MLP; in_proj_qkv/out_proj = fused GDN.
LAYERS = [
    ("model.language_model.layers.20.mlp.down_proj", "down_proj (MLP)"),
    ("model.language_model.layers.20.mlp.gate_proj", "gate_proj (MLP)"),
    ("model.language_model.layers.20.linear_attn.in_proj_qkv", "in_proj_qkv (GDN fused)"),
    ("model.language_model.layers.20.linear_attn.out_proj", "out_proj (GDN)"),
]


def load_op():
    if hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):
        return
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
        return
    except Exception:
        pass
    ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)


def get_layer_tensors(prefix):
    """Load qweight/qzeros/scales for `prefix` from the right shard via the index."""
    from safetensors import safe_open

    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for sfx in ("qweight", "qzeros", "scales"):
        key = f"{prefix}.{sfx}"
        shard = idx[key]
        with safe_open(os.path.join(MODEL, shard), framework="pt", device="cpu") as f:
            out[sfx] = f.get_tensor(key)
    return out["qweight"], out["qzeros"], out["scales"]


def build_woqgemm_ref(qw, qz, sc):
    """Build the auto_round_kernel woqgemm reference EXACTLY as woq_shim._XpuWoqGptqKernel does."""
    from auto_round_kernel.qlinear import QuantLinearGPTQ

    in_f = qw.shape[0] * 8
    out_f = qw.shape[1]
    ql = QuantLinearGPTQ(4, GROUP, True, in_f, out_f, False, weight_dtype=torch.bfloat16)
    ql.qweight.data = qw.clone()
    ql.qzeros.data = qz.clone()
    ql.scales.data = sc.to(torch.float16).clone()
    ql = ql.to(DEV)
    ql.post_init()
    return ql, in_f, out_f


def convert_auto_gptq_to_op_B(qw):
    """auto_gptq qweight [K/8,N] (contiguous, stride[0]==N) -> op B [K/8,N] with stride[0]==1.
    Pure relayout: B = qw.t().contiguous().t() keeps logical values, gives NT (stride[0]==1)."""
    B_contig = qw.t().contiguous()          # [N, K/8] contiguous (the backing buffer)
    B = B_contig.t()                        # [K/8, N] view, stride[0]==1
    return B, B_contig


def relerr(a, b):
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def main():
    print("=== w4a8_from_woq_probe (auto_gptq -> int4_gemm op layout) ===", flush=True)
    print("torch", torch.__version__, "xpu_avail", torch.xpu.is_available(), flush=True)
    load_op()
    has16 = hasattr(torch.ops._xpu_C, "int4_gemm_w4a16")
    has8 = hasattr(torch.ops._xpu_C, "int4_gemm_w4a8")
    print("ops: int4_gemm_w4a16=", has16, " int4_gemm_w4a8=", has8, flush=True)
    assert has16 and has8, "ops not registered"

    overall_ok = True
    for prefix, label in LAYERS:
        print(f"\n========== {label}  [{prefix.split('layers.20.')[1]}] ==========", flush=True)
        qw, qz, sc = get_layer_tensors(prefix)
        print(f"  qweight {tuple(qw.shape)} {qw.dtype}  qzeros {tuple(qz.shape)} {qz.dtype}  "
              f"scales {tuple(sc.shape)} {sc.dtype}", flush=True)

        # qzeros sanity: for sym auto_round, ARK ignores qzeros (zero fixed at 8). Print what
        # the qzeros would unpack to so we can confirm it is uniform (a sym ckpt packs a constant).
        wf = torch.arange(0, 32, 4, dtype=torch.int32)
        zz = (qz.unsqueeze(2) >> wf.view(1, 1, -1)) & 0xF      # [K/g, N/8, 8] in 0..15
        uvals, ucnt = torch.unique(zz, return_counts=True)
        print(f"  qzeros nibble values (0..15): {uvals.tolist()} counts {ucnt.tolist()}", flush=True)

        ql, in_f, out_f = build_woqgemm_ref(qw, qz, sc)
        K, N = in_f, out_f
        print(f"  K(in)={K}  N(out)={N}  group={GROUP}", flush=True)

        qw_x = qw.to(DEV)
        B, B_contig = convert_auto_gptq_to_op_B(qw_x)
        B_scale = sc.to(DEV).to(torch.float16).contiguous()    # [K/g, N]
        B_zp = torch.tensor([8], dtype=torch.int8, device=DEV)
        assert B.stride()[0] == 1, f"B not NT (stride0={B.stride()[0]})"
        print(f"  op B {tuple(B.shape)} stride{tuple(B.stride())}  B_scale {tuple(B_scale.shape)}", flush=True)

        for M in (1, 2048):
            x16 = torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.1
            ref = ql(x16).to(torch.float32)                    # woqgemm reference (fp16 internally)

            # --- decode path: int4_gemm_w4a16 (fp16 act) ---
            out16 = torch.ops._xpu_C.int4_gemm_w4a16(x16, B, None, B_scale, B_zp, GROUP, None)
            re16 = relerr(out16, ref)
            f16 = bool(torch.isfinite(out16).all())

            # --- prefill path: int4_gemm_w4a8 (per-token sym int8 act) ---
            amax = x16.abs().amax(-1, keepdim=True).clamp_(min=1e-5)
            xs = (amax / 127.0).to(torch.float16)
            xq = (x16 / xs).round().clamp_(-127, 127).to(torch.int8).contiguous()
            xz = torch.zeros_like(amax, dtype=torch.int32).contiguous()
            out8 = torch.ops._xpu_C.int4_gemm_w4a8(xq, xs.contiguous(), xz, B, B_scale, B_zp, GROUP, None, None)
            re8 = relerr(out8, ref)
            f8 = bool(torch.isfinite(out8).all())

            tag16 = "OK " if (re16 < 1e-2 and f16) else "BAD"
            tag8 = "OK " if (re8 < 5e-2 and f8) else "BAD"   # w4a8 looser (int8 act-quant error)
            print(f"  M={M:>4}  w4a16 relerr={re16:.3e} finite={f16} [{tag16}]   "
                  f"w4a8 relerr={re8:.3e} finite={f8} [{tag8}]", flush=True)
            if M == 1 and not (re16 < 1e-2 and f16):
                overall_ok = False
            if not (re8 < 5e-2 and f8):
                overall_ok = False

        del ql, qw_x, B, B_contig, B_scale
        if hasattr(torch, "xpu"):
            torch.xpu.empty_cache()

    print(f"\n=== GATE {'PASS' if overall_ok else 'FAIL'} ===", flush=True)
    print("(w4a16 relerr<1e-2 on decode = the op matches woqgemm; serve is numerically justified)", flush=True)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
