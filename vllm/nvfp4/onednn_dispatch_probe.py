# ONEDNN_VERBOSE dispatch probe: what oneDNN primitive does each op lower to at prefill M?
# Answers the crux -- does s8-src x per-16-K-group-scaled-weight dispatch to INT8 XMX
# (jit:gemm:xe with src_s8/wei_s8), or does the per-group weight scale force a float
# (bf16/f16) decompression path (= no int8 speedup)?
#
# Run with ONEDNN_VERBOSE=dispatch,profile_exec so the onednn_verbose,... lines print.
# One call each at M=2048 on the real gate_proj shape (N=17408 K=5120).
import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401

DEV = "xpu"
torch.manual_seed(0)
N, K, M = 17408, 5120, 2048

xb = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1
xs8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=DEV)
asc = torch.rand(M, 1, device=DEV, dtype=torch.float32) * 0.01 + 0.001
azp = torch.zeros(M, 1, dtype=torch.int8, device=DEV)

packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=DEV)
nv_scale = torch.rand(K // 16, N, device=DEV, dtype=torch.bfloat16) * 0.1 + 0.01
s8w = torch.randint(-12, 13, (N, K), dtype=torch.int8, device=DEV)
pc_scale = torch.rand(N, device=DEV, dtype=torch.float32) * 0.01 + 0.001
bf16w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02
i4 = torch.randint(-(2**31), 2**31, (N, K // 8), dtype=torch.int32, device=DEV)
i4_scale = torch.rand(K // 16, N, device=DEV, dtype=torch.bfloat16) * 0.1 + 0.01
i4_zp = torch.zeros(K // 16, N // 8, dtype=torch.int32, device=DEV)


def banner(s):
    print(f"\n##### DISPATCH: {s} #####", flush=True)


banner("bf16 F.linear")
torch.nn.functional.linear(xb, bf16w); torch.xpu.synchronize()

banner("nvfp4_gemm_w4a16 (f4_e2m1 weight, bf16 src, per-16-K-group scale) -- CURRENT")
torch.ops._xpu_C.nvfp4_gemm_w4a16(xb, packed.t(), None, nv_scale, 16); torch.xpu.synchronize()

banner("int8_gemm_w8a8 (s8 src per-token, s8 weight per-CHANNEL scale)")
torch.ops._xpu_C.int8_gemm_w8a8(xs8, asc, None, s8w.t(), pc_scale, None, None, torch.bfloat16)
torch.xpu.synchronize()

banner("int4_gemm_w4a8 (s8 src per-token, int4 weight per-16-K-GROUP scale) -- block-scaled probe")
torch.ops._xpu_C.int4_gemm_w4a8(xs8, asc, azp, i4.t(), i4_scale, i4_zp, 16, None, None)
torch.xpu.synchronize()

banner("int8_gemm_w8a16 (bf16 src, s8 weight per-16-K-group scale) -- decode path")
torch.ops._xpu_C.int8_gemm_w8a16(xb, s8w.t(), nv_scale, None); torch.xpu.synchronize()

if hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a8"):
    s8scale_pc = pc_scale  # reuse per-channel for a per-channel dispatch line
    banner("nvfp4_gemm_w4a8 (s8 src per-token, s8 weight per-16-K-GROUP scale) -- NEW block-scaled INT8")
    torch.ops._xpu_C.nvfp4_gemm_w4a8(xs8, asc, s8w.t(), nv_scale, 16, torch.bfloat16)
    torch.xpu.synchronize()
    banner("nvfp4_gemm_w4a8 (s8 src per-token, s8 weight PER-CHANNEL scale) -- control")
    torch.ops._xpu_C.nvfp4_gemm_w4a8(xs8, asc, s8w.t(), s8scale_pc, 16, torch.bfloat16)
    torch.xpu.synchronize()
else:
    print("\n(nvfp4_gemm_w4a8 not in this .so -- run with nvfp4pref_kernel mounted)", flush=True)

print("\n##### DONE #####", flush=True)
