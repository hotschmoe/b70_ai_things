# 10_triton_dot_widths.py -- EXPERIMENT A.3: what operand widths does Intel-XPU
# triton's tl.dot accept, and does int8 tl.dot emit DPAS or silently upcast?
# We compile+run tiny tl.dot kernels for int8/int16/fp16 operands and time them.
# ONEDNN not involved (this is triton codegen). We report success/fail + timing.
import time, sys
import torch
import triton
import triton.language as tl

DEV = "xpu"

def line(*a): print(*a, flush=True)

# --- int8 x int8 -> int32 accumulate dot ---
@triton.jit
def dot_i8(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])  # [M,K] i8
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])  # [K,N] i8
    acc = tl.dot(a, b, out_dtype=tl.int32)                      # [M,N] i32
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

@triton.jit
def dot_f16(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M); offs_n = tl.arange(0, N); offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

def try_dot(name, kern, dt_in, dt_out, M=16, N=16, K=64):
    try:
        if dt_in in (torch.int8, torch.int16, torch.int32):
            a = torch.randint(-8, 8, (M, K), dtype=dt_in, device=DEV)
            b = torch.randint(-8, 8, (K, N), dtype=dt_in, device=DEV)
        else:
            a = (torch.randn(M, K, device=DEV) * 0.1).to(dt_in)
            b = (torch.randn(K, N, device=DEV) * 0.1).to(dt_in)
        c = torch.zeros((M, N), dtype=dt_out, device=DEV)
        kern[(1,)](a, b, c, M, N, K)
        torch.xpu.synchronize()
        # reference
        ref = (a.float() @ b.float())
        err = (c.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        # time
        for _ in range(5): kern[(1,)](a, b, c, M, N, K)
        torch.xpu.synchronize(); t0 = time.time()
        for _ in range(50): kern[(1,)](a, b, c, M, N, K)
        torch.xpu.synchronize()
        ms = (time.time() - t0) / 50 * 1e3
        line(f"  {name:22s} OK  rel-err {err:.4f}  {ms:.4f} ms  in={dt_in} out={dt_out}")
        return True
    except Exception as e:
        line(f"  {name:22s} FAIL {type(e).__name__}: {str(e)[:140]}")
        return False

line("=== triton tl.dot operand-width probe (Intel XPU) ===")
line("triton", triton.__version__)
line("--- int8 x int8 -> int32 ---")
try_dot("i8xi8->i32 (M16N16K64)", dot_i8, torch.int8, torch.int32)
line("--- fp16 x fp16 -> fp32 (baseline) ---")
try_dot("f16xf16->f32", dot_f16, torch.float16, torch.float32)
line("--- int16 x int16 -> int32 ---")
try_dot("i16xi16->i32", dot_i8, torch.int16, torch.int32)
line("DONE")
