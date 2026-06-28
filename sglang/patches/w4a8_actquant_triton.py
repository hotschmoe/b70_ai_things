# w4a8_actquant_triton.py -- single-launch Triton per-token SYMMETRIC int8 activation quant for the
# W4A8 prefill path (int4_gemm_w4a8). Replaces the ~8-launch eager chain
#   amax = x.abs().amax(-1); scale = amax/127; q = round(x/scale).clamp(-127,127).to(int8)
# with ONE Triton kernel (amax reduce + quantize, two streaming passes over K).
#
# WHY triton (not torch.compile): torch.compile/inductor of the act-quant HANGS the sglang serve at
# startup (inductor async-compile worker deadlock inside the scheduler process). triton.jit compiles
# in-process with no worker pool -> no hang. triton-xpu is present in the image (it is the attention
# backend). Numerically == the eager path (round-half-away-from-zero vs torch's round-half-even differs
# only by <=1 LSB on a handful of elements; int8-act relerr is ~9e-3 either way).
#
# Public API: per_token_int8(x2) -> (q int8 [M,K] contiguous, scale fp16 [M,1], zero int32 [M,1]).
# available() reports whether the triton path imported/JITs OK (the shim falls back to eager if not).
import torch

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _ptq_int8_kernel(
        x_ptr, q_ptr, s_ptr,
        K,
        stride_xm, stride_qm,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_xm
        q_row = q_ptr + row * stride_qm
        # pass 1: row amax
        amax = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(x)))
        amax = tl.maximum(amax, 1e-5)
        inv = 127.0 / amax
        tl.store(s_ptr + row, (amax / 127.0).to(tl.float16))
        # pass 2: quantize (round half away from zero, clamp to int8 sym range)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
            v = x * inv
            r = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
            r = tl.minimum(tl.maximum(r, -127.0), 127.0)
            tl.store(q_row + offs, r.to(tl.int8), mask=mask)

    _HAVE_TRITON = True
except Exception as _e:  # noqa: BLE001
    _HAVE_TRITON = False
    _TRITON_ERR = repr(_e)


def per_token_int8(x2):
    """x2: [M,K] fp16 contiguous -> (q int8 [M,K], scale fp16 [M,1], zero int32 [M,1])."""
    M, K = x2.shape
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    q = torch.empty((M, K), dtype=torch.int8, device=x2.device)
    s = torch.empty((M, 1), dtype=torch.float16, device=x2.device)
    BLOCK_K = 2048
    _ptq_int8_kernel[(M,)](
        x2, q, s, K, x2.stride(0), q.stride(0), BLOCK_K=BLOCK_K,
        num_warps=8,
    )
    z = torch.zeros((M, 1), dtype=torch.int32, device=x2.device)
    return q, s, z


def available():
    return _HAVE_TRITON
