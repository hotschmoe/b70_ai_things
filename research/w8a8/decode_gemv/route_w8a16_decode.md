# route_w8a16_decode -- the REAL W8A8 decode lever (additive, env-gated)

The int8 GEMM is already at the BW roofline (FINDINGS.md). The avoidable cost at
small M (decode + MTP verify) is the per-token INT8 ACTIVATION QUANT. The fix is
not a new kernel -- it is to route small-M linear calls through the quant-free
`int8_gemm_w8a16` op (f16 activation x s8 weight, per-channel scale, one launch),
which reads the SAME s8 weight bytes as the s8s8 path but skips the act-quant.
Reserve the s8s8 `int8_gemm_w8a8` / `..._fusedq` (int8 activation -> XMX 2x
compute) for large-M prefill, where compute -- not BW -- is the limit.

Crossover: `int8_gemm_w8a16` stays weight-BW-bound (i.e. optimal, same time as
M=1) up to M ~= 157 on down_proj (f16 compute vs 581 GB/s). MTP verify is M~6.
So a threshold of 64 is safe and conservative; only true prefill (M>64) takes the
s8s8 route. Env: `B70_W8A16_M_MAX` (default 64).

W8A16 is also MORE accurate at decode (f16 act, relerr ~9e-3 vs s8 act ~1.3e-2).

Land only when the microbench (bench_decode_gemv.py) shows W8A16(graph) faster-or-
equal at M in {1,2,4,6,8} AND a serve sweep stays coherent (AGENTS.md rule).

--------------------------------------------------------------------------------
## Patch A -- sglang: sglang/patches/w8a8_shim.py  (weight already in NT form)

The FUSED path already routes `M == 1` to `int8_gemm_w8a16`. Extend the threshold
to cover the MTP verify batch and light concurrency. In `_apply_fused`:

    -        if M == 1:
    +        W8A16_M_MAX = int(os.environ.get("B70_W8A16_M_MAX", "64"))
    +        if M <= W8A16_M_MAX:
                 out = torch.ops._xpu_C.int8_gemm_w8a16(xf, layer.B_nt, layer.wscale_n, b)  # decode+MTP, quant-free
             else:
                 # ... unchanged s8s8 prefill (act-quant + int8_gemm_w8a8) ...

`layer.B_nt` is already the `[K,N]` s8 NT view (stride0==1) and `layer.wscale_n`
the `[N]` f16 per-channel scale -- exactly what the op wants. No repack.

--------------------------------------------------------------------------------
## Patch B -- vllm: vllm/contrib/vllm_int8_xpu/xpu_int8.py  (needs an NT weight view)

The vllm path always calls `int8_gemm_w8a8_fusedq` (act-quant at EVERY M, incl.
M=1). Two additive changes:

1. In `process_weights_after_loading`, keep an NT weight view for the w8a16 op.
   Current code stores `w_q = weight.t().contiguous()` = `[K,N]` contiguous
   (stride0==N, the NN form int8_gemm_w8a8 wants). Add an NT view from the
   original `[N,K]` contiguous backing plus an `[N]` f16 scale:

        # after the existing replace_parameter() calls:
        w_NK = weight.contiguous()              # [N,K] s8 backing (keep alive)
        layer._w8a16_B_backing = w_NK
        layer._w8a16_B_nt = w_NK.t()            # [K,N] view, stride0==1 (NT)
        layer._w8a16_wscale = weight_scale.reshape(-1).to(torch.float16)  # [N]

2. In `apply_weights`, route small M to the quant-free op:

        x_2d = x.reshape(-1, x.shape[-1])
        M = x_2d.shape[0]
        W8A16_M_MAX = int(os.environ.get("B70_W8A16_M_MAX", "64"))
        if M <= W8A16_M_MAX and hasattr(layer, "_w8a16_B_nt"):
            out = torch.ops._xpu_C.int8_gemm_w8a16(
                x_2d.to(torch.float16), layer._w8a16_B_nt,
                layer._w8a16_wscale, bias)
            return out.reshape(x.shape[:-1] + (out.size(-1),))
        # else fall through to the existing fusedq / two-step s8s8 prefill path

Register a `register_fake` for `int8_gemm_w8a16` (mirror the existing
`_fake_int8_gemm`) so XPU graph capture / dynamo tracing sees a shape-consistent
op: output `A.shape[:-1] + (B.size(-1),)`, dtype `A.dtype`.

--------------------------------------------------------------------------------
## Rollback
`B70_W8A16_M_MAX=1` restores the old behavior (sglang: M==1 only; vllm: fusedq at
M>1). `B70_W8A16_M_MAX=0` forces the s8s8 path at all M. Both changes are additive
and env-gated; production defaults are unchanged until the env is set.
