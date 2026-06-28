#!/usr/bin/env python3
# gdn_fused_conv_probe.py -- B70 reproduction for the GDN conv-fusion lever (W4A8_PLAN.md
# "GDN conv-fusion"). Answers, at the real Qwen3.6-27B decode shape (B=1):
#   (1) how expensive is the per-layer causal_conv1d_update launch?      [conv-cost microbench]
#   (2) is the fused conv+recurrence kernel bitwise-correct + deterministic?  [correctness gate]
#   (3) does fusing one launch out actually buy per-layer time (eager)?  [per-layer bench]
#
# The fused kernel under test is the SHIPPED-candidate launcher in
# sglang/patches/gdn_fused_conv.py (fused_conv_recurrent_packed_decode); the reference is the
# unfused 2-kernel decode path the GDN backend actually runs:
#   causal_conv1d_update(...) ; fused_recurrent_gated_delta_rule_packed_decode(...).
#
# Real shapes (config.json text_config): HV=48 v-heads, H=16 k-heads, K=V=128, conv width=4;
#   qkv_dim = 2*H*K + HV*V = 10240; 48 GDN (linear-attn) layers / token.
#
# Usage (card 0, sglang-xpu:mtp or :woq, NO serve):
#   ZE_AFFINITY_MASK=0 python3 sglang/gdn_fused_conv_probe.py
# NOTE: the EAGER per-layer win here does NOT translate to the GRAPH=1 serve (XPUGraph capture
#   already removes the host launch overhead) -- see the e2e A/B in W4A8_PLAN.md. This probe
#   measures the eager ceiling + correctness; the e2e verdict is the serve bench.
import os, sys, time
import torch, triton

# import path: the patches dir (ships gdn_fused_conv.py) + the baked fla path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches"))
from gdn_fused_conv import fused_conv_recurrent_packed_decode  # noqa: E402
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import causal_conv1d_update  # noqa: E402
from sglang.srt.layers.attention.fla.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode,
)

DEV = "xpu"
B = int(os.environ.get("B", "1"))
H, HV, K, V, W = 16, 48, 128, 128, 4
QKV_DIM = 2 * H * K + HV * V  # 10240
STATE_LEN = W - 1             # 3 (decode usage)
NUM_SLOTS = max(B + 4, 8)
SCALE = K ** -0.5
ITERS = int(os.environ.get("ITERS", "400"))
WARMUP = int(os.environ.get("WARMUP", "80"))


def make(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(B, QKV_DIM, generator=g, dtype=torch.float32) * 0.1).to(DEV, torch.bfloat16)
    cw = (torch.randn(QKV_DIM, W, generator=g, dtype=torch.float32) * 0.2).to(DEV, torch.bfloat16)
    cb = (torch.randn(QKV_DIM, generator=g, dtype=torch.float32) * 0.1).to(DEV, torch.bfloat16)
    cs = (torch.randn(NUM_SLOTS, QKV_DIM, STATE_LEN, generator=g, dtype=torch.float32) * 0.1).to(DEV, torch.bfloat16)
    a = torch.randn(B, HV, generator=g, dtype=torch.float32).to(DEV, torch.bfloat16)
    b = torch.randn(B, HV, generator=g, dtype=torch.float32).to(DEV, torch.bfloat16)
    A_log = torch.randn(HV, generator=g, dtype=torch.float32).to(DEV)
    dt_bias = torch.randn(HV, generator=g, dtype=torch.float32).to(DEV)
    ssm = (torch.randn(NUM_SLOTS, HV, V, K, generator=g, dtype=torch.float32) * 0.1).to(DEV)
    cache_idx = (torch.arange(B, dtype=torch.int64) % NUM_SLOTS).to(DEV)
    conv_idx = cache_idx.clone().to(torch.int32)
    return x, cw, cb, cs, a, b, A_log, dt_bias, ssm, cache_idx, conv_idx


def relerr(x, y):
    x = x.float(); y = y.float()
    return (x - y).abs().max().item() / (y.abs().max().item() + 1e-12)


def reference(inp):
    x, cw, cb, cs, a, b, A_log, dt_bias, ssm, cache_idx, conv_idx = inp
    cs_r = cs.clone(); ssm_r = ssm.clone()
    post = causal_conv1d_update(x, cs_r, cw, cb, "silu", conv_state_indices=conv_idx)
    out_r = post.new_empty(B, 1, HV, V)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=post, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=SCALE,
        initial_state=ssm_r, out=out_r, ssm_state_indices=cache_idx,
        use_qk_l2norm_in_kernel=True)
    return out_r.clone(), cs_r.clone(), ssm_r.clone()


def candidate(inp):
    x, cw, cb, cs, a, b, A_log, dt_bias, ssm, cache_idx, conv_idx = inp
    cs_c = cs.clone(); ssm_c = ssm.clone()
    out = fused_conv_recurrent_packed_decode(
        in_proj_out=x, conv_state=cs_c, conv_w=cw, conv_bias=cb,
        a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=SCALE,
        ssm_state=ssm_c, cache_indices=cache_idx,
        num_v_heads=HV, head_v_dim=V, num_k_heads=H, head_k_dim=K)
    return out.clone(), cs_c.clone(), ssm_c.clone()


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.xpu.synchronize()
    last = None
    for _ in range(2):
        torch.xpu.synchronize(); t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.xpu.synchronize(); t1 = time.perf_counter()
        last = (t1 - t0) / ITERS * 1e6
    return last


def main():
    inp = make(0)
    x, cw, cb, cs, a, b, A_log, dt_bias, ssm, cache_idx, conv_idx = inp
    reference(inp); candidate(inp); torch.xpu.synchronize()

    print("=== (2) correctness: fused vs unfused 2-kernel reference (target relerr < 1e-5) ===")
    out_r, cs_r, ssm_r = reference(inp)
    out_c, cs_c, ssm_c = candidate(inp)
    print(f"  out      relerr = {relerr(out_c, out_r):.3e}")
    print(f"  conv_st  relerr = {relerr(cs_c, cs_r):.3e}")
    print(f"  ssm_st   relerr = {relerr(ssm_c, ssm_r):.3e}")
    N = int(os.environ.get("NDET", "30"))
    r0o, r0c, _ = candidate(inp); nbad = 0
    for _ in range(N):
        o_i, cs_i, _ = candidate(inp)
        if (o_i.float() - r0o.float()).abs().max().item() > 0 or \
           (cs_i.float() - r0c.float()).abs().max().item() > 0:
            nbad += 1
    print(f"  determinism over {N} repeats (identical inputs): nondeterministic = {nbad}/{N}")

    print("=== (1)/(3) eager cost (warm, discard-1st) ===")
    us_conv = bench(lambda: causal_conv1d_update(x, cs.clone(), cw, cb, "silu", conv_state_indices=conv_idx))
    csl = cs.clone(); ssml = ssm.clone()
    post0 = causal_conv1d_update(x, csl, cw, cb, "silu", conv_state_indices=conv_idx)
    out_tmp = post0.new_empty(B, 1, HV, V)
    us_recur = bench(lambda: fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=post0, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=SCALE,
        initial_state=ssml, out=out_tmp, ssm_state_indices=cache_idx, use_qk_l2norm_in_kernel=True))
    csu = cs.clone(); ssmu = ssm.clone()
    def unfused():
        post = causal_conv1d_update(x, csu, cw, cb, "silu", conv_state_indices=conv_idx)
        o = post.new_empty(B, 1, HV, V)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=post, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=SCALE,
            initial_state=ssmu, out=o, ssm_state_indices=cache_idx, use_qk_l2norm_in_kernel=True)
    us_unfused = bench(unfused)
    csf = cs.clone(); ssmf = ssm.clone()
    us_fused = bench(lambda: fused_conv_recurrent_packed_decode(
        in_proj_out=x, conv_state=csf, conv_w=cw, conv_bias=cb, a=a, b=b, A_log=A_log,
        dt_bias=dt_bias, scale=SCALE, ssm_state=ssmf, cache_indices=cache_idx,
        num_v_heads=HV, head_v_dim=V, num_k_heads=H, head_k_dim=K))
    print(f"  conv alone        : {us_conv:7.2f} us/call  (x48 = {us_conv*48/1000:.2f} ms)")
    print(f"  recurrence alone  : {us_recur:7.2f} us/call")
    print(f"  unfused (2 launch): {us_unfused:7.2f} us/call")
    print(f"  FUSED  (1 launch) : {us_fused:7.2f} us/call")
    save = us_unfused - us_fused
    print(f"  eager savings/layer = {save:7.2f} us  ->  x48 = {save*48/1000:.2f} ms  "
          f"(EAGER e2e est {save*48/1000/36.6*100:.1f}% of a 36.6 ms step;")
    print(f"   NOTE: the GRAPH=1 serve A/B shows this does NOT realize -- graph capture already "
          f"removes host launch overhead. See W4A8_PLAN.md.)")


if __name__ == "__main__":
    main()
