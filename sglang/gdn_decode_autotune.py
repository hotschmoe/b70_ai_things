#!/usr/bin/env python3
# gdn_decode_autotune.py -- B70 isolated microbench + autotune for the GDN (gated-delta-net)
# decode triton kernel used on the W4A8/int4 sglang-XPU daily driver.
#
# Target kernel: fused_recurrent_gated_delta_rule_packed_decode_kernel
#   (sglang/srt/layers/attention/fla/fused_recurrent.py == sglang/patches/fused_recurrent.py)
#   This is the M=1 packed-decode variant the GDN backend calls per linear-attn layer at decode
#   (gating fused inside; use_qk_l2norm_in_kernel=True). 48 such layers / token on Qwen3.6-27B.
#
# Real Qwen3.6-27B GDN shapes (from config.json): HV=48 v-heads, H=16 k-heads, K=V=128, conv=4.
#   qkv_dim = 2*H*K + HV*V = 2*16*128 + 48*128 = 10240
#   Default launch: BK = npo2(K) = 128, BV = min(npo2(V),32) = 32, NV = cdiv(V,BV) = 4,
#                   grid = (NV, B*HV) = (4, 48) = 192 programs (B=1); num_stages=3, num_warps=1.
#
# Sweep BV cap {8,16,32,64,128} (grid parallelism), num_warps {1,2,4,8}, num_stages {1,2,3}.
# Numerics: BV tiling is bitwise-identical (V-rows independent, K-reduction full); warps/stages
# checked vs the default kernel for relerr < 1e-5 (fp32 SSM math). Warm timing, discard 1st run.
#
# Usage (card 0, sglang-xpu:woq or :mtp, NO serve):
#   ZE_AFFINITY_MASK=0 python3 sglang/gdn_decode_autotune.py
# B env var overrides decode batch (default 1 == the captured bs=1 graph shape).

import os
import time

import torch
import triton

from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_packed_decode_kernel,
)

DEV = "xpu"
B = int(os.environ.get("B", "1"))
HV, H, K, V = 48, 16, 128, 128
QKV_DIM = 2 * H * K + HV * V  # 10240
NUM_SLOTS = max(B + 2, 8)
SCALE = K ** -0.5
ITERS = int(os.environ.get("ITERS", "300"))
WARMUP = int(os.environ.get("WARMUP", "50"))


def make_inputs(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mixed_qkv = (torch.randn(B, QKV_DIM, generator=g, dtype=torch.float32) * 0.1).to(DEV, torch.bfloat16)
    a = torch.randn(B, HV, generator=g, dtype=torch.float32).to(DEV, torch.bfloat16)
    b = torch.randn(B, HV, generator=g, dtype=torch.float32).to(DEV, torch.bfloat16)
    A_log = torch.randn(HV, generator=g, dtype=torch.float32).to(DEV)
    dt_bias = torch.randn(HV, generator=g, dtype=torch.float32).to(DEV)
    ssm0 = (torch.randn(NUM_SLOTS, HV, V, K, generator=g, dtype=torch.float32) * 0.1).to(DEV)
    cache_indices = (torch.arange(B, dtype=torch.int64) % NUM_SLOTS).to(DEV)
    return mixed_qkv, a, b, A_log, dt_bias, ssm0, cache_indices


def launch(mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices, out,
           bv_cap, num_stages, num_warps):
    """Parameterized re-implementation of fused_recurrent_gated_delta_rule_packed_decode's
    launch (kernel unchanged) so we can sweep BV cap / num_stages / num_warps."""
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), bv_cap)
    NV = triton.cdiv(V, BV)
    grid = (NV, B * HV)
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        o=out, h0=ssm_states, ht=ssm_states, ssm_state_indices=cache_indices, scale=SCALE,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0), stride_b_tok=b.stride(0),
        stride_init_state_token=ssm_states.stride(0),
        stride_final_state_token=ssm_states.stride(0),
        stride_indices_seq=cache_indices.stride(0),
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        SOFTPLUS_THRESHOLD=20.0, USE_QK_L2NORM_IN_KERNEL=True,
        num_warps=num_warps, num_stages=num_stages,
    )


def ref_output(inp):
    """Reference == the shipped launcher (BV=32, stages=3, warps=1)."""
    mixed_qkv, a, b, A_log, dt_bias, ssm0, cache_indices = inp
    ssm = ssm0.clone()
    out = mixed_qkv.new_empty(B, 1, HV, V)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        scale=SCALE, initial_state=ssm, out=out, ssm_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return out.clone(), ssm.clone()


def cand_output(inp, bv_cap, num_stages, num_warps):
    mixed_qkv, a, b, A_log, dt_bias, ssm0, cache_indices = inp
    ssm = ssm0.clone()
    out = mixed_qkv.new_empty(B, 1, HV, V)
    launch(mixed_qkv, a, b, A_log, dt_bias, ssm, cache_indices, out,
           bv_cap, num_stages, num_warps)
    return out.clone(), ssm.clone()


def relerr(x, y):
    x = x.float(); y = y.float()
    d = (x - y).abs().max().item()
    s = y.abs().max().item() + 1e-12
    return d / s


def bench(inp, bv_cap, num_stages, num_warps):
    mixed_qkv, a, b, A_log, dt_bias, ssm0, cache_indices = inp
    ssm = ssm0.clone()
    out = mixed_qkv.new_empty(B, 1, HV, V)

    def run_n(n):
        for _ in range(n):
            launch(mixed_qkv, a, b, A_log, dt_bias, ssm, cache_indices, out,
                   bv_cap, num_stages, num_warps)

    # warmup (compile + warm the card; B70 idle-downclocks)
    run_n(WARMUP)
    torch.xpu.synchronize()
    # discard-1st: two timed passes, report the 2nd (warm)
    times = []
    for _ in range(2):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        run_n(ITERS)
        torch.xpu.synchronize()
        times.append((time.perf_counter() - t0) / ITERS * 1e3)  # ms/call
    return times[1], times[0]  # warm, cold


def main():
    print(f"# GDN packed-decode autotune | B={B} HV={HV} H={H} K={K} V={V} "
          f"qkv_dim={QKV_DIM} | ITERS={ITERS} WARMUP={WARMUP}")
    inp = make_inputs()
    ref_out, ref_ssm = ref_output(inp)

    # default config baseline
    base_bv = min(triton.next_power_of_2(V), 32)
    print(f"# default: BV={base_bv} NV={triton.cdiv(V,base_bv)} "
          f"grid=({triton.cdiv(V,base_bv)},{B*HV}) stages=3 warps=1")
    base_warm, _ = bench(inp, 32, 3, 1)
    print(f"# baseline warm = {base_warm*1e3:.2f} us/call ({base_warm:.5f} ms)\n")

    bv_caps = [8, 16, 32, 64, 128]
    warps_list = [1, 2, 4, 8]
    stages_list = [1, 2, 3]

    rows = []
    for bv_cap in bv_caps:
        for nw in warps_list:
            for ns in stages_list:
                BV = min(triton.next_power_of_2(V), bv_cap)
                NV = triton.cdiv(V, BV)
                try:
                    o, s = cand_output(inp, bv_cap, ns, nw)
                    re_o = relerr(o, ref_out)
                    re_s = relerr(s, ref_ssm)
                    warm, cold = bench(inp, bv_cap, ns, nw)
                except Exception as e:
                    print(f"BV={BV:3d} NV={NV} warps={nw} stages={ns}  FAIL {type(e).__name__}: {str(e)[:80]}")
                    continue
                speed = base_warm / warm
                ok = "OK " if (re_o < 1e-5 and re_s < 1e-5) else "NUM"
                rows.append((warm, speed, BV, NV, nw, ns, re_o, re_s, ok))
                print(f"BV={BV:3d} NV={NV} grid=({NV},{B*HV})={NV*B*HV:4d} warps={nw} stages={ns}  "
                      f"warm={warm*1e3:7.2f}us  {speed:5.3f}x  relerr o={re_o:.2e} s={re_s:.2e}  {ok}")

    print("\n# ===== TOP (numerically-clean, relerr<1e-5) by speed =====")
    clean = sorted([r for r in rows if r[8] == "OK "], key=lambda r: r[0])
    for warm, speed, BV, NV, nw, ns, re_o, re_s, ok in clean[:10]:
        print(f"  BV={BV:3d} NV={NV} warps={nw} stages={ns}  warm={warm*1e3:7.2f}us  {speed:5.3f}x")
    print(f"\n# baseline = {base_warm*1e3:.2f}us; best-clean = {clean[0][0]*1e3:.2f}us "
          f"= {clean[0][1]:.3f}x (BV={clean[0][2]} warps={clean[0][4]} stages={clean[0][5]})"
          if clean else "# no clean config")


if __name__ == "__main__":
    main()
