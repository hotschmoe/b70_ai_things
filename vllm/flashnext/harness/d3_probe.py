#!/usr/bin/env python3
"""D3 microbench: stock SYCL GDN vs fused decode, FA2 vs paged decode.

Card 1, inside xpu-run. No serve. Prints CONFIG/RESULT lines and writes
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/d3-probe.json.

Ornith geometry only: 30 GDN layers would multiply the per-layer times.
This process times one layer.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback

import torch

sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_gdn_xpu")

OUT_PATH = "/mnt/vm_8tb/b70/build/vllm-flashnext/logs/d3-probe.json"
ROOF = 608.0  # GB/s, D2's card roof
ROWS = []


def say(msg: str) -> None:
    print(msg, flush=True)
    ROWS.append(msg)


def stats(a: torch.Tensor, b: torch.Tensor) -> str:
    d = (a.float() - b.float()).abs()
    denom = a.float().norm().clamp_min(1e-8)
    cos = torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1), b.float().reshape(1, -1)
    ).item()
    return f"max_abs={d.max().item():.4e} mean_abs={d.mean().item():.4e} cos={cos:.6f}"


def replay_ms(fn, nreplay: int) -> tuple[float, str]:
    fn()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(graph):
            fn()
    except Exception as exc:
        return -1.0, f"capture_fail {exc!r}"
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(nreplay):
        graph.replay()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / float(nreplay)
    del graph
    torch.xpu.empty_cache()
    return ms, "graph"


def gbps(nbytes: int, ms: float) -> float:
    if ms <= 0:
        return 0.0
    return (nbytes / (ms * 1e-3)) / 1e9


def byte_model() -> None:
    conv_state = 3 * 8192 * 2
    conv_w = 8192 * 4 * 2
    ssm = 32 * 128 * 128 * 4
    qkvz = 12288 * 2
    ba = 64 * 2
    outs = 2 * 32 * 128 * 2
    # read conv + write conv + read ssm + write ssm + weights + activations
    traffic = conv_state * 2 + conv_w + ssm * 2 + qkvz + ba + outs
    say(
        "BYTES one GDN layer T=1 non-spec: "
        f"conv_state={conv_state} B (fp16 [3,8192]) "
        f"conv_w={conv_w} B ssm={ssm} B fp32 [32,128,128] "
        f"read+write_ssm={ssm * 2} B qkvz={qkvz} B ba={ba} B "
        f"core+out={outs} B traffic={traffic} B "
        f"roof_us={traffic / (ROOF * 1e3):.2f}"
    )
    say(
        "BYTES 30 GDN layers state traffic "
        f"{30 * ssm * 2 / 1e6:.2f} MB roof {30 * ssm * 2 / ROOF / 1e6:.3f} ms. "
        "D2 qkvz GEMV 0.093 ms/layer (540 GB/s) dominates the 4.79 ms class."
    )
    say(
        "ATTN bytes/token fp16 KV GQA read-once: "
        f"{2 * 2 * 256 * 2} B. "
        "S=32 ~0.1 us roof, S=8192 ~27 us, S=32768 ~110 us, "
        "S=131072 ~442 us, S=200000 ~674 us per full-attn layer. "
        "10 layers at 200k ~6.7 ms at the roof if KV is read once."
    )


def make_gdn(T: int, rows: int, slots: int, seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    dev = "xpu"
    qkvz = torch.randn(T, 12288, dtype=torch.float16, device=dev)
    ba = torch.randn(T, 64, dtype=torch.float16, device=dev)
    conv = (torch.randn(slots, rows, 8192, generator=g) * 0.1).to(dev).half()
    ssm = (torch.randn(slots, 32, 128, 128, generator=g) * 0.05).to(dev)
    conv_w = (torch.randn(8192, 4, generator=g) * 0.1).to(dev).half()
    A_log = torch.randn(32, generator=g).to(dev)
    dt_bias = torch.randn(32, generator=g).to(dev).half()
    norm_w = (torch.randn(128, generator=g) * 0.1 + 1).to(dev)
    return qkvz, ba, conv, ssm, conv_w, A_log, dt_bias, norm_w


def ref_nonspec(qkvz, ba, conv, ssm, conv_w, A_log, dt_bias, norm_w, eps=1e-6):
    """CPU oracle for non-spec, slot 0, has_initial, packed Qwen3.5 layout."""
    hist = conv[0].detach().float().cpu().clone()
    state = ssm[0].detach().float().cpu().clone()
    w = conv_w.detach().float().cpu()
    A = A_log.detach().float().cpu()
    dt = dt_bias.detach().float().cpu()
    gamma = norm_w.detach().float().cpu()
    scale = 1.0 / math.sqrt(128.0)
    cores = []
    outs = []
    T = qkvz.shape[0]
    for t in range(T):
        src = qkvz[t].detach().float().cpu()
        x = torch.zeros(8192)
        x[:2048] = src[:2048]
        x[2048:4096] = src[2048:4096]
        x[4096:8192] = src[4096:8192]
        window = torch.stack([hist[0], hist[1], hist[2], x], 0)
        acc = (window * w.transpose(0, 1)).sum(0)
        hist = torch.stack([hist[1], hist[2], x], 0)
        y = acc / (1.0 + torch.exp(-acc))
        y = y.half().float()
        q = y[:2048].view(16, 128)
        k = y[2048:4096].view(16, 128)
        v = y[4096:8192].view(32, 128)
        z = src[8192:].half().float().view(32, 128)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        b = ba[t, :32].detach().float().cpu()
        a = ba[t, 32:].detach().float().cpu()
        beta = 1.0 / (1.0 + torch.exp(-b))
        ax = a + dt
        sp = torch.where(ax < 20, torch.log(1.0 + torch.exp(ax)), ax)
        g = torch.exp(-torch.exp(A) * sp)
        core = torch.empty(32, 128)
        for h in range(32):
            hk = h // 2
            S = state[h] * g[h]
            kv = S @ k[hk]
            delta = (v[h] - kv) * beta[h]
            S = S + delta[:, None] * k[hk][None, :]
            state[h] = S
            core[h] = S @ q[hk]
        core_h = core.half().float()
        inv = torch.rsqrt(core_h.pow(2).mean(-1, keepdim=True) + eps)
        out = core_h * inv * gamma * (z / (1.0 + torch.exp(-z)))
        cores.append(core.half())
        outs.append(out.half())
    return (
        torch.stack(cores),
        torch.stack(outs),
        hist.half(),
        state,
    )


def stock_nonspec(core, z, qkvz, ba, conv, ssm, conv_w, A_log, dt_bias, cu, slot, has_initial):
    torch.ops._xpu_C.gdn_attention(
        core,
        z,
        qkvz,
        ba,
        16,
        32,
        128,
        128,
        conv,
        ssm,
        conv_w,
        None,
        "silu",
        A_log,
        dt_bias,
        0,
        1,
        0,
        has_initial,
        cu,
        None,
        slot,
        None,
        None,
        None,
        None,
        qkvz.shape[0],
        1,
        True,
    )


def run_gdn() -> None:
    say("--- GDN non-spec ---")
    for T in (1, 3):
        qkvz, ba, conv0, ssm0, conv_w, A_log, dt_bias, norm_w = make_gdn(
            T, rows=3, slots=4, seed=10 + T
        )
        cu = torch.tensor([0, T], dtype=torch.int32, device="xpu")
        slot = torch.tensor([0], dtype=torch.int32, device="xpu")
        has_initial = torch.ones(1, dtype=torch.bool, device="xpu")
        core_s = torch.zeros(T, 32, 128, dtype=torch.float16, device="xpu")
        z_s = torch.empty_like(core_s)
        conv_s = conv0.clone()
        ssm_s = ssm0.clone()
        say(f"CONFIG gdn stock call T={T}")
        try:
            stock_nonspec(
                core_s, z_s, qkvz, ba, conv_s, ssm_s, conv_w, A_log, dt_bias, cu, slot, has_initial
            )
            torch.xpu.synchronize()
        except Exception as exc:
            say(f"RESULT stock T={T} FAIL {exc!r}")
            traceback.print_exc()
            continue
        core_f = torch.zeros_like(core_s)
        out_f = torch.empty_like(core_s)
        conv_f = conv0.clone()
        ssm_f = ssm0.clone()
        say(f"CONFIG gdn fused call T={T}")
        try:
            torch.ops.b70_gdn.decode_fused(
            core_f,
            out_f,
            qkvz,
            ba,
            conv_f,
            ssm_f,
            conv_w,
            A_log,
            dt_bias,
            norm_w,
            cu,
            slot,
            has_initial,
            None,
            None,
            1e-6,
        )
            torch.xpu.synchronize()
        except Exception as exc:
            say(f"RESULT fused T={T} FAIL {exc!r}")
            traceback.print_exc()
            continue
        z_ref = qkvz[:, 8192:].reshape(T, 32, 128)
        gated = torch.nn.functional.silu(z_s.float())
        var = core_s.float().pow(2).mean(-1, keepdim=True)
        stock_out = (core_s.float() * torch.rsqrt(var + 1e-6) * norm_w * gated).half()
        say(f"RESULT T={T} core fused_vs_stock {stats(core_f, core_s)}")
        say(f"RESULT T={T} z stock_vs_proj {stats(z_s, z_ref)}")
        say(f"RESULT T={T} out fused_vs_stock_rms {stats(out_f, stock_out)}")
        say(f"RESULT T={T} conv_state fused_vs_stock {stats(conv_f, conv_s)}")
        say(f"RESULT T={T} ssm fused_vs_stock {stats(ssm_f, ssm_s)}")
        try:
            rc, ro, rh, rs = ref_nonspec(qkvz, ba, conv0, ssm0, conv_w, A_log, dt_bias, norm_w)
            say(f"RESULT T={T} core fused_vs_ref {stats(core_f.cpu(), rc)}")
            say(f"RESULT T={T} out fused_vs_ref {stats(out_f.cpu(), ro)}")
            say(f"RESULT T={T} conv fused_vs_ref {stats(conv_f[0].cpu(), rh)}")
            say(f"RESULT T={T} ssm fused_vs_ref {stats(ssm_f[0].cpu(), rs)}")
            say(f"RESULT T={T} core stock_vs_ref {stats(core_s.cpu(), rc)}")
        except Exception as exc:
            say(f"RESULT T={T} ref FAIL {exc!r}")

        def run_stock():
            stock_nonspec(
                core_s, z_s, qkvz, ba, conv_s, ssm_s, conv_w, A_log, dt_bias, cu, slot, has_initial
            )

        def run_fused():
            torch.ops.b70_gdn.decode_fused(
                core_f,
                out_f,
                qkvz,
                ba,
                conv_f,
                ssm_f,
                conv_w,
                A_log,
                dt_bias,
                norm_w,
                cu,
                slot,
                has_initial,
                None,
                None,
                1e-6,
            )

        ms_s, how_s = replay_ms(run_stock, 30)
        ms_f, how_f = replay_ms(run_fused, 30)
        traffic = 3 * 8192 * 2 * 2 + 8192 * 4 * 2 + 32 * 128 * 128 * 4 * 2 + T * (12288 * 2 + 128)
        say(
            f"RESULT T={T} time stock {ms_s:.4f} ms ({how_s}) "
            f"fused {ms_f:.4f} ms ({how_f}) "
            f"traffic {traffic} B fused_GBps {gbps(traffic, ms_f):.1f} "
            f"roof_us {traffic / ROOF / 1e6:.2f}"
        )


def run_gdn_spec() -> None:
    say("--- GDN spec T=3 (MTP verify shape, num_spec_tokens=3) ---")
    T = 3
    qkvz, ba, conv0, ssm0, conv_w, A_log, dt_bias, norm_w = make_gdn(
        T, rows=5, slots=4, seed=77
    )
    token_indx = torch.arange(T, dtype=torch.int32, device="xpu")
    num_accepted = torch.tensor([1], dtype=torch.int32, device="xpu")
    slot = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="xpu")
    cu = torch.tensor([0, T], dtype=torch.int32, device="xpu")
    core_s = torch.zeros(T, 32, 128, dtype=torch.float16, device="xpu")
    z_s = torch.empty_like(core_s)
    conv_s = conv0.clone()
    ssm_s = ssm0.clone()
    try:
        torch.ops._xpu_C.gdn_attention(
            core_s,
            z_s,
            qkvz,
            ba,
            16,
            32,
            128,
            128,
            conv_s,
            ssm_s,
            conv_w,
            None,
            "silu",
            A_log,
            dt_bias,
            0,
            0,
            1,
            None,
            None,
            None,
            None,
            cu,
            token_indx,
            slot,
            num_accepted,
            T,
            1,
            True,
        )
        torch.xpu.synchronize()
    except Exception as exc:
        say(f"RESULT spec stock FAIL {exc!r}")
        traceback.print_exc()
        return
    core_f = torch.zeros_like(core_s)
    out_f = torch.empty_like(core_s)
    conv_f = conv0.clone()
    ssm_f = ssm0.clone()
    has_initial = torch.ones(1, dtype=torch.bool, device="xpu")
    say("CONFIG gdn fused spec call")
    try:
        torch.ops.b70_gdn.decode_fused(
            core_f,
            out_f,
            qkvz,
            ba,
            conv_f,
            ssm_f,
            conv_w,
            A_log,
            dt_bias,
            norm_w,
            cu,
            slot,
            has_initial,
            token_indx,
            num_accepted,
            1e-6,
        )
        torch.xpu.synchronize()
    except Exception as exc:
        say(f"RESULT spec fused FAIL {exc!r}")
        traceback.print_exc()
        return
    say(f"RESULT spec core fused_vs_stock {stats(core_f, core_s)}")
    say(f"RESULT spec conv fused_vs_stock {stats(conv_f, conv_s)}")
    say(f"RESULT spec ssm fused_vs_stock {stats(ssm_f, ssm_s)}")
    z_ref = qkvz[:, 8192:].reshape(T, 32, 128)
    say(f"RESULT spec z stock_vs_proj {stats(z_s, z_ref)}")

    def run_stock():
        torch.ops._xpu_C.gdn_attention(
            core_s, z_s, qkvz, ba, 16, 32, 128, 128, conv_s, ssm_s, conv_w, None,
            "silu", A_log, dt_bias, 0, 0, 1, None, None, None, None, cu, token_indx,
            slot, num_accepted, T, 1, True,
        )

    def run_fused():
        torch.ops.b70_gdn.decode_fused(
            core_f, out_f, qkvz, ba, conv_f, ssm_f, conv_w, A_log, dt_bias, norm_w,
            cu, slot, has_initial, token_indx, num_accepted, 1e-6,
        )

    ms_s, how_s = replay_ms(run_stock, 20)
    ms_f, how_f = replay_ms(run_fused, 20)
    say(f"RESULT spec time stock {ms_s:.4f} ms ({how_s}) fused {ms_f:.4f} ms ({how_f})")


def fa_direct(q, k, v, out, cu, seqused, bt, scale, num_splits, S, T):
    """Long XPU schema from vllm-xpu-kernels flash_api.cpp varlen_fwd.

    T=1 paged decode ignores is_causal inside the kernel. num_splits None
    lets get_num_splits pick the split count. This is the C1 path.
    """
    dummy = torch.zeros_like(cu)
    return torch.ops._vllm_fa2_C.varlen_fwd(
        q,
        k,
        v,
        out,
        cu,
        dummy,
        seqused,
        None,
        bt,
        None,
        T,
        S,
        0.0,
        None,
        None,
        None,
        scale,
        None,
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        None,
        num_splits,
        False,
        None,
        None,
    )


def fa_serve(q, k, v, out, cu, seqused, bt, scale, S, T):
    """Serve path: vllm_xpu_kernels.flash_attn_varlen_func.

    T>1 causal paged batches (MTP verify) are rewritten to independent
    qlen=1 decodes before the C op. num_splits_kv stays None, so the
    decode kernel uses get_num_splits.
    """
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

    return flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=T,
        cu_seqlens_q=cu,
        max_seqlen_k=S,
        seqused_k=seqused,
        block_table=bt,
        softmax_scale=scale,
        causal=True,
        fa_version=2,
        num_splits=0,
        out=out,
    )


def build_cache(S: int, block: int):
    nblocks = (S + block - 1) // block
    # LBNHC physical: [blocks, kv_heads, block, 2*dim]
    raw = torch.randn(nblocks, 2, block, 512, dtype=torch.float16, device="xpu")
    viewed = raw.transpose(1, 2)  # [blocks, block, 2, 512]
    k, v = viewed.split(256, dim=-1)
    bt = torch.arange(nblocks, dtype=torch.int32, device="xpu").view(1, nblocks)
    return raw, k, v, bt


def run_attn() -> None:
    say("--- attention decode ---")
    scale = 1.0 / math.sqrt(256.0)
    # S=1 exact: softmax weight is 1, out equals V.
    raw, k, v, bt = build_cache(1, 64)
    # Plant a known V in the physical layout: [block, head, token, 256:512]
    raw.zero_()
    raw[:, :, 0, 256:] = 0.25
    T = 1
    q = torch.randn(T, 16, 256, dtype=torch.float16, device="xpu")
    cu = torch.tensor([0, T], dtype=torch.int32, device="xpu")
    seqused = torch.tensor([1], dtype=torch.int32, device="xpu")
    out = torch.empty_like(q)
    try:
        torch.ops.b70_gdn.paged_decode(out, q, k, v, bt, seqused, cu, scale, 1, None, None, None)
        torch.xpu.synchronize()
    except Exception as exc:
        say(f"RESULT attn S=1 ours FAIL {exc!r}")
        traceback.print_exc()
        out = None
    target = torch.full_like(q, 0.25)
    say(
        f"CONFIG LBNHC k shape={tuple(k.shape)} stride={tuple(k.stride())} "
        f"v stride={tuple(v.stride())}"
    )
    if out is not None:
        say(f"RESULT attn S=1 ours_vs_V {stats(out, target)}")
    try:
        out_fa = torch.empty_like(q)
        fa_serve(q, k, v, out_fa, cu, seqused, bt, scale, 1, T)
        torch.xpu.synchronize()
        say(f"RESULT attn S=1 fa_vs_V {stats(out_fa, target)}")
        if out is not None:
            say(f"RESULT attn S=1 ours_vs_fa {stats(out, out_fa)}")
    except Exception as exc:
        say(f"RESULT attn FA smoke FAIL {exc!r}")
        traceback.print_exc()

    seqs = (32, 8192, 32768, 131072, 200000)
    blocks = (64, 1088)
    for S in seqs:
        for block in blocks:
            for T in (1, 3):
                say(f"CONFIG attn S={S} block={block} T={T}")
                try:
                    raw, k, v, bt = build_cache(S, block)
                except RuntimeError as exc:
                    say(f"RESULT attn alloc S={S} block={block} FAIL {exc!r}")
                    torch.xpu.empty_cache()
                    continue
                q = torch.randn(T, 16, 256, dtype=torch.float16, device="xpu")
                cu = torch.tensor([0, T], dtype=torch.int32, device="xpu")
                seqused = torch.tensor([S], dtype=torch.int32, device="xpu")
                # 2 kv heads * (K+V) * fp16, read once per query token.
                nbytes = S * 2 * 2 * 256 * 2
                nreplay = 8 if S >= 32768 else 20
                out_ours = torch.empty_like(q)
                for splits in (1, 8, 32):
                    partial_o = partial_m = partial_l = None
                    if splits > 1:
                        partial_o = torch.empty(splits, T, 16, 256, dtype=torch.float32, device="xpu")
                        partial_m = torch.empty(splits, T, 16, dtype=torch.float32, device="xpu")
                        partial_l = torch.empty_like(partial_m)

                    def run_ours(splits=splits, partial_o=partial_o, partial_m=partial_m, partial_l=partial_l):
                        torch.ops.b70_gdn.paged_decode(
                            out_ours, q, k, v, bt, seqused, cu, scale, splits,
                            partial_o, partial_m, partial_l,
                        )

                    try:
                        if splits == 1:
                            run_ours()
                            torch.xpu.synchronize()
                        ms, how = replay_ms(run_ours, nreplay)
                    except Exception as exc:
                        say(
                            f"RESULT attn ours S={S} block={block} T={T} "
                            f"splits={splits} FAIL {exc!r}"
                        )
                        traceback.print_exc()
                        continue
                    # T queries re-read KV.
                    moved = nbytes * T
                    say(
                        f"RESULT attn ours S={S} block={block} T={T} splits={splits} "
                        f"{ms:.4f} ms ({how}) GBps {gbps(moved, ms):.1f} "
                        f"roof_us {moved / ROOF / 1e6:.1f}"
                    )
                out_fa = torch.empty_like(q)

                def run_serve(out_fa=out_fa):
                    fa_serve(q, k, v, out_fa, cu, seqused, bt, scale, S, T)

                try:
                    run_serve()
                    torch.xpu.synchronize()
                    ms, how = replay_ms(run_serve, nreplay)
                    say(
                        f"RESULT attn fa_serve S={S} block={block} T={T} "
                        f"{ms:.4f} ms ({how}) GBps {gbps(nbytes * T, ms):.1f}"
                    )
                except Exception as exc:
                    say(f"RESULT attn fa_serve S={S} block={block} T={T} FAIL {exc!r}")
                    traceback.print_exc()
                if T == 1:
                    out_fa1 = torch.empty_like(q)

                    def run_fa1(out_fa1=out_fa1):
                        fa_direct(q, k, v, out_fa1, cu, seqused, bt, scale, 1, S, T)

                    try:
                        run_fa1()
                        torch.xpu.synchronize()
                        ms, how = replay_ms(run_fa1, nreplay)
                        say(
                            f"RESULT attn fa_splits1 S={S} block={block} T={T} "
                            f"{ms:.4f} ms ({how}) GBps {gbps(nbytes, ms):.1f}"
                        )
                    except Exception as exc:
                        say(f"RESULT attn fa_splits1 S={S} block={block} FAIL {exc!r}")
                        traceback.print_exc()
                # Compare splits=1 against the serve wrapper on this cache.
                out_fa = torch.empty_like(q)
                out_ours.zero_()
                try:
                    fa_serve(q, k, v, out_fa, cu, seqused, bt, scale, S, T)
                    torch.ops.b70_gdn.paged_decode(
                        out_ours, q, k, v, bt, seqused, cu, scale, 1, None, None, None
                    )
                    torch.xpu.synchronize()
                    say(
                        f"RESULT attn ours_vs_fa S={S} block={block} T={T} "
                        f"{stats(out_ours, out_fa)}"
                    )
                except Exception as exc:
                    say(f"RESULT attn compare FAIL S={S} {exc!r}")
                    traceback.print_exc()
                del raw, k, v, bt, q
                torch.xpu.empty_cache()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("all", "gdn", "attn"), default="all")
    args = parser.parse_args()
    global OUT_PATH
    if args.only != "all":
        OUT_PATH = OUT_PATH.replace(".json", f"-{args.only}.json")
    say("CONFIG device card via ZE_AFFINITY_MASK, torch " + torch.__version__)
    say(f"CONFIG xpu_count={torch.xpu.device_count()} name={torch.xpu.get_device_name(0)}")
    import vllm_xpu_kernels._xpu_C  # registers gdn_attention on torch.ops._xpu_C
    import vllm  # fa2 and the rest of the runtime

    say(f"CONFIG vllm {getattr(vllm, '__version__', '?')}")
    import b70_gdn_xpu

    b70_gdn_xpu.register()
    ops = dir(torch.ops._xpu_C)
    for name in (
        "gdn_attention",
        "causal_conv1d_non_spec",
        "gated_delta_rule_non_spec",
        "causal_conv1d_spec",
        "gated_delta_rule_spec",
    ):
        say(f"CONFIG op {name} {name in ops}")
    say(f"CONFIG fa2 {hasattr(torch.ops, '_vllm_fa2_C')}")
    say("CONFIG fa path vllm_xpu_kernels long varlen_fwd; num_splits None = get_num_splits")
    try:
        byte_model()
        if args.only in ("all", "gdn"):
            run_gdn()
            run_gdn_spec()
        if args.only in ("all", "attn"):
            run_attn()
    finally:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w", encoding="ascii") as fh:
            json.dump({"lines": ROWS}, fh, indent=2)
            fh.write("\n")
        say(f"WROTE {OUT_PATH}")


if __name__ == "__main__":
    main()
