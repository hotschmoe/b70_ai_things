#!/usr/bin/env python3
"""D1: graph-timed short-context paged FA vs upstream Triton decode.

One physical card, pinned by xpu-run. No weights or server load. Compare
normal and grouped upstream split-K implementations at the real 24/2/256
shape and the real 1600-token cache page. Metadata max-K stays 131072.
"""

import json
import os
import signal

import torch

from vllm.v1.attention.ops.triton_decode_attention import (
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
)
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
from d1_fa_plan import fill_plan


def main():
    signal.alarm(180)
    torch.manual_seed(41021)
    dev = "xpu:0"
    hq, hkv, dim, page, cap = 24, 2, 256, 1600, 131072
    nb = (cap + page - 1) // page
    q = torch.randn(1, hq, dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(nb, page, hkv, dim, device=dev, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.arange(nb, device=dev, dtype=torch.int32)[None, :]
    cu = torch.tensor([0, 1], device=dev, dtype=torch.int32)
    lens = torch.tensor([128], device=dev, dtype=torch.int32)
    scale = torch.ones((), device=dev, dtype=torch.float32)
    candidates = []

    def fa():
        return flash_attn_varlen_func(
            q=q, k=k, v=v, max_seqlen_q=1, cu_seqlens_q=cu,
            seqused_k=lens, max_seqlen_k=cap, block_table=table, causal=True,
        )

    candidates.append(("fa", fa))
    zeros = torch.zeros_like(cu)
    for splits in (4, 8, 16):
        counts = torch.empty((1,), device=dev, dtype=torch.int32)
        work = torch.empty((splits, 4), device=dev, dtype=torch.int32)
        out = torch.empty_like(q)

        def planned(splits=splits, counts=counts, work=work, out=out):
            fill_plan(lens, counts, work, splits)
            result, _ = torch.ops._vllm_fa2_C.varlen_fwd(
                q, k, v, out, cu, zeros, lens, None, table, None,
                1, cap, 0.0, None, None, None, dim ** -0.5, None,
                False, True, -1, -1, 0.0, False, None, splits,
                False, counts, work,
            )
            return result

        candidates.append((f"fa-plan-{splits}", planned))
    for name, func in (("normal", decode_attention_fwd_normal),
                       ("grouped", decode_attention_fwd_grouped)):
        for splits in (4, 8, 16):
            out = torch.empty_like(q)
            lse = torch.empty((1, hq), device=dev, dtype=torch.float32)
            mid = torch.empty((1, hq, splits, dim + 1), device=dev,
                              dtype=torch.float32)

            def run(func=func, splits=splits, out=out, lse=lse, mid=mid):
                func(q, k, v, out, lse, table, lens, mid, splits,
                     dim ** -0.5, page, k_scale=scale, v_scale=scale)
                return out

            candidates.append((f"{name}-{splits}", run))

    graphs = []
    stream = torch.xpu.Stream()
    for name, run in candidates:
        try:
            for _ in range(3):
                run()
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.stream(stream):
                with torch.xpu.graph(graph, stream=stream):
                    output = run()
            torch.xpu.synchronize()
            graphs.append((name, graph, output))
            print(json.dumps(dict(event="captured", name=name)), flush=True)
        except Exception as exc:
            print(json.dumps(dict(event="candidate_error", name=name,
                                  error=repr(exc))), flush=True)
            raise

    bad = 0
    for length in (128, 512, 900, 1024, 1536, 2047, 2048, 4096):
        lens.fill_(length)
        # FP32 reference, computed independently of either attention kernel.
        kk = k.flatten(0, 1)[:length].float().repeat_interleave(hq // hkv, 1)
        vv = v.flatten(0, 1)[:length].float().repeat_interleave(hq // hkv, 1)
        scores = torch.einsum("hd,thd->ht", q[0].float(), kk) * dim ** -0.5
        ref = torch.einsum("ht,thd->hd", scores.softmax(-1), vv)[None]
        base = None
        for name, graph, output in graphs:
            for _ in range(3):
                graph.replay()
            torch.xpu.synchronize()
            actual = output.float()
            maxabs = float((actual - ref).abs().max())
            rms = float(((actual - ref) ** 2).mean().sqrt())
            ok = bool(torch.allclose(actual, ref, atol=0.003, rtol=0.03))
            if name == "fa":
                base = output.clone()
            equal = bool(torch.equal(output, base))
            timings = []
            for _ in range(3):
                start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
                start.record()
                for _ in range(32):
                    graph.replay()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end) * 1000 / 32)
            print(json.dumps(dict(event="result", name=name, length=length,
                                  graph_us=sorted(timings)[1], maxabs=maxabs,
                                  rms=rms, reference_ok=ok, fa_bitwise=equal)), flush=True)
            bad += not ok
    print(json.dumps(dict(event="done", bad=bad, device=torch.xpu.get_device_name(0),
                          card=os.environ.get("XPU_RUN_CARD"))), flush=True)
    if bad:
        raise SystemExit(1)
    signal.alarm(0)


if __name__ == "__main__":
    main()
