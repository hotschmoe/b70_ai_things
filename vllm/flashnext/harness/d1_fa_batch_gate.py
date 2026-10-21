#!/usr/bin/env python3
"""Native compact FA: actual enabled B2/B4/B8 graph shape correctness."""

import json
import signal

import torch

from d1_fa_plan import paged_attention
# Register the unchanged native op.
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func  # noqa: F401


def main():
    signal.alarm(180)
    torch.manual_seed(41022)
    device = "xpu:0"
    page, blocks, logical_pages = 1600, 96, 82
    k = torch.randn(blocks, page, 2, 256, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    bad = 0
    for batch in (2, 4, 8):
        q = torch.randn(batch, 24, 256, device=device, dtype=torch.bfloat16)
        cu = torch.arange(batch+1, device=device, dtype=torch.int32)
        lens = torch.ones(batch, device=device, dtype=torch.int32)
        table = torch.empty(batch, logical_pages, device=device, dtype=torch.int32)
        table.copy_(torch.arange(batch*logical_pages, device=device, dtype=torch.int32)
                    .reshape(batch, logical_pages) % blocks)
        storage = torch.full((batch+2, 24, 256), 127, device=device, dtype=torch.bfloat16)
        out = storage[1:-1]

        def run():
            return paged_attention(q, k, v, out, cu, lens, table,
                                   131072, 16)

        run()
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            result = run()
        lengths = [(129, 900), (512, 1024), (2051, 1536), (4096, 17)]
        if batch == 2:
            lengths.append((32768, 131071))
        for case, active_lens in enumerate(lengths):
            values = list(active_lens) + [0] * (batch-2)
            lens.copy_(torch.tensor(values, device=device, dtype=torch.int32))
            # Change page identities after capture, preserving table address.
            pages = (torch.arange(batch*logical_pages, device=device, dtype=torch.int32)
                     .reshape(batch, logical_pages) + 3*case) % blocks
            table.copy_(pages.flip(1))
            graph.replay()
            torch.xpu.synchronize()
            first = result[:2].clone()
            for _ in range(3):
                graph.replay()
            torch.xpu.synchronize()
            repeat = bool(torch.equal(first, result[:2]))
            canary = bool((storage[[0, batch+1]] == 127).all())
            errors, numeric = [], []
            for row, length in enumerate(active_lens):
                kk = k[table[row].long()].flatten(0, 1)[:length].float()
                vv = v[table[row].long()].flatten(0, 1)[:length].float()
                # Keep the 12 query heads grouped under each KV head, so
                # the near-131K reference does not replicate K/V 12 times.
                qq = q[row].float().reshape(2, 12, 256)
                scores = torch.einsum("kgd,tkd->kgt", qq, kk) / 16
                ref = torch.einsum("kgt,tkd->kgd", scores.softmax(-1), vv).reshape(24, 256)
                actual = result[row].float()
                errors.append(float((actual-ref).abs().max()))
                numeric.append(bool(torch.allclose(actual, ref, atol=0.003, rtol=0.03)))
            ok = all(numeric) and repeat and canary
            bad += not ok
            print(json.dumps(dict(batch=batch, case=case, lens=values,
                                  changed_pages=True, active_numeric=numeric,
                                  maxabs=errors, exact_replay=repeat,
                                  output_canaries=canary, ok=ok)), flush=True)
        del graph
    print(json.dumps(dict(event="done", cases=13, bad=bad)), flush=True)
    signal.alarm(0)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
