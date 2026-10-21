#!/usr/bin/env python3
"""Exercise the actual private FA interface with vLLM's scale views."""

import json
import os
import signal

import torch
import d1_fa_plan  # Match real serving: initialize vLLM before native FA.
import vllm_xpu_kernels.flash_attn_interface as interface


def main():
    signal.alarm(120)
    torch.manual_seed(41023)
    dev = "xpu:0"
    batch = 2
    q = torch.randn(batch, 24, 256, device=dev, dtype=torch.bfloat16)
    k = torch.randn(4, 1600, 2, 256, device=dev, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.tensor([[0, 1], [3, 2]], device=dev, dtype=torch.int32)
    lens = torch.tensor([345, 900], device=dev, dtype=torch.int32)
    cu = torch.arange(batch+1, device=dev, dtype=torch.int32)
    # Exact backend storage contract: expanded view of one FP32 scalar.
    k_scale = torch.ones((), device=dev).expand(batch, 2)
    v_scale = torch.ones((), device=dev).expand(batch, 2)
    out = torch.empty_like(q)
    original = interface.torch.ops._vllm_fa2_C.varlen_fwd
    seen = []

    def spy(*args):
        seen.append(dict(q_same=args[13] is None,
                         k_same=args[14] is k_scale,
                         v_same=args[15] is v_scale,
                         splits=args[25], plan=args[27] is not None,
                         work=args[28] is not None))
        return original(*args)

    interface.torch.ops._vllm_fa2_C.varlen_fwd = spy

    def run():
        return interface.flash_attn_varlen_func(
            q=q, k=k, v=v, out=out, max_seqlen_q=1,
            cu_seqlens_q=cu, seqused_k=lens, max_seqlen_k=131072,
            block_table=table, causal=True, k_descale=k_scale,
            v_descale=v_scale,
        )

    try:
        run()
        torch.xpu.synchronize()
        assert seen and all(all(x.values()) for x in seen)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            result = run()
    finally:
        interface.torch.ops._vllm_fa2_C.varlen_fwd = original
    assert interface.__dict__.get("_d1_activation_noted") is True
    for a, b in ((345, 900), (512, 1024), (1984, 2051)):
        lens.copy_(torch.tensor([a, b], device=dev, dtype=torch.int32))
        table.copy_(table.flip(1))
        graph.replay()
        torch.xpu.synchronize()
        first = result.clone()
        graph.replay()
        torch.xpu.synchronize()
        assert torch.equal(result, first)
        errors = []
        for row, length in enumerate((a, b)):
            kk = k[table[row].long()].flatten(0, 1)[:length].float()
            vv = v[table[row].long()].flatten(0, 1)[:length].float()
            qq = q[row].float().reshape(2, 12, 256)
            scores = torch.einsum("kgd,tkd->kgt", qq, kk) / 16
            ref = torch.einsum("kgt,tkd->kgd", scores.softmax(-1), vv).reshape(24, 256)
            errors.append(float((result[row].float()-ref).abs().max()))
            torch.testing.assert_close(result[row].float(), ref, atol=0.003, rtol=0.03)
        print(json.dumps(dict(lens=[a,b], active=True, scale_objects_forwarded=True,
                              exact_replay=True, maxabs=errors)), flush=True)
    print(json.dumps(dict(event="D1_FA_WRAPPER_GATE_OK", native_calls=seen,
                          overlay=interface.__file__, flag=os.environ.get("B70_D1_FA_PLAN"))), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
