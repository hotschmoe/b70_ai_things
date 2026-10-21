"""Scratch-only bounded FA graph/reinstantiate + exact PLE mixed-T21 probe."""
import argparse
import json
import os
import signal
import time
from types import SimpleNamespace

import torch
from d1_fa_plan import fill_plan
from vllm.models.qwen4_exp.gguf.mtp_ple import PLEHistory
import vllm_xpu_kernels.flash_attn_interface as interface


def emit(**values):
    print(json.dumps(values), flush=True)


def ple_gather(ring, rows):
    # The exact failing operator: BF16 [233,10,10240] ring indexed by
    # two contiguous int64 [T] arrays. No PLE host table/model load.
    slots = torch.full((rows,), 21, device="xpu", dtype=torch.int64)
    past = torch.arange(rows, device="xpu", dtype=torch.int64) - 9
    begin = time.perf_counter()
    gathered = ring[slots, past.remainder(10)]
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - begin
    assert bool((gathered == 1).all())
    emit(event="ple_gather", rows=rows, seconds=elapsed)


def ple_mixed(ring):
    # Reproduce20 fresh prefill rows plus1 cached decode at position15.
    # Source PLEHistory.conv: hc4*hidden2560, kernel4, dilation3, width1.
    ple = SimpleNamespace(hc=4, n_embd=2560, hc_dim=10240, hist=9,
                          kern=4, dil=3,
                          conv1d=SimpleNamespace(weight=torch.full(
                              (10240, 1, 4), 0.125, device="xpu", dtype=torch.bfloat16)))
    state = PLEHistory(ple, width=1, max_rows=12288)
    state.ring = ring
    state.slots = torch.tensor([21]*20+[20], device="xpu", dtype=torch.int64)
    state.positions = torch.tensor(list(range(20))+[15], device="xpu", dtype=torch.int64)
    state.starts = torch.tensor([0]*20+[20], device="xpu", dtype=torch.int64)
    state.ends = torch.tensor([20]*20+[21], device="xpu", dtype=torch.int64)
    normalized = torch.ones(21, 4, 2560, device="xpu", dtype=torch.bfloat16)
    begin = time.perf_counter()
    result = state.conv(normalized)
    torch.xpu.synchronize()
    # Current rows and history are all1, so only initial causal lags differ.
    weights = [sum(p >= lag for lag in (0,3,6,9))*0.125 for p in range(20)]+[0.5]
    reference = torch.nn.functional.silu(torch.tensor(weights)).to(torch.bfloat16)
    reference = reference.to("xpu")[:,None].expand(21,10240)
    torch.testing.assert_close(result, reference, atol=0.003, rtol=0.01)
    emit(event="ple_mixed21", seconds=time.perf_counter()-begin, reference_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("original", "temporary", "persistent"), required=True)
    args = parser.parse_args()
    signal.alarm(60)
    torch.manual_seed(41024)
    assert getattr(torch.xpu.XPUGraph, "_d1_reclaim", False), "need real reclaim500 overlay"
    emit(event="start", arm=args.arm, torch=torch.__version__, reclaim=os.environ.get("B70_XPU_CG_RECLAIM"))
    # Real backend storage order: (block,head,page,2*D), transposed views.
    kv = torch.randn(16,2,1600,512, device="xpu", dtype=torch.bfloat16).transpose(1,2)
    k, v = kv.split(256, dim=-1)
    ring = torch.ones(233,10,10240, device="xpu", dtype=torch.bfloat16)
    graphs = {}
    held = []
    pool = None
    os.environ["B70_D1_FA_PLAN"] = "0" if args.arm == "original" else "16"
    for batch in (8,4,3,2,1):
        q = torch.randn(batch,24,256,device="xpu",dtype=torch.bfloat16)
        lens = torch.full((batch,),900,device="xpu",dtype=torch.int32)
        table = torch.arange(batch*2,device="xpu",dtype=torch.int32).reshape(batch,2) %16
        cu = torch.arange(batch+1,device="xpu",dtype=torch.int32)
        scales = torch.ones((),device="xpu").expand(batch,2)
        outs = [torch.empty_like(q) for _ in range(12)]
        buffers = [(torch.empty(batch,device="xpu",dtype=torch.int32),
                    torch.empty(batch*16,4,device="xpu",dtype=torch.int32),
                    torch.zeros_like(cu)) for _ in range(12)]

        def run():
            for i in range(12):
                if args.arm != "persistent":
                    interface.flash_attn_varlen_func(
                        q=q,k=k,v=v,out=outs[i],max_seqlen_q=1,
                        cu_seqlens_q=cu,seqused_k=lens,max_seqlen_k=131072,
                        block_table=table,causal=True,k_descale=scales,v_descale=scales)
                else:
                    counts, work, zeros = buffers[i]
                    fill_plan(lens,counts,work,16)
                    torch.ops._vllm_fa2_C.varlen_fwd(
                        q,k,v,outs[i],cu,zeros,lens,None,table,None,
                        1,131072,0.0,None,scales,scales,1/16,None,
                        False,True,-1,-1,0.0,False,None,16,False,counts,work)
        run()
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph(keep_graph=True)
        with torch.xpu.graph(graph,pool=pool):
            run()
        if pool is None:
            pool = graph.pool()
        graphs[batch] = (graph,lens,outs[-1],q,table)
        held.append((q,lens,table,cu,scales,outs,buffers))
    for batch, total in ((1,1501),(2,501)):
        graph,lens,last,q,table = graphs[batch]
        for step in range(1,total+1):
            graph.replay()
            if step % 500 == 0:
                torch.xpu.synchronize()
                assert bool(last.isfinite().all())
                for row in range(batch):
                    kk=k[table[row].long()].flatten(0,1)[:900].float()
                    vv=v[table[row].long()].flatten(0,1)[:900].float()
                    qq=q[row].float().reshape(2,12,256)
                    scores=torch.einsum("kgd,tkd->kgt",qq,kk)/16
                    reference=torch.einsum("kgt,tkd->kgd",scores.softmax(-1),vv).reshape(24,256)
                    torch.testing.assert_close(last[row].float(),reference,atol=0.003,rtol=0.03)
                emit(event="boundary", arm=args.arm,batch=batch,replays=step,
                     allocated=torch.xpu.memory_allocated(),reserved=torch.xpu.memory_reserved(),
                     reference_ok=True,fds=len(os.listdir("/proc/self/fd")))
                if batch == 1 and step == 500:
                    ple_gather(ring,8192)
                if batch == 2:
                    ple_mixed(ring)
        torch.xpu.synchronize()
    emit(event="done", arm=args.arm, ok=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
