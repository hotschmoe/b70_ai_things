#!/usr/bin/env python3
# allgather_emul_test.py -- isolate the run-26 logits-corruption hypothesis: the capture-time
# all_gather emulation (zero-pad + push-AR-SUM + reshape) goes WRONG on REPLAY because the zero
# PADDING half of `out` is not re-zeroed each replay -> stale data from the previous replay bleeds
# into the peer's gathered slice via the sum. Replicates _push_all_gather EXACTLY inside a captured
# graph, replays 4x with DIFFERENT inputs each time, and checks each replay vs a concat reference.
#
# EXPECT (if hypothesis true): replay 0 may look ok-ish (pool zero) but replay >=1 shows the peer's
# slice contaminated by the PREVIOUS replay's input (stale padding). Run 2 ranks in sglang-xpu:mtp.
import ctypes
import os

import torch
import torch.distributed as dist

SO = os.environ.get("PUSH_SO", "/work/gm/libxpu_push_ar_bisect.so")
MAXB = int(os.environ.get("MAXB", str(128 << 20)))
FIX = os.environ.get("FIX", "0") == "1"   # FIX=1 -> explicit in-graph out.zero_() each replay


def run(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29657"
    torch.xpu.set_device(rank)
    dist.init_process_group(backend="xccl", rank=rank, world_size=world)
    lib = ctypes.CDLL(SO)
    lib.ar_setup_torch.restype = ctypes.c_int
    lib.ar_setup_torch.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.c_long]
    lib.ar_exchange.restype = ctypes.c_int
    lib.ar_exchange.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.ar_graph_spin_init.restype = ctypes.c_int
    lib.ar_graph_spin_init.argtypes = [ctypes.c_long]
    lib.ar_allreduce_graph_spin.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                                            ctypes.c_long, ctypes.c_int, ctypes.c_long]
    lib.ar_graph_new_capture.restype = None; lib.ar_graph_new_capture.argtypes = []
    q0 = torch.xpu.current_stream().sycl_queue
    assert lib.ar_setup_torch(rank, ctypes.c_ulonglong(q0), MAXB) == 0
    assert lib.ar_exchange(rank, b"/tmp/ag_emul_test.sock") == 0
    assert lib.ar_graph_spin_init(MAXB) == 0
    ws = world; dev = f"xpu:{rank}"
    T, V = 8, 4096                                  # [tokens, vocab_shard]
    dim = -1; d = dim + 2

    # static input buffer (sglang fills this per forward); we refill it per replay
    inp = torch.zeros((T, V), device=dev, dtype=torch.bfloat16)

    def emulate_capture():
        lib.ar_graph_new_capture()
        out = torch.zeros((ws, T, V), device=dev, dtype=torch.bfloat16)   # EXACT _push_all_gather
        q = torch.xpu.current_stream().sycl_queue
        if FIX:
            out.zero_()                                                    # explicit recorded re-zero
        out[rank].copy_(inp)
        lib.ar_allreduce_graph_spin(ctypes.c_ulonglong(q), ctypes.c_ulonglong(out.data_ptr()),
                                    out.numel() * 2, 1, MAXB)
        res = out.movedim(0, d).reshape((T,) + (ws * V,)).contiguous()
        return out, res

    s = torch.xpu.Stream()
    bad = 0
    with torch.xpu.stream(s):
        inp.fill_(float(rank + 1)); torch.xpu.synchronize(); dist.barrier()
        g = torch.xpu.XPUGraph(); g.capture_begin()
        out, res = emulate_capture()
        g.capture_end(); dist.barrier()
        # replay with DIFFERENT inputs each time. reference gather = concat of both ranks' inp.
        for it in range(4):
            myval = float((rank + 1) * 10 + it)           # distinct per rank per iter
            inp.fill_(myval)
            torch.xpu.synchronize(); dist.barrier()
            g.replay(); torch.xpu.synchronize()
            # reference: gathered[:, :V] = rank0's inp value, [:, V:] = rank1's inp value
            r0 = float(10 + it); r1 = float(20 + it)      # rank0 myval, rank1 myval
            got0 = float(res[:, :V].min()), float(res[:, :V].max())
            got1 = float(res[:, V:].min()), float(res[:, V:].max())
            ok = got0 == (r0, r0) and got1 == (r1, r1)
            if not ok:
                bad += 1
                print(f"[r{rank}] replay{it} WRONG: slice0={got0} (want {r0})  slice1={got1} (want {r1})", flush=True)
            else:
                print(f"[r{rank}] replay{it} OK  slice0={r0} slice1={r1}", flush=True)
            dist.barrier()
    dist.destroy_process_group()
    print(f"[r{rank}] {'PASS' if bad == 0 else 'FAIL bad=' + str(bad)}  (FIX={FIX})", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.spawn(run, args=(2,), nprocs=2, join=True)
