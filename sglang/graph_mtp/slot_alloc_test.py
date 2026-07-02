#!/usr/bin/env python3
# slot_alloc_test.py -- validate the run-25 bump-pointer PAYLOAD slot fix in ar_allreduce_graph_spin.
# Reproduces the exact failing shape: ONE captured graph with ~130 AR nodes of varying size INCLUDING
# a ~11MB node (the logits all_gather that overran its 1MB slot -> "!" garbage). Then a SECOND graph
# (after ar_graph_new_capture reset) to prove cross-graph arena reuse + global flag ids are correct.
#
# Correctness: each buffer is filled to (rank+1) before every replay; a correct push-allreduce sum
# leaves every element == 1.0 + 2.0 == 3.0 on both ranks. Any slot aliasing/overrun corrupts some
# buffer -> caught. Run: PUSH_SO=/work/gm/libxpu_push_ar_bisect.so inside sglang-xpu:mtp, 2 ranks.
import ctypes
import os

import torch
import torch.distributed as dist

SO = os.environ.get("PUSH_SO", "/work/gm/libxpu_push_ar_bisect.so")
MAXB = int(os.environ.get("MAXB", str(64 << 20)))
BIG_MB = float(os.environ.get("BIG_MB", "11"))


def build_sizes():
    # numel (bf16) list mimicking a verify graph: many small hidden-state ARs + a big logits gather.
    small = 5120  # ~10KB, a typical [tokens, hidden] decode AR slice
    sizes = [small] * 128
    sizes.insert(64, int(BIG_MB * (1 << 20) / 2))   # ~11MB node in the middle
    sizes.insert(30, 4096 * 44)                     # a ~360KB medium
    return sizes


def record_graph(lib, rank, sizes):
    lib.ar_graph_new_capture()                      # reset payload bump-pointer (the capture_begin hook)
    bufs = [torch.full((n,), float(rank + 1), device=f"xpu:{rank}", dtype=torch.bfloat16) for n in sizes]
    torch.xpu.synchronize()
    dist.barrier()
    g = torch.xpu.XPUGraph()
    g.capture_begin()
    q = torch.xpu.current_stream().sycl_queue
    for b in bufs:
        lib.ar_allreduce_graph_spin(ctypes.c_ulonglong(q), ctypes.c_ulonglong(b.data_ptr()),
                                    b.numel() * 2, 1, MAXB)
    g.capture_end()
    dist.barrier()
    return g, bufs


def check(bufs, rank, tag):
    bad = 0
    for i, b in enumerate(bufs):
        mn = float(b.min()); mx = float(b.max())
        if mn != 3.0 or mx != 3.0:
            bad += 1
            if bad <= 5:
                print(f"[r{rank}] {tag} node {i} n={b.numel()} CORRUPT min={mn} max={mx}", flush=True)
    if bad == 0:
        print(f"[r{rank}] {tag} ALL {len(bufs)} nodes == 3.0  OK", flush=True)
    return bad


def run(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29656"
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
    lib.ar_graph_new_capture.restype = None
    lib.ar_graph_new_capture.argtypes = []

    q0 = torch.xpu.current_stream().sycl_queue
    assert lib.ar_setup_torch(rank, ctypes.c_ulonglong(q0), MAXB) == 0
    assert lib.ar_exchange(rank, b"/tmp/slot_alloc_test.sock") == 0
    assert lib.ar_graph_spin_init(MAXB) == 0
    print(f"[r{rank}] setup ok, MAXB={MAXB>>20}MB", flush=True)

    sizes = build_sizes()
    total_mb = sum(n * 2 for n in sizes) / 1e6
    print(f"[r{rank}] graph: {len(sizes)} nodes, {total_mb:.1f}MB payload, big={BIG_MB}MB", flush=True)

    s = torch.xpu.Stream()
    bad_total = 0
    with torch.xpu.stream(s):
        # GRAPH A
        gA, bufsA = record_graph(lib, rank, sizes)
        print(f"[r{rank}] graph A captured ({len(bufsA)} nodes)", flush=True)
        for it in range(4):
            for b in bufsA:
                b.fill_(float(rank + 1))
            torch.xpu.synchronize(); dist.barrier()
            gA.replay(); torch.xpu.synchronize()
            bad_total += check(bufsA, rank, f"A-replay{it}")
            dist.barrier()
        # GRAPH B (smaller; tests cross-graph arena reuse + global flag ids). new_capture resets cursor.
        sizesB = [5120] * 40 + [int(2 * (1 << 20) / 2)]
        gB, bufsB = record_graph(lib, rank, sizesB)
        print(f"[r{rank}] graph B captured ({len(bufsB)} nodes)", flush=True)
        # interleave A and B replays -- the serving pattern that stresses arena reuse across graphs
        for it in range(4):
            for b in bufsA:
                b.fill_(float(rank + 1))
            for b in bufsB:
                b.fill_(float(rank + 1))
            torch.xpu.synchronize(); dist.barrier()
            gA.replay(); gB.replay(); torch.xpu.synchronize()
            bad_total += check(bufsA, rank, f"AB-A{it}")
            bad_total += check(bufsB, rank, f"AB-B{it}")
            dist.barrier()

    dist.destroy_process_group()
    print(f"[r{rank}] {'PASS' if bad_total == 0 else 'FAIL bad=' + str(bad_total)}", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.spawn(run, args=(2,), nprocs=2, join=True)
