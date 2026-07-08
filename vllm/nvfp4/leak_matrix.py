#!/usr/bin/env python3
# leak_matrix.py -- OP-LEVEL diagnostic for the MTP-graph NEO linear_stream.h:84 leak.
#
# Root-cause hypothesis (docs/20260707_dd_mtp_piecewise_neo_abort.md): a oneCCL collective
# recorded into a torch.xpu.XPUGraph (SYCL command_graph) RE-APPENDS command-list commands on
# every replay -> NEO LinearStream overflow (abort at compute-runtime linear_stream.h:84). Our
# custom push-AR posted-write all-reduce (native L0, no oneCCL) is claimed to record as a STATIC
# node and NOT accumulate. This script records ONE collective into a raw XPUGraph and replays it
# REPLAYS times on 2 XPU ranks, reporting whether it aborts (LEAK) or survives (CLEAN) -- isolating
# WHICH transport accumulates, without a full 27B model serve (dev loop ~1 min vs ~5 min).
#
# COLL modes (env COLL=):
#   oneccl_ar     : dist.all_reduce(z)                       [oneCCL all-reduce]          expect LEAK
#   oneccl_ag     : dist.all_gather_into_tensor(out, z)      [oneCCL all-gather]          expect LEAK
#   block3_oneccl : padded-buf + dist.all_reduce             [CURRENT sitecustomize (3)]  expect LEAK
#   pushar        : ar_allreduce_graph(z)                    [push-AR posted-write]       expect CLEAN
#   block3_pushar : padded-buf + push-AR all_reduce          [the FIX for block (3)]      expect CLEAN
#
# Env: REPLAYS (default 1_000_000), PRINT_EVERY (10_000), SYNC_EVERY (100),
#      PUSH_SO (push-AR .so; needed for pushar / block3_pushar), NUMEL (5120), MASTER_PORT (29655).
import ctypes
import os
import sys
import time

import torch
import torch.distributed as dist

COLL = os.environ.get("COLL", "oneccl_ar")
REPLAYS = int(os.environ.get("REPLAYS", "1000000"))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "10000"))
SYNC_EVERY = int(os.environ.get("SYNC_EVERY", "100"))
NUMEL = int(os.environ.get("NUMEL", "5120"))
SO = os.environ.get("PUSH_SO", "/opt/push_ar/prebuilt/libxpu_push_ar_graph.so")
DT = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


def _bind_pushar(lib):
    lib.ar_setup_torch.restype = ctypes.c_int
    lib.ar_setup_torch.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.c_long]
    lib.ar_exchange.restype = ctypes.c_int
    lib.ar_exchange.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.ar_allreduce_ptr_dt.argtypes = [ctypes.c_ulonglong, ctypes.c_long, ctypes.c_int]
    lib.ar_allreduce_graph.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                                       ctypes.c_long, ctypes.c_int]


def run(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29655")
    torch.xpu.set_device(rank)
    dist.init_process_group(backend="xccl", rank=rank, world_size=world)
    grp = dist.group.WORLD
    dev = f"xpu:{rank}"

    need_pushar = COLL in ("pushar", "block3_pushar")
    lib = None
    if need_pushar:
        lib = ctypes.CDLL(SO)
        _bind_pushar(lib)
        q0 = torch.xpu.current_stream().sycl_queue
        rc = lib.ar_setup_torch(rank, ctypes.c_ulonglong(q0), 64 << 20)
        assert rc == 0, f"ar_setup rc={rc}"
        sock = f"/tmp/leak_matrix_{os.environ['MASTER_PORT']}.sock".encode()
        rc = lib.ar_exchange(rank, sock)
        assert rc == 0, f"ar_exchange rc={rc}"
        print(f"[r{rank}] push-AR setup ok", flush=True)

    x = torch.full((NUMEL,), float(rank + 1), device=dev, dtype=torch.bfloat16)

    # capture ONE collective into a raw XPUGraph
    s = torch.xpu.Stream()
    with torch.xpu.stream(s):
        torch.xpu.synchronize()
        dist.barrier()
        # buffers the captured collective reads/writes (same address every replay, as vLLM requires)
        z = x.clone()
        buf = torch.zeros((world,) + tuple(z.size()), dtype=z.dtype, device=dev)  # block3 padded buffer
        out_ag = torch.empty((world * NUMEL,), dtype=z.dtype, device=dev)         # oneccl_ag output
        g = torch.xpu.XPUGraph()
        g.capture_begin()
        if COLL == "oneccl_ar":
            dist.all_reduce(z, group=grp)
        elif COLL == "oneccl_ag":
            dist.all_gather_into_tensor(out_ag, z, group=grp)
        elif COLL == "block3_oneccl":
            buf.zero_()
            buf[rank] = z
            dist.all_reduce(buf, group=grp)
        elif COLL == "pushar":
            q = torch.xpu.current_stream().sycl_queue
            lib.ar_allreduce_graph(ctypes.c_ulonglong(q), ctypes.c_ulonglong(z.data_ptr()),
                                   z.numel() * z.element_size(), DT[z.dtype])
        elif COLL == "block3_pushar":
            buf.zero_()
            buf[rank] = z
            q = torch.xpu.current_stream().sycl_queue
            lib.ar_allreduce_graph(ctypes.c_ulonglong(q), ctypes.c_ulonglong(buf.data_ptr()),
                                   buf.numel() * buf.element_size(), DT[buf.dtype])
        else:
            raise SystemExit(f"unknown COLL={COLL}")
        g.capture_end()
        print(f"[r{rank}] captured COLL={COLL}; replaying up to {REPLAYS} ...", flush=True)
        torch.xpu.synchronize()
        dist.barrier()

        t0 = time.time()
        last_t, last_it, r0 = t0, 0, None
        for it in range(1, REPLAYS + 1):
            g.replay()
            if it % SYNC_EVERY == 0:
                torch.xpu.synchronize()
            if it % PRINT_EVERY == 0 and rank == 0:
                torch.xpu.synchronize()
                now = time.time()
                inst = (it - last_it) / (now - last_t)   # window rate -- decay = accumulation
                cum = it / (now - t0)
                if r0 is None:
                    r0 = inst
                print(f"[r0] replay {it}/{REPLAYS}  inst={inst:.0f}/s cum={cum:.0f}/s "
                      f"(inst/first={inst / r0:.2f})  z0={float(z[0]):.1f}", flush=True)
                last_t, last_it = now, it
        torch.xpu.synchronize()
        dt = time.time() - t0
    dist.barrier()
    dist.destroy_process_group()
    print(f"[r{rank}] CLEAN: COLL={COLL} survived {REPLAYS} replays in {dt:.1f}s -- NO linear_stream abort", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    print(f"=== leak_matrix COLL={COLL} REPLAYS={REPLAYS} NUMEL={NUMEL} ===", flush=True)
    mp.spawn(run, args=(2,), nprocs=2, join=True)
    print(f"=== COLL={COLL} PASS (both ranks clean) ===", flush=True)
