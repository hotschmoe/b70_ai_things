#!/usr/bin/env python3
# leak_matrix_reclaim.py -- test whether a graph-replay command-list RECLAIM resets the per-replay
# accumulation (the inst-rate decay) that overflows NEO LinearStream. Records ONE collective into a
# torch.xpu.XPUGraph(keep_graph=True) on 2 ranks, replays REPLAYS times, and periodically applies a
# reclaim. If the inst rate DECAYS then JUMPS BACK UP right after each reclaim, that reclaim works.
#
# RECLAIM modes (env RECLAIM=):
#   none    : baseline (expect monotonic decay)
#   reinst  : g.instantiate() every EVERY replays  (re-finalize exec graph from the kept modifiable graph;
#             the doc's "resets only on re-instantiation" -- cheap, no re-trace)
#   rotate  : replay on a FRESH torch.xpu.Stream every EVERY replays (fresh immediate command list;
#             ALSO probes whether cross-stream replay of a captured graph is even legal on XPU)
#   reset   : g.reset()+recapture every EVERY replays (full recapture control -- expensive upper bound)
#
# Env: COLL (pushar|oneccl_ar), RECLAIM (none|reinst|rotate|reset), EVERY (2000), REPLAYS (60000),
#      PRINT_EVERY (2000), SYNC_EVERY (100), NUMEL (5120), PUSH_SO, MASTER_PORT.
import ctypes, os, sys, time
import torch
import torch.distributed as dist

COLL = os.environ.get("COLL", "pushar")
RECLAIM = os.environ.get("RECLAIM", "reinst")
EVERY = int(os.environ.get("EVERY", "2000"))
REPLAYS = int(os.environ.get("REPLAYS", "60000"))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "2000"))
SYNC_EVERY = int(os.environ.get("SYNC_EVERY", "100"))
NUMEL = int(os.environ.get("NUMEL", "5120"))
SO = os.environ.get("PUSH_SO", "/opt/push_ar/prebuilt/libxpu_push_ar_graph.so")
DT = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


def _bind(lib):
    lib.ar_setup_torch.restype = ctypes.c_int
    lib.ar_setup_torch.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.c_long]
    lib.ar_exchange.restype = ctypes.c_int
    lib.ar_exchange.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.ar_allreduce_graph.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_long, ctypes.c_int]


def run(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29760")
    torch.xpu.set_device(rank)
    dist.init_process_group(backend="xccl", rank=rank, world_size=world)
    grp = dist.group.WORLD
    dev = f"xpu:{rank}"
    lib = None
    if COLL == "pushar":
        lib = ctypes.CDLL(SO); _bind(lib)
        q0 = torch.xpu.current_stream().sycl_queue
        assert lib.ar_setup_torch(rank, ctypes.c_ulonglong(q0), 64 << 20) == 0
        assert lib.ar_exchange(rank, f"/tmp/reclaim_{os.environ['MASTER_PORT']}.sock".encode()) == 0

    x = torch.full((NUMEL,), float(rank + 1), device=dev, dtype=torch.bfloat16)
    z = x.clone()

    def record_into(gr):
        # (re)record the chosen collective into the modifiable graph gr
        if COLL == "pushar":
            q = torch.xpu.current_stream().sycl_queue
            lib.ar_allreduce_graph(ctypes.c_ulonglong(q), ctypes.c_ulonglong(z.data_ptr()),
                                   z.numel() * z.element_size(), DT[z.dtype])
        else:  # oneccl_ar
            dist.all_reduce(z, group=grp)

    cap_s = torch.xpu.Stream()
    with torch.xpu.stream(cap_s):
        torch.xpu.synchronize(); dist.barrier()
        g = torch.xpu.XPUGraph(keep_graph=True)
        g.capture_begin(); record_into(g); g.capture_end()
        g.instantiate()
        print(f"[r{rank}] captured COLL={COLL} keep_graph=True; RECLAIM={RECLAIM} EVERY={EVERY}", flush=True)
        torch.xpu.synchronize(); dist.barrier()

        rot_s = cap_s
        t0 = time.time(); last_t, last_it, r0, post = t0, 0, None, {}
        for it in range(1, REPLAYS + 1):
            # apply reclaim
            if it % EVERY == 0:
                if RECLAIM == "reinst":
                    g.instantiate()                 # re-finalize exec graph (destroys previous exec)
                elif RECLAIM == "rotate":
                    rot_s = torch.xpu.Stream()       # fresh stream => fresh immediate command list
                elif RECLAIM == "reset":
                    g.reset(); g.capture_begin(); record_into(g); g.capture_end(); g.instantiate()
                post[it] = True
            if RECLAIM == "rotate" and rot_s is not cap_s:
                with torch.xpu.stream(rot_s):
                    g.replay()
            else:
                g.replay()
            if it % SYNC_EVERY == 0:
                torch.xpu.synchronize()
            if it % PRINT_EVERY == 0 and rank == 0:
                torch.xpu.synchronize()
                now = time.time()
                inst = (it - last_it) / (now - last_t)
                if r0 is None:
                    r0 = inst
                tag = "  <-- reclaim applied" if (it in post or (it - PRINT_EVERY + EVERY) in post) else ""
                print(f"[r0] replay {it}/{REPLAYS}  inst={inst:.0f}/s (inst/first={inst / r0:.2f}){tag}", flush=True)
                last_t, last_it = now, it
        torch.xpu.synchronize(); dt = time.time() - t0
    dist.barrier(); dist.destroy_process_group()
    print(f"[r{rank}] DONE COLL={COLL} RECLAIM={RECLAIM}: {REPLAYS} replays in {dt:.1f}s, no abort", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    print(f"=== reclaim test COLL={COLL} RECLAIM={RECLAIM} EVERY={EVERY} REPLAYS={REPLAYS} ===", flush=True)
    mp.spawn(run, args=(2,), nprocs=2, join=True)
    print(f"=== COLL={COLL} RECLAIM={RECLAIM} PASS ===", flush=True)
