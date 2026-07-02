#!/usr/bin/env python3
# ar_integration_test.py -- reproduce the run-26/iso-AR failure: recorded push-AR all_reduce is
# numerically correct in ISOLATION (slot_alloc_test) but garbles in the real serve. The one pattern
# the isolation tests never covered: the AR INPUT is produced by a torch op INSIDE the captured graph
# (matmul -> clone -> push-AR -> matmul), with NO sync between. If the raw-queue push-AR kernels race
# the preceding/following torch ops (missing graph dependency edge), the reduced value is wrong.
#
# Both ranks use DISTINCT x so h differs; a correct all_reduce leaves h == h0+h1 on both ranks.
# Reference computed eagerly. Run 2 ranks in sglang-xpu:mtp.
import ctypes
import os

import torch
import torch.distributed as dist

SO = os.environ.get("PUSH_SO", "/work/gm/libxpu_push_ar_bisect.so")
MAXB = int(os.environ.get("MAXB", str(128 << 20)))
NLAYERS = int(os.environ.get("NLAYERS", "8"))   # chain of matmul+AR, like transformer layers


def run(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29658"
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
    assert lib.ar_exchange(rank, b"/tmp/ar_integ_test.sock") == 0
    assert lib.ar_graph_spin_init(MAXB) == 0
    dev = f"xpu:{rank}"
    M, K, N = 8, 512, 512
    torch.manual_seed(1234 + rank)                 # DISTINCT per rank
    x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1)
    Ws = [(torch.randn(K, N, device=dev, dtype=torch.bfloat16) * 0.05) for _ in range(NLAYERS)]

    def push_ar(t):                                # mirror push_ar_xpu all_reduce: clone + spin-AR
        out = t.clone()
        q = torch.xpu.current_stream().sycl_queue
        lib.ar_allreduce_graph_spin(ctypes.c_ulonglong(q), ctypes.c_ulonglong(out.data_ptr()),
                                    out.numel() * 2, 1, MAXB)
        return out

    def forward(reduce_fn):
        h = x
        for W in Ws:
            h = h @ W                              # activation produced INSIDE the region
            h = reduce_fn(h)                        # all-reduce it
        return h

    # eager reference using real oneCCL all_reduce (dist)
    def ccl_reduce(t):
        o = t.clone(); dist.all_reduce(o); return o
    ref = forward(ccl_reduce); torch.xpu.synchronize()

    # captured push-AR version
    s = torch.xpu.Stream()
    with torch.xpu.stream(s):
        torch.xpu.synchronize(); dist.barrier()
        lib.ar_graph_new_capture()
        g = torch.xpu.XPUGraph(); g.capture_begin()
        out = forward(push_ar)
        final = out.clone()
        g.capture_end(); dist.barrier()
        g.replay(); torch.xpu.synchronize()

    diff = (final.float() - ref.float()).abs().max().item()
    rel = diff / (ref.float().abs().max().item() + 1e-6)
    ok = rel < 0.02
    print(f"[r{rank}] captured-pushAR vs eager-oneCCL: max_abs_diff={diff:.4f} rel={rel:.4f} "
          f"{'OK' if ok else 'MISMATCH -> integration/ordering bug CONFIRMED'}", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.spawn(run, args=(2,), nprocs=2, join=True)
