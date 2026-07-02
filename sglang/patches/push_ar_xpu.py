# push_ar_xpu.py -- port of vllm/contrib/vllm_push_allreduce/_push_ar_patch.py to SGLANG.
# Monkeypatches sglang's XpuCommunicator.all_reduce to use the hand-rolled PUSH all-reduce
# (libxpu_push_ar_graph.so, C ABI, torch-independent -- see scripts/106 + P2P_GPU.md J.7-J.12/K).
# On dual B70 this beats oneCCL on decode latency (~34-45us vs ~85-88us) and prefill bandwidth
# (~10.6 vs ~9.4 GB/s), and does its OWN L0-IPC P2P (independent of CCL_TOPO_P2P_ACCESS, so the
# serve keeps P2PACCESS=0 -- no H.13 wedge surface).
#
# sglang specifics vs the vLLM patch:
#   - import path: sglang.srt.distributed.device_communicators.xpu_communicator
#   - sglang's XpuCommunicator has no .rank; use dist.get_rank(self.group).
#   - sglang creates XpuCommunicators for SEVERAL 2-rank groups (world, tp, ...). The .so supports
#     ONE pair rendezvous per process, so only the FIRST group to all_reduce engages (module flag);
#     the rest keep oneCCL. TP group all_reduces first (model forward) -> it gets the push path.
#   - rendezvous sock: PUSH_AR_SOCK (default /tmp/sglang_push_ar.sock -- both TP scheduler procs
#     share the container /tmp; one serve per container).
#
# Activation: import push_ar_xpu; push_ar_xpu.install()  (chained from woq_shim under
# B70_XPU_PUSH_AR=1) + PUSH_AR_SO=/path/to/libxpu_push_ar_graph.so.
# Env knobs (same semantics as the vLLM patch):
#   PUSH_AR_DISABLE=1     -> no-op
#   PUSH_AR_MAXB          -> scratch bytes cap (default 128 MiB)
#   PUSH_AR_MIN_NUMEL     -> engage only for numel >= this (0 = all; default 0)
#   PUSH_AR_GRAPH=1       -> use ar_allreduce_graph during XPUGraph capture (K.6 capturable path)
#   B70_PUSH_AR_STATS=1   -> count calls/bytes on rank 0, print every PUSH_AR_STATS_EVERY (2000)
import os
import ctypes
import threading

_lib = None
_lib_lock = threading.Lock()
_engaged_owner = None  # id() of the communicator that owns the pair rendezvous


def _load_lib(so):
    global _lib
    if _lib is not None:
        return _lib
    with _lib_lock:
        if _lib is None:
            lib = ctypes.CDLL(so)  # dlopen pulls libsycl/ze_loader -> L0 init (deferred on purpose, J.15)
            lib.ar_setup_torch.restype = ctypes.c_int
            lib.ar_setup_torch.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.c_long]
            lib.ar_exchange.restype = ctypes.c_int
            lib.ar_exchange.argtypes = [ctypes.c_int, ctypes.c_char_p]
            lib.ar_allreduce_ptr_dt.argtypes = [ctypes.c_ulonglong, ctypes.c_long, ctypes.c_int]
            if hasattr(lib, "ar_allreduce_graph"):
                lib.ar_allreduce_graph.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                                                   ctypes.c_long, ctypes.c_int]
            # PREFERRED capturable path (2026-07-02): pure-SYCL spin-kernel sync. The K.6 native-command
            # variant records but HANGS XPUGraph capture_end on DPC++ 2025.3/torch 2.12 (bisect: even an
            # EMPTY ext_codeplay_enqueue_native_command breaks finalize). The spin variant records and
            # replays correctly (5/5 microbench, device-side seq counters).
            if hasattr(lib, "ar_allreduce_graph_spin"):
                lib.ar_graph_spin_init.restype = ctypes.c_int
                lib.ar_graph_spin_init.argtypes = [ctypes.c_long]
                lib.ar_allreduce_graph_spin.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                                                        ctypes.c_long, ctypes.c_int, ctypes.c_long]
            # run-25 payload-slot fix: reset the per-graph payload bump-pointer at each capture_begin.
            if hasattr(lib, "ar_graph_new_capture"):
                lib.ar_graph_new_capture.restype = None
                lib.ar_graph_new_capture.argtypes = []
            _lib = lib
    return _lib


def on_capture_begin():
    """Reset the push-AR payload bump-pointer at the START of each captured graph. Safe no-op until the
    push collective is loaded+engaged. Hooked from _B70XPUGraph.capture_begin (woq_shim)."""
    lib = _lib
    if lib is not None and _engaged_owner is not None and hasattr(lib, "ar_graph_new_capture"):
        try:
            lib.ar_graph_new_capture()
        except Exception:
            pass


def install():
    if os.environ.get("PUSH_AR_DISABLE") == "1":
        return
    so = os.environ.get("PUSH_AR_SO", "/work/push_ar/libxpu_push_ar_graph.so")
    if not os.path.exists(so):
        print(f"[push-ar] SO not found at {so}; leaving oneCCL all_reduce in place", flush=True)
        return
    try:
        import torch
        import torch.distributed as dist
        from sglang.srt.distributed.device_communicators.xpu_communicator import XpuCommunicator
    except Exception as e:
        print(f"[push-ar] import failed ({e}); not patching", flush=True)
        return

    MAXB = int(os.environ.get("PUSH_AR_MAXB", str(128 << 20)))
    MIN_NUMEL = int(os.environ.get("PUSH_AR_MIN_NUMEL", "0"))
    SOCK = os.environ.get("PUSH_AR_SOCK", "/tmp/sglang_push_ar.sock")
    GRAPH = os.environ.get("PUSH_AR_GRAPH") == "1"
    STATS = os.environ.get("B70_PUSH_AR_STATS") == "1"
    STATS_EVERY = int(os.environ.get("PUSH_AR_STATS_EVERY", "2000"))
    _is_capturing = getattr(torch.xpu, "is_current_stream_capturing", lambda: False)
    DT = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}

    _orig_all_reduce = XpuCommunicator.all_reduce
    _lock = threading.Lock()
    _stats = {"n": 0, "bytes": 0, "n_small": 0, "n_fallback": 0}

    def _lazy_init(self):
        global _engaged_owner
        ready = getattr(self, "_push_ar_ready", None)
        if ready is not None:
            return ready
        with _lock:
            ready = getattr(self, "_push_ar_ready", None)
            if ready is not None:
                return ready
            ok = False
            # one pair rendezvous per process: first 2-rank group wins, others keep oneCCL
            if _engaged_owner is None and self.world_size == 2:
                try:
                    lib = _load_lib(so)
                    rank = dist.get_rank(self.group)
                    qaddr = torch.xpu.current_stream().sycl_queue
                    rc = lib.ar_setup_torch(rank, ctypes.c_ulonglong(qaddr), MAXB)
                    if rc == 0:
                        rc = lib.ar_exchange(rank, SOCK.encode())
                        ok = (rc == 0)
                    if ok and hasattr(lib, "ar_allreduce_graph_spin"):
                        lib.ar_graph_spin_init(MAXB)
                    if not ok:
                        print(f"[push-ar] setup/exchange rc={rc}; falling back to oneCCL", flush=True)
                except Exception as e:
                    print(f"[push-ar] lazy init exception {e}; falling back", flush=True)
                    ok = False
                if ok:
                    _engaged_owner = id(self)
                    self._push_ar_rank = dist.get_rank(self.group)
                    if self._push_ar_rank == 0:
                        print("[push-ar] ENGAGED: sglang XpuCommunicator.all_reduce -> push collective "
                              f"(sock={SOCK}, graph={GRAPH}, min_numel={MIN_NUMEL})", flush=True)
            self._push_ar_ready = ok
            return ok

    def all_reduce(self, input_):
        if (self.world_size == 2 and input_.is_contiguous()
                and input_.dtype in DT
                and input_.numel() * input_.element_size() <= MAXB
                and _lazy_init(self)):
            lib = _load_lib(so)
            if STATS:
                _stats["n"] += 1
                _stats["bytes"] += input_.numel() * input_.element_size()
                if input_.numel() < 65536:
                    _stats["n_small"] += 1
                if _stats["n"] % STATS_EVERY == 0 and getattr(self, "_push_ar_rank", 1) == 0:
                    print(f"[push-ar-stats] calls={_stats['n']} small(<64k-elem)={_stats['n_small']} "
                          f"fallback={_stats['n_fallback']} MB={_stats['bytes']/1e6:.1f}", flush=True)
            # CAPTURED decode (K.6): device-side L0-event sync records into torch's XPUGraph.
            if GRAPH and _is_capturing():
                out = input_.clone()
                q = torch.xpu.current_stream().sycl_queue
                if hasattr(lib, "ar_allreduce_graph_spin"):
                    lib.ar_allreduce_graph_spin(ctypes.c_ulonglong(q), ctypes.c_ulonglong(out.data_ptr()),
                                                out.numel() * out.element_size(), DT[input_.dtype], MAXB)
                elif hasattr(lib, "ar_allreduce_graph"):
                    lib.ar_allreduce_graph(ctypes.c_ulonglong(q), ctypes.c_ulonglong(out.data_ptr()),
                                           out.numel() * out.element_size(), DT[input_.dtype])
                else:
                    return _orig_all_reduce(self, input_)
                return out
            if not _is_capturing() and input_.numel() >= MIN_NUMEL:
                out = input_.clone()
                lib.ar_allreduce_ptr_dt(ctypes.c_ulonglong(out.data_ptr()),
                                        out.numel() * out.element_size(), DT[input_.dtype])
                return out
        if STATS:
            _stats["n_fallback"] += 1
        return _orig_all_reduce(self, input_)

    XpuCommunicator.all_reduce = all_reduce

    # ---- capture-time ALL_GATHER via zero-padded push-AR (PUSH_AR_GRAPH=1 only) ----
    # oneCCL all_gather has NO SYCL-Graph-recordable impl (the vLLM splitting_ops finding; recording it
    # HANGS capture on sglang). all_gather == sum of zero-padded per-rank slices, so during capture we
    # emulate it with ONE recorded push-AR on a [ws, *shape] buffer. Mirrors GroupCoordinator.all_gather
    # concat semantics (stack on dim0 -> movedim -> reshape). Eager path unchanged.
    GATHER_GRAPH = os.environ.get("PUSH_AR_GATHER_GRAPH", "1") == "1"  # 0 -> leave all_gather on oneCCL
    if GRAPH and GATHER_GRAPH:
        try:
            import sglang.srt.distributed.parallel_state as _ps

            _orig_all_gather = _ps.GroupCoordinator.all_gather

            def _push_all_gather(self, input_, dim=-1, output_tensor_list=None):
                comm = getattr(self, "xpu_communicator", None)
                if (
                    _is_capturing()
                    and output_tensor_list is None
                    and self.world_size == 2
                    and comm is not None
                    and getattr(comm, "_push_ar_ready", False)
                    and input_.is_contiguous()
                    and input_.dtype in DT
                ):
                    lib2 = _load_lib(so)
                    ws = self.world_size
                    input_size = input_.size()
                    d = dim + input_.dim() if dim < 0 else dim
                    rank = getattr(self, "rank_in_group", None)
                    if rank is None:
                        rank = dist.get_rank(self.device_group)
                    out = torch.zeros((ws,) + tuple(input_size), dtype=input_.dtype, device=input_.device)
                    out[rank].copy_(input_)
                    q = torch.xpu.current_stream().sycl_queue
                    if hasattr(lib2, "ar_allreduce_graph_spin"):
                        lib2.ar_allreduce_graph_spin(ctypes.c_ulonglong(q), ctypes.c_ulonglong(out.data_ptr()),
                                                     out.numel() * out.element_size(), DT[input_.dtype], MAXB)
                    else:
                        lib2.ar_allreduce_graph(ctypes.c_ulonglong(q), ctypes.c_ulonglong(out.data_ptr()),
                                                out.numel() * out.element_size(), DT[input_.dtype])
                    out = out.movedim(0, d).reshape(
                        input_size[:d] + (ws * input_size[d],) + input_size[d + 1:]
                    )
                    return out.contiguous()
                return _orig_all_gather(self, input_, dim, output_tensor_list)

            _ps.GroupCoordinator.all_gather = _push_all_gather
            print("[push-ar] capture-time all_gather -> zero-padded push-AR (recordable)", flush=True)
        except Exception as e:
            print(f"[push-ar] all_gather capture patch FAILED: {e}", flush=True)

    print(f"[push-ar] patched sglang XpuCommunicator.all_reduce (so={so}, maxB={MAXB})", flush=True)
