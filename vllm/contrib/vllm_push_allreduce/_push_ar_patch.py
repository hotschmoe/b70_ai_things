# contrib/vllm_push_allreduce/sitecustomize.py
# Monkeypatch vLLM-XPU's XpuCommunicator.all_reduce to use the hand-rolled PUSH all-reduce
# (libxpu_push_ar_torch.so, see scripts/106). On dual B70 this beats oneCCL on BOTH decode latency
# and prefill bandwidth (P2P_GPU.md J.10/J.11), and -- crucially -- it does its OWN L0-IPC P2P,
# INDEPENDENT of CCL_TOPO_P2P_ACCESS, so the serve runs P2PACCESS=0 (oneCCL host-staged warmup
# succeeds, no H.13 DEVICE_LOST wedge) while the model's allreduces go over the 11 GB/s posted-write path.
#
# Activation: put this dir on PYTHONPATH AND set PUSH_AR_SO=/path/to/libxpu_push_ar_torch.so.
# Only engages for world_size==2 contiguous bf16/fp16/fp32 tensors <= PUSH_AR_MAXB; everything else
# (world>2, non-contig, odd dtype, oversize) falls back to the original dist.all_reduce. Disable with
# PUSH_AR_DISABLE=1.
#
# CAVEAT (graph capture): the op uses host .wait() + a CPU spin barrier + a ctypes call, so it is NOT
# SYCL-graph-capturable. Run the serve EAGER (GRAPH=0), or mark the TP allreduce as a graph splitting op.
#
# IMPORTANT (J.15): the SYCL/L0 .so is loaded LAZILY (first all_reduce), NOT at sitecustomize/startup time.
# Loading it at interpreter startup initializes Level-Zero before vLLM's own XPU init and breaks GRAPH=1
# (VLLM_COMPILE) model construction (rotary-emb cos/sin crash). Deferring the dlopen fixes that.
import os, ctypes, threading

_lib = None          # ctypes handle, loaded on first use
_lib_lock = threading.Lock()

def _load_lib(so):
    global _lib
    if _lib is not None:
        return _lib
    with _lib_lock:
        if _lib is None:
            lib = ctypes.CDLL(so)  # dlopen here pulls in libsycl/libze_loader -> L0 init (deferred on purpose)
            lib.ar_setup_torch.restype = ctypes.c_int
            lib.ar_setup_torch.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.c_long]
            lib.ar_exchange.restype = ctypes.c_int
            lib.ar_exchange.argtypes = [ctypes.c_int, ctypes.c_char_p]
            lib.ar_allreduce_ptr_dt.argtypes = [ctypes.c_ulonglong, ctypes.c_long, ctypes.c_int]
            # capturable decode path (libxpu_push_ar_graph.so, P2P_GPU K.6). Guarded: the older
            # libxpu_push_ar_torch.so has no graph symbol -> capture routing simply stays off.
            if hasattr(lib, "ar_allreduce_graph"):
                lib.ar_allreduce_graph.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong,
                                                   ctypes.c_long, ctypes.c_int]
            _lib = lib
    return _lib

def _install():
    if os.environ.get("PUSH_AR_DISABLE") == "1":
        return
    so = os.environ.get("PUSH_AR_SO", "/tmp/libxpu_push_ar_torch.so")
    if not os.path.exists(so):
        print(f"[push_ar] SO not found at {so}; leaving oneCCL all_reduce in place", flush=True)
        return
    try:
        import torch
        from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator
    except Exception as e:
        print(f"[push_ar] import failed ({e}); not patching", flush=True)
        return

    MAXB = int(os.environ.get("PUSH_AR_MAXB", str(128 << 20)))  # 128 MiB scratch
    SOCK_DIR = os.environ.get("PUSH_AR_SOCKDIR", "/tmp")
    # MIN_NUMEL gate: engage push-ar only for tensors with >= this many elements. Set it ABOVE the
    # captured decode sizes (e.g. max_num_seqs*(1+spec)*hidden) so captured decode allreduces fall back
    # to oneCCL (graph-recordable) while large EAGER prefill allreduces use push-ar. 0 = engage all
    # (the J.14 eager A/B). This is the capture-gated production mode (J.15).
    MIN_NUMEL = int(os.environ.get("PUSH_AR_MIN_NUMEL", "0"))
    # PUSH_AR_GRAPH=1 enables the CAPTURABLE decode path (K.6): during torch XPUGraph capture, route the
    # all-reduce through ar_allreduce_graph (device-side L0-event sync -> records into the graph), so DECODE
    # all-reduces use the 11 GB/s push transport instead of falling back to oneCCL. Eager (prefill) still uses
    # the host-barrier ar_allreduce_ptr_dt. Default off -> byte-for-byte the proven J.21 prefill-only behavior.
    GRAPH = os.environ.get("PUSH_AR_GRAPH") == "1"
    _is_capturing = getattr(torch.xpu, "is_current_stream_capturing", lambda: False)
    DT = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}

    _orig_all_reduce = XpuCommunicator.all_reduce
    _lock = threading.Lock()

    def _lazy_init(self):
        # one-time setup on first TP all_reduce; keyed off the 2-rank pairwise algorithm.
        if getattr(self, "_push_ar_ready", None) is not None:
            return self._push_ar_ready
        with _lock:
            if getattr(self, "_push_ar_ready", None) is not None:
                return self._push_ar_ready
            ok = False
            try:
                lib = _load_lib(so)  # dlopen the SYCL/L0 .so now (NOT at startup -- see header note)
                if self.world_size == 2:
                    qaddr = torch.xpu.current_stream().sycl_queue
                    rc = lib.ar_setup_torch(self.rank, ctypes.c_ulonglong(qaddr), MAXB)
                    if rc == 0:
                        # sockpath shared by the 2 workers of THIS group (unique per master port)
                        port = os.environ.get("MASTER_PORT", "0")
                        sock = os.path.join(SOCK_DIR, f"vllm_push_ar_{port}.sock").encode()
                        rc = lib.ar_exchange(self.rank, sock)
                        ok = (rc == 0)
                    if not ok:
                        print(f"[push_ar] setup/exchange rc={rc}; falling back to oneCCL", flush=True)
            except Exception as e:
                print(f"[push_ar] lazy init exception {e}; falling back", flush=True)
                ok = False
            self._push_ar_ready = ok
            if ok and self.rank == 0:
                print("[push_ar] ENGAGED: XpuCommunicator.all_reduce -> push collective (P2PACCESS-independent)",
                      flush=True)
            return ok

    def all_reduce(self, input_):
        if (self.world_size == 2 and input_.is_contiguous()
                and input_.dtype in DT
                and input_.numel() * input_.element_size() <= MAXB
                and _lazy_init(self)):
            lib = _load_lib(so)
            # CAPTURED decode: device-side event sync -> records into torch's XPUGraph (K.6). Capture happens
            # on a dedicated stream, so pass THAT stream's queue (NOT the setup queue) per call.
            if GRAPH and hasattr(lib, "ar_allreduce_graph") and _is_capturing():
                out = input_.clone()
                q = torch.xpu.current_stream().sycl_queue
                lib.ar_allreduce_graph(ctypes.c_ulonglong(q), ctypes.c_ulonglong(out.data_ptr()),
                                       out.numel() * out.element_size(), DT[input_.dtype])
                return out
            # EAGER (prefill / warmup): host-barrier push, gated by MIN_NUMEL (small -> oneCCL fallback).
            if input_.numel() >= MIN_NUMEL:
                out = input_.clone()
                lib.ar_allreduce_ptr_dt(ctypes.c_ulonglong(out.data_ptr()),
                                        out.numel() * out.element_size(), DT[input_.dtype])
                return out
        return _orig_all_reduce(self, input_)

    XpuCommunicator.all_reduce = all_reduce
    print(f"[push_ar] patched XpuCommunicator.all_reduce (so={so}, maxB={MAXB})", flush=True)

try:
    _install()
except Exception as e:
    print(f"[push_ar] install failed: {e}", flush=True)
